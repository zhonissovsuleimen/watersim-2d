#include "parameters.h"
#include <cuda_runtime.h>
#include <device_atomic_functions.h>

//input size is NUM_PARTICLES, output size is 2
__global__ inline void count(int* d_input, int* d_output, int bitOffset){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < NUM_PARTICLES) {
    int output_id = (d_input[id] >> bitOffset) & 1;
    atomicAdd(&d_output[output_id], 1);
  }
}

//input size is 2, output size is 2
__global__ inline void countPrefixSum(int* d_input, int* d_output){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  //using single thread since the length of the array is only 2 
  if (id == 0) {
    d_output[0] = d_input[0];
    d_output[1] = d_input[0] + d_input[1];
  }
}

//input size is NUM_PARTICLES, output size is NUM_PARTICLES
__global__ inline void populateAuxilaryArrays(int* d_input, int* d_output_bit0, int* d_output_bit1, int bitOffset){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < NUM_PARTICLES) {
    d_output_bit1[id] = (d_input[id] >> bitOffset) & 1;
    d_output_bit0[id] = 1 - d_output_bit1[id];
    // d_output_bit0[id] = !d_output_bit1[id];
  }
}

//input size is NUM_PARTICLES, output size is NUM_PARTICLES
__global__ inline void prefixSum(int* d_input, int* d_output){
  __shared__ int block[THREADS_PER_BLOCK];

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int local_id = threadIdx.x;
  if (id < NUM_PARTICLES) {

    block[local_id] = d_input[id];
    __syncthreads();

    for(int offset = 1; offset < THREADS_PER_BLOCK; offset *= 2){
      if(local_id >= offset){
        block[local_id] += block[local_id - offset];
      }
      __syncthreads();
    }

    d_output[id] = block[local_id];
    __syncthreads();

    for(int blockOffset = THREADS_PER_BLOCK; blockOffset < NUM_PARTICLES; blockOffset *= 2){
      if(id >= blockOffset){
        d_output[id] += d_output[id - blockOffset + (THREADS_PER_BLOCK - local_id - 1)];
      }
      __syncthreads();
    }
  }  
}

__global__ inline void combinePrefixSum(int* d_input, int* d_output, int* d_prefix_bit0, int* d_prefix_bit1, int bitOffset){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < NUM_PARTICLES) {
    int bit = (d_input[id] >> bitOffset) & 1;
    d_output[id] = bit ? d_prefix_bit1[id] : d_prefix_bit0[id];
  }
}

__global__ inline void reorder(int* d_input, int* d_output, int* d_count, int* d_prefix, int bitOffset){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < NUM_PARTICLES) {
    int bit = (d_input[id] >> bitOffset) & 1;
    int output_id = d_prefix[id] + (bit ? d_count[0] : 0) - 1;
    d_output[output_id] = d_input[id];
  }
}

__host__ inline void radixSort(int* d_input, int* d_output, int* d_count, int* d_prefix_bit0, int* d_prefix_bit1, int* d_prefix){
  for(int bitOffset = 0; bitOffset < 31; bitOffset++){
    cudaMemset(d_count, 0, 2 * sizeof(int));
    count<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_count, bitOffset);
    countPrefixSum<<<1, 1>>>(d_count, d_count);

    populateAuxilaryArrays<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_prefix_bit0, d_prefix_bit1, bitOffset);
    prefixSum<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_prefix_bit0, d_prefix_bit0);
    prefixSum<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_prefix_bit1, d_prefix_bit1);
    combinePrefixSum<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_prefix, d_prefix_bit0, d_prefix_bit1, bitOffset);

    reorder<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, d_count, d_prefix, bitOffset);
    cudaMemcpy(d_input, d_output, NUM_PARTICLES * sizeof(int), cudaMemcpyDeviceToDevice);
  }
}