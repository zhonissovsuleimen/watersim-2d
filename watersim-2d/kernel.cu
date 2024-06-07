#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "smoothing_kernels.cuh"
#include "calc.cuh"

#include "parameters.h"
#include "renderer.h"
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <random>

__host__ void initialize(Vector2* h_pos, Vector2* h_vel, float* h_density, Vector2* h_pressure) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disx(SIM_MIN_X + PARTICLE_RADIUS, SIM_MAX_X - PARTICLE_RADIUS);
  std::uniform_real_distribution<float> disy(SIM_MIN_Y + PARTICLE_RADIUS, SIM_MAX_Y - PARTICLE_RADIUS);
  std::uniform_real_distribution<float> disv(-10.0f, 10.0f);

  for (int i = 0; i < NUM_PARTICLES; i++) {
    h_pos[i].x = disx(gen);
    h_pos[i].y = disy(gen);
    h_vel[i].x = 0.0f;
    h_vel[i].y = 0.0f;
    h_density[i] = 0.0f;
    h_pressure[i].x = 0.0f;
    h_pressure[i].y = 0.0f;
  }
}

__host__ void initializeGrid(int gridSize, int* h_gridCellLookup, int* h_gridStartIndexLookup, int* h_gridCountLookup) {
  for (int i = 0; i < gridSize; i++) {
    h_gridCellLookup[i] = 0;
    h_gridStartIndexLookup[i] = 0;
    h_gridCountLookup[i] = 0;
  }
}

__global__ void updateCellLookup(Vector2* pos, int* gridCellLookup, int gridWidth, int gridHeight) {
  //assuming numberOfParticles < gridWidth * gridHeight for now
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < NUM_PARTICLES) {
    gridCellLookup[i] = calcCellIndex(pos[i].x, pos[i].y);
  }
}

__global__ void checkBoundaries(Vector2* pos, Vector2* vel){
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if(i < NUM_PARTICLES) {
    //collision with the floor/ceiling
    if(pos[i].y < SIM_MIN_Y + PARTICLE_RADIUS) {
      pos[i].y = SIM_MIN_Y + PARTICLE_RADIUS;
      vel[i].y = MULTIPLIER_DAMPING * -vel[i].y;
    }else if(pos[i].y > SIM_MAX_Y - PARTICLE_RADIUS) {
      pos[i].y = SIM_MAX_Y - PARTICLE_RADIUS;
      vel[i].y = MULTIPLIER_DAMPING * -vel[i].y;
    }

    //collision with the walls
    if(pos[i].x < SIM_MIN_X + PARTICLE_RADIUS) {
      pos[i].x = SIM_MIN_X + PARTICLE_RADIUS;
      vel[i].x = MULTIPLIER_DAMPING * -vel[i].x;
    }else if(pos[i].x > SIM_MAX_X - PARTICLE_RADIUS) {
      pos[i].x = SIM_MAX_X - PARTICLE_RADIUS;
      vel[i].x = MULTIPLIER_DAMPING * -vel[i].x;
    }
  }
}

__global__ void applyGravity(Vector2* vel, float deltaTime) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < NUM_PARTICLES) {
    vel[i].y -= FORCE_GRAVITY / (PARTICLE_MASS * deltaTime);
  }
}

__global__ void updatePressure(Vector2* pos, float* density, Vector2* pressure) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < NUM_PARTICLES) {
    pressure[i] = calcPressureForce(i, pos, density);
  }
}

__global__ void updateDensity(Vector2* pos, float* density) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < NUM_PARTICLES) {
    density[i] = calcDensity(pos[i], pos);
  }
}

__global__ void update(Vector2* pos, Vector2* vel, Vector2* pressure, float deltaTime) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < NUM_PARTICLES) {
    vel[i].x += MULTIPLIER_FORCE_PRESSURE * pressure[i].x/(PARTICLE_MASS * deltaTime);
    vel[i].y += MULTIPLIER_FORCE_PRESSURE *  pressure[i].y/(PARTICLE_MASS * deltaTime);

    pos[i].x += vel[i].x / deltaTime;
    pos[i].y += vel[i].y / deltaTime;
  }
}

int main() {
  //grid parameters
  float gridLength = PARTICLE_RADIUS * MULTIPLIER_SMOOTHING_RADIUS;
  int gridWidth = std::ceil((SIM_MAX_X - SIM_MIN_X) / gridLength);
  int gridHeight = std::ceil((SIM_MAX_Y - SIM_MIN_Y) / gridLength);
  //assuming numberOfParticles < gridWidth * gridHeight for now
  int gridSize = gridWidth * gridHeight; 

  //memory allocation sizes
  size_t vectorBytes = NUM_PARTICLES * sizeof(Vector2);
  size_t floatBytes = NUM_PARTICLES * sizeof(float);
  size_t intBytes = gridSize * sizeof(int);

  //host vector pointers
  Vector2* h_pos, * h_vel;
  float* h_density;
  Vector2* h_pressure;

  int* h_gridCellLookup;
  int* h_gridStartIndexLookup;
  int* h_gridCountLookup;

  //allocate memory for the host vectors
  h_pos = (Vector2*)malloc(vectorBytes);
  h_vel = (Vector2*)malloc(vectorBytes);
  h_density = (float*)malloc(floatBytes);
  h_pressure = (Vector2*)malloc(vectorBytes);
  h_gridCellLookup = (int*)malloc(intBytes);
  h_gridStartIndexLookup = (int*)malloc(intBytes);
  h_gridCountLookup = (int*)malloc(intBytes);

  //device vector pointers
  Vector2* d_pos, * d_vel;
  float* d_density;
  Vector2* d_pressure;

  int* d_gridCellLookup;
  int* d_gridStartIndexLookup;
  int* d_gridCountLookup;

  //allocate memory for the device vectors
  cudaMalloc(&d_pos, vectorBytes);
  cudaMalloc(&d_vel, vectorBytes);
  cudaMalloc(&d_density, floatBytes);
  cudaMalloc(&d_pressure, vectorBytes);
  cudaMalloc(&d_gridCellLookup, intBytes);
  cudaMalloc(&d_gridStartIndexLookup, intBytes);
  cudaMalloc(&d_gridCountLookup, intBytes);

  //initialize the particles
  initialize(h_pos, h_vel, h_density, h_pressure);
  initializeGrid(gridSize, h_gridCellLookup, h_gridStartIndexLookup, h_gridCountLookup);

  //initialize opengl renderer
  Renderer renderer;
  float triangleLength = PARTICLE_RADIUS * 2 * sqrt(3);  //triangle length from inner circle radius
  renderer.init(1920, 1080, triangleLength);
  renderer.setBoundaries(SIM_MIN_X, SIM_MAX_X, SIM_MIN_Y, SIM_MAX_Y);

  //timer initialization
  std::chrono::steady_clock::time_point timer = std::chrono::steady_clock::now();
  std::chrono::duration<float, std::micro> microDif;

  while (renderer.render(h_pos, NUM_PARTICLES)) {
    //copy the host vectors to the device vectors
    cudaMemcpy(d_pos, h_pos, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_density, h_density, floatBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pressure, h_pressure, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridCellLookup, h_gridCellLookup, intBytes, cudaMemcpyHostToDevice);

    // updateKeyLookup<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>> (d_pos, d_gridKeyLookup, gridSize);
    

    //deltatime
    microDif = std::chrono::steady_clock::now() - timer;
    timer = std::chrono::steady_clock::now();

    //update particles
    checkBoundaries<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>> (d_pos, d_vel);
    applyGravity<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>> (d_vel, microDif.count());
    updateDensity<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>> (d_pos, d_density);
    updatePressure<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>> (d_pos, d_density, d_pressure);
    update<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>> (d_pos, d_vel, d_pressure, microDif.count());

    //copy the device vectors to the host vectors
    cudaMemcpy(h_pos, d_pos, vectorBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel, d_vel, vectorBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_density, d_density, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pressure, d_pressure, vectorBytes, cudaMemcpyDeviceToHost);
  }

  //free memory
  free(h_pos);
  free(h_vel);
  free(h_density);
  free(h_pressure);
  cudaFree(d_pos);
  cudaFree(d_vel);
  cudaFree(d_density);
  cudaFree(d_pressure);

  return 0;
}