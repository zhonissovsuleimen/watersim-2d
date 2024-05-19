#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "renderer.h"
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <random>

//simulation boundaries
#define min_x 0.0f
#define max_x 1920.0f
#define min_y 0.0f
#define max_y 1080.0f

#define PI 3.1415926535f

#define numberOfParticles 3000
#define particleRadius 6.0f
#define particleMass 1.0f
#define targetDensity 10.0f

#define damping 0.0f
#define gravityForce 10.0f

#define smoothingRadiusMultiplier 4.0f 
#define pressureForceMultiplier 100000.0f


__host__ void initialize(Vector2* h_pos, Vector2* h_vel, float* h_density, Vector2* h_pressure) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disx(min_x + particleRadius, max_x - particleRadius);
  std::uniform_real_distribution<float> disy(min_y + particleRadius, max_y - particleRadius);
  std::uniform_real_distribution<float> disv(-10.0f, 10.0f);

  for (int i = 0; i < numberOfParticles; i++) {
    h_pos[i].x = disx(gen);
    h_pos[i].y = disy(gen);
    h_vel[i].x = 0.0f;
    h_vel[i].y = 0.0f;
    h_density[i] = 0.0f;
    h_pressure[i].x = 0.0f;
    h_pressure[i].y = 0.0f;
  }
}

__device__ float smoothingKernel(float smoothingRadius, float distance) {
  float q = distance / smoothingRadius;
  float influence = 0.0f;
  // float volume = (2 * PI * smoothingRadius * smoothingRadius) / 5.0f;
  if(q >= 0 && q <= 0.5f){
    influence = 6 * (q * q * q - q * q) + 1;
  } else if (q > 0.5f && q <= 1.0f){
    influence = 2 * (1 - q) * (1 - q) * (1 - q);
    // volume *= 0.5f;
  }
  return influence;
}

__device__ float smoothingKernelDerivative(float smoothingRadius, float distance) {
  float q = distance / smoothingRadius;
  float influence = 0.0f;
  if(q >= 0 && q <= 0.5f){
    influence = (15*distance*(3*distance - 2*smoothingRadius)) / (PI * std::pow(smoothingRadius, 5));
  } else if (q > 0.5f && q <= 1.0f){
    influence = 6*(smoothingRadius - distance)*(smoothingRadius - distance) / std::pow(smoothingRadius, 3);
  }
  return influence;
}

__device__ float smoothingKernelTest(float smoothingRadius, float distance) {
  if (distance > smoothingRadius) return 0.0f;

  float value = smoothingRadius - distance;
  // float volume = (PI * std::pow(smoothingRadius, 4)) / 6.0f;
  return value * value;
}

__device__ float smoothingKernelDerivativeTest(float smoothingRadius, float distance) {
  if (distance > smoothingRadius) return 0.0f;

  float value = smoothingRadius - distance;
  
  float scale = 12/(PI * std::pow(smoothingRadius, 4));
  return value * scale;
}

__device__ float calcDistance(Vector2 p1, Vector2 p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

__global__ void checkBoundaries(Vector2* pos, Vector2* vel){
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if(i < numberOfParticles) {
    //collision with the floor/ceiling
    if(pos[i].y < min_y + particleRadius) {
      pos[i].y = min_y + particleRadius;
      vel[i].y = damping * -vel[i].y;
    }else if(pos[i].y > max_y - particleRadius) {
      pos[i].y = max_y - particleRadius;
      vel[i].y = damping * -vel[i].y;
    }

    //collision with the walls
    if(pos[i].x < min_x + particleRadius) {
      pos[i].x = min_x + particleRadius;
      vel[i].x = damping * -vel[i].x;
    }else if(pos[i].x > max_x - particleRadius) {
      pos[i].x = max_x - particleRadius;
      vel[i].x = damping * -vel[i].x;
    }
  }
}

__device__ float calcDensity(Vector2 particle, Vector2* pos) {
  float density = 0.0f;
  for(int i = 0; i < numberOfParticles; i++) {
    float distance = calcDistance(particle, pos[i]);
    density += particleMass * smoothingKernel(smoothingRadiusMultiplier * particleRadius, distance);
  }
  return density;
}

__device__ Vector2 calcPressureForce(int particleId, Vector2* pos, float* density) {
  Vector2 pressureForce = {0.0f, 0.0f};
  for(int j = 0; j < numberOfParticles; j++) {
    if (j == particleId) continue;
    float distance = calcDistance(pos[particleId], pos[j]);
    float directionX, directionY;
    if (distance == 0){
      //pseudo random
      directionX = ((int)density[particleId] % 2 == 0 ? 1.0f : -1.0f);
      directionY = ((int)density[j] % 2 == 0 ? 1.0f : -1.0f);
    }else{
      directionX = (pos[particleId].x - pos[j].x)/distance;
      directionY = (pos[particleId].y - pos[j].y)/distance;
    }
    
    float influence = smoothingKernelDerivative(smoothingRadiusMultiplier * particleRadius, distance);

    float pressureAtI = particleMass * (density[particleId] - targetDensity);
    float pressureAtJ = particleMass * (density[j] - targetDensity);
    float sharedPressure = -(pressureAtI + pressureAtJ) / 2.0f;
    
    pressureForce.x += sharedPressure * directionX * influence/density[particleId];
    pressureForce.y += sharedPressure * directionY * influence/density[particleId];
  }
  return pressureForce;
}

__global__ void applyGravity(Vector2* vel, float deltaTime) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numberOfParticles) {
    vel[i].y -= gravityForce / (particleMass * deltaTime);
  }
}

__global__ void updatePressure(Vector2* pos, float* density, Vector2* pressure) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numberOfParticles) {
    pressure[i] = calcPressureForce(i, pos, density);
  }
}

__global__ void updateDensity(Vector2* pos, float* density) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numberOfParticles) {
    density[i] = calcDensity(pos[i], pos);
  }
}

__global__ void update(Vector2* pos, Vector2* vel, Vector2* pressure, float deltaTime) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numberOfParticles) {
    vel[i].x += pressureForceMultiplier * pressure[i].x/(particleMass * deltaTime);
    vel[i].y += pressureForceMultiplier *  pressure[i].y/(particleMass * deltaTime);

    pos[i].x += vel[i].x / deltaTime;
    pos[i].y += vel[i].y / deltaTime;
  }
}

int main() {
  const unsigned int WIDTH = 1920;
  const unsigned int HEIGHT = 1080;

  //simulation parameters
  int NUM_THREADS = 256;
  int NUM_BLOCKS = (numberOfParticles + NUM_THREADS - 1) / NUM_THREADS;

  //host vector pointers
  Vector2* h_pos, * h_vel;
  float* h_density;
  Vector2* h_pressure;

  //device vector pointers
  Vector2* d_pos, * d_vel;
  float* d_density;
  Vector2* d_pressure;

  //size of the vectors in bytes
  size_t vectorBytes = numberOfParticles * sizeof(Vector2);
  size_t floatBytes = numberOfParticles * sizeof(float);

  //allocate memory for the host vectors
  h_pos = (Vector2*)malloc(vectorBytes);
  h_vel = (Vector2*)malloc(vectorBytes);
  h_density = (float*)malloc(floatBytes);
  h_pressure = (Vector2*)malloc(vectorBytes);

  //allocate memory for the device vectors
  cudaMalloc(&d_pos, vectorBytes);
  cudaMalloc(&d_vel, vectorBytes);
  cudaMalloc(&d_density, floatBytes);
  cudaMalloc(&d_pressure, vectorBytes);

  //initialize the particles
  initialize(h_pos, h_vel, h_density, h_pressure);

  //initialize opengl renderer
  Renderer renderer;
  
  //triangle length from inner circle radius
  float triangleLength = particleRadius * 2 * sqrt(3);
  renderer.init(WIDTH, HEIGHT, triangleLength);
  renderer.setBoundaries(min_x, max_x, min_y, max_y);

  std::chrono::steady_clock::time_point timer = std::chrono::steady_clock::now();
  std::chrono::duration<float, std::micro> microDif;
  while (renderer.render(h_pos, numberOfParticles)) {
  //copy the host vectors to the device vectors
    cudaMemcpy(d_pos, h_pos, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_density, h_density, floatBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pressure, h_pressure, vectorBytes, cudaMemcpyHostToDevice);

    //update the particles
    microDif = std::chrono::steady_clock::now() - timer;
    timer = std::chrono::steady_clock::now();
    checkBoundaries<<<NUM_BLOCKS, NUM_THREADS>>> (d_pos, d_vel);
    applyGravity<<<NUM_BLOCKS, NUM_THREADS>>> (d_vel, microDif.count());
    updateDensity<<<NUM_BLOCKS, NUM_THREADS>>> (d_pos, d_density);
    updatePressure<<<NUM_BLOCKS, NUM_THREADS>>> (d_pos, d_density, d_pressure);
    update<<<NUM_BLOCKS, NUM_THREADS>>> (d_pos, d_vel, d_pressure, microDif.count());

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