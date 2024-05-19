#pragma once
#include "parameters.h"
#include "renderer.h"
#include "smoothing_kernels.cuh"

__device__ inline float calcDistance(Vector2 p1, Vector2 p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

__device__ inline float calcDensity(Vector2 particle, Vector2* pos) {
  float density = 0.0f;
  for(int i = 0; i < NUM_PARTICLES; i++) {
    float distance = calcDistance(particle, pos[i]);
    density += PARTICLE_MASS * smoothingKernel(PARTICLE_RADIUS * MULTIPLIER_SMOOTHING_RADIUS, distance);
  }
  return density;
}

__device__ inline Vector2 calcPressureForce(int particleId, Vector2* pos, float* density) {
  Vector2 pressureForce = {0.0f, 0.0f};
  for(int j = 0; j < NUM_PARTICLES; j++) {
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
    
    float influence = smoothingKernelDerivative(PARTICLE_RADIUS * MULTIPLIER_SMOOTHING_RADIUS, distance);

    float pressureAtI = PARTICLE_MASS * (density[particleId] - DENSITY_TARGET);
    float pressureAtJ = PARTICLE_MASS * (density[j] - DENSITY_TARGET);
    float sharedPressure = -(pressureAtI + pressureAtJ) / 2.0f;
    
    pressureForce.x += sharedPressure * directionX * influence/density[particleId];
    pressureForce.y += sharedPressure * directionY * influence/density[particleId];
  }
  return pressureForce;
}

__device__ inline int calcCellIndex(float x, float y) {
  //copied from cuda kernel
  float gridLength = PARTICLE_RADIUS * MULTIPLIER_SMOOTHING_RADIUS;
  int gridWidth = std::ceil((SIM_MAX_X - SIM_MIN_X) / gridLength);
  int gridHeight = std::ceil((SIM_MAX_Y - SIM_MIN_Y) / gridLength);

  int cellX = (x - SIM_MIN_X) / gridLength;
  int cellY = (y - SIM_MIN_Y) / gridLength;
  return cellY * gridWidth + cellX;
}