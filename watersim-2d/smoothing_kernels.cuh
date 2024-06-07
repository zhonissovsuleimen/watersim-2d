#pragma once

__device__ inline float smoothingKernel(float smoothingRadius, float distance) {
  // float PI = 3.1415926535f;
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

__device__ inline float smoothingKernelDerivative(float smoothingRadius, float distance) {
  float PI = 3.1415926535f;
  float q = distance / smoothingRadius;
  float influence = 0.0f;
  if(q >= 0 && q <= 0.5f){
    influence = (15*distance*(3*distance - 2*smoothingRadius)) / (PI * std::pow(smoothingRadius, 5));
  } else if (q > 0.5f && q <= 1.0f){
    influence = 6*(smoothingRadius - distance)*(smoothingRadius - distance) / std::pow(smoothingRadius, 3);
  }
  return influence;
}

__device__ inline float smoothingKernelTest(float smoothingRadius, float distance) {
  if (distance > smoothingRadius) return 0.0f;
  // float PI = 3.1415926535f;

  float value = smoothingRadius - distance;
  // float volume = (PI * std::pow(smoothingRadius, 4)) / 6.0f;
  return value * value;
}

__device__ inline float smoothingKernelDerivativeTest(float smoothingRadius, float distance) {
  if (distance > smoothingRadius) return 0.0f;
  float PI = 3.1415926535f;

  float value = smoothingRadius - distance;
  
  float scale = 12/(PI * std::pow(smoothingRadius, 4));
  return value * scale;
}