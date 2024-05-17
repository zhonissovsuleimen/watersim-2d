#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "renderer.h"
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <random>

//opengl viewport boundaries
#define min_x -1.0f
#define max_x 1.0f
#define min_y -1.0f
#define max_y 1.0f

#define PI 3.1415926535f

#define numberOfParticles 1000
#define particleRadius 0.001f
#define particleMass 1.0f

#define damping 0.3f
#define gravity 9.8f
#define gravityModifier 0.1f


__host__ void particles_init(Vector2* h_pos, Vector2* h_vel) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < numberOfParticles; i++) {
	  h_pos[i].x = dis(gen);
    h_pos[i].y = dis(gen);
    h_vel[i].x = dis(gen);
    h_vel[i].y = dis(gen);
  }
}

__device__ float smoothingKernel(float radius, float distance) {
  if(distance > radius) return 0.0f;
  float value = (radius - distance) * (radius - distance) * (radius - distance);
  float volume = PI * std::pow(radius - distance, 5) / 10;
  return value / volume;
}

__device__ float calcDistance(Vector2 p1, Vector2 p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

__device__ void checkBoundaries(Vector2* pos, Vector2* vel){
  //collision with the floor/ceiling
  if(pos->y <= min_y + particleRadius) {
    pos->y = min_y + particleRadius;
    vel->y = damping * -vel->y;
  }else if(pos->y >= max_y - particleRadius) {
    pos->y = max_y - particleRadius;
    vel->y = damping * -vel->y;
  }

  //collision with the walls
  if(pos->x <= min_x + particleRadius) {
    pos->x = min_x + particleRadius;
    vel->x = damping * -vel->x;
  }else if(pos->x >= max_x - particleRadius) {
    pos->x = max_x - particleRadius;
    vel->x = damping * -vel->x;
  }
}

__global__ void update(Vector2* pos, Vector2* vel, float deltaMicro) {
  float deltaSeconds = deltaMicro / 1000000;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < numberOfParticles) {
    checkBoundaries(&pos[i], &vel[i]);

    //updating the velocity
    vel[i].y += ((-gravity * gravityModifier)/particleMass) * deltaSeconds;

    // updating the position
    pos[i].x = pos[i].x + vel[i].x * deltaSeconds;
    pos[i].y = pos[i].y + vel[i].y * deltaSeconds;
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

	//device vector pointers
	Vector2* d_pos, * d_vel;

	//size of the vectors in bytes
	size_t bytes = numberOfParticles * sizeof(Vector2);

	//allocate memory for the host vectors
	h_pos = (Vector2*)malloc(bytes);
	h_vel = (Vector2*)malloc(bytes);

	//allocate memory for the device vectors
	cudaMalloc(&d_pos, bytes);
	cudaMalloc(&d_vel, bytes);

	//initialize the particles
	particles_init(h_pos, h_vel);

	//initialize opengl renderer
	Renderer renderer;
  
  //triangle length from inner circle radius
  float triangleLength = particleRadius * 2 * sqrt(3);
  renderer.init(WIDTH, HEIGHT, triangleLength);

  std::chrono::steady_clock::time_point timer = std::chrono::steady_clock::now();
  std::chrono::duration<float, std::micro> microDif;
	while (renderer.render(h_pos, 2 * numberOfParticles)) {
		//copy the host vectors to the device vectors
		cudaMemcpy(d_pos, h_pos, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vel, h_vel, bytes, cudaMemcpyHostToDevice);

		//update the particles
    microDif = std::chrono::steady_clock::now() - timer;
    timer = std::chrono::steady_clock::now();
		update<<<NUM_BLOCKS, NUM_THREADS>>> (d_pos, d_vel, microDif.count());

		//copy the device vectors to the host vectors
		cudaMemcpy(h_pos, d_pos, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vel, d_vel, bytes, cudaMemcpyDeviceToHost);
	}

	//free memory
	free(h_pos);
	free(h_vel);
	cudaFree(d_pos);
	cudaFree(d_vel);

	return 0;
}