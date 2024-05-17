#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>


struct ParticleCenter {
  float x;
  float y;
};

class Renderer {
private:
	GLFWwindow* window;
	float triangle_length;
	GLuint vao;
	GLuint vbo;

	bool initShaders();
	void bind(ParticleCenter* positions, size_t size);

public:
	bool init(unsigned int width, unsigned int height, float triangle_length);
	bool render(ParticleCenter* position, size_t size);
};

extern "C" {
	bool init(unsigned int width, unsigned int height, float triangle_length);
	bool render(ParticleCenter* position, size_t size);
}