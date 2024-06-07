#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>


struct Vector2 {
  float x;
  float y;
};

class Renderer {
private:
	GLFWwindow* window;
	float triangle_length;
	GLuint vao;
	GLuint vbo;
	float min_x, max_x, min_y, max_y;

	bool initShaders();
	void bind(Vector2* positions, size_t size);
	void fitToScreen(float* vertexPos, size_t size);

public:
	bool init(unsigned int width, unsigned int height, float triangle_length);
	bool render(Vector2* position, size_t size);
	void setBoundaries(float min_x, float max_x, float min_y, float max_y);
};

extern "C" {
	bool init(unsigned int width, unsigned int height, float triangle_length);
	bool render(Vector2* position, size_t size);
	void setBoundaries(float min_x, float max_x, float min_y, float max_y);
}