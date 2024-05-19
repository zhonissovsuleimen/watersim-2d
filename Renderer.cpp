#pragma once

#include "renderer.h"
#include "shaders_glsl.h"

#include <iostream>

#include <thread>
#include <chrono>

GLint shaderProgram;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

void Renderer::setBoundaries(float min_x, float max_x, float min_y, float max_y) {
  this->min_x = min_x;
  this->max_x = max_x;
  this->min_y = min_y;
  this->max_y = max_y;
}

bool Renderer::initShaders() {
  // Initialize vertex shader
  std::cout << "Initializing vertex shader" << std::endl;
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, V_SHADER, NULL);
  glCompileShader(vertexShader);

  //Checking for errors
  GLint status;
  glGetProgramiv(vertexShader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    char infolog[512];
    glGetProgramInfoLog(vertexShader, 512, NULL, infolog);
    std::cerr << "Vertex shader compilation failed" << std::endl;
    std::cerr << infolog << std::endl;
    return false;
  }
  std::cout << "Initializing vertex shader complete" << std::endl;

  std::cout << "Initializing fragment shader" << std::endl;
  //Initialize fragment shader
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, F_SHADER, NULL);
  glCompileShader(fragmentShader);

  //Checking for errors
  glGetProgramiv(fragmentShader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    char infolog[512];
    glGetProgramInfoLog(fragmentShader, 512, NULL, infolog);
    std::cerr << "Fragment shader compilation failed" << std::endl;
    std::cerr << infolog << std::endl;
    return false;
  }
  std::cout << "Initializing fragment shader complete" << std::endl;

  //Linking vertex and fragment shaders
  std::cout << "Linking shaders into shader program" << std::endl;
  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  //Checking for errors
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
  if (status != GL_TRUE) {
    char infolog[512];
    glGetProgramInfoLog(shaderProgram, 512, NULL, infolog);
    std::cerr << "Shader linking failed" << std::endl;
    std::cerr << infolog << std::endl;
    return false;
  }
  std::cout << "Linking shaders complete" << std::endl;

  glUseProgram(shaderProgram);
  return true;
}

bool Renderer::init(unsigned int width, unsigned int height, float triangle_length) {
  this->triangle_length = triangle_length;
  std::cout << "Initializing OpenGL libraries" << std::endl;
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }
  std::cout << "Initializing OpenGL libraries complete" << std::endl;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // required for MacOS
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


  std::cout << "Creating OpenGL window" << std::endl;
  window = glfwCreateWindow(width, height, "2D water simualtion", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create OpenGL window" << std::endl;
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, key_callback); // escape key to close window

  glewExperimental = GL_TRUE;
  glewInit();
  glEnable(GL_DEPTH_TEST); // fixes issues with depth 
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  if (!initShaders()) { return false; }

  glGenBuffers(1, &vbo);
  glGenVertexArrays(1, &vao);

  return true;
}

void Renderer::fitToScreen(float* vertexPositions, size_t size) {
  float x_range = max_x - min_x;
  float y_range = max_y - min_y;

  for (int i = 0; i < size; i += 2) {
    vertexPositions[i] = (vertexPositions[i] - min_x) / x_range;
    vertexPositions[i] = 2 * vertexPositions[i] - 1;
    vertexPositions[i + 1] = (vertexPositions[i + 1] - min_y) / y_range;
    vertexPositions[i + 1] = 2 * vertexPositions[i + 1] - 1;
  }
}

float* generateTriangles(Vector2* positions, size_t positions_size, float triangle_length) {
  float* triangles = new float[6 * positions_size];
  const float sqrt3 = 1.7320508f;
  float sqrt3_times_length = sqrt3 * triangle_length;

  for (int i = 0; i < positions_size; i++) {
    float x = positions[i].x;
    float y = positions[i].y;

    triangles[6 * i] = x;
    triangles[6 * i + 1] = y + sqrt3_times_length / 3;

    triangles[6 * i + 2] = x - triangle_length / 2;
    triangles[6 * i + 3] = y - sqrt3_times_length / 6;

    triangles[6 * i + 4] = x + triangle_length / 2;
    triangles[6 * i + 5] = y - sqrt3_times_length / 6;
  }
  return triangles;
}

void Renderer::bind(Vector2* positions, size_t size) {
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  float* triangles = generateTriangles(positions, size, this->triangle_length);
  fitToScreen(triangles, 6 * size);
  glBufferData(GL_ARRAY_BUFFER, 6 * size * sizeof(float), triangles, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
  glEnableVertexAttribArray(0);
  delete[] triangles;
}

void unbind() {
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

bool Renderer::render(Vector2* positions, size_t size) {
  if (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    bind(positions, size);
    glDrawArrays(GL_TRIANGLES, 0, 3 * size);
    unbind();

    glfwSwapBuffers(window);
    glfwPollEvents();

    return true;
  }
  else {
    glfwDestroyWindow(window);
    glfwTerminate();
    return false;
  }
}