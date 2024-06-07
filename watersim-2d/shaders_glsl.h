#pragma once

const char* FRAGMENT_SHADER = R"(
#version 410 core

out vec4 fragColor;

void main() {
  fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}


)";

const char* VERTEX_SHADER = R"(

#version 410 core

layout (location = 0) in vec2 inPosition;

void main() {
  gl_Position = vec4(inPosition.xy, 0.0, 1.0);
}


)";

const char* const F_SHADER[] = { FRAGMENT_SHADER };
const char* const V_SHADER[] = { VERTEX_SHADER };

