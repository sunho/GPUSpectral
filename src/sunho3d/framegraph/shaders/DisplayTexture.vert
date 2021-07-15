#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 pos;

void main() {
    gl_Position = vec4(inPosition,0.5, 1.0);
    pos = inPosition;
}
