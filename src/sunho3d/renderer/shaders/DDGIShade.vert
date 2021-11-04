#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 pos;

void main() {
    vec2 p = vec2(inPosition.x, -inPosition.y);
    gl_Position = vec4(p,1.0, 1.0);
    pos = p;
}
