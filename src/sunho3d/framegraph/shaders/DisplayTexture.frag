#version 450

#define MAX_LIGHTS 64

layout(location = 0) in vec2 inPos;
layout(location = 0) out vec4 outColor;

layout(set=1,binding = 0) uniform sampler2D tex;

void main() {
    vec2 uv = inPos / 2.0 + 0.5;
    uv.y *= -1;
    vec4 diffuse = texture(tex, uv);
    outColor = diffuse;
}
