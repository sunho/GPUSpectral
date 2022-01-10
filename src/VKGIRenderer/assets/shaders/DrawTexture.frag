#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 inPos;
layout(location = 0) out vec4 outColor;

layout(set=0,binding = 0) uniform sampler2D tex;

void main() {
    vec2 uv = inPos / 2.0 + 0.5;
    vec4 diffuse = texture(tex, uv);
    outColor = diffuse;
}
