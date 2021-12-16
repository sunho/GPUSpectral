#version 450
#pragma shader_stage(fragment)

#define MAX_LIGHTS 64

layout(location = 0) in vec2 inPos;
layout(location = 0) out vec4 outColor;

layout(set=1,binding = 0) uniform sampler2D tex;

vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0);
}

void main() {
    vec2 uv = inPos / 2.0 + 0.5;
    vec4 diffuse = texture(tex, uv);
    outColor = vec4(ACESFilm(vec3(diffuse)),1.0);
}
