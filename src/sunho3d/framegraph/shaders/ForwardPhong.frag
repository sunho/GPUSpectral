#version 450

#define MAX_LIGHTS 64

layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 pos;

layout(location = 0) out vec4 outColor;

struct Light {
    vec3 pos;
    vec3 dir;
    vec2 RI;
};

layout(binding = 1) uniform LightUniformBuffer {
    Light light[MAX_LIGHTS];
    int numLights;
} light;

layout(binding = 0) uniform sampler2D diffuseMap;

void main() {
    vec4 diffuse = texture(diffuseMap, uv);
    vec4 color = diffuse * 0.2;
    for (int i = 0; i < light.numLights; i++) {
        vec3 lightV = light.light[i].pos - pos;
        vec3 light = normalize(lightV);
        float dis = length(lightV);
        float lightI = 1.0/(dis*dis);
        color +=  max(lightI* diffuse * dot(light,normalize(normal)),0.0f);
    }
    outColor = color;
}
