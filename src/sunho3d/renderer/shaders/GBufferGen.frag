#version 450

#define MAX_LIGHTS 64

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inPos;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outDiffuse;

layout(binding = 0) uniform TransformUniformBuffer {
    mat4 MVP;
    mat4 model;
    mat4 invModelT;
    vec3 cameraPos;
} transform;

layout(set=1,binding = 0) uniform sampler2D diffuseMap;

void main() {
    vec3 normal = normalize(inNormal);
    vec3 v = normalize(transform.cameraPos - inPos);
    vec4 diffuse = texture(diffuseMap, inUV);
    outNormal = vec4(normal, 0.0);
    outPos = vec4(inPos, 0.0);
    outDiffuse = diffuse;
}
