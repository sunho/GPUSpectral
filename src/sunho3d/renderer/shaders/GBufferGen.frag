#version 450

#define MAX_LIGHTS 64

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

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

layout(binding = 1) uniform MaterialBuffer {
    Material material;
} material;

layout(set=1,binding = 0) uniform sampler2D diffuseMap;

void main() {
    vec3 normal = normalize(inNormal);
    vec3 v = normalize(transform.cameraPos - inPos);
    if (material.material.typeID != MATERIAL_DIFFUSE_COLOR) {
        vec3 diffuse = texture(diffuseMap, inUV).rgb;
        outDiffuse = vec4(diffuse, 1.0);
    } else {
        outDiffuse = vec4(material.material.diffuseColor, 1.0);
    }
    outNormal = vec4(normal, 0.0);
    outPos = vec4(inPos, 0.0);
}
