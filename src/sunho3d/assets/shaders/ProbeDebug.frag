#version 450
#pragma shader_stage(fragment)

#define MAX_LIGHTS 64

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "probe.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inWorldPos;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outDiffuse;
layout(location = 3) out vec4 outEmission;

layout(binding = 0) uniform ProbeDebugUniformBuffer {
    mat4 MVP;
    mat4 model;
    SceneInfo sceneInfo;
    int probeId;
} uniforms;

layout(set=1,binding = 0) uniform sampler2D probeIrradianceMap;
layout(set=1,binding = 1) uniform sampler2D probeDepthMap;

void main() {
    outDiffuse = vec4(0.0);
    vec3 normal = normalize(inPos);
    ivec2 startOffset = probeIDToIRDTexOffset(uniforms.probeId);
    vec2 uv = octahedronMap(normal);
    vec2 tuv = (startOffset + uv * IRD_MAP_SIZE) / textureSize(probeIrradianceMap,0);
    vec4 irradiance = textureLod(probeIrradianceMap, tuv, 0);

    ivec2 depthStartOffset = probeIDToDepthTexOffset(uniforms.probeId);
    vec2 depthUv = octahedronMap(normal);
    vec2 depthTuv = (depthStartOffset + depthUv * DEPTH_MAP_SIZE) / textureSize(probeDepthMap,0);
    vec4 dist = texture(probeDepthMap, depthTuv);

    vec3 color = vec3(dist.x);
    outEmission = vec4((normal+1.0)/2.0, 1.0);
    outNormal = vec4(0.0);
    outPos = vec4(0.0);
}
