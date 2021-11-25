#version 450
#pragma shader_stage(vertex)

#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTex;

layout(location = 0) out vec3 pos;
layout(location = 1) out vec3 worldPos;

layout(binding = 0) uniform ProbeDebugUniformBuffer {
    mat4 MVP;
    mat4 model;
    SceneInfo sceneInfo;
    int probeId;
} uniforms;

void main() {
    gl_Position = uniforms.MVP*vec4(inPosition, 1.0);
    worldPos = vec3(uniforms.model*vec4(inPosition,1.0));
    pos = vec3(inPosition.x, inPosition.y, inPosition.z);
    
}
