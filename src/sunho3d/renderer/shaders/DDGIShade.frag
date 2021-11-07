#version 450

#define MAX_LIGHTS 64
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

#include "probe.glsl"

layout(location = 0) in vec2 inPos;
layout(location = 0) out vec4 outColor;

struct Light {
    vec3 pos;
    vec3 dir;
    vec2 RI;
};

layout(std140, binding = 0) uniform SceneBuffer {
    uvec2 frameSize;
    uint instanceNum;
    Instance instances[MAX_INSTANCES];
    uvec3 gridNum;
    vec3 sceneSize;
} sceneBuffer;

layout(std140, binding = 1) uniform LightUniformBuffer {
    Light light[MAX_LIGHTS];
    mat4 lightVP[MAX_LIGHTS];
    int numLights;
} light;

layout(set=1,binding = 0) uniform sampler2D positionBuffer;
layout(set=1,binding = 1) uniform sampler2D normalBuffer;
layout(set=1,binding = 2) uniform sampler2D diffuseBuffer;
layout(set=1,binding = 3) uniform sampler2D probeIrradianceMap;

layout( push_constant ) uniform PushConstants {
    vec3 cameraPos;
} constants;


void main() {
    vec2 uv = inPos / 2.0 + 0.5;

    vec3 normal = vec3(texture(positionBuffer, uv));
    vec3 pos = vec3(texture(normalBuffer, uv));
    vec3 v = normalize(constants.cameraPos - pos);
    vec4 diffuse = texture(diffuseBuffer, uv);
    vec4 color = diffuse * 0.35;
    uvec3 grid = posToGrid(pos, sceneBuffer.gridNum, sceneBuffer.sceneSize); 
    uint p0 = gridToProbeID(grid + uvec3(0,0,0), sceneBuffer.gridNum);
    uint p1 = gridToProbeID(grid + uvec3(0,0,1), sceneBuffer.gridNum);
    uint p2 = gridToProbeID(grid + uvec3(0,1,0), sceneBuffer.gridNum);
    uint p3 = gridToProbeID(grid + uvec3(0,1,1), sceneBuffer.gridNum);
    uint p4 = gridToProbeID(grid + uvec3(1,0,0), sceneBuffer.gridNum);
    uint p5 = gridToProbeID(grid + uvec3(1,0,1), sceneBuffer.gridNum);
    uint p6 = gridToProbeID(grid + uvec3(1,1,0), sceneBuffer.gridNum);
    uint p7 = gridToProbeID(grid + uvec3(1,1,1), sceneBuffer.gridNum);
    


    for (int i = 0; i < light.numLights; i++) {
        vec3 lightV = light.light[i].pos - pos;
        vec3 light = normalize(lightV);
        vec3 h = normalize(light + v);
        float dis = length(lightV);
        float lightI = 1.0/(dis*dis);
        color += lightI* diffuse * max(dot(light,normal),0.0f);
    }
    outColor = color;
}
