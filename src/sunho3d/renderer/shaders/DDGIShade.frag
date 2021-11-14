#version 450

#define MAX_LIGHTS 64
#extension GL_GOOGLE_include_directive : require
#define IRD_MAP_SIZE 8
#define IRD_MAP_PROBE_COLS 8

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

    uint numProbe = sceneBuffer.gridNum.x * sceneBuffer.gridNum.y * sceneBuffer.gridNum.z;
    vec3 pos = vec3(texture(positionBuffer, uv));
    vec3 normal = vec3(texture(normalBuffer, uv));
    vec3 v = normalize(constants.cameraPos - pos);
    vec4 diffuse = texture(diffuseBuffer, uv);
    vec4 color = vec4(0.0);
    ivec3 grid = posToGrid(pos, sceneBuffer.gridNum, sceneBuffer.sceneSize);
    vec3 gridSize = sceneBuffer.sceneSize * 2.0 / sceneBuffer.gridNum;
    int mainProbeID = gridToProbeID(grid, sceneBuffer.gridNum);
    vec3 mainProbePos = probeIDToPos(mainProbeID, sceneBuffer.gridNum, sceneBuffer.sceneSize);
    vec3 tt = (pos - mainProbePos) / gridSize;
    float sumWeight = 0.01;
    vec4 sumIrradiance = vec4(0.0);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                ivec3 offset = ivec3(i,j,k);
                ivec3 ogrid = grid + offset;
                int probeID = gridToProbeID(ogrid, sceneBuffer.gridNum);
                if (probeID < 0 || probeID > numProbe) {
                    continue;
                }   
                vec3 probePos = probeIDToPos(probeID, sceneBuffer.gridNum, sceneBuffer.sceneSize);
                vec3 interp = mix(1.0-tt, tt, vec3(offset));
                float weight = interp.x * interp.y * interp.z;

                // back face cull
                vec3 probeToPoint = pos-probePos;
                vec3 lightDir = normalize(-probeToPoint);
                weight *= max(0.05, dot(lightDir, normal));

                vec2 startOffset = vec2((probeID % IRD_MAP_PROBE_COLS) * IRD_MAP_SIZE, (probeID / IRD_MAP_PROBE_COLS) * IRD_MAP_SIZE);
                vec2 uv = octahedronMap(normal);
                vec2 tuv = (startOffset + uv * IRD_MAP_SIZE) / textureSize(probeIrradianceMap,0);
                vec4 irradiance = texture(probeIrradianceMap, tuv);
                
                sumIrradiance += irradiance * weight;
                sumWeight += weight;
            }
        }
    }

    color += diffuse * (sumIrradiance / sumWeight) / (2.0 * M_PI);

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
