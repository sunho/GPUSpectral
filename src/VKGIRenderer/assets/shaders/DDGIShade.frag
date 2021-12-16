#version 450
#pragma shader_stage(fragment)

#define MAX_LIGHTS 64
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.glsl"

#include "probe.glsl"

layout(location = 0) in vec2 inPos;
layout(location = 0) out vec4 outColor;

struct Light {
    vec3 pos;
    vec3 dir;
    vec2 RI;
    vec4 radiance;
};

layout(std140, binding = 0) uniform _SceneBuffer{
    SceneBuffer sceneBuffer;
};

layout(std140, binding = 1) uniform LightUniformBuffer {
    Light light[MAX_LIGHTS];
    mat4 lightVP[MAX_LIGHTS];
    int numLights;
} light;

layout(set=1,binding = 0) uniform sampler2D positionBuffer;
layout(set=1,binding = 1) uniform sampler2D normalBuffer;
layout(set=1,binding = 2) uniform sampler2D diffuseBuffer;
layout(set=1,binding = 3) uniform sampler2D emissionBuffer;
layout(set=1,binding = 4) uniform sampler2D probeIrradianceMap;
layout(set=1,binding = 5) uniform sampler2D probeDepthMap;
layout(set=1,binding = 6) uniform samplerCube pointShadowMaps[MAX_LIGHTS];

layout( push_constant ) uniform PushConstants {
    vec3 cameraPos;
} constants;

void main() {
    vec2 uv = inPos / 2.0 + 0.5;

    uint numProbe = sceneBuffer.sceneInfo.gridNum.x * sceneBuffer.sceneInfo.gridNum.y * sceneBuffer.sceneInfo.gridNum.z;
    vec3 pos = vec3(texture(positionBuffer, uv));
    vec3 normal = vec3(texture(normalBuffer, uv));
    vec3 v = normalize(constants.cameraPos - pos);
    vec4 diffuse = texture(diffuseBuffer, uv);
    vec4 color = vec4(0.0);
    ivec3 grid = posToGrid(pos, sceneBuffer.sceneInfo);
    vec3 gridSize = sceneBuffer.sceneInfo.sceneSize * 2.0 / vec3(sceneBuffer.sceneInfo.gridNum);
    int mainProbeID = gridToProbeID(grid, sceneBuffer.sceneInfo);
    ivec3 grid2 = probIDToGrid(mainProbeID, sceneBuffer.sceneInfo);
    vec3 mainProbePos = probeIDToPos(mainProbeID, sceneBuffer.sceneInfo);
    vec3 tt = (pos - mainProbePos) / gridSize;
    float sumWeight = 0.01;
    vec4 sumIrradiance = vec4(0.0);
    //outColor = vec4(grid / vec3(sceneBuffer.sceneInfo.gridNum), 0.0);
    //return;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                ivec3 offset = ivec3(i,j,k);
                ivec3 ogrid = grid + offset;
                ogrid = clamp(ogrid, ivec3(0), ivec3(sceneBuffer.sceneInfo.gridNum - 1));
                int probeID = gridToProbeID(ogrid, sceneBuffer.sceneInfo);
     
                // if (ogrid.x < 0 || ogrid.x >= sceneBuffer.sceneInfo.gridNum.x || ogrid.y < 0 || ogrid.y >= sceneBuffer.sceneInfo.gridNum.y || ogrid.z < 0 || ogrid.z >= sceneBuffer.sceneInfo.gridNum.z) {
                //     outColor = vec4(1.0, 0.0, 0.0, 1.0);
                //     return;
                //     continue;
                // }   

                vec3 probePos = probeIDToPos(probeID, sceneBuffer.sceneInfo);
                vec3 interp = mix(1.0-tt, tt, vec3(offset));
                float weight = interp.x * interp.y * interp.z;

                // back face cull
                vec3 probeToPoint = pos-probePos;
                vec3 lightDir = normalize(-probeToPoint);
                float distToProbe = length(pos + 0.1*normal - probePos);
                //distToProbe += 0.05;
                weight *= max(0.00, dot(lightDir, normal));

                // visibility testing
                vec2 depthUv = vec2(getDepthTexOffset(probeID, octahedronMap(lightDir))) / textureSize(probeDepthMap,0);
                vec4 dist = texture(probeDepthMap, depthUv);

                float mean = dist.x;
                float variance = abs(dist.y - (mean * mean));
                float t_sub_mean = distToProbe - mean;
                float chebychev = variance / (variance + (t_sub_mean * t_sub_mean));
              weight *= ((distToProbe <= mean) ? 1.0 : max(chebychev, 0.0));

                vec2 irdUv = vec2(getIRDTexOffset(probeID, octahedronMap(normal))) / textureSize(probeIrradianceMap,0);
                vec4 irradiance = texture(probeIrradianceMap, irdUv);
                
                sumIrradiance += irradiance * weight;
                sumWeight += weight;
            }
        }
    }

    vec4 emission = texture(emissionBuffer, uv);
    vec4 brdf = diffuse / M_PI;
    color += (brdf) * (sumIrradiance / sumWeight);
    color += emission;

    for (int i = 0; i < light.numLights; i++) {
        vec3 lightV = light.light[i].pos - pos;
        vec4 lightR =  light.light[i].radiance;
        float depth = length(lightV);
        vec3 p = normalize(-lightV);
        //p.y *= -1.0;
        //p.z *= -1.0;
        //p.x *= -1.0;
        float dist = texture(pointShadowMaps[nonuniformEXT(i)], vec3(p)).r;
        dist *= 25.0f;
        if (depth - 0.01 > dist) {
            continue;
        }
        vec3 light = normalize(lightV);
        vec3 h = normalize(light + v);
        float dis = length(lightV);
        float lightI = 0.5/(dis*dis);
        color += diffuse * lightR / M_PI * max(dot(light,normal),0.0f);
    }
    outColor = vec4(vec3(color), 1.0);
}