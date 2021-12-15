
#pragma once

#include "Common.cuh"

struct TriangleLight {
    float3 positions[3];
    float3 radinace;
};

struct LightData {
    Array<TriangleLight> triangleLights;
};

HOSTDEVICE CUDAINLINE float3 sampleTriangleLight(const TriangleLight& light, SamplerState& sampler, float& pdf) {
    // barycentric coordinates
    // u = 1 - sqrt(e_1)
    // v = e_2 * sqrt(e_1)
    float e1 = randUniform(sampler);
    float e2 = randUniform(sampler);
    float u = 1.0f - sqrt(e1);
    float v = e2 * sqrt(e1);
    float w = 1.0f - u - v;
    float A = 0.5f * abs(length(cross(light.positions[2] - light.positions[0], light.positions[1] - light.positions[0])));
    pdf = 1.0f / A;
    return u * light.positions[0] + v * light.positions[1] + w * light.positions[2];
}

HOSTDEVICE CUDAINLINE void sampleLight(const LightData& lightData, SamplerState& sampler, float3& p, float& pdf, float3& emission) {
    uint32_t lightIndex = randInt(sampler) % lightData.triangleLights.size();
    const TriangleLight& light = lightData.triangleLights[lightIndex];
    emission = light.radinace;
    p = sampleTriangleLight(light, sampler, pdf);
    pdf /= (float)lightData.triangleLights.size();
}
