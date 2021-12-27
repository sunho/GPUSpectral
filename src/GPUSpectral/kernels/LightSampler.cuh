
#pragma once

#include "Common.cuh"

struct LightOutput {
    float3 position;
    float3 emission;
    float pdf;
};

struct TriangleLight {
    float3 positions[3];
    float3 radiance;

    HOSTDEVICE CUDAINLINE float3 getPower() const {
        float A = 0.5f * abs(length(cross(positions[2] - positions[0], positions[1] - positions[0])));
        return A * M_PI * radiance;
    }

    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 pos, LightOutput* output) const {
        // barycentric coordinates
        // u = 1 - sqrt(e_1)
        // v = e_2 * sqrt(e_1)
        float e1 = randUniform(sampler);
        float e2 = randUniform(sampler);
        float u = 1.0f - sqrt(e1);
        float v = e2 * sqrt(e1);
        float w = 1.0f - u - v;
        float3 v0 = positions[0];
        float3 v1 = positions[1];
        float3 v2 = positions[2];
        float A = 0.5f * abs(length(cross(positions[2] - positions[0], positions[1] - positions[0])));
        float3 normal = normalize( cross( v1-v0, v2-v0 ) );
        float3 lightPos = u * positions[0] + v * positions[1] + w * positions[2];
        float ldist = length(lightPos - pos);
        float3 l = normalize(lightPos - pos);
        output->pdf = ldist * ldist / (fabs(dot(-l, normal)) * A);
        output->position = lightPos;
        output->emission = radiance * float(dot(-l, normal) > 0);
    }
};


struct PieceDist {
    Array<float> pdfs;

    HOSTDEVICE CUDAINLINE int sample(SamplerState& sampler, float& pdf) const {
        float u = randUniform(sampler);
        float cdf = 0.0f;
        int outIndex = -1;
        for (size_t i = 0; i < pdfs.size(); ++i) {
            cdf += pdfs[i];
            if (u <= cdf && outIndex == -1) {
                pdf = pdfs[i];
                outIndex = i;
            }
        }
        return outIndex;
    }
};

struct EnvmapLight {
    cudaTextureObject_t envmap;
    float2 size;
    float3 center;
    float radius;
    PieceDist yDist;
    Array<PieceDist> xDists;
    mat4 toWorld;
    mat4 toWorldInv;

    HOSTDEVICE CUDAINLINE float3 getPower() const {
        return make_float3(tex2D<float4>(envmap, 0.5, 0.5)) * M_PI * M_PI * 4.0 * radius * radius;
    }

    // Mitsuba env map coordinate system
    // -Z axis is mapped to u = 0
    // +X axis is mapped to u = pi /2
    // and v is 0.0 if y is positive (y axis looks down to the bottom)
    // When we load texture, y axis is opposite (y axsis looks up to the top)
    // atan2f(z,x) = 0 when it's at +X axis
    // u = (phi + pi/2) / (2 * pi)
    // v = -theta / pi
    HOSTDEVICE CUDAINLINE float3 sample(SamplerState& sampler, float3 pos, LightOutput *output) const {
        float ypdf;
        int y = yDist.sample(sampler, ypdf);
        float xpdf;
        int x = xDists[y].sample(sampler, xpdf);
        float2 uv = make_float2(x, y) / size;
        float phi = uv.x * 2 * M_PI - M_PI/2.0f;
        float theta = -uv.y * M_PI;
        float3 w = make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        w = normalize(make_float3(toWorld * make_float4(w, 0.0f)));
        output->position = center + w * radius;
        output->emission = make_float3(tex2D<float4>(envmap, uv.x, uv.y));
        /*printf("e: %f %f %f\n", output->emission.x, output->emission.y, output->emission.z);
        printf("p: %f %f %f\n", output->position.x, output->position.y, output->position.z);
        printf("pdf: %f %f\n", xpdf, ypdf);*/
        output->pdf = ypdf * xpdf / (fabs(2.0f * M_PI * M_PI* sin(theta))); // jacobian 2 * pi^2 from scaling, sin from unifrom sphere mapping
    }

    HOSTDEVICE CUDAINLINE float3 lookupEmission(float3 wi) const {
        wi = normalize(make_float3(toWorldInv * make_float4(wi, 0.0f)));
        float2 uv = make_float2(atan2f(wi.z, wi.x) / (2.0f * M_PI),
            acos(clamp(wi.y, -1.0f, 1.0f))/ M_PI);
        uv.y *= -1;
        return make_float3(tex2D<float4>(envmap, uv.x + 0.25f, uv.y));
    }
};

struct LightData {
    Array<TriangleLight> triangleLights;
    EnvmapLight envmapLight;
    PieceDist lightDist;
};

HOSTDEVICE CUDAINLINE void sampleLight(const LightData& lightData, SamplerState& sampler, float3 pos, LightOutput* output) {
    float selPdf;
    const int lightIndex = lightData.lightDist.sample(sampler, selPdf);
    if (lightIndex < lightData.triangleLights.size()) {
        const TriangleLight& light = lightData.triangleLights[lightIndex];
        light.sample(sampler, pos, output);
    }
    else {
        lightData.envmapLight.sample(sampler, pos, output);
    }
    output->pdf *= selPdf;
}
