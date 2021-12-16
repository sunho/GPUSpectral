#pragma once

#include "Common.cuh":

enum BSDFType: uint16_t {
    BSDF_DIFFUSE = 0,
    BSDF_SMOOTH_DIELECTRIC
};

struct BSDFHandle {
    HOSTDEVICE CUDAINLINE BSDFHandle() { } 
    BSDFHandle(BSDFType type, uint32_t index) : handle((static_cast<uint32_t>(type) << 16) | (index & 0xFFFF)){
    }
    HOSTDEVICE CUDAINLINE BSDFType type() const {
        return static_cast<BSDFType>((handle >> 16) & 0xffff);
    }
    HOSTDEVICE CUDAINLINE uint32_t index() const {
        return handle & 0xFFFF;
    }
    uint32_t handle;
};

struct DiffuseBSDF {
    float3 reflectance;
};

struct SmoothDielectricBSDF {
    float iorIn;
    float iorOut; // usually 1.0
};

struct BSDFData {
    Array<DiffuseBSDF> diffuseBSDFs;
    Array<SmoothDielectricBSDF> smoothDielectricBSDFs;
};

struct BSDFOutput {
    float3 bsdf;
    float pdf;
    bool isDelta;
};

HOSTDEVICE CUDAINLINE void sampleDiffuseBSDF(const DiffuseBSDF& bsdf, SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) {
    wi = randDirHemisphere(sampler);
    output.bsdf = bsdf.reflectance / M_PI;
    output.pdf = 1.0 / (2 * M_PI);
    output.isDelta = false;
}

HOSTDEVICE CUDAINLINE float fresnelDielectric(float no, float cosTho, float nt, float cosTht) {
    float a = nt * cosTho - no * cosTht;
    float ad = nt * cosTho + no * cosTht;
    float b = no * cosTho - nt * cosTht;
    float bd = no * cosTho + nt * cosTht;
    float A = a*a / (ad*ad);
    float B = b*b / (bd*bd);
    return 0.5f * (A + B);
}

HOSTDEVICE CUDAINLINE bool refract(float3 wo, float3 n, float no, float nt, float3& wt, float& cosTht) {
    float sinTho = sqrt(wo.x * wo.x + wo.y * wo.y); // sqrt((cos(phi)sin(theta))^2 + ((sin(phi)sin(theta))^2)
    float sqrtTerm = 1.0f - ((no * no) / (nt * nt)) * (sinTho * sinTho);
    if (sqrtTerm <= 0.0f) {
        return false;
    }
    cosTht = sqrt(sqrtTerm);
    wt = (no / nt) * (-wo) + ((no / nt) * dot(wo, n) - cosTht) * n;
    return true;
}

// bsdf for reflection: F_r/|cos(th_o)|
// bsdf for refraction: (n_o^2/n_t^2)(1-F_r)/|cos(th_o)|
HOSTDEVICE CUDAINLINE void sampleSmoothDielectricBSDF(const SmoothDielectricBSDF& bsdf, SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) {
    bool entering = wo.z > 0.0f;
    float no = entering ? bsdf.iorOut : bsdf.iorIn;
    float nt = entering ? bsdf.iorIn : bsdf.iorOut;
    float cosTho = wo.z;
    float cosTht;
    float3 wt;
    if (!refract(wo, faceforward(make_float3(0.0f, 0.0f, 1.0f), wo, make_float3(0.0f, 0.0f, 1.0f)), no, nt, wt, cosTht)) {
        // total internal reflection
        wi.z = wo.z;
        wi.x = -wo.x;
        wi.y = -wo.y;
        output.bsdf = 1.0f*make_float3(1/abs(cosTho));
        output.isDelta = true;
        output.pdf = 1.0f;
        return;
    }
    float Fr = fresnelDielectric(no, abs(cosTho), nt, cosTht);
    float u = randUniform(sampler);
    // reflection with prob Fr
    // refraction with prob 1 - Fr
    if (u < Fr) {
        // reflect
        wi.z = wo.z;
        wi.x = -wo.x;
        wi.y = -wo.y;
        output.bsdf = Fr * make_float3(1/abs(cosTho));
        output.isDelta = true;
        output.pdf = Fr;
    } else {
        // refract
        wi = wt;

        output.bsdf = make_float3(((no*no)/(nt*nt))*(1.0f-Fr)/abs(cosTht));
        output.isDelta = true;
        output.pdf = 1.0f - Fr;
    }
}

HOSTDEVICE CUDAINLINE void evalDiffuseBSDF(const DiffuseBSDF& bsdf, float3 wo, float3 wi, BSDFOutput& output) {
    output.bsdf = bsdf.reflectance / M_PI;
    output.pdf = 1.0 / (2 * M_PI);
    output.isDelta = false;
}

HOSTDEVICE CUDAINLINE void evalSmoothDielectricBSDF(const SmoothDielectricBSDF& bsdf, float3 wo, float3 wi, BSDFOutput& output) {
    output.bsdf = make_float3(0.0f);
    output.pdf = 1.0f;
    output.isDelta = true;
}

HOSTDEVICE CUDAINLINE void sampleBSDF(const BSDFData& data, SamplerState& sampler, const BSDFHandle& handle, float3 wo, float3& wi, BSDFOutput& output) {
    switch (handle.type()) {
    case BSDF_DIFFUSE: {
        sampleDiffuseBSDF(data.diffuseBSDFs[handle.index()], sampler, wo, wi, output);
        break;
    };
    case BSDF_SMOOTH_DIELECTRIC: {
        sampleSmoothDielectricBSDF(data.smoothDielectricBSDFs[handle.index()], sampler, wo, wi, output);
        break;
    }
    }
}

HOSTDEVICE CUDAINLINE void evalBSDF(const BSDFData& data, const BSDFHandle& handle, float3 wo, float3 wi, BSDFOutput& output) {
    switch (handle.type()) {
    case BSDF_DIFFUSE: {
        evalDiffuseBSDF(data.diffuseBSDFs[handle.index()], wo, wi, output);
        break;
    };
    case BSDF_SMOOTH_DIELECTRIC: {
        evalSmoothDielectricBSDF(data.smoothDielectricBSDFs[handle.index()], wo, wi, output);
        break;
    }
    }
}



