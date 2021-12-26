#pragma once

#define M_PI           3.14159265358979323846

#include "VectorMath.cuh"
#include <stdint.h>
#include <cassert>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <math.h>
#if defined(__CUDACC__) || defined(__CUDABE__)
#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#define CUDAINLINE __forceinline__
#define CONST_STATIC_INIT( ... )
#else
#define HOST 
#define DEVICE __device__
#define HOSTDEVICE 
#define CUDAINLINE inline 
#define CONST_STATIC_INIT( ... ) = __VA_ARGS__

// TODO figure out better way
template <typename T>
static __device__ typename T tex2DLod(cudaTextureObject_t obj, float x, float y, float level)
{
}

template <typename T>
static __device__ typename T tex2D(cudaTextureObject_t obj, float x, float y)
{
}

#endif
#define NUMBERCHECK(num) if (isnan(num)) { printf("nan detected; %s line %d\n", __FILE__, __LINE__); }
#define isvalid(num) (!isnan(num) && !isinf(num))

template <typename T>
struct Array {
    HOSTDEVICE CUDAINLINE Array()  {
    }
    HOST void allocDevice(size_t size) {
        size_ = size;
        cudaMalloc(&data_, size * sizeof(T));
    }

    HOST void freeDevice() {
        cudaFree(data_);
    }
    
    HOSTDEVICE CUDAINLINE const T* data() const {
        return data_;
    }
    
    HOSTDEVICE CUDAINLINE T* data() {
        return data_;
    }

    HOSTDEVICE CUDAINLINE size_t size() const {
        return size_;
    }

    HOSTDEVICE CUDAINLINE const T& operator[] (const size_t idx) const {
        return data_[idx];
    }

    HOSTDEVICE CUDAINLINE T& operator[] (const size_t idx) {
        return data_[idx];
    }

    HOST void upload(const T* src) {
        cudaMemcpy(data_, src, sizeof(T)*size_, cudaMemcpyHostToDevice);
    }
private:
    T* data_;
    size_t size_;
};

HOSTDEVICE CUDAINLINE uint32_t pcgHash(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct SamplerState {
    HOSTDEVICE CUDAINLINE SamplerState() : seed(0) {
    }

    HOSTDEVICE CUDAINLINE SamplerState(uint32_t seed) : seed(seed) {
    }
    uint32_t seed;
};

HOSTDEVICE CUDAINLINE uint32_t randInt(SamplerState& sampler)
{
    uint32_t state = sampler.seed;
    sampler.seed = sampler.seed * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}


HOSTDEVICE CUDAINLINE float randUniform(SamplerState& sampler) {
    return randInt(sampler) * (1.0 / float(0xffffffffu));
}

// http://corysimon.github.io/articles/uniformdistn-on-sphere/
// tldr; find pdf function by f(v)*dA = 1 = f(phi,theta) * dphi * dtheta
// dA = sin(phi) * dphi * dtheta
// inverse transform marginal pdf of phi
HOSTDEVICE CUDAINLINE float3 randDirSphere(SamplerState& sampler) {
    float2 u = make_float2(randUniform(sampler), randUniform(sampler));
    float theta = 2.0 * M_PI * u.x;
    float phi = acos(1.0 - 2.0 * u.y);
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    return make_float3(x, y, z);
}

HOSTDEVICE CUDAINLINE float3 randDirHemisphere(SamplerState& sampler) {
    float2 u = make_float2(randUniform(sampler), randUniform(sampler));
    float theta = 2.0 * M_PI * u.x;
    float phi = acos(1.0 - u.y);
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    return make_float3(x, y, z);
}

HOSTDEVICE CUDAINLINE float2 sampleConcentric(SamplerState& sampler) {
    float2 sample = make_float2(randUniform(sampler), randUniform(sampler));
    float2 u = 2.0f * sample - 1.0f;
    if (u.x == 0.0f && u.y == 0.0f) {
        return make_float2(0.0f, 0.0f);
    }
    float r, th;
    if (fabs(u.x) > fabs(u.y)) {
        r = u.x;
        th = (M_PI / 4.0f) * (u.y / u.x);
    }
    else {
        r = u.y;
        th = M_PI / 2.0f - (M_PI / 4.0f) * (u.x / u.y);
    }
    return make_float2(r * cos(th), r * sin(th));
}

HOSTDEVICE CUDAINLINE float3 randCosineHemisphere(SamplerState& sampler) {
    float2 u = sampleConcentric(sampler);
    float z = sqrt(fmaxf(0.0f, 1.0f - u.x * u.x - u.y * u.y));
    return make_float3(u.x, u.y, z);
}

HOSTDEVICE CUDAINLINE float cosineHemispherePdf(float3 wo) {
    return fmaxf(abs(wo.z) / M_PI, 0.000001f);
}

struct Onb
{
  HOSTDEVICE CUDAINLINE Onb(const float3& normal) {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  HOSTDEVICE CUDAINLINE void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  HOSTDEVICE CUDAINLINE float3 transform(float3 p) const {
      return make_float3(dot(p, m_tangent), dot(p, m_binormal), dot(p, m_normal));
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

HOSTDEVICE CUDAINLINE float3 ACESFilm(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

HOSTDEVICE CUDAINLINE float3 reinhard(float3 c) {
    return pow(c / (c + 1.0f), make_float3(1.0f / 2.2f));
}

HOSTDEVICE CUDAINLINE float3 filmMap(float3 c) {
    c *= 1.0f;
    float3 x = fmaxf(make_float3(0.0f), c - 0.004f);
    return (x * (6.2f * x + 0.5f)) / (x * (6.2f * x + 1.7f) + 0.06f);
}

HOSTDEVICE CUDAINLINE float3 rayDir(float2 size, float2 fragCoord, float fov, float ratio) {
    float2 xy = fragCoord - size / 2.0f;
    float z = (fmaxf(size.x,size.y)/2.0f) / tan(fov / 2.0f);
    auto dir = normalize(make_float3(-xy.x, xy.y, z));
    return dir;
}

HOSTDEVICE CUDAINLINE float3 sampleHalf(SamplerState& sampler, float alpha) {
    float2 u = make_float2(randUniform(sampler), randUniform(sampler));
    float phi = 2.0f * M_PI * u.x;
    // 1 + tan^2 = sec^2
    // 1 / (1+tan^2) = cos^2
    float logSample = log(1.0f - u.y);
    if (isinf(logSample)) logSample = 0.0f;
    float tan2 = -alpha * alpha * logSample;
    float cost = 1.0f / sqrt(1.0f + tan2); // denominator is never 0.0
    float sint = sqrt(fmaxf(0.0f, 1.0f - cost* cost));
    return make_float3(cos(phi) * sint, sin(phi) * sint, cost);
}

HOSTDEVICE CUDAINLINE float sphericalPhi(float3 wi) {
    float p = atan2(wi.y, wi.x);
    return p < 0.0f ? (p + 2 * M_PI) : p;
}

HOSTDEVICE CUDAINLINE float sphericalTheta(float3 wi) {
    return acos(clamp(wi.z,-1.0f, 1.0f));
}

HOSTDEVICE CUDAINLINE float beckmannD(float3 wh, float alpha) {
    float cos2 = wh.z * wh.z;
    float tan2 = (wh.x * wh.x + wh.y * wh.y) / cos2;
    float a = exp(-tan2 / (alpha * alpha));
    float b = M_PI * alpha * alpha * cos2 * cos2;
    return a / b;
}

HOSTDEVICE CUDAINLINE float ggxD(float3 wh, float alpha) {
    float cos2 = wh.z * wh.z;
    float tan2 = (wh.x * wh.x + wh.y * wh.y) / cos2;
    if (isinf(tan2)) { return 0.0f; };
    float b = (1.0f + tan2 / (alpha * alpha));
    float a = M_PI * alpha * alpha * cos2 * cos2 * b * b;
    return 1.0f / a;
}

HOSTDEVICE CUDAINLINE float ggxLambda(float3 wh, float alpha) {
    float cos2 = wh.z * wh.z;
    float tan2 = (wh.x * wh.x + wh.y * wh.y) / cos2;
    if (isinf(tan2)) { return 0.0f; };
    float a = -1.0f + sqrt(1.0f + alpha*alpha*tan2);
    return 0.5f * a;
}

HOSTDEVICE CUDAINLINE float ggxMask(float3 wo, float3 wi, float alpha) {
    return 1.0f / (1.0f + ggxLambda(wo, alpha) + ggxLambda(wi, alpha));
}

HOSTDEVICE CUDAINLINE float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

