#pragma once

#define M_PI           3.14159265358979323846

#include "../math/VectorMath.cuh"
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#if defined(__CUDACC__) || defined(__CUDABE__)
#define HOSTDEVICE __host__ __device__
#define CUDAINLINE __forceinline__
#define CONST_STATIC_INIT( ... )
#else
#define HOSTDEVICE 
#define CUDAINLINE inline 
#define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif

template <typename T>
struct Array {
    HOSTDEVICE CUDAINLINE Array()  {
    }
    
    HOSTDEVICE CUDAINLINE Array(size_t size) : size_(size) {
        cudaMalloc(&data_, size_ * sizeof(T));
    }

    HOSTDEVICE void free() {
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

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

