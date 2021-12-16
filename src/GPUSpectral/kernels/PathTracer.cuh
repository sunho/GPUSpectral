#pragma once

#include <optix.h>
#include <vector_types.h>
#include "LightSampler.cuh"
#include "BSDFSampler.cuh"

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};

typedef Record<EmptyData> EmptyRecord;

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION,
    RAY_TYPE_COUNT
};

struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;
    float fov;

    LightData lightData;
    BSDFData bsdfData;
    OptixTraversableHandle handle;
};

struct RayGenData
{
};

struct MissData
{
    float4 bg_color;
};

struct HitGroupData
{
    float3  emission_color;
    uint32_t twofaced;
    BSDFHandle bsdf;
    float4* vertices;
};

