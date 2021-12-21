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

struct GPUCamera {
    float3       eye;
    float3       U;
    float3       V;
    float3       W;
    float fov;
};

struct GPUScene {
    LightData lightData;
    BSDFData bsdfData;
    OptixTraversableHandle tlas;
};

struct Params
{
    unsigned int subframeIndex;
    float4* accumBuffer;
    uchar4* frameBuffer;
    unsigned int width;
    unsigned int height;
    unsigned int spp;

    GPUCamera camera;
    GPUScene scene;
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
    float4* normals;
};

