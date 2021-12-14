#pragma once

#include <optix.h>
#include <vector_types.h>
#include "LightSampler.h"

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
    RAY_TYPE_COUNT
};

struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
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

    ParallelogramLight     light; // TODO: make light list
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
    float3  diffuse_color;
    float4* vertices;
};

