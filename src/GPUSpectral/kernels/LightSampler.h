#pragma once

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct TriangleLight {
    float4 positions[3];
    float4 radinace;
};

struct LightData {
    TriangleLight* triangleLights;
};
