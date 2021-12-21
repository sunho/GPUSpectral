#pragma once

#include <optix.h>
#include "Scene.h"
#include "CudaTLAS.h"
class Renderer;
struct CudaSBT {
    CudaSBT(Renderer& renderer, OptixDeviceContext context, CudaTLAS& tlas, const Scene& scene);

    CUdeviceptr raygenRecord;
    CUdeviceptr missRecordBase;
    CUdeviceptr hitgroupRecordBase;
    OptixShaderBindingTable sbt{};
};
