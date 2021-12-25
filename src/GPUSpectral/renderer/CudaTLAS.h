#pragma once

#include <optix.h>
#include <vector>
#include "Scene.h"

class Renderer;

struct CudaTLAS {
    CudaTLAS(Renderer& renderer, OptixDeviceContext context, const Scene& scene);

    OptixTraversableHandle         gasHandle;
    CUdeviceptr                    gasOutputBuffer;
    CUdeviceptr                    devicePositions;
    CUdeviceptr                    deviceNormals;
    CUdeviceptr                    deviceUVs;
};
