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
    CUdeviceptr                    deviceUvs;

    std::vector<float4> positions;
    std::vector<float4> normals;
    std::vector<float4> uvs;
    std::vector<int> matIndices;
private:
    void fillData(Renderer& renderer, const Scene& scene);
};
