#pragma once

#include "../utils/CudaUtil.h"
#include "../kernels/VectorMath.cuh"
#include "../kernels/PathTracer.cuh"
#include "CudaSBT.h"
#include "CudaTLAS.h"
#include "CudaPipeline.h"
#include "Scene.h"
#include <filesystem>
#include <functional>
#include <optix.h>
#include <vector>
#include <span>

struct RenderConfig {
    int width;
    int height;
};

struct Mesh {
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
};

class Renderer;

struct RenderState {
    RenderState(Renderer& renderer, OptixDeviceContext context, const Scene& scene, const RenderConfig& config);

    Scene scene;
    LightData deviceLightData;
    BSDFData deviceBSDFData;
    CudaTLAS tlas;
    CudaSBT sbt;
    CUstream stream;
    Params  params;
    Params* dParams;
};

class Renderer {
public:
    friend class CudaSBT;
    Renderer(const std::string& basePath);
    ~Renderer();

    int addMesh(const Mesh& mesh);
    Mesh* getMesh(int meshId);

    void setScene(const Scene& scene, const RenderConfig& config);
    void render(int spp);

    std::string loadKernel(const std::string& name);
    std::string assetPath(const std::string& filename) { return (baseFsPath / "assets" / filename).string(); }

private:
    static OptixDeviceContext createDeviceContext();

    std::string basePath;
    std::filesystem::path baseFsPath;

    std::vector<Mesh> meshes;
    OptixDeviceContext context;
    CudaPipeline pipeline;

    std::unique_ptr<RenderState> state;
};