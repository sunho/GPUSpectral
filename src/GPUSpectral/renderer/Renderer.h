#pragma once

#include "../kernels/VectorMath.cuh"
#include "../kernels/PathTracer.cuh"
#include "Scene.h"
#include <filesystem>
#include <functional>
#include <optix.h>
#include <vector>
#include <span>

struct Mesh {
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
};

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

struct CudaPipeline {
    CudaPipeline(Renderer& renderer, OptixDeviceContext context);

    OptixProgramGroup              raygenProgGroup;
    OptixProgramGroup              radianceMissGroup;
    OptixProgramGroup              radianceHitGroup;
    OptixProgramGroup              shadowHitGroup;
    OptixProgramGroup              shadowMissGroup;
    
    OptixModule                    ptxModule;
    OptixPipeline                  pipeline;
private:
    void initModule(Renderer& renderer, OptixDeviceContext context);
    void initProgramGroups(Renderer& renderer, OptixDeviceContext context);
    void initPipeline(Renderer& renderer, OptixDeviceContext context);
};

struct CudaSBT {
    CudaSBT(Renderer& renderer, OptixDeviceContext context, CudaTLAS& tlas, const Scene& scene);

    CUdeviceptr raygenRecord;
    CUdeviceptr missRecordBase;
    CUdeviceptr hitgroupRecordBase;
    OptixShaderBindingTable sbt{};
};

struct RenderState {
    RenderState(Renderer& renderer, OptixDeviceContext context, const Scene& scene);

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

    void setScene(const Scene& scene);
    void render();

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