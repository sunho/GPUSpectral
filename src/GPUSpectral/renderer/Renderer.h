#pragma once

#include "../math/VectorMath.h"
#include "../kernels/path_tracer.h"
#include "Scene.h"
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
    CudaTLAS(Renderer& renderer, const Scene& scene);

    OptixTraversableHandle         gasHandle;
    CUdeviceptr                    gasOutputBuffer;
    CUdeviceptr                    devicePositions;
    CUdeviceptr                    deviceNormals;
    CUdeviceptr                    deviceUvs;
};

struct CudaPipeline {
    CudaPipeline(Renderer& renderer, OptixDeviceContext context);

    OptixProgramGroup              raygenProgGroup;
    OptixProgramGroup              radianceMissGroup;
    OptixProgramGroup              radianceHitGroup;
    OptixProgramGroup              shadowMissGroup;
    OptixProgramGroup              shadowHitGroup;
    
    OptixModule                    ptxModule;
    OptixPipeline                  pipeline;
private:
    void initModule(Renderer& renderer, OptixDeviceContext context);
    void initProgramGroups(Renderer& renderer, OptixDeviceContext context);
    void initPipeline(Renderer& renderer, OptixDeviceContext context);
};

struct CudaSBT {
    CudaSBT(Renderer& renderer, const Scene& scene);

    CUdeviceptr raygenRecord;
    CUdeviceptr missRecordBase;
    CUdeviceptr hitgroupRecordBase;
    OptixShaderBindingTable sbt;
};

struct RenderState {
    RenderState(Renderer& renderer, const Scene& scene);

    Scene scene;
    CudaTLAS tlas;
    CudaSBT sbt;
    CUstream stream;
    Params  params;
    Params* dParams;
};

class Renderer {
public:
    Renderer(const std::string& basePath);
    ~Renderer();

    int addMesh(const Mesh& mesh);
    Mesh* getMesh(int meshId);

    void setScene(const Scene& scene);
    void render();

    std::string loadKernel(const std::string& name);
    std::string assetPath(const std::string& filename);

private:
    static OptixDeviceContext createDeviceContext();

    std::string basePath;

    std::vector<Mesh> meshes;
    OptixDeviceContext context;
    CudaPipeline pipeline;

    std::unique_ptr<RenderState> state;
};