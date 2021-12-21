#include "Renderer.h"

#include <filesystem>
#include <fstream>
#include <stb_image_write.h>

#include <optix.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

Renderer::Renderer(const std::string& basePath) : 
    basePath(basePath), 
    baseFsPath(basePath),
    context(createDeviceContext()), 
    pipeline(*this, context) {
}

static void contextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

OptixDeviceContext Renderer::createDeviceContext() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCallback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    return context;
}

Renderer::~Renderer() {

}

std::string Renderer::loadKernel(const std::string& fileName) {
    std::ifstream is(baseFsPath / "kernels" / fileName, std::ios::binary);
    std::string includedSource((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    return includedSource;
}

Mesh* Renderer::getMesh(int id) {
    return &meshes[id];
       
}

int Renderer::addMesh(const Mesh& mesh) {
    int id = meshes.size();
    meshes.push_back(mesh);
    return id;
}

void Renderer::setScene(const Scene& scene, const RenderConfig& config) {
    state = std::make_unique<RenderState>(*this, context, scene, config);
}

RenderState::RenderState(Renderer& renderer, OptixDeviceContext context, const Scene& scene, const RenderConfig& config) :
    scene(scene), tlas(renderer, context, scene), sbt(renderer, context, tlas, scene) {

    params.subframe_index = 0u;
    params.handle = tlas.gasHandle;
    params.eye = scene.camera.eye;
    params.U =scene.camera.u;
    params.V = scene.camera.v;
    params.W = scene.camera.w;
    params.fov = scene.camera.fov;
    params.width = config.width;
    params.height = config.height;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
    ));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.frame_buffer), params.width * params.height * 4));

    deviceLightData.triangleLights.allocDevice(scene.triangleLights.size());
    deviceLightData.triangleLights.upload(scene.triangleLights.data());
    params.lightData = deviceLightData;

    {
    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
        deviceBSDFData.BSDFFIELD##s.allocDevice(scene.BSDFFIELD##s.size()); \
        deviceBSDFData.BSDFFIELD##s.upload(scene.BSDFFIELD##s.data());
    #include "../kernels/BSDF.inc"
    #undef BSDFDefinition
        params.bsdfData = deviceBSDFData;
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParams), sizeof(Params)));
}


void Renderer::render(int spp) {
    state->params.samples_per_launch = spp;
    state->params.subframe_index++;
    // Launch
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(state->dParams),
        &state->params, sizeof(Params),
        cudaMemcpyHostToDevice, state->stream
    ));

    OPTIX_CHECK(optixLaunch(
        pipeline.pipeline,
        state->stream,
        reinterpret_cast<CUdeviceptr>(state->dParams),
        sizeof(Params),
        &state->sbt.sbt,
        state->params.width,   // launch width
        state->params.height,  // launch height
        1                     // launch depth
    ));
    cudaDeviceSynchronize();

    std::vector<uint32_t> pixels(state->params.width * state->params.height);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(pixels.data()),
        reinterpret_cast<void*>(state->params.frame_buffer),
        state->params.width * state->params.height * 4,
        cudaMemcpyDeviceToHost
    ));
    if (!std::filesystem::is_directory("output")) {
        std::filesystem::create_directory("output");
    }
    stbi_flip_vertically_on_write(true);
    auto filename = std::string("output/output") + std::to_string(state->params.subframe_index) + ".jpg";
    stbi_write_jpg(filename.c_str(), state->params.width, state->params.height, 4, pixels.data(), state->params.width * 4);
}


