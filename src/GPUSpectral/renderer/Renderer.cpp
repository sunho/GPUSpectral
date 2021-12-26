#include "Renderer.h"
#include <numeric>

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

Mesh* Renderer::getMesh(MeshId id) {
    return &meshes[id];
}

MeshId Renderer::addMesh(const Mesh& mesh) {
    MeshId outId = nextHandleId++;
    meshes.emplace(outId, mesh);
    return outId;
}

TextureId Renderer::createTexture(TextureFormat format, uint32_t width, uint32_t height) {
    TextureId outId = nextHandleId++;
    textures.emplace(outId, std::move(Texture(*this, format, width, height)));
    return outId;
}

Texture* Renderer::getTexture(TextureId id) {
    return &textures[id];
}

void Renderer::setScene(const Scene& scene, const RenderConfig& config) {
    state = std::make_unique<RenderState>(*this, context, scene, config);
}

static EnvmapLight createEnvmapLight(Texture& envmap, const Scene& scene) {
    EnvmapLight outLight = {
        .envmap = envmap.getTextureObject(),
        .size = make_float2(envmap.getWidth(), envmap.getHeight()),
    };
    const auto bbox = scene.sceneData.getBoundingBox();
    outLight.center = (bbox.maxs + bbox.mins) / 2.0f;
    outLight.radius = fmaxf((bbox.maxs - bbox.mins) / 2.0f);
    float yTotalWeight = 0.0f;
    std::vector<float> ypdf;
    std::vector<PieceDist> xDists;
    for (size_t i = 0; i < envmap.getHeight(); ++i) {
        std::vector<float> xpdf;
        float xTotalWeight = 0.0f;
        for (size_t j = 0; j < envmap.getWidth(); ++j) {
            float3 tex = make_float3(envmap.texelFetch(j, i));
            float v = static_cast<float>(i) / envmap.getHeight();
            float weight = length(tex) * sin(v * M_PI);
            xTotalWeight += weight;
            xpdf.push_back(weight);
        }
        for (size_t j = 0; j < envmap.getWidth(); ++j) {
            xpdf[j] /= xTotalWeight;
        }
        PieceDist xdist{};
        xdist.pdfs.allocDevice(xpdf.size());
        xdist.pdfs.upload(xpdf.data());
        xDists.push_back(xdist);
        ypdf.push_back(xTotalWeight);
        yTotalWeight += xTotalWeight;
    }
    for (size_t i = 0; i < envmap.getHeight(); ++i) {
        ypdf[i] /= yTotalWeight;
    }
    PieceDist yDist;
    yDist.pdfs.allocDevice(ypdf.size());
    yDist.pdfs.upload(ypdf.data());
    outLight.xDists.allocDevice(xDists.size());
    outLight.xDists.upload(xDists.data());
    outLight.yDist = yDist;
    return outLight;
}

static std::vector<float> getTriangleLightWeights(const std::vector<TriangleLight>& lights) {
    std::vector<float> outTable;
    for (auto& light : lights) {
        float3 power = light.getPower();
        float pp = length(power);
        outTable.push_back(pp);
    }
    return outTable;
}


RenderState::RenderState(Renderer& renderer, OptixDeviceContext context, const Scene& scene, const RenderConfig& config) :
    scene(scene), tlas(renderer, context, scene), sbt(renderer, context, tlas, scene) {

    params.subframeIndex = 0u;
    params.scene.tlas= tlas.gasHandle;
    params.camera.eye = scene.camera.eye;
    params.camera.U =scene.camera.u;
    params.camera.V = scene.camera.v;
    params.camera.W = scene.camera.w;
    params.camera.fov = scene.camera.fov;
    params.width = config.width;
    params.height = config.height;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accumBuffer),
        params.width * params.height * sizeof(float4)
    ));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.frameBuffer), params.width * params.height * 4));

    deviceLightData.triangleLights.allocDevice(scene.triangleLights.size());
    deviceLightData.triangleLights.upload(scene.triangleLights.data());

    auto pdfs = getTriangleLightWeights(scene.triangleLights);
    if (scene.envMap) {
        Texture& envMapTex = *renderer.getTexture(scene.envMap);
        auto envLight = createEnvmapLight(envMapTex, scene);
        envLight.toWorld = scene.envMapTransform;
        envLight.toWorldInv = scene.envMapTransform.inverse();
        float mediumWeight = length(make_float3(envMapTex.texelFetch(envMapTex.getWidth() / 2, envMapTex.getHeight() / 2)));
        float power = M_PI * M_PI * 4.0f * envLight.radius * envLight.radius * mediumWeight;
        pdfs.push_back(power);
        deviceLightData.envmapLight = envLight;
    }
    const float totalPdf = std::accumulate(pdfs.begin(), pdfs.end(), 0.0f);
    for (size_t i = 0; i < pdfs.size(); ++i) {
        pdfs[i] /= totalPdf;
    }
    PieceDist lightDist{};
    lightDist.pdfs.allocDevice(pdfs.size());
    lightDist.pdfs.upload(pdfs.data());
    deviceLightData.lightDist = lightDist;
    params.scene.lightData = deviceLightData;

    {
    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
        deviceBSDFData.BSDFFIELD##s.allocDevice(scene.BSDFFIELD##s.size()); \
        deviceBSDFData.BSDFFIELD##s.upload(scene.BSDFFIELD##s.data());
    #include "../kernels/BSDF.inc"
    #undef BSDFDefinition
        params.scene.bsdfData = deviceBSDFData;
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParams), sizeof(Params)));
}


void Renderer::render(int spp) {
    state->params.spp= spp;
    state->params.subframeIndex++;
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
        reinterpret_cast<void*>(state->params.frameBuffer),
        state->params.width * state->params.height * 4,
        cudaMemcpyDeviceToHost
    ));
    if (!std::filesystem::is_directory("output")) {
        std::filesystem::create_directory("output");
    }
    stbi_flip_vertically_on_write(true);
    auto filename = std::string("output/output") + std::to_string(state->params.subframeIndex) + ".jpg";
    stbi_write_jpg(filename.c_str(), state->params.width, state->params.height, 4, pixels.data(), state->params.width * 4);
}


