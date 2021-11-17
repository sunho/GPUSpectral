#pragma once

#include "framegraph/FrameGraph.h"

#include "../Camera.h"
#include "../Transform.h"
#include "../Window.h"
#include "../backend/vulkan/VulkanDriver.h"
#include "../utils/ResourceList.h"

namespace sunho3d {

#define IRD_MAP_SIZE 8
#define IRD_MAP_PROBE_COLS 8

struct MaterialBuffer {
    glm::vec4 specular;
    float phong;
    int pad[3];
};

struct TransformBuffer {
    glm::mat4 MVP;
    glm::mat4 model;
    glm::mat4 invModelT;
    glm::vec4 cameraPos;
};

#define MATERIAL_DIFFUSE_TEXTURE 1
#define MATERIAL_DIFFUSE_COLOR 2
#define MATERIAL_EMISSION 3

struct InstanceMaterial {
    glm::vec3 diffuseColor;
    int diffuseMapIndex;
    int typeID;
    float pad[3];
};

struct Instance {
    glm::mat4 transform;
    uint32_t meshIndex;
    float pad[3];
    InstanceMaterial material;
};

static constexpr const size_t MAX_INSTANCES = 64;
static constexpr const size_t RAYS_PER_PROBE = 32;

struct ForwardRTSceneBuffer {
    glm::uvec2 frameSize;
    uint32_t instanceNum;
    uint32_t pad;
    std::array<Instance, MAX_INSTANCES> instances;
};

struct DDGIPushConstants {
    uint32_t globalRngState;
};

struct DDGISceneInfo {
    glm::uvec3 gridNum;
    float pad1;
    glm::vec3 sceneSize;
    float pad2;
    glm::vec3 sceneCenter;
    float pad3;
};

struct DDGISceneBuffer {
    glm::uvec2 frameSize;
    uint32_t instanceNum;
    uint32_t pad;
    std::array<Instance, MAX_INSTANCES> instances;
    DDGISceneInfo sceneInfo;
};

class Scene;
class Renderer;

struct InflightData {
    Handle<HwFence> fence;
    Handle<HwInflight> handle{};
    std::unique_ptr<FrameGraph> rg;

    std::vector<RTInstance> instances;
    //ddgi
    Handle<HwTLAS> tlas;
};

struct DDGIPassContext {
    Handle<HwTexture> probeIrradiance;
    Handle<HwTexture> probeMeanDistance;
    float hysteresis{ 0.75 };
};

struct GBuffer {
    GBuffer(VulkanDriver& driver, uint32_t width, uint32_t height) {
        positionBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, width, height);
        normalBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, width, height);
        diffuseBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, width, height);
        emissionBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, width, height);
        depthBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::DEPTH32F, width, height);
        RenderAttachments atts = {};
        atts.colors[0] = positionBuffer;
        atts.colors[1] = normalBuffer;
        atts.colors[2] = diffuseBuffer;
        atts.colors[3] = emissionBuffer;
        atts.depth = depthBuffer;

        renderTarget = driver.createRenderTarget(width, height, atts);
    }
    Handle<HwTexture> positionBuffer;
    Handle<HwTexture> normalBuffer;
    Handle<HwTexture> diffuseBuffer;
    Handle<HwTexture> emissionBuffer;
    Handle<HwTexture> depthBuffer;
    Handle<HwRenderTarget> renderTarget;
};

constexpr static size_t MAX_INFLIGHTS = 3; 

class Renderer : public IdResource {
  public:
    Renderer(Window* window);
    ~Renderer();

    VulkanDriver& getDriver() {
        return driver;
    }
    void run(Scene* scene);

  private:
    void registerPrograms();
    void rasterSuite(Scene* scene);
    void deferSuite(Scene* scene);
    void rtSuite(Scene* scene);
    void ddgiSuite(Scene* scene);

    Handle<HwBufferObject> createTransformBuffer(FrameGraph& rg, const Camera& camera, const glm::mat4& model, const glm::mat4& modelInvT);

    Handle<HwProgram> fowradPassProgram;
    Handle<HwProgram> forwardRTProgram;
    Handle<HwProgram> blitProgram;
    Handle<HwProgram> gbufferGenProgram;
    Handle<HwProgram> ddgiProbeRayGenProgram;
    Handle<HwProgram> ddgiProbeRayShadeProgram;
    Handle<HwProgram> ddgiShadeProgram;
    Handle<HwProgram> ddgiProbeUpdateProgram;
    Handle<HwProgram> deferredRenderProgram;

    Handle<HwRenderTarget> surfaceRenderTarget;
    Handle<HwPrimitive> quadPrimitive;

    std::array<InflightData, MAX_INFLIGHTS> inflights;
    
    std::unordered_map<uint32_t, Handle<HwBLAS> > blasMap;

    DDGIPassContext ddgiContext;
    std::unique_ptr<GBuffer> gbuffer;
    std::unique_ptr<GBuffer> rayGbuffer;
    Handle<HwTexture> probeTexture;
    Handle<HwTexture> probeDistTexture;
    Handle<HwTexture> probeDistSquareTexture;

    VulkanDriver driver;
    Window* window;

    size_t currentFrame{0};
};

}  // namespace sunho3d