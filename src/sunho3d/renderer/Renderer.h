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

struct PointShadowUniformBuffer {
    glm::mat4 lightVP;
    glm::vec3 lightPos;
    float farPlane;
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

#define MAX_MESH_COUNT 64
static constexpr const size_t MAX_INSTANCES = 64;
static constexpr const size_t RAYS_PER_PROBE = 32;

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

struct SceneBuffer {
    glm::uvec2 frameSize;
    uint32_t instanceNum;
    uint32_t pad;
    std::array<Instance, MAX_INSTANCES> instances;
    DDGISceneInfo sceneInfo;
};

class Scene;
struct SceneData;
struct LightData;
class Renderer;

struct InflightData {
    Handle<HwFence> fence;
    Handle<HwInflight> handle{};
    Handle<HwTLAS> tlas{};
    std::unique_ptr<FrameGraph> rg;


    Handle<HwBufferObject> sceneBuffer;
    Handle<HwBufferObject> lightBuffer;
    std::vector<RTInstance> instances;
    std::vector<Handle<HwBufferObject>> materialBuffers;
    std::vector<Handle<HwBufferObject>> transformBuffers;
    std::vector<TextureHandle> diffuseMap;
    std::vector<Handle<HwTexture>> shadowMaps;
    std::vector<InstanceMaterial> materials;
    std::vector<Handle<HwBufferObject>> vertexPositionBuffers;
    std::vector<Handle<HwBufferObject>> vertexNormalBuffers;
    std::vector<Handle<HwBufferObject>> vertexUVBuffers;

    void reset(VulkanDriver& driver) {
        instances.clear();
        materialBuffers.clear();
        transformBuffers.clear();
        diffuseMap.clear();
        materials.clear();
        vertexPositionBuffers.clear();
        vertexNormalBuffers.clear();
        vertexUVBuffers.clear();
        shadowMaps.clear();
    }
};

struct InflightContext {
    FrameGraph* rg;
    InflightData* data;
    SceneData* sceneData;
    Scene* scene;
};

struct GBuffer {
    GBuffer(VulkanDriver& driver, uint32_t width, uint32_t height) {
        positionBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, width, height, 1);
        normalBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, width, height, 1);
        diffuseBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, width, height, 1);
        emissionBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, width, height, 1);
        depthBuffer = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::DEPTH32F, 1, width, height, 1);
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

class Engine;
class Renderer : public IdResource {
  public:
    Renderer(Engine& engine, Window* window);
    ~Renderer();

    VulkanDriver& getDriver() {
        return driver;
    }
    void run(Scene* scene);

  private:
    Handle<HwProgram> loadComputeShader(const std::string& filename);
    Handle<HwProgram> loadGraphicsShader(const std::string& vertFilename, const std::string& fragFilename);
    void registerPrograms();
    void deferSuite(InflightContext& ctx);
    void rtSuite(InflightContext& ctx);
    void ddgiSuite(InflightContext& ctx);
    void prepareSceneData(InflightContext& context);
    Handle<HwTexture> buildPointShadowMap(InflightContext& ctx, LightData light);

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
    Handle<HwProgram> pointShadowGenProgram;

    Handle<HwRenderTarget> surfaceRenderTarget;
    Handle<HwPrimitive> quadPrimitive;

    std::array<InflightData, MAX_INFLIGHTS> inflights;
    
    std::unordered_map<uint32_t, Handle<HwBLAS> > blasCache;
    std::unordered_map<uint32_t, Handle<HwTexture>> shadowMapCache;

    std::unique_ptr<GBuffer> gbuffer;
    std::unique_ptr<GBuffer> rayGbuffer;
    Handle<HwTexture> probeTexture;
    Handle<HwTexture> probeDistTexture;
    Handle<HwTexture> probeDistSquareTexture;

    VulkanDriver driver;
    Engine& engine;
    Window* window;

    size_t currentFrame{0};
};

}  // namespace sunho3d