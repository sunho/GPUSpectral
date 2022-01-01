#pragma once

#include "framegraph/FrameGraph.h"

#include "../Camera.h"
#include "../Transform.h"
#include "../Window.h"
#include "../backend/vulkan/VulkanDriver.h"
#include "../utils/ResourceList.h"
#include "../Loader.h"

namespace VKGIRenderer {

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
static constexpr const size_t RAYS_PER_PROBE = 64;

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
    SceneBuffer cpuSceneBuffer;
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

constexpr static size_t MAX_INFLIGHTS = 2; 

class Engine;
class Renderer : public IdResource {
  public:
    Renderer(Engine& engine, Window* window);
    ~Renderer();

    VulkanDriver& getDriver() {
        return driver;
    }
    void run(const Scene& scene);

  private:
    Handle<HwProgram> loadShader(const std::string& filename);
    void registerShader(const std::string& shaderName, const std::string& filename);
    void render(InflightContext& context, const Scene& scene);
    Handle<HwProgram> getShaderProgram(const std::string& shaderName);
    void registerPrograms();
    void prepareSceneData(InflightContext& context);

    Handle<HwRenderTarget> surfaceRenderTarget;
    Handle<HwPrimitive> quadPrimitive;

    std::array<InflightData, MAX_INFLIGHTS> inflights;

    std::unordered_map<uint32_t, Handle<HwBLAS>> blasCache;
    std::unordered_map<std::string, Handle<HwProgram> > programs;

    VulkanDriver driver;
    Engine& engine;
    Window* window;

    size_t currentFrame{0};
};

}  // namespace VKGIRenderer