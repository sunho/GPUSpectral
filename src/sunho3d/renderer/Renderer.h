#pragma once

#include "framegraph/FrameGraph.h"

#include "../Camera.h"
#include "../Transform.h"
#include "../Window.h"
#include "../backend/vulkan/VulkanDriver.h"
#include "../utils/ResourceList.h"

namespace sunho3d {

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

struct Instance {
    glm::mat4 transform;
    uint32_t vertexStart;
    float pad[3];
};

static constexpr const size_t MAX_INSTANCES = 64;

struct ForwardRTSceneBuffer {
    glm::uvec2 frameSize;
    uint32_t instanceNum;
    uint32_t pad;
    std::array<Instance, MAX_INSTANCES> instances;
};

class Scene;
class Renderer;

struct InflightData {
    Handle<HwFence> fence;
    Handle<HwInflight> handle{};
    std::unique_ptr<FrameGraph> rg;

    std::vector<RTInstance> instances;

    //ddgi

};

struct DDGIPassContext {
    Handle<HwTexture> probeIrradiance;
    Handle<HwTexture> probeMeanDistance;
    int gridSize{ 32 };
    float sceneSize;
    float hysteresis{ 0.75 };
    
};

constexpr static size_t MAX_INFLIGHTS = 2; 

class Renderer : public IdResource {
  public:
    Renderer(Window* window);
    ~Renderer();

    VulkanDriver& getDriver() {
        return driver;
    }
    void run(Scene* scene);

  private:
    void rasterSuite(Scene* scene);
    void rtSuite(Scene* scene);
    void ddgiSuite(Scene* scene);

    Handle<HwUniformBuffer> createTransformBuffer(FrameGraph& rg, const Camera& camera, const glm::mat4& model);

    Handle<HwProgram> fowradPassProgram;
    Handle<HwProgram> forwardRTProgram;
    Handle<HwProgram> quadDrawProgram;
    Handle<HwProgram> blitProgram;
    Handle<HwRenderTarget> surfaceRenderTarget;
    Handle<HwPrimitive> quadPrimitive;

    std::array<InflightData, MAX_INFLIGHTS> inflights;
    
    std::unordered_map<uint32_t, Handle<HwBLAS> > blasMap;

    DDGIPassContext ddgiContext;

    VulkanDriver driver;
    Window* window;

    size_t currentFrame{0};
};

}  // namespace sunho3d
