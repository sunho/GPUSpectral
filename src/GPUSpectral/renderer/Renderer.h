#pragma once

#include "framegraph/FrameGraph.h"

#include "Camera.h"
#include "Transform.h"
#include "Mesh.h"
#include "../engine/Window.h"
#include "../backend/vulkan/VulkanDriver.h"
#include "../utils/ResourceList.h"

namespace GPUSpectral {

using MeshPtr = std::shared_ptr<Mesh>;

class Scene;
struct SceneData;
struct LightData;
class Renderer;

struct InflightData {
    Handle<HwFence> fence;
    Handle<HwInflight> handle{};
    Handle<HwTLAS> tlas{};
    Handle<HwBufferObject> instanceBuffer{};
    std::unique_ptr<FrameGraph> rg;
};

struct InflightContext {
    FrameGraph* rg;
    InflightData* data;
    SceneData* sceneData;
    Scene* scene;
};

constexpr static size_t MAX_INFLIGHTS = 2; 


class RendererImpl {
public:
    virtual void render(InflightContext& ctx, const Scene& scene) = 0;
};

class Engine;
class Renderer : public IdResource {
  public:
    Renderer(Engine& engine, Window* window);
    ~Renderer();

    VulkanDriver& getDriver() {
        return driver;
    }
    void run(const Scene& scene);
    void setRendererImpl(std::unique_ptr<RendererImpl> renderer) { this->impl = std::move(renderer);  }

    MeshPtr createMesh(const std::span<Mesh::Vertex> vertices, const std::span<uint32_t>& indices);

    Handle<HwRenderTarget> surfaceRenderTarget;
    Handle<HwPrimitive> quadPrimitive;
    Handle<HwProgram> getShaderProgram(const std::string& shaderName);
  private:
    Handle<HwProgram> loadShader(const std::string& filename);
    void registerShader(const std::string& shaderName, const std::string& filename);
    void render(InflightContext& context, const Scene& scene);
    void registerPrograms();
    void prepareSceneData(InflightContext& context, const Scene& scene);

    std::array<InflightData, MAX_INFLIGHTS> inflights;

    std::unordered_map<uint32_t, Handle<HwBLAS>> blasCache;
    std::unordered_map<std::string, Handle<HwProgram> > programs;
    std::unique_ptr<RendererImpl> impl;

    VulkanDriver driver;
    Engine& engine;
    Window* window;

    uint32_t nextMeshId{ 1 };

    size_t currentFrame{0};
};

}  // namespace GPUSpectral