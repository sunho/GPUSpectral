#pragma once

#include "framegraph/FrameGraph.h"

#include "Camera.h"
#include "Transform.h"
#include "Mesh.h"
#include "../engine/Window.h"
#include "../backend/vulkan/VulkanDriver.h"

namespace GPUSpectral {

using MeshPtr = std::shared_ptr<Mesh>;

class Scene;
struct SceneData;
struct LightData;
class Renderer;

constexpr static size_t MAX_INFLIGHTS = 2; 

class RenderPassCreator {
public:
    virtual void createRenderPass(FrameGraph& fg, const Scene& scene) = 0;
};

class Engine;
class Renderer {
  public:
    struct InflightData {
        Handle<HwFence> fence;
        Handle<HwInflight> handle{};
        std::unique_ptr<FrameGraph> rg;
    };

    Renderer(Engine& engine, Window* window);
    ~Renderer();

    void addRenderPassCreator(std::unique_ptr<RenderPassCreator> creator);

    HwDriver& getDriver() {
        return *driver;
    }
    void run(const Scene& scene);

    MeshPtr createMesh(const std::span<Mesh::Vertex> vertices, const std::span<uint32_t>& indices);
    Handle<HwBLAS> getOrCreateBLAS(const MeshPtr& meshPtr);

    Handle<HwRenderTarget> surfaceRenderTarget;
    Handle<HwPrimitive> quadPrimitive;
    Handle<HwProgram> getShaderProgram(const std::string& shaderName);
  private:
    Handle<HwProgram> loadShader(const std::string& filename);

    std::array<InflightData, MAX_INFLIGHTS> inflights;

    std::unordered_map<uint32_t, Handle<HwBLAS>> blasCache;
    std::unordered_map<std::string, Handle<HwProgram> > programs;
    std::list<std::unique_ptr<RenderPassCreator>> renderPassCreators;

    std::unique_ptr<HwDriver> driver;
    Engine& engine;
    Window* window;

    uint32_t nextMeshId{ 1 };

    size_t currentFrame{0};
};

}  // namespace GPUSpectral