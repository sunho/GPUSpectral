#pragma once

#include <sunho3d/framegraph/FrameGraph.h>

#include "Camera.h"
#include "Transform.h"
#include "Window.h"
#include "backend/vulkan/VulkanDriver.h"
#include "utils/ResourceList.h"

namespace sunho3d {

struct MaterialBuffer {
    glm::vec4 specular;
    float phong;
    int pad[3];
};

class Scene;
class Renderer;

class RenderGraph {
  public:
    RenderGraph(Renderer& renderer);
    ~RenderGraph();
    void reset();
    void addRenderPass(const std::string& name, std::vector<FGResource> inputs, std::vector<FGResource> outputs, RenderPass::RenderPassFunc func);
    void submit();

#define SCRATCH_IMPL(RESOURCENAME, METHODNAME)                        \
    template <typename... ARGS>                                       \
    Handle<Hw##RESOURCENAME> METHODNAME(ARGS&&... args) {             \
        auto handle = driver.METHODNAME(std::forward<ARGS>(args)...); \
        destroyers.push_back([handle, this]() {                       \
            this->driver.destroy##RESOURCENAME(handle);               \
        });                                                           \
        return handle;                                                \
    }

    SCRATCH_IMPL(RenderTarget, createDefaultRenderTarget)
    SCRATCH_IMPL(RenderTarget, createRenderTarget)
    SCRATCH_IMPL(Texture, createTexture)
    SCRATCH_IMPL(UniformBuffer, createUniformBuffer)

#undef SCRATCH_IMPL

    template <typename T>
    FGResource declareResource(const std::string& name) {
        return fg.declareResource<T>(name);
    }

    template <typename T>
    void defineResource(const FGResource& resource, const T& t) {
        fg.defineResource<T>(resource, t);
    }

  private:
    template <typename T, typename... ARGS>
    T callDriverMethod(T (VulkanDriver::*mf)(ARGS...), ARGS&&... args) noexcept {
        return (driver.*mf)(std::forward<ARGS>(args)...);
    }

    std::vector<std::function<void()>> destroyers;
    Renderer& parent;
    VulkanDriver& driver;
    FrameGraph fg;
};

struct InflightData {
    Handle<HwFence> fence;
    Handle<HwInflight> handle{};
    std::unique_ptr<RenderGraph> rg;
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
    Handle<HwUniformBuffer> createTransformBuffer(RenderGraph& rg, const Camera& camera, const glm::mat4& model);

    Handle<HwProgram> fowradPassProgram;
    Handle<HwProgram> quadDrawProgram;
    Handle<HwRenderTarget> surfaceRenderTarget;
    Handle<HwPrimitive> quadPrimitive;

    std::array<InflightData, MAX_INFLIGHTS> inflights;
    VulkanDriver driver;
    Window* window;

    size_t currentFrame{0};
};

}  // namespace sunho3d
