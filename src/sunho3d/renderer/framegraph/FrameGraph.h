#pragma once

#include <unordered_map>

#include "../../utils/FixedVector.h"
#include "../../backend/vulkan/VulkanDriver.h"
#include "RenderPass.h"
#include "Resource.h"

namespace sunho3d {
class Renderer;
}

class FrameGraph {
  public:
    FrameGraph(sunho3d::Renderer& renderer);
    ~FrameGraph();

    void reset();
    void addRenderPass(const std::string& name, std::vector<FGResource> inputs, std::vector<FGResource> outputs, RenderPass::RenderPassFunc func);
    void submit();

#define SCRATCH_IMPL(RESOURCENAME, METHODNAME)                        \
    template <typename... ARGS>                                       \
    Handle<Hw##RESOURCENAME> METHODNAME##SC(ARGS&&... args) {         \
        auto handle = driver.METHODNAME(std::forward<ARGS>(args)...); \
        destroyers.push_back([handle, this]() {                       \
            this->driver.destroy##RESOURCENAME(handle);               \
        });                                                           \
        return handle;                                                \
    }

    SCRATCH_IMPL(RenderTarget, createDefaultRenderTarget)
    SCRATCH_IMPL(RenderTarget, createRenderTarget)
    SCRATCH_IMPL(BufferObject, createBufferObject)
    SCRATCH_IMPL(Texture, createTexture)
    SCRATCH_IMPL(UniformBuffer, createUniformBuffer)

#undef SCRATCH_IMPL

    void addRenderPass(const RenderPass& pass);
    void compile();
    void run();

    template <typename T>
    FGResource declareResource(const std::string& name) {
        resources.emplace(nextId, FixedVector<char>(sizeof(T)));
        FGResource resource = {
            .name = name,
            .id = nextId++
        };
        return resource;
    }

    template <typename T>
    T getResource(const FGResource& resource) {
        auto& data = resources.find(resource.id)->second;
        return *reinterpret_cast<T*>(data.data());
    }

    template <typename T>
    void defineResource(const FGResource& resource, const T& t) {
        auto& data = resources.find(resource.id)->second;
        auto addr = reinterpret_cast<T*>(data.data());
        *addr = t;
    }

  private:
    std::unordered_map<uint32_t, FixedVector<char>> resources;
    std::vector<RenderPass> renderPasses;
    std::vector<size_t> runOrder;
    uint32_t nextId{ 1 };

    template <typename T, typename... ARGS>
    T callDriverMethod(T (sunho3d::VulkanDriver::*mf)(ARGS...), ARGS&&... args) noexcept {
        return (driver.*mf)(std::forward<ARGS>(args)...);
    }

    std::vector<std::function<void()>> destroyers;
    sunho3d::Renderer& parent;
    sunho3d::VulkanDriver& driver;
};
