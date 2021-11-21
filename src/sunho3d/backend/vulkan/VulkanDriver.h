#pragma once

#include <sunho3d/Window.h>
#include <vulkan/vulkan.h>

#include <map>
#include <optional>
#include <vector>
#include <cassert>
#include <Tracy.hpp>
#include <TracyVulkan.hpp>


#include "../DriverBase.h"
#include "../PipelineState.h"
#include "VulkanBuffer.h"
#include "VulkanDevice.h"
#include "VulkanHandles.h"
#include "VulkanPipelineCache.h"
#include "VulkanTexture.h"
#include "VulkanRays.h"

namespace sunho3d {
using HandleData = std::vector<char>;

struct VulkanInflight : public HwInflight {
    VulkanInflight() = delete;
    static vk::CommandBuffer createCommandBuffer(VulkanDevice& device) {
        auto cmdInfo = vk::CommandBufferAllocateInfo()
            .setCommandBufferCount(1)
            .setCommandPool(device.commandPool);
        return device.device.allocateCommandBuffers(cmdInfo).front();
    }
    VulkanInflight(VulkanDevice& device, VulkanRayTracer& tracer) : 
            device(device), 
            cmd(createCommandBuffer(device)), 
            rayFrameContext(tracer, device, cmd) {
        imageSemaphore = device.semaphorePool.acquire();
        renderSemaphore = device.semaphorePool.acquire();
    }
    ~VulkanInflight() {
        device.device.freeCommandBuffers(device.commandPool, 1, &cmd);
        device.semaphorePool.recycle(renderSemaphore);
        device.semaphorePool.recycle(imageSemaphore);
    }
    vk::Fence inflightFence{};
    vk::Semaphore imageSemaphore{};
    vk::Semaphore renderSemaphore{};
    VulkanDevice& device;
    vk::CommandBuffer cmd{};
    VulkanRayFrameContext rayFrameContext;
};

struct DriverContext { 
    VulkanRenderTarget *currentRenderTarget{};
    vk::RenderPass currentRenderPass{};
    Viewport viewport{};
    VulkanInflight* inflight{};
    TracyVkCtx tracyContext{ nullptr };
    std::string profileSectionName;
};

class VulkanDriver {
  public:
    VulkanDriver(Window *window);
    ~VulkanDriver();

    VulkanDriver(const VulkanDriver &) = delete;
    VulkanDriver &operator=(const VulkanDriver &) = delete;
    VulkanDriver(VulkanDriver &&) = delete;
    VulkanDriver &operator=(VulkanDriver &&) = delete;

#define DECL_COMMAND(R, N, ARGS, PARAMS) R N(ARGS);
#define DECL_VOIDCOMMAND(N, ARGS, PARAMS) void N(ARGS);

#include "../Command.inc"

#undef DECL_VOIDCOMMAND
#undef DECL_COMMAND

  private:
    void setupDebugMessenger();
    VulkanBindings translateBindingMap(const BindingMap& binds);
    std::string profileZoneName(std::string zoneName);
    

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);

    VkDebugUtilsMessengerEXT debugMessenger;

    template <typename Dp, typename B>
    Handle<B> allocHandle() {
        handles[nextId] = HandleData(sizeof(Dp));
        return Handle<B>(nextId++);
    }

    template <typename Dp, typename B>
    Dp *handleCast(Handle<B> handle) noexcept {
        if (!handle)
            return nullptr;
        auto iter = handles.find(handle.getId());
        assert(iter != handles.end());
        HandleData &data = iter->second;
        return reinterpret_cast<Dp *>(data.data());
    }

    template <typename Dp, typename B>
    const Dp *handleConstCast(const Handle<B> &handle) noexcept {
        if (!handle)
            return nullptr;
        auto iter = handles.find(handle.getId());
        HandleData &data = iter->second;
        return reinterpret_cast<const Dp *>(data.data());
    }

    template <typename Dp, typename B, typename... ARGS>
    Dp *constructHandle(Handle<B> &handle, ARGS &&... args) noexcept {
        if (!handle)
            return nullptr;
        auto iter = handles.find(handle.getId());
        HandleData &data = iter->second;
        Dp *addr = reinterpret_cast<Dp *>(data.data());
        new (addr) Dp(std::forward<ARGS>(args)...);
        return addr;
    }

    template <typename Dp, typename B>
    void destructHandle(const Handle<B> &handle) noexcept {
        auto iter = handles.find(handle.getId());
        HandleData &data = iter->second;
        reinterpret_cast<Dp *>(data.data())->~Dp();
        handles.erase(handle.getId());
    }

    std::map<HandleBase::HandleId, HandleData> handles;
    HandleBase::HandleId nextId{ 0 };

    std::unique_ptr<VulkanDevice> device;
    std::unique_ptr<VulkanRayTracer> rayTracer;
    DriverContext context;
};

}  // namespace sunho3d
