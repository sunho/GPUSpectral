#pragma once

#include <sunho3d/Window.h>

#include "../DriverBase.h"
#include "../PipelineState.h"
#include "VulkanPipelineCache.h"
#include "VulkanHandles.h"
#include "VulkanContext.h"
#include "VulkanBuffer.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <map>
#include <optional>

namespace sunho3d {

using HandleData = std::vector<char>;

class VulkanDriver {
public:
    VulkanDriver(Window* window);
    ~VulkanDriver();
    
    VulkanDriver(const VulkanDriver&) = delete;
    VulkanDriver& operator=(const VulkanDriver&) = delete;
    VulkanDriver(VulkanDriver&&) = delete;
    VulkanDriver& operator=(VulkanDriver&&) = delete;

#define DECL_COMMAND(R, N, ARGS, PARAMS) R N(ARGS);
#define DECL_VOIDCOMMAND(N, ARGS, PARAMS) void N(ARGS);

#include "../Command.inc"

#undef DECL_VOIDCOMMAND
#undef DECL_COMMAND
private:
    void setupDebugMessenger();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

    VkDebugUtilsMessengerEXT debugMessenger;
    
    template<typename Dp, typename B>
    Handle<B> alloc_handle() {
        handles[nextId] = HandleData(sizeof(Dp));
        return Handle<B>(nextId++);
    }

    template<typename Dp, typename B>
    Dp* handle_cast(Handle<B> handle) noexcept {
        if (!handle) return nullptr;
        auto iter = handles.find(handle.getId());
        assert(iter != handles.end());
        HandleData& data = iter->second;
        return reinterpret_cast<Dp*>(data.data());
    }

    template<typename Dp, typename B>
    const Dp* handle_const_cast(const Handle<B>& handle) noexcept {
        if (!handle) return nullptr;
        auto iter = handles.find(handle.getId());
        HandleData& data = iter->second;
        return reinterpret_cast<const Dp*>(data.data());
    }

    template<typename Dp, typename B, typename ... ARGS>
    Dp* construct_handle(Handle<B>& handle, ARGS&& ... args) noexcept {
        if (!handle) return nullptr;
        auto iter = handles.find(handle.getId());
        HandleData& data = iter->second;
        Dp* addr = reinterpret_cast<Dp*>(data.data());
        new(addr) Dp(std::forward<ARGS>(args)...);
        return addr;
    }

    template<typename Dp, typename B>
    void destruct_handle(const Handle<B>& handle) noexcept {
        auto iter = handles.find(handle.getId());
        HandleData& data = iter->second;
        reinterpret_cast<Dp*>(data.data())->~Dp();
        handles.erase(handle.getId());
    }
    
    std::map<HandleBase::HandleId, HandleData> handles;
    HandleBase::HandleId nextId { 0 };

    VulkanPipelineCache pipelineCache;
    VulkanContext context;
    VulkanSurfaceContext surface;
};

}
