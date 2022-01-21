#pragma once
#include <GPUSpectral/engine/Window.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "VulkanWSI.h"

#include "../DriverTypes.h"

struct VulkanSwapContext;
struct VulkanTexture;
struct VulkanSurfaceContext;
class VulkanPipelineCache;

class SemaphorePool {
  public:
    SemaphorePool() = delete;
    SemaphorePool(VulkanDevice& device)
        : device(device) {
    }
    ~SemaphorePool();

    void recycle(vk::Semaphore sem);
    vk::Semaphore acquire();

  private:
    std::vector<vk::Semaphore> semaphores;
    VulkanDevice& device;
};

struct AllocatedBuffer {
    vk::Buffer buffer;
    VmaAllocation allocation;
    void map(VulkanDevice& device, void** data);
    void unmap(VulkanDevice& device);
    void destroy(VulkanDevice& device);
};

struct AllocatedImage {
    vk::Image image;
    VmaAllocation allocation;
    void map(VulkanDevice& device, void** data);
    void unmap(VulkanDevice& device);
    void destroy(VulkanDevice& device);
};

class VulkanDevice {
  public:
    friend class VulkanWSI;
    friend class AllocatedBuffer;
    friend class AllocatedImage;
    VulkanDevice() = delete;
    VulkanDevice(GPUSpectral::Window* window);
    ~VulkanDevice();
    vk::Instance instance{};
    vk::Device device{};
    vk::PhysicalDevice physicalDevice{};
    vk::Queue graphicsQueue{};
    uint32_t queueFamily{};
    vk::CommandPool commandPool{};
    std::unique_ptr<VulkanWSI> wsi{};
    std::unique_ptr<VulkanPipelineCache> cache{};
    VmaAllocator allocator{};
    SemaphorePool semaphorePool;
    vk::DispatchLoaderDynamic dld;
    size_t shaderGroupHandleSize;
    size_t shaderGroupHandleSizeAligned;

    void immediateSubmit(std::function<void(vk::CommandBuffer)> func);
    AllocatedBuffer allocateBuffer(vk::BufferCreateInfo info, VmaMemoryUsage usage);
    AllocatedImage allocateImage(vk::ImageCreateInfo info, VmaMemoryUsage usage);
    uint64_t getBufferDeviceAddress(vk::Buffer buffer);

  private:
    vkb::Device vkbDevice{};
    vkb::Instance vkbInstance{};

    struct {
        vk::Fence uploadFence{};
        vk::CommandPool commandPool{};
    } uploadContext;
};