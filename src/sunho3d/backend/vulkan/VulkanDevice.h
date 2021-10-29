#pragma once
#include <VkBootstrap.h>
#include <sunho3d/Window.h>
#include "VulkanWSI.h"
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <vector>

#include "../DriverTypes.h"

struct VulkanSwapContext;
struct VulkanTexture;
struct VulkanSurfaceContext;
class VulkanPipelineCache;

class SemaphorePool {
public:
    SemaphorePool() = delete;
    SemaphorePool(VulkanDevice& device) : device(device) { }
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
    VulkanDevice(sunho3d::Window* window);
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

    void immediateSubmit(std::function<void(vk::CommandBuffer)> func);
    AllocatedBuffer allocateBuffer(vk::BufferCreateInfo info, VmaMemoryUsage usage);
    AllocatedImage allocateImage(vk::ImageCreateInfo info, VmaMemoryUsage usage);
private:
    vkb::Device vkbDevice{};
    vkb::Instance vkbInstance{};

    struct {
        vk::Fence uploadFence{};
        vk::CommandPool commandPool{};
    } uploadContext;
};