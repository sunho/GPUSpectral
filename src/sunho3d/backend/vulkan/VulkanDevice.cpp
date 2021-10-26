#include "VulkanDevice.h"
#include "VulkanWSI.h"
#include "VulkanPipelineCache.h"

VulkanDevice::VulkanDevice(sunho3d::Window* window) : semaphorePool(*this) {
    // Init vulkan instance
    vkb::InstanceBuilder builder;
    auto instRet = builder.set_app_name("Example Vulkan Application")
                      .request_validation_layers ()
                      .use_default_debug_messenger ()
                      .build ();
    if (!instRet) {
      throw std::runtime_error("error init vulkan instance");
    }
    vkbInstance = instRet.value();
    instance = vkbInstance.instance;

    // Create wsi instance and surface
    wsi = std::make_unique<VulkanWSI>(window, this);

    // Init vulkan device
    vkb::PhysicalDeviceSelector selector{ vkbInstance };
    auto physRet = selector.set_surface(wsi->surface)
                    .add_required_extension("VK_KHR_ray_tracing_pipeline")
                    .add_required_extension_features(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR)
                    .add_required_extension_features(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR)
                       .select ();
    if (!physRet) {
       throw std::runtime_error("error get physical device");
    }

    vkb::DeviceBuilder device_builder{ physRet.value () };

    auto devRet = device_builder.build();
    if (!devRet) {
       throw std::runtime_error("error building device");
    }
    vkbDevice = devRet.value();
    device = vkbDevice.device;
    physicalDevice = vkbDevice.physical_device;

    // Init graphics queue
    auto graphicsQueueRet = vkbDevice.get_queue(vkb::QueueType::graphics);
    if (!graphicsQueueRet) {
       throw std::runtime_error("error getting graphics queue");
    }
    graphicsQueue = graphicsQueueRet.value();

    // Init wsi
    wsi->initSwapchain();

    // Init upload context:
    auto fi = vk::FenceCreateInfo();
    auto info = vk::CommandPoolCreateInfo()
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(vkbDevice.get_queue_index(vkb::QueueType::graphics).value());
    
    uploadContext.uploadFence = device.createFence(fi);
    uploadContext.commandPool = device.createCommandPool(info);

    // Init allocator    
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = vkbDevice.physical_device.physical_device;
    allocatorInfo.device = vkbDevice.device;
    allocatorInfo.instance = vkbInstance.instance;
    vmaCreateAllocator(&allocatorInfo, &allocator);

    cache = std::make_unique<VulkanPipelineCache>(*this);
}

VulkanDevice::~VulkanDevice() {
    vmaDestroyAllocator(allocator);

    device.destroyCommandPool(uploadContext.commandPool);
    device.destroyFence(uploadContext.uploadFence);
    vkb::destroy_device(vkbDevice);
    vkb::destroy_instance(vkbInstance);
}

AllocatedBuffer VulkanDevice::allocateBuffer(vk::BufferCreateInfo info, VmaMemoryUsage usage) {
    AllocatedBuffer out;
    VkBuffer buffer;
    VkBufferCreateInfo ci = info;

    VmaAllocationCreateInfo alloc = {};
    alloc.usage = usage;
    if (vmaCreateBuffer(allocator, &ci, &alloc,
        &buffer,
        &out.allocation,
        nullptr) != VK_SUCCESS) {
        throw std::runtime_error("create buffer error");
    }
    out.buffer = buffer;
    return out;
}

AllocatedImage VulkanDevice::allocateImage(vk::ImageCreateInfo info, VmaMemoryUsage usage) {
    AllocatedImage out = {};

    VkImageCreateInfo ie = info;
    VkImage image;
    VmaAllocationCreateInfo alloc = {};
    alloc.usage = usage;
	if (vmaCreateImage(allocator, &ie, &alloc, &image, &out.allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("create image error");
    }
    out.image = image;
    return out;
}

SemaphorePool::~SemaphorePool() {
    for (auto sem : semaphores) {
        device.device.destroySemaphore(sem);
    }
}

void SemaphorePool::recycle(vk::Semaphore sem) {
    semaphores.push_back(sem);
}

vk::Semaphore SemaphorePool::acquire() {
    if (semaphores.empty()) {
        auto sem = device.device.createSemaphore(vk::SemaphoreCreateInfo());
        return sem;
    }
    auto sem = semaphores.back();
    semaphores.pop_back();
    return sem;
}
