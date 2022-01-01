#include "VulkanDevice.h"
#include "VulkanWSI.h"
#include "VulkanPipelineCache.h"
#include <VKGIRenderer/utils/Util.h>

VulkanDevice::VulkanDevice(VKGIRenderer::Window* window) : semaphorePool(*this) {
    // Init vulkan instance
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT physicalDeviceDescriptorIndexingFeatures{};
    physicalDeviceDescriptorIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
    physicalDeviceDescriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    physicalDeviceDescriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    physicalDeviceDescriptorIndexingFeatures.runtimeDescriptorArray                     = VK_TRUE;
    physicalDeviceDescriptorIndexingFeatures.descriptorBindingVariableDescriptorCount   = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    accelFeature.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
    rtPipelineFeature.rayTracingPipeline = VK_TRUE;
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bdaFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR };
    bdaFeature.bufferDeviceAddress = VK_TRUE;
    
   
    vkb::InstanceBuilder builder;

    auto instRet = builder.set_app_name("Example Vulkan Application")
                    .require_api_version(VK_API_VERSION_1_2)
                      .request_validation_layers ()
                    .enable_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)
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
                    .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
                    //.add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
                    .add_required_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
                    .add_required_extension(VK_EXT_SHADER_SUBGROUP_BALLOT_EXTENSION_NAME)
                    .add_required_extension(VK_KHR_MAINTENANCE3_EXTENSION_NAME)
                    .add_required_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
                    .add_required_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)
                    .add_required_extension_features(physicalDeviceDescriptorIndexingFeatures)
                    .add_required_extension_features(accelFeature)
                    .add_required_extension_features(rtPipelineFeature)
        .add_required_extension_features(bdaFeature)

                       .select ();
    if (!physRet) {
       throw std::runtime_error("error get physical device");
    }

    vkb::DeviceBuilder device_builder{ physRet.value () };

    auto devRet = device_builder
        .build();
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
    queueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    auto fi = vk::FenceCreateInfo();
    auto ci = vk::CommandPoolCreateInfo()
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(vkbDevice.get_queue_index(vkb::QueueType::graphics).value());
    
    uploadContext.uploadFence = device.createFence(fi);
    uploadContext.commandPool = device.createCommandPool(ci);

    auto ci2 = vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(vkbDevice.get_queue_index(vkb::QueueType::graphics).value());
    commandPool = device.createCommandPool(ci2);

    // Init allocator    
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = vkbDevice.physical_device.physical_device;
    allocatorInfo.device = vkbDevice.device;
    allocatorInfo.instance = vkbInstance.instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &allocator);

    cache = std::make_unique<VulkanPipelineCache>(*this);

    vk::DynamicLoader dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    dld = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr, device);

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties = {};
    rayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 deviceProperties2{};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties2.pNext = &rayTracingPipelineProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);
    shaderGroupHandleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
    shaderGroupHandleSizeAligned = align(rayTracingPipelineProperties.shaderGroupHandleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);
}

VulkanDevice::~VulkanDevice() {
    vmaDestroyAllocator(allocator);

    device.destroyCommandPool(uploadContext.commandPool);
    device.destroyFence(uploadContext.uploadFence);
    vkb::destroy_device(vkbDevice);
    vkb::destroy_instance(vkbInstance);
}

void VulkanDevice::immediateSubmit(std::function<void(vk::CommandBuffer)> func) {
    auto cmdInfo = vk::CommandBufferAllocateInfo()
        .setCommandBufferCount(1)
        .setCommandPool(uploadContext.commandPool);
    auto cmd = device.allocateCommandBuffers(cmdInfo).front();

    auto beginInfo = vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(beginInfo);
	func(cmd);
    cmd.end();

    auto submitInfo = vk::SubmitInfo()
        .setCommandBufferCount(1)
        .setPCommandBuffers(&cmd);
    graphicsQueue.submit(submitInfo, uploadContext.uploadFence);

    device.waitForFences(1, &uploadContext.uploadFence, true, UINT64_MAX);
	device.resetFences(1, &uploadContext.uploadFence);

    device.resetCommandPool(uploadContext.commandPool);
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

uint64_t VulkanDevice::getBufferDeviceAddress(vk::Buffer buffer) {
    VkBufferDeviceAddressInfoKHR ai = {};
    ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    ai.buffer = buffer;
    return dld.vkGetBufferDeviceAddressKHR(device, &ai);
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

void AllocatedBuffer::map(VulkanDevice& device, void** data) {
    vmaMapMemory(device.allocator, allocation, data);
}

void AllocatedBuffer::unmap(VulkanDevice& device) {
    vmaUnmapMemory(device.allocator, allocation);
}

void AllocatedBuffer::destroy(VulkanDevice& device) {
    vmaDestroyBuffer(device.allocator, buffer, allocation);
}

void AllocatedImage::map(VulkanDevice& device, void** data) {
    vmaMapMemory(device.allocator, allocation, data);
}

void AllocatedImage::unmap(VulkanDevice& device) {
    vmaUnmapMemory(device.allocator, allocation);
}

void AllocatedImage::destroy(VulkanDevice& device) {
    vmaDestroyImage(device.allocator, image, allocation);
}
