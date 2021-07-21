#pragma once

#include <vulkan/vulkan.h>
#include <stdexcept>

#include "../DriverBase.h"
#include "../Handles.h"
#include "VulkanContext.h"

struct VulkanBufferObject : public HwBufferObject {
    explicit VulkanBufferObject(VulkanContext &context, uint32_t size, BufferUsage usage);
    ~VulkanBufferObject();
    void upload(const BufferDescriptor &descriptor);
    VkDeviceMemory memory;
    VkBuffer buffer;

  private:
    VulkanContext &context;
};
