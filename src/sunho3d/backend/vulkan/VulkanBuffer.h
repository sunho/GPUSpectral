#pragma once

#include <vulkan/vulkan.hpp>
#include <stdexcept>

#include "../DriverBase.h"
#include "../Handles.h"
#include "VulkanDevice.h"

struct VulkanBufferObject : public HwBufferObject {
    explicit VulkanBufferObject(VulkanDevice &device, uint32_t size, BufferUsage usage);
    ~VulkanBufferObject();
    void upload(const BufferDescriptor &descriptor);
    std::vector<char> download();
    vk::Buffer buffer;

  private:
    VulkanDevice &device;
    AllocatedBuffer _buffer;
};
