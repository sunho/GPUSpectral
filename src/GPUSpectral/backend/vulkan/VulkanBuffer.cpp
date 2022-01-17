#include "VulkanBuffer.h"
#include "VulkanTypes.h"

#include <vulkan/vulkan.hpp>
#include <stdexcept>
#include <Tracy.hpp>

VulkanBufferObject::VulkanBufferObject(VulkanDevice &device, uint32_t size, BufferUsage usage, BufferType type)
    : device(device), HwBufferObject(size), type(type) {
    ZoneScopedN("Buffer create")
    auto bufferInfo = vk::BufferCreateInfo()
        .setSize(size)
        .setUsage(translateBufferUsage(usage) | vk::BufferUsageFlagBits::eTransferSrc);
    VmaMemoryUsage allocType = VMA_MEMORY_USAGE_GPU_ONLY;
    if (type == BufferType::HOST_COHERENT) {
        allocType = VMA_MEMORY_USAGE_CPU_ONLY;
    } else {
        bufferInfo.usage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    _buffer = device.allocateBuffer(bufferInfo, allocType);
    buffer = _buffer.buffer;
    if (type == BufferType::HOST_COHERENT) {
        _buffer.map(device, &mapped);
    }
}

VulkanBufferObject::~VulkanBufferObject() {
    if (mapped) {
        _buffer.unmap(device);
    }
    _buffer.destroy(device);
}

void VulkanBufferObject::uploadSync(const BufferDescriptor& descriptor) {
    ZoneScopedN("Buffer upload sync")
    size_t uploadSize = descriptor.size ? descriptor.size : size;
    auto bi = vk::BufferCreateInfo()
                  .setSize(uploadSize)
                  .setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    auto staging = device.allocateBuffer(bi, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data;
    staging.map(device, &data);
    memcpy(data, descriptor.data, uploadSize);
    staging.unmap(device);
    device.immediateSubmit([=](vk::CommandBuffer cmd) {
        vk::BufferCopy copy = {};
        copy.dstOffset = 0;
        copy.srcOffset = 0;
        copy.size = uploadSize;
        cmd.copyBuffer(staging.buffer, _buffer.buffer, 1, &copy);
    });

}
void VulkanBufferObject::copy(vk::CommandBuffer cmd, const VulkanBufferObject& obj) {
    vk::BufferCopy copy = {};
    copy.dstOffset = 0;
    copy.srcOffset = 0;
    copy.size = size;
    cmd.copyBuffer(obj.buffer, _buffer.buffer, 1, &copy);
}

std::vector<char> VulkanBufferObject::download() {
    std::vector<char> buf(size);
    auto bi = vk::BufferCreateInfo()
            .setSize(size)
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    auto staging = device.allocateBuffer(bi, VMA_MEMORY_USAGE_CPU_ONLY);
    device.immediateSubmit([=](vk::CommandBuffer cmd) {
		vk::BufferCopy copy = {};
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = size;
        cmd.copyBuffer(_buffer.buffer, staging.buffer, 1, &copy);
	});
    void* data;
    staging.map(device, &data);
    memcpy(buf.data(), data, size);
    staging.unmap(device);
    return buf;
}
