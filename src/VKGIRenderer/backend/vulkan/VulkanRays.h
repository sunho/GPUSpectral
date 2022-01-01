#pragma once
#include <VKGIRenderer/backend/DriverBase.h>

#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "VulkanHandles.h"

class VulkanRayFrameContext;
struct VulkanBLAS;
struct VulkanTLAS;

struct VulkanRTInstance {
    VulkanBLAS* blas{};
    glm::mat4x4 transfom;
};

struct VulkanRTSceneDescriptor {
    std::vector<VulkanRTInstance> instances;
};

struct VulkanBLAS : public HwBLAS {
    VulkanBLAS(VulkanDevice& device, vk::CommandBuffer cmd, VulkanPrimitive* primitive, VulkanBufferObject** scratch);
    ~VulkanBLAS();
    VkAccelerationStructureKHR handle{};
    std::unique_ptr<VulkanBufferObject> buffer{};
    uint64_t deviceAddress{};
};

struct VulkanTLAS : public HwTLAS {
    VulkanTLAS(VulkanDevice& device, const VulkanRTSceneDescriptor& scene, VulkanBufferObject** scratch);
    ~VulkanTLAS();
    VkAccelerationStructureKHR handle{};
    std::unique_ptr<VulkanBufferObject> buffer{};
};


