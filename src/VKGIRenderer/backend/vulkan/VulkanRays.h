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
    glm::mat4x3 transfom;
};

struct VulkanRTSceneDescriptor {
    std::vector<VulkanRTInstance> instances;
};

class VulkanRayTracer {
public:
    friend class VulkanRayFrameContext;
    friend class VulkanBLAS;
    friend class VulkanTLAS;
    VulkanRayTracer(VulkanDevice& device) {};
    ~VulkanRayTracer() {};

    void buildBLAS(VulkanRayFrameContext& frame, VulkanBLAS* blas, VulkanPrimitive* primitive) {};
    void buildTLAS(VulkanRayFrameContext& frame, VulkanTLAS* tlas, const VulkanRTSceneDescriptor& descriptor) {};
    void intersectRays(VulkanRayFrameContext& frame, VulkanTLAS* tlas, uint32_t rayCount, VulkanBufferObject* raysBuffer, VulkanBufferObject* hitBuffer) {};

  private:
};

class VulkanRayFrameContext {
public:
    friend class VulkanRayTracer;
    VulkanRayFrameContext(VulkanRayTracer& tracer, VulkanDevice& device, vk::CommandBuffer cmd) {};
    ~VulkanRayFrameContext() {};

    
    VulkanBufferObject* acquireTemporaryBuffer(size_t size);

private:
    vk::CommandBuffer cmd{};
    std::list<VulkanBufferObject> tempBuffers{};
};

struct VulkanBLAS : public HwBLAS {
    VulkanBLAS(VulkanRayTracer& parent) : parent(parent) {
    
    }
    ~VulkanBLAS() {
    }
    std::unique_ptr<VulkanBufferObject> geometry{};
    VulkanRayTracer& parent;
};

struct VulkanTLAS : public HwTLAS {
    VulkanTLAS(VulkanRayTracer& parent) : parent(parent) {
    
    }
    ~VulkanTLAS() {
    }
    std::unique_ptr<VulkanBufferObject> scene{};
    VulkanRayTracer& parent;
};


