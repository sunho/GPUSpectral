#pragma once
#include <radeonrays_vlk.h>
#include <sunho3d/backend/DriverBase.h>

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
    VulkanRayTracer(VulkanDevice& device);
    ~VulkanRayTracer();

    void buildBLAS(VulkanRayFrameContext& frame, VulkanBLAS* blas, VulkanPrimitive* primitive);
    void buildTLAS(VulkanRayFrameContext& frame, VulkanTLAS* tlas, const VulkanRTSceneDescriptor& descriptor);
    void intersectRays(VulkanRayFrameContext& frame, VulkanTLAS* tlas, uint32_t rayCount, VulkanBufferObject* raysBuffer, VulkanBufferObject* hitBuffer);

private:
    VulkanDevice& device;
    RRContext context{};
};

class VulkanRayFrameContext {
public:
    friend class VulkanRayTracer;
    VulkanRayFrameContext(VulkanRayTracer& tracer, VulkanDevice& device, vk::CommandBuffer cmd);
    ~VulkanRayFrameContext();
    
    VulkanBufferObject* acquireTemporaryBuffer(size_t size);

private:
    VulkanRayTracer& tracer;
    VulkanDevice& device;
    vk::CommandBuffer cmd{};
    RRCommandStream stream{};
    std::list<VulkanBufferObject> tempBuffers{};
};

struct VulkanBLAS : public HwBLAS {
    VulkanBLAS() = default;
    ~VulkanBLAS() = default;
    std::unique_ptr<VulkanBufferObject> geometry{};
};

struct VulkanTLAS : public HwTLAS {
    VulkanTLAS() = default;
    ~VulkanTLAS() = default;
    std::unique_ptr<VulkanBufferObject> scene{};
};


