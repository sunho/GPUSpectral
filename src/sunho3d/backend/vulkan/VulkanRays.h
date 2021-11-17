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
    friend class VulkanBLAS;
    friend class VulkanTLAS;
    VulkanRayTracer(VulkanDevice& device);
    ~VulkanRayTracer();

    void buildBLAS(VulkanRayFrameContext& frame, VulkanBLAS* blas, VulkanPrimitive* primitive);
    void buildTLAS(VulkanRayFrameContext& frame, VulkanTLAS* tlas, const VulkanRTSceneDescriptor& descriptor);
    void intersectRays(VulkanRayFrameContext& frame, VulkanTLAS* tlas, uint32_t rayCount, VulkanBufferObject* raysBuffer, VulkanBufferObject* hitBuffer);
    RRDevicePtr getDevicePtr(vk::Buffer buffer);

  private:
    VulkanDevice& device;
    RRContext context{};
};

class VulkanRayFrameContext {
public:
    friend class VulkanRayTracer;
    VulkanRayFrameContext(VulkanRayTracer& tracer, VulkanDevice& device, vk::CommandBuffer cmd);
    ~VulkanRayFrameContext();

    RRDevicePtr getTempDevicePtr(vk::Buffer buffer);
    
    VulkanBufferObject* acquireTemporaryBuffer(size_t size);

private:
    VulkanRayTracer& tracer;
    VulkanDevice& device;
    vk::CommandBuffer cmd{};
    RRCommandStream stream{};
    std::list<VulkanBufferObject> tempBuffers{};
    std::list<RRDevicePtr> ptrs{};
};

struct VulkanBLAS : public HwBLAS {
    VulkanBLAS(VulkanRayTracer& parent) : parent(parent) {
    
    }
    ~VulkanBLAS() {
        rrReleaseDevicePtr(parent.context, geometryPtr);
    }
    std::unique_ptr<VulkanBufferObject> geometry{};
    RRDevicePtr geometryPtr{};
    VulkanRayTracer& parent;
};

struct VulkanTLAS : public HwTLAS {
    VulkanTLAS(VulkanRayTracer& parent) : parent(parent) {
    
    }
    ~VulkanTLAS() {
        rrReleaseDevicePtr(parent.context, scenePtr);
    }
    std::unique_ptr<VulkanBufferObject> scene{};
    RRDevicePtr scenePtr{};
    VulkanRayTracer& parent;
};


