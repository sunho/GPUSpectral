#include "VulkanRays.h"
#include <radeonrays_vlk.h>
#include <glm/gtc/type_ptr.hpp>
#include <Tracy.hpp>

VulkanRayTracer::VulkanRayTracer(VulkanDevice& device) : device(device) {
    auto res = rrCreateContextVk(RR_API_VERSION, device.device, device.physicalDevice, device.graphicsQueue, device.queueFamily, &context);
    if (res != RR_SUCCESS) {
        throw std::runtime_error("error create radeon rays context");
    }
}

VulkanRayTracer::~VulkanRayTracer() {
    rrDestroyContext(context);
}

void VulkanRayTracer::buildBLAS(VulkanRayFrameContext& frame, VulkanBLAS* blas, VulkanPrimitive* primitive) {
    RRDevicePtr vertex_ptr = nullptr;
    RRDevicePtr index_ptr  = nullptr;
    rrGetDevicePtrFromVkBuffer(context, primitive->vertex->buffers[0]->buffer, 0, &vertex_ptr);
    rrGetDevicePtrFromVkBuffer(context, primitive->index->buffer->buffer, 0, &index_ptr);

    RRGeometryBuildInput gbi = {};
    RRTriangleMeshPrimitive mesh = {};
    gbi.triangle_mesh_primitives = &mesh;
    gbi.primitive_type = RR_PRIMITIVE_TYPE_TRIANGLE_MESH;
    mesh.vertices = vertex_ptr;
    mesh.triangle_indices = index_ptr;
    mesh.vertex_count = primitive->vertex->vertexCount;
    mesh.vertex_stride = primitive->vertex->attributes[0].stride;
    mesh.triangle_count = primitive->index->count / 3;
    mesh.index_type = RR_INDEX_TYPE_UINT32;
    gbi.primitive_count = 1;

    RRBuildOptions options;
    options.build_flags = RR_BUILD_FLAG_BITS_PREFER_FAST_BUILD;

    RRMemoryRequirements geometry_reqs;
    auto res = rrGetGeometryBuildMemoryRequirements(context, &gbi, &options, &geometry_reqs);
    if (res != RR_SUCCESS) {
        throw std::runtime_error("error getting mem requirements");
    }

    auto tempBuffer = frame.acquireTemporaryBuffer(geometry_reqs.temporary_build_buffer_size);
    blas->geometry = std::make_unique<VulkanBufferObject>(device, geometry_reqs.result_buffer_size, BufferUsage::STORAGE);

    RRDevicePtr geometry_ptr = getDevicePtr(blas->geometry->buffer);
    blas->geometryPtr = geometry_ptr;
    RRDevicePtr scratch_ptr = frame.getTempDevicePtr(tempBuffer->buffer);

    
    res = rrCmdBuildGeometry(context, RR_BUILD_OPERATION_BUILD, &gbi, &options, scratch_ptr, geometry_ptr, frame.stream);
    if (res != RR_SUCCESS) {
        throw std::runtime_error("error writing blas build commands");
    }
}

void VulkanRayTracer::buildTLAS(VulkanRayFrameContext& frame, VulkanTLAS* tlas, const VulkanRTSceneDescriptor& descriptor) {
    ZoneScopedN("rt tlas internal")
    std::vector<RRInstance> instances;
    RRSceneBuildInput sbi = {};

    for (auto& o : descriptor.instances) {
        RRInstance instance = {};
        instance.geometry = o.blas->geometryPtr;
        glm::mat3x4 tt = glm::transpose(o.transfom);
        memcpy(&instance.transform, glm::value_ptr(tt), 4*3*4);
        instances.push_back(instance);
    }
    sbi.instances = instances.data();
    sbi.instance_count = instances.size();

    RRBuildOptions options = {};
    options.build_flags = RR_BUILD_FLAG_BITS_PREFER_FAST_BUILD;
    RRMemoryRequirements reqs = {};
    auto res = rrGetSceneBuildMemoryRequirements(context, &sbi, &options, &reqs);
    if (res != RR_SUCCESS) {
        throw std::runtime_error("error get mem requirements");
    }
    
    auto tempBuffer = frame.acquireTemporaryBuffer(reqs.temporary_build_buffer_size);
    tlas->scene = std::make_unique<VulkanBufferObject>(device, reqs.result_buffer_size, BufferUsage::STORAGE);

    RRDevicePtr scene_ptr = getDevicePtr(tlas->scene->buffer);
    tlas->scenePtr = scene_ptr;
    RRDevicePtr scratch_ptr = frame.getTempDevicePtr(tempBuffer->buffer);

    {
        ZoneScopedN("rt tlas internal build scene")
        res = rrCmdBuildScene(context, &sbi, &options, scratch_ptr, scene_ptr, frame.stream);
    }
    if (res != RR_SUCCESS) {
        throw std::runtime_error("error build scene");
    }
}

void VulkanRayTracer::intersectRays(VulkanRayFrameContext& frame, VulkanTLAS* tlas, uint32_t rayCount, VulkanBufferObject* raysBuffer, VulkanBufferObject* hitBuffer) {
    size_t scratch_size;
    rrGetTraceMemoryRequirements(context, rayCount, &scratch_size);

    auto tempBuffer = frame.acquireTemporaryBuffer(scratch_size);

    RRDevicePtr rays_ptr = frame.getTempDevicePtr(raysBuffer->buffer);
    RRDevicePtr hits_ptr = frame.getTempDevicePtr(hitBuffer->buffer);
    RRDevicePtr scratch_ptr = frame.getTempDevicePtr(tempBuffer->buffer);

    auto res = rrCmdIntersect(context,
                                 tlas->scenePtr,
                                 RR_INTERSECT_QUERY_CLOSEST,
                                 rays_ptr,
                                 rayCount,
                                 nullptr,
                                 RR_INTERSECT_QUERY_OUTPUT_FULL_HIT,
                                 hits_ptr,
                                 scratch_ptr,
                                 frame.stream);
    
    if (res != RR_SUCCESS) {
        //std::cout << res << std::endl;
        throw std::runtime_error("error writing intersect commands");
    }
}

RRDevicePtr VulkanRayTracer::getDevicePtr(vk::Buffer buffer) {
    RRDevicePtr ptr;
    rrGetDevicePtrFromVkBuffer(context, buffer, 0, &ptr);
    return ptr;
}

VulkanRayFrameContext::VulkanRayFrameContext(VulkanRayTracer& tracer, VulkanDevice& device, vk::CommandBuffer cmd) : tracer(tracer), device(device), cmd(cmd) {
    auto res = rrGetCommandStreamFromVkCommandBuffer(tracer.context, cmd, &stream);
    if (res != RR_SUCCESS) {
        throw std::runtime_error("error create radeon rays command stream");
    }
}

VulkanRayFrameContext::~VulkanRayFrameContext() {
    for (auto ptr : ptrs) {
        rrReleaseDevicePtr(tracer.context, ptr);
    }
    rrReleaseExternalCommandStream(tracer.context, stream);
    rrFreeDescriptorSets(tracer.context);
}

RRDevicePtr VulkanRayFrameContext::getTempDevicePtr(vk::Buffer buffer) {
    RRDevicePtr ptr;
    rrGetDevicePtrFromVkBuffer(tracer.context, buffer, 0, &ptr);
    ptrs.push_back(ptr);
    return ptr;
}

VulkanBufferObject* VulkanRayFrameContext::acquireTemporaryBuffer(size_t size) {
    tempBuffers.emplace_back(device, size, BufferUsage::STORAGE);
    return &tempBuffers.back();
}

