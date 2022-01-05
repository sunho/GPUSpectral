#include "VulkanRays.h"
#include "VulkanHandles.h"
#include <glm/gtc/type_ptr.hpp>
#include <Tracy.hpp>

VulkanBLAS::VulkanBLAS(VulkanDevice& device, vk::CommandBuffer cmd, VulkanPrimitive* primitive, VulkanBufferObject** scratch) {
	VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress = {};
	VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress = {};

	vertexBufferDeviceAddress.deviceAddress = device.getBufferDeviceAddress(primitive->vertex->buffers[0]->buffer);
	indexBufferDeviceAddress.deviceAddress = device.getBufferDeviceAddress(primitive->index->buffer->buffer);

	uint32_t numTriangles = static_cast<uint32_t>(primitive->vertex->vertexCount) / 3;
	uint32_t maxVertex = primitive->vertex->vertexCount;

	// Build
	VkAccelerationStructureGeometryKHR accelerationStructureGeometry = {};
	accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	accelerationStructureGeometry.geometry.triangles.vertexFormat = (VkFormat)translateElementFormat(primitive->vertex->attributes[0].type, false, false);
	accelerationStructureGeometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
	accelerationStructureGeometry.geometry.triangles.maxVertex = maxVertex;
	accelerationStructureGeometry.geometry.triangles.vertexStride = primitive->vertex->attributes[0].stride;
	accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
	accelerationStructureGeometry.geometry.triangles.indexData = indexBufferDeviceAddress;
	accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
	accelerationStructureGeometry.geometry.triangles.transformData.hostAddress = nullptr;

	// Get size info
	VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = {};
	accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationStructureBuildGeometryInfo.geometryCount = 1;
	accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

	VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = {};
	accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	device.dld.vkGetAccelerationStructureBuildSizesKHR(
		device.device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&accelerationStructureBuildGeometryInfo,
		&numTriangles,
		&accelerationStructureBuildSizesInfo);
	
	buffer = std::make_unique<VulkanBufferObject>(device, accelerationStructureBuildSizesInfo.accelerationStructureSize, BufferUsage::BDA | BufferUsage::ACCELERATION_STRUCTURE, BufferType::DEVICE);
	// Acceleration structure
	VkAccelerationStructureCreateInfoKHR accelerationStructureCreate_info{};
	accelerationStructureCreate_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	accelerationStructureCreate_info.buffer = buffer->buffer;
	accelerationStructureCreate_info.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
	accelerationStructureCreate_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	device.dld.vkCreateAccelerationStructureKHR(device.device, &accelerationStructureCreate_info, nullptr, &handle);
	// AS device address
	VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
	accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	accelerationDeviceAddressInfo.accelerationStructure = handle;
	deviceAddress = device.dld.vkGetAccelerationStructureDeviceAddressKHR(device.device, &accelerationDeviceAddressInfo);

	// Create a small scratch buffer used during build of the bottom level acceleration structure
	*scratch = new VulkanBufferObject(device, accelerationStructureBuildSizesInfo.buildScratchSize, BufferUsage::BDA | BufferUsage::STORAGE  | BufferUsage::ACCELERATION_STRUCTURE_INPUT , BufferType::DEVICE);

	VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = {};
	accelerationBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	accelerationBuildGeometryInfo.dstAccelerationStructure = handle;
	accelerationBuildGeometryInfo.geometryCount = 1;
	accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
	accelerationBuildGeometryInfo.scratchData.deviceAddress = device.getBufferDeviceAddress((*scratch)->buffer);

	VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
	accelerationStructureBuildRangeInfo.primitiveCount = numTriangles;
	accelerationStructureBuildRangeInfo.primitiveOffset = 0;
	accelerationStructureBuildRangeInfo.firstVertex = 0;
	accelerationStructureBuildRangeInfo.transformOffset = 0;
	std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };
	device.dld.vkCmdBuildAccelerationStructuresKHR(
		cmd,
		1,
		&accelerationBuildGeometryInfo,
		accelerationBuildStructureRangeInfos.data());
}

VulkanBLAS::~VulkanBLAS() {
}

VulkanTLAS::VulkanTLAS(VulkanDevice& device, vk::CommandBuffer cmd, const VulkanRTSceneDescriptor& scene, VulkanBufferObject** scratch) {
	std::vector< VkAccelerationStructureInstanceKHR> blasInstances;
	for (auto& instance : scene.instances) {
		VkTransformMatrixKHR transformMatrix = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f };

		VkAccelerationStructureInstanceKHR accInstance{};
		accInstance.transform = transformMatrix;
		accInstance.instanceCustomIndex = 0;
		accInstance.mask = 0xFF;
		accInstance.instanceShaderBindingTableRecordOffset = 0;
		accInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		blasInstances.push_back(accInstance);
	}

	instanceBuffer = std::make_unique<VulkanBufferObject>(device, sizeof(VkAccelerationStructureInstanceKHR) * scene.instances.size(), BufferUsage::BDA, BufferType::HOST_COHERENT);
	memcpy(instanceBuffer->mapped, blasInstances.data(), blasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR));

	VkDeviceOrHostAddressConstKHR instance_data_device_address{};
	instance_data_device_address.deviceAddress = device.getBufferDeviceAddress(instanceBuffer->buffer);

	VkAccelerationStructureGeometryKHR acceleration_structure_geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
	acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	acceleration_structure_geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	acceleration_structure_geometry.geometry.instances.arrayOfPointers = VK_FALSE;
	acceleration_structure_geometry.geometry.instances.data = instance_data_device_address;

	VkAccelerationStructureBuildGeometryInfoKHR acceleration_structure_build_geometry_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	acceleration_structure_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	acceleration_structure_build_geometry_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	acceleration_structure_build_geometry_info.geometryCount = 1;
	acceleration_structure_build_geometry_info.pGeometries = &acceleration_structure_geometry;

	const auto primitive_count = static_cast<uint32_t>(scene.instances.size());

	VkAccelerationStructureBuildSizesInfoKHR acceleration_structure_build_sizes_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	device.dld.vkGetAccelerationStructureBuildSizesKHR(
		device.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&acceleration_structure_build_geometry_info,
		&primitive_count,
		&acceleration_structure_build_sizes_info);

	buffer = std::make_unique<VulkanBufferObject>(device, acceleration_structure_build_sizes_info.accelerationStructureSize, BufferUsage::STORAGE, BufferType::DEVICE);

	// Create the acceleration structure
	VkAccelerationStructureCreateInfoKHR acceleration_structure_create_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
	acceleration_structure_create_info.buffer = buffer->buffer;
	acceleration_structure_create_info.size = acceleration_structure_build_sizes_info.accelerationStructureSize;
	acceleration_structure_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	device.dld.vkCreateAccelerationStructureKHR(device.device, &acceleration_structure_create_info, nullptr, &handle);

	// The actual build process starts here
	*scratch = new VulkanBufferObject(device, acceleration_structure_build_sizes_info.buildScratchSize, BufferUsage::STORAGE | BufferUsage::BDA, BufferType::CPU_TO_GPU);

	VkAccelerationStructureBuildGeometryInfoKHR acceleration_build_geometry_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	acceleration_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	acceleration_build_geometry_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	acceleration_build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	acceleration_build_geometry_info.dstAccelerationStructure = handle;
	acceleration_build_geometry_info.geometryCount = 1;
	acceleration_build_geometry_info.pGeometries = &acceleration_structure_geometry;
	acceleration_build_geometry_info.scratchData.deviceAddress = device.getBufferDeviceAddress((*scratch)->buffer);

	VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_range_info;
	acceleration_structure_build_range_info.primitiveCount = primitive_count;
	acceleration_structure_build_range_info.primitiveOffset = 0;
	acceleration_structure_build_range_info.firstVertex = 0;
	acceleration_structure_build_range_info.transformOffset = 0;
	std::vector<VkAccelerationStructureBuildRangeInfoKHR*> acceleration_build_structure_range_infos = { &acceleration_structure_build_range_info };

	device.dld.vkCmdBuildAccelerationStructuresKHR(
		cmd,
		1,
		&acceleration_build_geometry_info,
		acceleration_build_structure_range_infos.data());

}

VulkanTLAS::~VulkanTLAS() {
}
