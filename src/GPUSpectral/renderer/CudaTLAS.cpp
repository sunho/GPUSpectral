#include "CudaTLAS.h"
#include "Renderer.h"

#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

CudaTLAS::CudaTLAS(Renderer& renderer, OptixDeviceContext context, const Scene& scene) {
    const size_t vertices_size_in_bytes = scene.sceneData.positions.size() * sizeof(float4);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePositions), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(devicePositions),
        scene.sceneData.positions.data(), vertices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceNormals), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceNormals),
        scene.sceneData.normals.data(), vertices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    const size_t uvVerticesSizeInBytes = scene.sceneData.uvs.size() * sizeof(float2);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceUVs), uvVerticesSizeInBytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(deviceUVs),
        scene.sceneData.uvs.data(), uvVerticesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr  d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = scene.sceneData.matIndices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        scene.sceneData.matIndices.data(),
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    //
    // Build triangle GAS
    //
    std::vector<uint32_t> triangle_input_flags(scene.materials.size(), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float4);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(scene.sceneData.positions.size());
    triangle_input.triangleArray.vertexBuffers = &devicePositions;
    triangle_input.triangleArray.flags = triangle_input_flags.data();
    triangle_input.triangleArray.numSbtRecords = scene.materials.size();
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &triangle_input,
        1,  // num_build_inputs
        &gas_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
    ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,                                  // CUDA stream
        &accel_options,
        &triangle_input,
        1,                                  // num build inputs
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gasHandle,
        &emitProperty,                      // emitted property list
        1                                   // num emitted properties
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gasOutputBuffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(context, 0, gasHandle, gasOutputBuffer, compacted_gas_size, &gasHandle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        gasOutputBuffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
