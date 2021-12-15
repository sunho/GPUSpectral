#include "Renderer.h"
#include "../utils/CudaUtil.h"

#include <filesystem>
#include <fstream>
#include <stb_image_write.h>

#include <optix.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

Renderer::Renderer(const std::string& basePath) : 
    basePath(basePath), 
    baseFsPath(basePath),
    context(createDeviceContext()), 
    pipeline(*this, context) {
}

static void contextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

OptixDeviceContext Renderer::createDeviceContext() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCallback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    return context;
}

Renderer::~Renderer() {

}

std::string Renderer::loadKernel(const std::string& fileName) {
    std::ifstream is(baseFsPath / "kernels" / fileName, std::ios::binary);
    std::string includedSource((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    return includedSource;
}

Mesh* Renderer::getMesh(int id) {
    return &meshes[id];
       
}

int Renderer::addMesh(const Mesh& mesh) {
    int id = meshes.size();
    meshes.push_back(mesh);
    return id;
}

void Renderer::setScene(const Scene& scene) {
    state = std::make_unique<RenderState>(*this, context, scene);
}

void Renderer::render() {
    state->params.width = 768;
    state->params.height = 768;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state->params.accum_buffer),
        state->params.width * state->params.height * sizeof(float4)
    ));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->params.frame_buffer), state->params.width * state->params.height * 4));
    state->params.subframe_index++;
    // Launch
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(state->dParams),
        &state->params, sizeof(Params),
        cudaMemcpyHostToDevice, state->stream
    ));

    OPTIX_CHECK(optixLaunch(
        pipeline.pipeline,
        state->stream,
        reinterpret_cast<CUdeviceptr>(state->dParams),
        sizeof(Params),
        &state->sbt.sbt,
        state->params.width,   // launch width
        state->params.height,  // launch height
        1                     // launch depth
    ));
    cudaDeviceSynchronize();

    std::vector<uint32_t> pixels(state->params.width * state->params.height);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(pixels.data()),
        reinterpret_cast<void*>(state->params.frame_buffer),
        state->params.width * state->params.height * 4,
        cudaMemcpyDeviceToHost
    ));
    stbi_write_jpg("jpg_test_.jpg", state->params.width, state->params.height, 4, pixels.data(), state->params.width * 4);
}

static OptixPipelineCompileOptions createPipelineCompileOption() {
    OptixPipelineCompileOptions options = {};
    options.usesMotionBlur = false;
    options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    options.numPayloadValues = 2;
    options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    options.pipelineLaunchParamsVariableName = "params";
    return options;
}

CudaPipeline::CudaPipeline(Renderer& renderer, OptixDeviceContext context) {
    initModule(renderer, context);
    initProgramGroups(renderer, context);
    initPipeline(renderer, context);
}

void CudaPipeline::initModule(Renderer& renderer, OptixDeviceContext context) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    auto pipelineCompileOptions = createPipelineCompileOption();
    
    std::string inputStr = renderer.loadKernel("PathTracer.ptx");
    size_t      inputSize = inputStr.size();

    const char* input = inputStr.data();
    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        input,
        inputSize,
        log,
        &sizeof_log,
        &ptxModule
    ));
}

void CudaPipeline::initProgramGroups(Renderer& renderer, OptixDeviceContext context) {
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptxModule;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygenProgGroup
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptxModule;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log, &sizeof_log,
            &radianceMissGroup
        ));

        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = nullptr;  // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &shadowMissGroup
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = ptxModule;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &radianceHitGroup
        ));

        memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = ptxModule;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &shadowHitGroup
        ));
    }
}

void CudaPipeline::initPipeline(Renderer& renderer, OptixDeviceContext context) {
    OptixProgramGroup program_groups[] =
    {
        raygenProgGroup,
        radianceMissGroup,
        radianceHitGroup,
        shadowHitGroup,
        shadowMissGroup
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    auto pipelineCompileOptions = createPipelineCompileOption();

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context,
        &pipelineCompileOptions,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(raygenProgGroup, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(radianceMissGroup, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(radianceHitGroup, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(shadowHitGroup, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(shadowMissGroup, &stack_sizes));

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}

CudaTLAS::CudaTLAS(Renderer& renderer, OptixDeviceContext context, const Scene& scene) {
    fillData(renderer, scene);
    const size_t vertices_size_in_bytes = positions.size() * sizeof(float4);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePositions), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(devicePositions),
        positions.data(), vertices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr  d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = matIndices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        matIndices.data(),
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    const size_t trinagleLight= triangleLights.size() * sizeof(TriangleLight);
    lightData.triangleLights = Array<TriangleLight>(triangleLights.size());
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(lightData.triangleLights.data()),
        triangleLights.data(),
        triangleLights.size() * sizeof(TriangleLight),
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
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(positions.size());
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

void CudaTLAS::fillData(Renderer& renderer, const Scene& scene) {
    for (auto& obj : scene.renderObjects) {
        auto mesh = renderer.getMesh(obj.meshId);
        for (auto& pos : mesh->positions) {
            positions.push_back(obj.transform * float4(pos.x, pos.y, pos.z, 1.0f));
        }

        auto invT = obj.transform.transpose().inverse();
        for (auto& nor : mesh->normals) {
            normals.push_back(invT * float4(nor.x, nor.y, nor.z, 0.0f));
        }
        
        for (auto& uv : mesh->uvs) {
            uvs.push_back(float4(uv.x, uv.y, 0.0f, 0.0f));
        }

        for (size_t i = 0; i < mesh->positions.size() / 3; ++i) {
            matIndices.push_back(obj.materialId);
        }
        auto material = scene.materials[obj.materialId];
        if (material.emission.x != 0.0f || material.emission.y != 0.0f || material.emission.z != 0.0f) {
            for (size_t i = 0; i < mesh->positions.size(); i+=3) {
                TriangleLight light = {};
                float3 pos0 = mesh->positions[i];
                float3 pos1 = mesh->positions[i+1];
                float3 pos2 = mesh->positions[i+2];
                light.positions[0] = make_float3(obj.transform * float4(pos0.x, pos0.y, pos0.z, 1.0f));
                light.positions[1] = make_float3(obj.transform * float4(pos1.x, pos1.y, pos1.z, 1.0f));
                light.positions[2] = make_float3(obj.transform * float4(pos2.x, pos2.y, pos2.z, 1.0f));
                light.radinace = material.emission;
                triangleLights.push_back(light);
            }
        }
    }
}

CudaSBT::CudaSBT(Renderer& renderer, OptixDeviceContext context, CudaTLAS& tlas, const Scene& scene) {
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(Record<RayGenData>);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    Record<RayGenData> rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(renderer.pipeline.raygenProgGroup, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));


    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof(Record<MissData>);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    Record<MissData> ms_sbt[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(renderer.pipeline.radianceMissGroup, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(renderer.pipeline.shadowMissGroup, &ms_sbt[1]));
    ms_sbt[1].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(Record<HitGroupData>);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * scene.materials.size()
    ));

    std::vector<Record<HitGroupData>> hitgroup_records(RAY_TYPE_COUNT * scene.materials.size());
    for (int i = 0; i < scene.materials.size(); ++i)
    {
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

            OPTIX_CHECK(optixSbtRecordPackHeader(renderer.pipeline.radianceHitGroup, &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.emission_color = scene.materials[i].emission;
            hitgroup_records[sbt_idx].data.diffuse_color = scene.materials[i].color;
            hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(tlas.devicePositions);
        }

        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
            memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);

            OPTIX_CHECK(optixSbtRecordPackHeader(renderer.pipeline.shadowHitGroup, &hitgroup_records[sbt_idx]));
        }
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records.data(),
        hitgroup_record_size * RAY_TYPE_COUNT * scene.materials.size(),
        cudaMemcpyHostToDevice
    ));

    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_records;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    sbt.missRecordCount = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    sbt.hitgroupRecordCount = RAY_TYPE_COUNT * scene.materials.size();
}


RenderState::RenderState(Renderer& renderer, OptixDeviceContext context, const Scene& scene) :
    scene(scene), tlas(renderer, context, scene), sbt(renderer, context, tlas, scene) {

    params.samples_per_launch = 10240;
    params.subframe_index = 0u;

    params.handle = tlas.gasHandle;
    params.eye = scene.camera.eye;
    params.U = scene.camera.u;
    params.V = scene.camera.v;
    params.W = scene.camera.w;

    params.lightData = tlas.lightData;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParams), sizeof(Params)));
}

