#include "CudaPipeline.h"
#include "Renderer.h"

#include <optix.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

static OptixPipelineCompileOptions createPipelineCompileOption() {
    OptixPipelineCompileOptions options = {};
    options.usesMotionBlur = false;
    options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    options.numPayloadValues = 4;
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
