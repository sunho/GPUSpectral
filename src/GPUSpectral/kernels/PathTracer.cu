#include <optix.h>
#include <vector_types.h>
#include <vector_functions.hpp>
#include "PathTracer.cuh"
#include "VectorMath.cuh"
#include <math.h>

extern "C" {
    __constant__ Params params;
}

struct RadiancePRD {
    float3       emitted;
    float3       radiance;
    float3       weight;
    float3       origin;
    float3       direction;
    
    SamplerState sampler;
    int          wasDelta;
    int          countEmitted;
    int          done;
    int          pad;
};

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ RadiancePRD* getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}

static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        ) {
    // TODO: deduce stride from num ray-types passed in params
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}

static __forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    unsigned int occluded = 0u;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RAY_TYPE_OCCLUSION,      // SBT offset
        RAY_TYPE_COUNT,          // SBT stride
        RAY_TYPE_OCCLUSION,      // missSBTIndex
        occluded);
    return occluded;
}

extern "C" __global__ void __closesthit__occlusion() {
    setPayloadOcclusion(true);
}


extern "C" __global__ void __raygen__rg() {
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;
    SamplerState sampler(pcgHash(idx.y * w + idx.x));
    float3 result = make_float3(0.0f);
    int i = params.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2( randUniform(sampler), randUniform(sampler) );

        const float2 fragcord = make_float2(static_cast<float>(idx.x) + subpixel_jitter.x, static_cast<float>(idx.y) + subpixel_jitter.y);
        const float3 rd = rayDir(make_float2(w, h), fragcord, params.fov);
        float3 ray_direction = normalize(rd.x*U + rd.y*V + rd.z*W);
        float3 ray_origin    = eye;

        RadiancePRD prd;
        prd.weight = make_float3(1.f);
        prd.countEmitted = true;
        prd.wasDelta = false;
        prd.done         = false;
        prd.sampler = sampler;
        prd.direction = ray_direction;

        int depth = 0;
        for( ;; )
        {
            prd.emitted      = make_float3(0.f);
            prd.radiance     = make_float3(0.f);
            traceRadiance(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    &prd );

            result += prd.emitted;
            result += prd.radiance;

            if( prd.done  || depth >= 8 ) // TODO RR, variable for depth
                break;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    }
    while( --i );

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index  = launch_index.y * params.width + launch_index.x;
    float3         accum_color  = result / static_cast<float>( params.samples_per_launch );

    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( ACESFilm(accum_color) );
}

extern "C" __global__ void __miss__radiance() {
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3( rt_data->bg_color * 3.0 ) * prd->weight;
    prd->done      = true;
}

extern "C" __global__ void __closesthit__radiance() {
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = rt_data->twofaced ? faceforward( N_0, -ray_dir, N_0 ) : N_0;
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();

    SamplerState sampler = prd->sampler;

    Onb onb(N);
    float3 wo = normalize(onb.transform(-ray_dir));

    float3 lightPos;
    float lightPdf;
    float3 lightEmission;
    sampleLight(params.lightData, sampler, lightPos, lightPdf, lightEmission);

    float3 L = normalize(lightPos - P);
    float3 wL = onb.transform(L);
    float Ldist = length(lightPos - P);
    const float NoL   = dot( N, L );

    if (NoL > 0.0f) {
        BSDFOutput lightBsdfRes;
        evalBSDF(params.bsdfData, rt_data->bsdf, wo, wL, lightBsdfRes);
        const bool occluded = traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,         // tmin
            Ldist - 0.01f  // tmax
        );

        if (!occluded) {
            prd->radiance = NoL * lightBsdfRes.bsdf * prd->weight * lightEmission / lightPdf;
        }
    }

    BSDFOutput bsdfRes;
    float3 wi;
    sampleBSDF(params.bsdfData, sampler, rt_data->bsdf, wo, wi, bsdfRes);
    float NoW = abs(dot(wi, make_float3(0.0f, 0.0f, 1.0f)));
    onb.inverse_transform(wi);

    if( prd->countEmitted)
        prd->emitted = prd->weight * rt_data->emission_color;

    if (prd->wasDelta)
        prd->emitted = prd->weight * rt_data->emission_color;

    prd->countEmitted = false;
    prd->origin    = P;
    prd->direction = wi;
    prd->weight *= bsdfRes.bsdf * NoW / bsdfRes.pdf;
    prd->wasDelta = bsdfRes.isDelta;
    prd->sampler = sampler;
}
