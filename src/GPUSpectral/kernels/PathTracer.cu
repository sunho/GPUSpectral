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
    float        directWeight;
    
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
        occluded
        );
    return occluded;
}

extern "C" __global__ void __closesthit__occlusion() {
    setPayloadOcclusion(true);
}

extern "C" __global__ void __raygen__rg() {
    const int    w = params.width;
    const int    h = params.height;
    const float3 eye = params.camera.eye;
    const float3 U = params.camera.U;
    const float3 V = params.camera.V;
    const float3 W = params.camera.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframeIndex;
    float imageAspectRatio = 1.0f;
    SamplerState sampler(pcgHash(tea<4>(idx.y * w + idx.x, params.subframeIndex)));
    float3 result = make_float3(0.0f);
    int i = params.spp;
    do {
        const float2 subpixel_jitter = make_float2(randUniform(sampler), randUniform(sampler));

        const float2 fragcord = make_float2(static_cast<float>(idx.x) + subpixel_jitter.x, static_cast<float>(idx.y) + subpixel_jitter.y);
        const float3 rd = rayDir(make_float2(w, h), fragcord, params.camera.fov, imageAspectRatio);
        float3 ray_direction = normalize(rd.x * U + rd.y * V + rd.z * W);
        float3 ray_origin = eye;

        RadiancePRD prd;
        prd.weight = make_float3(1.f);
        prd.countEmitted = true;
        prd.wasDelta = false;
        prd.done = false;
        prd.sampler = sampler;
        prd.direction = ray_direction;

        int depth = 0;
        for (;; )
        {
            prd.emitted = make_float3(0.f);
            prd.radiance = make_float3(0.f);
            traceRadiance(
                params.scene.tlas,
                ray_origin,
                ray_direction,
                0.000f,
                1e16f,
                &prd);

            result += prd.emitted;
            result += prd.radiance;

            if (depth >= 17) {
                break;
            }

            if (prd.done || depth >= 17) // TODO RR, variable for depth
                break;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    } while (--i);

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;
    float3         accum_color = result / static_cast<float>(params.spp);

    if (subframe_index > 0)
    {
        const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accumBuffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    if (!isfinite(accum_color)) {
        params.accumBuffer[image_index] = make_float4(accum_color, 1.0f);
        params.frameBuffer[image_index] = make_color(make_float3(0.f, 0.0, 10.0));
        printf("nan detected in framebuffer\n");
    }
    else {
        params.accumBuffer[image_index] = make_float4(accum_color, 1.0f);
        params.frameBuffer[image_index] = make_color(accum_color);
    }
}

extern "C" __global__ void __miss__radiance() {
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3(0.0, 0.0, 0.0);
    prd->emitted = make_float3(0.0f);
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
    const float3 n0   = make_float3( rt_data->normals[ vert_idx_offset+0 ] );
    const float3 n1   = make_float3( rt_data->normals[ vert_idx_offset+1 ] );
    const float3 n2   = make_float3( rt_data->normals[ vert_idx_offset+2 ] );
    float2 bary = optixGetTriangleBarycentrics();
    float3 SN = normalize(bary.x * n1 + bary.y * n2 + (1.0f - bary.x - bary.y) * n0);
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );
    RadiancePRD* prd = getPRD();

    float3 N = N_0;
    if (dot(N, -ray_dir) < 0) {
        /*if (!isTransimissionBSDF(rt_data->bsdf.type())) {
            prd->done = true;
            return;
        }*/
        if (rt_data->twofaced) {
            N *= -1.0f;
            SN *= -1.0f;
        }
    }
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    SamplerState sampler = prd->sampler;

    Onb onb(SN);
    Onb geoOnb(N);
    float3 wo = normalize(onb.transform(-ray_dir));

    LightOutput lightRes;
    sampleLight(params.scene.lightData, sampler, &lightRes);

    float3 L = normalize(lightRes.position - P);
    float3 wL = onb.transform(L);
    float Ldist = length(lightRes.position - P);
    const float NoL = fabs(dot(SN, L));
    float lightPdf = Ldist * Ldist / fabs(dot(-L, lightRes.normal)) * lightRes.pdf;

    BSDFOutput bsdfRes;
    float3 wi;
    sampleBSDF(params.scene.bsdfData, sampler, rt_data->bsdf, wo, wi, bsdfRes);
    float NoW = abs(dot(wi, make_float3(0.0f, 0.0f, 1.0f)));
    onb.inverse_transform(wi);

    if (dot(wi, N) <= 0.0f && !isTransimissionBSDF(rt_data->bsdf.type())) {
        prd->done = true;
        return;
    }

    if (!isvalid(bsdfRes.pdf) || !isvalid(bsdfRes.bsdf) || bsdfRes.pdf == 0.0f) {
        prd->done = true;
        return;
    }

    float3 direct = make_float3(0.0f);
    if (!bsdfRes.isDelta) {
        BSDFOutput lightBsdfRes;
        evalBSDF(params.scene.bsdfData, rt_data->bsdf, wo, wL, lightBsdfRes);

        if ((dot(N, L) > 0.0f && dot(lightRes.normal, -L) > 0) || isTransimissionBSDF(rt_data->bsdf.type())) {
            bool occluded = traceOcclusion(
                params.scene.tlas,
                P,
                L,
                0.00001f,
                Ldist - 0.01f
            );
            if (!occluded && isvalid(lightPdf) && isvalid(lightBsdfRes.bsdf) && lightPdf != 0.0f) {
                float w = powerHeuristic(1, lightPdf, 1, bsdfRes.pdf);
                direct += w * NoL * lightBsdfRes.bsdf * prd->weight * lightRes.emission / lightPdf;
            }
        }
    }
    float lightFlag = dot(N, -ray_dir) > 0 ? 1.0f : 0.0f;
    if (prd->countEmitted) {
        prd->emitted = prd->weight * rt_data->emission_color * lightFlag;
    } else {
        if (prd->wasDelta) {
            prd->emitted = prd->weight * rt_data->emission_color * lightFlag;
        } else {
            direct += rt_data->emission_color * lightFlag * prd->directWeight * prd->weight;
        }
    }
    prd->radiance = direct;
    prd->directWeight = powerHeuristic(1, bsdfRes.pdf, 1, lightPdf);
    prd->countEmitted = false;
    prd->origin = P + 0.0001f*faceforward(N, wi, N);
    prd->direction = wi;
    prd->weight *= bsdfRes.bsdf * NoW / bsdfRes.pdf;
    prd->wasDelta = bsdfRes.isDelta;
    prd->sampler = sampler;
}
