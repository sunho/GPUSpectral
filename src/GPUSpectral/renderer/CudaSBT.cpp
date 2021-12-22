#include "CudaSBT.h"
#include "Renderer.h"
#include <optix_stubs.h>

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
            hitgroup_records[sbt_idx].data.bsdf = scene.materials[i].bsdf;
            hitgroup_records[sbt_idx].data.twofaced = scene.materials[i].twofaced;
            hitgroup_records[sbt_idx].data.facenormals = scene.materials[i].facenormals;
            hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(tlas.devicePositions);
            hitgroup_records[sbt_idx].data.normals= reinterpret_cast<float4*>(tlas.deviceNormals);
            hitgroup_records[sbt_idx].data.uvs = reinterpret_cast<float2*>(tlas.deviceUVs);
        }

        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
            memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
            hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(tlas.devicePositions);
            hitgroup_records[sbt_idx].data.emission_color = scene.materials[i].emission;
            hitgroup_records[sbt_idx].data.uvs = reinterpret_cast<float2*>(tlas.deviceUVs);
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

