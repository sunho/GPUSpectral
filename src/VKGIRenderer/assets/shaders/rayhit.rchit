#version 460
#pragma shader_stage(closest)

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require

#include "pt_common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload prd;
layout(location = 2) rayPayloadEXT bool shadowed;
hitAttributeEXT vec3 attribs;

struct DiffuseBSDF {
    vec3 diffuse;
    uint hasTexture;
};

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer DiffuseBSDFBuffer
{
   	DiffuseBSDF values[];
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, std430, set = 0) uniform _RenderState {
	RenderState renderState;
};
 
vec2 sampleConcentric(inout uint seed) {
    vec2 s = vec2(randUniform(seed), randUniform(seed));
    vec2 u = 2.0f * s - 1.0f;
    if (u.x == 0.0f && u.y == 0.0f) {
        return vec2(0.0f, 0.0f);
    }
    float r, th;
    if (abs(u.x) > abs(u.y)) {
        r = u.x;
        th = (M_PI / 4.0f) * (u.y / u.x);
    }
    else {
        r = u.y;
        th = M_PI / 2.0f - (M_PI / 4.0f) * (u.x / u.y);
    }
    return vec2(r * cos(th), r * sin(th));
}

vec3 randCosineHemisphere(inout uint seed) {
    vec2 u = sampleConcentric(seed);
    float z = sqrt(max(0.0f, 1.0f - u.x * u.x - u.y * u.y));
    return vec3(u.x, u.y, z);
}

float cosineHemispherePdf(vec3 wo) {
    return max(abs(wo.z) / M_PI, 0.000001f);
}

void main()
{
	ivec3 index = ivec3(3 * gl_PrimitiveID, 3 * gl_PrimitiveID + 1, 3 * gl_PrimitiveID + 2);
	InstanceBuffer instanceBuffer = InstanceBuffer(renderState.scene.instances);
	Instance instance = instanceBuffer.values[gl_InstanceID];
	PositionBuffer posBuffer = PositionBuffer(instance.positionBuffer);
	NormalBuffer normalBuffer = NormalBuffer(instance.normalBuffer);

	vec3 pos0 = posBuffer.positions[index.x];
	vec3 pos1 = posBuffer.positions[index.y];
	vec3 pos2 = posBuffer.positions[index.z];
	pos0 = (gl_ObjectToWorldEXT * vec4(pos0, 1.0)).xyz;
	pos1 = (gl_ObjectToWorldEXT * vec4(pos1, 1.0)).xyz;
	pos2 = (gl_ObjectToWorldEXT * vec4(pos2, 1.0)).xyz;

	vec3 normal0 = normalBuffer.normals[index.x];
	vec3 normal1 = normalBuffer.normals[index.y];
	vec3 normal2 = normalBuffer.normals[index.z];
	normal0 = (instance.transformInvT * vec4(normal0, 0.0)).xyz;
	normal1 = (instance.transformInvT * vec4(normal1, 0.0)).xyz;
	normal2 = (instance.transformInvT * vec4(normal2, 0.0)).xyz;

	const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	// vec3 position = barycentrics.x * pos0 + barycentrics.y * pos1 + barycentrics.z * pos2;
    vec3 position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 normal = normalize(barycentrics.x * normal0 + barycentrics.y * normal1 + barycentrics.z * normal2);
	
    normal = normalize(cross(pos1 - pos0, pos2 - pos0));
    vec3 rayDir = prd.direction;
//   vec3 N = normal;
    if (dot(normal, -rayDir) < 0) {
        // if (rt_data->twofaced) {
            normal *= -1.0f;
            // SN *= -1.0f;
        // }
    }

    // if (rt_data->facenormals) {
    //     SN = N;
    // }
	Onb onb = onbCreate(normal);
    vec3 wo = normalize(onbTransform(onb, -rayDir));
	uint bsdfI = bsdfIndex(instance.bsdf);
    vec3 diffuse = DiffuseBSDFBuffer(renderState.scene.diffuseBSDFs).values[bsdfI].diffuse;

    vec3 wi = randCosineHemisphere(prd.seed);
    float pdf = cosineHemispherePdf(wi);
    float NoW = abs(dot(wi, vec3(0.0f, 0.0f, 1.0f)));
    prd.emitted = prd.weight * instance.emission;
    wi = onbUntransform(onb, wi);
    prd.origin = position;
    prd.direction = wi;
    prd.weight *= diffuse / M_PI * NoW / pdf;
    //prd->wasDelta = bsdfRes.isDelta;
    // LightOutput lightRes;
    // sampleLight(params.scene.lightData, sampler, P, &lightRes);

    // float3 L = normalize(lightRes.position - P);
    // float3 wL = onb.transform(L);
    // float Ldist = length(lightRes.position - P);
    // const float NoL = fabs(dot(SN, L));
    // float lightPdf = lightRes.pdf;

    // BSDFOutput bsdfRes;
    // float3 wi;
    // sampleBSDF(params.scene.bsdfData, sampler, uv, rt_data->bsdf, wo, wi, bsdfRes);
    // float NoW = abs(dot(wi, make_float3(0.0f, 0.0f, 1.0f)));
    // onb.inverse_transform(wi);

    // BSDFOutput lightBsdfRes;
    // evalBSDF(params.scene.bsdfData,rt_data->bsdf, uv, wo, wL, lightBsdfRes);

    // float3 direct = make_float3(0.0f);
    // bool neeDone = false;
    // if (params.nee) {
    //     if (!bsdfRes.isDelta) {
    //         if ((dot(N, -ray_dir) > 0 && dot(N, L) > 0) || isTransimissionBSDF(rt_data->bsdf.type())) {
    //             bool occluded = traceOcclusion(
    //                 params.scene.tlas,
    //                 P,
    //                 L,
    //                 0.001f,
    //                 Ldist - 0.01f
    //             );
    //             if (!occluded && isvalid(lightPdf) && isvalid(lightBsdfRes.bsdf) && lightPdf != 0.0f) {
    //                 float w = powerHeuristic(1, lightPdf, 1, bsdfRes.pdf);
    //                 direct += w * NoL * lightBsdfRes.bsdf * prd->weight * lightRes.emission / lightPdf;
    //                 neeDone = true;
    //                 /*if (direct.x > 100.0) {
    //                     printf("bsdf: %d pos: %f %f %f\n", rt_data->bsdf.type(), P.x, P.y, P.z);
    //                     printf("light pdf: %f %f bsdf pdf: %f \n", lightPdf, lightRes.pdf, bsdfRes.pdf);
    //                     printf("bsdf: %f %f %f\n", lightBsdfRes.bsdf.x, lightBsdfRes.bsdf.y, lightBsdfRes.bsdf.z);
    //                     printf("NEE: %f %f %f\n", direct.x, direct.y, direct.z);
    //                     printf("first hit: %d \n", prd->countEmitted);
    //                 }*/
    //             }
    //         }
    //     }
    // }

    /*if (prd->printDebug) {
        printf("bsdf: %f %f %f\n", bsdfRes.bsdf.x, bsdfRes.bsdf.y, bsdfRes.bsdf.z);
        printf("light bsdf: %f %f %f\n", lightBsdfRes.bsdf.x, lightBsdfRes.bsdf.y, lightBsdfRes.bsdf.z);
        printf("first hit: %d \n", prd->countEmitted);
    }*/

    // float lightFlag = dot(N, -ray_dir) > 0 ? 1.0f : 0.0f;
    // // This is adding term for the last path
    // // we should account for it before termninating by invalid path tests
    // if (params.nee && !prd->countEmitted && !prd->wasDelta) {
    //     float3 est = prd->directWeight * rt_data->emission_color * lightFlag * prd->weight;
    //     direct += est;
    //     /*if (prd->printDebug && est.x > 0.5) {
    //         printf("bsdf: %d pos: %f %f %f\n", rt_data->bsdf.type(), P.x, P.y, P.z);
    //         printf("light pdf: %f bsdf pdf: %f \n", lightPdf, bsdfRes.pdf);
    //         printf("BSDF estimate: %f %f %f\n", est.x, est.y, est.z);
    //         printf("first hit: %d \n", prd->countEmitted);
    //     }*/
    // }
    // prd->radiance = direct;
    // if (!params.nee || prd->countEmitted || prd->wasDelta) {
    //     prd->emitted = prd->weight * rt_data->emission_color * lightFlag;
    // }
    // // sampled invalid hemisphere becase we used shading normal
    // if (dot(wi, N) <= 0.0f && !isTransimissionBSDF(rt_data->bsdf.type())) {
    //     prd->done = true;
    //     return;
    // }
    // // light leak from self intersection
    // // TODO this is not working for two faced mesh
    // if (dot(N, -ray_dir) <= 0.0f && !isTransimissionBSDF(rt_data->bsdf.type())) {
    //     prd->done = true;
    //     return;
    // }
    // // singular bsdf or pdf
    // if (!isvalid(bsdfRes.pdf) || !isvalid(bsdfRes.bsdf) || bsdfRes.pdf == 0.0f) {
    //     prd->done = true;
    //     return;
    // }
    // if (neeDone) {
    //     prd->directWeight = powerHeuristic(1, bsdfRes.pdf, 1, lightPdf);
    // }
    // else {
    //     prd->directWeight = 1.0f;
    // }
    //prd->directWeight = 0.5f;


}