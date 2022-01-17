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

struct TriangleLight {
    vec4 positions[3];
    vec3 emission;
};

struct DiffuseBSDF {
    vec3 reflectance;
    uint hasTexture;
};

struct SmoothDielectricBSDF {
    float iorIn;
    float iorOut; // usually 1.0
};

struct SmoothConductorBSDF {
    float iorIn;
    float iorOut; // usually 1.0
};

struct SmoothFloorBSDF {
    vec3 diffuse;
    float R0;
};

struct SmoothPlasticBSDF {
    vec3 diffuse;
    float iorIn;
    float iorOut; // usually 1.0
    float R0;
};

struct RoughConductorBSDF {
    vec3 eta;
    vec3 k;
    vec3 reflectance;
    float alpha;
    int hasTexture;
};

struct RoughPlasticBSDF {
    vec3 diffuse;
    float iorIn;
    float iorOut; // usually 1.0
    float R0;
    float alpha;
    int hasTexture;
};

struct RoughFloorBSDF {
    vec3 diffuse;
    float R0;
    float alpha;
};

#define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer BSDFNAME##Buffer \
{ \
   	BSDFNAME values[]; \
};
#include "BSDF.inc"
#undef BSDFDefinition

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer TriangleLightBuffer
{
   	TriangleLight values[];
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, std430, set = 0) uniform _RenderState {
	RenderState renderState;
};
 
vec2 sampleConcentric() {
    vec2 s = vec2(randUniform(), randUniform());
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

vec3 randCosineHemisphere() {
    vec2 u = sampleConcentric();
    float z = sqrt(max(0.0f, 1.0f - u.x * u.x - u.y * u.y));
    return vec3(u.x, u.y, z);
}

float cosineHemispherePdf(vec3 wo) {
    return max(abs(wo.z) / M_PI, 0.000001f);
}

struct LightOutput {
    vec3 position;
    vec3 emission;
    float pdf;
};

LightOutput sampleTrangleLight(TriangleLight light, vec3 pos)  {
    // barycentric coordinates
    // u = 1 - sqrt(e_1)
    // v = e_2 * sqrt(e_1)
    float e1 = randUniform();
    float e2 = randUniform();
    float u = 1.0f - sqrt(e1);
    float v = e2 * sqrt(e1);
    float w = 1.0f - u - v;
    vec3 v0 = light.positions[0].xyz;
    vec3 v1 = light.positions[1].xyz;
    vec3 v2 = light.positions[2].xyz;
    float A = 0.5f * abs(length(cross(v2 - v0, v1 - v0)));
    vec3 normal = normalize( cross( v1-v0, v2-v0 ) );
    vec3 lightPos = u * v0 + v * v1 + w * v2;
    float ldist = length(lightPos - pos);
    vec3 l = normalize(lightPos - pos);
    LightOutput res;
    res.position = lightPos;
    res.emission = light.emission * float(dot(-l, normal) > 0);
    res.pdf = ldist * ldist / (abs(dot(-l, normal)) * A);
    return res;
}

LightOutput sampleLight(vec3 pos) {
    uint lightIdx = randPcg() % renderState.scene.numLights;
    TriangleLight light = TriangleLightBuffer(renderState.scene.triangleLights).values[lightIdx];
    LightOutput res = sampleTrangleLight(light, pos);
    res.pdf *= 1.0f / float(renderState.scene.numLights);
    return res;
}

vec3 sampleHalf(float alpha) {
    vec2 u = vec2(randUniform(), randUniform());
    float phi = 2.0f * M_PI * u.x;
    // 1 + tan^2 = sec^2
    // 1 / (1+tan^2) = cos^2
    float logSample = log(1.0f - u.y);
    if (isinf(logSample)) logSample = 0.0f;
    float tan2 = -alpha * alpha * logSample;
    float cost = 1.0f / sqrt(1.0f + tan2); // denominator is never 0.0
    float sint = sqrt(max(0.0f, 1.0f - cost* cost));
    return vec3(cos(phi) * sint, sin(phi) * sint, cost);
}

float sphericalPhi(vec3 wi) {
    float p = atan(wi.y, wi.x);
    return p < 0.0f ? (p + 2 * M_PI) : p;
}

float sphericalTheta(vec3 wi) {
    return acos(clamp(wi.z,-1.0f, 1.0f));
}

float beckmannD(vec3 wh, float alpha) {
    float cos2 = wh.z * wh.z;
    float tan2 = (wh.x * wh.x + wh.y * wh.y) / cos2;
    float a = exp(-tan2 / (alpha * alpha));
    float b = M_PI * alpha * alpha * cos2 * cos2;
    return a / b;
}

float ggxD(vec3 wh, float alpha) {
    float cos2 = wh.z * wh.z;
    float tan2 = (wh.x * wh.x + wh.y * wh.y) / cos2;
    if (isinf(tan2)) { return 0.0f; };
    float b = (1.0f + tan2 / (alpha * alpha));
    float a = M_PI * alpha * alpha * cos2 * cos2 * b * b;
    return 1.0f / a;
}

float ggxLambda(vec3 wh, float alpha) {
    float cos2 = wh.z * wh.z;
    float tan2 = (wh.x * wh.x + wh.y * wh.y) / cos2;
    if (isinf(tan2)) { return 0.0f; };
    float a = -1.0f + sqrt(1.0f + alpha*alpha*tan2);
    return 0.5f * a;
}

float ggxMask(vec3 wo, vec3 wi, float alpha) {
    return 1.0f / (1.0f + ggxLambda(wo, alpha) + ggxLambda(wi, alpha));
}

float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

struct BSDFOutput {
    vec3 bsdf;
    float pdf;
    bool isDelta;
};

float fresnel(float no, float cosTho, float nt, float cosTht) {
    float a = nt * cosTho - no * cosTht;
    float ad = nt * cosTho + no * cosTht;
    float b = no * cosTho - nt * cosTht;
    float bd = no * cosTho + nt * cosTht;
    float A = a*a / (ad*ad);
    float B = b*b / (bd*bd);
    return 0.5f * (A + B);
}

float fresnel(vec3 wo, float no, float nt) {
    float sinTho = sqrt(max(wo.x * wo.x + wo.y * wo.y, 0.0f)); // sqrt((cos(phi)sin(theta))^2 + ((sin(phi)sin(theta))^2)
    float sqrtTerm = 1.0f - ((no * no) / (nt * nt)) * (sinTho * sinTho);
    if (sqrtTerm <= 0.0f) {
        return 1.0f;
    }
    float cosTht = sqrt(sqrtTerm);
    float cosTho = abs(wo.z);
    return fresnel(no, cosTho, nt, cosTht);
}

float fresnel(float cosTho, float no, float nt) {
    float sinTho = sqrt(max(1.0f - cosTho*cosTho,0.0f));
    float sqrtTerm = 1.0f - ((no * no) / (nt * nt)) * (sinTho * sinTho);
    if (sqrtTerm <= 0.0f) {
        return 1.0f;
    }
    float cosTht = sqrt(sqrtTerm);
    return fresnel(no, cosTho, nt, cosTht);
}

vec3 fresnelConductor(vec3 wo, vec3 eta, vec3 k) {
    float cosTho = abs(wo.z);
    float cos2 = cosTho * cosTho;
    float sin2 = 1.0f - cos2;
    float tan2 = sin2 / cos2;
    vec3 n2 = eta * eta;
    vec3 k2 = k * k;
    vec3 c = n2 - k2 - sin2;
    vec3 a2b2 = sqrt(c * c + 4.0f * n2 * k2);
    vec3 a2 = 0.5f * (a2b2 + n2 - k2 - sin2);
    vec3 a = sqrt(a2);
    vec3 t = a2b2 - 2 * a * cosTho + cos2;
    vec3 tt = a2b2 + 2 * a * cosTho + cos2;
    vec3 Rs = t / tt;
    vec3 d = cos2 * a2b2 - 2 * a * cosTho * sin2 + sin2 * sin2;
    vec3 dd = cos2 * a2b2 + 2 * a * cosTho * sin2 + sin2 * sin2;
    vec3 Rp = Rs * (d / dd);
    return 0.5f * (Rp + Rs);
}

vec3 FresnelDieletricConductor(vec3 Eta, vec3 Etak, float CosTheta)
{
    float CosTheta2 = CosTheta * CosTheta;
    float SinTheta2 = 1 - CosTheta2;
    vec3 Eta2 = Eta * Eta;
    vec3 Etak2 = Etak * Etak;

    vec3 t0 = Eta2 - Etak2 - SinTheta2;
    vec3 a2plusb2 = sqrt(t0 * t0 + 4 * Eta2 * Etak2);
    vec3 t1 = a2plusb2 + CosTheta2;
    vec3 a = sqrt(0.5f * (a2plusb2 + t0));
    vec3 t2 = 2 * a * CosTheta;
    vec3 Rs = (t1 - t2) / (t1 + t2);

    vec3 t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
    vec3 t4 = t2 * SinTheta2;
    vec3 Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (Rp + Rs);
}

bool refractRay(vec3 wo, vec3 n, float no, float nt, out vec3 wt) {
    float sinTho = sqrt(max(wo.x * wo.x + wo.y * wo.y,0.0f)); // sqrt((cos(phi)sin(theta))^2 + ((sin(phi)sin(theta))^2)
    float sqrtTerm = 1.0f - ((no * no) / (nt * nt)) * (sinTho * sinTho);
    if (sqrtTerm <= 0.0f) {
        return false;
    }
    float cosTht = sqrt(sqrtTerm);
    wt = (no / nt) * (-wo) + ((no / nt) * dot(wo, n) - cosTht) * n;
    return true;
}

float coupledDiffuseTerm(float R0, float cosTho, float cosThi) {
    float k = 21.0f / (20.0f * M_PI * (1.0f - R0));
    float a = 1.0f - cosTho;
    float b = 1.0f - cosThi;
    float a5 = a * a * a * a * a;
    float b5 = b * b * b * b * b;
    return k * (1.0f - a5)* (1.0f - b5);
}

float fresnelBlendDiffuseTerm(float R0, float cosTho, float cosThi) {
    float k = 28.0f / (23.0f * M_PI);
    float a = 1.0f - 0.5f*cosTho;
    float b = 1.0f - 0.5f*cosThi;
    float a5 = a * a * a * a * a;
    float b5 = b * b * b * b * b;
    return k * (1.0f - R0) * (1.0f - a5)* (1.0f - b5);
}

// R_i
float internalScatterEscapeFraction(float R0, float no, float nt) {
    float Re = (M_PI * 20.0f * R0 + 1.0f) / 21.0f;
    float eta = no / nt;
    return 1.0f - eta * eta * (1.0f - Re);
}

float schlickFresnel(float R0, float cosTho) {
    float a = 1.0f - cosTho;
    float a5 = a * a * a * a * a;
    return R0 + a5 * (1.0f - R0);
}

#define BSDF_DIFFUSE 0
#define BSDF_SMOOTH_DIELECTRIC 1
#define BSDF_SMOOTH_CONDUCTOR 2
#define BSDF_SMOOTH_PLASTIC 3
#define BSDF_ROUGH_CONDUCTOR 4
#define BSDF_SMOOTH_FLOOR 5
#define BSDF_ROUGH_FLOOR 6
#define BSDF_ROUGH_PLASTIC 7

void diffuseBSDFSample(DiffuseBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    vec3 kD = bsdf.reflectance;
    wi = randCosineHemisphere();
    res.bsdf = kD / M_PI;
    res.pdf = cosineHemispherePdf(wi);
    res.isDelta = false;
    // NUMBERCHECK(res.bsdf)
    // NUMBERCHECK(res.pdf)
}

void diffuseBSDFEval(DiffuseBSDF bsdf, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    vec3 kD = bsdf.reflectance;
    res.bsdf = kD / M_PI;
    res.pdf = cosineHemispherePdf(wi);
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}

// bsdf for reflection: F_r/|cos(th_o)|
// bsdf for refraction: (n_t^2/n_o^2)(1-F_r)/|cos(th_o)|
void smoothDielectricBSDFSample(SmoothDielectricBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    bool entering = wo.z > 0.0f;
    float no = entering ? bsdf.iorOut : bsdf.iorIn;
    float nt = entering ? bsdf.iorIn : bsdf.iorOut;
    float cosTho = wo.z;
    vec3 wt;
    if (!refractRay(wo, faceforward(vec3(0.0f, 0.0f, 1.0f), -wo, vec3(0.0f, 0.0f, 1.0f)), no, nt, wt)) {
        // total internal reflection
        wi.z = wo.z;
        wi.x = -wo.x;
        wi.y = -wo.y;
        res.bsdf = 1.0f * vec3(1.0 / abs(cosTho));
        res.isDelta = true;
        res.pdf = 1.0f;
        return;
    }
    float Fr = fresnel(no, abs(cosTho), nt, abs(wt.z));
    float u = randUniform();
    // reflection with prob Fr
    // refraction with prob 1 - Fr
    if (u < Fr) {
        // reflect
        wi.z = wo.z;
        wi.x = -wo.x;
        wi.y = -wo.y;
        res.bsdf = Fr * vec3(1 / abs(cosTho));
        res.isDelta = true;
        res.pdf = Fr;
    } else {
        // refract
        wi = wt;

        res.bsdf = vec3(((no * no) / (nt * nt)) * (1.0f - Fr) / abs(wt.z));
        res.isDelta = true;
        res.pdf = 1.0f - Fr;
    }
}

void smoothDielectricBSDFEval(SmoothDielectricBSDF bsdf, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    res.bsdf = vec3(0.0f);
    res.pdf = 1.0f;
    res.isDelta = true;
}

void smoothConductorBSDFSample(SmoothConductorBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    float no = bsdf.iorOut;
    float nt = bsdf.iorIn;
    float Fr = nt == 0.0f ? 1.0f : fresnel(wo, no, nt);
    wi.z = wo.z;
    wi.x = -wo.x;
    wi.y = -wo.y;
    res.bsdf = Fr*vec3(1.0f/abs(wo.z));
    res.isDelta = true;
    res.pdf = 1.0f;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}

void smoothConductorBSDFEval(SmoothConductorBSDF bsdf, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    res.bsdf = vec3(0.0f);
    res.pdf = 1.0f;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
    res.isDelta = true;
}

void smoothFloorBSDFSample(SmoothFloorBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    float Fr = schlickFresnel(bsdf.R0, abs(wo.z));
    float u = randUniform();
    if (u < Fr) {
        wi.z = wo.z;
        wi.x = -wo.x;
        wi.y = -wo.y;
        res.bsdf = bsdf.diffuse * coupledDiffuseTerm(bsdf.R0, abs(wo.z), abs(wi.z)) + Fr * vec3(1.0f / abs(wo.z));
        res.pdf = Fr;
        res.isDelta = true;
        // NUMBERCHECK(output.bsdf)
        // NUMBERCHECK(output.pdf)
    } 
    else {
        wi = randCosineHemisphere();
        res.bsdf = bsdf.diffuse * coupledDiffuseTerm(bsdf.R0, abs(wo.z), abs(wi.z));
        res.pdf = (1.0f - Fr) * cosineHemispherePdf(wi);
        res.isDelta = false;
        // NUMBERCHECK(output.bsdf)
        // NUMBERCHECK(output.pdf)
    }
}

void smoothFloorBSDFEval(SmoothFloorBSDF bsdf, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    float Fr = schlickFresnel(bsdf.R0, abs(wo.z));
    res.bsdf = bsdf.diffuse * coupledDiffuseTerm(bsdf.R0, abs(wo.z), abs(wi.z));
    res.pdf = (1.0f - Fr) * cosineHemispherePdf(wi);
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}


void smoothPlasticBSDFSample(SmoothPlasticBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    float u = randUniform();
    float no = bsdf.iorOut;
    float nt = bsdf.iorIn;
    float Fri = fresnel(abs(wo.z), no, nt);
    if (u < Fri) {
        wi.z = wo.z;
        wi.x = -wo.x;
        wi.y = -wo.y;
        float Fro = fresnel(abs(wi.z), no, nt);
        float Ri = internalScatterEscapeFraction(bsdf.R0, no, nt);
        float eta = no / nt;
        res.bsdf = Fri * vec3(1.0f / abs(wo.z));
        res.pdf = Fri;
        res.isDelta = true;
        // NUMBERCHECK(output.bsdf)
        // NUMBERCHECK(output.pdf)
    } 
    else {
        wi = randCosineHemisphere();
        float Fro = fresnel(abs(wi.z), no, nt);
        float Ri = internalScatterEscapeFraction(bsdf.R0, no, nt);
        float eta = no / nt;
        vec3 d= bsdf.diffuse * eta * eta * (1.0f - Fri) * (1.0f - Fro)/ (M_PI * (1.0f - bsdf.diffuse * Ri));
        res.bsdf = d;
        res.pdf = (1.0f - Fri) * cosineHemispherePdf(wi);
        res.isDelta = false;
        // NUMBERCHECK(output.bsdf)
        // NUMBERCHECK(output.pdf)
    }
}

void smoothPlasticBSDFEval(SmoothPlasticBSDF bsdf, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    float no = bsdf.iorOut;
    float nt = bsdf.iorIn;
    float Fri = fresnel(abs(wo.z), no, nt);
    float Fro = fresnel(abs(wi.z), no, nt);
    float Ri = internalScatterEscapeFraction(bsdf.R0, no, nt);
    float eta = no / nt;
    vec3 d = bsdf.diffuse * (1.0f - Fri) * (1.0f - Fro) * eta * eta / (M_PI * (1.0f - bsdf.diffuse * Ri));
    res.bsdf = d;
    res.pdf = (1.0f - Fri) * cosineHemispherePdf(wi);
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}

void roughConductorBSDFSample(RoughConductorBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    vec3 Fr = FresnelDieletricConductor(bsdf.eta, bsdf.k, abs(wo.z));
    vec3 wh = sampleHalf(bsdf.alpha);
    if (wh.z <= 0.0f) {
        wh *= -1.0f;
    }
    wi = normalize(-wo + 2 * dot(wh, wo) * wh);
    res.bsdf = bsdf.reflectance * Fr * ggxD(wh, bsdf.alpha) * ggxMask(wo, wi, bsdf.alpha) / (4.0f * abs(wi.z) * abs(wo.z));
    res.pdf = beckmannD(wh, bsdf.alpha) * abs(wh.z) / (4.0f * abs(dot(wo,wh)));
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}

void roughConductorBSDFEval(RoughConductorBSDF bsdf,vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    vec3 Fr = FresnelDieletricConductor(bsdf.eta, bsdf.k, abs(wo.z));
    vec3 wh = normalize((wo + wi));
    res.bsdf = Fr * bsdf.reflectance * ggxD(wh, bsdf.alpha) * ggxMask(wo, wi, bsdf.alpha) / (4.0f * abs(wi.z) * abs(wo.z));
    res.pdf = beckmannD(wh, bsdf.alpha) * abs(wh.z) / (4.0f * abs(dot(wo,wh)));
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}

void roughPlasticBSDFSample(RoughPlasticBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    float u = randUniform();
    float no = bsdf.iorOut;
    float nt = bsdf.iorIn;
    float eta = no / nt;

    if (u < 0.5f) {
        vec3 wh = sampleHalf(bsdf.alpha);
        if (wh.z <= 0.0f) {
            wh *= -1.0f;
        }
        wi = normalize(-wo + 2 * dot(wh, wo) * wh);
    } 
    else {
        wi = randCosineHemisphere();
    }

    vec3 wh = normalize(wi + wo);
    float Fri = fresnel(abs(dot(wh, wo)), no, nt);
    float Fro = fresnel(abs(dot(wh, wi)), no, nt);
    float Ri = internalScatterEscapeFraction(bsdf.R0, no, nt);

    vec3 kD = bsdf.diffuse;
    vec3 specular = vec3(Fri) * ggxD(wh, bsdf.alpha) * ggxMask(wo, wi, bsdf.alpha) / (4.0f * abs(wo.z) * abs(wi.z));
    vec3 d = kD * (1.0f - Fri) * (1.0f - Fro) * eta * eta / (M_PI * (1.0f - kD * Ri));
    res.pdf = 0.5f * beckmannD(wh, bsdf.alpha) * abs(wh.z) / (4.0f * abs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
    res.bsdf = d + specular;
    res.isDelta = false;

    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}
    
void roughPlasticBSDFEval(RoughPlasticBSDF bsdf, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    float no = bsdf.iorOut;
    float nt = bsdf.iorIn;
    vec3 wh = normalize(wi + wo);
    float Fri = fresnel(abs(dot(wh, wo)), no, nt);
    float Fro = fresnel(abs(dot(wh, wi)), no, nt);
    float Ri = internalScatterEscapeFraction(bsdf.R0, no, nt);
    float eta = no / nt;

    vec3 kD = bsdf.diffuse;
    vec3 specular = vec3(Fri) * ggxD(wh, bsdf.alpha) * ggxMask(wo, wi, bsdf.alpha) / (4.0f * abs(wo.z) * abs(wi.z));
    vec3 d = kD * (1.0f - Fri) * (1.0f - Fro) * eta * eta / (M_PI * (1.0f - kD * Ri));
    res.pdf = 0.5f * max(beckmannD(wh, bsdf.alpha) * abs(wh.z), 0.01) / (4.0f * abs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
    res.bsdf = d + specular;
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}
void roughFloorBSDFSample(RoughFloorBSDF bsdf, vec2 uv, vec3 wo, out vec3 wi, out BSDFOutput res) {
    float u = randUniform();
    if (u < 0.5f) {
        vec3 wh = sampleHalf(bsdf.alpha);
        if (wh.z <= 0.0f) {
            wh *= -1.0f;
        }
        wi = normalize(-wo + 2 * dot(wh, wo) * wh);
    } 
    else {
        wi = randCosineHemisphere();
    }
    vec3 wh = normalize(wi + wo);
    float Fr = schlickFresnel(bsdf.R0, abs(dot(wo, wh)));
    vec3 d = bsdf.diffuse * fresnelBlendDiffuseTerm(bsdf.R0, abs(wo.z), abs(wi.z));
    vec3 specular = vec3(Fr) * ggxD(wh, bsdf.alpha)/ (4.0f * abs(dot(wo,wh))* max(abs(wo.z), abs(wi.z)));
    res.pdf = 0.5f * beckmannD(wh, bsdf.alpha) * abs(wh.z) / (4.0f * abs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
    res.bsdf = d + specular;
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}

void roughFloorBSDFEval(RoughFloorBSDF bsdf, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    vec3 wh = normalize(wi + wo);
    float Fr = schlickFresnel(bsdf.R0, abs(dot(wo,wh)));
    vec3 d = bsdf.diffuse * fresnelBlendDiffuseTerm(bsdf.R0, abs(wo.z), abs(wi.z));
    float ggx = ggxD(wh, bsdf.alpha);
    vec3 specular = vec3(Fr) * ggx / (4.0f * abs(dot(wo,wh)) * max(abs(wo.z), abs(wi.z)));
    res.pdf = 0.5f * beckmannD(wh, bsdf.alpha) * abs(wh.z) / (4.0f * abs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
    res.bsdf = d + specular;
    res.isDelta = false;
    // NUMBERCHECK(output.bsdf)
    // NUMBERCHECK(output.pdf)
}


bool isTransimissionBSDF(uint type) {
    switch (type) {
    case BSDF_SMOOTH_DIELECTRIC:
        return true;
    default:
        return false;
    }
}


void sampleBSDF(vec2 uv, BSDFHandle handle, vec3 wo, out vec3 wi, out BSDFOutput res) {
    switch (bsdfType(handle)) {
        #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
            case BSDF_##BSDFTYPE: { \
                BSDFNAME bsdf = BSDFNAME##Buffer(renderState.scene.BSDFFIELD##s).values[bsdfIndex(handle)]; \
                BSDFFIELD##Sample(bsdf, uv, wo, wi, res); \
                break; \
             }
#include "BSDF.inc"
#undef BSDFDefinition
    }
}

void evalBSDF(BSDFHandle handle, vec2 uv, vec3 wo, vec3 wi, out BSDFOutput res) {
    switch (bsdfType(handle)) {
        #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
            case BSDF_##BSDFTYPE: { \
                BSDFNAME bsdf = BSDFNAME##Buffer(renderState.scene.BSDFFIELD##s).values[bsdfIndex(handle)]; \
                BSDFFIELD##Eval(bsdf, uv, wo, wi, res); \
                break; }
#include "BSDF.inc"

#undef BSDFDefinition
    }
}

#define NEE false

bool isvalid(float x) {
    return isnan(x) == false && isinf(x) == false;
}

bool isvalid(vec3 x) {
    return isvalid(x.x) && isvalid(x.y) && isvalid(x.z);
}

void main()
{
    rngState = prd.seed;

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
	vec3 SN = normalize(barycentrics.x * normal0 + barycentrics.y * normal1 + barycentrics.z * normal2);
    vec3 N = normalize(cross(pos1 - pos0, pos2 - pos0));

    vec3 rayDir = prd.direction;
//   vec3 N = normal;
    if (dot(N, -rayDir) < 0) {
        // if (rt_data->twofaced) {
        if (instance.twofaced == 1 && vec3(instance.emission) == vec3(0.0)) {
            N *= -1.0f;
            SN *= -1.0f;
        }
            // SN *= -1.0f;
        // }
        
    }

    // if (rt_data->facenormals) {
    //     SN = N;
    // }
	Onb onb = onbCreate(SN);
    vec3 wo = normalize(onbTransform(onb, -rayDir));
    BSDFOutput bsdfRes;
    vec3 wi;
    sampleBSDF(vec2(0), instance.bsdf, wo, wi, bsdfRes);
    float NoW = abs(dot(wi, vec3(0.0f, 0.0f, 1.0f)));
    wi = onbUntransform(onb, wi);

    LightOutput lightRes = sampleLight(position);

    vec3 L = normalize(lightRes.position - position);
    vec3 wL = onbTransform(onb, L);
    float Ldist = length(lightRes.position - position);
    const float NoL = abs(dot(SN, L));
    float lightPdf = lightRes.pdf;

    BSDFOutput lightBsdfRes;
    evalBSDF(instance.bsdf, vec2(0), wo, wL, lightBsdfRes);

    // vec3 direct = vec3(0.0f);
    bool neeDone = false;
    // if (params.nee) 
    if (NEE) {
        if (!bsdfRes.isDelta) {
            if ((dot(N, -rayDir) > 0 && dot(N, L) > 0)  || isTransimissionBSDF(bsdfType(instance.bsdf))) {
                shadowed = true;  
                traceRayEXT(topLevelAS, 
                    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 
                    0xFF, 
                    0, 
                    0, 
                    1, 
                    position, 
                    0.01, 
                    L, 
                    Ldist - 0.01, 
                    2);

                if (!shadowed && lightPdf != 0.0f) {
                    float w = powerHeuristic(1, lightPdf, 1, bsdfRes.pdf);
                    prd.emitted += w * NoL * lightBsdfRes.bsdf * prd.weight * lightRes.emission / lightPdf;
                    neeDone = true;
                }
            }
        }
    }
    
    prd.seed = rngState;
    float lightFlag = dot(N, -rayDir) > 0 ? 1.0f : 0.0f;
    // This is adding term for the last path
    // we should account for it before termninating by invalid path tests
    if (NEE && prd.countEmitted == 0 && prd.wasDelta == 0) {
        prd.emitted += prd.directWeight * vec3(instance.emission) * lightFlag * prd.weight;
    }
    if (!NEE || prd.countEmitted == 1 || prd.wasDelta == 1) {
        prd.emitted += vec3(instance.emission) * lightFlag * prd.weight;
    }
    // sampled invalid hemisphere becase we used shading normal
    if (dot(wi, N) <= 0.0f && !isTransimissionBSDF(bsdfType(instance.bsdf))) {
        prd.done = 1;
        return;
    }
    // light leak from self intersection
    // TODO this is not working for two faced mesh
    if (dot(N, -rayDir) <= 0.0f && !isTransimissionBSDF(bsdfType(instance.bsdf))) {
        prd.done = 1;
        return;
    }
    // singular bsdf or pdf
    if (!isvalid(bsdfRes.pdf) || !isvalid(bsdfRes.bsdf) || bsdfRes.pdf == 0.0f) {
        prd.done = 1;
        return;
    }
    if (neeDone) {
        prd.directWeight = powerHeuristic(1, bsdfRes.pdf, 1, lightPdf);
    }
    else {
        prd.directWeight = 1.0f;
    }

    prd.countEmitted = 0;
    prd.origin = position + 0.001f*faceforward(N, -wi, N);;
    prd.direction = wi;
    prd.weight *= bsdfRes.bsdf * NoW / bsdfRes.pdf;
    prd.wasDelta = bsdfRes.isDelta ? 1 : 0;

}