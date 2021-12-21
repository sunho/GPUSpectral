#pragma once

#include "Common.cuh":

#define EPS 0.0000001f

struct BSDFOutput {
    float3 bsdf;
    float pdf;
    bool isDelta;
};

HOSTDEVICE CUDAINLINE float fresnel(float no, float cosTho, float nt, float cosTht) {
    float a = nt * cosTho - no * cosTht;
    float ad = nt * cosTho + no * cosTht;
    float b = no * cosTho - nt * cosTht;
    float bd = no * cosTho + nt * cosTht;
    float A = a*a / (ad*ad);
    float B = b*b / (bd*bd);
    return 0.5f * (A + B);
}

HOSTDEVICE CUDAINLINE float fresnel(float3 wo, float no, float nt) {
    float sinTho = sqrt(fmaxf(wo.x * wo.x + wo.y * wo.y, 0.0f)); // sqrt((cos(phi)sin(theta))^2 + ((sin(phi)sin(theta))^2)
    float sqrtTerm = 1.0f - ((no * no) / (nt * nt)) * (sinTho * sinTho);
    if (sqrtTerm <= 0.0f) {
        return 1.0f;
    }
    float cosTht = sqrt(sqrtTerm);
    float cosTho = abs(wo.z);
    return fresnel(no, cosTho, nt, cosTht);
}

HOSTDEVICE CUDAINLINE float fresnel(float cosTho, float no, float nt) {
    float sinTho = sqrt(fmaxf(1.0f - cosTho*cosTho,0.0f));
    float sqrtTerm = 1.0f - ((no * no) / (nt * nt)) * (sinTho * sinTho);
    if (sqrtTerm <= 0.0f) {
        return 1.0f;
    }
    float cosTht = sqrt(sqrtTerm);
    return fresnel(no, cosTho, nt, cosTht);
}

HOSTDEVICE CUDAINLINE float3 fresnelConductor(float3 wo, float3 eta, float3 k) {
    float cosTho = abs(wo.z);
    float cos2 = cosTho * cosTho;
    float sin2 = 1.0f - cos2;
    float tan2 = sin2 / cos2;
    float3 n2 = eta * eta;
    float3 k2 = k * k;
    float3 c = n2 - k2 - sin2;
    float3 a2b2 = sqrt(c * c + 4.0f * n2 * k2);
    float3 a2 = 0.5f * (a2b2 + n2 - k2 - sin2);
    float3 a = sqrt(a2);
    float3 t = a2b2 - 2 * a * cosTho + cos2;
    float3 tt = a2b2 + 2 * a * cosTho + cos2;
    float3 Rs = t / tt;
    float3 d = cos2 * a2b2 - 2 * a * cosTho * sin2 + sin2 * sin2;
    float3 dd = cos2 * a2b2 + 2 * a * cosTho * sin2 + sin2 * sin2;
    float3 Rp = Rs * (d / dd);
    return 0.5f * (Rp + Rs);
}

HOSTDEVICE CUDAINLINE float3 FresnelDieletricConductor(float3 Eta, float3 Etak, float CosTheta)
{
    float CosTheta2 = CosTheta * CosTheta;
    float SinTheta2 = 1 - CosTheta2;
    float3 Eta2 = Eta * Eta;
    float3 Etak2 = Etak * Etak;

    float3 t0 = Eta2 - Etak2 - SinTheta2;
    float3 a2plusb2 = sqrt(t0 * t0 + 4 * Eta2 * Etak2);
    float3 t1 = a2plusb2 + CosTheta2;
    float3 a = sqrt(0.5f * (a2plusb2 + t0));
    float3 t2 = 2 * a * CosTheta;
    float3 Rs = (t1 - t2) / (t1 + t2);

    float3 t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
    float3 t4 = t2 * SinTheta2;
    float3 Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (Rp + Rs);
}

HOSTDEVICE CUDAINLINE bool refract(float3 wo, float3 n, float no, float nt, float3& wt) {
    float sinTho = sqrt(fmaxf(wo.x * wo.x + wo.y * wo.y,0.0f)); // sqrt((cos(phi)sin(theta))^2 + ((sin(phi)sin(theta))^2)
    float sqrtTerm = 1.0f - ((no * no) / (nt * nt)) * (sinTho * sinTho);
    if (sqrtTerm <= 0.0f) {
        return false;
    }
    float cosTht = sqrt(sqrtTerm);
    wt = (no / nt) * (-wo) + ((no / nt) * dot(wo, n) - cosTht) * n;
    return true;
}

HOSTDEVICE CUDAINLINE float coupledDiffuseTerm(float R0, float cosTho, float cosThi) {
    float k = 21.0f / (20.0f * M_PI * (1.0f - R0));
    float a = 1.0f - cosTho;
    float b = 1.0f - cosThi;
    float a5 = a * a * a * a * a;
    float b5 = b * b * b * b * b;
    return k * (1.0f - a5)* (1.0f - b5);
}

HOSTDEVICE CUDAINLINE float fresnelBlendDiffuseTerm(float R0, float cosTho, float cosThi) {
    float k = 28.0f / (23.0f * M_PI);
    float a = 1.0f - 0.5f*cosTho;
    float b = 1.0f - 0.5f*cosThi;
    float a5 = a * a * a * a * a;
    float b5 = b * b * b * b * b;
    return k * (1.0f - R0) * (1.0f - a5)* (1.0f - b5);
}

// R_i
HOSTDEVICE CUDAINLINE float internalScatterEscapeFraction(float R0, float no, float nt) {
    float Re = (M_PI * 20.0f * R0 + 1.0f) / 21.0f;
    float eta = no / nt;
    return 1.0f - eta * eta * (1.0f - Re);
}

HOSTDEVICE CUDAINLINE float schlickFresnel(float R0, float cosTho) {
    float a = 1.0f - cosTho;
    float a5 = a * a * a * a * a;
    return R0 + a5 * (1.0f - R0);
}

enum BSDFType: uint16_t {
    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) BSDF_##BSDFTYPE,
    #include "BSDF.inc"
    #undef BSDFDefinition
};

struct BSDFHandle {
    HOSTDEVICE CUDAINLINE BSDFHandle() { } 
    BSDFHandle(BSDFType type, uint32_t index) : handle((static_cast<uint32_t>(type) << 16) | (index & 0xFFFF)){
    }
    HOSTDEVICE CUDAINLINE BSDFType type() const {
        return static_cast<BSDFType>((handle >> 16) & 0xffff);
    }
    HOSTDEVICE CUDAINLINE uint32_t index() const {
        return handle & 0xFFFF;
    }
    uint32_t handle;
};

struct DiffuseBSDF {
    float3 reflectance;
    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        wi = randCosineHemisphere(sampler);
        output.bsdf = reflectance / M_PI;
        output.pdf = cosineHemispherePdf(wi);
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }

    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        output.bsdf = reflectance / M_PI;
        output.pdf = cosineHemispherePdf(wi);
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }
};

struct SmoothDielectricBSDF {
    float iorIn;
    float iorOut; // usually 1.0

    // bsdf for reflection: F_r/|cos(th_o)|
    // bsdf for refraction: (n_t^2/n_o^2)(1-F_r)/|cos(th_o)|
    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        bool entering = wo.z > 0.0f;
        float no = entering ? iorOut : iorIn;
        float nt = entering ? iorIn : iorOut;
        float cosTho = wo.z;
        float3 wt;
        if (!refract(wo, faceforward(make_float3(0.0f, 0.0f, 1.0f), wo, make_float3(0.0f, 0.0f, 1.0f)), no, nt, wt)) {
            // total internal reflection
            wi.z = wo.z;
            wi.x = -wo.x;
            wi.y = -wo.y;
            output.bsdf = 1.0f * make_float3(1 / abs(cosTho));
            output.isDelta = true;
            output.pdf = 1.0f;
            return;
        }
        float Fr = fresnel(no, abs(cosTho), nt, abs(wt.z));
        float u = randUniform(sampler);
        // reflection with prob Fr
        // refraction with prob 1 - Fr
        if (u < Fr) {
            // reflect
            wi.z = wo.z;
            wi.x = -wo.x;
            wi.y = -wo.y;
            output.bsdf = Fr * make_float3(1 / abs(cosTho));
            output.isDelta = true;
            output.pdf = Fr;
        } else {
            // refract
            wi = wt;

            output.bsdf = make_float3(((no * no) / (nt * nt)) * (1.0f - Fr) / abs(wt.z));
            output.isDelta = true;
            output.pdf = 1.0f - Fr;
        }
    }

    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        output.bsdf = make_float3(0.0f);
        output.pdf = 1.0f;
        output.isDelta = true;
    }
};

struct SmoothConductorBSDF {
    float iorIn;
    float iorOut; // usually 1.0

    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        float no = iorOut;
        float nt = iorIn;
        float Fr = nt == 0.0f ? 1.0f : fresnel(wo, no, nt);
        wi.z = wo.z;
        wi.x = -wo.x;
        wi.y = -wo.y;
        output.bsdf = Fr*make_float3(1.0f/abs(wo.z));
        output.isDelta = true;
        output.pdf = 1.0f;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }

    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        output.bsdf = make_float3(0.0f);
        output.pdf = 1.0f;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
        output.isDelta = true;
    }
};

struct SmoothFloorBSDF {
    float3 diffuse;
    float R0;

    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        float Fr = schlickFresnel(R0, abs(wo.z));
        float u = randUniform(sampler);
        if (u < Fr) {
            wi.z = wo.z;
            wi.x = -wo.x;
            wi.y = -wo.y;
            output.bsdf = diffuse * coupledDiffuseTerm(R0, abs(wo.z), abs(wi.z)) + Fr * make_float3(1.0f / abs(wo.z));
            output.pdf = Fr;
            output.isDelta = true;
            NUMBERCHECK(output.bsdf)
            NUMBERCHECK(output.pdf)
        } 
        else {
            wi = randCosineHemisphere(sampler);
            output.bsdf = diffuse * coupledDiffuseTerm(R0, abs(wo.z), abs(wi.z));
            output.pdf = (1.0f - Fr) * cosineHemispherePdf(wi);
            output.isDelta = false;
            NUMBERCHECK(output.bsdf)
            NUMBERCHECK(output.pdf)
        }
    }

    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        float Fr = schlickFresnel(R0, fabs(wo.z));
        output.bsdf = diffuse * coupledDiffuseTerm(R0, fabs(wo.z), fabs(wi.z));
        output.pdf = (1.0f - Fr) * cosineHemispherePdf(wi);
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }
};

struct SmoothPlasticBSDF {
    float3 diffuse;
    float iorIn;
    float iorOut; // usually 1.0
    float R0;

    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        float u = randUniform(sampler);
        float no = iorOut;
        float nt = iorIn;
        float Fri = fresnel(fabs(wo.z), no, nt);
        if (u < Fri) {
            wi.z = wo.z;
            wi.x = -wo.x;
            wi.y = -wo.y;
            float Fro = fresnel(fabs(wi.z), no, nt);
            float Ri = internalScatterEscapeFraction(R0, no, nt);
            float eta = no / nt;
            float3 d = diffuse * eta * eta *(1.0f - Fri) * (1.0f - Fro) / (M_PI * (1.0f - diffuse * Ri));
            output.bsdf = d + Fri * make_float3(1.0f / abs(wo.z));
            output.pdf = Fri;
            output.isDelta = true;
            NUMBERCHECK(output.bsdf)
            NUMBERCHECK(output.pdf)
        } 
        else {
            wi = randCosineHemisphere(sampler);
            float Fro = fresnel(fabs(wi.z), no, nt);
            float Ri = internalScatterEscapeFraction(R0, no, nt);
            float eta = no / nt;
            float3 d= diffuse * eta * eta * (1.0f - Fri) * (1.0f - Fro)/ (M_PI * (1.0f - diffuse * Ri));
            output.bsdf = d;
            output.pdf = (1.0f - Fri) * cosineHemispherePdf(wi);
            output.isDelta = false;
            NUMBERCHECK(output.bsdf)
            NUMBERCHECK(output.pdf)
        }
    }

    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        float no = iorOut;
        float nt = iorIn;
        float Fri = fresnel(fabs(wo.z), no, nt);
        float Fro = fresnel(fabs(wi.z), no, nt);
        float Ri = internalScatterEscapeFraction(R0, no, nt);
        float eta = no / nt;
        float3 diffuse = diffuse * (1.0f - Fri) * (1.0f - Fro) * eta * eta / (M_PI * (1.0f - diffuse * Ri));
        output.bsdf = diffuse;
        output.pdf = (1.0f - Fri) * cosineHemispherePdf(wi);
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }
};

enum MicrofacetType : uint32_t {
    BECKMANN,
    GGX
};

struct RoughConductorBSDF {
    float3 eta;
    float3 k;
    float3 reflectance;
    float alpha;
    MicrofacetType distribution;

    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        float3 Fr = FresnelDieletricConductor(eta, k, fabs(wo.z));
        float3 wh = sampleHalf(sampler, alpha);
        if (wh.z <= 0.0f) {
            wh *= -1.0f;
        }
        wi = normalize(-wo + 2 * dot(wh, wo) * wh);
        output.bsdf = reflectance * Fr * ggxD(wh, alpha) * ggxMask(wo, wi, alpha);
        output.pdf = beckmannD(wh, alpha) * fabs(wh.z) / (4.0f * fabs(dot(wo,wh)));
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }

    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        float3 Fr = FresnelDieletricConductor(eta, k, fabs(wo.z));
        float3 wh = normalize((wo + wi));
        output.bsdf = Fr * reflectance * ggxD(wh, alpha) * ggxMask(wo, wi, alpha);
        output.pdf = beckmannD(wh, alpha) * fabs(wh.z) / (4.0f * fabs(dot(wo,wh)));
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
        output.isDelta = false;
    }
};

struct RoughPlasticBSDF {
    float3 diffuse;
    float iorIn;
    float iorOut; // usually 1.0
    float R0;
    float alpha;
    MicrofacetType distribution;

    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        float u = randUniform(sampler);
        float no = iorOut;
        float nt = iorIn;
        if (u < 0.5f) {
            float3 wh = sampleHalf(sampler, alpha);
            if (wh.z <= 0.0f) {
                wh *= -1.0f;
            }
            wi = normalize(-wo + 2 * dot(wh, wo) * wh);
        } 
        else {
            wi = randCosineHemisphere(sampler);
        }
        float3 wh = normalize(wi + wo);
        float Fri = fresnel(fabs(dot(wh, wo)), no, nt);
        float Fro = fresnel(fabs(dot(wh, wi)), no, nt);
        float Ri = internalScatterEscapeFraction(R0, no, nt);
        float eta = no / nt;
        float3 specular = make_float3(Fri) * ggxD(wh, alpha) * ggxMask(wo, wi, alpha) / (4.0f * fabs(wo.z)); //TODO do we need cos wi too?
        float3 d = diffuse * (1.0f - Fri) * (1.0f - Fro) * eta * eta / (M_PI * (1.0f - diffuse * Ri));
        output.pdf = 0.5f * beckmannD(wh, alpha) * fabs(wh.z) / (4.0f * abs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
        output.bsdf = d + specular;
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }
    
    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        float no = iorOut;
        float nt = iorIn;
        float3 wh = normalize(wi + wo);
        float Fri = fresnel(fabs(dot(wh, wo)), no, nt);
        float Fro = fresnel(fabs(dot(wh, wi)), no, nt);
        float Ri = internalScatterEscapeFraction(R0, no, nt);
        float eta = no / nt;
        float3 specular = make_float3(Fri) * ggxD(wh, alpha) * ggxMask(wo, wi, alpha) / (4.0f * fabs(wo.z)); //TODO do we need cos wi too?
        float3 d = diffuse * (1.0f - Fri) * (1.0f - Fro) * eta * eta / (M_PI * (1.0f - diffuse * Ri));
        output.pdf = 0.5f * fmaxf(beckmannD(wh, alpha) * fabs(wh.z), 0.01) / (4.0f * abs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
        output.bsdf = d + specular;
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }
};

struct RoughFloorBSDF {
    float3 diffuse;
    float R0;
    float alpha;
    MicrofacetType distribution;

    HOSTDEVICE CUDAINLINE void sample(SamplerState& sampler, float3 wo, float3& wi, BSDFOutput& output) const {
        float u = randUniform(sampler);
        if (u < 0.5f) {
            float3 wh = sampleHalf(sampler, alpha);
            if (wh.z <= 0.0f) {
                wh *= -1.0f;
            }
            wi = normalize(-wo + 2 * dot(wh, wo) * wh);
        } 
        else {
            wi = randCosineHemisphere(sampler);
        }
        float3 wh = normalize(wi + wo);
        float Fr = schlickFresnel(R0, fabs(dot(wo, wh)));
        float3 d = diffuse * fresnelBlendDiffuseTerm(R0, fabs(wo.z), fabs(wi.z));
        float3 specular = make_float3(Fr) * ggxD(wh, alpha)/ (4.0f * fabs(dot(wo,wh))* fmaxf(fabs(wo.z), fabs(wi.z)));
        output.pdf = 0.5f * beckmannD(wh, alpha) * fabs(wh.z) / (4.0f * abs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
        output.bsdf = d + specular;
        output.isDelta = false;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
    }

    HOSTDEVICE CUDAINLINE void eval(float3 wo, float3 wi, BSDFOutput& output) const {
        float3 wh = normalize(wi + wo);
        float Fr = schlickFresnel(R0, fabs(dot(wo,wh)));
        float3 d = diffuse * fresnelBlendDiffuseTerm(R0, abs(wo.z), abs(wi.z));
        float ggx = ggxD(wh, alpha);
        float3 specular = make_float3(Fr) * ggx / (4.0f * fabs(dot(wo,wh)) * fmaxf(fabs(wo.z), fabs(wi.z)));
        output.pdf = 0.5f * beckmannD(wh, alpha) * abs(wh.z) / (4.0f * fabs(dot(wo, wh))) + 0.5f * cosineHemispherePdf(wi);
        output.bsdf = d + specular;
        NUMBERCHECK(output.bsdf)
        NUMBERCHECK(output.pdf)
        output.isDelta = false;
    }
};

HOSTDEVICE CUDAINLINE bool isTransimissionBSDF(BSDFType type) {
    switch (type) {
    case BSDF_SMOOTH_DIELECTRIC:
        return true;
    default:
        return false;
    }
}

struct BSDFData {
    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) Array<BSDFNAME> BSDFFIELD##s;
    #include "BSDF.inc"
    #undef BSDFDefinition
};

HOSTDEVICE CUDAINLINE void sampleBSDF(const BSDFData& data, SamplerState& sampler, const BSDFHandle& handle, float3 wo, float3& wi, BSDFOutput& output) {
    switch (handle.type()) {
        #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
            case BSDF_##BSDFTYPE: { \
                data.BSDFFIELD##s[handle.index()].sample(sampler, wo, wi, output); \
                break; \
             }
        #include "BSDF.inc"
        #undef BSDFDefinition
    }
}

HOSTDEVICE CUDAINLINE void evalBSDF(const BSDFData& data, const BSDFHandle& handle, float3 wo, float3 wi, BSDFOutput& output) {
    switch (handle.type()) {
        #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) \
            case BSDF_##BSDFTYPE: { \
                data.BSDFFIELD##s[handle.index()].eval(wo, wi, output); \
                break; \
             }
        #include "BSDF.inc"
        #undef BSDFDefinition
    }
}



