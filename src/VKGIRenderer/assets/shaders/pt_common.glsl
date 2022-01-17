#define M_PI 3.14159265358979323846
#define DevicePtr uvec2

struct HitPayload {
    vec3 emitted;
    vec3 weight;
    vec3 origin;
    vec3 direction;
    float directWeight;
    uint seed;
    int wasDelta;
    int countEmitted;
    int done;
};

struct Camera {
    vec4 eye;
    mat4 view;
    float fov;
    // int pad[3];
};

struct RenderParams {
    int spp;
    int toneMap;
    int nee;
    uint timestamp;
};

#define BSDFHandle uint

uint bsdfHandle(uint type, uint index) {
    return (type << 16) | index;
}

uint bsdfType(uint handle) {
    return handle >> 16;
}

uint bsdfIndex(uint handle) {
    return handle & 0xffff;
}

struct Instance {
	mat4 transformInvT;
	uvec2 positionBuffer;
	uvec2 normalBuffer;
    vec4 emission;
    BSDFHandle bsdf;
    uint twofaced;
};

struct Scene {
    DevicePtr instances;
    DevicePtr triangleLights;
    #define BSDFDefinition(BSDFNAME, BSDFFIELD, BSDFTYPE) DevicePtr BSDFFIELD##s;
    #include "BSDF.inc"
    #undef BSDFDefinition
    int numLights;
    int a;
    int b;
    int c;
};

struct RenderState {
    Camera camera;
    Scene scene;
    RenderParams params;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer PositionBuffer
{
   	vec3 positions[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer NormalBuffer
{
   	vec3 normals[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer InstanceBuffer
{
   	Instance values[];
};

uint rngState;
uint randPcg()
{
    uint state = rngState;
    rngState = rngState * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint pcgHash(uint v)
{
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float randUniform() {
    return randPcg()*(1.0/float(0xffffffffu));
}

uint tea(uint val0, uint val1)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;

    for (uint n = 0; n < 4; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

struct Onb {
	vec3 tangent;
	vec3 binormal;
	vec3 normal;
};

Onb onbCreate(vec3 n) {
    Onb onb;
    onb.normal = normalize(n);
    if( abs(onb.normal.x) > abs(onb.normal.z) )
    {
        onb.binormal = vec3(-onb.normal.y, onb.normal.x, 0.0);
    }
    else
    {
        onb.binormal = vec3(0.0, -onb.normal.z, onb.normal.y);
    }

    onb.binormal = normalize(onb.binormal);
    onb.tangent = cross(onb.binormal, onb.normal);
    return onb;
}

vec3 onbUntransform(Onb o, vec3 v) {
	return o.tangent * v.x + o.binormal * v.y + o.normal * v.z;
}

vec3 onbTransform(Onb o, vec3 v) {
	return vec3(dot(v, o.tangent), dot(v, o.binormal), dot(v, o.normal));
}



// // http://corysimon.github.io/articles/uniformdistn-on-sphere/
// // tldr; find pdf function by f(v)*dA = 1 = f(phi,theta) * dphi * dtheta
// // dA = sin(phi) * dphi * dtheta
// // inverse transform marginal pdf of phi
// HOSTDEVICE CUDAINLINE float3 randDirSphere(SamplerState& sampler) {
//     float2 u = make_float2(randUniform(sampler), randUniform(sampler));
//     float theta = 2.0 * M_PI * u.x;
//     float phi = acos(1.0 - 2.0 * u.y);
//     float x = sin(phi) * cos(theta);
//     float y = sin(phi) * sin(theta);
//     float z = cos(phi);
//     return make_float3(x, y, z);
// }

// HOSTDEVICE CUDAINLINE float3 randDirHemisphere(SamplerState& sampler) {
//     float2 u = make_float2(randUniform(sampler), randUniform(sampler));
//     float theta = 2.0 * M_PI * u.x;
//     float phi = acos(1.0 - u.y);
//     float x = sin(phi) * cos(theta);
//     float y = sin(phi) * sin(theta);
//     float z = cos(phi);
//     return make_float3(x, y, z);
// }
