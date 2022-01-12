
struct HitPayload {
    vec4 output;
    vec3 weight;
    vec3 origin;
    vec3 direction;
    float directWeight;
    int seed;
    int wasDelta;
    int countEmitted;
    int done;
};

uint randPcg(inout uint rngState)
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

float randUniform(inout uint rngState) {
    return randPcg(rngState)*(1.0/float(0xffffffffu));
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

// HOSTDEVICE CUDAINLINE float2 sampleConcentric(SamplerState& sampler) {
//     float2 sample = make_float2(randUniform(sampler), randUniform(sampler));
//     float2 u = 2.0f * sample - 1.0f;
//     if (u.x == 0.0f && u.y == 0.0f) {
//         return make_float2(0.0f, 0.0f);
//     }
//     float r, th;
//     if (fabs(u.x) > fabs(u.y)) {
//         r = u.x;
//         th = (M_PI / 4.0f) * (u.y / u.x);
//     }
//     else {
//         r = u.y;
//         th = M_PI / 2.0f - (M_PI / 4.0f) * (u.x / u.y);
//     }
//     return make_float2(r * cos(th), r * sin(th));
// }

// HOSTDEVICE CUDAINLINE float3 randCosineHemisphere(SamplerState& sampler) {
//     float2 u = sampleConcentric(sampler);
//     float z = sqrt(fmaxf(0.0f, 1.0f - u.x * u.x - u.y * u.y));
//     return make_float3(u.x, u.y, z);
// }

// HOSTDEVICE CUDAINLINE float cosineHemispherePdf(float3 wo) {
//     return fmaxf(abs(wo.z) / M_PI, 0.000001f);
// }
