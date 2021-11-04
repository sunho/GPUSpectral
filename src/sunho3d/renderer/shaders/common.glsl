#define MAX_INSTANCES 32
#define M_PI 3.1415926535897932384626433832795

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct RayHit {
    vec2 bary;
    uint instId;
    uint primId;
};

struct Instance {
    mat4 transform;
    int vertexStart;
};

struct Ray {
    vec3 origin;
    float minTime;
    vec3 dir;
    float maxTime;
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

// http://corysimon.github.io/articles/uniformdistn-on-sphere/
// tldr; find pdf function by f(v)*dA = 1 = f(phi,theta) * dphi * dtheta
// dA = sin(phi) * dphi * dtheta
// inverse transform marginal pdf of phi
vec3 randDirSphere() {
    vec2 u = vec2(randUniform(), randUniform());
    float theta = 2.0 * M_PI * u.x;
    float phi = acos(1.0 - 2.0 * u.y);
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    return vec3(x,y,z);
}