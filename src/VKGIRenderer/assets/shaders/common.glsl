#define MAX_INSTANCES 64
#define MAX_DIFFUSE_MAPS 32
#define M_PI 3.1415926535897932384626433832795

#define MAX_MESH_COUNT 64
#define IRD_MAP_SIZE 8
#define IRD_MAP_PROBE_COLS 8
#define IRD_MAP_BORDER 1
#define IRD_MAP_TEX_SIZE 10

#define DEPTH_MAP_SIZE 24
#define DEPTH_MAP_PROBE_COLS 16
#define DEPTH_MAP_BORDER 1
#define DEPTH_MAP_TEX_SIZE 26
#define RAYS_PER_PROBE 64


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

struct SceneInfo {
    uvec3 gridNum;
    vec3 sceneSize;
    vec3 sceneCenter;
};


#define MATERIAL_DIFFUSE_TEXTURE 1
#define MATERIAL_DIFFUSE_COLOR 2
#define MATERIAL_EMISSION 3

struct Material {
    vec3 diffuseColor;
    int diffuseMapIndex;
    int typeID;
};

struct Instance {
    mat4 transform;
    int meshIndex;
    Material material;
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

vec3 ACESFilm(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0);
}

struct SceneBuffer {
    uvec2 frameSize;
    uint instanceNum;
    Instance instances[MAX_INSTANCES];
    SceneInfo sceneInfo;
};