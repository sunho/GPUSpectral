#pragma once

#include <vector>
#include <unordered_map>

#include "Camera.h"
#include "Entity.h"
#include "Light.h"
#include "utils/ResourceList.h"
#include <sunho3d/backend/DriverBase.h>

namespace sunho3d {
class Entity;
class VulkanDriver;
struct Geometry {
    Material* material;
    Handle<HwPrimitive> primitive;
    uint32_t vertexStart;
};

struct LightData {
    glm::vec4 pos;
    glm::vec4 dir;
    glm::vec4 RI;
};

static constexpr const size_t MAX_LIGHTS = 64;

struct LightBuffer {
    std::array<LightData, MAX_LIGHTS> lights;
    std::array<glm::mat4, MAX_LIGHTS> lightVP;
    int lightNum{};
    int pad[3];
};

struct SceneData {
    std::vector<glm::mat4> worldTransforms;
    std::vector<Geometry> geometries;
    LightBuffer lightBuffer;
    Handle<HwBufferObject> globalVertexBuffer;
};

class Renderer;
class Scene;

class GVetexBufferContainer {
public:
    GVetexBufferContainer(VulkanDriver& driver);
    ~GVetexBufferContainer();

    uint32_t getVertexStart(const Primitive& primitive);
    void registerPrimitiveIfNeccesary(const Primitive& primitive);

    Handle<HwBufferObject> getGPUBuffer() const;
private:
    void growBuffer();
    void uploadBuffer();

    std::unordered_map<uint32_t, uint32_t> vertexStarts;
    Handle<HwBufferObject> gpuBuffer;
    std::vector<Vertex> buffer;
    size_t currentSize{};
    size_t maxSize{256};
    VulkanDriver& driver;
};

struct DDGIConfig {
    glm::uvec3 gridNum{};
    glm::vec3 worldSize{};
};

class Scene : public IdResource {
  public:
    Scene(Renderer* renderer);
    void addEntity(Entity* entity);
    void addLight(Light* light);
    SceneData& getSceneData() {
        return sceneData;
    }
    Camera& getCamera() {
        return camera;
    }
    void prepare();
    DDGIConfig ddgi;
  private:
    void visitEntity(Entity* entity, const glm::mat4& currentTransform);

    std::vector<Entity*> entities;
    std::vector<Light*> lights;
    Camera camera;
    SceneData sceneData;
    Renderer* renderer;
    GVetexBufferContainer globalVertexBufferContainer; 

};

}  // namespace sunho3d
