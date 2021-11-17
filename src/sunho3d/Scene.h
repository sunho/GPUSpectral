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
    Primitive primitive;
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
    std::vector<glm::mat4> worldTransformsInvT;
    std::vector<Geometry> geometries;
    LightBuffer lightBuffer;
    Handle<HwBufferObject> globalVertexBuffer;
};

class Renderer;
class Scene;

struct DDGIConfig {
    glm::uvec3 gridNum{};
    glm::vec3 worldSize{};
    glm::vec3 gridOrigin{};
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
    void visitEntity(Entity* entity, const glm::mat4& currentTransform, const glm::mat4& currentTransformInvT);

    std::vector<Entity*> entities;
    std::vector<Light*> lights;
    Camera camera;
    SceneData sceneData;
    Renderer* renderer;

};

}  // namespace sunho3d
