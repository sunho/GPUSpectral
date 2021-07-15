#pragma once

#include <vector>

#include "Camera.h"
#include "Entity.h"
#include "Light.h"
#include "utils/ResourceList.h"

namespace sunho3d {
class Entity;

struct Geometry {
    Material* material;
    Handle<HwPrimitive> primitive;
};

struct LightData {
    glm::vec4 pos;
    glm::vec4 dir;
    glm::vec4 RI;
};

struct TransformBuffer {
    glm::mat4 MVP;
    glm::mat4 model;
    glm::mat4 invModelT;
    glm::vec4 cameraPos;
};

static constexpr const size_t MAX_LIGHTS = 64;

struct LightBuffer {
    std::array<LightData, MAX_LIGHTS> lights;
    int lightNum{};
    int pad[3];
};

struct SceneData {
    std::vector<glm::mat4> worldTransforms;
    std::vector<Geometry> geometries;
    LightBuffer lightBuffer;
};

class Renderer;

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

  private:
    void visitEntity(Entity* entity, const glm::mat4& currentTransform);

    std::vector<Entity*> entities;
    std::vector<Light*> lights;
    Camera camera;
    SceneData sceneData;
    Renderer* renderer;
};

}  // namespace sunho3d
