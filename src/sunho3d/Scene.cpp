#include "Scene.h"

using namespace sunho3d;

Scene::Scene(Renderer* renderer)
    : renderer(renderer) {
}

void Scene::addEntity(Entity* entity) {
    entities.push_back(entity);
}

void Scene::prepare() {
    sceneData.geometries.clear();
    sceneData.worldTransforms.clear();
    sceneData.lightBuffer.lightNum = 0;

    for (auto& entity : entities) {
        visitEntity(entity, glm::identity<glm::mat4>());
    }

    for (auto& light : lights) {
        auto& transform = light->getTransform();
        glm::vec4 pos = glm::vec4(transform.x, transform.y, transform.z, 1.0f);
        glm::vec4 dir = glm::vec4(light->getDirection(), 0.0f);
        glm::vec4 RI = glm::vec4(light->getRadius(), light->getIntensity(), 0.0f, 0.0f);
        sceneData.lightBuffer.lights[sceneData.lightBuffer.lightNum++] = {
            .pos = pos,
            .dir = dir,
            .RI = RI
        };
    }
}

void Scene::visitEntity(Entity* entity, const glm::mat4& currentTransform) {
    glm::mat4 nextTransform = entity->getTransform().toMatrix() * currentTransform;
    for (auto& prim : entity->getPrimitives()) {
        sceneData.geometries.push_back({ .primitive = prim.hwInstance,
                                         .material = prim.material });
        sceneData.worldTransforms.push_back(nextTransform);
    }
    for (auto& entity : entity->getNodes()) {
        visitEntity(entity, nextTransform);
    }
}

void Scene::addLight(sunho3d::Light* light) {
    lights.push_back(light);
}
