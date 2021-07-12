#include "Scene.h"


using namespace sunho3d;

Scene::Scene(uint32_t id) : IdResource(id) {
}

void Scene::addEntity(Entity *entity) {
    entities.push_back(entity);
}
