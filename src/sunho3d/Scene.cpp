#include "Scene.h"

using namespace sunho3d;

Scene::Scene() {
}

void Scene::addEntity(Entity *entity) {
    entities.push_back(entity);
}
