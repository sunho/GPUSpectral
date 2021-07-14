#include "Entity.h"

#include <iostream>

#include "Engine.h"

using namespace sunho3d;

Entity::Entity() {
}

void Entity::addNode(Entity *entity) {
    nodes.push_back(entity);
}

const std::vector<Entity *> &Entity::getNodes() {
    return nodes;
}

void Entity::addPrimitive(Primitive &&primitive) {
    primitives.push_back(primitive);
}
