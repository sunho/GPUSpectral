#include "Entity.h"
#include "Engine.h"

#include <iostream>

using namespace sunho3d;

Entity::Entity(uint32_t id) : IdResource(id) {
}

void Entity::addNode(Entity *entity) {
    nodes.push_back(entity);
}

const std::vector<Entity*>& Entity::getNodes() {
    return nodes;
}

void Entity::addPrimitive(Primitive &&primitive) {
    primitives.push_back(primitive);
}

