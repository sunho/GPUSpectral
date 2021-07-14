#include "Entity.h"

#include <iostream>

#include "Engine.h"

using namespace sunho3d;

Entity::Entity() {
}

void Entity::addNode(Entity* entity) {
    nodes.push_back(entity);
}

void Entity::addPrimitive(const Primitive& primitive) {
    primitives.push_back(primitive);
}
