#include "Entity.h"

#include <iostream>

#include "Engine.h"

using namespace sunho3d;

Entity::Entity() {
}

void Entity::addNode(Entity* entity) {
    nodes.push_back(entity);
}

Mesh::Mesh() {
}

void Mesh::addPrimitive(const Primitive& primitive) {
    primitives.push_back(primitive);
}
