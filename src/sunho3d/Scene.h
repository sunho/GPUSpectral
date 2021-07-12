#pragma once

#include <vector>
#include "ResourceList.h"

namespace sunho3d {

class Entity;
class Scene : public IdResource {
public:
    Scene(uint32_t id);
    void addEntity(Entity* entity);
//private:
    std::vector<Entity*> entities;
};

}
