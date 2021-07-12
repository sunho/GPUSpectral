#pragma once

#include <memory>
#include <list>
#include <stdlib.h>
#include "ResourceList.h"
#include "Window.h"
#include "Entity.h"
#include "Renderer.h"
#include "Scene.h"

namespace sunho3d {

class Engine {
public:
    Engine();
    ~Engine();
    
    Window* createWindow(size_t width, size_t height);
    Entity* createEntity();
    Renderer* createRenderer(Window* window, Scene* scene);
    Scene* createScene();
private:
    ResourceList<Window> windows;
    ResourceList<Renderer> renderers;
    ResourceList<Entity> entities;
    ResourceList<Scene> scenes;
};

}
