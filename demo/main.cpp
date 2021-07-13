#include <sunho3d/Engine.h>
#include <sunho3d/Loader.h>

int main() {
    sunho3d::Engine engine;
    sunho3d::Loader loader(engine);
    //sunho3d::Scene* scene = loader.loadGLTF("Unity2Skfb.gltf");
    sunho3d::Scene* scene = engine.createScene();
    scene->addEntity(loader.loadObj("Unity2Skfb.obj"));
    sunho3d::Window* window = engine.createWindow(600, 600);
    sunho3d::Renderer* renderer = engine.createRenderer(window, scene);
    renderer->run();
    
    return 0;
}
