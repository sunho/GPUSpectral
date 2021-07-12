#include <sunho3d/Engine.h>
#include <sunho3d/Loader.h>

int main() {
    sunho3d::Engine engine;
    sunho3d::Loader loader(engine);
    sunho3d::Scene* scene = loader.loadGLTF("anime.gltf");
    sunho3d::Window* window = engine.createWindow(600, 600);
    sunho3d::Renderer* renderer = engine.createRenderer(window, scene);
    renderer->run();
    
    return 0;
}
