#include <sunho3d/Engine.h>

int main() {
    sunho3d::Engine engine;
    sunho3d::Window* window = engine.createWindow(800, 600);
    sunho3d::Renderer* renderer = engine.createRenderer(window);
    renderer->run();
    
    return 0;
}
