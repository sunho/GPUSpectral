#include <sunho3d/Engine.h>
#include <sunho3d/Loader.h>
#include <sunho3d/Light.h>

int main() {
    sunho3d::Engine engine;
    sunho3d::Window* window = engine.createWindow(600, 600);
    sunho3d::Renderer* renderer = engine.createRenderer(window);
    sunho3d::Loader loader(engine, *renderer);
    
    //sunho3d::Scene* scene = loader.loadGLTF("Unity2Skfb.gltf");
    sunho3d::Scene* scene = engine.createScene(renderer);
    scene->addEntity(loader.loadObj("Unity2Skfb.obj"));
    scene->getCamera().lookAt(glm::vec3(0.0,0.5,3.0), glm::vec3(0.0,0.7,0.0), glm::vec3(0.0,1.0,0.0));
    scene->getCamera().setProjectionFov(glm::radians(45.0f), 1.0, 1.0f, 10.0f);
    auto l = new sunho3d::Light(sunho3d::Light::Type::DIRECTIONAL);
    l->setIntensity(1.0);
    l->setTransform({.x=0.0,.y=0.4,.z=1.0});
    scene->addLight(l);
    window->run([&](){
        renderer->run(scene);
    });
    
    return 0;
}
