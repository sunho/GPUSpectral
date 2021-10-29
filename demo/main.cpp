#include <sunho3d/Engine.h>
#include <sunho3d/Loader.h>
#include <sunho3d/Light.h>

#include <Windows.h>
#include <filesystem>

std::filesystem::path basePath() {
    char path[1024];
    GetModuleFileNameA(0, path, 1024);
    auto out = std::filesystem::path(path);
    return out.parent_path();
}

int main() {
    sunho3d::Engine engine;
    sunho3d::Window* window = engine.createWindow(1200, 1200);
    sunho3d::Renderer* renderer = engine.createRenderer(window);
    sunho3d::Loader loader(engine, *renderer);
    
    //sunho3d::Scene* scene = loader.loadGLTF("Unity2Skfb.gltf");
    sunho3d::Scene* scene = engine.createScene(renderer);
    auto neptune = loader.loadObj((basePath() / "assets" / "Unity2Skfb.obj").string());
    scene->addEntity(neptune);
    scene->getCamera().lookAt(glm::vec3(0.0,0.5,3.0), glm::vec3(0.0,0.7,0.0), glm::vec3(0.0,1.0,0.0));
    scene->getCamera().setProjectionFov(glm::radians(45.0f), 1.0, 1.0f, 10.0f);
    
    auto cube = loader.loadObj((basePath() / "assets" / "cube.obj").string());
    scene->addEntity(cube);
    auto t2 = cube->getTransform();
    t2.z = -1.7f;
    t2.sx = 30.0f;
    t2.sy = 30.0f;
    cube->setTransform(t2);
    auto l = new sunho3d::Light(sunho3d::Light::Type::DIRECTIONAL);
    l->setIntensity(0.6);
    l->setTransform({.x=-0.3,.y=1.3,.z=1.2});
    scene->addLight(l);
    
    auto t = neptune->getTransform();
    t.ry += 180.0f;
    //t.rz += 1.0f;
    neptune->setTransform(t);
    window->run([&](){
        renderer->run(scene);
    });
    
    return 0;
}
