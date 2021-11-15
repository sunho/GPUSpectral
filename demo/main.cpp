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
    sunho3d::Loader loader(engine, *renderer, basePath() / "assets");
    
    //sunho3d::Scene* scene = loader.loadGLTF("Unity2Skfb.gltf");

    auto scene = loader.loadMitsuba((basePath() / "assets" / "cornell-box" / "scene.xml").string());
    

    scene->ddgi.gridNum = glm::uvec3(12, 12, 12);
    scene->ddgi.worldSize = glm::vec3(2.0f,3.0f,2.0f);
    auto l = new sunho3d::Light(sunho3d::Light::Type::POINT);
    l->setIntensity(0.6);
    l->setTransform({.x=0.5,.y=-0.5,.z=6.0});
    scene->addLight(l);
    window->run([&](){
        //t.x += 0.1;
        //neptune->setTransform(t);
        renderer->run(scene);
    });
    
    return 0;
}
