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

    auto scene = loader.loadMitsuba((basePath() / "assets" / "staircase2" / "scene.xml").string());
    

    scene->ddgi.gridNum = glm::uvec3(32, 16, 32);
    scene->ddgi.worldSize = glm::vec3(15.0f,10.0f,15.0f);
    scene->ddgi.gridOrigin = glm::vec3(0.0f, 1.0f, 0.0f);
    /* auto l = new sunho3d::Light(sunho3d::Light::Type::POINT);
    l->setIntensity(0.6);
    l->setTransform({.x=0.5,.y=-0.5,.z=6.0});
    scene->addLight(l);*/
    glm::vec3 cameraPos = glm::vec3(6.0, 2.0, 1.0f);
    glm::vec3 origin = glm::vec3(-2.0, 1.0, 0.0);

    window->run([&](){
        //t.x += 0.1;
        //neptune->setTransform(t);
        int a = glfwGetKey(window->window, GLFW_KEY_A);
        if (a == GLFW_PRESS)
        {
            cameraPos.z -= 0.05f;
        }
        int d = glfwGetKey(window->window, GLFW_KEY_D);
        if (d == GLFW_PRESS)
        {
            cameraPos.x += 0.05f;
        }
        int w = glfwGetKey(window->window, GLFW_KEY_W);
        if (w == GLFW_PRESS)
        {
            cameraPos.z += 0.05f;
        }
        int s = glfwGetKey(window->window, GLFW_KEY_S);
        if (s == GLFW_PRESS)
        {
            cameraPos.x -= 0.05f;
        }
        scene->getCamera().lookAt(cameraPos, origin, glm::vec3(0.0, 1.0, 0.0));
        renderer->run(scene);
    });
    
    return 0;
}
