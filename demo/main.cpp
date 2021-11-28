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
    sunho3d::Engine engine(basePath(), basePath() / "assets");
    sunho3d::Window* window = engine.createWindow(1200, 1200);
    sunho3d::Renderer* renderer = engine.createRenderer(window);
    sunho3d::Loader loader(engine, *renderer);
    

    //sunho3d::Scene* scene = loader.loadGLTF("Unity2Skfb.gltf");
    
    auto scene = loader.loadMitsuba((basePath() / "assets" / "staircase2" / "scene.xml").string());
    scene->ddgi.gridNum = glm::uvec3(32, 16, 32);
    scene->ddgi.worldSize = glm::vec3(15.0f,10.0f,15.0f);
    scene->ddgi.gridOrigin = glm::vec3(0.0f, 1.0f, 0.0f);
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
   

    /*

    auto scene = loader.loadMitsuba((basePath() / "assets" / "cornell-box" / "scene.xml").string());
    scene->ddgi.gridNum = glm::uvec3(16, 16, 16);
    scene->ddgi.worldSize = glm::vec3(1.5f, 1.5f, 1.5f);
    scene->ddgi.gridOrigin = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraPos = glm::vec3(0.0, 1.0, 13.0f);
    glm::vec3 origin = glm::vec3(0.0, 1.0, 0.0);

    window->run([&]() {
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
    });*/
    return 0;
}
