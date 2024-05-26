#pragma once

#include <glm/glm.hpp>
#include <engine/util/keycode.hpp>
#include <engine/util/mousecode.hpp>
#include <engine/graphics/Camera.hpp>

struct GLFWwindow;

namespace en
{
    class Input
    {
    public:
        static void Init(GLFWwindow* windowHandle);
        static void Update();
        static bool IsKeyPressed(int keycode);
        static bool IsKeyReleased(int keycode);
        static bool IsMouseButtonPressed(int button);
        static glm::vec2 GetMousePos();
        static glm::vec2 GetMouseMove();

        static void HandleUserCamInput(Camera* cam, float deltaTime);

    private:
        static GLFWwindow* m_WindowHandle;
        static glm::vec2 m_MousePos;
        static glm::vec2 m_MouseMove;

        static glm::vec3 m_camPoints[100];

        static float m_CurrentTime;
        static bool m_UseCameraAnimationLoop;
    };
}
