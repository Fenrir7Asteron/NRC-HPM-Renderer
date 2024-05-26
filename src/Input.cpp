#include <engine/util/Input.hpp>
#include <engine/graphics/Window.hpp>

namespace en
{
    GLFWwindow* Input::m_WindowHandle = nullptr;
    glm::vec2 Input::m_MousePos = glm::vec2(0.0f);
    glm::vec2 Input::m_MouseMove = glm::vec2(0.0f);
    float Input::m_CurrentTime = 0.0f;
    bool Input::m_UseCameraAnimationLoop = false;
    glm::vec3 Input::m_camPoints[100] =
    {
        glm::vec3{64.000000f, 0.0f, 0.000000f},
        glm::vec3{63.873711f, 0.0f, 4.018593f},
        glm::vec3{63.495342f, 0.0f, 8.021327f},
        glm::vec3{62.866383f, 0.0f, 11.992405f},
        glm::vec3{61.989323f, 0.0f, 15.916154f},
        glm::vec3{60.867619f, 0.0f, 19.777088f},
        glm::vec3{59.505695f, 0.0f, 23.559971f},
        glm::vec3{57.908932f, 0.0f, 27.249874f},
        glm::vec3{56.083626f, 0.0f, 30.832237f},
        glm::vec3{54.036987f, 0.0f, 34.292919f},
        glm::vec3{51.777084f, 0.0f, 37.618259f},
        glm::vec3{49.312843f, 0.0f, 40.795139f},
        glm::vec3{46.653988f, 0.0f, 43.811020f},
        glm::vec3{43.811008f, 0.0f, 46.653999f},
        glm::vec3{40.795128f, 0.0f, 49.312855f},
        glm::vec3{37.618244f, 0.0f, 51.777096f},
        glm::vec3{34.292904f, 0.0f, 54.036995f},
        glm::vec3{30.832224f, 0.0f, 56.083633f},
        glm::vec3{27.249863f, 0.0f, 57.908936f},
        glm::vec3{23.559958f, 0.0f, 59.505699f},
        glm::vec3{19.777071f, 0.0f, 60.867622f},
        glm::vec3{15.916134f, 0.0f, 61.989326f},
        glm::vec3{11.992384f, 0.0f, 62.866386f},
        glm::vec3{8.021305f,  0.0f, 63.495342f},
        glm::vec3{4.018569f,  0.0f, 63.873711f},
        glm::vec3{-0.000026f, 0.0f, 64.000000f},
        glm::vec3{-4.018620f, 0.0f, 63.873711f},
        glm::vec3{-8.021356f, 0.0f, 63.495338f},
        glm::vec3{-11.99243f, 0.0f, 62.866379f},
        glm::vec3{-15.91618f, 0.0f, 61.989315f},
        glm::vec3{-19.77712f, 0.0f, 60.867607f},
        glm::vec3{-23.56000f, 0.0f, 59.505684f},
        glm::vec3{-27.24990f, 0.0f, 57.908916f},
        glm::vec3{-30.83227f, 0.0f, 56.083611f},
        glm::vec3{-34.29295f, 0.0f, 54.036964f},
        glm::vec3{-37.61829f, 0.0f, 51.777061f},
        glm::vec3{-40.79517f, 0.0f, 49.312820f},
        glm::vec3{-43.81104f, 0.0f, 46.653961f},
        glm::vec3{-46.65402f, 0.0f, 43.810982f},
        glm::vec3{-49.31287f, 0.0f, 40.795097f},
        glm::vec3{-51.77711f, 0.0f, 37.618214f},
        glm::vec3{-54.03701f, 0.0f, 34.292870f},
        glm::vec3{-56.08365f, 0.0f, 30.832188f},
        glm::vec3{-57.90895f, 0.0f, 27.249825f},
        glm::vec3{-59.50571f, 0.0f, 23.559919f},
        glm::vec3{-60.86763f, 0.0f, 19.777033f},
        glm::vec3{-61.98933f, 0.0f, 15.916095f},
        glm::vec3{-62.86639f, 0.0f, 11.992344f},
        glm::vec3{-63.49535f, 0.0f, 8.021264f},
        glm::vec3{-63.87371f, 0.0f, 4.018528f},
        glm::vec3{-64.00000f, 0.0f, -0.000067f},
        glm::vec3{-63.87370f, 0.0f, -4.018661f},
        glm::vec3{-63.49533f, 0.0f, -8.021397f},
        glm::vec3{-62.86637f, 0.0f, -11.99247f},
        glm::vec3{-61.98930f, 0.0f, -15.91622f},
        glm::vec3{-60.86759f, 0.0f, -19.77715f},
        glm::vec3{-59.50566f, 0.0f, -23.56004f},
        glm::vec3{-57.90889f, 0.0f, -27.24994f},
        glm::vec3{-56.08358f, 0.0f, -30.83230f},
        glm::vec3{-54.03694f, 0.0f, -34.29298f},
        glm::vec3{-51.77703f, 0.0f, -37.61832f},
        glm::vec3{-49.31279f, 0.0f, -40.79520f},
        glm::vec3{-46.65393f, 0.0f, -43.81107f},
        glm::vec3{-43.81095f, 0.0f, -46.65405f},
        glm::vec3{-40.79507f, 0.0f, -49.31289f},
        glm::vec3{-37.61819f, 0.0f, -51.77713f},
        glm::vec3{-34.29285f, 0.0f, -54.03702f},
        glm::vec3{-30.83216f, 0.0f, -56.08366f},
        glm::vec3{-27.24980f, 0.0f, -57.90896f},
        glm::vec3{-23.55989f, 0.0f, -59.50572f},
        glm::vec3{-19.77700f, 0.0f, -60.86764f},
        glm::vec3{-15.91607f, 0.0f, -61.98934f},
        glm::vec3{-11.99231f, 0.0f, -62.86640f},
        glm::vec3{-8.021238f, 0.0f, -63.495354f},
        glm::vec3{-4.018503f, 0.0f, -63.873714f},
        glm::vec3{0.000092f,  0.0f, -64.000000f},
        glm::vec3{4.018687f,  0.0f, -63.873703f},
        glm::vec3{8.021421f,  0.0f, -63.495331f},
        glm::vec3{11.992499f, 0.0f, -62.866367f},
        glm::vec3{15.916248f, 0.0f, -61.989296f},
        glm::vec3{19.777184f, 0.0f, -60.867584f},
        glm::vec3{23.560066f, 0.0f, -59.505657f},
        glm::vec3{27.249969f, 0.0f, -57.908886f},
        glm::vec3{30.832327f, 0.0f, -56.083576f},
        glm::vec3{34.293007f, 0.0f, -54.036930f},
        glm::vec3{37.618343f, 0.0f, -51.777023f},
        glm::vec3{40.795219f, 0.0f, -49.312778f},
        glm::vec3{43.811096f, 0.0f, -46.653915f},
        glm::vec3{46.654072f, 0.0f, -43.810932f},
        glm::vec3{49.312920f, 0.0f, -40.795048f},
        glm::vec3{51.777157f, 0.0f, -37.618160f},
        glm::vec3{54.037052f, 0.0f, -34.292816f},
        glm::vec3{56.083687f, 0.0f, -30.832130f},
        glm::vec3{57.908981f, 0.0f, -27.249765f},
        glm::vec3{59.505741f, 0.0f, -23.559856f},
        glm::vec3{60.867657f, 0.0f, -19.776968f},
        glm::vec3{61.989353f, 0.0f, -15.916030f},
        glm::vec3{62.866409f, 0.0f, -11.992278f},
        glm::vec3{63.495358f, 0.0f, -8.021198f},
        glm::vec3{63.873718f, 0.0f, -4.018462f}
    };

    void Input::Init(GLFWwindow* windowHandle)
    {
        float t = 0.0f;
        float pi2 = 3.14159265359f * 2.0f;
        float stepSize = pi2 / 100.0f / 2.0f;
        for (int i = 0; i < 100; ++i)
        {
            float offset = std::max(std::sin(t), 0.2f) * 2.5f;
            m_camPoints[i].x *= offset;
            m_camPoints[i].z *= offset;
            t += stepSize;
        }
        m_WindowHandle = windowHandle;
        Update();
    }

    void Input::Update()
    {
        double xpos;
        double ypos;
        glfwGetCursorPos(m_WindowHandle, &xpos, &ypos);
        glm::vec2 newMousePos = glm::vec2(xpos, ypos);

        m_MouseMove = newMousePos - m_MousePos;
        m_MousePos = newMousePos;
    }

    bool Input::IsKeyPressed(int keycode)
    {
        int state = glfwGetKey(m_WindowHandle, keycode);
        return state == GLFW_PRESS || state == GLFW_REPEAT;
    }

    bool Input::IsKeyReleased(int keycode)
    {
        int state = glfwGetKey(m_WindowHandle, keycode);
        return state == GLFW_RELEASE;
    }

    bool Input::IsMouseButtonPressed(int button)
    {
        int state = glfwGetMouseButton(m_WindowHandle, button);
        return state == GLFW_PRESS;
    }

    glm::vec2 Input::GetMousePos()
    {
        return m_MousePos;
    }

    glm::vec2 Input::GetMouseMove()
    {
        return m_MouseMove;
    }

    void Input::HandleUserCamInput(Camera* cam, float deltaTime)
    {
        bool cameraChanged = false;

        // Mouse input handling
        bool mouseRightPressed = en::Input::IsMouseButtonPressed(MOUSE_BUTTON_RIGHT);
        en::Window::EnableCursor(!mouseRightPressed);
        if (mouseRightPressed && !m_UseCameraAnimationLoop)
        {
            glm::vec2 mouseMove = 0.005f * -en::Input::GetMouseMove();
            cam->RotateViewDir(mouseMove.x, mouseMove.y);

            if (mouseMove != glm::vec2(0.0f, 0.0f))
                cameraChanged = true;
        }

        // Keyboard input handling
        if (!m_UseCameraAnimationLoop)
        {
            glm::vec3 camMove(0.0f, 0.0f, 0.0f);
            float camMoveSpeed = 20.0f * deltaTime;
            bool frontPressed = en::Input::IsKeyPressed(KEY_W);
            bool backPressed = en::Input::IsKeyPressed(KEY_S);
            bool leftPressed = en::Input::IsKeyPressed(KEY_A);
            bool rightPressed = en::Input::IsKeyPressed(KEY_D);
            bool upPressed = en::Input::IsKeyPressed(KEY_SPACE);
            bool downPressed = en::Input::IsKeyPressed(KEY_C);
            if (frontPressed && !backPressed)
                camMove.z = camMoveSpeed;
            else if (backPressed && !frontPressed)
                camMove.z = -camMoveSpeed;
            if (rightPressed && !leftPressed)
                camMove.x = camMoveSpeed;
            else if (leftPressed && !rightPressed)
                camMove.x = -camMoveSpeed;
            if (upPressed && !downPressed)
                camMove.y = camMoveSpeed;
            else if (downPressed && !upPressed)
                camMove.y = -camMoveSpeed;

            if (en::Input::IsKeyPressed(KEY_LEFT_SHIFT))
                camMove *= 10.0f;

            if (camMove != glm::vec3(0.f, 0.f, 0.f))
            {
                cam->Move(camMove);
                cameraChanged = true;
            }
        }
        else
        {
            float stepCount = 100.0f / 3.0f;
            m_CurrentTime += deltaTime * 10.0f;
            while (m_CurrentTime >= stepCount)
                m_CurrentTime -= stepCount;

            int p1 = ((int)(m_CurrentTime) * 3) % 100;
            int p2 = (p1 + 1) % 100;
            int p3 = (p1 + 2) % 100;
            int p4 = (p1 + 3) % 100;

            float t = m_CurrentTime;
            while (t > 1.0f)
            {
                t -= 1.0f;
            }

            float minT = (1 - t);
            float x = std::pow(minT, 3) * m_camPoints[p1].x
                + 3.0f * std::pow(minT, 2) * t * m_camPoints[p2].x
                + 3.0f * minT * std::pow(t, 2) * m_camPoints[p3].x
                + std::pow(t, 3) * m_camPoints[p4].x;

            float y = std::pow(minT, 3) * m_camPoints[p1].y
                + 3.0f * std::pow(minT, 2) * t * m_camPoints[p2].y
                + 3.0f * minT * std::pow(t, 2) * m_camPoints[p3].y
                + std::pow(t, 3) * m_camPoints[p4].y;

            float z = std::pow(minT, 3) * m_camPoints[p1].z
                + 3.0f * std::pow(minT, 2) * t * m_camPoints[p2].z
                + 3.0f * minT * std::pow(t, 2) * m_camPoints[p3].z
                + std::pow(t, 3) * m_camPoints[p4].z;

            glm::vec3 pos = glm::vec3(x, y, z);

            cam->SetPos(pos);
            cam->SetViewDir(-pos);
            cameraChanged = true;
        }

        cam->SetChanged(cameraChanged);
    }
}
