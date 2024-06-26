#pragma once

#include <engine/graphics/common.hpp>
#include <string>
#include <vector>

namespace en
{
	class Window
	{
	public:
		static void Init(uint32_t width, uint32_t height, bool resizable, const std::string& title);
		static void Update();
		static void Shutdown();

		static void WaitForUsableSize();
		static void EnableCursor(bool cursorEnabled);

		static bool IsSupported();
		static GLFWwindow* GetGLFWHandle();
		static uint32_t GetWidth();
		static uint32_t GetHeight();
		static bool IsClosed();
		static std::vector<const char*> GetVulkanExtensions();
		static VkSurfaceKHR CreateVulkanSurface(VkInstance vulkanInstance);

		static void SetTitle(const std::string& title);

	private:
		static bool m_Supported;

		static GLFWwindow* m_GLFWHandle;

		static uint32_t m_Width;
		static uint32_t m_Height;
	};
}
