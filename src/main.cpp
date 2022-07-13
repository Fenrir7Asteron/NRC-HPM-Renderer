#include <engine/util/Log.hpp>
#include <engine/graphics/Window.hpp>
#include <engine/graphics/VulkanAPI.hpp>
#include <engine/graphics/vulkan/Swapchain.hpp>
#include <engine/graphics/vulkan/CommandRecorder.hpp>
#include <engine/graphics/Camera.hpp>
#include <engine/graphics/renderer/ImGuiRenderer.hpp>
#include <imgui.h>
#include <engine/graphics/renderer/DensityPathTracer.hpp>
#include <engine/util/read_file.hpp>
#include <engine/graphics/vulkan/Texture3D.hpp>
#include <engine/objects/VolumeData.hpp>
#include <engine/util/Input.hpp>
#include <engine/util/Time.hpp>
#include <engine/graphics/DirLight.hpp>
#include <engine/compute/Matrix.hpp>
#include <mnist/mnist_reader.hpp>
#include <kompute/Kompute.hpp>
#include <engine/compute/MatmulOp.hpp>
#include <engine/compute/NeuralNetwork.hpp>
#include <engine/compute/SigmoidLayer.hpp>
#include <engine/compute/LinearLayer.hpp>
#include <engine/compute/KomputeManager.hpp>
#include <engine/compute/ReluLayer.hpp>
#include <filesystem>
#include <set>
#include <engine/util/openexr_helper.hpp>
#include <engine/graphics/renderer/NrcHpmRenderer.hpp>
#include <engine/compute/NrcDataset.hpp>
#include <thread>
#include <engine/graphics/NeuralRadianceCache.hpp>
#include <engine/graphics/PointLight.hpp>
#include <engine/graphics/HdrEnvMap.hpp>

en::NrcHpmRenderer* nrcHpmRenderer = nullptr;

std::vector<en::NrcInput> trainInputs;
std::vector<en::NrcTarget> trainTargets;

void RecordSwapchainCommandBuffer(VkCommandBuffer commandBuffer, VkImage image)
{
	uint32_t width = en::Window::GetWidth();
	uint32_t height = en::Window::GetHeight();

	VkCommandBufferBeginInfo beginInfo;
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.pNext = nullptr;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	VkResult result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
	if (result != VK_SUCCESS)
		en::Log::Error("Failed to begin VkCommandBuffer", true);

	en::vk::CommandRecorder::ImageLayoutTransfer(
		commandBuffer,
		image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_ACCESS_NONE_KHR,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT);

	if (nrcHpmRenderer != nullptr && en::ImGuiRenderer::IsInitialized())
	{
		VkImageCopy imageCopy;
		imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopy.srcSubresource.mipLevel = 0;
		imageCopy.srcSubresource.baseArrayLayer = 0;
		imageCopy.srcSubresource.layerCount = 1;
		imageCopy.srcOffset = { 0, 0, 0 };
		imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopy.dstSubresource.mipLevel = 0;
		imageCopy.dstSubresource.baseArrayLayer = 0;
		imageCopy.dstSubresource.layerCount = 1;
		imageCopy.dstOffset = { 0, 0, 0 };
		imageCopy.extent = { width, height, 1 };

		vkCmdCopyImage(
			commandBuffer,
			en::ImGuiRenderer::GetImage(),
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&imageCopy);
	}

	en::vk::CommandRecorder::ImageLayoutTransfer(
		commandBuffer,
		image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	result = vkEndCommandBuffer(commandBuffer);
	if (result != VK_SUCCESS)
		en::Log::Error("Failed to end VkCommandBuffer", true);
}

void SwapchainResizeCallback()
{
	en::Window::WaitForUsableSize();
	vkDeviceWaitIdle(en::VulkanAPI::GetDevice()); // TODO: causes error with multithreaded rendering

	uint32_t width = en::Window::GetWidth();
	uint32_t height = en::Window::GetHeight();
	nrcHpmRenderer->ResizeFrame(width, height);
	en::ImGuiRenderer::Resize(width, height);
	en::ImGuiRenderer::SetBackgroundImageView(nrcHpmRenderer->GetImageView());
}

void RunNrcHpm()
{
	std::string appName("NRC-HPM-Renderer");
	uint32_t width = 800;
	uint32_t height = width;

	// Start engine
	en::Log::Info("Starting " + appName);

	en::Window::Init(width, height, true, appName);
	en::Input::Init(en::Window::GetGLFWHandle());
	en::VulkanAPI::Init(appName);

	// Load data
	auto density3D = en::ReadFileDensity3D("data/cloud_sixteenth", 125, 85, 153);
	en::vk::Texture3D density3DTex(density3D, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
	en::VolumeData volumeData(&density3DTex);

	int hdrWidth, hdrHeight;
	std::vector<float> hdr4fData = en::ReadFileHdr4f("data/image/photostudio_4k.hdr", hdrWidth, hdrHeight);
	en::HdrEnvMap hdrEnvMap(hdr4fData, hdrWidth, hdrHeight);

	// Setup rendering
	en::Camera camera(
		glm::vec3(0.0f, 0.0f, -5.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		static_cast<float>(width) / static_cast<float>(height),
		glm::radians(60.0f),
		0.1f,
		100.0f);

	en::DirLight dirLight(-1.57f, 0.0f, glm::vec3(1.0f), 1.0f);
	en::PointLight pointLight(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 1.0f), 0.1f);

	en::vk::Swapchain swapchain(width, height, RecordSwapchainCommandBuffer, SwapchainResizeCallback);

	en::NeuralRadianceCache nrc(0.001f, 0.001f);

	nrcHpmRenderer = new en::NrcHpmRenderer(
		width, height,
		100, 100,
		camera,
		volumeData,
		dirLight, pointLight, hdrEnvMap,
		nrc);

	en::ImGuiRenderer::Init(width, height);
	en::ImGuiRenderer::SetBackgroundImageView(nrcHpmRenderer->GetImageView());

	swapchain.Resize(width, height); // Rerecords commandbuffers (needs to be called if renderer are created)

	// Main loop
	VkDevice device = en::VulkanAPI::GetDevice();
	VkQueue graphicsQueue = en::VulkanAPI::GetGraphicsQueue();
	VkResult result;
	size_t counter = 0;
	while (!en::Window::IsClosed())
	{
		if (counter % 100 == 0)
		{
			nrc.PrintWeights();
		}

		// Update
		en::Window::Update();
		en::Input::Update();
		en::Time::Update();

		width = en::Window::GetWidth();
		height = en::Window::GetHeight();

		float deltaTime = static_cast<float>(en::Time::GetDeltaTime());
		uint32_t fps = en::Time::GetFps();
		en::Input::HandleUserCamInput(&camera, deltaTime);
		en::Window::SetTitle(appName + " | Delta time: " + std::to_string(deltaTime) + "s | Fps: " + std::to_string(fps));

		// Physics
		camera.SetAspectRatio(width, height);
		camera.UpdateUniformBuffer();

		nrc.ResetStats();
		nrcHpmRenderer->Render(graphicsQueue);
		result = vkQueueWaitIdle(graphicsQueue);
		ASSERT_VULKAN(result);

		if (counter % 100 == 0)
		{
			const en::NeuralRadianceCache::StatsData& nrcStats = nrc.GetStats();
			en::Log::Info("NRC MSE Loss: " + std::to_string(nrcStats.mseLoss));
		}

		en::ImGuiRenderer::StartFrame();

		volumeData.RenderImGui();
		volumeData.Update(camera.HasChanged());
		dirLight.RenderImgui();
		pointLight.RenderImGui();

		en::ImGuiRenderer::EndFrame(graphicsQueue);
		result = vkQueueWaitIdle(graphicsQueue);
		ASSERT_VULKAN(result);

		swapchain.DrawAndPresent();

		counter++;
	}
	result = vkDeviceWaitIdle(device);
	ASSERT_VULKAN(result);

	// End
	density3DTex.Destroy();

	volumeData.Destroy();

	en::ImGuiRenderer::Shutdown();

	nrcHpmRenderer->Destroy();
	delete nrcHpmRenderer;

	nrc.Destroy();

	swapchain.Destroy(true);

	camera.Destroy();

	hdrEnvMap.Destroy();
	pointLight.Destroy();
	dirLight.Destroy();

	en::VulkanAPI::Shutdown();
	en::Window::Shutdown();

	en::Log::Info("Ending " + appName);
}

int main()
{
	RunNrcHpm();
	
	return 0;
}
