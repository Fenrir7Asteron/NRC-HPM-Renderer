#include <engine/cuda_common.hpp>
#include <engine/graphics/NeuralRadianceCache.hpp>
#include <engine/graphics/renderer/NrcHpmRenderer.hpp>
#include <engine/util/Log.hpp>
#include <engine/graphics/vulkan/CommandRecorder.hpp>
#include <glm/gtc/random.hpp>
#include <imgui.h>
#include <chrono>
#include <thread>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

namespace en
{
	VkDescriptorSetLayout NrcHpmRenderer::m_DescSetLayout;
	VkDescriptorPool NrcHpmRenderer::m_DescPool;

#ifdef _WIN64
	PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR = nullptr;

	HANDLE GetSemaphoreHandle(VkDevice device, VkSemaphore vkSemaphore)
	{
		if (fpGetSemaphoreWin32HandleKHR == nullptr)
		{
			fpGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
		}

		VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR = {};
		vulkanSemaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		vulkanSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
		vulkanSemaphoreGetWin32HandleInfoKHR.semaphore = vkSemaphore;
		vulkanSemaphoreGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

		HANDLE handle;
		fpGetSemaphoreWin32HandleKHR(device, &vulkanSemaphoreGetWin32HandleInfoKHR, &handle);
		return handle;
	}
#else
	PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR = nullptr;

	int GetSemaphoreHandle(VkDevice device, VkSemaphore semaphore)
	{
		Log::Info("Retreiving semaphore fd");

		if (fpGetSemaphoreFdKHR == nullptr)
		{
			fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
		}

		VkSemaphoreGetFdInfoKHR vulkanSemaphoreGetFdInfoKHR;
		vulkanSemaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
		vulkanSemaphoreGetFdInfoKHR.pNext = nullptr;
		vulkanSemaphoreGetFdInfoKHR.semaphore = semaphore;
		vulkanSemaphoreGetFdInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

		int fd;
		fpGetSemaphoreFdKHR(device, &vulkanSemaphoreGetFdInfoKHR, &fd);
		
		Log::Info("Semaphore fd has been retreived");
		return fd;
	}
#endif

	void NrcHpmRenderer::Init(VkDevice device)
	{
		// Create desc set layout
		uint32_t bindingIndex = 0;

		VkDescriptorSetLayoutBinding outputImageBinding;
		outputImageBinding.binding = bindingIndex++;
		outputImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		outputImageBinding.descriptorCount = 1;
		outputImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		outputImageBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding primaryRayColorImageBinding;
		primaryRayColorImageBinding.binding = bindingIndex++;
		primaryRayColorImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		primaryRayColorImageBinding.descriptorCount = 1;
		primaryRayColorImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		primaryRayColorImageBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding primaryRayInfoImageBinding;
		primaryRayInfoImageBinding.binding = bindingIndex++;
		primaryRayInfoImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		primaryRayInfoImageBinding.descriptorCount = 1;
		primaryRayInfoImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		primaryRayInfoImageBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcRayOriginImageBinding;
		nrcRayOriginImageBinding.binding = bindingIndex++;
		nrcRayOriginImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		nrcRayOriginImageBinding.descriptorCount = 1;
		nrcRayOriginImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcRayOriginImageBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcRayDirImageBinding;
		nrcRayDirImageBinding.binding = bindingIndex++;
		nrcRayDirImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		nrcRayDirImageBinding.descriptorCount = 1;
		nrcRayDirImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcRayDirImageBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcInferInputBufferBinding;
		nrcInferInputBufferBinding.binding = bindingIndex++;
		nrcInferInputBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcInferInputBufferBinding.descriptorCount = 1;
		nrcInferInputBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcInferInputBufferBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcInferOutputBufferBinding;
		nrcInferOutputBufferBinding.binding = bindingIndex++;
		nrcInferOutputBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcInferOutputBufferBinding.descriptorCount = 1;
		nrcInferOutputBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcInferOutputBufferBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcTrainInputBufferBinding;
		nrcTrainInputBufferBinding.binding = bindingIndex++;
		nrcTrainInputBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcTrainInputBufferBinding.descriptorCount = 1;
		nrcTrainInputBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcTrainInputBufferBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcTrainTargetBufferBinding;
		nrcTrainTargetBufferBinding.binding = bindingIndex++;
		nrcTrainTargetBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcTrainTargetBufferBinding.descriptorCount = 1;
		nrcTrainTargetBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcTrainTargetBufferBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcInferFilterBufferBinding;
		nrcInferFilterBufferBinding.binding = bindingIndex++;
		nrcInferFilterBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcInferFilterBufferBinding.descriptorCount = 1;
		nrcInferFilterBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcInferFilterBufferBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding nrcTrainRingBufferBinding;
		nrcTrainRingBufferBinding.binding = bindingIndex++;
		nrcTrainRingBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcTrainRingBufferBinding.descriptorCount = 1;
		nrcTrainRingBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		nrcTrainRingBufferBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding uniformBufferBinding;
		uniformBufferBinding.binding = bindingIndex++;
		uniformBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformBufferBinding.descriptorCount = 1;
		uniformBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		uniformBufferBinding.pImmutableSamplers = nullptr;

		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			outputImageBinding,
			primaryRayColorImageBinding,
			primaryRayInfoImageBinding,
			nrcRayOriginImageBinding,
			nrcRayDirImageBinding,
			nrcInferInputBufferBinding,
			nrcInferOutputBufferBinding,
			nrcTrainInputBufferBinding,
			nrcTrainTargetBufferBinding,
			nrcInferFilterBufferBinding,
			nrcTrainRingBufferBinding,
			uniformBufferBinding
		};

		VkDescriptorSetLayoutCreateInfo layoutCI;
		layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutCI.pNext = nullptr;
		layoutCI.flags = 0;
		layoutCI.bindingCount = bindings.size();
		layoutCI.pBindings = bindings.data();

		VkResult result = vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_DescSetLayout);
		ASSERT_VULKAN(result);

		// Create desc pool
		VkDescriptorPoolSize storageImagePS;
		storageImagePS.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		storageImagePS.descriptorCount = 5;

		VkDescriptorPoolSize storageBufferPS;
		storageBufferPS.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		storageBufferPS.descriptorCount = 6;

		VkDescriptorPoolSize uniformBufferPS;
		uniformBufferPS.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformBufferPS.descriptorCount = 1;

		std::vector<VkDescriptorPoolSize> poolSizes = { storageImagePS, storageBufferPS, uniformBufferPS };

		VkDescriptorPoolCreateInfo poolCI;
		poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolCI.pNext = nullptr;
		poolCI.flags = 0;
		poolCI.maxSets = 1;
		poolCI.poolSizeCount = poolSizes.size();
		poolCI.pPoolSizes = poolSizes.data();

		result = vkCreateDescriptorPool(device, &poolCI, nullptr, &m_DescPool);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::Shutdown(VkDevice device)
	{
		vkDestroyDescriptorPool(device, m_DescPool, nullptr);
		vkDestroyDescriptorSetLayout(device, m_DescSetLayout, nullptr);
	}

	NrcHpmRenderer::NrcHpmRenderer(
		uint32_t width,
		uint32_t height,
		bool blend,
		const Camera* camera,
		const AppConfig& appConfig,
		const HpmScene& hpmScene,
		NeuralRadianceCache& nrc)
		:
		m_RenderWidth(width),
		m_RenderHeight(height),
		m_TrainSpp(appConfig.trainSpp),
		m_PrimaryRayLength(appConfig.primaryRayLength),
		m_PrimaryRayProb(appConfig.primaryRayProb),
		m_TrainRayLength(appConfig.trainRayLength),
		m_ShouldBlend(blend),
		m_ClearShader("nrc/clear.comp", true),
		m_GenRaysShader("nrc/gen_rays.comp", true),
		m_PrepInferRaysShader("nrc/prep_infer_rays.comp", true),
		m_PrepTrainRaysShader("nrc/prep_train_rays.comp", true),
		m_RenderShader("nrc/render.comp", true),
		m_CommandPool(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, VulkanAPI::GetGraphicsQFI()),
		m_Camera(camera),
		m_HpmScene(hpmScene),
		m_Nrc(nrc),
		m_UniformBuffer(
			sizeof(UniformData), 
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
			{})
	{
		Log::Info("Creating NrcHpmRenderer");


		// Calc train subset
		const uint32_t trainPixelCount = appConfig.trainBatchCount * m_Nrc.GetTrainBatchSize();
		CalcTrainSubset(trainPixelCount);

		// Calc train ring buffer size
		m_TrainRingBufSize = static_cast<uint32_t>(appConfig.trainRingBufSize * static_cast<float>(m_TrainWidth * m_TrainHeight));

		// Init components
		VkDevice device = VulkanAPI::GetDevice();

		CreateSyncObjects(device);

		CreateNrcBuffers();
		m_Nrc.Init(
			m_RenderWidth * m_RenderHeight,
			reinterpret_cast<float*>(m_NrcInferInputDCuBuffer),
			reinterpret_cast<float*>(m_NrcInferOutputDCuBuffer),
			reinterpret_cast<float*>(m_NrcTrainInputDCuBuffer),
			reinterpret_cast<float*>(m_NrcTrainTargetDCuBuffer),
			m_CuExtCudaStartSemaphore, 
			m_CuExtCudaFinishedSemaphore);
		CreateNrcInferFilterBuffer();
		CreateNrcTrainRingBuffer();

		m_CommandPool.AllocateBuffers(3, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		m_PreCudaCommandBuffer = m_CommandPool.GetBuffer(0);
		m_PostCudaCommandBuffer = m_CommandPool.GetBuffer(1);
		m_RandomTasksCmdBuf = m_CommandPool.GetBuffer(2);

		CreatePipelineLayout(device);

		InitSpecializationConstants();

		CreateClearPipeline(device);
		CreateGenRaysPipeline(device);
		CreatePrepInferRaysPipeline(device);
		CreatePrepTrainRaysPipeline(device);
		CreateRenderPipeline(device);

		CreateOutputImage(device);
		CreatePrimaryRayColorImage(device);
		CreatePrimaryRayInfoImage(device);
		CreateNrcRayOriginImage(device);
		CreateNrcRayDirImage(device);

		AllocateAndUpdateDescriptorSet(device);

		CreateQueryPool(device);

		RecordPreCudaCommandBuffer();
		RecordPostCudaCommandBuffer();
	}

	void NrcHpmRenderer::Render(VkQueue queue, bool train)
	{
		// Check if camera moved
		if (m_Camera->HasChanged()) { m_BlendIndex = 1; }

		// Calc blending factor
		m_UniformData.blendFactor = 1.0 / static_cast<float>(m_BlendIndex);

		// Generate random
		m_UniformData.random = glm::linearRand(glm::vec4(0.0f), glm::vec4(1.0f));

		// Update uniform buffer
		m_UniformBuffer.SetData(sizeof(UniformData), &m_UniformData, 0, 0);

		// Update blending index
		if (m_ShouldBlend) { m_BlendIndex++; }
		
		// Pre cuda
		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_PreCudaCommandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &m_CudaStartSemaphore;

		VkResult result = vkQueueSubmit(queue, 1, &submitInfo, m_PreCudaFence);
		ASSERT_VULKAN(result);

		// Sync infer filter
		ASSERT_VULKAN(vkWaitForFences(VulkanAPI::GetDevice(), 1, &m_PreCudaFence, VK_TRUE, UINT64_MAX));
		m_NrcInferFilterStagingBuffer->GetData(m_NrcInferFilterBufferSize, m_NrcInferFilterData, 0, 0);
		ASSERT_VULKAN(vkResetFences(VulkanAPI::GetDevice(), 1, &m_PreCudaFence));

		// Cuda
		m_Nrc.InferAndTrain(reinterpret_cast<uint32_t*>(m_NrcInferFilterData), train);

		// Post cuda
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &m_CudaFinishedSemaphore;
		VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		submitInfo.pWaitDstStageMask = &waitStage;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_PostCudaCommandBuffer;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;

		result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::Destroy()
	{
		VkDevice device = VulkanAPI::GetDevice();

		m_CommandPool.Destroy();

		m_UniformBuffer.Destroy();

		vkDestroyQueryPool(device, m_QueryPool, nullptr);

		vkDestroyImageView(device, m_NrcRayDirImageView, nullptr);
		vkFreeMemory(device, m_NrcRayDirImageMemory, nullptr);
		vkDestroyImage(device, m_NrcRayDirImage, nullptr);

		vkDestroyImageView(device, m_NrcRayOriginImageView, nullptr);
		vkFreeMemory(device, m_NrcRayOriginImageMemory, nullptr);
		vkDestroyImage(device, m_NrcRayOriginImage, nullptr);

		vkDestroyImageView(device, m_PrimaryRayInfoImageView, nullptr);
		vkFreeMemory(device, m_PrimaryRayInfoImageMemory, nullptr);
		vkDestroyImage(device, m_PrimaryRayInfoImage, nullptr);

		vkDestroyImageView(device, m_PrimaryRayColorImageView, nullptr);
		vkFreeMemory(device, m_PrimaryRayColorImageMemory, nullptr);
		vkDestroyImage(device, m_PrimaryRayColorImage, nullptr);

		vkDestroyImageView(device, m_OutputImageView, nullptr);
		vkFreeMemory(device, m_OutputImageMemory, nullptr);
		vkDestroyImage(device, m_OutputImage, nullptr);

		vkDestroyPipeline(device, m_RenderPipeline, nullptr);
		m_RenderShader.Destroy();
		
		vkDestroyPipeline(device, m_PrepTrainRaysPipeline, nullptr);
		m_PrepTrainRaysShader.Destroy();

		vkDestroyPipeline(device, m_PrepInferRaysPipeline, nullptr);
		m_PrepInferRaysShader.Destroy();

		vkDestroyPipeline(device, m_GenRaysPipeline, nullptr);
		m_GenRaysShader.Destroy();

		vkDestroyPipeline(device, m_ClearPipeline, nullptr);
		m_ClearShader.Destroy();

		vkDestroyPipelineLayout(device, m_PipelineLayout, nullptr);
	
		m_NrcTrainRingBuffer->Destroy();
		delete m_NrcTrainRingBuffer;

		m_NrcInferFilterBuffer->Destroy();
		delete m_NrcInferFilterBuffer;
		m_NrcInferFilterStagingBuffer->Destroy();
		delete m_NrcInferFilterStagingBuffer;
		delete m_NrcInferFilterData;

		m_NrcTrainTargetBuffer->Destroy();
		delete m_NrcTrainTargetBuffer;
		ASSERT_CUDA(cudaDestroyExternalMemory(m_NrcTrainTargetCuExtMem));

		m_NrcTrainInputBuffer->Destroy();
		delete m_NrcTrainInputBuffer;
		ASSERT_CUDA(cudaDestroyExternalMemory(m_NrcTrainInputCuExtMem));

		m_NrcInferOutputBuffer->Destroy();
		delete m_NrcInferOutputBuffer;
		ASSERT_CUDA(cudaDestroyExternalMemory(m_NrcInferOutputCuExtMem));

		m_NrcInferInputBuffer->Destroy();
		delete m_NrcInferInputBuffer;
		ASSERT_CUDA(cudaDestroyExternalMemory(m_NrcInferInputCuExtMem));

		vkDestroyFence(device, m_PostCudaFence, nullptr);
		vkDestroyFence(device, m_PreCudaFence, nullptr);

		vkDestroySemaphore(device, m_CudaFinishedSemaphore, nullptr);
		ASSERT_CUDA(cudaDestroyExternalSemaphore(m_CuExtCudaFinishedSemaphore));
		
		vkDestroySemaphore(device, m_CudaStartSemaphore, nullptr);
		ASSERT_CUDA(cudaDestroyExternalSemaphore(m_CuExtCudaStartSemaphore));
	}

	void NrcHpmRenderer::ExportOutputImageToFile(VkQueue queue, const std::string& filePath) const
	{
		const size_t floatCount = m_RenderWidth * m_RenderHeight * 4;
		const size_t bufferSize = floatCount * sizeof(float);

		vk::Buffer vkBuffer(
			bufferSize,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			{});

		VkCommandBufferBeginInfo cmdBufBI;
		cmdBufBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmdBufBI.pNext = nullptr;
		cmdBufBI.flags = 0;
		cmdBufBI.pInheritanceInfo = nullptr;
		ASSERT_VULKAN(vkBeginCommandBuffer(m_RandomTasksCmdBuf, &cmdBufBI));

		VkBufferImageCopy region;
		region.bufferOffset = 0;
		region.bufferRowLength = m_RenderWidth;
		region.bufferImageHeight = m_RenderHeight;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { m_RenderWidth, m_RenderHeight, 1 };

		vkCmdCopyImageToBuffer(m_RandomTasksCmdBuf, m_OutputImage, VK_IMAGE_LAYOUT_GENERAL, vkBuffer.GetVulkanHandle(), 1, &region);

		ASSERT_VULKAN(vkEndCommandBuffer(m_RandomTasksCmdBuf));

		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_RandomTasksCmdBuf;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;

		ASSERT_VULKAN(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		ASSERT_VULKAN(vkQueueWaitIdle(queue));

		std::vector<float> buffer(floatCount);
		vkBuffer.GetData(bufferSize, buffer.data(), 0, 0);
		vkBuffer.Destroy();

		// Store in exr file
		if (TINYEXR_SUCCESS != SaveEXR(buffer.data(), m_RenderWidth, m_RenderHeight, 4, 0, filePath.c_str(), nullptr))
		{
			en::Log::Error("TINYEXR Error", true);
		}
	}

	void NrcHpmRenderer::EvaluateTimestampQueries()
	{
		VkDevice device = VulkanAPI::GetDevice();
		std::vector<uint64_t> queryResults(c_QueryCount);
		ASSERT_VULKAN(vkGetQueryPoolResults(
			device,
			m_QueryPool,
			0,
			c_QueryCount,
			sizeof(uint64_t) * c_QueryCount,
			queryResults.data(),
			sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT));
		vkResetQueryPool(device, m_QueryPool, 0, c_QueryCount);

		for (size_t i = 0; i < c_QueryCount - 1; i++)
		{
			m_TimePeriods[i] = c_TimestampPeriodInMS * static_cast<float>(queryResults[i + 1] - queryResults[i]);
		}
		m_TimePeriods[c_QueryCount - 1] = c_TimestampPeriodInMS * static_cast<float>(queryResults[c_QueryCount - 1] - queryResults[0]);
	}

	void NrcHpmRenderer::RenderImGui()
	{
		ImGui::Begin("NrcHpmRenderer");
		
		size_t periodIndex = 0;
		ImGui::Text("Clear Buffers Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("GenRays Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("PrepInferRays Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("Copy Infer Filter Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("PrepTrainRays Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("Cuda Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("Render Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("Total Time %f ms", m_TimePeriods[periodIndex++]);
		ImGui::Text("Theoretical FPS %f", 1000.0f / m_TimePeriods[c_QueryCount - 1]);

		ImGui::Checkbox("Show NRC", reinterpret_cast<bool*>(&m_UniformData.showNrc));

		ImGui::Checkbox("Blend", &m_ShouldBlend);
		ImGui::Text("Blend index %u", m_BlendIndex);
		if (ImGui::Button("Reset blending")) { m_BlendIndex = 1; }

		ImGui::End();
	}

	VkImage NrcHpmRenderer::GetImage() const
	{
		return m_OutputImage;
	}

	VkImageView NrcHpmRenderer::GetImageView() const
	{
		return m_OutputImageView;
	}

	bool NrcHpmRenderer::IsBlending() const
	{
		return m_ShouldBlend;
	}

	float NrcHpmRenderer::GetFrameTimeMS() const
	{
		return m_TimePeriods[c_QueryCount - 1];
	}

	float NrcHpmRenderer::GetLoss() const
	{
		return m_Nrc.GetLoss();
	}

	float NrcHpmRenderer::GetInferenceTime() const
	{
		return m_Nrc.GetInferenceTime();
	}

	float NrcHpmRenderer::GetTrainTime() const
	{
		return m_Nrc.GetTrainTime();
	}

	void NrcHpmRenderer::SetCamera(VkQueue queue, const Camera* camera)
	{
		// Set members
		m_BlendIndex = 1;
		m_Camera = camera;

		// Clear images
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;
		ASSERT_VULKAN(vkBeginCommandBuffer(m_RandomTasksCmdBuf, &beginInfo));

		VkClearColorValue clearColor = { 0.0f, 0.0f, 0.0f, 0.0f };
		VkImageSubresourceRange subresourceRange;
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.baseArrayLayer = 0;
		subresourceRange.layerCount = 1;
		vkCmdClearColorImage(m_RandomTasksCmdBuf, m_OutputImage, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &subresourceRange);
		vkCmdClearColorImage(m_RandomTasksCmdBuf, m_PrimaryRayColorImage, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &subresourceRange);
		vkCmdClearColorImage(m_RandomTasksCmdBuf, m_PrimaryRayInfoImage, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &subresourceRange);

		ASSERT_VULKAN(vkEndCommandBuffer(m_RandomTasksCmdBuf));

		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_RandomTasksCmdBuf;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;
		ASSERT_VULKAN(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		ASSERT_VULKAN(vkQueueWaitIdle(queue));

		// Rerecord cmd buf
		RecordPreCudaCommandBuffer();
		RecordPostCudaCommandBuffer();
	}

	void NrcHpmRenderer::SetBlend(bool blend)
	{
		m_ShouldBlend = blend;
		m_BlendIndex = 1;
	}

	const NeuralRadianceCache& NrcHpmRenderer::GetNrc()
	{
		return m_Nrc;
	}

	void NrcHpmRenderer::CalcTrainSubset(uint32_t trainPixelCount)
	{
		const uint32_t sqrt = std::sqrt(trainPixelCount);
		for (uint32_t factor = sqrt; factor >= 2; factor--)
		{
			if (trainPixelCount % factor == 0)
			{
				const uint32_t otherFactor = trainPixelCount / factor;
				const uint32_t biggerFactor = std::max(factor, otherFactor);
				const uint32_t smallerFactor = std::min(factor, otherFactor);
				
				if (m_RenderWidth > m_RenderHeight)
				{
					m_TrainWidth = biggerFactor;
					m_TrainHeight = smallerFactor;
				}
				else
				{
					m_TrainWidth = smallerFactor;
					m_TrainHeight = biggerFactor;
				}

				m_TrainXDist = m_RenderWidth / m_TrainWidth;
				m_TrainYDist = m_RenderHeight / m_TrainHeight;

				en::Log::Warn("Train pixel count: " + std::to_string(trainPixelCount) + ", m_TrainXDist: " + std::to_string(m_TrainXDist) + ", m_TrainYDist: " + std::to_string(m_TrainYDist));

				return;
			}
		}

		en::Log::Error("Could not find suitable division of trainPixelCount", true);
	}

	void NrcHpmRenderer::CreateSyncObjects(VkDevice device)
	{
		Log::Info("NrcHpmRenderer: Creating sync objects");

		// Create vk semaphore
		VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
		vulkanExportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
		vulkanExportSemaphoreCreateInfo.pNext = nullptr;
#ifdef _WIN64
		vulkanExportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		vulkanExportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

		VkSemaphoreCreateInfo semaphoreCI;
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphoreCI.pNext = &vulkanExportSemaphoreCreateInfo;
		semaphoreCI.flags = 0;

		VkResult result = vkCreateSemaphore(device, &semaphoreCI, nullptr, &m_CudaStartSemaphore);
		ASSERT_VULKAN(result);

		result = vkCreateSemaphore(device, &semaphoreCI, nullptr, &m_CudaFinishedSemaphore);
		ASSERT_VULKAN(result);

		// Export semaphore to cuda
		cudaExternalSemaphoreHandleDesc extCudaSemaphoreHD{};
#ifdef _WIN64
		extCudaSemaphoreHD.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
#else
		extCudaSemaphoreHD.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
#endif

#ifdef _WIN64
		extCudaSemaphoreHD.handle.win32.handle = GetSemaphoreHandle(device, m_CudaStartSemaphore);
		ASSERT_CUDA(cudaImportExternalSemaphore(&m_CuExtCudaStartSemaphore, &extCudaSemaphoreHD));
		
		extCudaSemaphoreHD.handle.win32.handle = GetSemaphoreHandle(device, m_CudaFinishedSemaphore);
		ASSERT_CUDA(cudaImportExternalSemaphore(&m_CuExtCudaFinishedSemaphore, &extCudaSemaphoreHD));
#else
		extCudaSemaphoreHD.handle.fd = GetSemaphoreHandle(device, m_CudaStartSemaphore);
		ASSERT_CUDA(cudaImportExternalSemaphore(&m_CuExtCudaStartSemaphore, &extCudaSemaphoreHD));
		
		extCudaSemaphoreHD.handle.fd= GetSemaphoreHandle(device, m_CudaFinishedSemaphore);
		ASSERT_CUDA(cudaImportExternalSemaphore(&m_CuExtCudaFinishedSemaphore, &extCudaSemaphoreHD));
#endif
	
		// Create fence
		VkFenceCreateInfo fenceCI{};
		fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCI.pNext = nullptr;
		fenceCI.flags = 0;
		ASSERT_VULKAN(vkCreateFence(device, &fenceCI, nullptr, &m_PreCudaFence));
		ASSERT_VULKAN(vkCreateFence(device, &fenceCI, nullptr, &m_PostCudaFence));
	}

	void NrcHpmRenderer::CreateNrcBuffers()
	{
		Log::Info("NrcHpmRenderer: Creating nrc buffers");

		// Calculate sizes
		const size_t inferCount = m_RenderWidth * m_RenderHeight;
		//inferCount += m_Nrc.GetInferBatchSize() - (inferCount % m_Nrc.GetTrainBatchSize());
		const size_t trainCount = m_TrainWidth * m_TrainHeight;

		m_NrcInferInputBufferSize = inferCount * NeuralRadianceCache::sc_InputCount * sizeof(float);
		m_NrcInferOutputBufferSize = inferCount * NeuralRadianceCache::sc_OutputCount * sizeof(float);
		m_NrcTrainInputBufferSize = trainCount * NeuralRadianceCache::sc_InputCount * sizeof(float);
		m_NrcTrainTargetBufferSize = trainCount * NeuralRadianceCache::sc_OutputCount * sizeof(float);

		// Create buffers
#ifdef _WIN64
		VkExternalMemoryHandleTypeFlagBits extMemType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		VkExternalMemoryHandleTypeFlagBits extMemType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

		Log::Info("Creating VkBuffers");
		m_NrcInferInputBuffer = new vk::Buffer(
			m_NrcInferInputBufferSize, 
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
			{},
			extMemType);

		m_NrcInferOutputBuffer = new vk::Buffer(
			m_NrcInferOutputBufferSize,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			{},
			extMemType);
		
		m_NrcTrainInputBuffer = new vk::Buffer(
			m_NrcTrainInputBufferSize,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			{},
			extMemType);
		
		m_NrcTrainTargetBuffer = new vk::Buffer(
			m_NrcTrainTargetBufferSize,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			{},
			extMemType);

		// Get cuda external memory
		Log::Info("Retreiving cuda external memory");
#ifdef _WIN64
		cudaExternalMemoryHandleDesc cuExtMemHandleDesc{};
		cuExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;

		cuExtMemHandleDesc.handle.win32.handle = m_NrcInferInputBuffer->GetMemoryWin32Handle();
		cuExtMemHandleDesc.size = m_NrcInferInputBufferSize;
		cudaError_t cudaResult = cudaImportExternalMemory(&m_NrcInferInputCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);

		cuExtMemHandleDesc.handle.win32.handle = m_NrcInferOutputBuffer->GetMemoryWin32Handle();
		cuExtMemHandleDesc.size = m_NrcInferOutputBufferSize;
		cudaResult = cudaImportExternalMemory(&m_NrcInferOutputCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);

		cuExtMemHandleDesc.handle.win32.handle = m_NrcTrainInputBuffer->GetMemoryWin32Handle();
		cuExtMemHandleDesc.size = m_NrcTrainInputBufferSize;
		cudaResult = cudaImportExternalMemory(&m_NrcTrainInputCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);

		cuExtMemHandleDesc.handle.win32.handle = m_NrcTrainTargetBuffer->GetMemoryWin32Handle();
		cuExtMemHandleDesc.size = m_NrcTrainTargetBufferSize;
		cudaResult = cudaImportExternalMemory(&m_NrcTrainTargetCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);
#else
		cudaExternalMemoryHandleDesc cuExtMemHandleDesc{};
		cuExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

		cuExtMemHandleDesc.handle.fd = m_NrcInferInputBuffer->GetMemoryFd();
		cuExtMemHandleDesc.size = m_NrcInferInputBufferSize;
		cudaError_t cudaResult = cudaImportExternalMemory(&m_NrcInferInputCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);

		cuExtMemHandleDesc.handle.fd = m_NrcInferOutputBuffer->GetMemoryFd();
		cuExtMemHandleDesc.size = m_NrcInferOutputBufferSize;
		cudaResult = cudaImportExternalMemory(&m_NrcInferOutputCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);

		cuExtMemHandleDesc.handle.fd = m_NrcTrainInputBuffer->GetMemoryFd();
		cuExtMemHandleDesc.size = m_NrcTrainInputBufferSize;
		cudaResult = cudaImportExternalMemory(&m_NrcTrainInputCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);

		cuExtMemHandleDesc.handle.fd = m_NrcTrainTargetBuffer->GetMemoryFd();
		cuExtMemHandleDesc.size = m_NrcTrainTargetBufferSize;
		cudaResult = cudaImportExternalMemory(&m_NrcTrainTargetCuExtMem, &cuExtMemHandleDesc);
		ASSERT_CUDA(cudaResult);
#endif

		// Get cuda buffer
		Log::Info("Mapping cuda external memory");
		cudaExternalMemoryBufferDesc cudaExtBufferDesc{};
		cudaExtBufferDesc.offset = 0;
		cudaExtBufferDesc.flags = 0;

		cudaExtBufferDesc.size = m_NrcInferInputBufferSize;
		cudaResult = cudaExternalMemoryGetMappedBuffer(&m_NrcInferInputDCuBuffer, m_NrcInferInputCuExtMem, &cudaExtBufferDesc);
		ASSERT_CUDA(cudaResult);

		cudaExtBufferDesc.size = m_NrcInferOutputBufferSize;
		cudaResult = cudaExternalMemoryGetMappedBuffer(&m_NrcInferOutputDCuBuffer, m_NrcInferOutputCuExtMem, &cudaExtBufferDesc);
		ASSERT_CUDA(cudaResult);

		cudaExtBufferDesc.size = m_NrcTrainInputBufferSize;
		cudaResult = cudaExternalMemoryGetMappedBuffer(&m_NrcTrainInputDCuBuffer, m_NrcTrainInputCuExtMem, &cudaExtBufferDesc);
		ASSERT_CUDA(cudaResult);

		cudaExtBufferDesc.size = m_NrcTrainTargetBufferSize;
		cudaResult = cudaExternalMemoryGetMappedBuffer(&m_NrcTrainTargetDCuBuffer, m_NrcTrainTargetCuExtMem, &cudaExtBufferDesc);
		ASSERT_CUDA(cudaResult);
	}

	void NrcHpmRenderer::CreateNrcInferFilterBuffer()
	{
		m_NrcInferFilterBufferSize = sizeof(uint32_t) * m_Nrc.GetInferBatchCount();
		m_NrcInferFilterData = malloc(m_NrcInferFilterBufferSize);
		
		m_NrcInferFilterStagingBuffer = new vk::Buffer(
			m_NrcInferFilterBufferSize,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			{});

		m_NrcInferFilterBuffer = new vk::Buffer(
			m_NrcInferFilterBufferSize,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			{});
	}

	void NrcHpmRenderer::CreateNrcTrainRingBuffer()
	{
		const VkDeviceSize headAndTailSize = 2 * sizeof(uint32_t);
		const VkDeviceSize rayInfoSize = 6 * sizeof(float) * m_TrainWidth * m_TrainHeight;
		m_NrcTrainRingBufferSize = headAndTailSize + rayInfoSize;

		m_NrcTrainRingBuffer = new vk::Buffer(
			m_NrcTrainRingBufferSize,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			{});

		vk::Buffer stagingBuffer(
			m_NrcTrainRingBufferSize,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			{});

		void* nrcTrainRingData = malloc(m_NrcTrainRingBufferSize);
		
		uint32_t* indexData = reinterpret_cast<uint32_t*>(nrcTrainRingData);
		indexData[0] = 0;
		indexData[1] = 0;

		float* rayData = reinterpret_cast<float*>(indexData + 2);
		for (size_t ray = 0; ray < m_TrainWidth * m_TrainHeight; ray++)
		{
			rayData[(6 * ray) + 0] = 0.0f;
			rayData[(6 * ray) + 1] = 0.0f;
			rayData[(6 * ray) + 2] = 0.0f;
			
			rayData[(6 * ray) + 3] = 0.0f;
			rayData[(6 * ray) + 4] = 0.0f;
			rayData[(6 * ray) + 5] = 1.0f;
		}

		stagingBuffer.SetData(m_NrcTrainRingBufferSize, nrcTrainRingData, 0, 0);
		vk::Buffer::Copy(&stagingBuffer, m_NrcTrainRingBuffer, m_NrcTrainRingBufferSize);

		stagingBuffer.Destroy();
	}

	void NrcHpmRenderer::CreatePipelineLayout(VkDevice device)
	{
		Log::Info("NrcHpmRenderer: Creating pipeline layout");

		std::vector<VkDescriptorSetLayout> layouts = {
			Camera::GetDescriptorSetLayout(),
			VolumeData::GetDescriptorSetLayout(),
			DirLight::GetDescriptorSetLayout(),
			PointLight::GetDescriptorSetLayout(),
			HdrEnvMap::GetDescriptorSetLayout(),
			m_DescSetLayout };

		VkPipelineLayoutCreateInfo layoutCreateInfo;
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCreateInfo.pNext = nullptr;
		layoutCreateInfo.flags = 0;
		layoutCreateInfo.setLayoutCount = layouts.size();
		layoutCreateInfo.pSetLayouts = layouts.data();
		layoutCreateInfo.pushConstantRangeCount = 0;
		layoutCreateInfo.pPushConstantRanges = nullptr;

		VkResult result = vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &m_PipelineLayout);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::InitSpecializationConstants()
	{
		const VkExtent3D volumeSizeUI = m_HpmScene.GetVolumeData()->GetExtent();
		glm::vec3 volumeSizeF = { volumeSizeUI.width, volumeSizeUI.height, volumeSizeUI.depth };
		volumeSizeF = glm::normalize(volumeSizeF) * 107.5f;

		// Fill struct
		m_SpecData.renderWidth = m_RenderWidth;
		m_SpecData.renderHeight = m_RenderHeight;
		m_SpecData.trainWidth = m_TrainWidth;
		m_SpecData.trainHeight = m_TrainHeight;
		m_SpecData.trainXDist = m_TrainXDist;
		m_SpecData.trainYDist = m_TrainYDist;
		m_SpecData.trainSpp = m_TrainSpp;
		m_SpecData.primaryRayLength = m_PrimaryRayLength;
		m_SpecData.primaryRayProb = m_PrimaryRayProb;
		m_SpecData.trainRingBufSize = m_TrainRingBufSize;
		m_SpecData.trainRayLength = m_TrainRayLength;

		m_SpecData.inferBatchSize = m_Nrc.GetInferBatchSize();
		m_SpecData.trainBatchSize = m_Nrc.GetTrainBatchSize();

		m_SpecData.volumeSizeX = volumeSizeF.x;
		m_SpecData.volumeSizeY = volumeSizeF.y;
		m_SpecData.volumeSizeZ = volumeSizeF.z;
		m_SpecData.volumeDensityFactor = m_HpmScene.GetVolumeData()->GetDensityFactor();
		m_SpecData.volumeG = m_HpmScene.GetVolumeData()->GetG();

		m_SpecData.hdrEnvMapStrength = m_HpmScene.GetHdrEnvMap()->GetStrength();

		// Init map entries
		uint32_t constantID = 0;

		VkSpecializationMapEntry renderWidthEntry;
		renderWidthEntry.constantID = constantID++;
		renderWidthEntry.offset = offsetof(SpecializationData, SpecializationData::renderWidth);
		renderWidthEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry renderHeightEntry;
		renderHeightEntry.constantID = constantID++;
		renderHeightEntry.offset = offsetof(SpecializationData, SpecializationData::renderHeight);
		renderHeightEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry trainWidthEntry;
		trainWidthEntry.constantID = constantID++;
		trainWidthEntry.offset = offsetof(SpecializationData, SpecializationData::trainWidth);
		trainWidthEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry trainHeightEntry;
		trainHeightEntry.constantID = constantID++;
		trainHeightEntry.offset = offsetof(SpecializationData, SpecializationData::trainHeight);
		trainHeightEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry trainXDistEntry{};
		trainXDistEntry.constantID = constantID++;
		trainXDistEntry.offset = offsetof(SpecializationData, SpecializationData::trainXDist);
		trainXDistEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry trainYDistEntry{};
		trainYDistEntry.constantID = constantID++;
		trainYDistEntry.offset = offsetof(SpecializationData, SpecializationData::trainXDist);
		trainYDistEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry trainSppEntry;
		trainSppEntry.constantID = constantID++;
		trainSppEntry.offset = offsetof(SpecializationData, SpecializationData::trainSpp);
		trainSppEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry primaryRayLengthEntry;
		primaryRayLengthEntry.constantID = constantID++;
		primaryRayLengthEntry.offset = offsetof(SpecializationData, SpecializationData::primaryRayLength);
		primaryRayLengthEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry primaryRayProbEntry{};
		primaryRayProbEntry.constantID = constantID++;
		primaryRayProbEntry.offset = offsetof(SpecializationData, SpecializationData::primaryRayProb);
		primaryRayProbEntry.size = sizeof(float);

		VkSpecializationMapEntry trainRingBufSizeEntry = {};
		trainRingBufSizeEntry.constantID = constantID++;
		trainRingBufSizeEntry.offset = offsetof(SpecializationData, SpecializationData::trainRingBufSize);
		trainRingBufSizeEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry trainRayLengthEntry{};
		trainRayLengthEntry.constantID = constantID++;
		trainRayLengthEntry.offset = offsetof(SpecializationData, SpecializationData::trainRayLength);
		trainRayLengthEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry inferBatchSizeEntry{};
		inferBatchSizeEntry.constantID = constantID++;
		inferBatchSizeEntry.offset = offsetof(SpecializationData, SpecializationData::inferBatchSize);
		inferBatchSizeEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry trainBatchSizeEntry;
		trainBatchSizeEntry.constantID = constantID++;
		trainBatchSizeEntry.offset = offsetof(SpecializationData, SpecializationData::trainBatchSize);
		trainBatchSizeEntry.size = sizeof(uint32_t);

		VkSpecializationMapEntry volumeSizeXEntry;
		volumeSizeXEntry.constantID = constantID++;
		volumeSizeXEntry.offset = offsetof(SpecializationData, SpecializationData::volumeSizeX);
		volumeSizeXEntry.size = sizeof(float);

		VkSpecializationMapEntry volumeSizeYEntry;
		volumeSizeYEntry.constantID = constantID++;
		volumeSizeYEntry.offset = offsetof(SpecializationData, SpecializationData::volumeSizeY);
		volumeSizeYEntry.size = sizeof(float);

		VkSpecializationMapEntry volumeSizeZEntry;
		volumeSizeZEntry.constantID = constantID++;
		volumeSizeZEntry.offset = offsetof(SpecializationData, SpecializationData::volumeSizeZ);
		volumeSizeZEntry.size = sizeof(float);

		VkSpecializationMapEntry volumeDensityFactorEntry;
		volumeDensityFactorEntry.constantID = constantID++;
		volumeDensityFactorEntry.offset = offsetof(SpecializationData, SpecializationData::volumeDensityFactor);
		volumeDensityFactorEntry.size = sizeof(float);

		VkSpecializationMapEntry volumeGEntry;
		volumeGEntry.constantID = constantID++;
		volumeGEntry.offset = offsetof(SpecializationData, SpecializationData::volumeG);
		volumeGEntry.size = sizeof(float);

		VkSpecializationMapEntry hdrEnvMapStrengthEntry;
		hdrEnvMapStrengthEntry.constantID = constantID++;
		hdrEnvMapStrengthEntry.offset = offsetof(SpecializationData, SpecializationData::hdrEnvMapStrength);
		hdrEnvMapStrengthEntry.size = sizeof(float);

		m_SpecMapEntries = {
			renderWidthEntry,
			renderHeightEntry,
			trainWidthEntry,
			trainHeightEntry,
			trainXDistEntry,
			trainYDistEntry,
			trainSppEntry,
			primaryRayLengthEntry,
			primaryRayProbEntry,
			trainRingBufSizeEntry,
			inferBatchSizeEntry,
			trainBatchSizeEntry,
			volumeSizeXEntry,
			volumeSizeYEntry,
			volumeSizeZEntry,
			volumeDensityFactorEntry,
			volumeGEntry,
			hdrEnvMapStrengthEntry
		};

		m_SpecInfo.mapEntryCount = m_SpecMapEntries.size();
		m_SpecInfo.pMapEntries = m_SpecMapEntries.data();
		m_SpecInfo.dataSize = sizeof(SpecializationData);
		m_SpecInfo.pData = &m_SpecData;
	}

	void NrcHpmRenderer::CreateClearPipeline(VkDevice device)
	{
		VkPipelineShaderStageCreateInfo shaderStage;
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.pNext = nullptr;
		shaderStage.flags = 0;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage.module = m_ClearShader.GetVulkanModule();
		shaderStage.pName = "main";
		shaderStage.pSpecializationInfo = &m_SpecInfo;

		VkComputePipelineCreateInfo pipelineCI;
		pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineCI.pNext = nullptr;
		pipelineCI.flags = 0;
		pipelineCI.stage = shaderStage;
		pipelineCI.layout = m_PipelineLayout;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
		pipelineCI.basePipelineIndex = 0;

		VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_ClearPipeline);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreateGenRaysPipeline(VkDevice device)
	{
		VkPipelineShaderStageCreateInfo shaderStage;
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.pNext = nullptr;
		shaderStage.flags = 0;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage.module = m_GenRaysShader.GetVulkanModule();
		shaderStage.pName = "main";
		shaderStage.pSpecializationInfo = &m_SpecInfo;

		VkComputePipelineCreateInfo pipelineCI;
		pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineCI.pNext = nullptr;
		pipelineCI.flags = 0;
		pipelineCI.stage = shaderStage;
		pipelineCI.layout = m_PipelineLayout;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
		pipelineCI.basePipelineIndex = 0;

		VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_GenRaysPipeline);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreatePrepInferRaysPipeline(VkDevice device)
	{
		VkPipelineShaderStageCreateInfo shaderStage;
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.pNext = nullptr;
		shaderStage.flags = 0;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage.module = m_PrepInferRaysShader.GetVulkanModule();
		shaderStage.pName = "main";
		shaderStage.pSpecializationInfo = &m_SpecInfo;

		VkComputePipelineCreateInfo pipelineCI;
		pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineCI.pNext = nullptr;
		pipelineCI.flags = 0;
		pipelineCI.stage = shaderStage;
		pipelineCI.layout = m_PipelineLayout;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
		pipelineCI.basePipelineIndex = 0;

		VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_PrepInferRaysPipeline);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreatePrepTrainRaysPipeline(VkDevice device)
	{
		VkPipelineShaderStageCreateInfo shaderStage;
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.pNext = nullptr;
		shaderStage.flags = 0;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage.module = m_PrepTrainRaysShader.GetVulkanModule();
		shaderStage.pName = "main";
		shaderStage.pSpecializationInfo = &m_SpecInfo;

		VkComputePipelineCreateInfo pipelineCI;
		pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineCI.pNext = nullptr;
		pipelineCI.flags = 0;
		pipelineCI.stage = shaderStage;
		pipelineCI.layout = m_PipelineLayout;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
		pipelineCI.basePipelineIndex = 0;

		VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_PrepTrainRaysPipeline);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreateRenderPipeline(VkDevice device)
	{
		VkPipelineShaderStageCreateInfo shaderStage;
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.pNext = nullptr;
		shaderStage.flags = 0;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage.module = m_RenderShader.GetVulkanModule();
		shaderStage.pName = "main";
		shaderStage.pSpecializationInfo = &m_SpecInfo;

		VkComputePipelineCreateInfo pipelineCI;
		pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineCI.pNext = nullptr;
		pipelineCI.flags = 0;
		pipelineCI.stage = shaderStage;
		pipelineCI.layout = m_PipelineLayout;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
		pipelineCI.basePipelineIndex = 0;

		VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_RenderPipeline);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreateOutputImage(VkDevice device)
	{
		VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

		// Create Image
		VkImageCreateInfo imageCI;
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.pNext = nullptr;
		imageCI.flags = 0;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = { m_RenderWidth, m_RenderHeight, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCI.queueFamilyIndexCount = 0;
		imageCI.pQueueFamilyIndices = nullptr;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

		VkResult result = vkCreateImage(device, &imageCI, nullptr, &m_OutputImage);
		ASSERT_VULKAN(result);

		// Image Memory
		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(device, m_OutputImage, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo;
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.pNext = nullptr;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = VulkanAPI::FindMemoryType(
			memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		result = vkAllocateMemory(device, &allocateInfo, nullptr, &m_OutputImageMemory);
		ASSERT_VULKAN(result);

		result = vkBindImageMemory(device, m_OutputImage, m_OutputImageMemory, 0);
		ASSERT_VULKAN(result);

		// Create image view
		VkImageViewCreateInfo imageViewCI;
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.pNext = nullptr;
		imageViewCI.flags = 0;
		imageViewCI.image = m_OutputImage;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.format = format;
		imageViewCI.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;

		result = vkCreateImageView(device, &imageViewCI, nullptr, &m_OutputImageView);
		ASSERT_VULKAN(result);

		// Change image layout
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		result = vkBeginCommandBuffer(m_RandomTasksCmdBuf, &beginInfo);
		ASSERT_VULKAN(result);

		vk::CommandRecorder::ImageLayoutTransfer(
			m_RandomTasksCmdBuf,
			m_OutputImage,
			VK_IMAGE_LAYOUT_PREINITIALIZED,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_NONE,
			VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		result = vkEndCommandBuffer(m_RandomTasksCmdBuf);
		ASSERT_VULKAN(result);

		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_RandomTasksCmdBuf;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;

		VkQueue queue = VulkanAPI::GetGraphicsQueue();
		result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		ASSERT_VULKAN(result);
		result = vkQueueWaitIdle(queue);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreatePrimaryRayColorImage(VkDevice device)
	{
		VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

		// Create Image
		VkImageCreateInfo imageCI;
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.pNext = nullptr;
		imageCI.flags = 0;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = { m_RenderWidth, m_RenderHeight, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCI.queueFamilyIndexCount = 0;
		imageCI.pQueueFamilyIndices = nullptr;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

		VkResult result = vkCreateImage(device, &imageCI, nullptr, &m_PrimaryRayColorImage);
		ASSERT_VULKAN(result);

		// Image Memory
		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(device, m_PrimaryRayColorImage, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo;
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.pNext = nullptr;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = VulkanAPI::FindMemoryType(
			memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		result = vkAllocateMemory(device, &allocateInfo, nullptr, &m_PrimaryRayColorImageMemory);
		ASSERT_VULKAN(result);

		result = vkBindImageMemory(device, m_PrimaryRayColorImage, m_PrimaryRayColorImageMemory, 0);
		ASSERT_VULKAN(result);

		// Create image view
		VkImageViewCreateInfo imageViewCI;
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.pNext = nullptr;
		imageViewCI.flags = 0;
		imageViewCI.image = m_PrimaryRayColorImage;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.format = format;
		imageViewCI.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;

		result = vkCreateImageView(device, &imageViewCI, nullptr, &m_PrimaryRayColorImageView);
		ASSERT_VULKAN(result);

		// Change image layout
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		result = vkBeginCommandBuffer(m_RandomTasksCmdBuf, &beginInfo);
		ASSERT_VULKAN(result);

		vk::CommandRecorder::ImageLayoutTransfer(
			m_RandomTasksCmdBuf,
			m_PrimaryRayColorImage,
			VK_IMAGE_LAYOUT_PREINITIALIZED,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_NONE,
			VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		result = vkEndCommandBuffer(m_RandomTasksCmdBuf);
		ASSERT_VULKAN(result);

		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_RandomTasksCmdBuf;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;

		VkQueue queue = VulkanAPI::GetGraphicsQueue();
		result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		ASSERT_VULKAN(result);
		result = vkQueueWaitIdle(queue);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreatePrimaryRayInfoImage(VkDevice device)
	{
		VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

		// Create Image
		VkImageCreateInfo imageCI;
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.pNext = nullptr;
		imageCI.flags = 0;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = { m_RenderWidth, m_RenderHeight, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCI.queueFamilyIndexCount = 0;
		imageCI.pQueueFamilyIndices = nullptr;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

		VkResult result = vkCreateImage(device, &imageCI, nullptr, &m_PrimaryRayInfoImage);
		ASSERT_VULKAN(result);

		// Image Memory
		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(device, m_PrimaryRayInfoImage, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo;
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.pNext = nullptr;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = VulkanAPI::FindMemoryType(
			memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		result = vkAllocateMemory(device, &allocateInfo, nullptr, &m_PrimaryRayInfoImageMemory);
		ASSERT_VULKAN(result);

		result = vkBindImageMemory(device, m_PrimaryRayInfoImage, m_PrimaryRayInfoImageMemory, 0);
		ASSERT_VULKAN(result);

		// Create image view
		VkImageViewCreateInfo imageViewCI;
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.pNext = nullptr;
		imageViewCI.flags = 0;
		imageViewCI.image = m_PrimaryRayInfoImage;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.format = format;
		imageViewCI.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;

		result = vkCreateImageView(device, &imageViewCI, nullptr, &m_PrimaryRayInfoImageView);
		ASSERT_VULKAN(result);

		// Change image layout
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		result = vkBeginCommandBuffer(m_RandomTasksCmdBuf, &beginInfo);
		ASSERT_VULKAN(result);

		vk::CommandRecorder::ImageLayoutTransfer(
			m_RandomTasksCmdBuf,
			m_PrimaryRayInfoImage,
			VK_IMAGE_LAYOUT_PREINITIALIZED,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_NONE,
			VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		result = vkEndCommandBuffer(m_RandomTasksCmdBuf);
		ASSERT_VULKAN(result);

		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_RandomTasksCmdBuf;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;

		VkQueue queue = VulkanAPI::GetGraphicsQueue();
		result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		ASSERT_VULKAN(result);
		result = vkQueueWaitIdle(queue);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreateNrcRayOriginImage(VkDevice device)
	{
		VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

		// Create Image
		VkImageCreateInfo imageCI;
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.pNext = nullptr;
		imageCI.flags = 0;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = { m_RenderWidth, m_RenderHeight, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT;
		imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCI.queueFamilyIndexCount = 0;
		imageCI.pQueueFamilyIndices = nullptr;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

		VkResult result = vkCreateImage(device, &imageCI, nullptr, &m_NrcRayOriginImage);
		ASSERT_VULKAN(result);

		// Image Memory
		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(device, m_NrcRayOriginImage, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo;
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.pNext = nullptr;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = VulkanAPI::FindMemoryType(
			memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		result = vkAllocateMemory(device, &allocateInfo, nullptr, &m_NrcRayOriginImageMemory);
		ASSERT_VULKAN(result);

		result = vkBindImageMemory(device, m_NrcRayOriginImage, m_NrcRayOriginImageMemory, 0);
		ASSERT_VULKAN(result);

		// Create image view
		VkImageViewCreateInfo imageViewCI;
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.pNext = nullptr;
		imageViewCI.flags = 0;
		imageViewCI.image = m_NrcRayOriginImage;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.format = format;
		imageViewCI.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;

		result = vkCreateImageView(device, &imageViewCI, nullptr, &m_NrcRayOriginImageView);
		ASSERT_VULKAN(result);

		// Change image layout
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		result = vkBeginCommandBuffer(m_RandomTasksCmdBuf, &beginInfo);
		ASSERT_VULKAN(result);

		vk::CommandRecorder::ImageLayoutTransfer(
			m_RandomTasksCmdBuf,
			m_NrcRayOriginImage,
			VK_IMAGE_LAYOUT_PREINITIALIZED,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_NONE,
			VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		result = vkEndCommandBuffer(m_RandomTasksCmdBuf);
		ASSERT_VULKAN(result);

		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_RandomTasksCmdBuf;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;

		VkQueue queue = VulkanAPI::GetGraphicsQueue();
		result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		ASSERT_VULKAN(result);
		result = vkQueueWaitIdle(queue);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::CreateNrcRayDirImage(VkDevice device)
	{
		VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

		// Create Image
		VkImageCreateInfo imageCI;
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.pNext = nullptr;
		imageCI.flags = 0;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent = { m_RenderWidth, m_RenderHeight, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_STORAGE_BIT;
		imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCI.queueFamilyIndexCount = 0;
		imageCI.pQueueFamilyIndices = nullptr;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

		VkResult result = vkCreateImage(device, &imageCI, nullptr, &m_NrcRayDirImage);
		ASSERT_VULKAN(result);

		// Image Memory
		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(device, m_NrcRayDirImage, &memoryRequirements);

		VkMemoryAllocateInfo allocateInfo;
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.pNext = nullptr;
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = VulkanAPI::FindMemoryType(
			memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		result = vkAllocateMemory(device, &allocateInfo, nullptr, &m_NrcRayDirImageMemory);
		ASSERT_VULKAN(result);

		result = vkBindImageMemory(device, m_NrcRayDirImage, m_NrcRayDirImageMemory, 0);
		ASSERT_VULKAN(result);

		// Create image view
		VkImageViewCreateInfo imageViewCI;
		imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCI.pNext = nullptr;
		imageViewCI.flags = 0;
		imageViewCI.image = m_NrcRayDirImage;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.format = format;
		imageViewCI.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;

		result = vkCreateImageView(device, &imageViewCI, nullptr, &m_NrcRayDirImageView);
		ASSERT_VULKAN(result);

		// Change image layout
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		result = vkBeginCommandBuffer(m_RandomTasksCmdBuf, &beginInfo);
		ASSERT_VULKAN(result);

		vk::CommandRecorder::ImageLayoutTransfer(
			m_RandomTasksCmdBuf,
			m_NrcRayDirImage,
			VK_IMAGE_LAYOUT_PREINITIALIZED,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_ACCESS_NONE,
			VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		result = vkEndCommandBuffer(m_RandomTasksCmdBuf);
		ASSERT_VULKAN(result);

		VkSubmitInfo submitInfo;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = nullptr;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_RandomTasksCmdBuf;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;

		VkQueue queue = VulkanAPI::GetGraphicsQueue();
		result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		ASSERT_VULKAN(result);
		result = vkQueueWaitIdle(queue);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::AllocateAndUpdateDescriptorSet(VkDevice device)
	{
		// Allocate
		VkDescriptorSetAllocateInfo descSetAI;
		descSetAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descSetAI.pNext = nullptr;
		descSetAI.descriptorPool = m_DescPool;
		descSetAI.descriptorSetCount = 1;
		descSetAI.pSetLayouts = &m_DescSetLayout;

		VkResult result = vkAllocateDescriptorSets(device, &descSetAI, &m_DescSet);
		ASSERT_VULKAN(result);

		// Write
		// Storage image writes
		uint32_t bindingIndex = 0;

		VkDescriptorImageInfo outputImageInfo;
		outputImageInfo.sampler = VK_NULL_HANDLE;
		outputImageInfo.imageView = m_OutputImageView;
		outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet outputImageWrite;
		outputImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		outputImageWrite.pNext = nullptr;
		outputImageWrite.dstSet = m_DescSet;
		outputImageWrite.dstBinding = bindingIndex++;
		outputImageWrite.dstArrayElement = 0;
		outputImageWrite.descriptorCount = 1;
		outputImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		outputImageWrite.pImageInfo = &outputImageInfo;
		outputImageWrite.pBufferInfo = nullptr;
		outputImageWrite.pTexelBufferView = nullptr;

		VkDescriptorImageInfo primaryRayColorImageInfo;
		primaryRayColorImageInfo.sampler = VK_NULL_HANDLE;
		primaryRayColorImageInfo.imageView = m_PrimaryRayColorImageView;
		primaryRayColorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet primaryRayColorImageWrite;
		primaryRayColorImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		primaryRayColorImageWrite.pNext = nullptr;
		primaryRayColorImageWrite.dstSet = m_DescSet;
		primaryRayColorImageWrite.dstBinding = bindingIndex++;
		primaryRayColorImageWrite.dstArrayElement = 0;
		primaryRayColorImageWrite.descriptorCount = 1;
		primaryRayColorImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		primaryRayColorImageWrite.pImageInfo = &primaryRayColorImageInfo;
		primaryRayColorImageWrite.pBufferInfo = nullptr;
		primaryRayColorImageWrite.pTexelBufferView = nullptr;

		VkDescriptorImageInfo primaryRayInfoImageInfo;
		primaryRayInfoImageInfo.sampler = VK_NULL_HANDLE;
		primaryRayInfoImageInfo.imageView = m_PrimaryRayInfoImageView;
		primaryRayInfoImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet primaryRayInfoImageWrite;
		primaryRayInfoImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		primaryRayInfoImageWrite.pNext = nullptr;
		primaryRayInfoImageWrite.dstSet = m_DescSet;
		primaryRayInfoImageWrite.dstBinding = bindingIndex++;
		primaryRayInfoImageWrite.dstArrayElement = 0;
		primaryRayInfoImageWrite.descriptorCount = 1;
		primaryRayInfoImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		primaryRayInfoImageWrite.pImageInfo = &primaryRayInfoImageInfo;
		primaryRayInfoImageWrite.pBufferInfo = nullptr;
		primaryRayInfoImageWrite.pTexelBufferView = nullptr;

		VkDescriptorImageInfo nrcRayOriginImageInfo;
		nrcRayOriginImageInfo.sampler = VK_NULL_HANDLE;
		nrcRayOriginImageInfo.imageView = m_NrcRayOriginImageView;
		nrcRayOriginImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet nrcRayOriginImageWrite;
		nrcRayOriginImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcRayOriginImageWrite.pNext = nullptr;
		nrcRayOriginImageWrite.dstSet = m_DescSet;
		nrcRayOriginImageWrite.dstBinding = bindingIndex++;
		nrcRayOriginImageWrite.dstArrayElement = 0;
		nrcRayOriginImageWrite.descriptorCount = 1;
		nrcRayOriginImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		nrcRayOriginImageWrite.pImageInfo = &nrcRayOriginImageInfo;
		nrcRayOriginImageWrite.pBufferInfo = nullptr;
		nrcRayOriginImageWrite.pTexelBufferView = nullptr;

		VkDescriptorImageInfo nrcRayDirImageInfo;
		nrcRayDirImageInfo.sampler = VK_NULL_HANDLE;
		nrcRayDirImageInfo.imageView = m_NrcRayDirImageView;
		nrcRayDirImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet nrcRayDirImageWrite;
		nrcRayDirImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcRayDirImageWrite.pNext = nullptr;
		nrcRayDirImageWrite.dstSet = m_DescSet;
		nrcRayDirImageWrite.dstBinding = bindingIndex++;
		nrcRayDirImageWrite.dstArrayElement = 0;
		nrcRayDirImageWrite.descriptorCount = 1;
		nrcRayDirImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		nrcRayDirImageWrite.pImageInfo = &nrcRayDirImageInfo;
		nrcRayDirImageWrite.pBufferInfo = nullptr;
		nrcRayDirImageWrite.pTexelBufferView = nullptr;

		// Storage buffer writes
		VkDescriptorBufferInfo nrcInferInputBufferInfo;
		nrcInferInputBufferInfo.buffer = m_NrcInferInputBuffer->GetVulkanHandle();
		nrcInferInputBufferInfo.offset = 0;
		nrcInferInputBufferInfo.range = m_NrcInferInputBufferSize;

		VkWriteDescriptorSet nrcInferInputBufferWrite;
		nrcInferInputBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcInferInputBufferWrite.pNext = nullptr;
		nrcInferInputBufferWrite.dstSet = m_DescSet;
		nrcInferInputBufferWrite.dstBinding = bindingIndex++;
		nrcInferInputBufferWrite.dstArrayElement = 0;
		nrcInferInputBufferWrite.descriptorCount = 1;
		nrcInferInputBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcInferInputBufferWrite.pImageInfo = nullptr;
		nrcInferInputBufferWrite.pBufferInfo = &nrcInferInputBufferInfo;
		nrcInferInputBufferWrite.pTexelBufferView = nullptr;

		VkDescriptorBufferInfo nrcInferOutputBufferInfo;
		nrcInferOutputBufferInfo.buffer = m_NrcInferOutputBuffer->GetVulkanHandle();
		nrcInferOutputBufferInfo.offset = 0;
		nrcInferOutputBufferInfo.range = m_NrcInferOutputBufferSize;

		VkWriteDescriptorSet nrcInferOutputBufferWrite;
		nrcInferOutputBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcInferOutputBufferWrite.pNext = nullptr;
		nrcInferOutputBufferWrite.dstSet = m_DescSet;
		nrcInferOutputBufferWrite.dstBinding = bindingIndex++;
		nrcInferOutputBufferWrite.dstArrayElement = 0;
		nrcInferOutputBufferWrite.descriptorCount = 1;
		nrcInferOutputBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcInferOutputBufferWrite.pImageInfo = nullptr;
		nrcInferOutputBufferWrite.pBufferInfo = &nrcInferOutputBufferInfo;
		nrcInferOutputBufferWrite.pTexelBufferView = nullptr;

		VkDescriptorBufferInfo nrcTrainInputBufferInfo;
		nrcTrainInputBufferInfo.buffer = m_NrcTrainInputBuffer->GetVulkanHandle();
		nrcTrainInputBufferInfo.offset = 0;
		nrcTrainInputBufferInfo.range = m_NrcTrainInputBufferSize;

		VkWriteDescriptorSet nrcTrainInputBufferWrite;
		nrcTrainInputBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcTrainInputBufferWrite.pNext = nullptr;
		nrcTrainInputBufferWrite.dstSet = m_DescSet;
		nrcTrainInputBufferWrite.dstBinding = bindingIndex++;
		nrcTrainInputBufferWrite.dstArrayElement = 0;
		nrcTrainInputBufferWrite.descriptorCount = 1;
		nrcTrainInputBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcTrainInputBufferWrite.pImageInfo = nullptr;
		nrcTrainInputBufferWrite.pBufferInfo = &nrcTrainInputBufferInfo;
		nrcTrainInputBufferWrite.pTexelBufferView = nullptr;

		VkDescriptorBufferInfo nrcTrainTargetBufferInfo;
		nrcTrainTargetBufferInfo.buffer = m_NrcTrainTargetBuffer->GetVulkanHandle();
		nrcTrainTargetBufferInfo.offset = 0;
		nrcTrainTargetBufferInfo.range = m_NrcTrainTargetBufferSize;

		VkWriteDescriptorSet nrcTrainTargetBufferWrite;
		nrcTrainTargetBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcTrainTargetBufferWrite.pNext = nullptr;
		nrcTrainTargetBufferWrite.dstSet = m_DescSet;
		nrcTrainTargetBufferWrite.dstBinding = bindingIndex++;
		nrcTrainTargetBufferWrite.dstArrayElement = 0;
		nrcTrainTargetBufferWrite.descriptorCount = 1;
		nrcTrainTargetBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcTrainTargetBufferWrite.pImageInfo = nullptr;
		nrcTrainTargetBufferWrite.pBufferInfo = &nrcTrainTargetBufferInfo;
		nrcTrainTargetBufferWrite.pTexelBufferView = nullptr;

		VkDescriptorBufferInfo nrcInferFilterBufferInfo;
		nrcInferFilterBufferInfo.buffer = m_NrcInferFilterBuffer->GetVulkanHandle();
		nrcInferFilterBufferInfo.offset = 0;
		nrcInferFilterBufferInfo.range = m_NrcInferFilterBufferSize;

		VkWriteDescriptorSet nrcInferFilterBufferWrite;
		nrcInferFilterBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcInferFilterBufferWrite.pNext = nullptr;
		nrcInferFilterBufferWrite.dstSet = m_DescSet;
		nrcInferFilterBufferWrite.dstBinding = bindingIndex++;
		nrcInferFilterBufferWrite.dstArrayElement = 0;
		nrcInferFilterBufferWrite.descriptorCount = 1;
		nrcInferFilterBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcInferFilterBufferWrite.pImageInfo = nullptr;
		nrcInferFilterBufferWrite.pBufferInfo = &nrcInferFilterBufferInfo;
		nrcInferFilterBufferWrite.pTexelBufferView = nullptr;

		VkDescriptorBufferInfo nrcTrainRingBufferInfo;
		nrcTrainRingBufferInfo.buffer = m_NrcTrainRingBuffer->GetVulkanHandle();
		nrcTrainRingBufferInfo.offset = 0;
		nrcTrainRingBufferInfo.range = m_NrcTrainRingBufferSize;

		VkWriteDescriptorSet nrcTrainRingBufferWrite;
		nrcTrainRingBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		nrcTrainRingBufferWrite.pNext = nullptr;
		nrcTrainRingBufferWrite.dstSet = m_DescSet;
		nrcTrainRingBufferWrite.dstBinding = bindingIndex++;
		nrcTrainRingBufferWrite.dstArrayElement = 0;
		nrcTrainRingBufferWrite.descriptorCount = 1;
		nrcTrainRingBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		nrcTrainRingBufferWrite.pImageInfo = nullptr;
		nrcTrainRingBufferWrite.pBufferInfo = &nrcTrainRingBufferInfo;
		nrcTrainRingBufferWrite.pTexelBufferView = nullptr;

		// Uniform buffer write
		VkDescriptorBufferInfo uniformBufferInfo;
		uniformBufferInfo.buffer = m_UniformBuffer.GetVulkanHandle();
		uniformBufferInfo.offset = 0;
		uniformBufferInfo.range = sizeof(UniformData);

		VkWriteDescriptorSet uniformBufferWrite;
		uniformBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		uniformBufferWrite.pNext = nullptr;
		uniformBufferWrite.dstSet = m_DescSet;
		uniformBufferWrite.dstBinding = bindingIndex++;
		uniformBufferWrite.dstArrayElement = 0;
		uniformBufferWrite.descriptorCount = 1;
		uniformBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformBufferWrite.pImageInfo = nullptr;
		uniformBufferWrite.pBufferInfo = &uniformBufferInfo;
		uniformBufferWrite.pTexelBufferView = nullptr;

		// Write writes
		std::vector<VkWriteDescriptorSet> writes = { 
			outputImageWrite,
			primaryRayColorImageWrite,
			primaryRayInfoImageWrite,
			nrcRayOriginImageWrite,
			nrcRayDirImageWrite,
			nrcInferInputBufferWrite,
			nrcInferOutputBufferWrite,
			nrcTrainInputBufferWrite,
			nrcTrainTargetBufferWrite,
			nrcInferFilterBufferWrite,
			nrcTrainRingBufferWrite,
			uniformBufferWrite
		};

		vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
	}

	void NrcHpmRenderer::CreateQueryPool(VkDevice device)
	{
		VkQueryPoolCreateInfo queryPoolCI;
		queryPoolCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolCI.pNext = nullptr;
		queryPoolCI.flags = 0;
		queryPoolCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
		queryPoolCI.queryCount = c_QueryCount;
		queryPoolCI.pipelineStatistics = 0;

		ASSERT_VULKAN(vkCreateQueryPool(device, &queryPoolCI, nullptr, &m_QueryPool));
	}

	void NrcHpmRenderer::RecordPreCudaCommandBuffer()
	{
		m_QueryIndex = 0;

		// Begin
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;
		
		VkResult result = vkBeginCommandBuffer(m_PreCudaCommandBuffer, &beginInfo);
		ASSERT_VULKAN(result);
		
		// Collect descriptor sets
		std::vector<VkDescriptorSet> descSets = { m_Camera->GetDescriptorSet() };
		const std::vector<VkDescriptorSet>& hpmSceneDescSets = m_HpmScene.GetDescriptorSets();
		descSets.insert(descSets.end(), hpmSceneDescSets.begin(), hpmSceneDescSets.end());
		descSets.push_back(m_DescSet);

		// Bind descriptor sets
		vkCmdBindDescriptorSets(
			m_PreCudaCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout,
			0, descSets.size(), descSets.data(),
			0, nullptr);

		// Reset query pool
		vkCmdResetQueryPool(m_PreCudaCommandBuffer, m_QueryPool, 0, c_QueryCount);
		
		// Timestamp
		vkCmdWriteTimestamp(m_PreCudaCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_QueryPool, m_QueryIndex++);

		// Clear buffers
		vkCmdFillBuffer(m_PreCudaCommandBuffer, m_NrcInferInputBuffer->GetVulkanHandle(), 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(m_PreCudaCommandBuffer, m_NrcInferOutputBuffer->GetVulkanHandle(), 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(m_PreCudaCommandBuffer, m_NrcTrainInputBuffer->GetVulkanHandle(), 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(m_PreCudaCommandBuffer, m_NrcTrainTargetBuffer->GetVulkanHandle(), 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(m_PreCudaCommandBuffer, m_NrcInferFilterBuffer->GetVulkanHandle(), 0, VK_WHOLE_SIZE, 0);

		// Clear using shader
		vkCmdBindPipeline(m_PreCudaCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_ClearPipeline);
		vkCmdDispatch(m_PreCudaCommandBuffer, 1, 1, 1);

		// Timestamp
		vkCmdWriteTimestamp(m_PreCudaCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_QueryPool, m_QueryIndex++);

		// Gen rays pipeline
		vkCmdBindPipeline(m_PreCudaCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_GenRaysPipeline);
		vkCmdDispatch(m_PreCudaCommandBuffer, m_RenderWidth / 32, m_RenderHeight, 1);

		// Timestamp
		vkCmdWriteTimestamp(m_PreCudaCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_QueryPool, m_QueryIndex++);

		// Prep infer rays
		vkCmdBindPipeline(m_PreCudaCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_PrepInferRaysPipeline);
		vkCmdDispatch(m_PreCudaCommandBuffer, m_RenderWidth / 32, m_RenderHeight, 1);

		// Timestamp
		vkCmdWriteTimestamp(m_PreCudaCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_QueryPool, m_QueryIndex++);

		// Copy nrc infer filter buffer to host
		VkBufferCopy nrcInferFilterCopy;
		nrcInferFilterCopy.srcOffset = 0;
		nrcInferFilterCopy.dstOffset = 0;
		nrcInferFilterCopy.size = m_NrcInferFilterBufferSize;
		vkCmdCopyBuffer(
			m_PreCudaCommandBuffer, 
			m_NrcInferFilterBuffer->GetVulkanHandle(), 
			m_NrcInferFilterStagingBuffer->GetVulkanHandle(), 
			1, 
			&nrcInferFilterCopy);

		// Timestamp
		vkCmdWriteTimestamp(m_PreCudaCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_QueryPool, m_QueryIndex++);

		// Prep train rays
		vkCmdBindPipeline(m_PreCudaCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_PrepTrainRaysPipeline);
		vkCmdDispatch(m_PreCudaCommandBuffer, m_TrainWidth / 32, m_TrainHeight, 1);

		// Timestamp
		vkCmdWriteTimestamp(m_PreCudaCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_QueryPool, m_QueryIndex++);
		
		// End
		result = vkEndCommandBuffer(m_PreCudaCommandBuffer);
		ASSERT_VULKAN(result);
	}

	void NrcHpmRenderer::RecordPostCudaCommandBuffer()
	{
		// Begin command buffer
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		VkResult result = vkBeginCommandBuffer(m_PostCudaCommandBuffer, &beginInfo);
		ASSERT_VULKAN(result);
		
		// Collect descriptor sets
		std::vector<VkDescriptorSet> descSets = { m_Camera->GetDescriptorSet() };
		const std::vector<VkDescriptorSet>& hpmSceneDescSets = m_HpmScene.GetDescriptorSets();
		descSets.insert(descSets.end(), hpmSceneDescSets.begin(), hpmSceneDescSets.end());
		descSets.push_back(m_DescSet);

		// Bind descriptor sets
		vkCmdBindDescriptorSets(
			m_PostCudaCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout,
			0, descSets.size(), descSets.data(),
			0, nullptr);

		// Timestamp
		vkCmdWriteTimestamp(m_PostCudaCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_QueryPool, m_QueryIndex++);

		// Render pipeline
		vkCmdBindPipeline(m_PostCudaCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_RenderPipeline);
		vkCmdDispatch(m_PostCudaCommandBuffer, m_RenderWidth / 32, m_RenderHeight, 1);

		// Timestamp
		vkCmdWriteTimestamp(m_PostCudaCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_QueryPool, m_QueryIndex++);
		
		// End command buffer
		result = vkEndCommandBuffer(m_PostCudaCommandBuffer);
		ASSERT_VULKAN(result);
	}
}
