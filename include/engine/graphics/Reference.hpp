#pragma once

#include <array>
#include <engine/graphics/Camera.hpp>
#include <engine/AppConfig.hpp>
#include <engine/HpmScene.hpp>
#include <engine/graphics/vulkan/CommandPool.hpp>
#include <engine/graphics/vulkan/Shader.hpp>
#include <engine/graphics/renderer/NrcHpmRenderer.hpp>
#include <engine/graphics/renderer/McHpmRenderer.hpp>

namespace en
{
	class Reference
	{
	public:
		struct Result
		{
			float mse; // MSE of "not reference" to reference
			float refMean; // Mean or reference image
			float ownMean; // Mean of "not reference" image
			float ownVar; // Variance of "not reference" image
			uint32_t validPixelCount; // Number of valid pixels

			float GetBias() const;
			float GetRelBias() const;
			float GetRelVar() const;
			float GetCV() const;
		};

		Reference(
			uint32_t width,
			uint32_t height,
			const AppConfig& appConfig,
			const HpmScene& scene,
			VkQueue queue);

		Result CompareNrc(NrcHpmRenderer& renderer, const Camera* oldCamera, VkQueue queue);
		Result CompareMc(McHpmRenderer& renderer, const Camera* oldCamera, VkQueue queue);
		void Destroy();

	private:
		struct SpecializationData
		{
			uint32_t width;
			uint32_t height;
		};

		uint32_t m_Width = 0;
		uint32_t m_Height = 0;

		VkDescriptorSetLayout m_DescSetLayout;
		VkDescriptorPool m_DescPool;
		VkDescriptorSet m_DescSet;

		vk::CommandPool m_CmdPool;
		VkCommandBuffer m_CmdBuf;

		SpecializationData m_SpecData = {};
		std::vector<VkSpecializationMapEntry> m_SpecEntries;
		VkSpecializationInfo m_SpecInfo;

		VkPipelineLayout m_PipelineLayout;
		
		vk::Shader m_Cmp1Shader;
		VkPipeline m_Cmp1Pipeline = VK_NULL_HANDLE;

		vk::Shader m_NormShader;
		VkPipeline m_NormPipeline = VK_NULL_HANDLE;

		vk::Shader m_Cmp2Shader;
		VkPipeline m_Cmp2Pipeline = VK_NULL_HANDLE;

		en::Camera* m_RefCamera = nullptr;
		VkImage m_RefImage = VK_NULL_HANDLE;
		VkDeviceMemory m_RefImageMemory = VK_NULL_HANDLE;
		VkImageView m_RefImageView = VK_NULL_HANDLE;

		vk::Buffer m_ResultStagingBuffer;
		vk::Buffer m_ResultBuffer;

		void CreateDescriptor();
		void UpdateDescriptor(VkImageView refImageView, VkImageView cmpImageView);

		void InitSpecInfo();
		void CreatePipelineLayout();
		void CreateCmp1Pipeline();
		void CreateNormPipeline();
		void CreateCmp2Pipeline();

		void RecordCmpCmdBuf();

		void CreateRefCameras();
		void CreateRefImages(VkQueue queue);
		void GenRefImages(const AppConfig& appConfig, const HpmScene& scene, VkQueue queue);
		void CopyToRefImage(uint32_t imageIdx, VkImage srcImage, VkQueue queue);
	};
}
