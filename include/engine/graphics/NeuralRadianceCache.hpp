#pragma once

#include <tiny-cuda-nn/config.h>
#include <engine/graphics/vulkan/Buffer.hpp>
#include <vector>
#include <engine/AppConfig.hpp>

namespace en
{
	class NeuralRadianceCache
	{
	public:
		NeuralRadianceCache(const AppConfig& appConfig, const uint32_t renderWidth, const uint32_t renderHeight);

		void Init(
			float renderWidth, 
			float renderHeight,
			float* dCuInferInput, 
			float* dCuInferOutput, 
			float* dCuTrainInput, 
			float* dCuTrainTarget,
			cudaExternalSemaphore_t cudaStartSemaphore,
			cudaExternalSemaphore_t cudaFinishedSemaphore);

		void InferAndTrain(const uint32_t* inferFilter, const uint32_t* trainFilter, bool train);

		void Destroy();

		float GetLoss() const;
		size_t GetInferBatchCount() const;
		size_t GetTrainBatchCount() const;
		size_t GetTrainBatchCountHorizontal() const;
		size_t GetTrainBatchCountVertical() const;
		uint32_t GetInferBatchSizeVertical() const;
		uint32_t GetInferBatchSizeHorizontal() const;
		uint32_t GetTrainBatchSizeVertical() const;
		uint32_t GetTrainBatchSizeHorizontal() const;

		static uint32_t sc_InputCount;
		static uint32_t sc_OutputCount;

	private:
		static const uint32_t sc_FilterFrameCountThreshold = 1;

		uint32_t m_InferBatchSize = 0;
		uint32_t m_TrainBatchSize = 0;
		uint32_t m_InferBatchSizeVertical = 0;
		uint32_t m_InferBatchSizeHorizontal = 0;
		uint32_t m_TrainBatchSizeVertical = 0;
		uint32_t m_TrainBatchSizeHorizontal = 0;
		uint32_t m_InferBatchCountVertical = 0;
		uint32_t m_InferBatchCountHorizontal = 0;
		const uint32_t m_TrainBatchCountVertical = 0;
		const uint32_t m_TrainBatchCountHorizontal = 0;
		const uint32_t m_TrainMaxBatchLevel = 0;

		tcnn::TrainableModel m_Model;

		tcnn::GPUMatrix<float> m_InferInput;
		tcnn::GPUMatrix<float> m_InferOutput;
		tcnn::GPUMatrix<float> m_TrainInput;
		tcnn::GPUMatrix<float> m_TrainTarget;

		std::vector<tcnn::GPUMatrix<float>> m_InferInputBatches;
		std::vector<tcnn::GPUMatrix<float>> m_InferOutputBatches;
		std::vector<std::vector<tcnn::GPUMatrix<float>>> m_TrainInputBatches;
		std::vector<std::vector<tcnn::GPUMatrix<float>>> m_TrainTargetBatches;

		cudaExternalSemaphore_t m_CudaStartSemaphore;
		cudaExternalSemaphore_t m_CudaFinishedSemaphore;

		float m_Loss = 0.0f;
		size_t m_TrainCounter = 0;

		void Inference(const uint32_t* inferFilter);
		void Train(const uint32_t* trainFilter);
		bool GetBatchesToTrain(const int32_t currentBatchLevel, const uint32_t minBatchIdx, const uint32_t maxBatchIdx, const uint32_t* trainFilter, std::vector<std::pair<uint32_t, uint32_t>>& batchesToTrain);
		bool IsBatchFilterPositive(const uint32_t minBatchIdx, const uint32_t maxBatchIdx, const uint32_t* trainFilter);
		void AwaitCudaStartSemaphore();
		void SignalCudaFinishedSemaphore();
		size_t GetLinearInferBatchIndex(size_t verticalIdx, size_t horizontalIdx);
		size_t GetLinearTrainBatchIndex(size_t verticalIdx, size_t horizontalIdx);
	};
}
