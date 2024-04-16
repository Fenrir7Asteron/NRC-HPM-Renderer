#include <engine/cuda_common.hpp>
#include <engine/graphics/NeuralRadianceCache.hpp>
#include <random>
#include <engine/util/Log.hpp>

namespace en
{
	uint32_t NeuralRadianceCache::sc_InputCount = 5;
	uint32_t NeuralRadianceCache::sc_OutputCount = 3;

	NeuralRadianceCache::NeuralRadianceCache(const AppConfig& appConfig, const uint32_t renderWidth, const uint32_t renderHeight) :
		m_TrainBatchCountVertical(appConfig.trainBatchVerticalCount),
		m_TrainBatchCountHorizontal(appConfig.trainBatchHorizontalCount),
		m_InferBatchSize(2 << (appConfig.log2InferBatchSize - 1)),
		m_TrainBatchSize(2 << (appConfig.log2TrainBatchSize - 1)),
		m_InferBatchSizeVertical(sqrt(2 << (appConfig.log2InferBatchSize - 1))),
		m_InferBatchSizeHorizontal(sqrt(2 << (appConfig.log2InferBatchSize - 1))),
		m_InferBatchCountVertical(ceil((float) renderHeight / m_InferBatchSizeVertical)),
		m_InferBatchCountHorizontal(ceil((float) renderWidth / m_InferBatchSizeHorizontal)),
		m_TrainBatchSizeVertical(sqrt(2 << (appConfig.log2TrainBatchSize - 1))),
		m_TrainBatchSizeHorizontal(sqrt(2 << (appConfig.log2TrainBatchSize - 1)))
	{
		nlohmann::json modelConfig = {
			{"loss", {
				{"otype", appConfig.lossFn}
			}},
			{"optimizer", {
				{"otype", "EMA"},
				{"decay", appConfig.emaDecay},
				{"nested", {
					{"otype", appConfig.optimizer},
					{"learning_rate", appConfig.learningRate},
					//{"l2_reg", 0.0001},
				}}
			}},
			appConfig.encoding.jsonConfig,
			{"network", {
				{"otype", "FullyFusedMLP"},
				{"activation", "ReLU"},
				{"output_activation", "None"},
				{"n_neurons", appConfig.nnWidth},
				{"n_hidden_layers", appConfig.nnDepth},
			}},
		};

		m_Model = tcnn::create_from_config(sc_InputCount, sc_OutputCount, modelConfig);
	}

	void NeuralRadianceCache::Init(
		float renderWidth, 
		float renderHeight,
		float* dCuInferInput,
		float* dCuInferOutput,
		float* dCuTrainInput,
		float* dCuTrainTarget,
		cudaExternalSemaphore_t cudaStartSemaphore,
		cudaExternalSemaphore_t cudaFinishedSemaphore)
	{
		const uint32_t inferCount = renderWidth * renderHeight;

		// Check if sample counts are compatible
		if (inferCount % 16 != 0) { en::Log::Error("NRC requires inferCount to be a multiple of 16", true); }

		// Init members
		m_CudaStartSemaphore = cudaStartSemaphore;
		m_CudaFinishedSemaphore = cudaFinishedSemaphore;

		// Init big buffer
		const uint32_t trainCount = m_TrainBatchCountVertical * m_TrainBatchCountHorizontal * m_TrainBatchSizeVertical * m_TrainBatchSizeHorizontal;

		m_InferInput = tcnn::GPUMatrix<float>(dCuInferInput, sc_InputCount, inferCount);
		m_InferOutput = tcnn::GPUMatrix<float>(dCuInferOutput, sc_OutputCount, inferCount);
		m_TrainInput = tcnn::GPUMatrix<float>(dCuTrainInput, sc_InputCount, trainCount);
		m_TrainTarget = tcnn::GPUMatrix<float>(dCuTrainTarget, sc_OutputCount, trainCount);

		// Init infer buffers
		
		uint32_t inferBatchCount = m_InferBatchCountVertical * m_InferBatchCountHorizontal;
		if (m_InferBatchSize % tcnn::BATCH_SIZE_GRANULARITY != 0) { en::Log::Error("NRC requires inferBatchSize to be a multiple of " + std::to_string(tcnn::BATCH_SIZE_GRANULARITY), true); }
		m_InferInputBatches.resize(inferBatchCount);
		m_InferOutputBatches.resize(inferBatchCount);

		const uint32_t inferLastBatchSizeVertical = renderHeight - ((m_InferBatchCountVertical - 1) * m_InferBatchSizeVertical);
		const uint32_t inferLastBatchSizeHorizontal = renderWidth - ((m_InferBatchCountHorizontal - 1) * m_InferBatchSizeHorizontal);
		uint32_t batchOffset = 0;

		for (uint32_t i = 0; i < m_InferBatchCountVertical; i++)
		{
			for (uint32_t j = 0; j < m_InferBatchCountHorizontal; j++)
			{
				const uint32_t linearBatchIdx = GetLinearInferBatchIndex(i, j);
				if (i < m_InferBatchCountVertical - 1 && j < m_InferBatchCountHorizontal - 1)
				{
					m_InferInputBatches[linearBatchIdx] = m_InferInput.slice_cols(batchOffset, m_InferBatchSize);
					m_InferOutputBatches[linearBatchIdx] = m_InferOutput.slice_cols(batchOffset, m_InferBatchSize);
					batchOffset += m_InferBatchSize;
				}
				else if (i == m_InferBatchCountVertical - 1 && j < m_InferBatchCountHorizontal - 1)
				{
					m_InferInputBatches[linearBatchIdx] = m_InferInput.slice_cols(batchOffset, inferLastBatchSizeVertical * m_InferBatchSizeHorizontal);
					m_InferOutputBatches[linearBatchIdx] = m_InferOutput.slice_cols(batchOffset, inferLastBatchSizeVertical * m_InferBatchSizeHorizontal);
					batchOffset += inferLastBatchSizeVertical * m_InferBatchSizeHorizontal;
				}
				else if (i < m_InferBatchCountVertical - 1 && j == m_InferBatchCountHorizontal - 1)
				{
					m_InferInputBatches[linearBatchIdx] = m_InferInput.slice_cols(batchOffset, inferLastBatchSizeHorizontal * m_InferBatchSizeVertical);
					m_InferOutputBatches[linearBatchIdx] = m_InferOutput.slice_cols(batchOffset, inferLastBatchSizeHorizontal * m_InferBatchSizeVertical);
					batchOffset += inferLastBatchSizeHorizontal * m_InferBatchSizeVertical;
				}
				else
				{
					m_InferInputBatches[linearBatchIdx] = m_InferInput.slice_cols(batchOffset, inferLastBatchSizeVertical * inferLastBatchSizeHorizontal);
					m_InferOutputBatches[linearBatchIdx] = m_InferOutput.slice_cols(batchOffset, inferLastBatchSizeVertical * inferLastBatchSizeHorizontal);
					batchOffset += inferLastBatchSizeVertical * inferLastBatchSizeHorizontal;
				}
			}
		}

		// Init train buffers
		const uint32_t trainBatchCount = m_TrainBatchCountVertical * m_TrainBatchCountHorizontal;
		if (m_TrainBatchSize % tcnn::BATCH_SIZE_GRANULARITY != 0) { en::Log::Error("NRC requires trainBatchSize to be a multiple of " + std::to_string(tcnn::BATCH_SIZE_GRANULARITY), true); }
		m_TrainInputBatches.resize(trainBatchCount);
		m_TrainTargetBatches.resize(trainBatchCount);

		for (uint32_t i = 0; i < trainBatchCount; i++)
		{
			m_TrainInputBatches[i] = m_TrainInput.slice_cols(i * m_TrainBatchSize, m_TrainBatchSize);
			m_TrainTargetBatches[i] = m_TrainTarget.slice_cols(i * m_TrainBatchSize, m_TrainBatchSize);
		}

		en::Log::Info("Infer batch offset" + std::to_string(batchOffset) + ", infer count" + std::to_string(inferCount));
		en::Log::Info("Infer batch count (V:" + std::to_string(m_InferBatchCountVertical)+ ", H:" + std::to_string(m_InferBatchCountHorizontal) + ")");
		en::Log::Info("Infer batch size (V:" + std::to_string(m_InferBatchSizeVertical) + ", H:" + std::to_string(m_InferBatchSizeHorizontal) + ")");
		en::Log::Info("Train batch count (V:" + std::to_string(m_TrainBatchCountVertical) + ", H:" + std::to_string(m_TrainBatchCountHorizontal) + ")");
		en::Log::Info("Train batch size (V:" + std::to_string(m_TrainBatchSizeVertical) + ", H:" + std::to_string(m_TrainBatchSizeHorizontal) + ")");
	}

	void NeuralRadianceCache::InferAndTrain(const uint32_t* inferFilter, const uint32_t* trainFilter, uint32_t* trainFilteredFrameCounter, bool train)
	{
		AwaitCudaStartSemaphore();
		Inference(inferFilter);
		if (train) { Train(trainFilter, trainFilteredFrameCounter); }
		SignalCudaFinishedSemaphore();
	}

	void NeuralRadianceCache::Destroy()
	{
	}

	float NeuralRadianceCache::GetLoss() const
	{
		return m_Loss;
	}

	size_t NeuralRadianceCache::GetInferBatchCount() const
	{
		return m_InferBatchCountVertical * m_InferBatchCountHorizontal;
	}

	size_t NeuralRadianceCache::GetTrainBatchCount() const
	{
		return m_TrainBatchCountVertical * m_TrainBatchCountHorizontal;
	}

	size_t NeuralRadianceCache::GetTrainBatchCountHorizontal() const
	{
		return m_TrainBatchCountHorizontal;
	}

	size_t NeuralRadianceCache::GetTrainBatchCountVertical() const
	{
		return m_TrainBatchCountVertical;
	}

	uint32_t NeuralRadianceCache::GetInferBatchSizeVertical() const
	{
		return m_InferBatchSizeVertical;
	}

	uint32_t NeuralRadianceCache::GetInferBatchSizeHorizontal() const
	{
		return m_InferBatchSizeHorizontal;
	}

	uint32_t NeuralRadianceCache::GetTrainBatchSizeVertical() const
	{
		return m_TrainBatchSizeVertical;
	}

	uint32_t NeuralRadianceCache::GetTrainBatchSizeHorizontal() const
	{
		return m_TrainBatchSizeHorizontal;
	}

	void NeuralRadianceCache::Inference(const uint32_t* inferFilter)
	{
		for (size_t i = 0; i < m_InferBatchCountVertical; i++)
		{
			for (int j = 0; j < m_InferBatchCountHorizontal; ++j)
			{
				const size_t linearBatchIndex = GetLinearInferBatchIndex(i, j);
				//en::Log::Info("Linear infer batch index " + std::to_string(linearBatchIndex)+ " has filter " + std::to_string(inferFilter[linearBatchIndex]));
				if (inferFilter[linearBatchIndex] > 0)
				{
					const tcnn::GPUMatrix<float>& inputBatch = m_InferInputBatches[linearBatchIndex];
					tcnn::GPUMatrix<float>& outputBatch = m_InferOutputBatches[linearBatchIndex];
					m_Model.network->inference(inputBatch, outputBatch);
				}
			}
		}
	}

	void NeuralRadianceCache::Train(const uint32_t* trainFilter, uint32_t* trainFilteredFrameCounter)
	{
		for (size_t i = 0; i < m_TrainBatchCountVertical; i++)
		{
			for (size_t j = 0; j < m_TrainBatchCountHorizontal; j++)
			{
				const size_t linearBatchIndex = GetLinearTrainBatchIndex(i, j);
				//en::Log::Info("Linear train batch index " + std::to_string(linearBatchIndex) + " has filter " + std::to_string(trainFilter[linearBatchIndex]));

				if (trainFilter[linearBatchIndex] <= 0)
				{
					trainFilteredFrameCounter[linearBatchIndex] = std::min(trainFilteredFrameCounter[linearBatchIndex] + 1, sc_FilterFrameCountThreshold);
				}
				else
				{
					trainFilteredFrameCounter[linearBatchIndex] = 0;
				}

				// Exclude batch from training if it filtered more than sc_FilterFrameCountThreshold times
				// Batch is filtered if not a single ray scattered inside it
				if (trainFilteredFrameCounter[linearBatchIndex] < sc_FilterFrameCountThreshold)
				{
					const tcnn::GPUMatrix<float>& inputBatch = m_TrainInputBatches[linearBatchIndex];
					const tcnn::GPUMatrix<float>& targetBatch = m_TrainTargetBatches[linearBatchIndex];
					auto forwardContext = m_Model.trainer->training_step(inputBatch, targetBatch);
					m_Loss = m_Model.trainer->loss(*forwardContext.get());
				}
			}
		}
	}

	void NeuralRadianceCache::AwaitCudaStartSemaphore()
	{
		cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
		memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
		extSemaphoreWaitParams.params.fence.value = 0;
		extSemaphoreWaitParams.flags = 0;

		cudaError_t error = cudaWaitExternalSemaphoresAsync(&m_CudaStartSemaphore, &extSemaphoreWaitParams, 1);
		ASSERT_CUDA(error);
	}

	void NeuralRadianceCache::SignalCudaFinishedSemaphore()
	{
		cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
		memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));
		extSemaphoreSignalParams.params.fence.value = 0;
		extSemaphoreSignalParams.flags = 0;

		cudaError_t error = cudaSignalExternalSemaphoresAsync(&m_CudaFinishedSemaphore, &extSemaphoreSignalParams, 1);
		ASSERT_CUDA(error);
	}

	size_t NeuralRadianceCache::GetLinearInferBatchIndex(size_t verticalBatchIdx, size_t horizontalBatchIdx)
	{
		return verticalBatchIdx * m_InferBatchCountHorizontal + horizontalBatchIdx;
	}

	size_t NeuralRadianceCache::GetLinearTrainBatchIndex(size_t verticalBatchIdx, size_t horizontalBatchIdx)
	{
		return verticalBatchIdx * m_TrainBatchCountHorizontal + horizontalBatchIdx;
	}
}
