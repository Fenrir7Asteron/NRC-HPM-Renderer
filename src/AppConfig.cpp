#include <engine/AppConfig.hpp>
#include <engine/util/Log.hpp>
#include <imgui.h>

namespace en
{
	AppConfig::NNEncodingConfig::NNEncodingConfig()
	{
	}

	AppConfig::NNEncodingConfig::NNEncodingConfig(uint32_t posID, uint32_t dirID) :
		posID(posID),
		dirID(dirID)
	{
		nlohmann::json posEncoding;
		switch (posID)
		{
		case 0:
			posEncoding = {
				{"otype", "HashGrid"},
				{"n_dims_to_encode", 3},
				{"n_levels", 16},
				{"n_features_per_level", 2},
				{"log2_hashmap_size", 19},
				{"base_resolution", 16},
				{"per_level_scale", 2.0},
			};
			break;
		case 1:
			posEncoding = {
				{"otype", "Identity"},
				{"n_dims_to_encode", 3}
			};
			break;
		case 2:
			posEncoding = {
				{"otype", "TriangleWave"},
				{"n_dims_to_encode", 3},
				{"n_frequencies", 12}
			};
			break;
		case 3:
			posEncoding = {
				{"otype", "Frequency"},
				{"n_dims_to_encode", 3},
				{"n_frequencies", 12},
			};
			break;
		default:
			Log::Error("NNEncodingConfig posID is invalid", true);
			break;
		}

		nlohmann::json dirEncoding;
		switch (dirID)
		{
		case 0:
			dirEncoding = {
				{"otype", "OneBlob"},
				{"n_dims_to_encode", 2},
				{"n_bins", 4},
			};
			break;
		case 1:
			dirEncoding = {
				{"otype", "Identity"},
				{"n_dims_to_encode", 2}
			};
			break;
		case 2:
			dirEncoding = {
				{"otype", "TriangleWave"},
				{"n_dims_to_encode", 2},
				{"n_frequencies", 4}
			};
			break;
		default:
			Log::Error("NNEncodingConfig dirID is invalid", true);
			break;
		}

		jsonConfig = { "encoding", {
			{"otype", "Composite"},
			{"reduction", "Concatenation"},
			{"nested", { posEncoding, dirEncoding }}
		}};
	}

	AppConfig::HpmSceneConfig::HpmSceneConfig()
	{
	}

	AppConfig::HpmSceneConfig::HpmSceneConfig(uint32_t id) :
		id(id)
	{
		switch (id)
		{
		case 0:
			dirLightStrength = 16.0f;
			pointLightStrength = 0.0f;
			hdrEnvMapPath = "";
			hdrEnvMapStrength = 0.0f;
			density = 0.6f;
			dynamic = false;
			break;
		case 1:
			dirLightStrength = 0.0f;
			pointLightStrength = 64.0f;
			hdrEnvMapPath = "";
			hdrEnvMapStrength = 0.0f;
			density = 0.6f;
			dynamic = false;
			break;
		case 2:
			dirLightStrength = 0.0f;
			pointLightStrength = 128.0f;
			hdrEnvMapPath = "";
			hdrEnvMapStrength = 0.0;
			density = 1.0f;
			dynamic = false;
			break;
		case 3:
			dirLightStrength = 16.0f;
			pointLightStrength = 0.0f;
			hdrEnvMapPath = "";
			hdrEnvMapStrength = 0.0;
			density = 0.25f;
			dynamic = false;
			break;
		case 4:
			dirLightStrength = 8.0f;
			pointLightStrength = 0.0f;
			hdrEnvMapPath = "";
			hdrEnvMapStrength = 0.0;
			density = 0.6f;
			dynamic = false;
			break;
		case 5:
			dirLightStrength = 0.0f;
			pointLightStrength = 0.0f;
			hdrEnvMapPath = "";
			hdrEnvMapStrength = 1.0f;
			density = 1.6f; // 0.8
			dynamic = false;
			break;
		default:
			Log::Error("HpmSceneConfig ID is invalid", true);
			break;
		}
	}

	AppConfig::AppConfig() {}

	AppConfig::AppConfig(const std::vector<char*>& argv)
	{
		if (argv.size() != 19) { Log::Error("Argument count does not match requirements for AppConfig", true); }

		size_t index = 1;

		lossFn = std::string(argv[index++]);
		optimizer = std::string(argv[index++]);
		learningRate = std::stof(argv[index++]);
		emaDecay = std::stof(argv[index++]);
		
		const uint32_t posID = std::stoi(argv[index++]);
		const uint32_t dirID = std::stoi(argv[index++]);
		encoding = NNEncodingConfig(posID, dirID);
		
		nnWidth = std::stoi(argv[index++]);
		nnDepth = std::stoi(argv[index++]);
		log2InferBatchSize = std::stoi(argv[index++]);
		log2TrainBatchSize = std::stoi(argv[index++]);
		trainBatchCount = std::stoi(argv[index++]);

		scene = HpmSceneConfig(std::stoi(argv[index++]));

		trainRingBufSize = std::stof(argv[index++]);
		trainSpp = std::stoi(argv[index++]);
		primaryRayLength = std::stoi(argv[index++]);
		primaryRayProb = std::stof(argv[index++]);
		trainRayLength = std::stoi(argv[index++]);
		enableBenchmarkOnStart = std::stoi(argv[index++]);
	}

	std::string AppConfig::GetName() const
	{
		std::string str = "";
		str += lossFn + "_";
		str += optimizer + "_";
		str += std::to_string(learningRate) + "_";
		str += std::to_string(emaDecay) + "_";
		str += std::to_string(encoding.posID) + "_";
		str += std::to_string(encoding.dirID) + "_";
		str += std::to_string(nnWidth) + "_";
		str += std::to_string(nnDepth) + "_";
		str += std::to_string(log2InferBatchSize) + "_";
		str += std::to_string(log2TrainBatchSize) + "_";
		str += std::to_string(trainBatchCount) + "_";
		str += std::to_string(scene.id) + "_";
		str += std::to_string(trainRingBufSize) + "_";
		str += std::to_string(trainSpp) + "_";
		str += std::to_string(primaryRayLength) + "_";
		str += std::to_string(primaryRayProb) + "_";
		str += std::to_string(trainRayLength);
		return str;
	}

	void AppConfig::RenderImGui() const
	{
		ImGui::Begin("AppConfig");
		ImGui::Text(lossFn.c_str());
		ImGui::Text(optimizer.c_str());
		ImGui::Text("Learning rate %f", learningRate);
		ImGui::Text("EMA Decay %f", emaDecay);
		ImGui::Text("Encoding (%d, %d)", encoding.posID, encoding.dirID);
		ImGui::Text("NN Width %d", nnWidth);
		ImGui::Text("NN Depth %d", nnDepth);
		ImGui::Text("Batch Sizes (%d, %d)", log2InferBatchSize, log2TrainBatchSize);
		ImGui::Text("Train Batch Count %d", trainBatchCount);
		ImGui::Text("Scene %d", scene.id);
		ImGui::Text("Train ring buffer size %f", trainRingBufSize);
		ImGui::Text("Train spp %d", trainSpp);
		ImGui::Text("Primary ray length %d", primaryRayLength);
		ImGui::Text("Primary ray prob %f", primaryRayProb);
		ImGui::Text("Train ray length %d", trainRayLength);
		ImGui::End();
	}
}
