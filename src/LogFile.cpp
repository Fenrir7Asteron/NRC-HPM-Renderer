#include <engine/util/LogFile.hpp>
#include <filesystem>
#include <engine/util/Log.hpp>

namespace en
{
	LogFile::LogFile(const std::string& filePath) : m_FilePath(filePath)
	{
		if (std::filesystem::exists(filePath))
		{
			Log::Info("Log file (" + filePath + ") already exists -> deleting it");
			std::filesystem::remove(filePath);
		}
	}

	LogFile::~LogFile()
	{
		if (m_File.is_open())
		{
			m_File.close();
		}
	}

	void LogFile::OpenLogFile()
	{
		Log::Info("Open file (" + m_FilePath + ")");
		m_File = std::ofstream(m_FilePath);
	}

	void LogFile::WriteLine(const std::string& line)
	{
		if (!m_File.is_open())
		{
			OpenLogFile();
		}

		m_File << line << std::endl;
	}
}
