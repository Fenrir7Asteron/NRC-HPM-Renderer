#pragma once

#include <string>
#include <fstream>

namespace en
{
	class LogFile
	{
	public:
		LogFile(const std::string& filePath);
		~LogFile();

		void OpenLogFile();
		void WriteLine(const std::string& line);

	private:
		std::string m_FilePath;
		std::ofstream m_File;
	};
}
