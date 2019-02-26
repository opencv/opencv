// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_LOGTAGCONFIGPARSER_HPP
#define OPENCV_CORE_LOGTAGCONFIGPARSER_HPP

#if 1 // if not already in precompiled headers
#include <string>
#include <vector>
#include <functional>
#endif

#include <opencv2/core/utils/logtag.hpp>

namespace cv {
namespace utils {
namespace logging {

class LogTagConfigParser
{
public:
	struct NameLevel
	{
		std::string name;
		LogLevel level;
	};

public:
	LogTagConfigParser();
	explicit LogTagConfigParser(const std::string& input);
	~LogTagConfigParser();

public:
	bool parse(const std::string& input);
    bool hasMalformed() const;
    void forEachParsed(std::function<void(const std::string&, LogLevel)> func) const;
    void forEachMalformed(std::function<void(const std::string&)> func) const;

private:
	void segmentTokens();
	void parseNameLevel(const std::string& s);
	static std::pair<LogLevel, bool> parseLogLevel(const std::string& s);
	static std::string toString(LogLevel level);

private:
	std::string m_input;
	std::vector<NameLevel> m_parsed;
	std::vector<std::string> m_malformed;
};

}}} //namespace

#endif
