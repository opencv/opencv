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
#include "logtagconfig.hpp"

namespace cv {
namespace utils {
namespace logging {

class LogTagConfigParser
{
public:
    LogTagConfigParser(LogLevel defaultUnconfiguredGlobalLevel = LOG_LEVEL_VERBOSE);
    explicit LogTagConfigParser(const std::string& input);
    ~LogTagConfigParser();

public:
    bool parse(const std::string& input);
    bool hasMalformed() const;
    const LogTagConfig& getGlobalConfig() const;
    const std::vector<LogTagConfig>& getFullNameConfigs() const;
    const std::vector<LogTagConfig>& getFirstPartConfigs() const;
    const std::vector<LogTagConfig>& getAnyPartConfigs() const;
    const std::vector<std::string>& getMalformed() const;

private:
    void segmentTokens();
    void parseNameAndLevel(const std::string& s);
    void parseWildcard(const std::string& name, LogLevel level);
    static std::pair<LogLevel, bool> parseLogLevel(const std::string& s);
    static std::string toString(LogLevel level);

private:
    std::string m_input;
    LogTagConfig m_parsedGlobal;
    std::vector<LogTagConfig> m_parsedFullName;
    std::vector<LogTagConfig> m_parsedFirstPart;
    std::vector<LogTagConfig> m_parsedAnyPart;
    std::vector<std::string> m_malformed;
};

}}} //namespace

#endif
