// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <precomp.hpp>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <sstream>
#include <iostream>
#include <fstream>

namespace cv {
namespace utils {
namespace logging {

static LogLevel parseLogLevelConfiguration()
{
    static cv::String param_log_level = utils::getConfigurationParameterString("OPENCV_LOG_LEVEL", "INFO");
    if (param_log_level == "DISABLED" || param_log_level == "disabled" ||
        param_log_level == "0" || param_log_level == "OFF" || param_log_level == "off")
        return LOG_LEVEL_SILENT;
    if (param_log_level == "FATAL" || param_log_level == "fatal")
        return LOG_LEVEL_FATAL;
    if (param_log_level == "ERROR" || param_log_level == "error")
        return LOG_LEVEL_ERROR;
    if (param_log_level == "WARNING" || param_log_level == "warning" ||
        param_log_level == "WARNINGS" || param_log_level == "warnings" ||
        param_log_level == "WARN" || param_log_level == "warn")
        return LOG_LEVEL_WARNING;
    if (param_log_level == "INFO" || param_log_level == "info")
        return LOG_LEVEL_INFO;
    if (param_log_level == "DEBUG" || param_log_level == "debug")
        return LOG_LEVEL_DEBUG;
    if (param_log_level == "VERBOSE" || param_log_level == "verbose")
        return LOG_LEVEL_VERBOSE;
    std::cerr << "ERROR: Unexpected logging level value: " << param_log_level << std::endl;
    return LOG_LEVEL_INFO;
}

static LogLevel& getLogLevelVariable()
{
    static LogLevel g_logLevel = parseLogLevelConfiguration();
    return g_logLevel;
}

LogLevel setLogLevel(LogLevel logLevel)
{
    LogLevel old = getLogLevelVariable();
    getLogLevelVariable() = logLevel;
    return old;
}

LogLevel getLogLevel()
{
    return getLogLevelVariable();
}

namespace internal {

void writeLogMessage(LogLevel logLevel, const char* message)
{
    const int threadID = cv::utils::getThreadID();
    std::ostream* out = (logLevel <= LOG_LEVEL_WARNING) ? &std::cerr : &std::cout;
    std::stringstream ss;
    switch (logLevel)
    {
    case LOG_LEVEL_FATAL:   ss << "[FATAL:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_ERROR:   ss << "[ERROR:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_WARNING: ss << "[ WARN:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_INFO:    ss << "[ INFO:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_DEBUG:   ss << "[DEBUG:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_VERBOSE: ss << message << std::endl; break;
    default:
        return;
    }
    (*out) << ss.str();
    if (logLevel <= LOG_LEVEL_WARNING)
        (*out) << std::flush;
}

} // namespace

}}} // namespace
