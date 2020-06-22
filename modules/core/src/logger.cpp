// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "utils/logtagmanager.hpp"
#include "utils/logtagconfigparser.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

#ifdef __ANDROID__
# include <android/log.h>
#endif

namespace cv {
namespace utils {
namespace logging {

namespace internal
{

// Combining several things that require static dynamic initialization in a
// well-defined order into a struct.
//
struct GlobalLoggingInitStruct
{
public:
#if defined NDEBUG
    static const bool m_isDebugBuild = false;
#else
    static const bool m_isDebugBuild = true;
#endif

public:
    static LogLevel m_defaultUnconfiguredGlobalLevel;

public:
    LogTagManager logTagManager;

    GlobalLoggingInitStruct()
        : logTagManager(m_defaultUnconfiguredGlobalLevel)
    {
        (void)getInitializationMutex();  // ensure initialization of global objects

        applyConfigString();
        handleMalformed();
    }

private:
    void applyConfigString()
    {
        logTagManager.setConfigString(utils::getConfigurationParameterString("OPENCV_LOG_LEVEL", ""));
    }

    void handleMalformed()
    {
        // need to print warning for malformed log tag config strings?
        if (m_isDebugBuild)
        {
            const auto& parser = logTagManager.getConfigParser();
            if (parser.hasMalformed())
            {
                const auto& malformedList = parser.getMalformed();
                for (const auto& malformed : malformedList)
                {
                    std::cout << "Malformed log level config: \"" << malformed << "\"\n";
                }
                std::cout.flush();
            }
        }
    }
};

LogLevel GlobalLoggingInitStruct::m_defaultUnconfiguredGlobalLevel = GlobalLoggingInitStruct::m_isDebugBuild
                ? LOG_LEVEL_INFO
                : LOG_LEVEL_WARNING;


// Static dynamic initialization guard function for the combined struct
// just defined above
//
// An initialization guard function guarantees that outside code cannot
// accidentally see not-yet-dynamically-initialized data, by routing
// all outside access request to this function, so that this function
// has a chance to run the initialization code if necessary.
//
// An initialization guard function only guarantees initialization upon
// the first call to this function.
//
static GlobalLoggingInitStruct& getGlobalLoggingInitStruct()
{
    static GlobalLoggingInitStruct globalLoggingInitInstance;
    return globalLoggingInitInstance;
}

// To ensure that the combined struct defined above is initialized even
// if the initialization guard function wasn't called, a dummy static
// instance of a struct is defined below, which will call the
// initialization guard function.
//
struct GlobalLoggingInitCall
{
    GlobalLoggingInitCall()
    {
        getGlobalLoggingInitStruct();
        (void)getGlobalLogTag();  // complete initialization of logger structures
    }
};

static GlobalLoggingInitCall globalLoggingInitCall;

static LogTagManager& getLogTagManager()
{
    static LogTagManager& logTagManagerInstance = getGlobalLoggingInitStruct().logTagManager;
    return logTagManagerInstance;
}

static LogLevel& getLogLevelVariable()
{
    static LogLevel& refGlobalLogLevel = getGlobalLogTag()->level;
    return refGlobalLogLevel;
}

LogTag* getGlobalLogTag()
{
    static LogTag* globalLogTagPtr = getGlobalLoggingInitStruct().logTagManager.get("global");
    return globalLogTagPtr;
}

} // namespace

void registerLogTag(LogTag* plogtag)
{
    if (!plogtag || !plogtag->name)
    {
        return;
    }
    internal::getLogTagManager().assign(plogtag->name, plogtag);
}

void setLogTagLevel(const char* tag, LogLevel level)
{
    if (!tag)
    {
        return;
    }
    internal::getLogTagManager().setLevelByFullName(std::string(tag), level);
}

LogLevel getLogTagLevel(const char* tag)
{
    if (!tag)
    {
        return getLogLevel();
    }
    const LogTag* ptr = internal::getLogTagManager().get(std::string(tag));
    if (!ptr)
    {
        return getLogLevel();
    }
    return ptr->level;
}

LogLevel setLogLevel(LogLevel logLevel)
{
    // note: not thread safe, use sparingly and do not critically depend on outcome
    LogLevel& refGlobalLevel = internal::getLogLevelVariable();
    const LogLevel old = refGlobalLevel;
    refGlobalLevel = logLevel;
    return old;
}

LogLevel getLogLevel()
{
    return internal::getLogLevelVariable();
}

namespace internal {

void writeLogMessage(LogLevel logLevel, const char* message)
{
    const int threadID = cv::utils::getThreadID();
    std::ostringstream ss;
    switch (logLevel)
    {
    case LOG_LEVEL_FATAL:   ss << "[FATAL:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_ERROR:   ss << "[ERROR:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_WARNING: ss << "[ WARN:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_INFO:    ss << "[ INFO:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_DEBUG:   ss << "[DEBUG:" << threadID << "] " << message << std::endl; break;
    case LOG_LEVEL_VERBOSE: ss << message << std::endl; break;
    case LOG_LEVEL_SILENT: return;  // avoid compiler warning about incomplete switch
    case ENUM_LOG_LEVEL_FORCE_INT: return;  // avoid compiler warning about incomplete switch
    }
#ifdef __ANDROID__
    int android_logLevel = ANDROID_LOG_INFO;
    switch (logLevel)
    {
    case LOG_LEVEL_FATAL:   android_logLevel = ANDROID_LOG_FATAL; break;
    case LOG_LEVEL_ERROR:   android_logLevel = ANDROID_LOG_ERROR; break;
    case LOG_LEVEL_WARNING: android_logLevel = ANDROID_LOG_WARN; break;
    case LOG_LEVEL_INFO:    android_logLevel = ANDROID_LOG_INFO; break;
    case LOG_LEVEL_DEBUG:   android_logLevel = ANDROID_LOG_DEBUG; break;
    case LOG_LEVEL_VERBOSE: android_logLevel = ANDROID_LOG_VERBOSE; break;
    default:
        break;
    }
    __android_log_print(android_logLevel, "OpenCV/" CV_VERSION, "%s", ss.str().c_str());
#endif
    std::ostream* out = (logLevel <= LOG_LEVEL_WARNING) ? &std::cerr : &std::cout;
    (*out) << ss.str();
    if (logLevel <= LOG_LEVEL_WARNING)
        (*out) << std::flush;
}

void writeLogMessageEx(LogLevel logLevel, const char* tag, const char* file, int line, const char* func, const char* message)
{
    std::ostringstream strm;
    if (tag)
    {
        strm << tag << " ";
    }
    if (file)
    {
        strm << file << " ";
    }
    if (line > 0)
    {
        strm << "(" << line << ") ";
    }
    if (func)
    {
        strm << func << " ";
    }
    strm << message;
    writeLogMessage(logLevel, strm.str().c_str());
}

} // namespace

}}} // namespace
