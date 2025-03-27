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
#include <atomic>

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
    CV_SINGLETON_LAZY_INIT_REF(GlobalLoggingInitStruct, new GlobalLoggingInitStruct());
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

namespace //unnamed
{
    std::atomic<WriteLogMessageFuncType> stc_userWriteLogMessageFunc{};
    std::atomic<WriteLogMessageExFuncType> stc_userWriteLogMessageExFunc{};
} //unnamed

static int getShowTimestampMode()
{
    static bool param_timestamp_enable = utils::getConfigurationParameterBool("OPENCV_LOG_TIMESTAMP", true);
    static bool param_timestamp_ns_enable = utils::getConfigurationParameterBool("OPENCV_LOG_TIMESTAMP_NS", false);
    return (param_timestamp_enable ? 1 : 0) + (param_timestamp_ns_enable ? 2 : 0);
}

void writeLogMessage(LogLevel logLevel, const char* message)
{
    WriteLogMessageFuncType userFunc = stc_userWriteLogMessageFunc.load();
    if (userFunc && userFunc != writeLogMessage)
    {
        (*userFunc)(logLevel, message);
        return;
    }

    const int threadID = cv::utils::getThreadID();

    std::string message_id;
    switch (getShowTimestampMode())
    {
        case 1: message_id = cv::format("%d@%0.3f", threadID, getTimestampNS() * 1e-9); break;
        case 1+2: message_id = cv::format("%d@%llu", threadID, (long long unsigned int)getTimestampNS()); break;
        default: message_id = cv::format("%d", threadID); break;
    }

    std::ostringstream ss;
    switch (logLevel)
    {
    case LOG_LEVEL_FATAL:   ss << "[FATAL:" << message_id << "] " << message << std::endl; break;
    case LOG_LEVEL_ERROR:   ss << "[ERROR:" << message_id << "] " << message << std::endl; break;
    case LOG_LEVEL_WARNING: ss << "[ WARN:" << message_id << "] " << message << std::endl; break;
    case LOG_LEVEL_INFO:    ss << "[ INFO:" << message_id << "] " << message << std::endl; break;
    case LOG_LEVEL_DEBUG:   ss << "[DEBUG:" << message_id << "] " << message << std::endl; break;
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
    {
        (*out) << std::flush;
    }
}

static const char* stripSourceFilePathPrefix(const char* file)
{
    CV_Assert(file);
    const char* pos = file;
    const char* strip_pos = NULL;
    char ch = 0;
    while ((ch = pos[0]) != 0)
    {
        ++pos;
        if (ch == '/' || ch == '\\')
            strip_pos = pos;
    }
    if (strip_pos == NULL || strip_pos == pos/*eos*/)
        return file;
    return strip_pos;
}

void writeLogMessageEx(LogLevel logLevel, const char* tag, const char* file, int line, const char* func, const char* message)
{
    WriteLogMessageExFuncType userFunc = stc_userWriteLogMessageExFunc.load();
    if (userFunc && userFunc != writeLogMessageEx)
    {
        (*userFunc)(logLevel, tag, file, line, func, message);
        return;
    }

    std::ostringstream strm;
    if (tag)
    {
        strm << tag << ' ';
    }
    if (file)
    {
        strm << stripSourceFilePathPrefix(file);
        if (line > 0)
        {
            strm << ':' << line;
        }
        strm << ' ';
    }
    if (func)
    {
        strm << func << ' ';
    }
    strm << message;
    writeLogMessage(logLevel, strm.str().c_str());
}

void replaceWriteLogMessage(WriteLogMessageFuncType f)
{
    if (f == writeLogMessage)
    {
        f = nullptr;
    }
    stc_userWriteLogMessageFunc.store(f);
}

void replaceWriteLogMessageEx(WriteLogMessageExFuncType f)
{
    if (f == writeLogMessageEx)
    {
        f = nullptr;
    }
    stc_userWriteLogMessageExFunc.store(f);
}

} // namespace

}}} // namespace
