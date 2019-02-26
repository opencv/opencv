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

#if 0
static LogLevel parseLogLevelConfiguration()
{
    static cv::String param_log_level = utils::getConfigurationParameterString("OPENCV_LOG_LEVEL",
#if defined NDEBUG
            "WARNING"
#else
            "INFO"
#endif
    );
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
#endif

namespace internal
{
    // Combining several things that require static dynamic initialization in a
    // well-defined order into a struct.
    //
    struct GlobalLoggingInitStruct
    {
    public:
#if defined NDEBUG
        static constexpr bool m_isDebugBuild = false;
#else
        static constexpr bool m_isDebugBuild = true;
#endif

    public:
        static constexpr LogLevel m_defaultGlobalLogLevelNoConfig =
            m_isDebugBuild ? LOG_LEVEL_DEBUG : LOG_LEVEL_WARNING;

    public:
        LogTagManager logTagManager;
        LogTag logTagGlobal;
        cv::String logTagConfigString;
        LogTagConfigParser logTagConfigParser;

        GlobalLoggingInitStruct()
            : logTagManager()
            , logTagGlobal("global", m_defaultGlobalLogLevelNoConfig)
            , logTagConfigString()
            , logTagConfigParser()
        {
            logTagManager.assign(logTagGlobal.name, &logTagGlobal);
            if (loadConfigString())
            {
                if (!tryParseConfigString())
                {
                    handleMalformed();
                }
                // should we still try to apply some of the specs if it is partly
                // malformed, partly parsable? what are the consequences?
                // (if it is partly malformed, the remaining "parsable" part
                // may be nonsensical despite being seemingly parsable)
                applyConfigString();
            }
        }

    private:
        struct ParsedName
        {
            std::string trimmedNamePart;
            bool isGlobal;
            bool hasPrefixWildcard;
            bool hasSuffixWildcard;
        };

    private:
        bool loadConfigString()
        {
            logTagConfigString = utils::getConfigurationParameterString("OPENCV_LOG_LEVEL", "");
            return !logTagConfigString.empty();
        }

        bool tryParseConfigString()
        {
            logTagConfigParser.parse(logTagConfigString);
            return !logTagConfigParser.hasMalformed();
        }

        void handleMalformed()
        {
            // need to print warning for malformed log tag config strings?
            if (m_isDebugBuild)
            {
                auto func = [&](const std::string& malformed)
                {
                    std::cout << "Malformed log level config: \"" << malformed << "\"\n";
                };
                logTagConfigParser.forEachMalformed(func);
                std::cout.flush();
            }
        }

        void applyConfigString()
        {
            auto func = [&](const std::string& name, LogLevel level)
            {
                ParsedName parsed = parseName(name);
                if (parsed.isGlobal)
                {
                    logTagGlobal.level = level;
                    return;
                }
                auto applyFunc = [&](LogTag* pLogTag)
                {
                    pLogTag->level = level;
                };
                if (parsed.hasPrefixWildcard)
                {
                    logTagManager.forEach_byAnyPart(parsed.trimmedNamePart, applyFunc);
                }
                else if (parsed.hasSuffixWildcard)
                {
                    logTagManager.forEach_byFirstPart(parsed.trimmedNamePart, applyFunc);
                }
                else
                {
                    logTagManager.invoke(parsed.trimmedNamePart, applyFunc);
                }
            };
            logTagConfigParser.forEachParsed(func);
        }

        ParsedName parseName(const std::string& name)
        {
            constexpr size_t npos = std::string::npos;
            const size_t len = name.length();
            ParsedName parsed{ "global", true, false, false };
            if (len == 0u)
            {
                return parsed;
            }
            const bool hasPrefixWildcard = (name[0u] == '*');
            if (len == 1u && hasPrefixWildcard)
            {
                return parsed;
            }
            const size_t first = name.find_first_not_of("*.");
            if (first == npos && hasPrefixWildcard)
            {
                return parsed;
            }
            const bool hasSuffixWildcard = (name[len - 1u] == '*');
            const size_t last = name.find_last_not_of("*.");
            parsed.trimmedNamePart = name.substr(first, last - first + 1u);
            parsed.isGlobal = (parsed.trimmedNamePart == "global");
            parsed.hasPrefixWildcard = hasPrefixWildcard;
            parsed.hasSuffixWildcard = hasSuffixWildcard;
            return parsed;
        }
    };

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
        static LogLevel& refGlobalLogLevel = getGlobalLoggingInitStruct().logTagGlobal.level;
        return refGlobalLogLevel;
    }

    LogTag* getGlobalLogTag()
    {
        static LogTag& refGlobalLogTag = getGlobalLoggingInitStruct().logTagGlobal;
        return &refGlobalLogTag;
    }
}

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
    auto func = [level](LogTag* plogtag)
    {
        plogtag->level = level;
    };
    internal::getLogTagManager().invoke(tag, func);
}

LogLevel getLogTagLevel(const char* tag)
{
    cv::utils::logging::LogLevel level;
    auto func = [&level](LogTag* plogtag)
    {
        level = plogtag->level;
    };
    internal::getLogTagManager().invoke(tag, func);
    return level;
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
    default:
        return;
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
