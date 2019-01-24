// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <sstream>
#include <iostream>
#include <fstream>
#include <memory>

#ifdef __ANDROID__
# include <android/log.h>
#endif

namespace cv {
namespace utils {
namespace logging {
namespace hash_utility
{
    /**
    The 64-bit of Fnv1a is used because it allows character-wise calculation,
    and also because the 64-bit version of the function has smaller chance of
    collisions than the 32-bit version.
    */
    struct Fnv1a
    {
        static constexpr uint64_t basis = 0xcbf29ce484222325uLL;
        static constexpr uint64_t prime = 0x100000001b3uLL;

        static uint64_t open()
        {
            return basis;
        }

        static uint64_t update(uint64_t state, uint8_t octet)
        {
            return (state ^ octet) * prime;
        }

        static uint64_t close(uint64_t state)
        {
            return state;
        }

        /**
        @brief Updates the hash value with characters read from a string.
        @detail
        The function stops at the null character.
        Caller is responsible for opening and closing; this function is not
        responsible for either.
        By calling updateHash() on more than one strings, it can be used to
        effectively compute the hash from a string formed from their concatenations.
        */
        static uint64_t updateString(uint64_t hash, const char* s)
        {
            // Caller responsible for Fnv1a::open() and Fnv1a::close()
            const char* s2 = s;
            if (s2) // in case s is nullptr
            {
                while (*s2)
                {
                    hash = Fnv1a::update(hash, (uint8_t)(*s2));
                    ++s2;
                }
            }
            return hash;
        }
    };
}// hash_utility

LogScope::LogScope()
    : m_name()
    , m_parent(nullptr)
    , m_scopeLevel(0)
    , m_fullHashLazy()
    , m_dataLazy(nullptr)
{
}

LogScope::LogScope(const char* name, const LogScope& parent)
    : m_name(name)
    , m_parent(std::addressof(parent))
    , m_scopeLevel(parent.m_scopeLevel + 1)
    , m_fullHashLazy(0u)
    , m_dataLazy(nullptr)
{
}

LogScope::~LogScope()
{
}

LogScope& LogScope::getRoot()
{
    static LogScope stc_rootLogScope{};
    return stc_rootLogScope;
}

int LogScope::scopeLevel() const
{
    return m_scopeLevel;
}

const std::string& LogScope::name() const
{
    return m_name;
}

std::string LogScope::fullName() const
{
    if (m_scopeLevel == 0)
    {
        return "";
    }
    else if (m_scopeLevel == 1)
    {
        return m_name;
    }
    else
    {
        return m_parent->fullName() + "." + m_name;
    }
}

uint64_t LogScope::fullNameHash() const
{
    if (m_fullHashLazy)
    {
        return m_fullHashLazy;
    }
    uint64_t hash = internal_fullHash_unclosed();
    hash = hash_utility::Fnv1a::close(hash);
    m_fullHashLazy = hash;
    return hash;
}

uint64_t LogScope::internal_fullHash_unclosed() const
{
    uint64_t hash;
    if (m_scopeLevel > 1)
    {
        hash = m_parent->internal_fullHash_unclosed();
        hash = hash_utility::Fnv1a::update(hash, '.');
    }
    else
    {
        hash = hash_utility::Fnv1a::open();
    }
    hash = hash_utility::Fnv1a::updateString(hash, m_name.c_str());
    return hash;
}

int LogScope::getLogThreshold() const
{
    LogThreshold* data = m_dataLazy;
    if (!data)
    {
        data = std::addressof(LogManager().getThresholdForScope(*this));
        m_dataLazy = data;
    }
    int logLevel = LogManager::getLogThreshold(*data);
    if (logLevel != CV_LOG_LEVEL_CHECK_PARENT)
    {
        return logLevel;
    }
    if (m_parent)
    {
        return m_parent->getLogThreshold();
    }
    return getLogLevel();
}

// ======
//
// LogManager::StaticData class
//
// Purpose: to resolve static initialization order issues.
//
// ======

class LogManager::StaticData
{
public:
    MutexType m_mutex;
    std::unordered_map<uint64_t, std::unique_ptr<LogThreshold>> m_logScopeDataMap;

public:
    StaticData()
    {
    }

    ~StaticData()
    {
    }
};


// ======
// LogThreshold methods
// ======

constexpr LogThreshold::LogThreshold()
    : m_launchTimeLogLevel(CV_LOG_LEVEL_CHECK_PARENT)
    , m_runTimeLogLevel(CV_LOG_LEVEL_CHECK_PARENT)
{
}

// ======
//
// LogManager methods
//
// ======

LogManager::StaticData& LogManager::internal_getStaticData()
{
    static StaticData stc_data;
    return stc_data;
}

LogThreshold& LogManager::getThresholdForScope(const LogScope& scope)
{
    return getThresholdForHash(getHash(scope));
}

LogThreshold& LogManager::getThresholdForScope(const char* fullScopeName)
{
    return getThresholdForHash(getHash(fullScopeName));
}

LogThreshold& LogManager::getThresholdForScope(const std::string& fullScopeName)
{
    return getThresholdForHash(getHash(fullScopeName.c_str()));
}

void LogManager::setLogThreshold(const LogScope& scope, int logThresholdLevel, LogConfigLayer configLayer)
{
    setLogThreshold(getThresholdForScope(scope), logThresholdLevel, configLayer);
}

void LogManager::setLogThreshold(const char* fullScopeName, int logThresholdLevel, LogConfigLayer configLayer)
{
    setLogThreshold(getThresholdForScope(fullScopeName), logThresholdLevel, configLayer);
}

void LogManager::setLogThreshold(const std::string& fullScopeName, int logThresholdLevel, LogConfigLayer configLayer)
{
    setLogThreshold(getThresholdForScope(fullScopeName), logThresholdLevel, configLayer);
}

void LogManager::setLogThreshold(uint64_t hash, int logThresholdLevel, LogConfigLayer configLayer)
{
    setLogThreshold(getThresholdForHash(hash), logThresholdLevel, configLayer);
}

LogThreshold& LogManager::getThresholdForHash(uint64_t hash)
{
    StaticData& staticData = internal_getStaticData();
    LockType lock(staticData.m_mutex);
    auto iter = staticData.m_logScopeDataMap.find(hash);
    if (iter == staticData.m_logScopeDataMap.end())
    {
        iter = staticData.m_logScopeDataMap.emplace(hash, std::make_unique<LogThreshold>()).first;
    }
    return *(iter->second);
}

uint64_t LogManager::getHash(const LogScope& scope)
{
    return scope.fullNameHash();
}

uint64_t LogManager::getHash(const char* fullScopeName)
{
    uint64_t hash = hash_utility::Fnv1a::open();
    hash = hash_utility::Fnv1a::updateString(hash, fullScopeName);
    hash = hash_utility::Fnv1a::close(hash);
    return hash;
}

void LogManager::setLogThreshold(LogThreshold& data, int logThresholdLevel, LogConfigLayer configLayer)
{
    switch (configLayer)
    {
    case LogConfigLayer::LOG_CONFIG_LAUNCH:
        data.m_launchTimeLogLevel = logThresholdLevel;
        return;
    case LogConfigLayer::LOG_CONFIG_RUNTIME:
        data.m_runTimeLogLevel = logThresholdLevel;
        return;
    default:
        throw std::invalid_argument("LogManager::setLogThreshold(...), invalid value for enum LogConfigLayer");
    }
}

int LogManager::getLogThreshold(const LogThreshold& data)
{
    int runTimeLogLevel = data.m_runTimeLogLevel;
    if (runTimeLogLevel != CV_LOG_LEVEL_CHECK_PARENT)
    {
        return runTimeLogLevel;
    }
    int launchTimeLogLevel = data.m_launchTimeLogLevel;
    if (launchTimeLogLevel != CV_LOG_LEVEL_CHECK_PARENT)
    {
        return launchTimeLogLevel;
    }
    return CV_LOG_LEVEL_CHECK_PARENT;
}

std::ostream& operator << (std::ostream& o, const LogMeta& meta)
{
    if (meta.scope && (meta.scope->scopeLevel() > 0))
    {
        std::string s = meta.scope->fullName();
        o << "(scope:" << s << ") ";
    }
    if (meta.loc.file && meta.loc.file[0])
    {
        o << "(file: " << meta.loc.file << ") ";
    }
    if (meta.loc.line >= 1)
    {
        o << "(line: " << meta.loc.line << ") ";
    }
    if (meta.loc.func && meta.loc.func[0])
    {
        o << "(func: " << meta.loc.func << ") ";
    }
    if (meta.objInfo.typeName && meta.objInfo.typeName[0])
    {
        o << "(class: " << meta.objInfo.typeName << ") ";
    }
    if (meta.objInfo.ptr)
    {
        o << "(this_ptr: " << meta.objInfo.ptr << ") ";
    }
    return o;
}

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

/**
Default log filter.
This default log filter preserves old behavior: it compares the message log level
against the global log level. Other log message attributes are not checked.

This filter must fit function prototype:
typedef bool(*LogFilterFunctionType) (const ::cv::utils::logging::LogMeta&);
*/

static bool defaultLogFilter(const LogMeta& meta)
{
    // For example, compare this to original code, macro CV_LOG_FATAL
    // ...
    // if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_FATAL) break;
    // ...
    // LOG_LEVEL_FATAL is the level of the message itself;
    // getLogLevel is the global log level filter threshold.
    //
    return (getLogLevelVariable() >= meta.level);
}

static LogFilterFunctionType& getLogFilterVariable()
{
    static LogFilterFunctionType g_logFilter = defaultLogFilter;
    return g_logFilter;
}

LogFilterFunctionType setLogFilter(LogFilterFunctionType filter)
{
    LogFilterFunctionType old = getLogFilterVariable();
    getLogFilterVariable() = filter;
    return old;
}

LogFilterFunctionType getLogFilter()
{
    return getLogFilterVariable();
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

} // namespace

}}} // namespace
