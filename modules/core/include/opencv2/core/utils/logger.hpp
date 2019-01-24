// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_LOGGER_HPP
#define OPENCV_LOGGER_HPP

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <atomic>
#include <limits.h> // INT_MAX

#include "logger.defines.hpp"

//! @addtogroup core_logging
// This section describes OpenCV logging utilities.
//
//! @{

namespace cv {
namespace utils {
namespace logging {

//! Supported logging levels and their semantic
enum LogLevel {
    LOG_LEVEL_SILENT = 0,              //!< for using in setLogVevel() call
    LOG_LEVEL_FATAL = 1,               //!< Fatal (critical) error (unrecoverable internal error)
    LOG_LEVEL_ERROR = 2,               //!< Error message
    LOG_LEVEL_WARNING = 3,             //!< Warning message
    LOG_LEVEL_INFO = 4,                //!< Info message
    LOG_LEVEL_DEBUG = 5,               //!< Debug message. Disabled in the "Release" build.
    LOG_LEVEL_VERBOSE = 6,             //!< Verbose (trace) messages. Requires verbosity level. Disabled in the "Release" build.
#ifndef CV_DOXYGEN
    ENUM_LOG_LEVEL_FORCE_INT = INT_MAX
#endif
};

//! Config layers for allowing config changes made during runtime to have
//! higher priority over launch-time configs
enum class LogConfigLayer
{
    LOG_CONFIG_LAUNCH = 0, //!< for canned configs loaded from file, command-line, or environment strings
    LOG_CONFIG_RUNTIME = 1, //!< for programmatic config changes higher priority than "LAUNCH" (canned configs)
#ifndef CV_DOXYGEN
    ENUM_LOG_CONFIG_FORCE_INT = INT_MAX
#endif
};


#if 1 // TENTATIVE

// Macros that expand to default declarations of special class methods

//! DC: Default (argumentless) constructor
#define CV_DECLARE_MEMBER_DEFAULTS_DC(ClassName) ClassName() = default;

//! CC: Copy constructor
#define CV_DECLARE_MEMBER_DEFAULTS_CC(ClassName) ClassName(const ClassName&) = default;

//! NOCC: Copy constructor is deleted
#define CV_DECLARE_MEMBER_DEFAULTS_NOCC(ClassName) ClassName(const ClassName&) = delete;

//! MC: Move constructor
#define CV_DECLARE_MEMBER_DEFAULTS_MC(ClassName) ClassName(ClassName&&) = default;

//! NOMC: Move constructor is deleted
#define CV_DECLARE_MEMBER_DEFAULTS_NOMC(ClassName) ClassName(ClassName&&) = delete;

//! CA: Copy assignment operator
#define CV_DECLARE_MEMBER_DEFAULTS_CA(ClassName) ClassName& operator = (const ClassName&) = default;

//! NOCA: Copy assignment operator is deleted
#define CV_DECLARE_MEMBER_DEFAULTS_NOCA(ClassName) ClassName& operator = (const ClassName&) = delete;

//! MA: Move assignment operator
#define CV_DECLARE_MEMBER_DEFAULTS_MA(ClassName) ClassName& operator = (ClassName&&) = default;

//! NOMA: Move assignment operator is deleted
#define CV_DECLARE_MEMBER_DEFAULTS_NOMA(ClassName) ClassName& operator = (ClassName&&) = delete;

//! ALL: All of above is declared default (constructor: argumentless, copy, move; assignment: copy, move)
#define CV_DECLARE_MEMBER_DEFAULTS_ALL(ClassName) \
    CV_DECLARE_MEMBER_DEFAULTS_DC(ClassName) \
    CV_DECLARE_MEMBER_DEFAULTS_CC(ClassName) \
    CV_DECLARE_MEMBER_DEFAULTS_MC(ClassName) \
    CV_DECLARE_MEMBER_DEFAULTS_CA(ClassName) \
    CV_DECLARE_MEMBER_DEFAULTS_MA(ClassName)

#endif // TENTATIVE

// class forward declaration
class LogThreshold;

/**
@brief Identifies the scope from which a log message is generated.

LogScope is intended to be a static fixture. The recommended hierarchy is:
- Root (which is nameless - not to be used for logging)
- Module
- Class
- Function

Other hierarchies can also be used. However, the string formed from the
fully qualified name, in  the style of "ModuleOne.ClassA.FunctionX", must
be globally unique across all hierarchies.

Recommended insertion (injection) into C++ code
- For "module", declare as namespace-level extern (global static), and then
  point a macro to that name. The macro should only be visible to files
  inside that module.
- For "class", declare as public static member of that class.
- For "function", declare as a static local inside that function.
*/
class CV_EXPORTS LogScope
{
private:
    // For within-class use by LogScope::getRoot() only. Initializes a nameless, global scope.
    LogScope();

private:
    CV_DECLARE_MEMBER_DEFAULTS_NOCC(LogScope);
    CV_DECLARE_MEMBER_DEFAULTS_NOMC(LogScope);
    CV_DECLARE_MEMBER_DEFAULTS_NOCA(LogScope);
    CV_DECLARE_MEMBER_DEFAULTS_NOMA(LogScope);

public:
    // Initializes a nested scope with the specified name and the parent.
    LogScope(const char* name, const LogScope& parent);

    // Retrieves the nameless, global scope.
    static LogScope& getRoot();

public:
    ~LogScope();

public:
    /**
    @brief The nesting level of scopes.
    The nesting level of the global scope is defined as zero.
    Each nesting increments the level by one.
    @remark This is not the log filtering level.
    */
    int scopeLevel() const;

    /**
    @brief Returns the simple name of this scope.
    */
    const std::string& name() const;

    /**
    @brief Returns the full name of this scope, which is
    the concatenation of all scope names joined with a period (".")
    */
    std::string fullName() const;

    /**
    @brief Returns the hash value computed from the full name.
    */
    uint64_t fullNameHash() const;

    /**
    @brief Returns the current filtering threshold applied to the log level
    of messages associated with this scope.
    */
    int getLogThreshold() const;

private:
    /**
    @brief Returns the hash value computed from the full name, without closing,
    so that more substrings can be concatenated.
    */
    uint64_t internal_fullHash_unclosed() const;

private:
    std::string m_name;
    const LogScope* m_parent;
    int m_scopeLevel;
    mutable std::atomic<uint64_t> m_fullHashLazy;
    mutable std::atomic<LogThreshold*> m_dataLazy;
};

// Logging-related configuration data (mutable) associated with a scope.
// Currently it just contains log level filtering threshold.
class CV_EXPORTS LogThreshold
{
public:
    // Log level filtering threshold for the associated scope for LogConfigLayer::LOG_CONFIG_LAUNCH
    int m_launchTimeLogLevel;

    // Log level filtering threshold for the associated scope for LogConfigLayer::LOG_CONFIG_RUNTIME
    int m_runTimeLogLevel;

public:
    // Sets all log levels to special value CV_LOG_LEVEL_CHECK_PARENT, which means there is no
    // filtering threshold in effect in this scope for any config layer.
    constexpr LogThreshold();

public:
    // Destructor (trivial)
    ~LogThreshold() = default;
};

/**
@brief A global singleton manager for logging configurations.

Configurable aspects are:

Log level filtering threshold can be specified for a "scope". A "scope" can be module, class,
function, or something else. Internally, a "scope" is identified by a fully qualified name,
for example "ModuleOne.ClassA.FunctionX".

Thread-local configs can be specified. Details to emerge later.

String formatting can be specified. Details to emerge later.

Logging backends (sinks) can be specified. Details to emerge later.
*/
class CV_EXPORTS LogManager
{
public:
    using MutexType = std::recursive_mutex;
    using LockType = std::lock_guard<MutexType>;

    //! @{
public:
    static LogThreshold& getThresholdForScope(const LogScope& scope);
    static LogThreshold& getThresholdForScope(const char* fullScopeName);
    static LogThreshold& getThresholdForScope(const std::string& fullScopeName);
    static LogThreshold& getThresholdForHash(uint64_t hash);
    //! @}

    //! @{
public:
    static void setLogThreshold(const LogScope& scope, int logThresholdLevel, LogConfigLayer configLayer = LogConfigLayer::LOG_CONFIG_RUNTIME);
    static void setLogThreshold(const char* fullScopeName, int logThresholdLevel, LogConfigLayer configLayer = LogConfigLayer::LOG_CONFIG_RUNTIME);
    static void setLogThreshold(const std::string& fullScopeName, int logThresholdLevel, LogConfigLayer configLayer = LogConfigLayer::LOG_CONFIG_RUNTIME);
    static void setLogThreshold(uint64_t hash, int logThresholdLevel, LogConfigLayer configLayer = LogConfigLayer::LOG_CONFIG_RUNTIME);
    //! @}

    //! @{
public:
    static uint64_t getHash(const LogScope& scope);
    static uint64_t getHash(const char* fullScopeName);
    static void setLogThreshold(LogThreshold& data, int logThresholdLevel, LogConfigLayer configLayer = LogConfigLayer::LOG_CONFIG_RUNTIME);
    static int getLogThreshold(const LogThreshold& data);
    //! @}

private:
    class StaticData;
    static StaticData& internal_getStaticData();
};

/**
Information about the current object (the "this" pointer) collected alongside each log message.
*/
class CV_EXPORTS LogObjInfo
{
public:
    CV_DECLARE_MEMBER_DEFAULTS_ALL(LogObjInfo);

public:
    /**
    Name of the class, or the string obtained from C++11 "std::type_info(typeid(Class)).name()".
    Might contain a string prefix such as "class Class" depending on compiler vendor.
    Might be nullptr. If not nullptr, the string needs to be static lifetime or const literal.
    */
    const char* typeName;

    /**
    Type-erased pointer to object. Use std::addressof() to initialize.
    */
    const void* ptr;

public:
    LogObjInfo(const char* typeName_, const void* ptr_)
        : typeName(typeName_)
        , ptr(ptr_)
    {
    }

public:
    template <class C>
    LogObjInfo(const C& c)
        : typeName(typeid(C).name())
        , ptr(std::addressof(c))
    {
    }
};

/**
Information about the line of code (source file, line number, etc) collected alongside each log message
*/
class CV_EXPORTS LogLoc
{
public:
    CV_DECLARE_MEMBER_DEFAULTS_ALL(LogLoc);

public:
    /**
    Source file name, likely from expansion of predefined __FILE__ macro.
    Might be nullptr. If not nullptr, the string needs to be static lifetime or const literal.
    */
    const char* file;

    /**
    Line number, likely from expansion of predefined __LINE__ macro.
    Might be zero or negative, which indicates unavailability.
    */
    int line;

    /**
    Function name, likely from expansion of CV_Func macro, which is similar to __func__.
    Actual formatting is compiler vendor specific and might be inconsistent.
    Could be decorated or undecorated.
    Might possibly contain qualifiers.
    Might possibly contain argument list.
    Might be utterly garbage.
    Might be nullptr. If not nullptr, the string needs to be static lifetime or const literal.
    */
    const char* func;

public:
    LogLoc(const char* file_, int line_, const char* func_)
        : file(file_)
        , line(line_)
        , func(func_)
    {
    }
};

#if 0 // TENTATIVE
/**
Information about parallelized work collected alongside each log message.

This information is only available when executed inside a cv::ParallelLoopBody method
invoked via cv::parallel_for_

@todo Currently empty and not implemented yet.
*/
class CV_EXPORTS LogPar
{
public:
    CV_DECLARE_MEMBER_DEFAULTS_ALL(LogPar);

public:
public:
};
#endif // TENTATIVE

/**
Wraps the verbosity level of a VERBOSE log message.

Because OpenCV splits the VERBOSE log level into sub-levels of verbosity via the
argument "v" in macro CV_LOG_VERBOSE(tag, v, ...) a wrapper class is needed so that
it can be updated to LogMeta class via the stream insertion operator.
*/
class CV_EXPORTS LogVerboseLevel
{
public:
    CV_DECLARE_MEMBER_DEFAULTS_ALL(LogVerboseLevel);

public:
    int verboseLevel;

public:
    LogVerboseLevel(int verboseLevel_)
        : verboseLevel(verboseLevel_)
    {
    }
};

/**
Attributes collected alongside each logged message.
*/
class CV_EXPORTS LogMeta
{
public:
    CV_DECLARE_MEMBER_DEFAULTS_CC(LogMeta);
    CV_DECLARE_MEMBER_DEFAULTS_MC(LogMeta);
    CV_DECLARE_MEMBER_DEFAULTS_CA(LogMeta);
    CV_DECLARE_MEMBER_DEFAULTS_MA(LogMeta);

public:
    LogLevel level;
    int verboseLevel;
    const LogScope* scope;
    int thread;
    LogObjInfo objInfo;
    LogLoc loc;

#if 0 // TENTATIVE
    LogPar par;
#endif // TENTATIVE

public:
    LogMeta(LogLevel level_, const LogScope* scope_ = std::addressof(LogScope::getRoot()), int thread_ = cv::utils::getThreadID())
        : level(level_)
        , verboseLevel()
        , scope(scope_)
        , thread(thread_)
        , objInfo()
        , loc()
#if 0
        , par()
#endif
    {
    }

    LogMeta()
        : LogMeta(LOG_LEVEL_VERBOSE, nullptr, cv::utils::getThreadID())
    {
    }

public:
    /**
    A stream insertion operator that does nothing. This allows the use of "nullptr"
    when invoking the CV_LOGMETA(..., metaArgs, ...) to indicate that metaArgs is empty.
    */
    LogMeta& operator << (std::nullptr_t)
    {
        return *this;
    }

    /**
    Updates the log level of this message via the stream insertion operator.
    */
    LogMeta& operator << (LogLevel level_)
    {
        this->level = level_;
        return *this;
    }

    /**
    Updates the scope info via the stream insertion operator.
    */
    LogMeta& operator << (const LogScope& scope_)
    {
        this->scope = std::addressof(scope_);
        return *this;
    }

    /**
    Updates the current class and object info via the stream insertion operator.
    */
    LogMeta& operator << (const LogObjInfo& objInfo_)
    {
        this->objInfo = objInfo_;
        return *this;
    }

    /**
    Updates the line of code info via the stream insertion operator.
    */
    LogMeta& operator << (const LogLoc& loc_)
    {
        this->loc = loc_;
        return *this;
    }

#if 0 // TENTATIVE
    /**
    Updates the parallelized work info via the stream insertion operator.
    */
    LogMeta& operator << (const LogPar& par_)
    {
        this->par = par_;
        return *this;
    }
#endif // TENTATIVE

    /**
    Updates the verbose level via the stream insertion operator.
    @note Verbose level is only relevant when the message log level is VERBOSE.
    */
    LogMeta& operator << (const LogVerboseLevel& verboseLevel_)
    {
        this->verboseLevel = verboseLevel_.verboseLevel;
        return *this;
    }
};

//! Serializes the log message metadata to a string that is inserted into an std::ostream.
CV_EXPORTS std::ostream& operator << (std::ostream& o, const LogMeta& meta);

/** Set global logging level
@return previous logging level
*/
CV_EXPORTS LogLevel setLogLevel(LogLevel logLevel);

/** Get global logging level */
CV_EXPORTS LogLevel getLogLevel();

/**
The function prototype for global log message filter based on attributes inside a LogMeta.
*/
typedef bool(*LogFilterFunctionType) (const ::cv::utils::logging::LogMeta&);

/**
Sets global log message filter based on attributes provided in a LogMeta.
*/
CV_EXPORTS LogFilterFunctionType setLogFilter(LogFilterFunctionType filter);

/**
Gets the current global log message filter.
*/
CV_EXPORTS LogFilterFunctionType getLogFilter();

namespace internal {
/** Write log message */
CV_EXPORTS void writeLogMessage(LogLevel logLevel, const char* message);
} // namespace

/**
 * \def CV_LOG_STRIP_LEVEL
 *
 * Define CV_LOG_STRIP_LEVEL=CV_LOG_LEVEL_[DEBUG|INFO|WARN|ERROR|FATAL|DISABLED] to compile out anything at that and before that logging level
 */
#ifndef CV_LOG_STRIP_LEVEL
# if defined NDEBUG
#   define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG
# else
#   define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE
# endif
#endif

/**
@todo Decide whether we actually want to keep the "tag" argument in the new macro.
@todo Decide whether the for(;;) { ...; break; } encasing is still needed.
@remark Apparently the for-break encasing allows short-circuit evaluation inside the macro.
*/
#define CV_LOGMETA(tag, level, metaArgs, ...) \
    for(;;) { \
        ::cv::utils::logging::LogMeta meta_; \
        meta_ << ::cv::utils::logging::LogLevel(level) << metaArgs; \
        if (!((::cv::utils::logging::getLogFilter())(meta_))) break; \
        std::stringstream ss; \
        ss << meta_; \
        ss << __VA_ARGS__; \
        cv::utils::logging::internal::writeLogMessage(level, ss.str().c_str()); \
        break; \
    }

/**
Collects the line-of-code information so that it can be added to the log message metadata.
*/
#define CV_LOG_LOC() (::cv::utils::logging::LogLoc(__FILE__, __LINE__, CV_Func))

/**
Collects the type name and address of the current object (via "this") so that it can be added
to the log message metadata. This macro is only valid when inside a class non-static member
function.
*/
#define CV_LOG_THIS() (::cv::utils::logging::LogObjInfo(*this))

#define CV_LOG_FATAL(tag, ...) CV_LOGMETA(tag, cv::utils::logging::LOG_LEVEL_FATAL, nullptr, __VA_ARGS__)
#define CV_LOG_ERROR(tag, ...) CV_LOGMETA(tag, cv::utils::logging::LOG_LEVEL_ERROR, nullptr, __VA_ARGS__)
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_WARN
#define CV_LOG_WARNING(tag, ...)
#else
#define CV_LOG_WARNING(tag, ...) CV_LOGMETA(tag, cv::utils::logging::LOG_LEVEL_WARNING, nullptr, __VA_ARGS__)
#endif
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_INFO
#define CV_LOG_INFO(tag, ...)
#else
#define CV_LOG_INFO(tag, ...) CV_LOGMETA(tag, cv::utils::logging::LOG_LEVEL_INFO, nullptr, __VA_ARGS__)
#endif
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_DEBUG
#define CV_LOG_DEBUG(tag, ...)
#else
#define CV_LOG_DEBUG(tag, ...) CV_LOGMETA(tag, cv::utils::logging::LOG_LEVEL_DEBUG, nullptr, __VA_ARGS__)
#endif
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_VERBOSE
#define CV_LOG_VERBOSE(tag, v, ...)
#else
#define CV_LOG_VERBOSE(tag, v, ...) CV_LOGMETA(tag, cv::utils::logging::LOG_LEVEL_VERBOSE, ::cv::utils::logging::LogVerboseLevel(v), __VA_ARGS__)
#endif


}}} // namespace

//! @}

#endif // OPENCV_LOGGER_HPP
