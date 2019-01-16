// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_LOGGER_HPP
#define OPENCV_LOGGER_HPP

#include <iostream>
#include <sstream>
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

#if 1 // TENTATIVE

// Macros that expand to default declarations of special class methods

// DC: Default (argumentless) constructor
#define DECLARE_MEMBER_DEFAULTS_DC(ClassName) ClassName() = default;

// CC: Copy constructor
#define DECLARE_MEMBER_DEFAULTS_CC(ClassName) ClassName(const ClassName&) = default;

// MC: Move constructor
#define DECLARE_MEMBER_DEFAULTS_MC(ClassName) ClassName(ClassName&&) = default;

// CA: Copy assignment operator
#define DECLARE_MEMBER_DEFAULTS_CA(ClassName) ClassName& operator = (const ClassName&) = default;

// MA: Move assignment operator
#define DECLARE_MEMBER_DEFAULTS_MA(ClassName) ClassName& operator = (ClassName&&) = default;

// ALL: All of above
#define DECLARE_MEMBER_DEFAULTS_ALL(ClassName) \
    DECLARE_MEMBER_DEFAULTS_DC(ClassName) \
    DECLARE_MEMBER_DEFAULTS_CC(ClassName) \
    DECLARE_MEMBER_DEFAULTS_MC(ClassName) \
    DECLARE_MEMBER_DEFAULTS_CA(ClassName) \
    DECLARE_MEMBER_DEFAULTS_MA(ClassName)

#endif // TENTATIVE

/**
Information about the module that generates a log message.
*/
class LogModuleInfo
{
public:
    DECLARE_MEMBER_DEFAULTS_ALL(LogModuleInfo);

public:
    /**
    Name of the OpenCV module or component.
    Might be nullptr. If not nullptr, the string needs to be static lifetime or const literal.
    */
    const char* name;

public:
    LogModuleInfo(const char* name_)
        : name(name_)
    {
    }
};

/**
Information about the current object (the "this" pointer) collected alongside each log message.
*/
class LogObjInfo
{
public:
    DECLARE_MEMBER_DEFAULTS_ALL(LogObjInfo);

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
class LogLoc
{
public:
    DECLARE_MEMBER_DEFAULTS_ALL(LogLoc);

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
class LogPar
{
public:
    DECLARE_MEMBER_DEFAULTS_ALL(LogPar);

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
class LogVerboseLevel
{
public:
    DECLARE_MEMBER_DEFAULTS_ALL(LogVerboseLevel);

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
class LogMeta
{
public:
    DECLARE_MEMBER_DEFAULTS_CC(LogMeta);
    DECLARE_MEMBER_DEFAULTS_MC(LogMeta);
    DECLARE_MEMBER_DEFAULTS_CA(LogMeta);
    DECLARE_MEMBER_DEFAULTS_MA(LogMeta);

public:
    LogLevel level;
    int verboseLevel;
    const char* moduleName;
    int thread;
    LogObjInfo objInfo;
    LogLoc loc;

#if 0 // TENTATIVE
    LogPar par;
#endif // TENTATIVE

public:
    LogMeta(LogLevel level_, const char* moduleName_ = nullptr, int thread_ = cv::utils::getThreadID())
        : level(level_)
        , verboseLevel()
        , moduleName(moduleName_)
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
    Updates the module info via the stream insertion operator.
    */
    LogMeta& operator << (const LogModuleInfo& moduleInfo)
    {
        this->moduleName = moduleInfo.name;
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

#if 1 // TENTATIVE
/**
Serializes the log message metadata to a string that is inserted into an std::ostream.
*/
inline std::ostream& operator << (std::ostream& o, const LogMeta& meta)
{
    if (meta.moduleName && meta.moduleName[0])
    {
        o << "(module:" << meta.moduleName << ") ";
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
#endif // TENTATIVE

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
