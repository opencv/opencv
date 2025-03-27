// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_LOGGER_HPP
#define OPENCV_LOGGER_HPP

#include <iostream>
#include <sstream>
#include <limits.h> // INT_MAX

#include "logger.defines.hpp"
#include "logtag.hpp"

namespace cv {
namespace utils {
namespace logging {

//! @addtogroup core_logging
//! @{

/** Set global logging level
@return previous logging level
*/
CV_EXPORTS LogLevel setLogLevel(LogLevel logLevel);
/** Get global logging level */
CV_EXPORTS LogLevel getLogLevel();

CV_EXPORTS void registerLogTag(cv::utils::logging::LogTag* plogtag);

CV_EXPORTS void setLogTagLevel(const char* tag, cv::utils::logging::LogLevel level);

CV_EXPORTS cv::utils::logging::LogLevel getLogTagLevel(const char* tag);

namespace internal {

/** Get global log tag */
CV_EXPORTS cv::utils::logging::LogTag* getGlobalLogTag();

/** Write log message */
CV_EXPORTS void writeLogMessage(LogLevel logLevel, const char* message);

/** Write log message */
CV_EXPORTS void writeLogMessageEx(LogLevel logLevel, const char* tag, const char* file, int line, const char* func, const char* message);

/**
 * @brief Function pointer type for writeLogMessage. Used by replaceWriteLogMessage.
 */
typedef void (*WriteLogMessageFuncType)(LogLevel, const char*);

/**
 * @brief Function pointer type for writeLogMessageEx. Used by replaceWriteLogMessageEx.
 */
typedef void (*WriteLogMessageExFuncType)(LogLevel, const char*, const char*, int, const char*, const char*);

/**
 * @brief Replaces the OpenCV writeLogMessage function with a user-defined function.
 * @note The user-defined function must have the same signature as writeLogMessage.
 * @note The user-defined function must accept arguments that can be potentially null.
 * @note The user-defined function must be thread-safe, as OpenCV logging may be called
 *       from multiple threads.
 * @note The user-defined function must not perform any action that can trigger
 *       deadlocks or infinite loop. Many OpenCV functions are not re-entrant.
 * @note Once replaced, logs will not go through the OpenCV writeLogMessage function.
 * @note To restore, call this function with a nullptr.
 */
CV_EXPORTS void replaceWriteLogMessage(WriteLogMessageFuncType f);

/**
 * @brief Replaces the OpenCV writeLogMessageEx function with a user-defined function.
 * @note The user-defined function must have the same signature as writeLogMessage.
 * @note The user-defined function must accept arguments that can be potentially null.
 * @note The user-defined function must be thread-safe, as OpenCV logging may be called
 *       from multiple threads.
 * @note The user-defined function must not perform any action that can trigger
 *       deadlocks or infinite loop. Many OpenCV functions are not re-entrant.
 * @note Once replaced, logs will not go through any of the OpenCV logging functions
 *       such as writeLogMessage or writeLogMessageEx, until their respective restore
 *       methods are called.
 * @note To restore, call this function with a nullptr.
 */
CV_EXPORTS void replaceWriteLogMessageEx(WriteLogMessageExFuncType f);

} // namespace

struct LogTagAuto
    : public LogTag
{
    inline LogTagAuto(const char* _name, LogLevel _level)
        : LogTag(_name, _level)
    {
        registerLogTag(this);
    }
};

/**
 * \def CV_LOG_STRIP_LEVEL
 *
 * Define CV_LOG_STRIP_LEVEL=CV_LOG_LEVEL_[DEBUG|INFO|WARN|ERROR|FATAL|SILENT] to compile out anything at that and before that logging level
 */
#ifndef CV_LOG_STRIP_LEVEL
# if defined NDEBUG
#   define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG
# else
#   define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE
# endif
#endif

#define CV_LOGTAG_PTR_CAST(expr) static_cast<const cv::utils::logging::LogTag*>(expr)

// CV_LOGTAG_EXPAND_NAME is intended to be re-defined (undef and then define again)
// to allows logging users to use a shorter name argument when calling
// CV_LOG_WITH_TAG or its related macros such as CV_LOG_INFO.
//
// This macro is intended to modify the tag argument as a string (token), via
// preprocessor token pasting or metaprogramming techniques. A typical usage
// is to apply a prefix, such as
// ...... #define CV_LOGTAG_EXPAND_NAME(tag) cv_logtag_##tag
//
// It is permitted to re-define to a hard-coded expression, ignoring the tag.
// This would work identically like the CV_LOGTAG_FALLBACK macro.
//
// Important: When the logging macro is called with tag being NULL, a user-defined
// CV_LOGTAG_EXPAND_NAME may expand it into cv_logtag_0, cv_logtag_NULL, or
// cv_logtag_nullptr. Use with care. Also be mindful of C++ symbol redefinitions.
//
// If there is significant amount of logging code with tag being NULL, it is
// recommended to use (re-define) CV_LOGTAG_FALLBACK to inject locally a default
// tag at the beginning of a compilation unit, to minimize lines of code changes.
//
#define CV_LOGTAG_EXPAND_NAME(tag) tag

// CV_LOGTAG_FALLBACK is intended to be re-defined (undef and then define again)
// by any other compilation units to provide a log tag when the logging statement
// does not specify one. The macro needs to expand into a C++ expression that can
// be static_cast into (cv::utils::logging::LogTag*). Null (nullptr) is permitted.
#define CV_LOGTAG_FALLBACK nullptr

// CV_LOGTAG_GLOBAL is the tag used when a log tag is not specified in the logging
// statement nor the compilation unit. The macro needs to expand into a C++
// expression that can be static_cast into (cv::utils::logging::LogTag*). Must be
// non-null. Do not re-define.
#define CV_LOGTAG_GLOBAL cv::utils::logging::internal::getGlobalLogTag()

#define CV_LOG_WITH_TAG(tag, msgLevel, extra_check0, extra_check1, ...) \
    for(;;) { \
        extra_check0; \
        const auto cv_temp_msglevel = (cv::utils::logging::LogLevel)(msgLevel); \
        if (cv_temp_msglevel >= (CV_LOG_STRIP_LEVEL)) break; \
        auto cv_temp_logtagptr = CV_LOGTAG_PTR_CAST(CV_LOGTAG_EXPAND_NAME(tag)); \
        if (!cv_temp_logtagptr) cv_temp_logtagptr = CV_LOGTAG_PTR_CAST(CV_LOGTAG_FALLBACK); \
        if (!cv_temp_logtagptr) cv_temp_logtagptr = CV_LOGTAG_PTR_CAST(CV_LOGTAG_GLOBAL); \
        if (cv_temp_logtagptr && (cv_temp_msglevel > cv_temp_logtagptr->level)) break; \
        extra_check1; \
        std::stringstream cv_temp_logstream; \
        cv_temp_logstream << __VA_ARGS__; \
        cv::utils::logging::internal::writeLogMessageEx( \
            cv_temp_msglevel, \
            (cv_temp_logtagptr ? cv_temp_logtagptr->name : nullptr), \
            __FILE__, \
            __LINE__, \
            CV_Func, \
            cv_temp_logstream.str().c_str()); \
        break; \
    }

#define CV_LOG_FATAL(tag, ...)   CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_FATAL, , , __VA_ARGS__)
#define CV_LOG_ERROR(tag, ...)   CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_ERROR, , , __VA_ARGS__)
#define CV_LOG_WARNING(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_WARNING, , , __VA_ARGS__)
#define CV_LOG_INFO(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_INFO, , , __VA_ARGS__)
#define CV_LOG_DEBUG(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_DEBUG, , , __VA_ARGS__)
#define CV_LOG_VERBOSE(tag, v, ...) CV_LOG_WITH_TAG(tag, (cv::utils::logging::LOG_LEVEL_VERBOSE + (int)(v)), , , __VA_ARGS__)

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_INFO
#undef CV_LOG_INFO
#define CV_LOG_INFO(tag, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_DEBUG
#undef CV_LOG_DEBUG
#define CV_LOG_DEBUG(tag, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_VERBOSE
#undef CV_LOG_VERBOSE
#define CV_LOG_VERBOSE(tag, v, ...)
#endif

//! @cond IGNORED
#define CV__LOG_ONCE_CHECK_PRE \
    static bool _cv_log_once_ ## __LINE__ = false; \
    if (_cv_log_once_ ## __LINE__) break;

#define CV__LOG_ONCE_CHECK_POST \
    _cv_log_once_ ## __LINE__ = true;

#define CV__LOG_IF_CHECK(logging_cond) \
    if (!(logging_cond)) break;

//! @endcond


// CV_LOG_ONCE_XXX macros

#define CV_LOG_ONCE_ERROR(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_ERROR, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, __VA_ARGS__)
#define CV_LOG_ONCE_WARNING(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_WARNING, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, __VA_ARGS__)
#define CV_LOG_ONCE_INFO(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_INFO, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, __VA_ARGS__)
#define CV_LOG_ONCE_DEBUG(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_DEBUG, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, __VA_ARGS__)
#define CV_LOG_ONCE_VERBOSE(tag, v, ...) CV_LOG_WITH_TAG(tag, (cv::utils::logging::LOG_LEVEL_VERBOSE + (int)(v)), CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, __VA_ARGS__)

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_INFO
#undef CV_LOG_ONCE_INFO
#define CV_LOG_ONCE_INFO(tag, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_DEBUG
#undef CV_LOG_ONCE_DEBUG
#define CV_LOG_ONCE_DEBUG(tag, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_VERBOSE
#undef CV_LOG_ONCE_VERBOSE
#define CV_LOG_ONCE_VERBOSE(tag, v, ...)
#endif


// CV_LOG_IF_XXX macros

#define CV_LOG_IF_FATAL(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_FATAL, , CV__LOG_IF_CHECK(logging_cond), __VA_ARGS__)
#define CV_LOG_IF_ERROR(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_ERROR, , CV__LOG_IF_CHECK(logging_cond), __VA_ARGS__)
#define CV_LOG_IF_WARNING(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_WARNING, , CV__LOG_IF_CHECK(logging_cond), __VA_ARGS__)
#define CV_LOG_IF_INFO(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_INFO, , CV__LOG_IF_CHECK(logging_cond), __VA_ARGS__)
#define CV_LOG_IF_DEBUG(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_DEBUG, , CV__LOG_IF_CHECK(logging_cond), __VA_ARGS__)
#define CV_LOG_IF_VERBOSE(tag, v, logging_cond, ...) CV_LOG_WITH_TAG(tag, (cv::utils::logging::LOG_LEVEL_VERBOSE + (int)(v)), , CV__LOG_IF_CHECK(logging_cond), __VA_ARGS__)

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_INFO
#undef CV_LOG_IF_INFO
#define CV_LOG_IF_INFO(tag, logging_cond, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_DEBUG
#undef CV_LOG_IF_DEBUG
#define CV_LOG_IF_DEBUG(tag, logging_cond, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_VERBOSE
#undef CV_LOG_IF_VERBOSE
#define CV_LOG_IF_VERBOSE(tag, v, logging_cond, ...)
#endif


//! @}

}}} // namespace

#endif // OPENCV_LOGGER_HPP
