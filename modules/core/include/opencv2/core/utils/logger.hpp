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

//! @addtogroup core_logging
// This section describes OpenCV logging utilities.
//
//! @{

namespace cv {
namespace utils {
namespace logging {

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

} // namespace

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

#define CV_LOG_WITH_TAG(tag, msgLevel, ...) \
    for(;;) { \
        const auto cv_temp_msglevel = (cv::utils::logging::LogLevel)(msgLevel); \
        if (cv_temp_msglevel >= (CV_LOG_STRIP_LEVEL)) break; \
        auto cv_temp_logtagptr = (const cv::utils::logging::LogTag*)(tag); \
        if (!cv_temp_logtagptr) cv_temp_logtagptr = (const cv::utils::logging::LogTag*)(CV_LOGTAG_FALLBACK); \
        if (!cv_temp_logtagptr) cv_temp_logtagptr = (const cv::utils::logging::LogTag*)(CV_LOGTAG_GLOBAL); \
        if (cv_temp_logtagptr && (cv_temp_msglevel > cv_temp_logtagptr->level)) break; \
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

#define CV_LOG_FATAL(tag, ...)   CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_FATAL, __VA_ARGS__)
#define CV_LOG_ERROR(tag, ...)   CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_ERROR, __VA_ARGS__)
#define CV_LOG_WARNING(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_WARNING, __VA_ARGS__)
#define CV_LOG_INFO(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_INFO, __VA_ARGS__)
#define CV_LOG_DEBUG(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define CV_LOG_VERBOSE(tag, v, ...) CV_LOG_WITH_TAG(tag, (cv::utils::logging::LOG_LEVEL_VERBOSE + (int)(v)), __VA_ARGS__)

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_INFO
# undef CV_LOG_INFO
# define CV_LOG_INFO(tag, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_DEBUG
# undef CV_LOG_DEBUG
# define CV_LOG_DEBUG(tag, ...)
#endif

#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_VERBOSE
# undef CV_LOG_VERBOSE
# define CV_LOG_VERBOSE(tag, v, ...)
#endif

#if 0
#define CV_LOG_FATAL(tag, ...)   for(;;) { if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_FATAL) break; std::stringstream ss; ss << __VA_ARGS__; cv::utils::logging::internal::writeLogMessage(cv::utils::logging::LOG_LEVEL_FATAL, ss.str().c_str()); break; }
#define CV_LOG_ERROR(tag, ...)   for(;;) { if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_ERROR) break; std::stringstream ss; ss << __VA_ARGS__; cv::utils::logging::internal::writeLogMessage(cv::utils::logging::LOG_LEVEL_ERROR, ss.str().c_str()); break; }
#define CV_LOG_WARNING(tag, ...) for(;;) { if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_WARNING) break; std::stringstream ss; ss << __VA_ARGS__; cv::utils::logging::internal::writeLogMessage(cv::utils::logging::LOG_LEVEL_WARNING, ss.str().c_str()); break; }
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_INFO
#define CV_LOG_INFO(tag, ...)
#else
#define CV_LOG_INFO(tag, ...)    for(;;) { if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_INFO) break; std::stringstream ss; ss << __VA_ARGS__; cv::utils::logging::internal::writeLogMessage(cv::utils::logging::LOG_LEVEL_INFO, ss.str().c_str()); break; }
#endif
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_DEBUG
#define CV_LOG_DEBUG(tag, ...)
#else
#define CV_LOG_DEBUG(tag, ...)   for(;;) { if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_DEBUG) break; std::stringstream ss; ss << __VA_ARGS__; cv::utils::logging::internal::writeLogMessage(cv::utils::logging::LOG_LEVEL_DEBUG, ss.str().c_str()); break; }
#endif
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_VERBOSE
#define CV_LOG_VERBOSE(tag, v, ...)
#else
#define CV_LOG_VERBOSE(tag, v, ...) for(;;) { if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_VERBOSE) break; std::stringstream ss; ss << "[VERB" << v << ":" << cv::utils::getThreadID() << "] " << __VA_ARGS__; cv::utils::logging::internal::writeLogMessage(cv::utils::logging::LOG_LEVEL_VERBOSE, ss.str().c_str()); break; }
#endif
#endif

}}} // namespace

//! @}

#endif // OPENCV_LOGGER_HPP
