// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_LOGGER_HPP
#define OPENCV_LOGGER_HPP

#include <iostream>
#include <sstream>
#include <limits.h> // INT_MAX

#include "logger.defines.hpp"

namespace cv {
namespace utils {
namespace logging {

//! @addtogroup core_logging
//! @{

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

/** Set global logging level
@return previous logging level
*/
CV_EXPORTS LogLevel setLogLevel(LogLevel logLevel);
/** Get global logging level */
CV_EXPORTS LogLevel getLogLevel();

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

#define CV_LOG_WITH_TAG(tag, msgLevel, extra_check0, extra_check1, msg_prefix, ...) \
    for(;;) { \
        extra_check0; \
        if (cv::utils::logging::getLogLevel() < msgLevel) break; \
        extra_check1; \
        std::stringstream ss; ss msg_prefix <<  __VA_ARGS__; \
        cv::utils::logging::internal::writeLogMessage(msgLevel, ss.str().c_str()); \
        break; \
    }

#define CV_LOG_FATAL(tag, ...)   CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_FATAL, , , , __VA_ARGS__)
#define CV_LOG_ERROR(tag, ...)   CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_ERROR, , , , __VA_ARGS__)
#define CV_LOG_WARNING(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_WARNING, , , , __VA_ARGS__)
#define CV_LOG_INFO(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_INFO, , , , __VA_ARGS__)
#define CV_LOG_DEBUG(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_DEBUG, , , , __VA_ARGS__)
#define CV_LOG_VERBOSE(tag, v, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_VERBOSE, , , << "[VERB" << v << ":" << cv::utils::getThreadID() << "] ", __VA_ARGS__)

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

#define CV_LOG_ONCE_ERROR(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_ERROR, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, , __VA_ARGS__)
#define CV_LOG_ONCE_WARNING(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_WARNING, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, , __VA_ARGS__)
#define CV_LOG_ONCE_INFO(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_INFO, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, , __VA_ARGS__)
#define CV_LOG_ONCE_DEBUG(tag, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_DEBUG, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, , __VA_ARGS__)
#define CV_LOG_ONCE_VERBOSE(tag, v, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_VERBOSE, CV__LOG_ONCE_CHECK_PRE, CV__LOG_ONCE_CHECK_POST, << "[VERB" << v << ":" << cv::utils::getThreadID() << "] ", __VA_ARGS__)

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

#define CV_LOG_IF_FATAL(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_FATAL, , CV__LOG_IF_CHECK(logging_cond), , __VA_ARGS__)
#define CV_LOG_IF_ERROR(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_ERROR, , CV__LOG_IF_CHECK(logging_cond), , __VA_ARGS__)
#define CV_LOG_IF_WARNING(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_WARNING, , CV__LOG_IF_CHECK(logging_cond), , __VA_ARGS__)
#define CV_LOG_IF_INFO(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_INFO, , CV__LOG_IF_CHECK(logging_cond), , __VA_ARGS__)
#define CV_LOG_IF_DEBUG(tag, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_DEBUG, , CV__LOG_IF_CHECK(logging_cond), , __VA_ARGS__)
#define CV_LOG_IF_VERBOSE(tag, v, logging_cond, ...) CV_LOG_WITH_TAG(tag, cv::utils::logging::LOG_LEVEL_VERBOSE, , CV__LOG_IF_CHECK(logging_cond), << "[VERB" << v << ":" << cv::utils::getThreadID() << "] ", __VA_ARGS__)

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
