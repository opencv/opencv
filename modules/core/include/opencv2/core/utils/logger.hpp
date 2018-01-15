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


}}} // namespace

//! @}

#endif // OPENCV_LOGGER_HPP
