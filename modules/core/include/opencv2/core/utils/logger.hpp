// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_LOGGING_HPP
#define OPENCV_LOGGING_HPP

#include <iostream>
#include <sstream>
#include <limits.h> // INT_MAX

// TODO This file contains just interface part with implementation stubs.

//! @addtogroup core_logging
// This section describes OpenCV logging utilities.
//
//! @{

namespace cv {
namespace utils {
namespace logging {

// Supported logging levels and their semantic
#define CV_LOG_LEVEL_SILENT 0          //!< for using in setLogVevel() call
#define CV_LOG_LEVEL_FATAL 1           //!< Fatal (critical) error (unrecoverable internal error)
#define CV_LOG_LEVEL_ERROR 2           //!< Error message
#define CV_LOG_LEVEL_WARN 3            //!< Warning message
#define CV_LOG_LEVEL_INFO 4            //!< Info message
#define CV_LOG_LEVEL_DEBUG 5           //!< Debug message. Disabled in the "Release" build.
#define CV_LOG_LEVEL_VERBOSE 6         //!< Verbose (trace) messages. Requires verbosity level. Disabled in the "Release" build.

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


#define CV_LOG_FATAL(tag, ...)   for(;;) { std::stringstream ss; ss << "[FATAL:" << cv::utils::getThreadID() << "] " << __VA_ARGS__ << std::endl; std::cerr << ss.str() << std::flush; break; }
#define CV_LOG_ERROR(tag, ...)   for(;;) { std::stringstream ss; ss << "[ERROR:" << cv::utils::getThreadID() << "] " << __VA_ARGS__ << std::endl; std::cerr << ss.str() << std::flush; break; }
#define CV_LOG_WARNING(tag, ...) for(;;) { std::stringstream ss; ss << "[ WARN:" << cv::utils::getThreadID() << "] " << __VA_ARGS__ << std::endl; std::cout << ss.str() << std::flush; break; }
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_INFO
#define CV_LOG_INFO(tag, ...)
#else
#define CV_LOG_INFO(tag, ...)    for(;;) { std::stringstream ss; ss << "[ INFO:" << cv::utils::getThreadID() << "] " << __VA_ARGS__ << std::endl; std::cout << ss.str(); break; }
#endif
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_DEBUG
#define CV_LOG_DEBUG(tag, ...)
#else
#define CV_LOG_DEBUG(tag, ...)   for(;;) { std::stringstream ss; ss << "[DEBUG:" << cv::utils::getThreadID() << "] " << __VA_ARGS__ << std::endl; std::cout << ss.str(); break; }
#endif
#if CV_LOG_STRIP_LEVEL <= CV_LOG_LEVEL_VERBOSE
#define CV_LOG_VERBOSE(tag, v, ...)
#else
#define CV_LOG_VERBOSE(tag, v, ...) for(;;) { std::stringstream ss; ss << "[VERB" << v << ":" << cv::utils::getThreadID() << "] " << __VA_ARGS__ << std::endl; std::cout << ss.str(); break; }
#endif


}}} // namespace

//! @}

#endif // OPENCV_LOGGING_HPP
