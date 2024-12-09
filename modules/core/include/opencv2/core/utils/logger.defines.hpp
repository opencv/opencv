// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_LOGGER_DEFINES_HPP
#define OPENCV_LOGGER_DEFINES_HPP

//! @addtogroup core_logging
//! @{

// Supported logging levels and their semantic
#define CV_LOG_LEVEL_SILENT 0          //!< for using in setLogLevel() call
#define CV_LOG_LEVEL_FATAL 1           //!< Fatal (critical) error (unrecoverable internal error)
#define CV_LOG_LEVEL_ERROR 2           //!< Error message
#define CV_LOG_LEVEL_WARN 3            //!< Warning message
#define CV_LOG_LEVEL_INFO 4            //!< Info message
#define CV_LOG_LEVEL_DEBUG 5           //!< Debug message. Disabled in the "Release" build.
#define CV_LOG_LEVEL_VERBOSE 6         //!< Verbose (trace) messages. Requires verbosity level. Disabled in the "Release" build.

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

}}} // namespace

//! @}

#endif // OPENCV_LOGGER_DEFINES_HPP
