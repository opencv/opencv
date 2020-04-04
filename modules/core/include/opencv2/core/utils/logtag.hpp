// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_LOGTAG_HPP
#define OPENCV_CORE_LOGTAG_HPP

#include "opencv2/core/cvstd.hpp"
#include "logger.defines.hpp"

namespace cv {
namespace utils {
namespace logging {

struct LogTag
{
    const char* name;
    LogLevel level;

    inline LogTag(const char* _name, LogLevel _level)
        : name(_name)
        , level(_level)
    {}
};

}}}

#endif
