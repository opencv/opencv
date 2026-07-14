// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_LOGTAGCONFIG_HPP
#define OPENCV_CORE_LOGTAGCONFIG_HPP

#if 1 // if not already in precompiled headers
#include <opencv2/core/utils/logger.defines.hpp>
#include <string>
#endif

namespace cv {
namespace utils {
namespace logging {

struct LogTagConfig
{
    std::string namePart;
    LogLevel level;
    bool isGlobal;
    bool hasPrefixWildcard;
    bool hasSuffixWildcard;

    LogTagConfig()
        : namePart()
        , level()
        , isGlobal()
        , hasPrefixWildcard()
        , hasSuffixWildcard()
    {
    }

    LogTagConfig(const std::string& _namePart, LogLevel _level, bool _isGlobal = false,
        bool _hasPrefixWildcard = false, bool _hasSuffixWildcard = false)
        : namePart(_namePart)
        , level(_level)
        , isGlobal(_isGlobal)
        , hasPrefixWildcard(_hasPrefixWildcard)
        , hasSuffixWildcard(_hasSuffixWildcard)
    {
    }

    LogTagConfig(const LogTagConfig&) = default;
    LogTagConfig(LogTagConfig&&) = default;
    ~LogTagConfig() = default;
};

}}}

#endif
