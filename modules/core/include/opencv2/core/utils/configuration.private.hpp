// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CONFIGURATION_PRIVATE_HPP
#define OPENCV_CONFIGURATION_PRIVATE_HPP

#include "opencv2/core/cvstd.hpp"
#include <vector>
#include <string>

namespace cv { namespace utils {

typedef std::vector<std::string> Paths;
CV_EXPORTS bool getConfigurationParameterBool(const char* name, bool defaultValue = false);
CV_EXPORTS size_t getConfigurationParameterSizeT(const char* name, size_t defaultValue = 0);
CV_EXPORTS std::string getConfigurationParameterString(const char* name, const std::string & defaultValue = std::string());
CV_EXPORTS Paths getConfigurationParameterPaths(const char* name, const Paths &defaultValue = Paths());

}} // namespace

#endif // OPENCV_CONFIGURATION_PRIVATE_HPP
