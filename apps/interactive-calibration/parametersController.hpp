// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef PARAMETERS_CONTROLLER_HPP
#define PARAMETERS_CONTROLLER_HPP

#include <string>

#include <opencv2/core.hpp>

#include "calibCommon.hpp"

namespace calib {

class parametersController
{
protected:
    captureParameters mCapParams;
    internalParameters mInternalParameters;

    bool loadFromFile(const std::string& inputFileName);
public:
    parametersController();
    parametersController(cv::Ptr<captureParameters> params);

    captureParameters getCaptureParameters() const;
    internalParameters getInternalParameters() const;

    bool loadFromParser(cv::CommandLineParser& parser);
};

}

#endif
