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
