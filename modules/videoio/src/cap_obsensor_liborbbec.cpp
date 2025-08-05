// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
* Copyright(C) 2024 by ORBBEC Technology., Inc.
* Authors:
*   Huang Zhenchang <yufeng@orbbec.com>
*   Yu Shuai <daiyin@orbbec.com>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "precomp.hpp"

#if defined(HAVE_OBSENSOR) && defined(HAVE_OBSENSOR_ORBBEC_SDK)
#include "libobsensor/ObSensor.hpp"
#include "cap_obsensor_liborbbec.hpp"

namespace cv
{
Ptr<IVideoCapture> create_obsensor_capture(int index)
{
    return makePtr<VideoCapture_obsensor>(index);
}

VideoCapture_obsensor::VideoCapture_obsensor(int)
{
    ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
    config = std::make_shared<ob::Config>();
    pipe = std::make_shared<ob::Pipeline>();
    auto colorProfiles = pipe->getStreamProfileList(OB_SENSOR_COLOR);
    auto colorProfile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
    config->enableStream(colorProfile->as<ob::VideoStreamProfile>());

    auto depthProfiles = pipe->getStreamProfileList(OB_SENSOR_DEPTH);
    auto depthProfile = depthProfiles->getProfile(OB_PROFILE_DEFAULT);
    config->enableStream(depthProfile->as<ob::VideoStreamProfile>());

    config->setAlignMode(ALIGN_D2C_SW_MODE);

    pipe->start(config, [&](std::shared_ptr<ob::FrameSet> frameset) {
        std::unique_lock<std::mutex> lk(videoFrameMutex);
        colorFrame = frameset->colorFrame();
        depthFrame = frameset->depthFrame();
    });

    auto param = pipe->getCameraParam();
    camParam.p1[0] = param.rgbIntrinsic.fx;
    camParam.p1[1] = param.rgbIntrinsic.fy;
    camParam.p1[2] = param.rgbIntrinsic.cx;
    camParam.p1[3] = param.rgbIntrinsic.cy;
}

VideoCapture_obsensor::~VideoCapture_obsensor(){
    pipe->stop();
}

double VideoCapture_obsensor::getProperty(int propIdx) const
{
    double rst = 0.0;
    propIdx = propIdx & (~CAP_OBSENSOR_GENERATORS_MASK);
    switch (propIdx)
    {
    case CAP_PROP_OBSENSOR_INTRINSIC_FX:
        rst = camParam.p1[0];
        break;
    case CAP_PROP_OBSENSOR_INTRINSIC_FY:
        rst = camParam.p1[1];
        break;
    case CAP_PROP_OBSENSOR_INTRINSIC_CX:
        rst = camParam.p1[2];
        break;
    case CAP_PROP_OBSENSOR_INTRINSIC_CY:
        rst = camParam.p1[3];
        break;
    case CAP_PROP_POS_MSEC:
    case CAP_PROP_OBSENSOR_RGB_POS_MSEC:
        if (grabbedColorFrame)
        {
            rst = grabbedColorFrame->globalTimeStampUs();
            if (rst == 0.0)
            {
                CV_LOG_ONCE_WARNING(NULL, "Camera reports zero global timestamp. System timestamp is used instead.");
                rst = grabbedColorFrame->systemTimeStamp();
            }
        }
        break;
    case CAP_PROP_OBSENSOR_DEPTH_POS_MSEC:
        if (grabbedDepthFrame)
        {
            rst = grabbedDepthFrame->systemTimeStamp();
            if (rst == 0.0)
            {
                CV_LOG_ONCE_WARNING(NULL, "Camera reports zero global timestamp. System timestamp is used instead.");
                rst = grabbedDepthFrame->systemTimeStamp();
            }
        }
        break;
    }

    return rst;
}

bool VideoCapture_obsensor::setProperty(int, double)
{
    return false;
}

bool VideoCapture_obsensor::grabFrame()
{
    std::unique_lock<std::mutex> lk(videoFrameMutex);
    grabbedColorFrame = colorFrame;
    grabbedDepthFrame = depthFrame;

    return grabbedColorFrame || grabbedDepthFrame;
}

bool VideoCapture_obsensor::retrieveFrame(int outputType, cv::OutputArray frame)
{
    switch (outputType)
    {
    case CAP_OBSENSOR_BGR_IMAGE:
        if(grabbedColorFrame != nullptr){
            auto format = grabbedColorFrame->format();
            if(format != OB_FORMAT_MJPEG){
                CV_LOG_WARNING(NULL, "Unsupported color frame format");
                return false;
            }
            auto mjpgMat = Mat(1, grabbedColorFrame->dataSize() , CV_8UC1, grabbedColorFrame->data()).clone();
            auto bgrMat = imdecode(mjpgMat, IMREAD_COLOR);
            if(bgrMat.empty()){
                CV_LOG_WARNING(NULL, "Failed to decode color frame");
                return false;
            }
            bgrMat.copyTo(frame);
            return true;
        }
        break;
    case CAP_OBSENSOR_DEPTH_MAP:
        if(grabbedDepthFrame != nullptr){
            auto format = grabbedDepthFrame->format();
            if(format != OB_FORMAT_Y16){
                CV_LOG_WARNING(NULL, "Unsupported depth frame format");
                return false;
            }
            Mat(grabbedDepthFrame->height(), grabbedDepthFrame->width(), CV_16UC1, grabbedDepthFrame->data()).copyTo(frame);
            return true;
        }
        break;
    default:
        return false;
    }

    return false;
}

int VideoCapture_obsensor::getCaptureDomain()
{
    return CAP_OBSENSOR;
}

bool VideoCapture_obsensor::isOpened() const
{
    return true;
}

}
#endif
