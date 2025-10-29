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
Ptr<IVideoCapture> create_obsensor_capture(int index, const cv::VideoCaptureParameters& params)
{
    return makePtr<VideoCapture_obsensor>(index, params);
}

VideoCapture_obsensor::VideoCapture_obsensor(int, const cv::VideoCaptureParameters& params)
{
    ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
    config = std::make_shared<ob::Config>();
    pipe = std::make_shared<ob::Pipeline>();
#if ORBBEC_SDK_VERSION_MAJOR != 1
    alignFilter = std::make_shared<ob::Align>(OB_STREAM_COLOR);
#endif

    int color_width = params.get<double>(CAP_PROP_FRAME_WIDTH, OB_WIDTH_ANY);
    int color_height = params.get<double>(CAP_PROP_FRAME_HEIGHT, OB_HEIGHT_ANY);
    int color_fps = params.get<double>(CAP_PROP_FPS, OB_FPS_ANY);

    auto colorProfiles = pipe->getStreamProfileList(OB_SENSOR_COLOR);
    if (color_width == OB_WIDTH_ANY && color_height == OB_HEIGHT_ANY && color_fps == OB_FPS_ANY)
    {
        CV_LOG_INFO(NULL, "Use default color stream profile");
        auto colorProfile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
        config->enableStream(colorProfile->as<ob::VideoStreamProfile>());
    }
    else
    {
        CV_LOG_INFO(NULL, "Looking for custom color profile " << color_width << "x" << color_height << "@" << color_fps << " fps");
        auto colorProfile = colorProfiles->getVideoStreamProfile(color_width, color_height, OB_FORMAT_MJPG, color_fps);
        config->enableStream(colorProfile->as<ob::VideoStreamProfile>());
    }

    int depth_width = params.get<double>(CAP_PROP_OBSENSOR_DEPTH_WIDTH, OB_WIDTH_ANY);
    int depth_height = params.get<double>(CAP_PROP_OBSENSOR_DEPTH_HEIGHT, OB_HEIGHT_ANY);
    int depth_fps = params.get<double>(CAP_PROP_OBSENSOR_DEPTH_FPS, OB_FPS_ANY);

    auto depthProfiles = pipe->getStreamProfileList(OB_SENSOR_DEPTH);
    if (depth_width == OB_WIDTH_ANY && depth_height == OB_HEIGHT_ANY && depth_fps == OB_FPS_ANY)
    {
        CV_LOG_INFO(NULL, "Use default depth stream profile");
        auto depthProfile = depthProfiles->getProfile(OB_PROFILE_DEFAULT);
        config->enableStream(depthProfile->as<ob::VideoStreamProfile>());
    }
    else
    {
        CV_LOG_INFO(NULL, "Looking for custom color profile " << depth_width << "x" << depth_height << "@" << depth_fps << " fps");
        auto depthProfile = depthProfiles->getVideoStreamProfile(depth_width, depth_height, OB_FORMAT_Y14, depth_fps);
        config->enableStream(depthProfile->as<ob::VideoStreamProfile>());
    }

#if ORBBEC_SDK_VERSION_MAJOR == 1
    config->setAlignMode(ALIGN_D2C_SW_MODE);
#else
    config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    pipe->enableFrameSync();
#endif

    pipe->start(config, [&](std::shared_ptr<ob::FrameSet> frameset) {
        std::unique_lock<std::mutex> lk(videoFrameMutex);
#if ORBBEC_SDK_VERSION_MAJOR == 1
        colorFrame = frameset->colorFrame();
        depthFrame = frameset->depthFrame();
#else
        auto alignFrameSet = alignFilter->process(frameset);
        if (alignFrameSet) {
            colorFrame = alignFrameSet->as<ob::FrameSet>()->colorFrame();
            depthFrame = alignFrameSet->as<ob::FrameSet>()->depthFrame();
        }
#endif
    });

    auto param = pipe->getCameraParam();
    camParam.intrinsicColor[0] = param.rgbIntrinsic.fx;
    camParam.intrinsicColor[1] = param.rgbIntrinsic.fy;
    camParam.intrinsicColor[2] = param.rgbIntrinsic.cx;
    camParam.intrinsicColor[3] = param.rgbIntrinsic.cy;

    camParam.distortionColor[0] = param.depthDistortion.k1;
    camParam.distortionColor[1] = param.depthDistortion.k2;
    camParam.distortionColor[2] = param.depthDistortion.k3;
    camParam.distortionColor[3] = param.depthDistortion.k4;
    camParam.distortionColor[4] = param.depthDistortion.k5;
    camParam.distortionColor[5] = param.depthDistortion.k6;
    camParam.distortionColor[6] = param.depthDistortion.p1;
    camParam.distortionColor[7] = param.depthDistortion.p2;
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
        rst = camParam.intrinsicColor[0];
        break;
    case CAP_PROP_OBSENSOR_INTRINSIC_FY:
        rst = camParam.intrinsicColor[1];
        break;
    case CAP_PROP_OBSENSOR_INTRINSIC_CX:
        rst = camParam.intrinsicColor[2];
        break;
    case CAP_PROP_OBSENSOR_INTRINSIC_CY:
        rst = camParam.intrinsicColor[3];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_K1:
        rst = camParam.distortionColor[0];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_K2:
        rst = camParam.distortionColor[1];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_K3:
        rst = camParam.distortionColor[2];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_K4:
        rst = camParam.distortionColor[3];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_K5:
        rst = camParam.distortionColor[4];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_K6:
        rst = camParam.distortionColor[5];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_P1:
        rst = camParam.distortionColor[6];
        break;
    case CAP_PROP_OBSENSOR_COLOR_DISTORTION_P2:
        rst = camParam.distortionColor[7];
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

bool VideoCapture_obsensor::setProperty(int prop, double)
{
    switch(prop)
    {
        case CAP_PROP_OBSENSOR_DEPTH_WIDTH:
        case CAP_PROP_OBSENSOR_DEPTH_HEIGHT:
        case CAP_PROP_OBSENSOR_DEPTH_FPS:
            CV_LOG_WARNING(NULL, "CAP_PROP_OBSENSOR_DEPTH_WIDTH, CAP_PROP_OBSENSOR_DEPTH_HEIGHT, CAP_PROP_OBSENSOR_DEPTH_FPS options are supported during camera initialization only");
            break;
        case CAP_PROP_FRAME_WIDTH:
        case CAP_PROP_FRAME_HEIGHT:
        case CAP_PROP_FPS:
            CV_LOG_WARNING(NULL, "CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS options are supported during camera initialization only");
            break;
    }

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
