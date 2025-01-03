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

#ifndef _CAP_LIBORBBEC_HPP_
#define _CAP_LIBORBBEC_HPP_

#if defined(HAVE_OBSENSOR) && defined(HAVE_OBSENSOR_ORBBEC_SDK)

#include <libobsensor/ObSensor.hpp>
#include <mutex>

namespace cv
{

struct CameraParam
{
    float    p0[4];
    float    p1[4];
    float    p2[9];
    float    p3[3];
    float    p4[5];
    float    p5[5];
    uint32_t p6[2];
    uint32_t p7[2];
};

class VideoCapture_obsensor : public IVideoCapture
{
public:
    VideoCapture_obsensor(int index);
    virtual ~VideoCapture_obsensor();

    virtual double getProperty(int propIdx) const CV_OVERRIDE;
    virtual bool setProperty(int propIdx, double propVal) CV_OVERRIDE;

    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int outputType, OutputArray frame) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE;

protected:
    std::mutex                 videoFrameMutex;
    std::shared_ptr<ob::VideoFrame> colorFrame;
    std::shared_ptr<ob::VideoFrame> depthFrame;
    std::shared_ptr<ob::VideoFrame> grabbedColorFrame;
    std::shared_ptr<ob::VideoFrame> grabbedDepthFrame;
    std::shared_ptr<ob::Pipeline> pipe;
    std::shared_ptr<ob::Config> config;
    CameraParam camParam;
};

}

#endif
#endif
