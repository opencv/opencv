// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
* Copyright(C) 2022 by ORBBEC Technology., Inc.
* Authors:
*   Huang Zhenchang <yufeng@orbbec.com>
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

#ifndef OPENCV_VIDEOIO_CAP_OBSENSOR_CAPTURE_HPP
#define OPENCV_VIDEOIO_CAP_OBSENSOR_CAPTURE_HPP

#include <map>
#include <mutex>
#include <condition_variable>

#include "cap_obsensor/obsensor_stream_channel_interface.hpp"

#if defined(HAVE_OBSENSOR) && !defined(HAVE_OBSENSOR_ORBBEC_SDK)

namespace cv {
class VideoCapture_obsensor : public IVideoCapture
{
public:
    VideoCapture_obsensor(int index);
    virtual ~VideoCapture_obsensor();

    virtual double getProperty(int propIdx) const CV_OVERRIDE;
    virtual bool setProperty(int propIdx, double propVal) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int outputType, OutputArray frame) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE {
        return CAP_OBSENSOR;
    }
    virtual bool isOpened() const CV_OVERRIDE {
        return isOpened_;
    }

private:
    bool isOpened_;
    std::vector<Ptr<obsensor::IStreamChannel>> streamChannelGroup_;

    std::mutex frameMutex_;
    std::condition_variable frameCv_;

    Mat depthFrame_;
    Mat colorFrame_;

    Mat grabbedDepthFrame_;
    Mat grabbedColorFrame_;

    obsensor::CameraParam camParam_;
    int camParamScale_;
};
} // namespace cv::
#endif // HAVE_OBSENSOR
#endif // OPENCV_VIDEOIO_CAP_OBSENSOR_CAPTURE_HPP
