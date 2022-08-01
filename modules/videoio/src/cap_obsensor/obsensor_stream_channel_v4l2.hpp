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

#ifndef OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_V4L2_HPP
#define OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_V4L2_HPP
#ifdef HAVE_OBSENSOR_V4L2

#include "obsensor_uvc_stream_channel.hpp"

#include <mutex>
#include <condition_variable>
#include <thread>

namespace cv {
namespace obsensor {
#define MAX_FRAME_BUFFER_NUM 4
struct V4L2FrameBuffer
{
    uint32_t length = 0;
    uint8_t* ptr = nullptr;
};

int xioctl(int fd, int req, void* arg);

class V4L2Context
{
public:
    ~V4L2Context() {}
    static V4L2Context& getInstance();

    std::vector<UvcDeviceInfo> queryUvcDeviceInfoList();
    Ptr<IStreamChannel> createStreamChannel(const UvcDeviceInfo& devInfo);

private:
    V4L2Context() noexcept {}
};

class V4L2StreamChannel : public IUvcStreamChannel
{
public:
    V4L2StreamChannel(const UvcDeviceInfo& devInfo);
    virtual ~V4L2StreamChannel() noexcept;

    virtual void start(const StreamProfile& profile, FrameCallback frameCallback) override;
    virtual void stop() override;

private:
    void grabFrame();

    virtual bool setXu(uint8_t ctrl, const uint8_t* data, uint32_t len) override;
    virtual bool getXu(uint8_t ctrl, uint8_t** data, uint32_t* len) override;

private:
    int devFd_;

    V4L2FrameBuffer frameBuffList[MAX_FRAME_BUFFER_NUM];

    StreamState streamState_;
    std::mutex streamStateMutex_;
    std::condition_variable streamStateCv_;

    std::thread grabFrameThread_;

    FrameCallback frameCallback_;
    StreamProfile currentProfile_;

    std::vector<uint8_t> xuRecvBuf_;
    std::vector<uint8_t> xuSendBuf_;
};
}} // namespace cv::obsensor::
#endif // HAVE_OBSENSOR_V4L2
#endif // OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_V4L2_HPP
