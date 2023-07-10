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

#ifndef OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_INTERFACE_HPP
#define OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_INTERFACE_HPP

#ifdef HAVE_OBSENSOR

#include "../precomp.hpp" // #include "precomp.hpp : compile error on linux

#include <functional>
#include <vector>
#include <memory>

namespace cv {
namespace obsensor {

#define OBSENSOR_CAM_VID 0x2bc5 // usb vid
#define OBSENSOR_ASTRA2_PID 0x0660 // pid of Orbbec Astra 2 Camera
#define OBSENSOR_GEMINI2_PID 0x0670 // pid of Orbbec Gemini 2 Camera
#define OBSENSOR_FEMTO_MEGA_PID 0x0669 // pid of Orbbec Femto Mega Camera

enum StreamType
{
    OBSENSOR_STREAM_IR = 1,
    OBSENSOR_STREAM_COLOR = 2,
    OBSENSOR_STREAM_DEPTH = 3,
};

enum FrameFormat
{
    FRAME_FORMAT_UNKNOWN = -1,
    FRAME_FORMAT_YUYV = 0,
    FRAME_FORMAT_MJPG = 5,
    FRAME_FORMAT_Y16 = 8,
    FRAME_FORMAT_Y14 = 9,
};

enum PropertyId
{
    DEPTH_TO_COLOR_ALIGN = 42,
    CAMERA_PARAM = 1001,
};

struct Frame
{
    FrameFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t dataSize;
    uint8_t* data;
};

struct StreamProfile
{
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    FrameFormat format;
};

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

typedef std::function<void(Frame*)> FrameCallback;
class IStreamChannel
{
public:
    virtual ~IStreamChannel() noexcept {}
    virtual void start(const StreamProfile& profile, FrameCallback frameCallback) = 0;
    virtual void stop() = 0;
    virtual bool setProperty(int propId, const uint8_t* data, uint32_t dataSize) = 0;
    virtual bool getProperty(int propId, uint8_t* recvData, uint32_t* recvDataSize) = 0;

    virtual StreamType streamType() const = 0;
    virtual uint16_t getPid() const =0;
};

// "StreamChannelGroup" mean a group of stream channels from same one physical device
std::vector<Ptr<IStreamChannel>> getStreamChannelGroup(uint32_t groupIdx = 0);

}} // namespace cv::obsensor::
#endif // HAVE_OBSENSOR
#endif // OPENCV_VIDEOIO_OBSENSOR_STREAM_CHANNEL_INTERFACE_HPP
