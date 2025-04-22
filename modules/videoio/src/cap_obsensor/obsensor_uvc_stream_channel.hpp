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

#ifndef OPENCV_VIDEOIO_OBSENSOR_UVC_STREAM_CHANNEL_HPP
#define OPENCV_VIDEOIO_OBSENSOR_UVC_STREAM_CHANNEL_HPP
#include "obsensor_stream_channel_interface.hpp"

#ifdef HAVE_OBSENSOR
namespace cv {
namespace obsensor {
#define XU_MAX_DATA_LENGTH 1024
#define XU_UNIT_ID_COMMON 4
#define XU_UNIT_ID_G330 3

struct UvcDeviceInfo
{
    std::string id; // uvc sub-device id
    std::string name;
    std::string uid; // parent usb device id
    uint16_t vid;
    uint16_t pid;
    uint16_t mi; // uvc interface index
};

enum StreamState
{
    STREAM_STOPED = 0, // stoped or ready
    STREAM_STARTING = 1,
    STREAM_STARTED = 2,
    STREAM_STOPPING = 3,
};
struct Guid {
    uint32_t data1;
    uint16_t data2, data3;
    uint8_t  data4[8];
};

struct ObExtensionUnit {
    uint8_t unit;
    Guid id;
};

StreamType parseUvcDeviceNameToStreamType(const std::string& devName);
FrameFormat frameFourccToFormat(uint32_t fourcc);
uint32_t frameFormatToFourcc(FrameFormat);

struct OBExtensionParam {
    float bl;
    float bl2;
    float pd;
    float ps;
};

struct OBHardwareD2CParams {
    float scale;
    int left;
    int top;
    int right;
    int bottom;
};

class IFrameProcessor{
public:
    virtual void process(Frame* frame) = 0;
    virtual ~IFrameProcessor() = default;
};

class DepthFrameProcessor: public IFrameProcessor {
public:
    DepthFrameProcessor(const OBExtensionParam& parma);
    virtual ~DepthFrameProcessor();
    virtual void process(Frame* frame) override;

private:
    const OBExtensionParam param_;
    uint16_t* lookUpTable_;
};

class HardwareD2CProcessor: public IFrameProcessor {
public:
    HardwareD2CProcessor(const OBHardwareD2CParams& param);
    virtual ~HardwareD2CProcessor() = default;
    virtual void process(Frame* frame) override;

private:
    const OBHardwareD2CParams param_;
};

class DepthFrameUnpacker: public IFrameProcessor {
public:
    DepthFrameUnpacker();
    virtual ~DepthFrameUnpacker();
    virtual void process(Frame* frame) override;
private:
    const uint32_t OUT_DATA_SIZE = 1280*800*2;
    uint8_t *outputDataBuf_;
};


class IUvcStreamChannel : public IStreamChannel {
public:
    IUvcStreamChannel(const UvcDeviceInfo& devInfo);
    virtual ~IUvcStreamChannel() noexcept {}

    virtual bool setProperty(int propId, const uint8_t* data, uint32_t dataSize) override;
    virtual bool getProperty(int propId, uint8_t* recvData, uint32_t* recvDataSize) override;
    virtual StreamType streamType() const override;
    virtual uint16_t getPid() const override;

protected:
    virtual bool setXu(uint8_t ctrl, const uint8_t* data, uint32_t len) = 0;
    virtual bool getXu(uint8_t ctrl, uint8_t** data, uint32_t* len) = 0;

    bool initDepthFrameProcessor();
    bool initHardwareD2CProcessor();

protected:
    const UvcDeviceInfo devInfo_;
    const ObExtensionUnit xuUnit_;
    StreamType streamType_;
    Ptr<IFrameProcessor> depthFrameProcessor_;
    Ptr<IFrameProcessor> hardwareD2CProcessor_;
};
}} // namespace cv::obsensor::
#endif // HAVE_OBSENSOR
#endif // OPENCV_VIDEOIO_OBSENSOR_UVC_STREAM_CHANNEL_HPP
