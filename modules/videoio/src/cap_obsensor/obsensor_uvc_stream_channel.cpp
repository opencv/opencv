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

#if defined(HAVE_OBSENSOR_V4L2) ||  defined(HAVE_OBSENSOR_MSMF)

#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#if defined(HAVE_OBSENSOR_V4L2)
#include "obsensor_stream_channel_v4l2.hpp"
#elif defined(HAVE_OBSENSOR_MSMF)
#include "obsensor_stream_channel_msmf.hpp"
#endif // HAVE_OBSENSOR_V4L2

namespace cv {
namespace obsensor {
const uint8_t OB_EXT_CMD0[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x52, 0x00, 0x5B, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD1[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x54, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD2[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x56, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD3[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x58, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD4[16] = { 0x47, 0x4d, 0x02, 0x00, 0x03, 0x00, 0x60, 0x00, 0xed, 0x03, 0x00, 0x00 };
const uint8_t OB_EXT_CMD5[16] = { 0x47, 0x4d, 0x02, 0x00, 0x03, 0x00, 0x62, 0x00, 0xe9, 0x03, 0x00, 0x00 };

#if defined(HAVE_OBSENSOR_V4L2)
#define fourCc2Int(a, b, c, d) \
    ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))
#elif defined(HAVE_OBSENSOR_MSMF)
#define fourCc2Int(a, b, c, d) \
    (((uint32_t)(a) <<24) | ((uint32_t)(b) << 16) | ((uint32_t)(c) << 8) | (uint32_t)(d))
#endif // HAVE_OBSENSOR_V4L2

const std::map<uint32_t, FrameFormat> fourccToOBFormat = {
    {fourCc2Int('Y', 'U', 'Y', '2'), FRAME_FORMAT_YUYV},
    {fourCc2Int('M', 'J', 'P', 'G'), FRAME_FORMAT_MJPG},
    {fourCc2Int('Y', '1', '6', ' '), FRAME_FORMAT_Y16},
};

StreamType parseUvcDeviceNameToStreamType(const std::string& devName)
{
    std::string uvcDevName = devName;
    for (size_t i = 0; i < uvcDevName.length(); i++)
    {
        uvcDevName[i] = (char)tolower(uvcDevName[i]);
    }
    if (uvcDevName.find(" depth") != std::string::npos)
    {
        return OBSENSOR_STREAM_DEPTH;
    }
    else if (uvcDevName.find(" ir") != std::string::npos)
    {
        return OBSENSOR_STREAM_IR;
    }

    return OBSENSOR_STREAM_COLOR; // else
}

FrameFormat frameFourccToFormat(uint32_t fourcc)
{
    for (const auto& item : fourccToOBFormat)
    {
        if (item.first == fourcc)
        {
            return item.second;
        }
    }
    return FRAME_FORMAT_UNKNOWN;
}

uint32_t frameFormatToFourcc(FrameFormat fmt)
{
    for (const auto& item : fourccToOBFormat)
    {
        if (item.second == fmt)
        {
            return item.first;
        }
    }
    return 0;
}

std::vector<Ptr<IStreamChannel>> getStreamChannelGroup(uint32_t groupIdx)
{
    std::vector<Ptr<IStreamChannel>> streamChannelGroup;

#if defined(HAVE_OBSENSOR_V4L2)
    auto& ctx = V4L2Context::getInstance();
#elif defined(HAVE_OBSENSOR_MSMF)
    auto& ctx = MFContext::getInstance();
#endif // HAVE_OBSENSOR_V4L2

    auto uvcDevInfoList = ctx.queryUvcDeviceInfoList();

    std::map<std::string, std::vector<UvcDeviceInfo>> uvcDevInfoGroupMap;

    auto devInfoIter = uvcDevInfoList.begin();
    while (devInfoIter != uvcDevInfoList.end())
    {
        if (devInfoIter->vid != OBSENSOR_CAM_VID)
        {
            devInfoIter = uvcDevInfoList.erase(devInfoIter); // drop it
            continue;
        }
        devInfoIter++;
    }

    if (!uvcDevInfoList.empty() && uvcDevInfoList.size() <= 3)
    {
        uvcDevInfoGroupMap.insert({ "default", uvcDevInfoList });
    }
    else {
        for (auto& devInfo : uvcDevInfoList)
        {
            // group by uid
            uvcDevInfoGroupMap[devInfo.uid].push_back(devInfo); // todo: group by sn
        }
    }

    if (uvcDevInfoGroupMap.size() > groupIdx)
    {
        auto uvcDevInfoGroupIter = uvcDevInfoGroupMap.begin();
        std::advance(uvcDevInfoGroupIter, groupIdx);
        for (const auto& devInfo : uvcDevInfoGroupIter->second)
        {
            streamChannelGroup.emplace_back(ctx.createStreamChannel(devInfo));
        }
    }
    else
    {
        CV_LOG_ERROR(NULL, "Camera index out of range");
    }
    return streamChannelGroup;
}

DepthFrameProcessor::DepthFrameProcessor(const OBExtensionParam& param) : param_(param)
{
    double tempValue = 0;
    double rstValue = 0;
    lookUpTable_ = new uint16_t[4096];
    memset(lookUpTable_, 0, 4096 * 2);
    for (uint16_t oriValue = 0; oriValue < 4096; oriValue++)
    {
        if (oriValue == 0)
        {
            continue;
        }
        tempValue = 200.375 - (double)oriValue / 8;
        rstValue = (double)param_.pd / (1 + tempValue * param_.ps / param_.bl) * 10;
        if ((rstValue >= 40) && (rstValue <= 10000) && rstValue < 65536)
        {
            lookUpTable_[oriValue] = (uint16_t)rstValue;
        }
    }
}

DepthFrameProcessor::~DepthFrameProcessor()
{
    delete[] lookUpTable_;
}

void DepthFrameProcessor::process(Frame* frame)
{
    uint16_t* data = (uint16_t*)frame->data;
    for (uint32_t i = 0; i < frame->dataSize / 2; i++)
    {
        data[i] = lookUpTable_[data[i] & 0x0fff];
    }
}

IUvcStreamChannel::IUvcStreamChannel(const UvcDeviceInfo& devInfo) :
    devInfo_(devInfo),
    streamType_(parseUvcDeviceNameToStreamType(devInfo_.name))
{

}

StreamType IUvcStreamChannel::streamType() const {
    return streamType_;
}

bool IUvcStreamChannel::setProperty(int propId, const uint8_t* /*data*/, uint32_t /*dataSize*/)
{
    uint8_t* rcvData;
    uint32_t rcvLen;
    bool rst = true;
    switch (propId)
    {
    case DEPTH_TO_COLOR_ALIGN:
        // todo: value filling
        rst &= setXu(2, OB_EXT_CMD0, sizeof(OB_EXT_CMD0));
        rst &= getXu(2, &rcvData, &rcvLen);
        rst &= setXu(2, OB_EXT_CMD1, sizeof(OB_EXT_CMD1));
        rst &= getXu(2, &rcvData, &rcvLen);
        rst &= setXu(2, OB_EXT_CMD2, sizeof(OB_EXT_CMD2));
        rst &= getXu(2, &rcvData, &rcvLen);
        rst &= setXu(2, OB_EXT_CMD3, sizeof(OB_EXT_CMD3));
        rst &= getXu(2, &rcvData, &rcvLen);
        break;
    default:
        rst = false;
        break;
    }
    return rst;
}

bool IUvcStreamChannel::getProperty(int propId, uint8_t* recvData, uint32_t* recvDataSize)
{
    bool rst = true;
    uint8_t* rcvData;
    uint32_t rcvLen;
    switch (propId)
    {
    case CAMERA_PARAM:
        rst &= setXu(2, OB_EXT_CMD5, sizeof(OB_EXT_CMD5));
        rst &= getXu(2, &rcvData, &rcvLen);
        if (rst && OB_EXT_CMD5[6] == rcvData[6] && rcvData[8] == 0 && rcvData[9] == 0)
        {
            memcpy(recvData, rcvData + 10, rcvLen - 10);
            *recvDataSize = rcvLen - 10;
        }
        break;
    default:
        rst = false;
        break;
    }

    return rst;
}

bool IUvcStreamChannel::initDepthFrameProcessor()
{
    if (streamType_ == OBSENSOR_STREAM_DEPTH && setXu(2, OB_EXT_CMD4, sizeof(OB_EXT_CMD4)))
    {
        uint8_t* rcvData;
        uint32_t rcvLen;
        if (getXu(1, &rcvData, &rcvLen) && OB_EXT_CMD4[6] == rcvData[6] && rcvData[8] == 0 && rcvData[9] == 0)
        {
            depthFrameProcessor_ = makePtr<DepthFrameProcessor>(*(OBExtensionParam*)(rcvData + 10));
            return true;
        }
    }
    return false;
}
}} // namespace cv::obsensor::
#endif // HAVE_OBSENSOR_V4L2 || HAVE_OBSENSOR_MSMF
