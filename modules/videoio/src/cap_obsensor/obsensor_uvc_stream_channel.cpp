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
const ObExtensionUnit OBSENSOR_COMMON_XU_UNIT = { XU_UNIT_ID_COMMON, { 0xA55751A1, 0xF3C5, 0x4A5E, { 0x8D, 0x5A, 0x68, 0x54, 0xB8, 0xFA, 0x27, 0x16 } } };
const ObExtensionUnit OBSENSOR_G330_XU_UNIT = { XU_UNIT_ID_G330, { 0xC9606CCB, 0x594C, 0x4D25, { 0xaf, 0x47, 0xcc, 0xc4, 0x96, 0x43, 0x59, 0x95 } } };

const uint8_t OB_EXT_CMD0[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x52, 0x00, 0x5B, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD1[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x54, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD2[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x56, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD3[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x58, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD4[16] = { 0x47, 0x4d, 0x02, 0x00, 0x03, 0x00, 0x60, 0x00, 0xed, 0x03, 0x00, 0x00 };
const uint8_t OB_EXT_CMD5[16] = { 0x47, 0x4d, 0x02, 0x00, 0x03, 0x00, 0x62, 0x00, 0xe9, 0x03, 0x00, 0x00 };
const uint8_t OB_EXT_CMD6[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x7c, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD7[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0xfe, 0x12, 0x55, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD8[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0xfe, 0x13, 0x3f, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD9[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0xfa, 0x13, 0x4b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD11[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0xfe, 0x13, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD12[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0xfe, 0x13, 0x3f, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD13[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0xfa, 0x13, 0x4b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
const uint8_t OB_EXT_CMD14[16] = { 0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0xfa, 0x14, 0xd3, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };

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
    {fourCc2Int('Y', '1', '4', ' '), FRAME_FORMAT_Y14},
    {fourCc2Int('Z', '1', '6', ' '), FRAME_FORMAT_Y16}
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


DepthFrameUnpacker::DepthFrameUnpacker(){
    outputDataBuf_ = new uint8_t[OUT_DATA_SIZE];
}

DepthFrameUnpacker::~DepthFrameUnpacker() {
    delete[] outputDataBuf_;
}

#define ON_BITS(count) ((1 << count) - 1)
#define CREATE_MASK(count, offset) (ON_BITS(count) << offset)
#define TAKE_BITS(source, count, offset) ((source & CREATE_MASK(count, offset)) >> offset)
void DepthFrameUnpacker::process(Frame *frame){
    const uint8_t tarStep   = 16;
    const uint8_t srcStep   = 28;
    uint16_t *tar = (uint16_t *)outputDataBuf_;
    uint8_t *src = frame->data;

    uint32_t      pixelSize = frame->width * frame->height;
    for(uint32_t i = 0; i < pixelSize; i += tarStep) {
        tar[0] = (TAKE_BITS(src[0], 8, 0) << 6) | TAKE_BITS(src[1], 6, 2);
        tar[1] = (TAKE_BITS(src[1], 2, 0) << 12) | (TAKE_BITS(src[2], 8, 0) << 4) | TAKE_BITS(src[3], 4, 4);
        tar[2] = (TAKE_BITS(src[3], 4, 0) << 10) | (TAKE_BITS(src[4], 8, 0) << 2) | TAKE_BITS(src[5], 2, 6);
        tar[3] = (TAKE_BITS(src[5], 6, 0) << 8) | TAKE_BITS(src[6], 8, 0);

        tar[4] = (TAKE_BITS(src[7], 8, 0) << 6) | TAKE_BITS(src[8], 6, 2);
        tar[5] = (TAKE_BITS(src[8], 2, 0) << 12) | (TAKE_BITS(src[9], 8, 0) << 4) | TAKE_BITS(src[10], 4, 4);
        tar[6] = (TAKE_BITS(src[10], 4, 0) << 10) | (TAKE_BITS(src[11], 8, 0) << 2) | TAKE_BITS(src[12], 2, 6);
        tar[7] = (TAKE_BITS(src[12], 6, 0) << 8) | TAKE_BITS(src[13], 8, 0);

        tar[8]  = (TAKE_BITS(src[14], 8, 0) << 6) | TAKE_BITS(src[15], 6, 2);
        tar[9]  = (TAKE_BITS(src[15], 2, 0) << 12) | (TAKE_BITS(src[16], 8, 0) << 4) | TAKE_BITS(src[17], 4, 4);
        tar[10] = (TAKE_BITS(src[17], 4, 0) << 10) | (TAKE_BITS(src[18], 8, 0) << 2) | TAKE_BITS(src[19], 2, 6);
        tar[11] = (TAKE_BITS(src[19], 6, 0) << 8) | TAKE_BITS(src[20], 8, 0);

        tar[12] = (TAKE_BITS(src[21], 8, 0) << 6) | TAKE_BITS(src[22], 6, 2);
        tar[13] = (TAKE_BITS(src[22], 2, 0) << 12) | (TAKE_BITS(src[23], 8, 0) << 4) | TAKE_BITS(src[24], 4, 4);
        tar[14] = (TAKE_BITS(src[24], 4, 0) << 10) | (TAKE_BITS(src[25], 8, 0) << 2) | TAKE_BITS(src[26], 2, 6);
        tar[15] = (TAKE_BITS(src[26], 6, 0) << 8) | TAKE_BITS(src[27], 8, 0);

        src += srcStep;
        tar += tarStep;
    }
    frame->data = outputDataBuf_;
    frame->format = FRAME_FORMAT_Y16;
}

IUvcStreamChannel::IUvcStreamChannel(const UvcDeviceInfo& devInfo) :
    devInfo_(devInfo),
    xuUnit_(IS_OBSENSOR_GEMINI330_PID(devInfo.pid) ? OBSENSOR_G330_XU_UNIT : OBSENSOR_COMMON_XU_UNIT),
    streamType_(parseUvcDeviceNameToStreamType(devInfo_.name))
{

}

StreamType IUvcStreamChannel::streamType() const {
    return streamType_;
}

uint16_t IUvcStreamChannel::getPid() const {
    return devInfo_.pid;
};

bool IUvcStreamChannel::setProperty(int propId, const uint8_t* /*data*/, uint32_t /*dataSize*/)
{
    uint8_t* rcvData;
    uint32_t rcvLen;
    bool rst = true;
    switch (propId)
    {
    case DEPTH_TO_COLOR_ALIGN:
        if(OBSENSOR_GEMINI2_PID == devInfo_.pid ){
            rst &= setXu(2, OB_EXT_CMD8, sizeof(OB_EXT_CMD8));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD6, sizeof(OB_EXT_CMD6));
            rst &= getXu(2, &rcvData, &rcvLen);
        }else if(OBSENSOR_ASTRA2_PID == devInfo_.pid ){
            rst &= setXu(2, OB_EXT_CMD12, sizeof(OB_EXT_CMD12));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD6, sizeof(OB_EXT_CMD6));
            rst &= getXu(2, &rcvData, &rcvLen);
        }else if(OBSENSOR_GEMINI2L_PID == devInfo_.pid){
            rst &= setXu(2, OB_EXT_CMD11, sizeof(OB_EXT_CMD11));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD6, sizeof(OB_EXT_CMD6));
            rst &= getXu(2, &rcvData, &rcvLen);
        }else if(OBSENSOR_GEMINI2XL_PID == devInfo_.pid){
            rst &= setXu(2, OB_EXT_CMD11, sizeof(OB_EXT_CMD11));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD6, sizeof(OB_EXT_CMD6));
            rst &= getXu(2, &rcvData, &rcvLen);
        }else if(IS_OBSENSOR_GEMINI330_PID(devInfo_.pid)) {
            rst &= setXu(2, OB_EXT_CMD6, sizeof(OB_EXT_CMD6));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD14, sizeof(OB_EXT_CMD14));
            rst &= getXu(2, &rcvData, &rcvLen);
        }else{
            rst &= setXu(2, OB_EXT_CMD0, sizeof(OB_EXT_CMD0));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD1, sizeof(OB_EXT_CMD1));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD2, sizeof(OB_EXT_CMD2));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, OB_EXT_CMD3, sizeof(OB_EXT_CMD3));
            rst &= getXu(2, &rcvData, &rcvLen);
        }
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
        if(OBSENSOR_GEMINI2_PID == devInfo_.pid){
            // return default param
            CameraParam param;
            param.p0[0] = 519.342f;
            param.p0[1] = 519.043f;
            param.p0[2] = 319.41f;
            param.p0[3] = 240.839f;
            param.p1[0] = 519.342f;
            param.p1[1] = 519.043f;
            param.p1[2] = 319.41f;
            param.p1[3] = 240.839f;
            param.p6[0] = 640;
            param.p6[1] = 480;
            param.p7[0] = 640;
            param.p7[1] = 480;
            *recvDataSize = sizeof(CameraParam);
            memcpy(recvData, &param, *recvDataSize);
        }else if(OBSENSOR_GEMINI2L_PID == devInfo_.pid){
            // return default param
            CameraParam param;
            param.p0[0] = 688.87f;
            param.p0[1] = 688.922f;
            param.p0[2] = 644.317f;
            param.p0[3] = 354.382f;
            param.p1[0] = 688.87f;
            param.p1[1] = 688.922f;
            param.p1[2] = 644.317f;
            param.p1[3] = 354.382f;
            param.p6[0] = 1280;
            param.p6[1] = 720;
            param.p7[0] = 1280;
            param.p7[1] = 720;
            *recvDataSize = sizeof(CameraParam);
            memcpy(recvData, &param, *recvDataSize);
        }else if(OBSENSOR_GEMINI2XL_PID == devInfo_.pid){
            // return default param
            CameraParam param;
            param.p0[0] = 610.847f;
            param.p0[1] = 610.829f;
            param.p0[2] = 640.647f;
            param.p0[3] = 401.817f;
            param.p1[0] = 610.847f;
            param.p1[1] = 610.829f;
            param.p1[2] = 640.647f;
            param.p1[3] = 401.817f;
            param.p6[0] = 640;
            param.p6[1] = 480;
            param.p7[0] = 640;
            param.p7[1] = 480;
            *recvDataSize = sizeof(CameraParam);
            memcpy(recvData, &param, *recvDataSize);
        }
        else if(OBSENSOR_ASTRA2_PID == devInfo_.pid){
            // return default param
            CameraParam param;
            param.p0[0] = 558.151f;
            param.p0[1] = 558.003f;
            param.p0[2] = 312.546f;
            param.p0[3] = 241.169f;
            param.p1[0] = 558.151f;
            param.p1[1] = 558.003f;
            param.p1[2] = 312.546f;
            param.p1[3] = 241.169f;
            param.p6[0] = 640;
            param.p6[1] = 480;
            param.p7[0] = 640;
            param.p7[1] = 480;
            *recvDataSize = sizeof(CameraParam);
            memcpy(recvData, &param, *recvDataSize);
        }
        else if(OBSENSOR_FEMTO_MEGA_PID == devInfo_.pid){
            // return default param
            CameraParam param;
            param.p0[0] = 748.370f;
            param.p0[1] = 748.296f;
            param.p0[2] = 634.670f;
            param.p0[3] = 341.196f;
            param.p1[0] = 374.185f;
            param.p1[1] = 374.148f;
            param.p1[2] = 317.335f;
            param.p1[3] = 170.598f;
            param.p6[0] = 1280;
            param.p6[1] = 720;
            param.p7[0] = 640;
            param.p7[1] = 360;
            *recvDataSize = sizeof(CameraParam);
            memcpy(recvData, &param, *recvDataSize);
        }
        else if(IS_OBSENSOR_GEMINI330_SHORT_PID(devInfo_.pid)){
            // return default param
            CameraParam param;
            param.p0[0] = 460.656f;
            param.p0[1] = 460.782f;
            param.p0[2] = 320.985f;
            param.p0[3] = 233.921f;
            param.p1[0] = 460.656f;
            param.p1[1] = 460.782f;
            param.p1[2] = 320.985f;
            param.p1[3] = 233.921f;
            param.p6[0] = 640;
            param.p6[1] = 480;
            param.p7[0] = 640;
            param.p7[1] = 480;
            *recvDataSize = sizeof(CameraParam);
            memcpy(recvData, &param, *recvDataSize);
        }
        else if(IS_OBSENSOR_GEMINI330_LONG_PID(devInfo_.pid)){
            // return default param
            CameraParam param;
            param.p0[0] = 366.751f;
            param.p0[1] = 365.782f;
            param.p0[2] = 319.893f;
            param.p0[3] = 243.415f;
            param.p1[0] = 366.751f;
            param.p1[1] = 365.782f;
            param.p1[2] = 319.893f;
            param.p1[3] = 243.415f;
            param.p6[0] = 640;
            param.p6[1] = 480;
            param.p7[0] = 640;
            param.p7[1] = 480;
            *recvDataSize = sizeof(CameraParam);
            memcpy(recvData, &param, *recvDataSize);
        }
        else{
            rst &= setXu(2, OB_EXT_CMD5, sizeof(OB_EXT_CMD5));
            rst &= getXu(2, &rcvData, &rcvLen);
            if (rst && OB_EXT_CMD5[6] == rcvData[6] && rcvData[8] == 0 && rcvData[9] == 0)
            {
                memcpy(recvData, rcvData + 10, rcvLen - 10);
                *recvDataSize = rcvLen - 10;
            }
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
    if( OBSENSOR_ASTRA2_PID == devInfo_.pid){
        uint8_t* rcvData;
        uint32_t rcvLen;

        setXu(2, OB_EXT_CMD7, sizeof(OB_EXT_CMD7));
        getXu(2, &rcvData, &rcvLen);

        setXu(2, OB_EXT_CMD9, sizeof(OB_EXT_CMD9));
        getXu(2, &rcvData, &rcvLen);

        depthFrameProcessor_ = makePtr<DepthFrameUnpacker>();
        return true;
    }
    else if(OBSENSOR_GEMINI2_PID == devInfo_.pid || OBSENSOR_GEMINI2L_PID == devInfo_.pid){
        uint8_t* rcvData;
        uint32_t rcvLen;

        setXu(2, OB_EXT_CMD7, sizeof(OB_EXT_CMD7));
        getXu(2, &rcvData, &rcvLen);

        setXu(2, OB_EXT_CMD9, sizeof(OB_EXT_CMD9));
        getXu(2, &rcvData, &rcvLen);
        return true;
    }
    else if(OBSENSOR_GEMINI2XL_PID == devInfo_.pid){
        uint8_t* rcvData;
        uint32_t rcvLen;

        setXu(2, OB_EXT_CMD7, sizeof(OB_EXT_CMD7));
        getXu(2, &rcvData, &rcvLen);

        setXu(2, OB_EXT_CMD13, sizeof(OB_EXT_CMD13));
        getXu(2, &rcvData, &rcvLen);
        return true;
    }
    else if(IS_OBSENSOR_GEMINI330_PID(devInfo_.pid))
    {
        uint8_t* rcvData;
        uint32_t rcvLen;

        setXu(2, OB_EXT_CMD7, sizeof(OB_EXT_CMD7));
        getXu(2, &rcvData, &rcvLen);
        return true;
    }
    else if(streamType_ == OBSENSOR_STREAM_DEPTH && setXu(2, OB_EXT_CMD4, sizeof(OB_EXT_CMD4)))
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
