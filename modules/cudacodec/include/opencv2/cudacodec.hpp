/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CUDACODEC_HPP__
#define __OPENCV_CUDACODEC_HPP__

#ifndef __cplusplus
#  error cudacodec.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"

namespace cv { namespace cudacodec {

////////////////////////////////// Video Encoding //////////////////////////////////

// Works only under Windows.
// Supports olny H264 video codec and AVI files.

enum SurfaceFormat
{
    SF_UYVY = 0,
    SF_YUY2,
    SF_YV12,
    SF_NV12,
    SF_IYUV,
    SF_BGR,
    SF_GRAY = SF_BGR
};

struct CV_EXPORTS EncoderParams
{
    int P_Interval;      // NVVE_P_INTERVAL,
    int IDR_Period;      // NVVE_IDR_PERIOD,
    int DynamicGOP;      // NVVE_DYNAMIC_GOP,
    int RCType;          // NVVE_RC_TYPE,
    int AvgBitrate;      // NVVE_AVG_BITRATE,
    int PeakBitrate;     // NVVE_PEAK_BITRATE,
    int QP_Level_Intra;  // NVVE_QP_LEVEL_INTRA,
    int QP_Level_InterP; // NVVE_QP_LEVEL_INTER_P,
    int QP_Level_InterB; // NVVE_QP_LEVEL_INTER_B,
    int DeblockMode;     // NVVE_DEBLOCK_MODE,
    int ProfileLevel;    // NVVE_PROFILE_LEVEL,
    int ForceIntra;      // NVVE_FORCE_INTRA,
    int ForceIDR;        // NVVE_FORCE_IDR,
    int ClearStat;       // NVVE_CLEAR_STAT,
    int DIMode;          // NVVE_SET_DEINTERLACE,
    int Presets;         // NVVE_PRESETS,
    int DisableCabac;    // NVVE_DISABLE_CABAC,
    int NaluFramingType; // NVVE_CONFIGURE_NALU_FRAMING_TYPE
    int DisableSPSPPS;   // NVVE_DISABLE_SPS_PPS

    EncoderParams();
    explicit EncoderParams(const String& configFile);

    void load(const String& configFile);
    void save(const String& configFile) const;
};

class CV_EXPORTS EncoderCallBack
{
public:
    enum PicType
    {
        IFRAME = 1,
        PFRAME = 2,
        BFRAME = 3
    };

    virtual ~EncoderCallBack() {}

    //! callback function to signal the start of bitstream that is to be encoded
    //! callback must allocate host buffer for CUDA encoder and return pointer to it and it's size
    virtual uchar* acquireBitStream(int* bufferSize) = 0;

    //! callback function to signal that the encoded bitstream is ready to be written to file
    virtual void releaseBitStream(unsigned char* data, int size) = 0;

    //! callback function to signal that the encoding operation on the frame has started
    virtual void onBeginFrame(int frameNumber, PicType picType) = 0;

    //! callback function signals that the encoding operation on the frame has finished
    virtual void onEndFrame(int frameNumber, PicType picType) = 0;
};

class CV_EXPORTS VideoWriter
{
public:
    virtual ~VideoWriter() {}

    //! writes the next frame from GPU memory
    virtual void write(InputArray frame, bool lastFrame = false) = 0;

    virtual EncoderParams getEncoderParams() const = 0;
};

//! create VideoWriter for specified output file (only AVI file format is supported)
CV_EXPORTS Ptr<VideoWriter> createVideoWriter(const String& fileName, Size frameSize, double fps, SurfaceFormat format = SF_BGR);
CV_EXPORTS Ptr<VideoWriter> createVideoWriter(const String& fileName, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);

//! create VideoWriter for user-defined callbacks
CV_EXPORTS Ptr<VideoWriter> createVideoWriter(const Ptr<EncoderCallBack>& encoderCallback, Size frameSize, double fps, SurfaceFormat format = SF_BGR);
CV_EXPORTS Ptr<VideoWriter> createVideoWriter(const Ptr<EncoderCallBack>& encoderCallback, Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR);

////////////////////////////////// Video Decoding //////////////////////////////////////////

enum Codec
{
    MPEG1 = 0,
    MPEG2,
    MPEG4,
    VC1,
    H264,
    JPEG,
    H264_SVC,
    H264_MVC,

    Uncompressed_YUV420 = (('I'<<24)|('Y'<<16)|('U'<<8)|('V')),   // Y,U,V (4:2:0)
    Uncompressed_YV12   = (('Y'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,V,U (4:2:0)
    Uncompressed_NV12   = (('N'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,UV  (4:2:0)
    Uncompressed_YUYV   = (('Y'<<24)|('U'<<16)|('Y'<<8)|('V')),   // YUYV/YUY2 (4:2:2)
    Uncompressed_UYVY   = (('U'<<24)|('Y'<<16)|('V'<<8)|('Y'))    // UYVY (4:2:2)
};

enum ChromaFormat
{
    Monochrome = 0,
    YUV420,
    YUV422,
    YUV444
};

struct FormatInfo
{
    Codec codec;
    ChromaFormat chromaFormat;
    int width;
    int height;
};

class CV_EXPORTS VideoReader
{
public:
    virtual ~VideoReader() {}

    virtual bool nextFrame(OutputArray frame) = 0;

    virtual FormatInfo format() const = 0;
};

class CV_EXPORTS RawVideoSource
{
public:
    virtual ~RawVideoSource() {}

    virtual bool getNextPacket(unsigned char** data, int* size, bool* endOfFile) = 0;

    virtual FormatInfo format() const = 0;
};

CV_EXPORTS Ptr<VideoReader> createVideoReader(const String& filename);
CV_EXPORTS Ptr<VideoReader> createVideoReader(const Ptr<RawVideoSource>& source);

}} // namespace cv { namespace cudacodec {

#endif /* __OPENCV_CUDACODEC_HPP__ */
