/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef _GRFMT_JPEG_H_
#define _GRFMT_JPEG_H_

#include "grfmt_base.hpp"
#include "bitstrm.hpp"

#ifdef HAVE_JPEG

// IJG-based Jpeg codec

namespace cv
{
/**
* @brief Jpeg markers that can be encountered in a Jpeg file
*/
enum AppMarkerTypes
{
    SOI = 0xD8, SOF0 = 0xC0, SOF2 = 0xC2, DHT = 0xC4,
    DQT = 0xDB, DRI = 0xDD, SOS = 0xDA,

    RST0 = 0xD0, RST1 = 0xD1, RST2 = 0xD2, RST3 = 0xD3,
    RST4 = 0xD4, RST5 = 0xD5, RST6 = 0xD6, RST7 = 0xD7,

    APP0 = 0xE0, APP1 = 0xE1, APP2 = 0xE2, APP3 = 0xE3,
    APP4 = 0xE4, APP5 = 0xE5, APP6 = 0xE6, APP7 = 0xE7,
    APP8 = 0xE8, APP9 = 0xE9, APP10 = 0xEA, APP11 = 0xEB,
    APP12 = 0xEC, APP13 = 0xED, APP14 = 0xEE, APP15 = 0xEF,

    COM = 0xFE, EOI = 0xD9
};


class JpegDecoder CV_FINAL : public BaseImageDecoder
{
public:

    JpegDecoder();
    virtual ~JpegDecoder();

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();

    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:

    FILE* m_f;
    void* m_state;

private:
    JpegDecoder(const JpegDecoder &); // copy disabled
    JpegDecoder& operator=(const JpegDecoder &); // assign disabled
};


class JpegEncoder CV_FINAL : public BaseImageEncoder
{
public:
    JpegEncoder();
    virtual ~JpegEncoder();

    bool isValidParam(const int key, const int value) const CV_OVERRIDE;

    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    ImageEncoder newEncoder() const CV_OVERRIDE;
};

}

#endif

#endif/*_GRFMT_JPEG_H_*/
