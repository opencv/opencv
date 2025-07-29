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

#ifndef _GRFMT_BMP_H_
#define _GRFMT_BMP_H_

#include "grfmt_base.hpp"

namespace cv
{

enum BmpCompression
{
    BMP_RGB = 0,
    BMP_RLE8 = 1,
    BMP_RLE4 = 2,
    BMP_BITFIELDS = 3
};


// Windows Bitmap reader
class BmpDecoder CV_FINAL : public BaseImageDecoder
{
public:

    BmpDecoder();
    ~BmpDecoder() CV_OVERRIDE;

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();

    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:

    void  initMask();
    void  maskBGRA(uchar* des, const uchar* src, int num, bool alpha_required);
    void  maskBGRAtoGray(uchar* des, const uchar* src, int num);

    enum Origin
    {
        ORIGIN_TL = 0,
        ORIGIN_BL = 1
    };

    RLByteStream    m_strm;
    PaletteEntry    m_palette[256];
    Origin          m_origin;
    int             m_bpp;
    int             m_offset;
    BmpCompression  m_rle_code;
    uint            m_rgba_mask[4];
    int             m_rgba_bit_offset[4];
    float           m_rgba_scale_factor[4];
};


// ... writer
class BmpEncoder CV_FINAL : public BaseImageEncoder
{
public:
    BmpEncoder();
    ~BmpEncoder() CV_OVERRIDE;

    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE;
};

}

#endif/*_GRFMT_BMP_H_*/
