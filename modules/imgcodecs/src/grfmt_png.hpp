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

#ifndef _GRFMT_PNG_H_
#define _GRFMT_PNG_H_

#ifdef HAVE_PNG

#include "grfmt_base.hpp"
#include "bitstrm.hpp"

namespace cv
{

class PngDecoder : public BaseImageDecoder
{
public:

    PngDecoder();
    virtual ~PngDecoder();

    bool  readData( Mat& img );
    bool  readHeader();
    void  close();

    ImageDecoder newDecoder() const;

protected:

    static void readDataFromBuf(void* png_ptr, uchar* dst, size_t size);

    int   m_bit_depth;
    void* m_png_ptr;  // pointer to decompression structure
    void* m_info_ptr; // pointer to image information structure
    void* m_end_info; // pointer to one more image information structure
    FILE* m_f;
    int   m_color_type;
    size_t m_buf_pos;
};


class PngEncoder : public BaseImageEncoder
{
public:
    PngEncoder();
    virtual ~PngEncoder();

    bool  isFormatSupported( int depth ) const;
    bool  write( const Mat& img, const std::vector<int>& params );

    ImageEncoder newEncoder() const;

protected:
    static void writeDataToBuf(void* png_ptr, uchar* src, size_t size);
    static void flushBuf(void* png_ptr);
};

}

#endif

#endif/*_GRFMT_PNG_H_*/
