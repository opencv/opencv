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

#ifndef _GRFMT_JASPER_H_
#define _GRFMT_JASPER_H_

#ifdef HAVE_JASPER

#include "grfmt_base.hpp"

namespace cv
{

class Jpeg2KDecoder CV_FINAL : public BaseImageDecoder
{
public:

    Jpeg2KDecoder();
    virtual ~Jpeg2KDecoder();

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();
    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:
    bool  readComponent8u( uchar *data, void *buffer, int step, int cmpt,
                           int maxval, int offset, int ncmpts );
    bool  readComponent16u( unsigned short *data, void *buffer, int step, int cmpt,
                            int maxval, int offset, int ncmpts );

    void *m_stream;
    void *m_image;
};


class Jpeg2KEncoder CV_FINAL : public BaseImageEncoder
{
public:
    Jpeg2KEncoder();
    virtual ~Jpeg2KEncoder();

    bool  isFormatSupported( int depth ) const CV_OVERRIDE;
    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    ImageEncoder newEncoder() const CV_OVERRIDE;

protected:
    bool  writeComponent8u( void *img, const Mat& _img );
    bool  writeComponent16u( void *img, const Mat& _img );
};

}

#endif

#endif/*_GRFMT_JASPER_H_*/
