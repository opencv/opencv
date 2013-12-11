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

#ifndef _GRFMT_EXR_H_
#define _GRFMT_EXR_H_

#ifdef HAVE_OPENEXR

#if defined __GNUC__ && defined __APPLE__
#  pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <ImfChromaticities.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImathBox.h>
#include "grfmt_base.hpp"

namespace cv
{

using namespace Imf;
using namespace Imath;

/* libpng version only */

class ExrDecoder : public BaseImageDecoder
{
public:

    ExrDecoder();
    ~ExrDecoder();

    int   type() const;
    bool  readData( Mat& img );
    bool  readHeader();
    void  close();

    ImageDecoder newDecoder() const;

protected:
    void  UpSample( uchar *data, int xstep, int ystep, int xsample, int ysample );
    void  UpSampleX( float *data, int xstep, int xsample );
    void  UpSampleY( uchar *data, int xstep, int ystep, int ysample );
    void  ChromaToBGR( float *data, int numlines, int step );
    void  RGBToGray( float *in, float *out );

    InputFile      *m_file;
    Imf::PixelType  m_type;
    Box2i           m_datawindow;
    bool            m_ischroma;
    const Channel  *m_red;
    const Channel  *m_green;
    const Channel  *m_blue;
    Chromaticities  m_chroma;
    int             m_bit_depth;
    bool            m_native_depth;
    bool            m_iscolor;
    bool            m_isfloat;
};


class ExrEncoder : public BaseImageEncoder
{
public:
    ExrEncoder();
    ~ExrEncoder();

    bool  isFormatSupported( int depth ) const;
    bool  write( const Mat& img, const std::vector<int>& params );
    ImageEncoder newEncoder() const;
};

}

#endif

#endif/*_GRFMT_EXR_H_*/
