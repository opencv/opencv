/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                            License Agreement
//                 For Open Source Computer Vision Library
//                         (3-clause BSD License)
//
//  Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
//  Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
//  Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
//  Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
//  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
//  Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
//  Third party copyrights are property of their respective owners.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright notice,
//      this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//
//    * Neither the names of the copyright holders nor the names of the contributors
//      may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall copyright holders or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.
//
//
//M*/

//Based on "imgcodecs/src/grfmt_pxm.hpp"
//Written by Dimitrios Katsaros <patcherwork@gmail.com>

#ifndef _OPENCV_PAM_HPP_
#define _OPENCV_PAM_HPP_

#include "grfmt_base.hpp"
#include "bitstrm.hpp"

namespace cv
{

class PAMDecoder : public BaseImageDecoder
{
public:

    PAMDecoder();
    virtual ~PAMDecoder();

    bool  readData( Mat& img );
    bool  readHeader();

    size_t signatureLength() const;
    bool checkSignature( const String& signature ) const;
    ImageDecoder newDecoder() const;

protected:

    RLByteStream    m_strm;
    int m_maxval, m_channels, m_sampledepth, m_offset,
        selected_fmt;
    bool bit_mode;
};


class PAMEncoder : public BaseImageEncoder
{
public:
    PAMEncoder();
    virtual ~PAMEncoder();

    bool  isFormatSupported( int depth ) const;
    bool  write( const Mat& img, const std::vector<int>& params );

    ImageEncoder newEncoder() const;
};

}

#endif /* _OPENCV_PAM_HPP_ */