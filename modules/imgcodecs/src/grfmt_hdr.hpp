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

#ifndef _GRFMT_HDR_H_
#define _GRFMT_HDR_H_

#include "grfmt_base.hpp"

#ifdef HAVE_IMGCODEC_HDR

namespace cv
{

// Radiance rgbe (.hdr) reader
class HdrDecoder CV_FINAL : public BaseImageDecoder
{
public:
    HdrDecoder();
    ~HdrDecoder() CV_OVERRIDE;
    bool readHeader() CV_OVERRIDE;
    bool readData( Mat& img ) CV_OVERRIDE;
    bool checkSignature( const String& signature ) const CV_OVERRIDE;
    ImageDecoder newDecoder() const CV_OVERRIDE;
    size_t signatureLength() const CV_OVERRIDE;
protected:
    String m_signature_alt;
    FILE *file;
};

// ... writer
class HdrEncoder CV_FINAL : public BaseImageEncoder
{
public:
    HdrEncoder();
    ~HdrEncoder() CV_OVERRIDE;
    bool write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    ImageEncoder newEncoder() const CV_OVERRIDE;
    bool isFormatSupported( int depth ) const CV_OVERRIDE;
protected:

};

}

#endif // HAVE_IMGCODEC_HDR

#endif/*_GRFMT_HDR_H_*/
