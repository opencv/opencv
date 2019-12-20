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

#ifndef _GRFMT_BASE_H_
#define _GRFMT_BASE_H_

#include "utils.hpp"
#include "bitstrm.hpp"

namespace cv
{

class BaseImageDecoder;
class BaseImageEncoder;
typedef Ptr<BaseImageEncoder> ImageEncoder;
typedef Ptr<BaseImageDecoder> ImageDecoder;

///////////////////////////////// base class for decoders ////////////////////////
class BaseImageDecoder
{
public:
    BaseImageDecoder();
    virtual ~BaseImageDecoder() {}

    int width() const { return m_width; }
    int height() const { return m_height; }
    virtual int type() const { return m_type; }

    virtual bool setSource( const String& filename );
    virtual bool setSource( const Mat& buf );
    virtual int setScale( const int& scale_denom );
    virtual bool readHeader() = 0;
    virtual bool readData( Mat& img ) = 0;

    /// Called after readData to advance to the next page, if any.
    virtual bool nextPage() { return false; }

    virtual size_t signatureLength() const;
    virtual bool checkSignature( const String& signature ) const;
    virtual ImageDecoder newDecoder() const;

protected:
    int  m_width;  // width  of the image ( filled by readHeader )
    int  m_height; // height of the image ( filled by readHeader )
    int  m_type;
    int  m_scale_denom;
    String m_filename;
    String m_signature;
    Mat m_buf;
    bool m_buf_supported;
};


///////////////////////////// base class for encoders ////////////////////////////
class BaseImageEncoder
{
public:
    BaseImageEncoder();
    virtual ~BaseImageEncoder() {}
    virtual bool isFormatSupported( int depth ) const;

    virtual bool setDestination( const String& filename );
    virtual bool setDestination( std::vector<uchar>& buf );
    virtual bool write( const Mat& img, const std::vector<int>& params ) = 0;
    virtual bool writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params);

    virtual String getDescription() const;
    virtual ImageEncoder newEncoder() const;

    virtual void throwOnEror() const;

protected:
    String m_description;

    String m_filename;
    std::vector<uchar>* m_buf;
    bool m_buf_supported;

    String m_last_error;
};

}

#endif/*_GRFMT_BASE_H_*/
