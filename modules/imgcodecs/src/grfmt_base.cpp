/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "precomp.hpp"

#include "grfmt_base.hpp"
#include "bitstrm.hpp"

namespace cv
{

ImageDecoder::Impl::Impl()
{
    m_width = m_height = 0;
    m_type = -1;
    m_buf_supported = false;
    m_scale_denom = 1;
}

bool ImageDecoder::Impl::setSource( const String& filename )
{
    m_filename = filename;
    m_buf.release();
    return true;
}

bool ImageDecoder::Impl::setSource( const Mat& buf )
{
    if( !m_buf_supported )
        return false;
    m_filename = String();
    m_buf = buf;
    return true;
}

size_t ImageDecoder::Impl::signatureLength() const
{
    return m_signature.size();
}

bool ImageDecoder::Impl::checkSignature( const String& signature ) const
{
    size_t len = signatureLength();
    return signature.size() >= len && memcmp( signature.c_str(), m_signature.c_str(), len ) == 0;
}

int ImageDecoder::Impl::setScale( const int& scale_denom )
{
    int temp = m_scale_denom;
    m_scale_denom = scale_denom;
    return temp;
}

bool ImageDecoder::Impl::checkDest( const Mat& dst, int dst_type ) const
{
    size_t have_size = dst.total() * dst.elemSize();
    size_t want_size = m_width * m_height * CV_ELEM_SIZE(dst_type);
    return have_size >= want_size;
}

String ImageDecoder::Impl::getDescription() const
{
    return m_description;
}

Ptr<ImageDecoder::Impl> ImageDecoder::Impl::newDecoder() const
{
    return Ptr<ImageDecoder::Impl>();
}

ImageEncoder::Impl::Impl()
{
    m_buf = 0;
    m_buf_supported = false;
}

bool  ImageEncoder::Impl::isFormatSupported( int depth ) const
{
    return depth == CV_8U;
}

String ImageEncoder::Impl::getDescription() const
{
    return m_description;
}

bool ImageEncoder::Impl::setDestination( const String& filename )
{
    m_filename = filename;
    m_buf = 0;
    return true;
}

bool ImageEncoder::Impl::setDestination( Mat& buf )
{
    if( !m_buf_supported )
        return false;
    m_buf = &buf;
    memset(m_buf->data, 0, m_buf->total() * m_buf->elemSize());
    m_filename = String();
    return true;
}

Ptr<ImageEncoder::Impl> ImageEncoder::Impl::newEncoder() const
{
    return Ptr<ImageEncoder::Impl>();
}

void ImageEncoder::Impl::throwOnEror() const
{
    if(!m_last_error.empty())
    {
        String msg = "Raw image encoder error: " + m_last_error;
        CV_Error( CV_BadImageSize, msg.c_str() );
    }
}

}

/* End of file. */
