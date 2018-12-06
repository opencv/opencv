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

#if defined _WIN32
#include <windows.h>
#endif

namespace cv
{

#if defined _WIN32
unsigned int code_page = AreFileApisANSI() ? CP_ACP : CP_OEMCP;
#endif

String toString(const WString& wstr)
{
#if defined _WIN32
    if(wstr.empty())
    {
        //Return the empty string
        return("");
    }

    //Calculate target buffer size (not including the zero terminator)
    int wstring_size = static_cast<int>(wstr.size());
    int length = WideCharToMultiByte(code_page, 0, wstr.c_str(), wstring_size, NULL, 0, NULL, NULL);
    if(length == 0)
    {
        //Conversion failed
        return("");
    }

    //The elements of a vector are stored contiguously, the standard does not guarantee this for a string
    std::vector<char> str(length, ' ');
    WideCharToMultiByte(code_page, 0, wstr.c_str(), wstring_size, str.data(), length, NULL, NULL);

    return(std::string(str.data(), length));
#else
    std::size_t size = wstr.size();

    //Overestimate number of code points
    std::vector<char> str(size, ' ');

    //Convert to a narrow character string
    size = std::wcstombs(str.data(), wstr.c_str(), size);
    if(size != String::npos)
    {
        //Shrink to fit
        return(String(str.data(), size));
    }

    //Conversion failed, return an empty string
    return("");
#endif
}

WString toWString(const String& str)
{
#if defined _WIN32
    if(str.empty())
    {
        //Return the empty wstring
        return(L"");
    }

    //Calculate target buffer size (not including the zero terminator)
    int string_size = static_cast<int>(str.size());
    int length = MultiByteToWideChar(code_page, 0, str.c_str(), string_size, NULL, 0);
    if(length == 0)
    {
        //Conversion failed
        return(L"");
    }

    //The elements of a vector are stored contiguously, the standard does not guarantee this for a string
    std::vector<wchar_t> wstr(length, L' ');

    //No error checking. We already know, that the conversion will succeed.
    MultiByteToWideChar(code_page, 0, str.c_str(), string_size, wstr.data(), length);

    return(WString(wstr.data(), length));
#else
    std::size_t size = str.size();

    //Overestimate number of code points
    std::vector<wchar_t> wstr(size, L' ');

    //Convert to a multibyte character string
    size = std::mbstowcs(wstr.data(), str.c_str(), size);
    if(size != String::npos)
    {
        //Shrink to fit
        return(WString(wstr.data(), size));
    }

    //Conversion failed, return an empty wstring
    return(L"");
#endif
}

BaseImageDecoder::BaseImageDecoder()
{
    m_width = m_height = 0;
    m_type = -1;
    m_buf_supported = false;
    m_scale_denom = 1;
}

bool BaseImageDecoder::setSource( const Pfad& filename )
{
    m_filename = filename;
    m_buf.release();
    return true;
}

bool BaseImageDecoder::setSource( const Mat& buf )
{
    if( !m_buf_supported )
        return false;
    m_filename = Pfad();
    m_buf = buf;
    return true;
}

size_t BaseImageDecoder::signatureLength() const
{
    return m_signature.size();
}

bool BaseImageDecoder::checkSignature( const String& signature ) const
{
    size_t len = signatureLength();
    return signature.size() >= len && memcmp( signature.c_str(), m_signature.c_str(), len ) == 0;
}

int BaseImageDecoder::setScale( const int& scale_denom )
{
    int temp = m_scale_denom;
    m_scale_denom = scale_denom;
    return temp;
}

ImageDecoder BaseImageDecoder::newDecoder() const
{
    return ImageDecoder();
}

BaseImageEncoder::BaseImageEncoder()
{
    m_buf = 0;
    m_buf_supported = false;
}

bool  BaseImageEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U;
}

Pfad BaseImageEncoder::getDescription() const
{
    return m_description;
}

bool BaseImageEncoder::setDestination( const Pfad& filename )
{
    m_filename = filename;
    m_buf = 0;
    return true;
}

bool BaseImageEncoder::setDestination( std::vector<uchar>& buf )
{
    if( !m_buf_supported )
        return false;
    m_buf = &buf;
    m_buf->clear();
    m_filename = Pfad();
    return true;
}

bool BaseImageEncoder::writemulti(const std::vector<Mat>&, const std::vector<int>& )
{
    return false;
}

ImageEncoder BaseImageEncoder::newEncoder() const
{
    return ImageEncoder();
}

void BaseImageEncoder::throwOnEror() const
{
    if(!m_last_error.empty())
    {
        String msg = "Raw image encoder error: " + m_last_error;
        CV_Error( Error::BadImageSize, msg.c_str() );
    }
}

}

/* End of file. */
