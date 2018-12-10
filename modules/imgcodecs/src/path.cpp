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

#include "path.hpp"

#if defined _WIN32
#include <windows.h>
#endif

namespace cv
{

Path::Path()
{
    
}

Path::Path(const String& value)
{
#if defined _WIN32
    m_path = toWString(value);
#else
    m_path = value;
#endif
}

Path::Path(const String::value_type* value)
{
#if defined _WIN32
    m_path = toWString(value);
#else
    m_path = value;
#endif
}

Path::Path(const WString& value)
{
#if defined _WIN32
    m_path = value;
#else
    m_path = toString(value);
#endif
}

Path::Path(const WString::value_type* value)
{
#if defined _WIN32
    m_path = value;
#else
    m_path = toString(value);
#endif
}

Path::Path(const Path& rhs)
{
    m_path = rhs.m_path;
}

Path::~Path()
{
    
}

Path& Path::operator=(const Path& rhs)
{
    m_path = rhs.m_path;
    return(*this);
}

size_t Path::size() const
{
    return( m_path.size() );
}

bool Path::empty() const
{
    return( m_path.empty() );
}

const Path::PathType::value_type* Path::c_str() const
{
    return( m_path.c_str() );
}

const Path::PathType::value_type* Path::firstOccurrence( Path::PathType::value_type character ) const
{
#if defined _WIN32
    return( wcschr( m_path.c_str(), character ) );
#else
    return( strchr( m_path.c_str(), character ) );
#endif
}

const Path::PathType::value_type* Path::lastOccurrence( Path::PathType::value_type character ) const
{
#if defined _WIN32
    return( wcsrchr( m_path.c_str(), character ) );
#else
    return( strrchr( m_path.c_str(), character ) );
#endif
}

void Path::tempPath()
{
#if defined _WIN32
#ifndef WINRT
    const wchar_t *temp_dir = _wgetenv(L"OPENCV_TEMP_PATH");
#endif

#ifdef WINRT
    RoInitialize(RO_INIT_MULTITHREADED);
    std::wstring temp_dir = GetTempPathWinRT();

    std::wstring temp_file = GetTempFileNameWinRT(L"ocv");
    if (temp_file.empty())
    {
        m_path = WString();
    }
    else
    {
        temp_file = temp_dir.append(std::wstring(L"\\")).append(temp_file);
        DeleteFileW(temp_file.c_str());

        m_path = temp_file;
    }
#else
    wchar_t temp_dir2[MAX_PATH] = { 0 };
    wchar_t temp_file[MAX_PATH] = { 0 };

    if (temp_dir == 0 || temp_dir[0] == 0)
    {
        ::GetTempPathW(sizeof(temp_dir2), temp_dir2);
        temp_dir = temp_dir2;
    }
    if(0 == ::GetTempFileNameW(temp_dir, L"ocv", 0, temp_file))
    {
        m_path = WString();
    }
    else
    {
        DeleteFileW(temp_file);

        m_path = temp_file;
    }
#endif
#else
    m_path = tempfile();
#endif
}

FILE* Path::openPath(const Path& mode) const
{
#if defined _WIN32
    return( _wfopen( m_path.c_str(), mode.c_str() ) );
#else
    return( fopen( m_path.c_str(), mode.c_str() ) );
#endif
}

int Path::removePath() const
{
#if defined _WIN32
    return(_wremove( m_path.c_str() ));
#else
    return(remove( m_path.c_str() ));
#endif
}

String Path::string() const
{
#if defined _WIN32
    return( toString( m_path ) );
#else
    return( m_path );
#endif
}

Path::PathType Path::native() const
{
    return( m_path );
}

WString Path::wstring() const
{
#if defined _WIN32
    return( m_path );
#else
    return( toWString( m_path ) );
#endif
}

String Path::toString(const WString& wstr)
{
    if(wstr.empty())
    {
        //Return the empty string
        return("");
    }

#if defined _WIN32
    UINT code_page = AreFileApisANSI() ? CP_ACP : CP_OEMCP;

    //Calculate target buffer size (not including the zero terminator)
    int wstring_size = static_cast<int>(wstr.size());
    int size = WideCharToMultiByte(code_page, WC_NO_BEST_FIT_CHARS, wstr.c_str(), wstring_size, NULL, 0, NULL, NULL);
    if(size == 0)
    {
        //Conversion failed
        return("");
    }

    //A string is contiguous with C++11
    String str(size, ' ');
    WideCharToMultiByte(code_page, WC_NO_BEST_FIT_CHARS, wstr.c_str(), wstring_size, &str[0], size, NULL, NULL);

    return(str);
#else
    std::size_t size = std::wcstombs(NULL, wstr.c_str(), 0);
    if(size == String::npos)
    {
        //Conversion failed, return an empty string
        return("");
    }

    //A string is contiguous with C++11
    String str(size, ' ');

    //Convert to a narrow character string
    size = std::wcstombs(&str[0], wstr.c_str(), size);
    return(str);
#endif
}

WString Path::toWString(const String& str)
{
    if(str.empty())
    {
        //Return the empty wstring
        return(L"");
    }

#if defined _WIN32
    UINT code_page = AreFileApisANSI() ? CP_ACP : CP_OEMCP;

    //Calculate target buffer size (not including the zero terminator)
    int string_size = static_cast<int>(str.size());
    int size = MultiByteToWideChar(code_page, MB_PRECOMPOSED, str.c_str(), string_size, NULL, 0);
    if(size == 0)
    {
        //Conversion failed
        return(L"");
    }

    //A wstring is contiguous with C++11
    WString wstr(size, L' ');

    //No error checking. We already know, that the conversion will succeed.
    MultiByteToWideChar(code_page, MB_PRECOMPOSED, str.c_str(), string_size, &wstr[0], size);

    return(wstr);
#else
    std::size_t size = str.size();

    //A wstring is contiguous with C++11
    WString wstr(size, L' ');

    //Convert to a multibyte character string
    size = std::mbstowcs(&wstr[0], str.c_str(), size);
    if(size != String::npos)
    {
        //Shrink to fit
        wstr.resize(size);
        return(wstr);
    }

    //Conversion failed, return an empty wstring
    return(L"");
#endif
}

int Path::toLower(int c)
{
    return( tolower( c ) );
}

wint_t Path::toLower(wint_t c)
{
    return( towlower( c ) );
}

int Path::isAlpaNumeric( int ch )
{
    return( isalnum( ch ) );
}

int Path::isAlpaNumeric( wint_t ch )
{
    return( iswalnum( ch ) );
}

}

/* End of file. */
