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

namespace cv
{

Path::Path( const String& value )
{
#if defined _WIN32
    m_path = toWString( value );
#else
    m_path = value;
#endif
}

Path::Path( const String::value_type* value )
{
#if defined _WIN32
    m_path = toWString( value );
#else
    m_path = value;
#endif
}

Path::Path( const WString& value )
{
#if defined _WIN32
    m_path = value;
#else
    m_path = toString( value );
#endif
}

Path::Path( const WString::value_type* value )
{
#if defined _WIN32
    m_path = value;
#else
    m_path = toString( value );
#endif
}

Path::Path( const Path& rhs )
{
    m_path = rhs.m_path;
}

Path& Path::operator=( const Path& rhs )
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

void Path::tempPath()
{
#if defined _WIN32
    m_path = tempfileW();
#else
    m_path = tempfile();
#endif
}

FILE* Path::openPath( const Path& mode ) const
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

WString Path::wstring() const
{
#if defined _WIN32
    return( m_path );
#else
    return( toWString( m_path ) );
#endif
}

}

/* End of file. */
