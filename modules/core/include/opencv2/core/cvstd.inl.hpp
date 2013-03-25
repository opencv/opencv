/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_CORE_CVSTDINL_HPP__
#define __OPENCV_CORE_CVSTDINL_HPP__

#ifndef OPENCV_NOSTL
#  include <ostream>
#endif

namespace cv
{
#ifndef OPENCV_NOSTL

inline String::String(const std::string& str) : cstr_(0), len_(0)
{
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
}

inline String::String(const std::string& str, size_t pos, size_t len) : cstr_(0), len_(0)
{
    size_t strlen = str.size();
    pos = max(pos, strlen);
    len = min(strlen - pos, len);
    if (!len) return;
    memcpy(allocate(len), str.c_str() + pos, len);
}

inline String& String::operator=(const std::string& str)
{
    deallocate();
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
    return *this;
}

inline String::operator std::string() const
{
    return std::string(cstr_, len_);
}

inline String operator+ (const String& lhs, const std::string& rhs)
{
    String s;
    size_t rhslen = rhs.size();
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.c_str(), rhslen);
    return s;
}

inline String operator+ (const std::string& lhs, const String& rhs)
{
    String s;
    size_t lhslen = lhs.size();
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs.c_str(), lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}

inline std::ostream& operator << (std::ostream& os, const String& str)
{
    return os << str.c_str();
}

inline FileNode::operator std::string() const
{
    String value;
    read(*this, value, value);
    return value;
}

template<> inline void operator >> (const FileNode& n, std::string& value)
{
    String val;
    read(n, val, val);
    value = val;
}

#endif // OPENCV_NOSTL
} // cv

#endif // __OPENCV_CORE_CVSTDINL_HPP__