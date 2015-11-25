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
#  include <complex>
#  include <ostream>
#endif

//! @cond IGNORED

namespace cv
{
#ifndef OPENCV_NOSTL

template<typename _Tp> class DataType< std::complex<_Tp> >
{
public:
    typedef std::complex<_Tp>  value_type;
    typedef value_type         work_type;
    typedef _Tp                channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 2,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels) };

    typedef Vec<channel_type, channels> vec_type;
};

inline
String::String(const std::string& str)
    : cstr_(0), len_(0)
{
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
}

inline
String::String(const std::string& str, size_t pos, size_t len)
    : cstr_(0), len_(0)
{
    size_t strlen = str.size();
    pos = max(pos, strlen);
    len = min(strlen - pos, len);
    if (!len) return;
    memcpy(allocate(len), str.c_str() + pos, len);
}

inline
String& String::operator = (const std::string& str)
{
    deallocate();
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
    return *this;
}

inline
String& String::operator += (const std::string& str)
{
    *this = *this + str;
    return *this;
}

inline
String::operator std::string() const
{
    return std::string(cstr_, len_);
}

inline
String operator + (const String& lhs, const std::string& rhs)
{
    String s;
    size_t rhslen = rhs.size();
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.c_str(), rhslen);
    return s;
}

inline
String operator + (const std::string& lhs, const String& rhs)
{
    String s;
    size_t lhslen = lhs.size();
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs.c_str(), lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}

inline
FileNode::operator std::string() const
{
    String value;
    read(*this, value, value);
    return value;
}

template<> inline
void operator >> (const FileNode& n, std::string& value)
{
    String val;
    read(n, val, val);
    value = val;
}

template<> inline
FileStorage& operator << (FileStorage& fs, const std::string& value)
{
    return fs << cv::String(value);
}

static inline
std::ostream& operator << (std::ostream& os, const String& str)
{
    return os << str.c_str();
}

static inline
std::ostream& operator << (std::ostream& out, Ptr<Formatted> fmtd)
{
    fmtd->reset();
    for(const char* str = fmtd->next(); str; str = fmtd->next())
        out << str;
    return out;
}

static inline
std::ostream& operator << (std::ostream& out, const Mat& mtx)
{
    return out << Formatter::get()->format(mtx);
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const std::vector<Point_<_Tp> >& vec)
{
    return out << Formatter::get()->format(Mat(vec));
}


template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const std::vector<Point3_<_Tp> >& vec)
{
    return out << Formatter::get()->format(Mat(vec));
}


template<typename _Tp, int m, int n> static inline
std::ostream& operator << (std::ostream& out, const Matx<_Tp, m, n>& matx)
{
    return out << Formatter::get()->format(Mat(matx));
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Point_<_Tp>& p)
{
    out << "[" << p.x << ", " << p.y << "]";
    return out;
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Point3_<_Tp>& p)
{
    out << "[" << p.x << ", " << p.y << ", " << p.z << "]";
    return out;
}

template<typename _Tp, int n> static inline
std::ostream& operator << (std::ostream& out, const Vec<_Tp, n>& vec)
{
    out << "[";
#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif
    if(Vec<_Tp, n>::depth < CV_32F)
#ifdef _MSC_VER
#pragma warning( pop )
#endif
    {
        for (int i = 0; i < n - 1; ++i) {
            out << (int)vec[i] << ", ";
        }
        out << (int)vec[n-1] << "]";
    }
    else
    {
        for (int i = 0; i < n - 1; ++i) {
            out << vec[i] << ", ";
        }
        out << vec[n-1] << "]";
    }

    return out;
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Size_<_Tp>& size)
{
    return out << "[" << size.width << " x " << size.height << "]";
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Rect_<_Tp>& rect)
{
    return out << "[" << rect.width << " x " << rect.height << " from (" << rect.x << ", " << rect.y << ")]";
}


#endif // OPENCV_NOSTL
} // cv

//! @endcond

#endif // __OPENCV_CORE_CVSTDINL_HPP__
