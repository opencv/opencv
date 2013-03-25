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

#ifndef __OPENCV_CORE_CVSTD_HPP__
#define __OPENCV_CORE_CVSTD_HPP__

#include <cstddef>
#include <cstring>

#include "opencv2/core/cvdef.h"

#ifndef OPENCV_NOSTL
#  include <string>
#endif

// import useful primitives from stl
#ifndef OPENCV_NOSTL_TRANSITIONAL
#  include <algorithm>
#  include <utility>
#  include <cmath>

namespace cv
{
    using std::min;
    using std::max;
    using std::abs;
    using std::swap;
}

#else
namespace cv
{
    template<typename T> inline T min(T a, T b) { return a < b ? a : b; }
    template<typename T> inline T max(T a, T b) { return a > b ? a : b; }
    template<typename T> inline T abs(T a) { return a < 0 ? -a : a; }
    template<typename T> inline void swap(T& a, T& b) { T tmp = a; a = b; b = tmp; }

    template<> inline uchar abs(uchar a) { return a; }
    template<> inline ushort abs(ushort a) { return a; }
    template<> inline uint abs(uint a) { return a; }
    template<> inline uint64 abs(uint64 a) { return a; }
}
#endif

namespace cv {

class CV_EXPORTS FileNode; //for string constructor from FileNode

class CV_EXPORTS String
{
public:
    typedef char value_type;
    typedef char& reference;
    typedef const char& const_reference;
    typedef char* pointer;
    typedef const char* const_pointer;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;

    typedef char* iterator;
    typedef const char* const_iterator;

    static const size_t npos = size_t(-1);

    explicit String() : cstr_(0), len_(0) {}

    String(const String& str) : cstr_(str.cstr_), len_(str.len_)
    {
        if (cstr_) CV_XADD(((int*)cstr_)-1, 1);
    }

    String(const String& str, size_t pos, size_t len = npos) : cstr_(0), len_(0)
    {
        pos = min(pos, str.len_);
        len = min(str.len_ - pos, len);
        if (!len) return;
        if (len == str.len_)
        {
            CV_XADD(((int*)str.cstr_)-1, 1);
            cstr_ = str.cstr_;
            len_ = str.len_;
            return;
        }
        memcpy(allocate(len), str.cstr_ + pos, len);
    }

    String(const char* s): cstr_(0), len_(0)
    {
        if (!s) return;
        size_t len = strlen(s);
        memcpy(allocate(len), s, len);
    }

    String(const char* s, size_t n): cstr_(0), len_(0)
    {
        if (!n) return;
        memcpy(allocate(n), s, n);
    }

    String(size_t n, char c): cstr_(0), len_(0)
    {
        memset(allocate(n), c, n);
    }

    String(const char* first, const char* last): cstr_(0), len_(0)
    {
        size_t len = (size_t)(last - first);
        memcpy(allocate(len), first, len);
    }

    template<typename Iterator>
    String(Iterator first, Iterator last): cstr_(0), len_(0)
    {
        size_t len = (size_t)(last - first);
        char* str = allocate(len);
        while (first != last)
        {
            *str++ = *first;
            ++first;
        }
    }

    ~String() { deallocate(); }

    String& operator=(const String& str)
    {
        deallocate();
        if (str.cstr_) CV_XADD(((int*)str.cstr_)-1, 1);
        cstr_ = str.cstr_;
        len_ = str.len_;
        return *this;
    }

    String& operator=(const char* s)
    {
        deallocate();
        if (!s) return *this;
        size_t len = strlen(s);
        memcpy(allocate(len), s, len);
        return *this;
    }

    String& operator=(char c)
    {
        deallocate();
        allocate(1)[0] = c;
        return *this;
    }

    size_t size() const { return len_; }
    size_t length() const { return len_; }

    char operator[](size_t idx) const { return cstr_[idx]; }
    char operator[](int idx) const    { return cstr_[idx]; }

    const char* begin() const { return cstr_; }
    const char* end() const { return len_ ? cstr_ + 1 : 0; }

    bool empty() const { return len_ == 0; }

    const char* c_str() const { return cstr_ ? cstr_ : ""; }

    void swap(String& str)
    {
        cv::swap(cstr_, str.cstr_);
        cv::swap(len_, str.len_);
    }

    void clear() { deallocate(); }

    int compare(const char* s) const
    {
        if (cstr_ == s) return 0;

        return strcmp(c_str(), s);
    }

    int compare(const String& str) const
    {
        if (cstr_ == str.cstr_) return 0;

        return strcmp(c_str(), str.c_str());
    }

    String substr(size_t pos = 0, size_t len = npos) const
    {
        return String(*this, pos, len);
    }

    size_t find(const char* s, size_t pos, size_t n) const
    {
        if (n == 0 || pos + n > len_) return npos;
        const char* lmax = cstr_ + len_ - n;
        for (const char* i = cstr_ + pos; i <= lmax; ++i)
        {
            size_t j = 0;
            while (j < n && s[j] == i[j]) ++j;
            if (j == n) return (size_t)(i - cstr_);
        }
        return npos;
    }

    size_t find(char c, size_t pos = 0) const
    {
        return find(&c, pos, 1);
    }

    size_t find(const String& str, size_t pos = 0) const
    {
        return find(str.c_str(), pos, str.len_);
    }

    size_t find(const char* s, size_t pos = 0) const
    {
        if (pos >= len_ || !s[0]) return npos;
        const char* lmax = cstr_ + len_;
        for (const char* i = cstr_ + pos; i < lmax; ++i)
        {
            size_t j = 0;
            while (s[j] && s[j] == i[j])
            {   if(i + j >= lmax) return npos;
                ++j;
            }
            if (!s[j]) return (size_t)(i - cstr_);
        }
        return npos;
    }

    size_t rfind(const char* s, size_t pos, size_t n) const
    {
        if (n > len_) return npos;
        if (pos > len_ - n) pos = len_ - n;
        for (const char* i = cstr_ + pos; i >= cstr_; --i)
        {
            size_t j = 0;
            while (j < n && s[j] == i[j]) ++j;
            if (j == n) return (size_t)(i - cstr_);
        }
        return npos;
    }

    size_t rfind(char c, size_t pos = npos) const
    {
        return rfind(&c, pos, 1);
    }

    size_t rfind(const String& str, size_t pos = npos) const
    {
        return rfind(str.c_str(), pos, str.len_);
    }

    size_t rfind(const char* s, size_t pos = npos) const
    {
        return rfind(s, pos, strlen(s));
    }

    size_t find_first_of(const char* s, size_t pos, size_t n) const
    {
        if (n == 0 || pos + n > len_) return npos;
        const char* lmax = cstr_ + len_;
        for (const char* i = cstr_ + pos; i < lmax; ++i)
        {
            for (size_t j = 0; j < n; ++j)
                if (s[j] == *i)
                    return (size_t)(i - cstr_);
        }
        return npos;
    }

    size_t find_first_of(char c, size_t pos = 0) const
    {
        return find_first_of(&c, pos, 1);
    }

    size_t find_first_of(const String& str, size_t pos = 0) const
    {
        return find_first_of(str.c_str(), pos, str.len_);
    }

    size_t find_first_of(const char* s, size_t pos = 0) const
    {
        if (pos >= len_ || !s[0]) return npos;
        const char* lmax = cstr_ + len_;
        for (const char* i = cstr_ + pos; i < lmax; ++i)
        {
            for (size_t j = 0; s[j]; ++j)
                if (s[j] == *i)
                    return (size_t)(i - cstr_);
        }
        return npos;
    }

    size_t find_last_of(const char* s, size_t pos, size_t n) const
    {
        if (pos >= len_) pos = len_ - 1;
        for (const char* i = cstr_ + pos; i >= cstr_; --i)
        {
            for (size_t j = 0; j < n; ++j)
                if (s[j] == *i)
                    return (size_t)(i - cstr_);
        }
        return npos;
    }

    size_t find_last_of(char c, size_t pos = npos) const
    {
        return find_last_of(&c, pos, 1);
    }

    size_t find_last_of(const String& str, size_t pos = npos) const
    {
        return find_last_of(str.c_str(), pos, str.len_);
    }

    size_t find_last_of(const char* s, size_t pos = npos) const
    {
        if (pos >= len_) pos = len_ - 1;
        for (const char* i = cstr_ + pos; i >= cstr_; --i)
        {
            for (size_t j = 0; s[j]; ++j)
                if (s[j] == *i)
                    return (size_t)(i - cstr_);
        }
        return npos;
    }

    friend String operator+ (const String& lhs, const String& rhs);
    friend String operator+ (const String& lhs, const char*   rhs);
    friend String operator+ (const char*   lhs, const String& rhs);
    friend String operator+ (const String& lhs, char          rhs);
    friend String operator+ (char          lhs, const String& rhs);

#ifndef OPENCV_NOSTL
    String(const std::string& str);
    String(const std::string& str, size_t pos, size_t len = npos);
    String& operator=(const std::string& str);
    operator std::string() const;

    friend String operator+ (const String& lhs, const std::string& rhs);
    friend String operator+ (const std::string& lhs, const String& rhs);
#endif

    explicit String(const FileNode& fn);

private:
    char*  cstr_;
    size_t len_;

    char* allocate(size_t len); // len_ without trailing 0
    void deallocate();
};

inline String operator+ (const String& lhs, const String& rhs)
{
    String s;
    s.allocate(lhs.len_ + rhs.len_);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.cstr_, rhs.len_);
    return s;
}

inline String operator+ (const String& lhs, const char* rhs)
{
    String s;
    size_t rhslen = strlen(rhs);
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs, rhslen);
    return s;
}

inline String operator+ (const char* lhs, const String& rhs)
{
    String s;
    size_t lhslen = strlen(lhs);
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs, lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}

inline String operator+ (const String& lhs, char rhs)
{
    String s;
    s.allocate(lhs.len_ + 1);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    s.cstr_[lhs.len_] = rhs;
    return s;
}

inline String operator+ (char lhs, const String& rhs)
{
    String s;
    s.allocate(rhs.len_ + 1);
    s.cstr_[0] = lhs;
    memcpy(s.cstr_ + 1, rhs.cstr_, rhs.len_);
    return s;
}

inline bool operator== (const String& lhs, const String& rhs) { return 0 == lhs.compare(rhs); }
inline bool operator== (const char*   lhs, const String& rhs) { return 0 == rhs.compare(lhs); }
inline bool operator== (const String& lhs, const char*   rhs) { return 0 == lhs.compare(rhs); }
inline bool operator!= (const String& lhs, const String& rhs) { return 0 != lhs.compare(rhs); }
inline bool operator!= (const char*   lhs, const String& rhs) { return 0 != rhs.compare(lhs); }
inline bool operator!= (const String& lhs, const char*   rhs) { return 0 != lhs.compare(rhs); }
inline bool operator<  (const String& lhs, const String& rhs) { return lhs.compare(rhs) <  0; }
inline bool operator<  (const char*   lhs, const String& rhs) { return rhs.compare(lhs) >  0; }
inline bool operator<  (const String& lhs, const char*   rhs) { return lhs.compare(rhs) <  0; }
inline bool operator<= (const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }
inline bool operator<= (const char*   lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }
inline bool operator<= (const String& lhs, const char*   rhs) { return lhs.compare(rhs) <= 0; }
inline bool operator>  (const String& lhs, const String& rhs) { return lhs.compare(rhs) >  0; }
inline bool operator>  (const char*   lhs, const String& rhs) { return rhs.compare(lhs) <  0; }
inline bool operator>  (const String& lhs, const char*   rhs) { return lhs.compare(rhs) >  0; }
inline bool operator>= (const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }
inline bool operator>= (const char*   lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }
inline bool operator>= (const String& lhs, const char*   rhs) { return lhs.compare(rhs) >= 0; }

} // cv

#ifndef OPENCV_NOSTL_TRANSITIONAL
namespace std
#else
namespace cv
#endif
{
    template<> inline void swap<cv::String>(cv::String& a, cv::String& b) { a.swap(b); }
}

#endif //__OPENCV_CORE_CVSTD_HPP__
