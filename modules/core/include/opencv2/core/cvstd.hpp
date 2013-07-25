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

#ifndef __cplusplus
#  error cvstd.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"

#include <cstddef>
#include <cstring>
#include <cctype>

#ifndef OPENCV_NOSTL
#  include <string>
#endif

// import useful primitives from stl
#ifndef OPENCV_NOSTL_TRANSITIONAL
#  include <algorithm>
#  include <utility>
#  include <cstdlib> //for abs(int)
#  include <cmath>

namespace cv
{
    using std::min;
    using std::max;
    using std::abs;
    using std::swap;
    using std::sqrt;
    using std::exp;
    using std::pow;
    using std::log;
}

namespace std
{
    static inline uchar abs(uchar a) { return a; }
    static inline ushort abs(ushort a) { return a; }
    static inline unsigned abs(unsigned a) { return a; }
    static inline uint64 abs(uint64 a) { return a; }
}

#else
namespace cv
{
    template<typename T> static inline T min(T a, T b) { return a < b ? a : b; }
    template<typename T> static inline T max(T a, T b) { return a > b ? a : b; }
    template<typename T> static inline T abs(T a) { return a < 0 ? -a : a; }
    template<typename T> static inline void swap(T& a, T& b) { T tmp = a; a = b; b = tmp; }

    template<> inline uchar abs(uchar a) { return a; }
    template<> inline ushort abs(ushort a) { return a; }
    template<> inline unsigned abs(unsigned a) { return a; }
    template<> inline uint64 abs(uint64 a) { return a; }
}
#endif

namespace cv {

//////////////////////////// memory management functions ////////////////////////////

/*!
  Allocates memory buffer

  This is specialized OpenCV memory allocation function that returns properly aligned memory buffers.
  The usage is identical to malloc(). The allocated buffers must be freed with cv::fastFree().
  If there is not enough memory, the function calls cv::error(), which raises an exception.

  \param bufSize buffer size in bytes
  \return the allocated memory buffer.
*/
CV_EXPORTS void* fastMalloc(size_t bufSize);

/*!
  Frees the memory allocated with cv::fastMalloc

  This is the corresponding deallocation function for cv::fastMalloc().
  When ptr==NULL, the function has no effect.
*/
CV_EXPORTS void fastFree(void* ptr);

/*!
  The STL-compilant memory Allocator based on cv::fastMalloc() and cv::fastFree()
*/
template<typename _Tp> class Allocator
{
public:
    typedef _Tp value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> class rebind { typedef Allocator<U> other; };

    explicit Allocator() {}
    ~Allocator() {}
    explicit Allocator(Allocator const&) {}
    template<typename U>
    explicit Allocator(Allocator<U> const&) {}

    // address
    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type count, const void* =0) { return reinterpret_cast<pointer>(fastMalloc(count * sizeof (_Tp))); }
    void deallocate(pointer p, size_type) { fastFree(p); }

    void construct(pointer p, const _Tp& v) { new(static_cast<void*>(p)) _Tp(v); }
    void destroy(pointer p) { p->~_Tp(); }

    size_type max_size() const { return cv::max(static_cast<_Tp>(-1)/sizeof(_Tp), 1); }
};



//////////////////// generic_type ref-counting pointer class for C/C++ objects ////////////////////////

/*!
  Smart pointer to dynamically allocated objects.

  This is template pointer-wrapping class that stores the associated reference counter along with the
  object pointer. The class is similar to std::smart_ptr<> from the recent addons to the C++ standard,
  but is shorter to write :) and self-contained (i.e. does add any dependency on the compiler or an external library).

  Basically, you can use "Ptr<MyObjectType> ptr" (or faster "const Ptr<MyObjectType>& ptr" for read-only access)
  everywhere instead of "MyObjectType* ptr", where MyObjectType is some C structure or a C++ class.
  To make it all work, you need to specialize Ptr<>::delete_obj(), like:

  \code
  template<> CV_EXPORTS void Ptr<MyObjectType>::delete_obj() { call_destructor_func(obj); }
  \endcode

  \note{if MyObjectType is a C++ class with a destructor, you do not need to specialize delete_obj(),
  since the default implementation calls "delete obj;"}

  \note{Another good property of the class is that the operations on the reference counter are atomic,
  i.e. it is safe to use the class in multi-threaded applications}
*/
template<typename _Tp> class Ptr
{
public:
    //! empty constructor
    Ptr();
    //! take ownership of the pointer. The associated reference counter is allocated and set to 1
    Ptr(_Tp* _obj);
    //! calls release()
    ~Ptr();
    //! copy constructor. Copies the members and calls addref()
    Ptr(const Ptr& ptr);
    template<typename _Tp2> Ptr(const Ptr<_Tp2>& ptr);
    //! copy operator. Calls ptr.addref() and release() before copying the members
    Ptr& operator = (const Ptr& ptr);
    //! increments the reference counter
    void addref();
    //! decrements the reference counter. If it reaches 0, delete_obj() is called
    void release();
    //! deletes the object. Override if needed
    void delete_obj();
    //! returns true iff obj==NULL
    bool empty() const;

    //! cast pointer to another type
    template<typename _Tp2> Ptr<_Tp2> ptr();
    template<typename _Tp2> const Ptr<_Tp2> ptr() const;

    //! helper operators making "Ptr<T> ptr" use very similar to "T* ptr".
    _Tp* operator -> ();
    const _Tp* operator -> () const;

    operator _Tp* ();
    operator const _Tp*() const;

    _Tp* obj; //< the object pointer.
    int* refcount; //< the associated reference counter
};



//////////////////////////////// string class ////////////////////////////////

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

    explicit String();
    String(const String& str);
    String(const String& str, size_t pos, size_t len = npos);
    String(const char* s);
    String(const char* s, size_t n);
    String(size_t n, char c);
    String(const char* first, const char* last);
    template<typename Iterator> String(Iterator first, Iterator last);
    explicit String(const FileNode& fn);
    ~String();

    String& operator=(const String& str);
    String& operator=(const char* s);
    String& operator=(char c);

    size_t size() const;
    size_t length() const;

    char operator[](size_t idx) const;
    char operator[](int idx) const;

    const char* begin() const;
    const char* end() const;

    const char* c_str() const;

    bool empty() const;
    void clear();

    int compare(const char* s) const;
    int compare(const String& str) const;

    void swap(String& str);
    String substr(size_t pos = 0, size_t len = npos) const;

    size_t find(const char* s, size_t pos, size_t n) const;
    size_t find(char c, size_t pos = 0) const;
    size_t find(const String& str, size_t pos = 0) const;
    size_t find(const char* s, size_t pos = 0) const;

    size_t rfind(const char* s, size_t pos, size_t n) const;
    size_t rfind(char c, size_t pos = npos) const;
    size_t rfind(const String& str, size_t pos = npos) const;
    size_t rfind(const char* s, size_t pos = npos) const;

    size_t find_first_of(const char* s, size_t pos, size_t n) const;
    size_t find_first_of(char c, size_t pos = 0) const;
    size_t find_first_of(const String& str, size_t pos = 0) const;
    size_t find_first_of(const char* s, size_t pos = 0) const;

    size_t find_last_of(const char* s, size_t pos, size_t n) const;
    size_t find_last_of(char c, size_t pos = npos) const;
    size_t find_last_of(const String& str, size_t pos = npos) const;
    size_t find_last_of(const char* s, size_t pos = npos) const;

    friend String operator+ (const String& lhs, const String& rhs);
    friend String operator+ (const String& lhs, const char*   rhs);
    friend String operator+ (const char*   lhs, const String& rhs);
    friend String operator+ (const String& lhs, char          rhs);
    friend String operator+ (char          lhs, const String& rhs);

    String toLowerCase() const;

#ifndef OPENCV_NOSTL
    String(const std::string& str);
    String(const std::string& str, size_t pos, size_t len = npos);
    String& operator=(const std::string& str);
    operator std::string() const;

    friend String operator+ (const String& lhs, const std::string& rhs);
    friend String operator+ (const std::string& lhs, const String& rhs);
#endif

private:
    char*  cstr_;
    size_t len_;

    char* allocate(size_t len); // len without trailing 0
    void deallocate();
};



/////////////////////////// cv::Ptr implementation ///////////////////////////

template<typename _Tp> inline
Ptr<_Tp>::Ptr()
    : obj(0), refcount(0) {}

template<typename _Tp> inline
Ptr<_Tp>::Ptr(_Tp* _obj)
    : obj(_obj)
{
    if(obj)
    {
        refcount = (int*)fastMalloc(sizeof(*refcount));
        *refcount = 1;
    }
    else
        refcount = 0;
}

template<typename _Tp> template<typename _Tp2>
Ptr<_Tp>::Ptr(const Ptr<_Tp2>& p)
    : obj(0), refcount(0)
{
    if (p.empty())
        return;

    _Tp* p_casted = dynamic_cast<_Tp*>(p.obj);
    if (!p_casted)
        return;

    obj = p_casted;
    refcount = p.refcount;
    addref();
}

template<typename _Tp> inline
Ptr<_Tp>::~Ptr()
{
    release();
}

template<typename _Tp> inline
void Ptr<_Tp>::addref()
{
    if( refcount )
        CV_XADD(refcount, 1);
}

template<typename _Tp> inline
void Ptr<_Tp>::release()
{
    if( refcount && CV_XADD(refcount, -1) == 1 )
    {
        delete_obj();
        fastFree(refcount);
    }
    refcount = 0;
    obj = 0;
}

template<typename _Tp> inline
void Ptr<_Tp>::delete_obj()
{
    if( obj )
        delete obj;
}

template<typename _Tp> inline
Ptr<_Tp>::Ptr(const Ptr<_Tp>& _ptr)
{
    obj = _ptr.obj;
    refcount = _ptr.refcount;
    addref();
}

template<typename _Tp> inline
Ptr<_Tp>& Ptr<_Tp>::operator = (const Ptr<_Tp>& _ptr)
{
    int* _refcount = _ptr.refcount;
    if( _refcount )
        CV_XADD(_refcount, 1);
    release();
    obj = _ptr.obj;
    refcount = _refcount;
    return *this;
}

template<typename _Tp> inline
_Tp* Ptr<_Tp>::operator -> ()
{
    return obj;
}

template<typename _Tp> inline
const _Tp* Ptr<_Tp>::operator -> () const
{
    return obj;
}

template<typename _Tp> inline
Ptr<_Tp>::operator _Tp* ()
{
    return obj;
}

template<typename _Tp> inline
Ptr<_Tp>::operator const _Tp*() const
{
    return obj;
}

template<typename _Tp> inline
bool Ptr<_Tp>::empty() const
{
    return obj == 0;
}

template<typename _Tp> template<typename _Tp2> inline
Ptr<_Tp2> Ptr<_Tp>::ptr()
{
    Ptr<_Tp2> p;
    if( !obj )
        return p;

    _Tp2* obj_casted = dynamic_cast<_Tp2*>(obj);
    if (!obj_casted)
        return p;

    if( refcount )
        CV_XADD(refcount, 1);

    p.obj = obj_casted;
    p.refcount = refcount;
    return p;
}

template<typename _Tp> template<typename _Tp2> inline
const Ptr<_Tp2> Ptr<_Tp>::ptr() const
{
    Ptr<_Tp2> p;
    if( !obj )
        return p;

    _Tp2* obj_casted = dynamic_cast<_Tp2*>(obj);
    if (!obj_casted)
        return p;

    if( refcount )
        CV_XADD(refcount, 1);

    p.obj = obj_casted;
    p.refcount = refcount;
    return p;
}

template<class _Tp, class _Tp2> static inline
bool operator == (const Ptr<_Tp>& a, const Ptr<_Tp2>& b)
{
    return a.refcount == b.refcount;
}

template<class _Tp, class _Tp2> static inline
bool operator != (const Ptr<_Tp>& a, const Ptr<_Tp2>& b)
{
    return a.refcount != b.refcount;
}



////////////////////////// cv::String implementation /////////////////////////

inline
String::String()
    : cstr_(0), len_(0)
{}

inline
String::String(const String& str)
    : cstr_(str.cstr_), len_(str.len_)
{
    if (cstr_)
        CV_XADD(((int*)cstr_)-1, 1);
}

inline
String::String(const String& str, size_t pos, size_t len)
    : cstr_(0), len_(0)
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

inline
String::String(const char* s)
    : cstr_(0), len_(0)
{
    if (!s) return;
    size_t len = strlen(s);
    memcpy(allocate(len), s, len);
}

inline
String::String(const char* s, size_t n)
    : cstr_(0), len_(0)
{
    if (!n) return;
    memcpy(allocate(n), s, n);
}

inline
String::String(size_t n, char c)
    : cstr_(0), len_(0)
{
    memset(allocate(n), c, n);
}

inline
String::String(const char* first, const char* last)
    : cstr_(0), len_(0)
{
    size_t len = (size_t)(last - first);
    memcpy(allocate(len), first, len);
}

template<typename Iterator> inline
String::String(Iterator first, Iterator last)
    : cstr_(0), len_(0)
{
    size_t len = (size_t)(last - first);
    char* str = allocate(len);
    while (first != last)
    {
        *str++ = *first;
        ++first;
    }
}

inline
String::~String()
{
    deallocate();
}

inline
String& String::operator=(const String& str)
{
    if (&str == this) return *this;

    deallocate();
    if (str.cstr_) CV_XADD(((int*)str.cstr_)-1, 1);
    cstr_ = str.cstr_;
    len_ = str.len_;
    return *this;
}

inline
String& String::operator=(const char* s)
{
    deallocate();
    if (!s) return *this;
    size_t len = strlen(s);
    memcpy(allocate(len), s, len);
    return *this;
}

inline
String& String::operator=(char c)
{
    deallocate();
    allocate(1)[0] = c;
    return *this;
}

inline
size_t String::size() const
{
    return len_;
}

inline
size_t String::length() const
{
    return len_;
}

inline
char String::operator[](size_t idx) const
{
    return cstr_[idx];
}

inline
char String::operator[](int idx) const
{
    return cstr_[idx];
}

inline
const char* String::begin() const
{
    return cstr_;
}

inline
const char* String::end() const
{
    return len_ ? cstr_ + 1 : 0;
}

inline
bool String::empty() const
{
    return len_ == 0;
}

inline
const char* String::c_str() const
{
    return cstr_ ? cstr_ : "";
}

inline
void String::swap(String& str)
{
    cv::swap(cstr_, str.cstr_);
    cv::swap(len_, str.len_);
}

inline
void String::clear()
{
    deallocate();
}

inline
int String::compare(const char* s) const
{
    if (cstr_ == s) return 0;
    return strcmp(c_str(), s);
}

inline
int String::compare(const String& str) const
{
    if (cstr_ == str.cstr_) return 0;
    return strcmp(c_str(), str.c_str());
}

inline
String String::substr(size_t pos, size_t len) const
{
    return String(*this, pos, len);
}

inline
size_t String::find(const char* s, size_t pos, size_t n) const
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

inline
size_t String::find(char c, size_t pos) const
{
    return find(&c, pos, 1);
}

inline
size_t String::find(const String& str, size_t pos) const
{
    return find(str.c_str(), pos, str.len_);
}

inline
size_t String::find(const char* s, size_t pos) const
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

inline
size_t String::rfind(const char* s, size_t pos, size_t n) const
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

inline
size_t String::rfind(char c, size_t pos) const
{
    return rfind(&c, pos, 1);
}

inline
size_t String::rfind(const String& str, size_t pos) const
{
    return rfind(str.c_str(), pos, str.len_);
}

inline
size_t String::rfind(const char* s, size_t pos) const
{
    return rfind(s, pos, strlen(s));
}

inline
size_t String::find_first_of(const char* s, size_t pos, size_t n) const
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

inline
size_t String::find_first_of(char c, size_t pos) const
{
    return find_first_of(&c, pos, 1);
}

inline
size_t String::find_first_of(const String& str, size_t pos) const
{
    return find_first_of(str.c_str(), pos, str.len_);
}

inline
size_t String::find_first_of(const char* s, size_t pos) const
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

inline
size_t String::find_last_of(const char* s, size_t pos, size_t n) const
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

inline
size_t String::find_last_of(char c, size_t pos) const
{
    return find_last_of(&c, pos, 1);
}

inline
size_t String::find_last_of(const String& str, size_t pos) const
{
    return find_last_of(str.c_str(), pos, str.len_);
}

inline
size_t String::find_last_of(const char* s, size_t pos) const
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

inline
String String::toLowerCase() const
{
    String res(cstr_, len_);

    for (size_t i = 0; i < len_; ++i)
        res.cstr_[i] = (char) ::tolower(cstr_[i]);

    return res;
}

// ************************* cv::String non-member functions *************************

inline
String operator + (const String& lhs, const String& rhs)
{
    String s;
    s.allocate(lhs.len_ + rhs.len_);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.cstr_, rhs.len_);
    return s;
}

inline
String operator + (const String& lhs, const char* rhs)
{
    String s;
    size_t rhslen = strlen(rhs);
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs, rhslen);
    return s;
}

inline
String operator + (const char* lhs, const String& rhs)
{
    String s;
    size_t lhslen = strlen(lhs);
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs, lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}

inline
String operator + (const String& lhs, char rhs)
{
    String s;
    s.allocate(lhs.len_ + 1);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    s.cstr_[lhs.len_] = rhs;
    return s;
}

inline
String operator + (char lhs, const String& rhs)
{
    String s;
    s.allocate(rhs.len_ + 1);
    s.cstr_[0] = lhs;
    memcpy(s.cstr_ + 1, rhs.cstr_, rhs.len_);
    return s;
}

static inline bool operator== (const String& lhs, const String& rhs) { return 0 == lhs.compare(rhs); }
static inline bool operator== (const char*   lhs, const String& rhs) { return 0 == rhs.compare(lhs); }
static inline bool operator== (const String& lhs, const char*   rhs) { return 0 == lhs.compare(rhs); }
static inline bool operator!= (const String& lhs, const String& rhs) { return 0 != lhs.compare(rhs); }
static inline bool operator!= (const char*   lhs, const String& rhs) { return 0 != rhs.compare(lhs); }
static inline bool operator!= (const String& lhs, const char*   rhs) { return 0 != lhs.compare(rhs); }
static inline bool operator<  (const String& lhs, const String& rhs) { return lhs.compare(rhs) <  0; }
static inline bool operator<  (const char*   lhs, const String& rhs) { return rhs.compare(lhs) >  0; }
static inline bool operator<  (const String& lhs, const char*   rhs) { return lhs.compare(rhs) <  0; }
static inline bool operator<= (const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }
static inline bool operator<= (const char*   lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }
static inline bool operator<= (const String& lhs, const char*   rhs) { return lhs.compare(rhs) <= 0; }
static inline bool operator>  (const String& lhs, const String& rhs) { return lhs.compare(rhs) >  0; }
static inline bool operator>  (const char*   lhs, const String& rhs) { return rhs.compare(lhs) <  0; }
static inline bool operator>  (const String& lhs, const char*   rhs) { return lhs.compare(rhs) >  0; }
static inline bool operator>= (const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }
static inline bool operator>= (const char*   lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }
static inline bool operator>= (const String& lhs, const char*   rhs) { return lhs.compare(rhs) >= 0; }

} // cv

#ifndef OPENCV_NOSTL_TRANSITIONAL
namespace std
#else
namespace cv
#endif
{
    template<> inline
    void swap<cv::String>(cv::String& a, cv::String& b)
    {
        a.swap(b);
    }
}

#endif //__OPENCV_CORE_CVSTD_HPP__
