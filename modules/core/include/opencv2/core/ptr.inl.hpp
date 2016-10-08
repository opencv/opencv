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
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
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
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_PTR_INL_HPP
#define OPENCV_CORE_PTR_INL_HPP

#include <algorithm>

//! @cond IGNORED

namespace cv {

template<typename Y>
void DefaultDeleter<Y>::operator () (Y* p) const
{
    delete p;
}

namespace detail
{

struct PtrOwner
{
    PtrOwner() : refCount(1)
    {}

    void incRef()
    {
        CV_XADD(&refCount, 1);
    }

    void decRef()
    {
        if (CV_XADD(&refCount, -1) == 1) deleteSelf();
    }

protected:
    /* This doesn't really need to be virtual, since PtrOwner is never deleted
       directly, but it doesn't hurt and it helps avoid warnings. */
    virtual ~PtrOwner()
    {}

    virtual void deleteSelf() = 0;

private:
    unsigned int refCount;

    // noncopyable
    PtrOwner(const PtrOwner&);
    PtrOwner& operator = (const PtrOwner&);
};

template<typename Y, typename D>
struct PtrOwnerImpl : PtrOwner
{
    PtrOwnerImpl(Y* p, D d) : owned(p), deleter(d)
    {}

    void deleteSelf()
    {
        deleter(owned);
        delete this;
    }

private:
    Y* owned;
    D deleter;
};


}

template<typename T>
Ptr<T>::Ptr() : owner(NULL), stored(NULL)
{}

template<typename T>
template<typename Y>
Ptr<T>::Ptr(Y* p)
  : owner(p
      ? new detail::PtrOwnerImpl<Y, DefaultDeleter<Y> >(p, DefaultDeleter<Y>())
      : NULL),
    stored(p)
{}

template<typename T>
template<typename Y, typename D>
Ptr<T>::Ptr(Y* p, D d)
  : owner(p
      ? new detail::PtrOwnerImpl<Y, D>(p, d)
      : NULL),
    stored(p)
{}

template<typename T>
Ptr<T>::Ptr(const Ptr& o) : owner(o.owner), stored(o.stored)
{
    if (owner) owner->incRef();
}

template<typename T>
template<typename Y>
Ptr<T>::Ptr(const Ptr<Y>& o) : owner(o.owner), stored(o.stored)
{
    if (owner) owner->incRef();
}

template<typename T>
template<typename Y>
Ptr<T>::Ptr(const Ptr<Y>& o, T* p) : owner(o.owner), stored(p)
{
    if (owner) owner->incRef();
}

template<typename T>
Ptr<T>::~Ptr()
{
    release();
}

template<typename T>
Ptr<T>& Ptr<T>::operator = (const Ptr<T>& o)
{
    Ptr(o).swap(*this);
    return *this;
}

template<typename T>
template<typename Y>
Ptr<T>& Ptr<T>::operator = (const Ptr<Y>& o)
{
    Ptr(o).swap(*this);
    return *this;
}

template<typename T>
void Ptr<T>::release()
{
    if (owner) owner->decRef();
    owner = NULL;
    stored = NULL;
}

template<typename T>
template<typename Y>
void Ptr<T>::reset(Y* p)
{
    Ptr(p).swap(*this);
}

template<typename T>
template<typename Y, typename D>
void Ptr<T>::reset(Y* p, D d)
{
    Ptr(p, d).swap(*this);
}

template<typename T>
void Ptr<T>::swap(Ptr<T>& o)
{
    std::swap(owner, o.owner);
    std::swap(stored, o.stored);
}

template<typename T>
T* Ptr<T>::get() const
{
    return stored;
}

template<typename T>
typename detail::RefOrVoid<T>::type Ptr<T>::operator * () const
{
    return *stored;
}

template<typename T>
T* Ptr<T>::operator -> () const
{
    return stored;
}

template<typename T>
Ptr<T>::operator T* () const
{
    return stored;
}


template<typename T>
bool Ptr<T>::empty() const
{
    return !stored;
}

template<typename T>
template<typename Y>
Ptr<Y> Ptr<T>::staticCast() const
{
    return Ptr<Y>(*this, static_cast<Y*>(stored));
}

template<typename T>
template<typename Y>
Ptr<Y> Ptr<T>::constCast() const
{
    return Ptr<Y>(*this, const_cast<Y*>(stored));
}

template<typename T>
template<typename Y>
Ptr<Y> Ptr<T>::dynamicCast() const
{
    return Ptr<Y>(*this, dynamic_cast<Y*>(stored));
}

#ifdef CV_CXX_MOVE_SEMANTICS

template<typename T>
Ptr<T>::Ptr(Ptr&& o) : owner(o.owner), stored(o.stored)
{
    o.owner = NULL;
    o.stored = NULL;
}

template<typename T>
Ptr<T>& Ptr<T>::operator = (Ptr<T>&& o)
{
    if (this == &o)
        return *this;

    release();
    owner = o.owner;
    stored = o.stored;
    o.owner = NULL;
    o.stored = NULL;
    return *this;
}

#endif


template<typename T>
void swap(Ptr<T>& ptr1, Ptr<T>& ptr2){
    ptr1.swap(ptr2);
}

template<typename T>
bool operator == (const Ptr<T>& ptr1, const Ptr<T>& ptr2)
{
    return ptr1.get() == ptr2.get();
}

template<typename T>
bool operator != (const Ptr<T>& ptr1, const Ptr<T>& ptr2)
{
    return ptr1.get() != ptr2.get();
}

template<typename T>
Ptr<T> makePtr()
{
    return Ptr<T>(new T());
}

template<typename T, typename A1>
Ptr<T> makePtr(const A1& a1)
{
    return Ptr<T>(new T(a1));
}

template<typename T, typename A1, typename A2>
Ptr<T> makePtr(const A1& a1, const A2& a2)
{
    return Ptr<T>(new T(a1, a2));
}

template<typename T, typename A1, typename A2, typename A3>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3)
{
    return Ptr<T>(new T(a1, a2, a3));
}

template<typename T, typename A1, typename A2, typename A3, typename A4>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4)
{
    return Ptr<T>(new T(a1, a2, a3, a4));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10, const A11& a11)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10, const A11& a11, const A12& a12)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12));
}
} // namespace cv

//! @endcond

#endif // OPENCV_CORE_PTR_INL_HPP
