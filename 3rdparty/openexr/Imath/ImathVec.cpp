///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////



//----------------------------------------------------------------------------
//
//      Specializations of the Vec2<T> and Vec3<T> templates.
//
//----------------------------------------------------------------------------

#include "ImathVec.h"

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// suppress exception specification warnings
#pragma warning(disable:4290)
#endif


namespace Imath {

namespace
{

template<class T>
bool
normalizeOrThrow(Vec2<T> &v)
{
    int axis = -1;
    for (int i = 0; i < 2; i ++)
    {
        if (v[i] != 0)
        {
            if (axis != -1)
            {
                throw IntVecNormalizeExc ("Cannot normalize an integer "
                                          "vector unless it is parallel "
                                          "to a principal axis");
            }
            axis = i;
        }
    }
    v[axis] = (v[axis] > 0) ? 1 : -1;
    return true;
}


template<class T>
bool
normalizeOrThrow(Vec3<T> &v)
{
    int axis = -1;
    for (int i = 0; i < 3; i ++)
    {
        if (v[i] != 0)
        {
            if (axis != -1)
            {
                throw IntVecNormalizeExc ("Cannot normalize an integer "
                                          "vector unless it is parallel "
                                          "to a principal axis");
            }
            axis = i;
        }
    }
    v[axis] = (v[axis] > 0) ? 1 : -1;
    return true;
}


template<class T>
bool
normalizeOrThrow(Vec4<T> &v)
{
    int axis = -1;
    for (int i = 0; i < 4; i ++)
    {
        if (v[i] != 0)
        {
            if (axis != -1)
            {
                throw IntVecNormalizeExc ("Cannot normalize an integer "
                                          "vector unless it is parallel "
                                          "to a principal axis");
            }
            axis = i;
        }
    }
    v[axis] = (v[axis] > 0) ? 1 : -1;
    return true;
}

}


// Vec2<short>

template <> 
short
Vec2<short>::length () const
{
    float lenF = Math<float>::sqrt ((float)dot (*this));
    short lenS = (short) (lenF + 0.5f);
    return lenS;
}

template <>
const Vec2<short> &
Vec2<short>::normalize ()
{
    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
const Vec2<short> &
Vec2<short>::normalizeExc () throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
const Vec2<short> &
Vec2<short>::normalizeNonNull ()
{
    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
Vec2<short>
Vec2<short>::normalized () const
{
    Vec2<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}

template <>
Vec2<short>
Vec2<short>::normalizedExc () const throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    Vec2<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}

template <>
Vec2<short>
Vec2<short>::normalizedNonNull () const
{
    Vec2<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}


// Vec2<int>

template <> 
int
Vec2<int>::length () const
{
    float lenF = Math<float>::sqrt ((float)dot (*this));
    int lenI = (int) (lenF + 0.5f);
    return lenI;
}

template <>
const Vec2<int> &
Vec2<int>::normalize ()
{
    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
const Vec2<int> &
Vec2<int>::normalizeExc () throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
const Vec2<int> &
Vec2<int>::normalizeNonNull ()
{
    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
Vec2<int>
Vec2<int>::normalized () const
{
    Vec2<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}

template <>
Vec2<int>
Vec2<int>::normalizedExc () const throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    Vec2<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}

template <>
Vec2<int>
Vec2<int>::normalizedNonNull () const
{
    Vec2<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}


// Vec3<short>

template <> 
short
Vec3<short>::length () const
{
    float lenF = Math<float>::sqrt ((float)dot (*this));
    short lenS = (short) (lenF + 0.5f);
    return lenS;
}

template <>
const Vec3<short> &
Vec3<short>::normalize ()
{
    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
const Vec3<short> &
Vec3<short>::normalizeExc () throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
const Vec3<short> &
Vec3<short>::normalizeNonNull ()
{
    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
Vec3<short>
Vec3<short>::normalized () const
{
    Vec3<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}

template <>
Vec3<short>
Vec3<short>::normalizedExc () const throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    Vec3<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}

template <>
Vec3<short>
Vec3<short>::normalizedNonNull () const
{
    Vec3<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}


// Vec3<int>

template <> 
int
Vec3<int>::length () const
{
    float lenF = Math<float>::sqrt ((float)dot (*this));
    int lenI = (int) (lenF + 0.5f);
    return lenI;
}

template <>
const Vec3<int> &
Vec3<int>::normalize ()
{
    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
const Vec3<int> &
Vec3<int>::normalizeExc () throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
const Vec3<int> &
Vec3<int>::normalizeNonNull ()
{
    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
Vec3<int>
Vec3<int>::normalized () const
{
    Vec3<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}

template <>
Vec3<int>
Vec3<int>::normalizedExc () const throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    Vec3<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}

template <>
Vec3<int>
Vec3<int>::normalizedNonNull () const
{
    Vec3<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}


// Vec4<short>

template <> 
short
Vec4<short>::length () const
{
    float lenF = Math<float>::sqrt ((float)dot (*this));
    short lenS = (short) (lenF + 0.5f);
    return lenS;
}

template <>
const Vec4<short> &
Vec4<short>::normalize ()
{
    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
const Vec4<short> &
Vec4<short>::normalizeExc () throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0) && (w == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
const Vec4<short> &
Vec4<short>::normalizeNonNull ()
{
    normalizeOrThrow<short>(*this);
    return *this;
}

template <>
Vec4<short>
Vec4<short>::normalized () const
{
    Vec4<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}

template <>
Vec4<short>
Vec4<short>::normalizedExc () const throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0) && (w == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    Vec4<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}

template <>
Vec4<short>
Vec4<short>::normalizedNonNull () const
{
    Vec4<short> v(*this);
    normalizeOrThrow<short>(v);
    return v;
}


// Vec4<int>

template <> 
int
Vec4<int>::length () const
{
    float lenF = Math<float>::sqrt ((float)dot (*this));
    int lenI = (int) (lenF + 0.5f);
    return lenI;
}

template <>
const Vec4<int> &
Vec4<int>::normalize ()
{
    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
const Vec4<int> &
Vec4<int>::normalizeExc () throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0) && (w == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
const Vec4<int> &
Vec4<int>::normalizeNonNull ()
{
    normalizeOrThrow<int>(*this);
    return *this;
}

template <>
Vec4<int>
Vec4<int>::normalized () const
{
    Vec4<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}

template <>
Vec4<int>
Vec4<int>::normalizedExc () const throw (Iex::MathExc)
{
    if ((x == 0) && (y == 0) && (z == 0) && (w == 0))
        throw NullVecExc ("Cannot normalize null vector.");

    Vec4<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}

template <>
Vec4<int>
Vec4<int>::normalizedNonNull () const
{
    Vec4<int> v(*this);
    normalizeOrThrow<int>(v);
    return v;
}

} // namespace Imath
