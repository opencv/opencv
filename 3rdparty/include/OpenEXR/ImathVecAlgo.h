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



#ifndef INCLUDED_IMATHVECALGO_H
#define INCLUDED_IMATHVECALGO_H

//-------------------------------------------------------------------------
//
//      This file contains algorithms applied to or in conjunction
//      with points (Imath::Vec2 and Imath::Vec3).
//	The assumption made is that these functions are called much
//	less often than the basic point functions or these functions
//	require more support classes.
//
//-------------------------------------------------------------------------

#include "ImathVec.h"
#include "ImathLimits.h"

namespace Imath {


//--------------------------------------------------------------
// Find the projection of vector t onto vector s (Vec2 and Vec3)
//--------------------------------------------------------------

template <class Vec> Vec	project (const Vec &s, const Vec &t);


//----------------------------------------------
// Find a vector which is perpendicular to s and
// in the same plane as s and t (Vec2 and Vec3)
//----------------------------------------------

template <class Vec> Vec	orthogonal (const Vec &s, const Vec &t);


//-----------------------------------------------
// Find the direction of a ray s after reflection
// off a plane with normal t (Vec2 and Vec3)
//-----------------------------------------------

template <class Vec> Vec	reflect (const Vec &s, const Vec &t);


//----------------------------------------------------------------------
// Find the vertex of triangle (v0, v1, v2), which is closest to point p
// (Vec2 and Vec3).
//----------------------------------------------------------------------

template <class Vec> Vec	closestVertex (const Vec &v0,
					       const Vec &v1,
					       const Vec &v2, 
					       const Vec &p);

//---------------
// Implementation
//---------------

template <class Vec>
Vec
project (const Vec &s, const Vec &t)
{
    Vec sNormalized = s.normalized();
    return sNormalized * (sNormalized ^ t);
}

template <class Vec>
Vec
orthogonal (const Vec &s, const Vec &t)
{
    return t - project (s, t);
}

template <class Vec>
Vec
reflect (const Vec &s, const Vec &t)
{
    return s - typename Vec::BaseType(2) * (s - project(t, s));
}

template <class Vec>
Vec
closestVertex(const Vec &v0,
              const Vec &v1,
              const Vec &v2, 
              const Vec &p)
{
    Vec nearest = v0;
    typename Vec::BaseType neardot = (v0 - p).length2();
    typename Vec::BaseType tmp     = (v1 - p).length2();

    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v1;
    }

    tmp = (v2 - p).length2();

    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v2;
    }

    return nearest;
}


} // namespace Imath

#endif
