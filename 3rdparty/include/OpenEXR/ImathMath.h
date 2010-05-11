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



#ifndef INCLUDED_IMATHMATH_H
#define INCLUDED_IMATHMATH_H

//----------------------------------------------------------------------------
//
//	ImathMath.h
//
//	This file contains template functions which call the double-
//	precision math functions defined in math.h (sin(), sqrt(),
//	exp() etc.), with specializations that call the faster
//	single-precision versions (sinf(), sqrtf(), expf() etc.)
//	when appropriate.
//
//	Example:
//
//	    double x = Math<double>::sqrt (3);	// calls ::sqrt(double);
//	    float  y = Math<float>::sqrt (3);	// calls ::sqrtf(float);
//
//	When would I want to use this?
//
//	You may be writing a template which needs to call some function
//	defined in math.h, for example to extract a square root, but you
//	don't know whether to call the single- or the double-precision
//	version of this function (sqrt() or sqrtf()):
//
//	    template <class T>
//	    T
//	    glorp (T x)
//	    {
//		return sqrt (x + 1);		// should call ::sqrtf(float)
//	    }					// if x is a float, but we
//						// don't know if it is
//
//	Using the templates in this file, you can make sure that
//	the appropriate version of the math function is called:
//
//	    template <class T>
//	    T
//	    glorp (T x, T y)
//	    {
//		return Math<T>::sqrt (x + 1);	// calls ::sqrtf(float) if x
//	    }					// is a float, ::sqrt(double)
//	    					// otherwise
//
//----------------------------------------------------------------------------

#include "ImathPlatform.h"
#include <math.h>

namespace Imath {


template <class T>
struct Math
{
   static T	acos  (T x)		{return ::acos (double(x));}	
   static T	asin  (T x)		{return ::asin (double(x));}
   static T	atan  (T x)		{return ::atan (double(x));}
   static T	atan2 (T x, T y)	{return ::atan2 (double(x), double(y));}
   static T	cos   (T x)		{return ::cos (double(x));}
   static T	sin   (T x)		{return ::sin (double(x));}
   static T	tan   (T x)		{return ::tan (double(x));}
   static T	cosh  (T x)		{return ::cosh (double(x));}
   static T	sinh  (T x)		{return ::sinh (double(x));}
   static T	tanh  (T x)		{return ::tanh (double(x));}
   static T	exp   (T x)		{return ::exp (double(x));}
   static T	log   (T x)		{return ::log (double(x));}
   static T	log10 (T x)		{return ::log10 (double(x));}
   static T	modf  (T x, T *iptr)
   {
        double ival;
        T rval( ::modf (double(x),&ival));
	*iptr = ival;
	return rval;
   }
   static T	pow   (T x, T y)	{return ::pow (double(x), double(y));}
   static T	sqrt  (T x)		{return ::sqrt (double(x));}
   static T	ceil  (T x)		{return ::ceil (double(x));}
   static T	fabs  (T x)		{return ::fabs (double(x));}
   static T	floor (T x)		{return ::floor (double(x));}
   static T	fmod  (T x, T y)	{return ::fmod (double(x), double(y));}
   static T	hypot (T x, T y)	{return ::hypot (double(x), double(y));}
};


template <>
struct Math<float>
{
   static float	acos  (float x)			{return ::acosf (x);}	
   static float	asin  (float x)			{return ::asinf (x);}
   static float	atan  (float x)			{return ::atanf (x);}
   static float	atan2 (float x, float y)	{return ::atan2f (x, y);}
   static float	cos   (float x)			{return ::cosf (x);}
   static float	sin   (float x)			{return ::sinf (x);}
   static float	tan   (float x)			{return ::tanf (x);}
   static float	cosh  (float x)			{return ::coshf (x);}
   static float	sinh  (float x)			{return ::sinhf (x);}
   static float	tanh  (float x)			{return ::tanhf (x);}
   static float	exp   (float x)			{return ::expf (x);}
   static float	log   (float x)			{return ::logf (x);}
   static float	log10 (float x)			{return ::log10f (x);}
   static float	modf  (float x, float *y)	{return ::modff (x, y);}
   static float	pow   (float x, float y)	{return ::powf (x, y);}
   static float	sqrt  (float x)			{return ::sqrtf (x);}
   static float	ceil  (float x)			{return ::ceilf (x);}
   static float	fabs  (float x)			{return ::fabsf (x);}
   static float	floor (float x)			{return ::floorf (x);}
   static float	fmod  (float x, float y)	{return ::fmodf (x, y);}
#if !defined(_MSC_VER)
   static float	hypot (float x, float y)	{return ::hypotf (x, y);}
#else
   static float hypot (float x, float y)	{return ::sqrtf(x*x + y*y);}
#endif
};


//--------------------------------------------------------------------------
// Compare two numbers and test if they are "approximately equal":
//
// equalWithAbsError (x1, x2, e)
//
//	Returns true if x1 is the same as x2 with an absolute error of
//	no more than e,
//	
//	abs (x1 - x2) <= e
//
// equalWithRelError (x1, x2, e)
//
//	Returns true if x1 is the same as x2 with an relative error of
//	no more than e,
//	
//	abs (x1 - x2) <= e * x1
//
//--------------------------------------------------------------------------

template <class T>
inline bool
equalWithAbsError (T x1, T x2, T e)
{
    return ((x1 > x2)? x1 - x2: x2 - x1) <= e;
}


template <class T>
inline bool
equalWithRelError (T x1, T x2, T e)
{
    return ((x1 > x2)? x1 - x2: x2 - x1) <= e * ((x1 > 0)? x1: -x1);
}



} // namespace Imath

#endif
