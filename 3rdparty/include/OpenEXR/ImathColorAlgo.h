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



#ifndef INCLUDED_IMATHCOLORALGO_H
#define INCLUDED_IMATHCOLORALGO_H


#include "ImathColor.h"
#include "ImathMath.h"
#include "ImathLimits.h"

namespace Imath {


//
//	Non-templated helper routines for color conversion.
//	These routines eliminate type warnings under g++.
//

Vec3<double>	hsv2rgb_d(const Vec3<double> &hsv);

Color4<double>	hsv2rgb_d(const Color4<double> &hsv);


Vec3<double>	rgb2hsv_d(const Vec3<double> &rgb);

Color4<double>	rgb2hsv_d(const Color4<double> &rgb);


//
//	Color conversion functions and general color algorithms
//
//	hsv2rgb(), rgb2hsv(), rgb2packed(), packed2rgb()
//	see each funtion definition for details.
//

template<class T> 
Vec3<T>  
hsv2rgb(const Vec3<T> &hsv)
{
    if ( limits<T>::isIntegral() )
    {
	Vec3<double> v = Vec3<double>(hsv.x / double(limits<T>::max()),
				      hsv.y / double(limits<T>::max()),
				      hsv.z / double(limits<T>::max()));
	Vec3<double> c = hsv2rgb_d(v);
	return Vec3<T>((T) (c.x * limits<T>::max()),
		       (T) (c.y * limits<T>::max()),
		       (T) (c.z * limits<T>::max()));
    }
    else
    {
	Vec3<double> v = Vec3<double>(hsv.x, hsv.y, hsv.z);
	Vec3<double> c = hsv2rgb_d(v);
	return Vec3<T>((T) c.x, (T) c.y, (T) c.z);
    }
}


template<class T> 
Color4<T>  
hsv2rgb(const Color4<T> &hsv)
{
    if ( limits<T>::isIntegral() )
    {
	Color4<double> v = Color4<double>(hsv.r / float(limits<T>::max()),
					  hsv.g / float(limits<T>::max()),
					  hsv.b / float(limits<T>::max()),
					  hsv.a / float(limits<T>::max()));
	Color4<double> c = hsv2rgb_d(v);
	return Color4<T>((T) (c.r * limits<T>::max()),
    	    	    	 (T) (c.g * limits<T>::max()),
    	    	    	 (T) (c.b * limits<T>::max()),
			 (T) (c.a * limits<T>::max()));
    }
    else
    {
	Color4<double> v = Color4<double>(hsv.r, hsv.g, hsv.g, hsv.a);
	Color4<double> c = hsv2rgb_d(v);
	return Color4<T>((T) c.r, (T) c.g, (T) c.b, (T) c.a);
    }
}


template<class T> 
Vec3<T>  
rgb2hsv(const Vec3<T> &rgb)
{
    if ( limits<T>::isIntegral() )
    {
	Vec3<double> v = Vec3<double>(rgb.x / double(limits<T>::max()),
				      rgb.y / double(limits<T>::max()),
				      rgb.z / double(limits<T>::max()));
	Vec3<double> c = rgb2hsv_d(v);
	return Vec3<T>((T) (c.x * limits<T>::max()),
		       (T) (c.y * limits<T>::max()),
		       (T) (c.z * limits<T>::max()));
    }
    else
    {
	Vec3<double> v = Vec3<double>(rgb.x, rgb.y, rgb.z);
	Vec3<double> c = rgb2hsv_d(v);
	return Vec3<T>((T) c.x, (T) c.y, (T) c.z);
    }
}


template<class T> 
Color4<T>  
rgb2hsv(const Color4<T> &rgb)
{
    if ( limits<T>::isIntegral() )
    {
	Color4<double> v = Color4<double>(rgb.r / float(limits<T>::max()),
					  rgb.g / float(limits<T>::max()),
					  rgb.b / float(limits<T>::max()),
					  rgb.a / float(limits<T>::max()));
	Color4<double> c = rgb2hsv_d(v);
	return Color4<T>((T) (c.r * limits<T>::max()),
    	    	    	 (T) (c.g * limits<T>::max()),
    	    	    	 (T) (c.b * limits<T>::max()),
			 (T) (c.a * limits<T>::max()));
    }
    else
    {
	Color4<double> v = Color4<double>(rgb.r, rgb.g, rgb.g, rgb.a);
	Color4<double> c = rgb2hsv_d(v);
	return Color4<T>((T) c.r, (T) c.g, (T) c.b, (T) c.a);
    }
}

template <class T>
PackedColor
rgb2packed(const Vec3<T> &c)
{
    if ( limits<T>::isIntegral() )
    {
	float x = c.x / float(limits<T>::max());
	float y = c.y / float(limits<T>::max());
	float z = c.z / float(limits<T>::max());
	return rgb2packed( V3f(x,y,z) );
    }
    else
    {
	return (  (PackedColor) (c.x * 255)		|
		(((PackedColor) (c.y * 255)) << 8)	|
		(((PackedColor) (c.z * 255)) << 16)	| 0xFF000000 );
    }
}

template <class T>
PackedColor
rgb2packed(const Color4<T> &c)
{
    if ( limits<T>::isIntegral() )
    {
	float r = c.r / float(limits<T>::max());
	float g = c.g / float(limits<T>::max());
	float b = c.b / float(limits<T>::max());
	float a = c.a / float(limits<T>::max());
	return rgb2packed( C4f(r,g,b,a) );
    }
    else
    {
	return (  (PackedColor) (c.r * 255)		|
		(((PackedColor) (c.g * 255)) << 8)	|
		(((PackedColor) (c.b * 255)) << 16)	|
		(((PackedColor) (c.a * 255)) << 24));
    }
}

//
//	This guy can't return the result because the template
//	parameter would not be in the function signiture. So instead,
//	its passed in as an argument.
//

template <class T>
void
packed2rgb(PackedColor packed, Vec3<T> &out)
{
    if ( limits<T>::isIntegral() )
    {
	T f = limits<T>::max() / ((PackedColor)0xFF);
	out.x =  (packed &     0xFF) * f;
	out.y = ((packed &   0xFF00) >>  8) * f;
	out.z = ((packed & 0xFF0000) >> 16) * f;
    }
    else
    {
	T f = T(1) / T(255);
	out.x =  (packed &     0xFF) * f;
	out.y = ((packed &   0xFF00) >>  8) * f;
	out.z = ((packed & 0xFF0000) >> 16) * f;
    }
}

template <class T>
void
packed2rgb(PackedColor packed, Color4<T> &out)
{
    if ( limits<T>::isIntegral() )
    {
	T f = limits<T>::max() / ((PackedColor)0xFF);
	out.r =  (packed &       0xFF) * f;
	out.g = ((packed &     0xFF00) >>  8) * f;
	out.b = ((packed &   0xFF0000) >> 16) * f;
	out.a = ((packed & 0xFF000000) >> 24) * f;
    }
    else
    {
	T f = T(1) / T(255);
	out.r =  (packed &       0xFF) * f;
	out.g = ((packed &     0xFF00) >>  8) * f;
	out.b = ((packed &   0xFF0000) >> 16) * f;
	out.a = ((packed & 0xFF000000) >> 24) * f;
    }
}


} // namespace Imath

#endif  
