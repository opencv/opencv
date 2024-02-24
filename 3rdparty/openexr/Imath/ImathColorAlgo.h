//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Color conversion functions and general color algorithms
//

#ifndef INCLUDED_IMATHCOLORALGO_H
#define INCLUDED_IMATHCOLORALGO_H

#include "ImathNamespace.h"
#include "ImathExport.h"

#include "ImathColor.h"
#include "ImathMath.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Non-templated helper routines for color conversion.
// These routines eliminate type warnings under g++.
//

///
/// Convert 3-channel hsv to rgb. Non-templated helper routine.
IMATH_EXPORT Vec3<double> hsv2rgb_d (const Vec3<double>& hsv) IMATH_NOEXCEPT;

///
/// Convert 4-channel hsv to rgb (with alpha). Non-templated helper routine.
IMATH_EXPORT Color4<double> hsv2rgb_d (const Color4<double>& hsv) IMATH_NOEXCEPT;

///
/// Convert 3-channel rgb to hsv. Non-templated helper routine.
IMATH_EXPORT Vec3<double> rgb2hsv_d (const Vec3<double>& rgb) IMATH_NOEXCEPT;

///
/// Convert 4-channel rgb to hsv. Non-templated helper routine.
IMATH_EXPORT Color4<double> rgb2hsv_d (const Color4<double>& rgb) IMATH_NOEXCEPT;

///
/// Convert 3-channel hsv to rgb.
///

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Vec3<T>
hsv2rgb (const Vec3<T>& hsv) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        Vec3<double> v = Vec3<double> (hsv.x / double (std::numeric_limits<T>::max()),
                                       hsv.y / double (std::numeric_limits<T>::max()),
                                       hsv.z / double (std::numeric_limits<T>::max()));
        Vec3<double> c = hsv2rgb_d (v);
        return Vec3<T> ((T) (c.x * std::numeric_limits<T>::max()),
                        (T) (c.y * std::numeric_limits<T>::max()),
                        (T) (c.z * std::numeric_limits<T>::max()));
    }
    else
    {
        Vec3<double> v = Vec3<double> (hsv.x, hsv.y, hsv.z);
        Vec3<double> c = hsv2rgb_d (v);
        return Vec3<T> ((T) c.x, (T) c.y, (T) c.z);
    }
}

///
/// Convert 4-channel hsv to rgb (with alpha).
///

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Color4<T>
hsv2rgb (const Color4<T>& hsv) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        Color4<double> v = Color4<double> (hsv.r / float (std::numeric_limits<T>::max()),
                                           hsv.g / float (std::numeric_limits<T>::max()),
                                           hsv.b / float (std::numeric_limits<T>::max()),
                                           hsv.a / float (std::numeric_limits<T>::max()));
        Color4<double> c = hsv2rgb_d (v);
        return Color4<T> ((T) (c.r * std::numeric_limits<T>::max()),
                          (T) (c.g * std::numeric_limits<T>::max()),
                          (T) (c.b * std::numeric_limits<T>::max()),
                          (T) (c.a * std::numeric_limits<T>::max()));
    }
    else
    {
        Color4<double> v = Color4<double> (hsv.r, hsv.g, hsv.b, hsv.a);
        Color4<double> c = hsv2rgb_d (v);
        return Color4<T> ((T) c.r, (T) c.g, (T) c.b, (T) c.a);
    }
}

///
/// Convert 3-channel rgb to hsv.
///

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Vec3<T>
rgb2hsv (const Vec3<T>& rgb) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        Vec3<double> v = Vec3<double> (rgb.x / double (std::numeric_limits<T>::max()),
                                       rgb.y / double (std::numeric_limits<T>::max()),
                                       rgb.z / double (std::numeric_limits<T>::max()));
        Vec3<double> c = rgb2hsv_d (v);
        return Vec3<T> ((T) (c.x * std::numeric_limits<T>::max()),
                        (T) (c.y * std::numeric_limits<T>::max()),
                        (T) (c.z * std::numeric_limits<T>::max()));
    }
    else
    {
        Vec3<double> v = Vec3<double> (rgb.x, rgb.y, rgb.z);
        Vec3<double> c = rgb2hsv_d (v);
        return Vec3<T> ((T) c.x, (T) c.y, (T) c.z);
    }
}

///
/// Convert 4-channel rgb to hsv (with alpha).
///

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Color4<T>
rgb2hsv (const Color4<T>& rgb) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        Color4<double> v = Color4<double> (rgb.r / float (std::numeric_limits<T>::max()),
                                           rgb.g / float (std::numeric_limits<T>::max()),
                                           rgb.b / float (std::numeric_limits<T>::max()),
                                           rgb.a / float (std::numeric_limits<T>::max()));
        Color4<double> c = rgb2hsv_d (v);
        return Color4<T> ((T) (c.r * std::numeric_limits<T>::max()),
                          (T) (c.g * std::numeric_limits<T>::max()),
                          (T) (c.b * std::numeric_limits<T>::max()),
                          (T) (c.a * std::numeric_limits<T>::max()));
    }
    else
    {
        Color4<double> v = Color4<double> (rgb.r, rgb.g, rgb.b, rgb.a);
        Color4<double> c = rgb2hsv_d (v);
        return Color4<T> ((T) c.r, (T) c.g, (T) c.b, (T) c.a);
    }
}

///
/// Convert 3-channel rgb to PackedColor
///

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 PackedColor
rgb2packed (const Vec3<T>& c) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        float x = c.x / float (std::numeric_limits<T>::max());
        float y = c.y / float (std::numeric_limits<T>::max());
        float z = c.z / float (std::numeric_limits<T>::max());
        return rgb2packed (V3f (x, y, z));
    }
    else
    {
        // clang-format off
	return (  (PackedColor) (c.x * 255)		|
		(((PackedColor) (c.y * 255)) << 8)	|
		(((PackedColor) (c.z * 255)) << 16)	| 0xFF000000 );
        // clang-format on
    }
}

///
/// Convert 4-channel rgb to PackedColor (with alpha)
///

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 PackedColor
rgb2packed (const Color4<T>& c) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        float r = c.r / float (std::numeric_limits<T>::max());
        float g = c.g / float (std::numeric_limits<T>::max());
        float b = c.b / float (std::numeric_limits<T>::max());
        float a = c.a / float (std::numeric_limits<T>::max());
        return rgb2packed (C4f (r, g, b, a));
    }
    else
    {
        // clang-format off
	return (  (PackedColor) (c.r * 255)		|
		(((PackedColor) (c.g * 255)) << 8)	|
		(((PackedColor) (c.b * 255)) << 16)	|
		(((PackedColor) (c.a * 255)) << 24));
        // clang-format on
    }
}

///
/// Convert PackedColor to 3-channel rgb. Return the result in the
/// `out` parameter.
///

template <class T>
IMATH_HOSTDEVICE void
packed2rgb (PackedColor packed, Vec3<T>& out) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        T f   = std::numeric_limits<T>::max() / ((PackedColor) 0xFF);
        out.x = (packed & 0xFF) * f;
        out.y = ((packed & 0xFF00) >> 8) * f;
        out.z = ((packed & 0xFF0000) >> 16) * f;
    }
    else
    {
        T f   = T (1) / T (255);
        out.x = (packed & 0xFF) * f;
        out.y = ((packed & 0xFF00) >> 8) * f;
        out.z = ((packed & 0xFF0000) >> 16) * f;
    }
}

///
/// Convert PackedColor to 4-channel rgb (with alpha). Return the
/// result in the `out` parameter.
///

template <class T>
IMATH_HOSTDEVICE void
packed2rgb (PackedColor packed, Color4<T>& out) IMATH_NOEXCEPT
{
    if (std::numeric_limits<T>::is_integer)
    {
        T f   = std::numeric_limits<T>::max() / ((PackedColor) 0xFF);
        out.r = (packed & 0xFF) * f;
        out.g = ((packed & 0xFF00) >> 8) * f;
        out.b = ((packed & 0xFF0000) >> 16) * f;
        out.a = ((packed & 0xFF000000) >> 24) * f;
    }
    else
    {
        T f   = T (1) / T (255);
        out.r = (packed & 0xFF) * f;
        out.g = ((packed & 0xFF00) >> 8) * f;
        out.b = ((packed & 0xFF0000) >> 16) * f;
        out.a = ((packed & 0xFF000000) >> 24) * f;
    }
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHCOLORALGO_H
