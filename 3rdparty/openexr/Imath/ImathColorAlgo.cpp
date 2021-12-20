//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

///
/// @file  ImathColorAlgo.cpp
///
/// @brief Implementation of non-template items declared in ImathColorAlgo.h
///

#include "ImathColorAlgo.h"

IMATH_INTERNAL_NAMESPACE_SOURCE_ENTER

Vec3<double>
hsv2rgb_d (const Vec3<double>& hsv) IMATH_NOEXCEPT
{
    double hue = hsv.x;
    double sat = hsv.y;
    double val = hsv.z;

    double x = 0.0, y = 0.0, z = 0.0;

    if (hue == 1)
        hue = 0;
    else
        hue *= 6;

    int i    = int (std::floor (hue));
    double f = hue - i;
    double p = val * (1 - sat);
    double q = val * (1 - (sat * f));
    double t = val * (1 - (sat * (1 - f)));

    switch (i)
    {
    case 0:
        x = val;
        y = t;
        z = p;
        break;
    case 1:
        x = q;
        y = val;
        z = p;
        break;
    case 2:
        x = p;
        y = val;
        z = t;
        break;
    case 3:
        x = p;
        y = q;
        z = val;
        break;
    case 4:
        x = t;
        y = p;
        z = val;
        break;
    case 5:
        x = val;
        y = p;
        z = q;
        break;
    }

    return Vec3<double> (x, y, z);
}

Color4<double>
hsv2rgb_d (const Color4<double>& hsv) IMATH_NOEXCEPT
{
    double hue = hsv.r;
    double sat = hsv.g;
    double val = hsv.b;

    double r = 0.0, g = 0.0, b = 0.0;

    if (hue == 1)
        hue = 0;
    else
        hue *= 6;

    int i    = int (std::floor (hue));
    double f = hue - i;
    double p = val * (1 - sat);
    double q = val * (1 - (sat * f));
    double t = val * (1 - (sat * (1 - f)));

    switch (i)
    {
    case 0:
        r = val;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = val;
        b = p;
        break;
    case 2:
        r = p;
        g = val;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = val;
        break;
    case 4:
        r = t;
        g = p;
        b = val;
        break;
    case 5:
        r = val;
        g = p;
        b = q;
        break;
    }

    return Color4<double> (r, g, b, hsv.a);
}

Vec3<double>
rgb2hsv_d (const Vec3<double>& c) IMATH_NOEXCEPT
{
    const double& x = c.x;
    const double& y = c.y;
    const double& z = c.z;

    double max   = (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    double min   = (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    double range = max - min;
    double val   = max;
    double sat   = 0;
    double hue   = 0;

    if (max != 0)
        sat = range / max;

    if (sat != 0)
    {
        double h;

        if (x == max)
            h = (y - z) / range;
        else if (y == max)
            h = 2 + (z - x) / range;
        else
            h = 4 + (x - y) / range;

        hue = h / 6.;

        if (hue < 0.)
            hue += 1.0;
    }
    return Vec3<double> (hue, sat, val);
}

Color4<double>
rgb2hsv_d (const Color4<double>& c) IMATH_NOEXCEPT
{
    const double& r = c.r;
    const double& g = c.g;
    const double& b = c.b;

    double max   = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
    double min   = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
    double range = max - min;
    double val   = max;
    double sat   = 0;
    double hue   = 0;

    if (max != 0)
        sat = range / max;

    if (sat != 0)
    {
        double h;

        if (r == max)
            h = (g - b) / range;
        else if (g == max)
            h = 2 + (b - r) / range;
        else
            h = 4 + (r - g) / range;

        hue = h / 6.;

        if (hue < 0.)
            hue += 1.0;
    }
    return Color4<double> (hue, sat, val, c.a);
}

IMATH_INTERNAL_NAMESPACE_SOURCE_EXIT
