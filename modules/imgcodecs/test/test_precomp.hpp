// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ts.hpp"
#include "opencv2/imgcodecs.hpp"

namespace cv {

static inline
void PrintTo(const ImreadModes& val, std::ostream* os)
{
    int v = val;
    if (v == IMREAD_UNCHANGED && (v & IMREAD_IGNORE_ORIENTATION) != 0)
    {
        CV_Assert(IMREAD_UNCHANGED == -1);
        *os << "IMREAD_UNCHANGED";
        return;
    }
    if ((v & IMREAD_COLOR) != 0)
    {
        CV_Assert(IMREAD_COLOR == 1);
        v &= ~IMREAD_COLOR;
        *os << "IMREAD_COLOR" << (v == 0 ? "" : " | ");
    }
    else
    {
        CV_Assert(IMREAD_GRAYSCALE == 0);
        *os << "IMREAD_GRAYSCALE" << (v == 0 ? "" : " | ");
    }
    if ((v & IMREAD_ANYDEPTH) != 0)
    {
        v &= ~IMREAD_ANYDEPTH;
        *os << "IMREAD_ANYDEPTH" << (v == 0 ? "" : " | ");
    }
    if ((v & IMREAD_ANYCOLOR) != 0)
    {
        v &= ~IMREAD_ANYCOLOR;
        *os << "IMREAD_ANYCOLOR" << (v == 0 ? "" : " | ");
    }
    if ((v & IMREAD_LOAD_GDAL) != 0)
    {
        v &= ~IMREAD_LOAD_GDAL;
        *os << "IMREAD_LOAD_GDAL" << (v == 0 ? "" : " | ");
    }
    if ((v & IMREAD_IGNORE_ORIENTATION) != 0)
    {
        v &= ~IMREAD_IGNORE_ORIENTATION;
        *os << "IMREAD_IGNORE_ORIENTATION" << (v == 0 ? "" : " | ");
    }
    if ((v & IMREAD_COLOR_RGB) != 0)
    {
        v &= ~IMREAD_COLOR_RGB;
        *os << "IMREAD_COLOR_RGB" << (v == 0 ? "" : " | ");
    }
    switch (v)
    {
        case IMREAD_UNCHANGED: return;
        case IMREAD_GRAYSCALE: return;
        case IMREAD_COLOR: return;
        case IMREAD_ANYDEPTH: return;
        case IMREAD_ANYCOLOR: return;
        case IMREAD_LOAD_GDAL: return;
        case IMREAD_REDUCED_GRAYSCALE_2: // fallthru
        case IMREAD_REDUCED_COLOR_2: *os << "REDUCED_2"; return;
        case IMREAD_REDUCED_GRAYSCALE_4: // fallthru
        case IMREAD_REDUCED_COLOR_4: *os << "REDUCED_4"; return;
        case IMREAD_REDUCED_GRAYSCALE_8: // fallthru
        case IMREAD_REDUCED_COLOR_8: *os << "REDUCED_8"; return;
        case IMREAD_IGNORE_ORIENTATION: return;
        case IMREAD_COLOR_RGB: return;
    } // don't use "default:" to emit compiler warnings
    *os << "IMREAD_UNKNOWN(" << (int)v << ")";
}

} // namespace

#endif
