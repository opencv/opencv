// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "stat.simd.hpp"
#include "stat.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv { namespace hal {

int normHamming(const uchar* a, int n)
{
    CV_INSTRUMENT_REGION()

    CV_CPU_DISPATCH(normHamming, (a, n),
        CV_CPU_DISPATCH_MODES_ALL);
}

int normHamming(const uchar* a, const uchar* b, int n)
{
    CV_INSTRUMENT_REGION()

    CV_CPU_DISPATCH(normHamming, (a, b, n),
        CV_CPU_DISPATCH_MODES_ALL);
}

}} //cv::hal
