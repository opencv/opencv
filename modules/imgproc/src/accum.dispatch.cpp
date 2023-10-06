 // This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "accum.simd.hpp"
#include "accum.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

DEF_ACC_INT_FUNCS(8u32f, uchar, float)
DEF_ACC_INT_FUNCS(8u64f, uchar, double)
DEF_ACC_INT_FUNCS(16u32f, ushort, float)
DEF_ACC_INT_FUNCS(16u64f, ushort, double)
DEF_ACC_FLT_FUNCS(32f, float, float)
DEF_ACC_FLT_FUNCS(32f64f, float, double)
DEF_ACC_FLT_FUNCS(64f, double, double)

} //cv::hal
