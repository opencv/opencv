// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "../../net_impl.hpp"
#include "../conv2_common.hpp"
#include "conv2_kernels.simd.hpp"
#include "layers/cpu_kernels/conv2_kernels.simd_declarations.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

ConvFunc getConvFunc(int depth, int C0)
{
    CV_CPU_DISPATCH(getConvFunc_, (depth, C0), CV_CPU_DISPATCH_MODES_ALL);
}

CV__DNN_INLINE_NS_END
}}
