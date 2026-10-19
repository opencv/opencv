// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "../conv2_common.hpp"
#include "conv2_deconv.simd.hpp"
#include "layers/cpu_kernels/conv2_deconv.simd_declarations.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

DeconvFunc getDeconvFunc(int depth)
{
    CV_CPU_DISPATCH(getDeconvFunc_, (depth), CV_CPU_DISPATCH_MODES_ALL);
}

CV__DNN_INLINE_NS_END
}}
