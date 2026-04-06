// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>
#include "activation_kernels.simd.hpp"
#include "layers/cpu_kernels/activation_kernels.simd_declarations.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

ActivationFunc getActivationFunc(int type)
{
    CV_CPU_DISPATCH(getActivationFunc_, (type), CV_CPU_DISPATCH_MODES_ALL);
}

CV__DNN_INLINE_NS_END
}}
