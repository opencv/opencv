// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_BBOX_UTILS_HPP
#define OPENCV_DNN_SRC_CUDA_BBOX_UTILS_HPP

#include "math.hpp"

#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    struct BoundingBox
    {
        float xmin, ymin, xmax, ymax;
    };

    template <bool NORMALIZED_BBOX>
    __device__ __forceinline__ float compute_bbox_size(BoundingBox bbox)
    {
        float width = bbox.xmax - bbox.xmin;
        float height = bbox.ymax - bbox.ymin;
        if (width < 0 || height < 0)
            return 0.0;

        if (!NORMALIZED_BBOX)
        {
            width += 1;
            height += 1;
        }

        using csl::device::mul_ftz;
        return mul_ftz(width, height);
    }

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA_BBOX_UTILS_HPP */
