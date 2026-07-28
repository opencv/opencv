// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_RVV_HAL_DNN_HPP
#define OPENCV_RVV_HAL_DNN_HPP

namespace cv { namespace rvv_hal { namespace dnn {

#if CV_HAL_RVV_1P0_ENABLED

// `state` is an opaque pointer to the DNN engine's cv::dnn::ConvState descriptor.
// It is passed as const void* on purpose: this header is pulled into every module
// through the generated custom_hal.hpp, so it must not depend on internal dnn
// headers. The implementations in src/dnn/ include conv2_common.hpp and cast back.
//
// Each hook computes only the block-plane range [task_start, task_end) of the
// output tensor (blocked NCHWc, N*C1 planes of C0 channels); OpenCV owns the
// parallel_for_ and never expects the HAL to spawn threads.

/* ############ maxpool32f ############ */

int maxpool32f(const float* inp_data, float* out_data,
               const void* state, int task_start, int task_end);

#undef cv_hal_dnn_maxpool32f
#define cv_hal_dnn_maxpool32f cv::rvv_hal::dnn::maxpool32f

/* ############ avgpool32f ############ */

int avgpool32f(const float* inp_data, float* out_data,
               const void* state, int count_include_pad,
               int task_start, int task_end);

#undef cv_hal_dnn_avgpool32f
#define cv_hal_dnn_avgpool32f cv::rvv_hal::dnn::avgpool32f

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::dnn

#endif // OPENCV_RVV_HAL_DNN_HPP
