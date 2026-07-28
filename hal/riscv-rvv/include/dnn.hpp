// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_RVV_HAL_DNN_HPP
#define OPENCV_RVV_HAL_DNN_HPP

namespace cv { namespace rvv_hal { namespace dnn {

#if CV_HAL_RVV_1P0_ENABLED

// Blocked NCHWc pooling, CV_32F. The geometry crosses the boundary as a flat, stable
// C argument list (no dnn types): C0 channel block; insize/outsize = the [3] input/output
// spatial dims in a fixed Z,Y,X frame (unused leading dims = 1); strides[3]; pads[6]
// (begin[0..2]+end[3..5]); inner[6] = the padding-free interior bounds; coordtab[ksize*3]
// = per-tap (dz,dy,dx); ofstab[ksize] = per-tap flat input offset for the interior.
// Each hook computes only the block-plane range [task_start, task_end) (N*C1 planes);
// OpenCV owns the parallel_for_ and never expects the HAL to spawn threads.

/* ############ maxpool32f ############ */

int maxpool32f(const float* inp_data, float* out_data, int C0,
               const int* insize, const int* outsize, const int* strides,
               const int* pads, const int* inner, const int* coordtab,
               const int* ofstab, int ksize, int task_start, int task_end);

#undef cv_hal_dnn_maxpool32f
#define cv_hal_dnn_maxpool32f cv::rvv_hal::dnn::maxpool32f

/* ############ avgpool32f ############ */

int avgpool32f(const float* inp_data, float* out_data, int C0,
               const int* insize, const int* outsize, const int* strides,
               const int* pads, const int* inner, const int* coordtab,
               const int* ofstab, int ksize, int count_include_pad,
               int task_start, int task_end);

#undef cv_hal_dnn_avgpool32f
#define cv_hal_dnn_avgpool32f cv::rvv_hal::dnn::avgpool32f

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::dnn

#endif // OPENCV_RVV_HAL_DNN_HPP
