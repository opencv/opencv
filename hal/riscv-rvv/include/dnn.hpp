// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_RVV_HAL_DNN_HPP
#define OPENCV_RVV_HAL_DNN_HPP

namespace cv { namespace rvv_hal { namespace dnn {

#if CV_HAL_RVV_1P0_ENABLED

// Blocked NCDHWc pooling / depthwise convolution, CV_32F. The geometry crosses the boundary as
// a flat, stable C argument list (no dnn types): C0 channel block; insize/outsize = the [3]
// input/output spatial dims in a fixed Z,Y,X frame (unused leading dims = 1); strides[3]; pads[6]
// (begin[0..2]+end[3..5]); inner[6] = the padding-free interior bounds; coordtab[ksize*3]
// = per-tap (dz,dy,dx); ofstab[ksize] = per-tap flat input offset for the interior.
// Each hook computes only the block-plane range [task_start, task_end) (N*C1 planes);
// OpenCV owns the parallel_for_ and never expects the HAL to spawn threads.

/* ############ maxpool3d32f ############ */

int maxpool3d32f(const float* inp_data, float* out_data, int C0,
               const int* insize, const int* outsize, const int* strides,
               const int* pads, const int* inner, const int* coordtab,
               const int* ofstab, int ksize, int task_start, int task_end);

#undef cv_hal_dnn_maxpool3d32f
#define cv_hal_dnn_maxpool3d32f cv::rvv_hal::dnn::maxpool3d32f

/* ############ avgpool3d32f ############ */

int avgpool3d32f(const float* inp_data, float* out_data, int C0,
               const int* insize, const int* outsize, const int* strides,
               const int* pads, const int* inner, const int* coordtab,
               const int* ofstab, int ksize, int count_include_pad,
               int task_start, int task_end);

#undef cv_hal_dnn_avgpool3d32f
#define cv_hal_dnn_avgpool3d32f cv::rvv_hal::dnn::avgpool3d32f

/* ############ depthwise_conv32f ############ */

// Depthwise convolution adds, on top of the geometry above, the fused epilogue out = act(in*W +
// bias, scaled): weights is the repacked C1*ksize*C0 tensor (block b at b*ksize*C0); scale/bias
// are optional per-channel [C] vectors (null => 1/0); residual (optional) is added before the
// activation out = min(s>=0 ? s : s*alpha, maxval), where alpha is prelu_slope[c] (if != null)
// else default_alpha.

int depthwise_conv32f(const float* inp_data, const float* residual_data,
                      float* out_data, const float* weights,
                      const float* scale, const float* bias,
                      int C, int C0, int C1,
                      const int* insize, const int* outsize, const int* strides,
                      const int* pads, const int* inner, const int* coordtab,
                      const int* ofstab, int ksize,
                      float maxval, float default_alpha, const float* prelu_slope,
                      int task_start, int task_end);

#undef cv_hal_dnn_depthwise_conv32f
#define cv_hal_dnn_depthwise_conv32f cv::rvv_hal::dnn::depthwise_conv32f

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::dnn

#endif // OPENCV_RVV_HAL_DNN_HPP
