// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "accelerate_hal_imgproc.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/utility.hpp"

#include <algorithm>
#include <Accelerate/Accelerate.h>


#if ((defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_11_0) || \
     (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0) || \
     (defined(__WATCH_OS_VERSION_MAX_ALLOWED) && __WATCH_OS_VERSION_MAX_ALLOWED >= __WATCHOS_7_0) || \
     (defined(__TV_OS_VERSION_MAX_ALLOWED) && __TV_OS_VERSION_MAX_ALLOWED >= __TVOS_14_0))

int accelerate_hal_sepFilter_stateless(const uchar* src_data, size_t src_step, int src_type,
                                       uchar* dst_data, size_t dst_step, int dst_type,
                                       int width, int height, int full_width, int full_height, int offset_x, int offset_y,
                                       const uchar* kernelx_data, int kernelx_len,
                                       const uchar* kernely_data, int kernely_len,
                                       int kernel_type, int anchor_x, int anchor_y, double delta, int borderType)
{
    if (kernel_type != CV_32F || (borderType != cv::BORDER_CONSTANT && borderType != cv::BORDER_REPLICATE) || anchor_x != -1 || anchor_y != -1) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    uchar* _dst_data = dst_data;
    size_t _dst_step = dst_step;
    const int elem_size = CV_ELEM_SIZE(src_type);
    cv::AutoBuffer<uchar> dst;

    // In-place processing is not allowed
    if (src_data == dst_data) {
        dst = cv::AutoBuffer<uchar>(width * height * elem_size);
        _dst_step = width * elem_size;
        _dst_data = dst.data();
    }

    const vImage_Buffer src_buffer{
        const_cast<void*>(static_cast<const void *>(src_data - offset_x * elem_size - src_step * offset_y)),
        static_cast<vImagePixelCount>(full_height),
        static_cast<vImagePixelCount>(full_width),
        src_step,
    };
    const vImage_Buffer dst_buffer{
        const_cast<void*>(static_cast<const void *>(_dst_data)),
        static_cast<vImagePixelCount>(height),
        static_cast<vImagePixelCount>(width),
        _dst_step,
    };
    const vImage_Flags flags = borderType == cv::BORDER_CONSTANT ? kvImageBackgroundColorFill : kvImageEdgeExtend;
    vImage_Error result;
    if (src_type == CV_8U && dst_type == CV_8U) {
        result = vImageSepConvolve_Planar8(&src_buffer, &dst_buffer, nullptr,
            offset_x, offset_y,
            reinterpret_cast<const float*>(kernelx_data), kernelx_len,
            reinterpret_cast<const float*>(kernely_data), kernely_len,
            delta, Pixel_F{}, flags);
    } else if (src_type == CV_16F && dst_type == CV_16F) {
#if ((defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_13_0) || \
     (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_16_0) || \
     (defined(__WATCH_OS_VERSION_MAX_ALLOWED) && __WATCH_OS_VERSION_MAX_ALLOWED >= __WATCHOS_9_0) || \
     (defined(__TV_OS_VERSION_MAX_ALLOWED) && __TV_OS_VERSION_MAX_ALLOWED >= __TVOS_16_0))
        result = vImageSepConvolve_Planar16F(&src_buffer, &dst_buffer, nullptr,
            offset_x, offset_y,
            reinterpret_cast<const float*>(kernelx_data), kernelx_len,
            reinterpret_cast<const float*>(kernely_data), kernely_len,
            delta, Pixel_F{}, flags);
#else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif
    } else if (src_type == CV_32F && dst_type == CV_32F) {
        result = vImageSepConvolve_PlanarF(&src_buffer, &dst_buffer, nullptr,
            offset_x, offset_y,
            reinterpret_cast<const float*>(kernelx_data), kernelx_len,
            reinterpret_cast<const float*>(kernely_data), kernely_len,
            delta, Pixel_F{}, flags);
    } else if (src_type == CV_8U && dst_type == CV_16U) {
        result = vImageSepConvolve_Planar8to16U(&src_buffer, &dst_buffer, nullptr,
            offset_x, offset_y,
            reinterpret_cast<const float*>(kernelx_data), kernelx_len,
            reinterpret_cast<const float*>(kernely_data), kernely_len,
            1.f, delta, Pixel_8{}, flags);
    } else {
        // vImageSepConvolve_Planar16U fails Imgproc_Sobel.accuracy tests
        // vImageSepConvolve_ARGB8888 fails SepFilter2D.Mat_BitExact test
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if (src_data == dst_data) {
        for (int i = 0; i < height; i++) {
            std::copy(_dst_data + i * _dst_step, _dst_data + (i + 1) * _dst_step, dst_data + i * dst_step);
        }
    }

    return result == kvImageNoError ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif
