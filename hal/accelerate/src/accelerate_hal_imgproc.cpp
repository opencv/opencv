// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "accelerate_hal_imgproc.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/utility.hpp"

#include <algorithm>
#include <atomic>
#include <Accelerate/Accelerate.h>

int accelerate_hal_sepFilter_stateless(const uchar* src_data, size_t src_step, int src_type,
                                       uchar* dst_data, size_t dst_step, int dst_type,
                                       int width, int height, int full_width, int full_height, int offset_x, int offset_y,
                                       const uchar* kernelx_data, int kernelx_len,
                                       const uchar* kernely_data, int kernely_len,
                                       int kernel_type, int anchor_x, int anchor_y, double delta, int borderType)
{
    // vImageSepConvolve_Planar16U fails Imgproc_Sobel.accuracy tests
    // vImageSepConvolve_ARGB8888 fails SepFilter2D.Mat_BitExact test
    const bool supported_types = (src_type == CV_8U && (dst_type == CV_8U || dst_type == CV_16U)) ||
                                 (src_type == CV_16F && dst_type == CV_16F) ||
                                 (src_type == CV_32F && dst_type == CV_32F);
    if (!supported_types || kernel_type != CV_32F || (borderType != cv::BORDER_CONSTANT && borderType != cv::BORDER_REPLICATE) ||
        anchor_x != -1 || anchor_y != -1) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    uchar* _dst_data = dst_data;
    size_t _dst_step = dst_step;
    const int elem_size = CV_ELEM_SIZE(src_type);
    cv::AutoBuffer<uchar> dst;

    // In-place processing is not allowed
    if (src_data == dst_data) {
        dst.allocate(width * height * elem_size);
        _dst_step = width * elem_size;
        _dst_data = dst.data();
    }

    const vImage_Buffer src_buffer{
        const_cast<void*>(static_cast<const void *>(src_data - offset_x * elem_size - src_step * offset_y)),
        static_cast<vImagePixelCount>(full_height),
        static_cast<vImagePixelCount>(full_width),
        src_step,
    };
    const vImage_Flags flags = (borderType == cv::BORDER_CONSTANT ? kvImageBackgroundColorFill : kvImageEdgeExtend) | kvImageDoNotTile;

    // Filters one stripe of dst rows on the calling thread; y is the stripe's first row in the source ROI
    auto convolve = [&](const vImage_Buffer* dst_buffer, vImagePixelCount y) -> vImage_Error {
        if (src_type == CV_8U && dst_type == CV_8U) {
            return vImageSepConvolve_Planar8(&src_buffer, dst_buffer, nullptr,
                offset_x, y,
                reinterpret_cast<const float*>(kernelx_data), kernelx_len,
                reinterpret_cast<const float*>(kernely_data), kernely_len,
                delta, Pixel_F{}, flags);
        } else if (src_type == CV_16F && dst_type == CV_16F) {
#if ((defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && defined(__MAC_13_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_13_0) || \
     (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && defined(__IPHONE_16_0) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_16_0) || \
     (defined(__WATCH_OS_VERSION_MAX_ALLOWED) && defined(__WATCHOS_9_0) && __WATCH_OS_VERSION_MAX_ALLOWED >= __WATCHOS_9_0) || \
     (defined(__TV_OS_VERSION_MAX_ALLOWED) && defined(__TVOS_16_0) && __TV_OS_VERSION_MAX_ALLOWED >= __TVOS_16_0))
            if (__builtin_available(macOS 13.0, iOS 16.0, watchOS 9.0, tvOS 16.0, *)) {
                return vImageSepConvolve_Planar16F(&src_buffer, dst_buffer, nullptr,
                    offset_x, y,
                    reinterpret_cast<const float*>(kernelx_data), kernelx_len,
                    reinterpret_cast<const float*>(kernely_data), kernely_len,
                    delta, Pixel_F{}, flags);
            }
#endif
        } else if (src_type == CV_32F && dst_type == CV_32F) {
            return vImageSepConvolve_PlanarF(&src_buffer, dst_buffer, nullptr,
                offset_x, y,
                reinterpret_cast<const float*>(kernelx_data), kernelx_len,
                reinterpret_cast<const float*>(kernely_data), kernely_len,
                delta, Pixel_F{}, flags);
        } else if (src_type == CV_8U && dst_type == CV_16U) {
            return vImageSepConvolve_Planar8to16U(&src_buffer, dst_buffer, nullptr,
                offset_x, y,
                reinterpret_cast<const float*>(kernelx_data), kernelx_len,
                reinterpret_cast<const float*>(kernely_data), kernely_len,
                1.f, delta, Pixel_8{}, flags);
        }
        return kvImageUnsupportedConversion;
    };

    // Split dst into stripes of rows (about 64K pixels each). cv::parallel_for_ picks how many threads
    // run them, so cv::setNumThreads() and nested parallel regions work as in the rest of OpenCV.
    std::atomic<bool> ok(true);
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& rows) {
        const vImage_Buffer dst_stripe{
            _dst_data + rows.start * _dst_step,
            static_cast<vImagePixelCount>(rows.end - rows.start),
            static_cast<vImagePixelCount>(width),
            _dst_step,
        };
        if (convolve(&dst_stripe, offset_y + rows.start) != kvImageNoError) {
            ok = false;
        }
    }, static_cast<double>(width) * height / (1 << 16));
    if (!ok) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if (src_data == dst_data) {
        for (int i = 0; i < height; i++) {
            std::copy(_dst_data + i * _dst_step, _dst_data + (i + 1) * _dst_step, dst_data + i * dst_step);
        }
    }

    return CV_HAL_ERROR_OK;
}
