// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"

namespace cv {

namespace ndsrvp {

template <typename type, typename vtype>
class operators_threshold_t {
public:
    virtual ~operators_threshold_t() {};
    virtual inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval)
    {
        (void)src;
        (void)thresh;
        (void)maxval;
        CV_Error(cv::Error::StsBadArg, "");
        return vtype();
    }
    virtual inline type scalar(const type& src, const type& thresh, const type& maxval)
    {
        (void)src;
        (void)thresh;
        (void)maxval;
        CV_Error(cv::Error::StsBadArg, "");
        return type();
    }
};

template <typename type, typename vtype>
class opThreshBinary : public operators_threshold_t<type, vtype> {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval) override
    {
        return (vtype)__nds__bpick((long)maxval, (long)0, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval) override
    {
        return src > thresh ? maxval : 0;
    }
};

template <typename type, typename vtype>
class opThreshBinaryInv : public operators_threshold_t<type, vtype> {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval) override
    {
        return (vtype)__nds__bpick((long)0, (long)maxval, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval) override
    {
        return src > thresh ? 0 : maxval;
    }
};

template <typename type, typename vtype>
class opThreshTrunc : public operators_threshold_t<type, vtype> {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval) override
    {
        (void)maxval;
        return (vtype)__nds__bpick((long)thresh, (long)src, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval) override
    {
        (void)maxval;
        return src > thresh ? thresh : src;
    }
};

template <typename type, typename vtype>
class opThreshToZero : public operators_threshold_t<type, vtype> {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval) override
    {
        (void)maxval;
        return (vtype)__nds__bpick((long)src, (long)0, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval) override
    {
        (void)maxval;
        return src > thresh ? src : 0;
    }
};

template <typename type, typename vtype>
class opThreshToZeroInv : public operators_threshold_t<type, vtype> {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval) override
    {
        (void)maxval;
        return (vtype)__nds__bpick((long)0, (long)src, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval) override
    {
        (void)maxval;
        return src > thresh ? 0 : src;
    }
};

template <typename type, typename vtype, int nlane>
static void threshold_op(const type* src_data, size_t src_step,
    type* dst_data, size_t dst_step,
    int width, int height, int cn,
    type thresh, type maxval, int thtype)
{
    int i, j;
    width *= cn;
    src_step /= sizeof(type);
    dst_step /= sizeof(type);
    vtype vthresh;
    vtype vmaxval;
    for (i = 0; i < nlane; i++) {
        vthresh[i] = thresh;
        vmaxval[i] = maxval;
    }

    operators_threshold_t<type, vtype>* op;
    switch (thtype) {
    case CV_HAL_THRESH_BINARY:
        op = new opThreshBinary<type, vtype>();
        break;
    case CV_HAL_THRESH_BINARY_INV:
        op = new opThreshBinaryInv<type, vtype>();
        break;
    case CV_HAL_THRESH_TRUNC:
        op = new opThreshTrunc<type, vtype>();
        break;
    case CV_HAL_THRESH_TOZERO:
        op = new opThreshToZero<type, vtype>();
        break;
    case CV_HAL_THRESH_TOZERO_INV:
        op = new opThreshToZeroInv<type, vtype>();
        break;
    default:
        CV_Error(cv::Error::StsBadArg, "");
        return;
    }

    for (i = 0; i < height; i++, src_data += src_step, dst_data += dst_step) {
        for (j = 0; j <= width - nlane; j += nlane) {
            vtype vs = *(vtype*)(src_data + j);
            *(vtype*)(dst_data + j) = op->vector(vs, vthresh, vmaxval);
        }
        for (; j < width; j++) {
            dst_data[j] = op->scalar(src_data[j], thresh, maxval);
        }
    }

    delete op;
    return;
}

int threshold(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step,
    int width, int height, int depth, int cn,
    double thresh, double maxValue, int thresholdType)
{
    if (width <= 255 && height <= 255) // slower at small size
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (depth == CV_8U) {
        threshold_op<uchar, uint8x8_t, 8>((uchar*)src_data, src_step, (uchar*)dst_data, dst_step, width, height, cn, (uchar)thresh, (uchar)maxValue, thresholdType);
        return CV_HAL_ERROR_OK;
    } else if (depth == CV_16S) {
        threshold_op<short, int16x4_t, 4>((short*)src_data, src_step, (short*)dst_data, dst_step, width, height, cn, (short)thresh, (short)maxValue, thresholdType);
        return CV_HAL_ERROR_OK;
    } else if (depth == CV_16U) {
        threshold_op<ushort, uint16x4_t, 4>((ushort*)src_data, src_step, (ushort*)dst_data, dst_step, width, height, cn, (ushort)thresh, (ushort)maxValue, thresholdType);
        return CV_HAL_ERROR_OK;
    } else {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

} // namespace ndsrvp

} // namespace cv
