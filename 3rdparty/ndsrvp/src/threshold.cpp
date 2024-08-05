// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

template <typename type, typename vtype>
struct opThreshBinary_t {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval)
    {
        return (vtype)__nds__bpick((long)maxval, (long)0, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval)
    {
        return src > thresh ? maxval : 0;
    }
};

template <typename type, typename vtype>
struct opThreshBinaryInv_t {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval)
    {
        return (vtype)__nds__bpick((long)0, (long)maxval, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval)
    {
        return src > thresh ? 0 : maxval;
    }
};

template <typename type, typename vtype>
struct opThreshTrunc_t {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval)
    {
        (void)maxval;
        return (vtype)__nds__bpick((long)thresh, (long)src, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval)
    {
        (void)maxval;
        return src > thresh ? thresh : src;
    }
};

template <typename type, typename vtype>
struct opThreshToZero_t {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval)
    {
        (void)maxval;
        return (vtype)__nds__bpick((long)src, (long)0, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval)
    {
        (void)maxval;
        return src > thresh ? src : 0;
    }
};

template <typename type, typename vtype>
struct opThreshToZeroInv_t {
    inline vtype vector(const vtype& src, const vtype& thresh, const vtype& maxval)
    {
        (void)maxval;
        return (vtype)__nds__bpick((long)0, (long)src, (long)(src > thresh));
    }
    inline type scalar(const type& src, const type& thresh, const type& maxval)
    {
        (void)maxval;
        return src > thresh ? 0 : src;
    }
};

template <typename type, typename vtype, int nlane,
    template <typename ttype, typename vttype> typename opThresh_t>
static inline void threshold_op(const uchar* src, size_t src_step,
    uchar* dst, size_t dst_step,
    int width, int height, int cn,
    double thresh_d, double maxval_d)
{
    int i, j;
    width *= cn;

    type* src_data = (type*)src;
    type* dst_data = (type*)dst;
    src_step /= sizeof(type);
    dst_step /= sizeof(type);

    type thresh = saturate_cast<type>(thresh_d);
    type maxval = saturate_cast<type>(maxval_d);
    vtype vthresh;
    vtype vmaxval;
    for (i = 0; i < nlane; i++) {
        vthresh[i] = thresh;
        vmaxval[i] = maxval;
    }

    opThresh_t<type, vtype> opThresh;

    for (i = 0; i < height; i++, src_data += src_step, dst_data += dst_step) {
        for (j = 0; j <= width - nlane; j += nlane) {
            *(vtype*)(dst_data + j) = opThresh.vector(*(vtype*)(src_data + j), vthresh, vmaxval);
        }
        for (; j < width; j++) {
            dst_data[j] = opThresh.scalar(src_data[j], thresh, maxval);
        }
    }

    return;
}

typedef void (*ThreshFunc)(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step,
    int width, int height, int cn,
    double thresh, double maxval);

int threshold(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step,
    int width, int height, int depth, int cn,
    double thresh, double maxValue, int thresholdType)
{
    static ThreshFunc thfuncs[4][5] =
    {
        {
            threshold_op<uchar, uint8x8_t, 8, opThreshBinary_t>,
            threshold_op<uchar, uint8x8_t, 8, opThreshBinaryInv_t>,
            threshold_op<uchar, uint8x8_t, 8, opThreshTrunc_t>, 
            threshold_op<uchar, uint8x8_t, 8, opThreshToZero_t>,
            threshold_op<uchar, uint8x8_t, 8, opThreshToZeroInv_t> },
        {
            threshold_op<char, int8x8_t, 8, opThreshBinary_t>,
            threshold_op<char, int8x8_t, 8, opThreshBinaryInv_t>,
            threshold_op<char, int8x8_t, 8, opThreshTrunc_t>, 
            threshold_op<char, int8x8_t, 8, opThreshToZero_t>,
            threshold_op<char, int8x8_t, 8, opThreshToZeroInv_t> },
        {
            threshold_op<ushort, uint16x4_t, 4, opThreshBinary_t>,
            threshold_op<ushort, uint16x4_t, 4, opThreshBinaryInv_t>,
            threshold_op<ushort, uint16x4_t, 4, opThreshTrunc_t>,
            threshold_op<ushort, uint16x4_t, 4, opThreshToZero_t>,
            threshold_op<ushort, uint16x4_t, 4, opThreshToZeroInv_t> },
        {
            threshold_op<short, int16x4_t, 4, opThreshBinary_t>,
            threshold_op<short, int16x4_t, 4, opThreshBinaryInv_t>,
            threshold_op<short, int16x4_t, 4, opThreshTrunc_t>,
            threshold_op<short, int16x4_t, 4, opThreshToZero_t>,
            threshold_op<short, int16x4_t, 4, opThreshToZeroInv_t> }
    };

    if(depth < 0 || depth > 3 || thresholdType < 0 || thresholdType > 4 || (width < 256 && height < 256))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    thfuncs[depth][thresholdType](src_data, src_step, dst_data, dst_step, width, height, cn, thresh, maxValue);
    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
