// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

namespace reduce_c_neon
{

// Optimized ReduceC support in this backend:
//
// | Input -> output type/channel       | SUM | AVG | MIN | MAX | SUM2 |
// |------------------------------------|:---:|:---:|:---:|:---:|:----:|
// | 8UC1/C3/C4 -> 8UC1/C3/C4          |  -  |  -  |  x  |  x  |  -   |
// | 8UC1/C3/C4 -> 32SC1/C3/C4         |  x  |  x  |  -  |  -  |  x*  |
// | 8UC1/C3/C4 -> 32FC1/C3/C4         |  x  |  x  |  -  |  -  |  x*  |
// | 8UC1/C3/C4 -> 64FC1/C3/C4         |  -  |  -  |  -  |  -  |  x*  |
// | 16UC1/C3/C4 -> 16UC1/C3/C4        |  -  |  -  |  x  |  x  |  -   |
// | 16UC1 -> 32FC1                     |  x  |  x  |  -  |  -  |  x*  |
// | 16UC3/C4 -> 32FC3/C4              |  x  |  x  |  -  |  -  |  -   |
// | 16UC1 -> 64FC1                     |  -  |  -  |  -  |  -  |  x*  |
// | 16UC3/C4 -> 64FC3/C4              |  -  |  -  |  -  |  -  |  -   |
// | 16SC1/C3/C4 -> 16SC1/C3/C4        |  -  |  -  |  x  |  x  |  -   |
// | 16SC1 -> 32FC1                     |  x  |  x  |  -  |  -  |  x*  |
// | 16SC3/C4 -> 32FC3/C4              |  x  |  x  |  -  |  -  |  -   |
// | 16SC1 -> 64FC1                     |  -  |  -  |  -  |  -  |  x*  |
// | 16SC3/C4 -> 64FC3/C4              |  -  |  -  |  -  |  -  |  -   |
// | 32FC1 -> 32FC1                     |  x  |  x  |  x  |  x  |  x   |
// | 32FC3 -> 32FC3                     |  x  |  x  |  -  |  -  |  -   |
// | 32FC4 -> 32FC4                     |  x  |  x  |  x  |  x  |  x   |
// | 32FC1 -> 64FC1                     | x*  | x*  |  -  |  -  |  x*  |
// | 32FC3/C4 -> 64FC3/C4              | x*  | x*  |  -  |  -  |  -   |
// | 64FC1 -> 64FC1                     | x*  | x*  | x*  | x*  |  x*  |
// | 64FC3/C4 -> 64FC3/C4              | x*  | x*  |  -  |  -  |  -   |
//
// 'x' in SUM/AVG denotes the existing shared universal-intrinsics kernel; 'x'
// in MIN/MAX/SUM2 denotes a native NEON kernel. '*' requires AArch64. For legal
// MIN/MAX/SUM2 combinations marked '-', and for other channel counts, dispatch
// uses the shared generic fallback.

template<bool isMax>
static void minMax8uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            uchar result = isMax ? 0 : UCHAR_MAX;
            int x = 0;
            uint8x16_t acc = vdupq_n_u8(result);
            for (; x <= cols - 16; x += 16)
            {
                uint8x16_t v = vld1q_u8(src + x);
                acc = isMax ? vmaxq_u8(acc, v) : vminq_u8(acc, v);
            }
            uchar lanes[16];
            vst1q_u8(lanes, acc);
            for (int i = 0; i < 16; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dst[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            ushort result = isMax ? 0 : USHRT_MAX;
            int x = 0;
            uint16x8_t acc = vdupq_n_u16(result);
            for (; x <= cols - 8; x += 8)
            {
                uint16x8_t v = vld1q_u16(src + x);
                acc = isMax ? vmaxq_u16(acc, v) : vminq_u16(acc, v);
            }
            ushort lanes[8];
            vst1q_u16(lanes, acc);
            for (int i = 0; i < 8; i++)
                result = isMax ? std::max(result, lanes[i]) : std::min(result, lanes[i]);
            for (; x < cols; x++)
                result = isMax ? std::max(result, src[x]) : std::min(result, src[x]);
            dstmat.ptr<ushort>(y)[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16sC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            short result = isMax ? SHRT_MIN : SHRT_MAX;
            int x = 0;
            int16x8_t acc = vdupq_n_s16(result);
            for (; x <= cols - 8; x += 8)
            {
                int16x8_t v = vld1q_s16(src + x);
                acc = isMax ? vmaxq_s16(acc, v) : vminq_s16(acc, v);
            }
            short lanes[8];
            vst1q_s16(lanes, acc);
            for (int i = 0; i < 8; i++)
                result = isMax ? std::max(result, lanes[i]) : std::min(result, lanes[i]);
            for (; x < cols; x++)
                result = isMax ? std::max(result, src[x]) : std::min(result, src[x]);
            dstmat.ptr<short>(y)[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const ushort initial = isMax ? 0 : USHRT_MAX;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            ushort* dst = dstmat.ptr<ushort>(y);
            uint16x8x4_t acc = {{
                vdupq_n_u16(initial), vdupq_n_u16(initial),
                vdupq_n_u16(initial), vdupq_n_u16(initial)
            }};
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                uint16x8x4_t v = vld4q_u16(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = isMax ? vmaxq_u16(acc.val[c], v.val[c])
                                       : vminq_u16(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                ushort lanes[8];
                vst1q_u16(lanes, acc.val[c]);
                ushort result = initial;
                for (int i = 0; i < 8; i++)
                    result = isMax ? std::max(result, lanes[i]) : std::min(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = isMax ? std::max(result, src[i * 4 + c])
                                   : std::min(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16sC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const short initial = isMax ? SHRT_MIN : SHRT_MAX;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            short* dst = dstmat.ptr<short>(y);
            int16x8x4_t acc = {{
                vdupq_n_s16(initial), vdupq_n_s16(initial),
                vdupq_n_s16(initial), vdupq_n_s16(initial)
            }};
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                int16x8x4_t v = vld4q_s16(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = isMax ? vmaxq_s16(acc.val[c], v.val[c])
                                       : vminq_s16(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                short lanes[8];
                vst1q_s16(lanes, acc.val[c]);
                short result = initial;
                for (int i = 0; i < 8; i++)
                    result = isMax ? std::max(result, lanes[i]) : std::min(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = isMax ? std::max(result, src[i * 4 + c])
                                   : std::min(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const ushort initial = isMax ? 0 : USHRT_MAX;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            ushort* dst = dstmat.ptr<ushort>(y);
            uint16x8x3_t acc = {{vdupq_n_u16(initial), vdupq_n_u16(initial), vdupq_n_u16(initial)}};
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                uint16x8x3_t v = vld3q_u16(src + x * 3);
                for (int c = 0; c < 3; c++)
                    acc.val[c] = isMax ? vmaxq_u16(acc.val[c], v.val[c])
                                       : vminq_u16(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 3; c++)
            {
                ushort lanes[8];
                vst1q_u16(lanes, acc.val[c]);
                ushort result = initial;
                for (int i = 0; i < 8; i++)
                    result = isMax ? std::max(result, lanes[i]) : std::min(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = isMax ? std::max(result, src[i * 3 + c])
                                   : std::min(result, src[i * 3 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16sC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const short initial = isMax ? SHRT_MIN : SHRT_MAX;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            short* dst = dstmat.ptr<short>(y);
            int16x8x3_t acc = {{vdupq_n_s16(initial), vdupq_n_s16(initial), vdupq_n_s16(initial)}};
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                int16x8x3_t v = vld3q_s16(src + x * 3);
                for (int c = 0; c < 3; c++)
                    acc.val[c] = isMax ? vmaxq_s16(acc.val[c], v.val[c])
                                       : vminq_s16(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 3; c++)
            {
                short lanes[8];
                vst1q_s16(lanes, acc.val[c]);
                short result = initial;
                for (int i = 0; i < 8; i++)
                    result = isMax ? std::max(result, lanes[i]) : std::min(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = isMax ? std::max(result, src[i * 3 + c])
                                   : std::min(result, src[i * 3 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax8uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            const uchar initial = isMax ? 0 : UCHAR_MAX;
            uint8x16x3_t acc = {{
                vdupq_n_u8(initial), vdupq_n_u8(initial), vdupq_n_u8(initial)
            }};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x3_t v = vld3q_u8(src + x * 3);
                for (int c = 0; c < 3; c++)
                    acc.val[c] = isMax ? vmaxq_u8(acc.val[c], v.val[c])
                                       : vminq_u8(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 3; c++)
            {
                uchar lanes[16];
                vst1q_u8(lanes, acc.val[c]);
                uchar result = initial;
                for (int i = 0; i < 16; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 3 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax8uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            const uchar initial = isMax ? 0 : UCHAR_MAX;
            uint8x16x4_t acc = {{
                vdupq_n_u8(initial), vdupq_n_u8(initial),
                vdupq_n_u8(initial), vdupq_n_u8(initial)
            }};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x4_t v = vld4q_u8(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = isMax ? vmaxq_u8(acc.val[c], v.val[c])
                                       : vminq_u8(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                uchar lanes[16];
                vst1q_u8(lanes, acc.val[c]);
                uchar result = initial;
                for (int i = 0; i < 16; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = src[0];
            int x = 0;
            float32x4_t acc = vdupq_n_f32(result);
            for (; x <= cols - 4; x += 4)
            {
                float32x4_t v = vld1q_f32(src + x);
                acc = isMax ? vmaxq_f32(acc, v) : vminq_f32(acc, v);
            }
            float lanes[4];
            vst1q_f32(lanes, acc);
            for (int i = 0; i < 4; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax32fC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            float32x4x4_t acc = {{
                vdupq_n_f32(initial), vdupq_n_f32(initial),
                vdupq_n_f32(initial), vdupq_n_f32(initial)
            }};
            int x = 0;
            for (; x <= cols - 4; x += 4)
            {
                float32x4x4_t v = vld4q_f32(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = isMax ? vmaxq_f32(acc.val[c], v.val[c])
                                       : vminq_f32(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                float lanes[4];
                vst1q_f32(lanes, acc.val[c]);
                float result = initial;
                for (int i = 0; i < 4; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

#if defined(__aarch64__) || defined(_M_ARM64)
template<bool isMax>
static void minMax64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const double initial = isMax ? std::numeric_limits<double>::lowest()
                                 : std::numeric_limits<double>::max();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const double* src = srcmat.ptr<double>(y);
            double result = initial;
            int x = 0;
            float64x2_t acc0 = vdupq_n_f64(initial);
            float64x2_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            for (; x <= cols - 8; x += 8)
            {
                float64x2_t v0 = vld1q_f64(src + x);
                float64x2_t v1 = vld1q_f64(src + x + 2);
                float64x2_t v2 = vld1q_f64(src + x + 4);
                float64x2_t v3 = vld1q_f64(src + x + 6);
                acc0 = isMax ? vmaxq_f64(acc0, v0) : vminq_f64(acc0, v0);
                acc1 = isMax ? vmaxq_f64(acc1, v1) : vminq_f64(acc1, v1);
                acc2 = isMax ? vmaxq_f64(acc2, v2) : vminq_f64(acc2, v2);
                acc3 = isMax ? vmaxq_f64(acc3, v3) : vminq_f64(acc3, v3);
            }
            acc0 = isMax ? vmaxq_f64(acc0, acc1) : vminq_f64(acc0, acc1);
            acc2 = isMax ? vmaxq_f64(acc2, acc3) : vminq_f64(acc2, acc3);
            acc0 = isMax ? vmaxq_f64(acc0, acc2) : vminq_f64(acc0, acc2);
            double lanes[2];
            vst1q_f64(lanes, acc0);
            for (int i = 0; i < 2; i++)
                result = isMax ? std::max(result, lanes[i]) : std::min(result, lanes[i]);
            for (; x < cols; x++)
                result = isMax ? std::max(result, src[x]) : std::min(result, src[x]);
            dstmat.ptr<double>(y)[0] = result;
        }
    });
    v_cleanup();
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
static inline uint32_t reduceSum2_8u_NEON(uint8x16_t v)
{
    uint16x8_t lo = vmull_u8(vget_low_u8(v), vget_low_u8(v));
    uint16x8_t hi = vmull_u8(vget_high_u8(v), vget_high_u8(v));
    return vaddvq_u32(vpaddlq_u16(lo)) + vaddvq_u32(vpaddlq_u16(hi));
}

template<typename DT>
static void sum2_8uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t result = 0;
            int x = 0;
            for (; x <= cols - 16; x += 16)
                result += reduceSum2_8u_NEON(vld1q_u8(src + x));
            for (; x < cols; x++)
                result += (uint32_t)src[x] * src[x];
            dst[0] = (DT)(int32_t)result;
        }
    });
    v_cleanup();
}

static void sum2_16u32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            float result = 0;
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint16x8_t v01 = vld1q_u16(src + x);
                uint16x8_t v23 = vld1q_u16(src + x + 8);
                float32x4_t v0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v01)));
                float32x4_t v1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v01)));
                float32x4_t v2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v23)));
                float32x4_t v3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v23)));
                acc0 = vmlaq_f32(acc0, v0, v0);
                acc1 = vmlaq_f32(acc1, v1, v1);
                acc2 = vmlaq_f32(acc2, v2, v2);
                acc3 = vmlaq_f32(acc3, v3, v3);
            }
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float lanes[4];
            vst1q_f32(lanes, acc0);
            for (int i = 0; i < 4; i++)
                result += lanes[i];
            for (; x < cols; x++)
            {
                float value = (float)src[x];
                result += value * value;
            }
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

static void sum2_16s32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            float result = 0;
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                int16x8_t v01 = vld1q_s16(src + x);
                int16x8_t v23 = vld1q_s16(src + x + 8);
                float32x4_t v0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v01)));
                float32x4_t v1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v01)));
                float32x4_t v2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v23)));
                float32x4_t v3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v23)));
                acc0 = vmlaq_f32(acc0, v0, v0);
                acc1 = vmlaq_f32(acc1, v1, v1);
                acc2 = vmlaq_f32(acc2, v2, v2);
                acc3 = vmlaq_f32(acc3, v3, v3);
            }
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float lanes[4];
            vst1q_f32(lanes, acc0);
            for (int i = 0; i < 4; i++)
                result += lanes[i];
            for (; x < cols; x++)
            {
                float value = (float)src[x];
                result += value * value;
            }
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

static void sum2_16u64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            uint64x2_t acc0 = vdupq_n_u64(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                uint16x8_t v = vld1q_u16(src + x);
                uint32x4_t sq0 = vmull_u16(vget_low_u16(v), vget_low_u16(v));
                uint32x4_t sq1 = vmull_u16(vget_high_u16(v), vget_high_u16(v));
                acc0 = vaddq_u64(acc0, vmovl_u32(vget_low_u32(sq0)));
                acc1 = vaddq_u64(acc1, vmovl_high_u32(sq0));
                acc2 = vaddq_u64(acc2, vmovl_u32(vget_low_u32(sq1)));
                acc3 = vaddq_u64(acc3, vmovl_high_u32(sq1));
            }
            acc0 = vaddq_u64(vaddq_u64(acc0, acc1), vaddq_u64(acc2, acc3));
            uint64_t lanes[2];
            vst1q_u64(lanes, acc0);
            uint64_t result = lanes[0] + lanes[1];
            for (; x < cols; x++)
                result += (uint64_t)src[x] * src[x];
            dstmat.ptr<double>(y)[0] = (double)result;
        }
    });
    v_cleanup();
}

static void sum2_16s64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            uint64x2_t acc0 = vdupq_n_u64(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                int16x8_t v = vld1q_s16(src + x);
                int32x4_t sq0 = vmull_s16(vget_low_s16(v), vget_low_s16(v));
                int32x4_t sq1 = vmull_s16(vget_high_s16(v), vget_high_s16(v));
                uint32x4_t usq0 = vreinterpretq_u32_s32(sq0);
                uint32x4_t usq1 = vreinterpretq_u32_s32(sq1);
                acc0 = vaddq_u64(acc0, vmovl_u32(vget_low_u32(usq0)));
                acc1 = vaddq_u64(acc1, vmovl_high_u32(usq0));
                acc2 = vaddq_u64(acc2, vmovl_u32(vget_low_u32(usq1)));
                acc3 = vaddq_u64(acc3, vmovl_high_u32(usq1));
            }
            acc0 = vaddq_u64(vaddq_u64(acc0, acc1), vaddq_u64(acc2, acc3));
            uint64_t lanes[2];
            vst1q_u64(lanes, acc0);
            uint64_t result = lanes[0] + lanes[1];
            for (; x < cols; x++)
            {
                int64_t value = src[x];
                result += (uint64_t)(value * value);
            }
            dstmat.ptr<double>(y)[0] = (double)result;
        }
    });
    v_cleanup();
}

static void sum2_32f64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float64x2_t acc0 = vdupq_n_f64(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                float32x4_t v01 = vld1q_f32(src + x);
                float32x4_t v23 = vld1q_f32(src + x + 4);
                float64x2_t v0 = vcvt_f64_f32(vget_low_f32(v01));
                float64x2_t v1 = vcvt_high_f64_f32(v01);
                float64x2_t v2 = vcvt_f64_f32(vget_low_f32(v23));
                float64x2_t v3 = vcvt_high_f64_f32(v23);
                acc0 = vmlaq_f64(acc0, v0, v0);
                acc1 = vmlaq_f64(acc1, v1, v1);
                acc2 = vmlaq_f64(acc2, v2, v2);
                acc3 = vmlaq_f64(acc3, v3, v3);
            }
            acc0 = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
            double lanes[2];
            vst1q_f64(lanes, acc0);
            double result = lanes[0] + lanes[1];
            for (; x < cols; x++)
            {
                double value = (double)src[x];
                result += value * value;
            }
            dstmat.ptr<double>(y)[0] = result;
        }
    });
    v_cleanup();
}

static void sum2_64f64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const double* src = srcmat.ptr<double>(y);
            float64x2_t acc0 = vdupq_n_f64(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                float64x2_t v0 = vld1q_f64(src + x);
                float64x2_t v1 = vld1q_f64(src + x + 2);
                float64x2_t v2 = vld1q_f64(src + x + 4);
                float64x2_t v3 = vld1q_f64(src + x + 6);
                acc0 = vmlaq_f64(acc0, v0, v0);
                acc1 = vmlaq_f64(acc1, v1, v1);
                acc2 = vmlaq_f64(acc2, v2, v2);
                acc3 = vmlaq_f64(acc3, v3, v3);
            }
            acc0 = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
            double lanes[2];
            vst1q_f64(lanes, acc0);
            double result = lanes[0] + lanes[1];
            for (; x < cols; x++)
                result += src[x] * src[x];
            dstmat.ptr<double>(y)[0] = result;
        }
    });
    v_cleanup();
}

template<typename DT>
static void sum2_8uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t results[3] = {0, 0, 0};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x3_t v = vld3q_u8(src + x * 3);
                for (int c = 0; c < 3; c++)
                    results[c] += reduceSum2_8u_NEON(v.val[c]);
            }
            for (int c = 0; c < 3; c++)
            {
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 3 + c];
                    results[c] += value * value;
                }
                dst[c] = (DT)(int32_t)results[c];
            }
        }
    });
    v_cleanup();
}

template<typename DT>
static void sum2_8uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t results[4] = {0, 0, 0, 0};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x4_t v = vld4q_u8(src + x * 4);
                for (int c = 0; c < 4; c++)
                    results[c] += reduceSum2_8u_NEON(v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 4 + c];
                    results[c] += value * value;
                }
                dst[c] = (DT)(int32_t)results[c];
            }
        }
    });
    v_cleanup();
}

#endif

static void sum2_32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = 0;
            int x = 0;
            float32x4_t acc = vdupq_n_f32(0);
            for (; x <= cols - 4; x += 4)
            {
                float32x4_t v = vld1q_f32(src + x);
                acc = vaddq_f32(acc, vmulq_f32(v, v));
            }
            float lanes[4];
            vst1q_f32(lanes, acc);
            for (int i = 0; i < 4; i++)
                result += lanes[i];
            for (; x < cols; x++)
                result += src[x] * src[x];
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

static void sum2_32fC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            float32x4x4_t acc = {{
                vdupq_n_f32(0), vdupq_n_f32(0),
                vdupq_n_f32(0), vdupq_n_f32(0)
            }};
            int x = 0;
            for (; x <= cols - 4; x += 4)
            {
                float32x4x4_t v = vld4q_f32(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = vmlaq_f32(acc.val[c], v.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                float lanes[4];
                float result = 0;
                vst1q_f32(lanes, acc.val[c]);
                for (int i = 0; i < 4; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                    result += src[i * 4 + c] * src[i * 4 + c];
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

} // namespace reduce_c_neon
