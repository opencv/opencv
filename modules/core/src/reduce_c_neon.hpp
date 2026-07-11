// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

namespace reduce_c_neon
{

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
