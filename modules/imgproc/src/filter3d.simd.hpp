// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at https://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void filterDepth(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes,
                 int firstRow);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if (CV_SIMD || CV_SIMD_SCALABLE)
// Each overload loads 2 * vlanes elements as two float32 vectors.
static inline void loadAsFloat(const uchar* in, v_float32& lo, v_float32& hi)
{
    v_uint32 lo32, hi32;
    v_expand(vx_load_expand(in), lo32, hi32);
    lo = v_cvt_f32(v_reinterpret_as_s32(lo32));
    hi = v_cvt_f32(v_reinterpret_as_s32(hi32));
}

static inline void loadAsFloat(const hfloat* in, v_float32& lo, v_float32& hi)
{
    lo = vx_load_expand(in);
    hi = vx_load_expand(in + VTraits<v_float32>::vlanes());
}

static inline void loadAsFloat(const float* in, v_float32& lo, v_float32& hi)
{
    lo = vx_load(in);
    hi = vx_load(in + VTraits<v_float32>::vlanes());
}
#endif

template<typename T>
static inline void accumulateDepth(const T* in, float* out, int count, float weight)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int lanes = VTraits<v_float32>::vlanes();
    const v_float32 w = vx_setall_f32(weight);
    for (; i <= count - 2 * lanes; i += 2 * lanes)
    {
        v_float32 lo, hi;
        loadAsFloat(in + i, lo, hi);
        v_store(out + i, v_fma(lo, w, vx_load(out + i)));
        v_store(out + i + lanes, v_fma(hi, w, vx_load(out + i + lanes)));
    }
#endif
    for (; i < count; ++i)
        out[i] += weight * static_cast<float>(in[i]);
}

template<typename T>
static void filterDepth_(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes,
                         int firstRow)
{
    const float* weights = kernel.ptr<float>();
    const int rowSize = dst.cols * dst.channels();
    for (int y = 0; y < dst.rows; ++y)
    {
        float* out = dst.ptr<float>(y);
        std::fill(out, out + rowSize, 0.f);
        for (size_t k = 0; k < planes.size(); ++k)
            if (planes[k] >= 0)
                accumulateDepth(src.ptr<T>(planes[k], firstRow + y), out, rowSize, weights[k]);
    }
    vx_cleanup();
}

void filterDepth(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes,
                 int firstRow)
{
    CV_INSTRUMENT_REGION();

    if (src.depth() == CV_8U)
        filterDepth_<uchar>(src, dst, kernel, planes, firstRow);
    else if (src.depth() == CV_16F)
        filterDepth_<hfloat>(src, dst, kernel, planes, firstRow);
    else
        filterDepth_<float>(src, dst, kernel, planes, firstRow);
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
