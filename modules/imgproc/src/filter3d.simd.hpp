// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at https://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void filterDepth16f(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template<typename T>
static inline void accumulateDepth(const T* in, float* out, int count, float weight)
{
    for (int i = 0; i < count; ++i)
        out[i] += weight * static_cast<float>(in[i]);
}

#if CV_AVX2 && CV_FP16
static inline void accumulateDepth(const hfloat* in, float* out, int count, float weight)
{
    const int lanes = VTraits<v_float32>::vlanes();
    const v_float32 w = vx_setall_f32(weight);
    int i = 0;
    for (; i <= count - lanes; i += lanes)
        v_store(out + i, v_fma(vx_load_expand(in + i), w, vx_load(out + i)));
    accumulateDepth<hfloat>(in + i, out + i, count - i, weight);
}
#endif

template<typename T>
static inline void filterDepth(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes)
{
    const float* weights = kernel.ptr<float>();
    const int rowSize = dst.cols * dst.channels();
    for (int y = 0; y < dst.rows; ++y)
    {
        float* out = dst.ptr<float>(y);
        std::fill(out, out + rowSize, 0.f);
        for (size_t k = 0; k < planes.size(); ++k)
            if (planes[k] >= 0)
                accumulateDepth(src.ptr<T>(planes[k], y), out, rowSize, weights[k]);
    }
}

void filterDepth16f(const Mat& src, Mat& dst, const Mat& kernel, const std::vector<int>& planes)
{
    filterDepth<hfloat>(src, dst, kernel, planes);
}

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
