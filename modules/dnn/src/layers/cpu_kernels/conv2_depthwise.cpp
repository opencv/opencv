// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "../../net_impl.hpp"
#include "../conv2_common.hpp"
#include "conv2_depthwise.simd.hpp"
#include "layers/cpu_kernels/conv2_depthwise.simd_declarations.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

DepthwiseConvFunc getDepthwiseConvFunc(int depth)
{
    CV_CPU_DISPATCH(getDepthwiseConvFunc_, (depth), CV_CPU_DISPATCH_MODES_ALL);
}

template <typename InpT, typename OutT>
static void repackDepthwiseWeightsBlock(const InpT* inpw, OutT* outw,
                                        int ksize, int C0, int currC0)
{
    for (int xy = 0; xy < ksize; xy++, inpw++, outw += C0) {
        for (int c0 = 0; c0 < currC0; c0++) {
            outw[c0] = OutT(inpw[ksize*c0]);
        }
    }
}

// C x 1 x Hk x Wk => C1 x Hk x Wk x C0
void repackDepthwiseConvWeights(const void* inpw__, int inptype_, void* outw__, int outtype_,
                                const MatShape& wshape, int C0_)
{
    CV_Assert(inptype_ == CV_32F || inptype_ == CV_16F || inptype_ == CV_16BF);
    CV_Assert(outtype_ == CV_32F || outtype_ == CV_16F || outtype_ == CV_16BF);
    CV_Assert(wshape.dims == 4 && wshape[1] == 1);

    int C1_ = (wshape[0] + C0_ - 1)/C0_;
    parallel_for_(Range(0, C1_), [&](const Range& r) {
        int inptype = inptype_, outtype = outtype_;
        size_t inpEsz = CV_ELEM_SIZE(inptype);
        size_t outEsz = CV_ELEM_SIZE(outtype);
        int C = wshape[0], C0 = C0_;
        int ksize = wshape[2]*wshape[3];

        for (int c1 = r.start; c1 < r.end; c1++) {
            const uint8_t* inpw_ = (const uint8_t*)inpw__ + c1*ksize*C0*inpEsz;
            uint8_t* outw_ = (uint8_t*)outw__ + c1*ksize*C0*outEsz;
            int currC0 = std::min(C - c1*C0, C0);
            if (currC0 < C0)
                memset(outw_, 0, ksize*C0*outEsz);

            if (inptype == CV_32F) {
                const float* inpw = (const float*)inpw_;
                if (outtype == CV_32F) {
                    float* outw = (float*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                } else if (outtype == CV_16F) {
                    hfloat* outw = (hfloat*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                } else if (outtype == CV_16BF) {
                    bfloat* outw = (bfloat*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                }
            } else if (inptype == CV_16F) {
                const hfloat* inpw = (const hfloat*)inpw_;
                if (outtype == CV_32F) {
                    float* outw = (float*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                } else if (outtype == CV_16F) {
                    hfloat* outw = (hfloat*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                } else if (outtype == CV_16BF) {
                    bfloat* outw = (bfloat*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                }
            } else if (inptype == CV_16BF) {
                const bfloat* inpw = (const bfloat*)inpw_;
                if (outtype == CV_32F) {
                    float* outw = (float*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                } else if (outtype == CV_16F) {
                    hfloat* outw = (hfloat*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                } else if (outtype == CV_16BF) {
                    bfloat* outw = (bfloat*)outw_;
                    repackDepthwiseWeightsBlock(inpw, outw, ksize, C0, currC0);
                }
            }
        }
    });
}

CV__DNN_INLINE_NS_END
}}
