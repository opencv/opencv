// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpConv.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "fast_convolution.hpp"

namespace cv { namespace dnn {

static void depthWiseBlock(const float *inptr, float *outptr, const float *weights, float biasval, int *ofstab, int *yxtab,
                           float minval, float maxval, int Hi, int Wi, int H0, int W0, int ksize, int pad_top, int pad_left,
                           int dilation_y, int stride_x, int stride_y, int inner_xleft, int inner_xright, int inner_ytop,
                           int inner_ybottom, bool ifMinMaxAct, bool useSIMD, bool is3x3)
{
#if CV_SIMD128
    v_float32x4 vminval = v_setall_f32(minval), vmaxval = v_setall_f32(maxval);

    v_float32x4 w0 = v_setall_f32(
            0.f), w1 = w0, w2 = w0, w3 = w0, w4 = w0, w5 = w0, w6 = w0, w7 = w0, w8 = w0, vbias = w0;
    if (useSIMD)
    {
        vbias = v_setall_f32(biasval);
        if (is3x3)
        {
            w0 = v_setall_f32(weights[0]);
            w1 = v_setall_f32(weights[1]);
            w2 = v_setall_f32(weights[2]);
            w3 = v_setall_f32(weights[3]);
            w4 = v_setall_f32(weights[4]);
            w5 = v_setall_f32(weights[5]);
            w6 = v_setall_f32(weights[6]);
            w7 = v_setall_f32(weights[7]);
            w8 = v_setall_f32(weights[8]);
        }
    }
#endif
    int dy0 = 1;
    for (int y0 = 0; y0 < H0; y0 += dy0, outptr += W0 * dy0)
    {
#if CV_SIMD128
        dy0 = inner_ytop <= y0 && y0 + 3 < inner_ybottom && is3x3 && stride_y == 1 && dilation_y == 1
              ? 3 : 1;
#endif
        int x0 = 0, x1 = y0 >= inner_ytop && y0 < inner_ybottom ? inner_xleft : W0;
        int yi_ = y0 * stride_y - pad_top;

        for (;;)
        {
            float s_0, s_1, s_2;
            if (dy0 == 3)
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    s_0 = s_1 = s_2 = biasval;
                    for (int k = 0; k < ksize; k++)
                    {
                        int dy = yxtab[k * 2];
                        int yi = yi_ + dy;
                        int xi = xi_ + yxtab[k * 2 + 1];
                        float w = weights[k];

                        if ((unsigned) xi < (unsigned) Wi)
                        {
                            s_0 += inptr[yi * Wi + xi] * w;
                            s_1 += inptr[(yi + 1) * Wi + xi] * w;
                            s_2 += inptr[(yi + 2) * Wi + xi] * w;
                        }
                    }
                    s_0 = std::min(std::max(s_0, minval), maxval);
                    s_1 = std::min(std::max(s_1, minval), maxval);
                    s_2 = std::min(std::max(s_2, minval), maxval);
                    outptr[x0] = s_0;
                    outptr[x0 + W0] = s_1;
                    outptr[x0 + W0 * 2] = s_2;
                }
            }
            else
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    s_0 = biasval;
                    for (int k = 0; k < ksize; k++) {
                        int dy = yxtab[k * 2];
                        int yi = yi_ + dy;
                        int xi = xi_ + yxtab[k * 2 + 1];
                        float w = weights[k];
                        if (((unsigned) yi < (unsigned) Hi) & ((unsigned) xi < (unsigned) Wi))
                            s_0 += inptr[yi * Wi + xi] * w;
                    }
                    s_0 = std::min(std::max(s_0, minval), maxval);
                    outptr[x0] = s_0;
                }
            }
            if (x0 == W0)
                break;
            x1 = inner_xright;
#if CV_SIMD128
            if (useSIMD)
            {
                if (is3x3)
                {
                    if (dy0 == 3)
                    {
                        for (; x0 <= x1 - FAST_VEC_NLANES; x0 += FAST_VEC_NLANES)
                        {
                            int xi_ = x0 * stride_x - pad_left;
                            const float *inptr_xi = inptr + Wi * yi_ + xi_;

                            v_float32x4 s0, s1, s2;
                            v_float32x4 x00 = v_load(inptr_xi);
                            v_float32x4 x01 = v_load(inptr_xi + 1);
                            v_float32x4 x02 = v_load(inptr_xi + 2);

                            v_float32x4 x10 = v_load(inptr_xi + Wi);
                            v_float32x4 x11 = v_load(inptr_xi + Wi + 1);
                            v_float32x4 x12 = v_load(inptr_xi + Wi + 2);

                            v_float32x4 x20 = v_load(inptr_xi + Wi * 2);
                            v_float32x4 x21 = v_load(inptr_xi + Wi * 2 + 1);
                            v_float32x4 x22 = v_load(inptr_xi + Wi * 2 + 2);

                            v_float32x4 x30 = v_load(inptr_xi + Wi * 3);
                            v_float32x4 x31 = v_load(inptr_xi + Wi * 3 + 1);
                            v_float32x4 x32 = v_load(inptr_xi + Wi * 3 + 2);

                            v_float32x4 x40 = v_load(inptr_xi + Wi * 4);
                            v_float32x4 x41 = v_load(inptr_xi + Wi * 4 + 1);
                            v_float32x4 x42 = v_load(inptr_xi + Wi * 4 + 2);

                            s0 = v_fma(x00, w0, vbias);
                            s1 = v_fma(x10, w0, vbias);
                            s2 = v_fma(x20, w0, vbias);

                            s0 = v_fma(x01, w1, s0);
                            s1 = v_fma(x11, w1, s1);
                            s2 = v_fma(x21, w1, s2);

                            s0 = v_fma(x02, w2, s0);
                            s1 = v_fma(x12, w2, s1);
                            s2 = v_fma(x22, w2, s2);

                            s0 = v_fma(x10, w3, s0);
                            s1 = v_fma(x20, w3, s1);
                            s2 = v_fma(x30, w3, s2);

                            s0 = v_fma(x11, w4, s0);
                            s1 = v_fma(x21, w4, s1);
                            s2 = v_fma(x31, w4, s2);

                            s0 = v_fma(x12, w5, s0);
                            s1 = v_fma(x22, w5, s1);
                            s2 = v_fma(x32, w5, s2);

                            s0 = v_fma(x20, w6, s0);
                            s1 = v_fma(x30, w6, s1);
                            s2 = v_fma(x40, w6, s2);

                            s0 = v_fma(x21, w7, s0);
                            s1 = v_fma(x31, w7, s1);
                            s2 = v_fma(x41, w7, s2);

                            s0 = v_fma(x22, w8, s0);
                            s1 = v_fma(x32, w8, s1);
                            s2 = v_fma(x42, w8, s2);

                            if (ifMinMaxAct)
                            {
                                s0 = v_min(v_max(s0, vminval), vmaxval);
                                s1 = v_min(v_max(s1, vminval), vmaxval);
                                s2 = v_min(v_max(s2, vminval), vmaxval);
                            }

                            v_store(outptr + x0, s0);
                            v_store(outptr + W0 + x0, s1);
                            v_store(outptr + W0 * 2 + x0, s2);
                        }
                    }
                    else
                    {
                        for (; x0 <= x1 - FAST_VEC_NLANES; x0 += FAST_VEC_NLANES)
                        {
                            int xi_ = x0 * stride_x - pad_left;
                            const float *inptr_xi = inptr + Wi * yi_ + xi_;
                            v_float32x4 s0 = v_fma(v_load(inptr_xi + ofstab[0]), w0, vbias);
                            v_float32x4 s1 = v_load(inptr_xi + ofstab[1]) * w1;
                            v_float32x4 s2 = v_load(inptr_xi + ofstab[2]) * w2;

                            s0 = v_fma(v_load(inptr_xi + ofstab[3]), w3, s0);
                            s1 = v_fma(v_load(inptr_xi + ofstab[4]), w4, s1);
                            s2 = v_fma(v_load(inptr_xi + ofstab[5]), w5, s2);

                            s0 = v_fma(v_load(inptr_xi + ofstab[6]), w6, s0);
                            s1 = v_fma(v_load(inptr_xi + ofstab[7]), w7, s1);
                            s2 = v_fma(v_load(inptr_xi + ofstab[8]), w8, s2);

                            s0 = s0 + s1 + s2;
                            if (ifMinMaxAct)
                                s0 = v_min(v_max(s0, vminval), vmaxval);
                            v_store(outptr + x0, s0);
                        }
                    }
                }
                else
                {
                    for (; x0 <= x1 - FAST_VEC_NLANES; x0 += FAST_VEC_NLANES)
                    {
                        int xi_ = x0 * stride_x - pad_left, k = 0;
                        const float *inptr_xi = inptr + Wi * yi_ + xi_;
                        v_float32x4 s0 = vbias;
                        for (; k <= ksize - 4; k += 4)
                        {
                            v_float32x4 v0 = v_load(inptr_xi + ofstab[k]);
                            v_float32x4 v1 = v_load(inptr_xi + ofstab[k + 1]);
                            v_float32x4 v2 = v_load(inptr_xi + ofstab[k + 2]);
                            v_float32x4 v3 = v_load(inptr_xi + ofstab[k + 3]);

                            v_float32x4 ww0 = v_setall_f32(weights[k]);
                            v_float32x4 ww1 = v_setall_f32(weights[k+1]);
                            v_float32x4 ww2 = v_setall_f32(weights[k+2]);
                            v_float32x4 ww3 = v_setall_f32(weights[k+3]);

                            s0 = v_fma(v0, ww0, s0);
                            s0 = v_fma(v1, ww1, s0);
                            s0 = v_fma(v2, ww2, s0);
                            s0 = v_fma(v3, ww3, s0);
                        }
                        for (; k < ksize; k++)
                            s0 = v_fma(v_load(inptr_xi + ofstab[k]),
                                       v_setall_f32(weights[k]), s0);
                        if (ifMinMaxAct)
                            s0 = v_min(v_max(s0, vminval), vmaxval);
                        v_store(outptr + x0, s0);
                    }
                }
            }
#endif
            if (dy0 == 3)
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    const float *inptr_xi = inptr + W0 * yi_ + xi_;
                    s_0 = s_1 = s_2 = biasval;
                    for (int k = 0; k < ksize; k++)
                    {
                        int inp_ofs = ofstab[k];
                        float w = weights[k];
                        s_0 += inptr_xi[inp_ofs] * w;
                        s_1 += inptr_xi[inp_ofs + Wi] * w;
                        s_2 += inptr_xi[inp_ofs + Wi * 2] * w;
                    }
                    if (ifMinMaxAct)
                    {
                        s_0 = std::min(std::max(s_0, minval), maxval);
                        s_1 = std::min(std::max(s_1, minval), maxval);
                        s_2 = std::min(std::max(s_2, minval), maxval);
                    }

                    outptr[x0] = s_0;
                    outptr[x0 + W0] = s_1;
                    outptr[x0 + W0 * 2] = s_2;
                }
            }
            else
            {
                for (; x0 < x1; x0++)
                {
                    int xi_ = x0 * stride_x - pad_left;
                    const float *inptr_xi = inptr + Wi * yi_ + xi_;
                    s_0 = biasval;
                    for (int k = 0; k < ksize; k++)
                    {
                        s_0 += inptr_xi[ofstab[k]] * weights[k];
                    }

                    if (ifMinMaxAct)
                        s_0 = std::min(std::max(s_0, minval), maxval);
                    outptr[x0] = s_0;
                }
            }
            x1 = W0;
        }
    }
}

void runDepthwise(InputArray _input, OutputArray _output, const Ptr<FastConv2d>& conv, float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct) {
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);
    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->K, Hk = conv->Hk, Wk = conv->Wk;
    int H0 = outputShape[2], W0 = outputShape[3], ngroups = conv->ngroups;

    const size_t inp_planesize = (size_t) Hi * Wi;
    const size_t out_planesize = (size_t) H0 * W0;

    CV_Assert(ngroups > 1 && ngroups == K && ngroups == C);

    int stride_y = conv->stride_y, stride_x = conv->stride_x;
    int dilation_y = conv->dilation_y, dilation_x = conv->dilation_x;

    int pad_top = conv->pad_top, pad_bottom = conv->pad_bottom;
    int pad_left = conv->pad_left, pad_right = conv->pad_right;

    int ksize = Hk * Wk, padded_ksize = ((ksize + FAST_VEC_NLANES - 1) / FAST_VEC_NLANES) * FAST_VEC_NLANES;

    const float *inp = input.ptr<float>();
    float *out = output.ptr<float>();

    std::vector<int> ofstab_(3 * padded_ksize, 0);
    int *ofstab = ofstab_.data();
    int *yxtab = ofstab + padded_ksize;

    for (int k = 0; k < padded_ksize; k++)
    {
        int y = k < ksize ? k / Wk : 0;
        int x = k < ksize ? k % Wk : 0;
        int dy = y * dilation_y, dx = x * dilation_x;
        yxtab[k * 2] = dy;
        yxtab[k * 2 + 1] = dx;
        ofstab[k] = dy * Wi + dx;
    }

    const float *weights0 = conv->weightsBuf.data(), *bias = conv->biasBuf.data();
    int inner_ytop = (pad_bottom + stride_y - 1) / stride_y, inner_ybottom = 3;
    int inner_xleft = (pad_left + stride_x - 1) / stride_x, inner_xright = 4;

    CV_Assert(ksize > 1 || (pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0));

    inner_xright = (Wi - (Wk - 1) * dilation_x + pad_left) / stride_x;
    inner_xright += inner_xright * stride_x - pad_left + (Wk - 1) * dilation_x < Wi;
    inner_ybottom = (Hi - (Hk - 1) * dilation_y + pad_top) / stride_y;
    inner_ybottom += inner_ybottom * stride_y - pad_top + (Hk - 1) * dilation_y < Hi;

    if (inner_xleft >= inner_xright || inner_ytop >= inner_ybottom)
    {
        inner_xleft = W0;
        inner_ytop = H0;
    }

    inner_ybottom = inner_ybottom < H0 ? inner_ybottom : H0;

    bool useSIMD = stride_x == 1 && inner_xleft < W0;
    bool is3x3 = Hk == 3 && Wk == 3;

    parallel_for_(Range(0, N * C), [&](const Range &r0) {
        for (int nc = r0.start; nc < r0.end; nc++)
        {
            int c = nc % C;
            const float *inptr = inp + inp_planesize * nc;
            float *outptr0 = out + out_planesize * nc;

            float biasval = bias[c];
            const float *weights = weights0 + c * padded_ksize;

#if CV_TRY_AVX2
            if (conv->useAVX2)
                opt_AVX2::depthWiseBlock_AVX2(inptr, outptr0, weights, biasval, ofstab, yxtab, minval, maxval, Hi, Wi, H0, W0, ksize,
                                         pad_top, pad_left, dilation_y, stride_x, stride_y, inner_xleft, inner_xright, inner_ytop,
                                         inner_ybottom, ifMinMaxAct, useSIMD, is3x3);
            else
#endif
            depthWiseBlock(inptr, outptr0, weights, biasval, ofstab, yxtab, minval, maxval, Hi, Wi, H0, W0, ksize,
                           pad_top, pad_left, dilation_y, stride_x, stride_y, inner_xleft, inner_xright, inner_ytop,
                           inner_ybottom, ifMinMaxAct, useSIMD, is3x3);

            if (activ)
                activ->forwardSlice(outptr0, outptr0, (int) out_planesize, out_planesize, c, c+1);
        }
    });
}

}} // namespace cv::dnn