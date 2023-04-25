// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "convolution.hpp"

#include "conv_depthwise.simd.hpp"
#include "layers/cpu_kernels/conv_depthwise.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv { namespace dnn {

void depthWiseBlockConv2D(const float* wptr,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w,
                                 int dilation_h, int dilation_w,
                                 int pad_t, int pad_l,
                                 const float* biasptr, const float* relu,
                                 const float* inptr_,
                                 int height, int width,
                                 float* outptr_,
                                 int out_d, int outH, int outW, bool fusedAdd);

void depthWiseBlockConv1D(const float* wptr,
                                 int kernel_w, int stride_w, int dilation_w, int pad_l,
                                 const float* biasptr, const float* relu,
                                 const float* inptr_, int width,
                                 float* outptr_,
                                 int out_d, int outW, bool fusedAdd);

void runDepthwise(InputArray _input, OutputArray _output, const Ptr<FastConv>& conv, ActivationLayer* activ_,
                  const std::vector<float>& reluslope, bool fusedAdd)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);

    CV_Assert(inputShape.size() == 3 || inputShape.size() == 4);
    CV_Assert(inputShape.size() == outputShape.size());

    int conv_dim = conv->conv_dim;
    CV_Assert((conv_dim == CONV_2D || conv_dim == CONV_1D) &&
            "DNN: Currently we do not support depth-wise for Convolution 3D!");

    ActivationLayer* activ = reluslope.empty() ? activ_ : nullptr;
    int N = inputShape[0], C = inputShape[1];

    int Hi = conv_dim == CONV_1D ? 1 : inputShape[inputShape.size() - 2];
    int Wi = inputShape[inputShape.size() - 1];

    int K = conv->K, Hk = conv->Hk, Wk = conv->Wk;

    int H0 = conv_dim == CONV_1D ? 1 : outputShape[outputShape.size() - 2];
    int W0 = outputShape[outputShape.size() - 1];
    int ngroups = conv->ngroups;

    const size_t inp_planesize = (size_t) Hi * Wi;
    const size_t out_planesize = (size_t) H0 * W0;

    CV_Assert(ngroups > 1 && ngroups == K && ngroups == C);

    int stride_h = conv->stride_h, stride_w = conv->stride_w;
    int dilation_h = conv->dilation_h, dilation_w = conv->dilation_w;

    int pad_top = conv->pad_top, pad_bottom = conv->pad_bottom;
    int pad_left = conv->pad_left, pad_right = conv->pad_right;

    int ksize = Hk * Wk;

    const int VEC_NLANES = 32;
    int padded_ksize = ((ksize + VEC_NLANES-1) / VEC_NLANES) * VEC_NLANES;

    const float *inp = input.ptr<float>();
    float *out = output.ptr<float>();

#if CV_TRY_AVX2 || CV_TRY_AVX || CV_TRY_RVV
    // TODO: remove the following limitation, need change code in conv_depthwise.simd.hpp.
    bool canRunOpt = Wi >= 16 + dilation_w*(Wk - 1) && !fusedAdd;
#endif
    std::vector<int> ofstab_(3 * ksize, 0);
    int *ofstab = ofstab_.data();
    int *yxtab = ofstab + ksize;

    for (int k = 0; k < ksize; k++)
    {
        int y = k < ksize ? k / Wk : 0;
        int x = k < ksize ? k % Wk : 0;
        int dy = y * dilation_h, dx = x * dilation_w;
        yxtab[k * 2] = dy;
        yxtab[k * 2 + 1] = dx;
        ofstab[k] = dy * Wi + dx;
    }

    const float *weights0 = conv->weightsBufPtr, *bias = conv->biasBuf.data();
    const float* relu = reluslope.data();
    CV_Assert(ksize > 1 || (pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0));

    parallel_for_(Range(0, N * C), [&](const Range &r0) {
    for (int nc = r0.start; nc < r0.end; nc++)
    {
        int c = nc % C;
        const float *inptr0 = inp + inp_planesize * nc;
        float *outptr0 = out + out_planesize * nc;

        const float *weights = weights0 + c * padded_ksize;

        if (conv_dim == CONV_2D)
        {
#if CV_TRY_AVX2
            if(canRunOpt && conv->useAVX2)
                opt_AVX2::fastDepthwiseConv(weights, Hk, Wk, stride_h, stride_w, dilation_h, dilation_w,
                                            pad_top, pad_left, bias, relu, inptr0, Hi, Wi, outptr0, c, H0, W0);
            else
#endif
#if CV_TRY_AVX
            if(canRunOpt && conv->useAVX)
                opt_AVX::fastDepthwiseConv(weights, Hk, Wk, stride_h, stride_w, dilation_h, dilation_w,
                                            pad_top, pad_left, bias, relu, inptr0, Hi, Wi, outptr0, c, H0, W0);
            else
#endif
#if CV_TRY_RVV
            if(canRunOpt && conv->useRVV)
                opt_RVV::fastDepthwiseConv(weights, Hk, Wk, stride_h, stride_w, dilation_h, dilation_w,
                                            pad_top, pad_left, bias, relu, inptr0, Hi, Wi, outptr0, c, H0, W0);
            else
#endif
            depthWiseBlockConv2D(weights, Hk, Wk, stride_h, stride_w, dilation_h, dilation_w,
                                 pad_top, pad_left, bias, relu, inptr0, Hi, Wi, outptr0, c, H0, W0, fusedAdd);
        }
        else // conv_dim == CONV_1D, spatial branch for depth-wise Conv1D.
        {
            depthWiseBlockConv1D(weights, Wk, stride_w, dilation_w, pad_left, bias, relu, inptr0, Wi, outptr0, c, W0, fusedAdd);
        }

        if (activ)
            activ->forwardSlice(outptr0, outptr0, (int) out_planesize, out_planesize, c, c+1);
    }});
}

/****************************************************************************************\
                                    SIMD and no-SIMD code for depthWiseBlockConv
\****************************************************************************************/

void depthWiseBlockConv2D(const float* wptr,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w,
                                 int dilation_h, int dilation_w,
                                 int pad_t, int pad_l,
                                 const float* biasptr, const float* relu,
                                 const float* inptr_,
                                 int height, int width,
                                 float* outptr_,
                                 int out_d, int outH, int outW, bool fusedAdd)
{
    const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
            w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
            w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    const int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const float* imgptr0 = inptr_ + in_i*width;
        const float* imgptr1 = imgptr0 + dilation_h*width;
        const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
        float out, w00 = w00_, w01 = w01_, w02 = w02_;
        float w20 = w20_, w21 = w21_, w22 = w22_;
        if (in_i < 0)
        {
            w00 = w01 = w02 = 0.f;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            w20 = w21 = w22 = 0.f;
            imgptr2 = imgptr1;
        }

        float* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                  imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                  imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
            if (fusedAdd)
                out += outptr[0];
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[0] = out;
            out_j = 1;
        }

#if CV_SIMD128
        const int VEC_NLANES = 4;
        if ((stride_w == 1 || (stride_w == 2 && dilation_w == 1)) && (outW1 - out_j) >= VEC_NLANES)
        {
            v_float32x4 vw00 = v_setall_f32(w00);
            v_float32x4 vw01 = v_setall_f32(w01);
            v_float32x4 vw02 = v_setall_f32(w02);
            v_float32x4 vw10 = v_setall_f32(w10);
            v_float32x4 vw11 = v_setall_f32(w11);
            v_float32x4 vw12 = v_setall_f32(w12);
            v_float32x4 vw20 = v_setall_f32(w20);
            v_float32x4 vw21 = v_setall_f32(w21);
            v_float32x4 vw22 = v_setall_f32(w22);
            v_float32x4 z = v_setzero_f32();
            v_float32x4 vbias = v_setall_f32(bias);
            v_float32x4 vrc = v_setall_f32(relu_coeff);

            if (stride_w == 1)
            {
                for (; out_j < outW1; out_j += VEC_NLANES)
                {
                    // Tail processing.
                    if (out_j > outW1 - VEC_NLANES)
                    {
                        // If fusedAdd is true, what is stored in outptr is not a meaningless value,
                        // but the number being added. And we should avoid use tail processing in this case.
                        // Because the tail process will make some elements compute twice,
                        // which will lead to result errors.
                        if (fusedAdd)
                            break;
                        out_j = outW1 - VEC_NLANES;
                    }

                    int in_j = out_j * stride_w - pad_l;
                    v_float32x4 v00 = v_load(imgptr0 + in_j),
                            v01 = v_load(imgptr0 + in_j + dilation_w),
                            v02 = v_load(imgptr0 + in_j + dilation_w*2),
                            v10 = v_load(imgptr1 + in_j),
                            v11 = v_load(imgptr1 + in_j + dilation_w),
                            v12 = v_load(imgptr1 + in_j + dilation_w*2),
                            v20 = v_load(imgptr2 + in_j),
                            v21 = v_load(imgptr2 + in_j + dilation_w),
                            v22 = v_load(imgptr2 + in_j + dilation_w*2);

                    v_float32x4 vout = v00*vw00 + v01*vw01 + v02*vw02 +
                                     v10*vw10 + v11*vw11 + v12*vw12 +
                                     v20*vw20 + v21*vw21 + v22*vw22 + vbias;
                    if (fusedAdd)
                        vout = v_load(outptr + out_j) + vout;
                    if (relu)
                        vout = v_select(vout > z, vout, vout*vrc);
                    v_store(outptr + out_j, vout);
                }
            }
            else // (stride_w == 2 && dilation_w == 1)
            {
                for (; out_j < outW1; out_j += VEC_NLANES)
                {
                    // Tail processing.
                    if (out_j > outW1 - VEC_NLANES)
                    {
                        if (fusedAdd)
                            break;
                        out_j = outW1 - VEC_NLANES;
                    }

                    int in_j = out_j * stride_w - pad_l;

                    v_float32x4 v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    v_load_deinterleave(imgptr0 + in_j, v00, v01);
                    v_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    v_load_deinterleave(imgptr1 + in_j, v10, v11);
                    v_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    v_load_deinterleave(imgptr2 + in_j, v20, v21);
                    v_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    v_float32x4 vout = v00 * vw00 + v01 * vw01 + v02 * vw02 +
                            v10 * vw10 + v11 * vw11 + v12 * vw12 +
                            v20 * vw20 + v21 * vw21 + v22 * vw22 + vbias;

                    if (fusedAdd)
                        vout = v_load(outptr + out_j) + vout;
                    if (relu)
                        vout = v_select(vout > z, vout, vout*vrc);
                    v_store(outptr + out_j, vout);
                }
            }
        }
#endif

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                  imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                  imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
            if (fusedAdd)
                out += outptr[out_j];
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            float s0 = 1.f, s1 = 1.f, s2 = 1.f;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0.f;
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0.f;
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0.f;
            }
            out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                  imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                  imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
            if (fusedAdd)
                out += outptr[out_j];
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }
    }
}

void depthWiseBlockConv1D(const float* wptr,
                                 int kernel_w, int stride_w, int dilation_w, int pad_l,
                                 const float* biasptr, const float* relu,
                                 const float* inptr_, int width,
                                 float* outptr_,
                                 int out_d, int outW, bool fusedAdd)
{
    const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2];
    const int outW1 = min(outW, (width - dilation_w * (kernel_w - 1) + pad_l)/stride_w);
    float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

    int out_j = 0;
    const float* imgptr0 = inptr_;
    float out, w00 = w00_, w01 = w01_, w02 = w02_;
    float* outptr = outptr_;

    if (pad_l > 0)
    {
        out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 + bias;
        if (fusedAdd)
            out += outptr[0];
        if (relu)
            out = out > 0.f ? out : out*relu_coeff;
        outptr[0] = out;
        out_j = 1;
    }

#if CV_SIMD128
    const int VEC_NLANES = 4;
    if ((stride_w == 1 || (stride_w == 2 && dilation_w == 1)) && (outW1 - out_j) >= VEC_NLANES)
    {
        v_float32x4 vw00 = v_setall_f32(w00);
        v_float32x4 vw01 = v_setall_f32(w01);
        v_float32x4 vw02 = v_setall_f32(w02);
        v_float32x4 z = v_setzero_f32();
        v_float32x4 vbias = v_setall_f32(bias);
        v_float32x4 vrc = v_setall_f32(relu_coeff);

        if( stride_w == 1 )
        {
            for( ; out_j < outW1; out_j += VEC_NLANES )
            {
                // Tail processing.
                if (out_j + VEC_NLANES > outW1)
                {
                    if (fusedAdd)
                        break;
                    out_j = outW1 - VEC_NLANES;
                }

                int in_j = out_j * stride_w - pad_l;
                v_float32x4 v00 = v_load(imgptr0 + in_j),
                        v01 = v_load(imgptr0 + in_j + dilation_w),
                        v02 = v_load(imgptr0 + in_j + dilation_w*2);

                v_float32x4 vout = v00*vw00 + v01*vw01 + v02*vw02 + vbias;
                if (fusedAdd)
                    vout = v_load(outptr + out_j) + vout;
                if (relu)
                    vout = v_select(vout > z, vout, vout*vrc);
                v_store(outptr + out_j, vout);
            }
        }
        else // (stride_w == 2 && dilation_w == 1)
        {
            for( ; out_j < outW1; out_j += VEC_NLANES )
            {
                // Tail processing.
                if (out_j + VEC_NLANES > outW1)
                {
                    if (fusedAdd)
                        break;
                    out_j = outW1 - VEC_NLANES;
                }

                int in_j = out_j * stride_w - pad_l;

                v_float32x4 v00, v01, v02, unused;
                v_load_deinterleave(imgptr0 + in_j, v00, v01);
                v_load_deinterleave(imgptr0 + in_j + 2, v02, unused);

                v_float32x4 vout = v00 * vw00 + v01 * vw01 + v02 * vw02 + vbias;

                if (fusedAdd)
                    vout = v_load(outptr + out_j) + vout;

                if (relu)
                    vout = v_select(vout > z, vout, vout*vrc);
                v_store(outptr + out_j, vout);
            }
        }
    }
#endif

    for (; out_j < outW1; out_j++)
    {
        int in_j = out_j * stride_w - pad_l;
        out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 + bias;
        if (fusedAdd)
            out += outptr[out_j];
        if (relu)
            out = out > 0.f ? out : out*relu_coeff;
        outptr[out_j] = out;
    }

    for (; out_j < outW; out_j++ )
    {
        int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
        float s0 = 1.f, s1 = 1.f, s2 = 1.f;
        if (in_j0 >= width)
        {
            in_j0 = 0;
            s0 = 0.f;
        }
        if (in_j1 >= width)
        {
            in_j1 = 0;
            s1 = 0.f;
        }
        if (in_j2 >= width)
        {
            in_j2 = 0;
            s2 = 0.f;
        }
        out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 + bias;
        if (fusedAdd)
            out += outptr[out_j];
        if (relu)
            out = out > 0.f ? out : out*relu_coeff;
        outptr[out_j] = out;
    }
}


}} // namespace cv::dnn
