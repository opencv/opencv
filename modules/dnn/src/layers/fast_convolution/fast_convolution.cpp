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
#include "fast_convolution.simd.hpp"

namespace cv { namespace dnn {

Ptr<FastConv2d> initFastConv2d(
        int ngroups,
        int K, int C, int Hk, int Wk,
        int stride_x, int stride_y,
        int dilation_x, int dilation_y,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end,
        float* srcWeights,
        float* srcBias)
{
    Ptr<FastConv2d> conv = makePtr<FastConv2d>();

    CV_Assert(ngroups > 0 && K > 0 && C > 0 && K % ngroups == 0);
    CV_Assert(Hk > 0 && Wk > 0);
    CV_Assert(stride_y > 0 && stride_x > 0);
    CV_Assert(dilation_y > 0 && dilation_x > 0);

    conv->K = K; conv->C = C; conv->Hk = Hk; conv->Wk = Wk;  // [K, iC, kH, kW]
    conv->stride_y = stride_y;
    conv->stride_x = stride_x;
    conv->dilation_y = dilation_y;
    conv->dilation_x = dilation_x;

    conv->ngroups = ngroups;
    conv->pad_top = pads_begin[0];
    conv->pad_bottom = pads_end[0];
    conv->pad_left = pads_begin[1];
    conv->pad_right = pads_end[1];

    // store bias; append some zero's to make sure that
    // we can always read FAST_CONV_MR elements starting from any valid index
    {
        int k = 0, nbias = K + FAST_CONV_MR-1;
        conv->biasBuf.reserve(nbias);
        float* biasBufPtr = conv->biasBuf.data();
        for(; k < K; k++)
            biasBufPtr[k] = srcBias ? srcBias[k] : 0.f;
        for(; k < nbias; k++)
            biasBufPtr[k] = 0.f;
    }

#if CV_NEON // For now, winograd is ARM platform only.
    if (ngroups == 1 && Hk ==3 && Wk == 3 && stride_x == 1 && stride_y == 1 && dilation_x == 1 && dilation_y ==1
        && K >= 16 && C >= 16 )
        conv->ifWinograd63 = true;
#else
    conv->ifWinograd63 = false;
#endif

    if (ngroups > 1 && ngroups == K && ngroups == C)
    {
        // for depth-wise convolutions on NCHW data we just preserve the weights in KCHW layout,
        // but add some padding to make the weights array layout more SIMD-friendly
        int ksize = Hk*Wk;
        int padded_ksize = ((ksize + FAST_VEC_NLANES-1)/FAST_VEC_NLANES)*FAST_VEC_NLANES;  // this code aims to let memory fit with vector size.
        int nweights = C*padded_ksize;
        conv->weightsBuf.reserve(nweights);
        float* weightsBufPtr = conv->weightsBuf.data();
        memset(weightsBufPtr, 0, nweights*sizeof(weightsBufPtr[0]));
        for(int c = 0; c < C; c++)
        {
            for (int k = 0; k < ksize; k++)
                weightsBufPtr[c*padded_ksize + k] = srcWeights[c*ksize + k];
        }
    }
    else
    {
        // The weights are packed as
        // ngroups x (ceil((K/ngroups)/FAST_CONV_MR)*FAST_CONV_MR) x (Cg*Hk*Wk) x FAST_CONV_MR tensor
        int Kg = K/ngroups, Cg = max(C/ngroups, 1);
        int Kg_aligned = ((Kg + FAST_CONV_MR - 1)/FAST_CONV_MR)*FAST_CONV_MR;
        size_t nweights = ngroups*Kg_aligned*Cg*Hk*Wk;
        conv->weightsBuf.reserve(nweights);
        float* weightsBufPtr = conv->weightsBuf.data();
        memset(weightsBufPtr, 0, nweights*sizeof(weightsBufPtr[0]));
        float* packed_wptr = weightsBufPtr;

        // pack the weight.
        for(int g = 0; g < ngroups; g++)
        {
            for(int k0 = 0; k0 < Kg_aligned; k0 += FAST_CONV_MR)
            {
                int dk = Kg - k0 < FAST_CONV_MR ? Kg - k0 : FAST_CONV_MR;
                for(int c = 0; c < Cg; c++)
                {
                    for(int yx = 0; yx < Hk*Wk; yx++, packed_wptr += FAST_CONV_MR)
                    {
                        const float* wptr = srcWeights + ((g*Kg + k0)*Cg + c)*Hk*Wk + yx;
                        int k = 0;
                        for(; k < dk; k++, wptr += Cg*Hk*Wk)
                            packed_wptr[k] = *wptr;
                        for(; k < FAST_CONV_MR; k++)
                            packed_wptr[k] = 0.f;
                    }
                }
            }
        }

        // Prepare Weight for Winograd F(6x6, 3x3)
        if (conv->ifWinograd63)
        {
            initWinograd63(conv, srcWeights, K, C);
        }
    }
    return conv;
}

static void packInput(float* inpbuf, const float* inptr, int* yxtab, int ksize, int Cg, int Hi, int Wi, int W0,
                         int pad_top, int pad_left, int stride_x, int stride_y, int yx0, int slice_len,
                         bool fast_1x1, bool partial0, bool s1d1p0, bool s1d1)
{
    const size_t inp_planesize = (size_t)Hi*Wi;

    if (fast_1x1)
    {
        /*
           super-fast branch for 1x1 convolutions with sy=sx=1.
           in this case each feature plane can be safely treated
           as 1D array and we just extract next portion
           of FAST_CONV_NR elements from each feature plane and
           put it together.
        */
        inptr += yx0;
        if (!partial0)
        {
            // Make special branch where memcpy() is called with a constant buffer size.
            // Compilers will likely unroll this loop properly.
            for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += FAST_CONV_NR)
                memcpy(inpbuf, inptr, FAST_CONV_NR * sizeof(inpbuf[0]));
        }
        else
        {
            for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += FAST_CONV_NR)
            {
                memcpy(inpbuf, inptr, slice_len * sizeof(inpbuf[0]));
                memset(inpbuf + slice_len, 0, (FAST_CONV_NR - slice_len) * sizeof(inpbuf[0]));
            }
        }
    }
    else if (s1d1p0)
    {
        /*
         slower, but still fast branch for sy=sx=1, dy=dx=1 and without padding,
         in this case we copy data from input tensors by chunks.
         */
        for (int c = 0; c < Cg; c++)
        {
            float *inpbuf_c = inpbuf + c * (FAST_CONV_NR * ksize);
            const float *inptr_c = inptr + c * inp_planesize;

            for (int k = 0; k < ksize; k++)
            {
                int y0 = yx0 / W0, x0 = yx0 % W0;
                int yi = y0 + yxtab[k * 2], xi = x0 + yxtab[k * 2 + 1];
                float *inpbuf_k = inpbuf_c + k * FAST_CONV_NR;
                int xi_0 = yxtab[k * 2 + 1];

                int i = 0;
                for (; i < slice_len;)
                {
                    const float *inptr_k = inptr_c + yi * Wi + xi;
                    int copy_len = std::min(slice_len - i, W0 - x0);
                    int di_z = (slice_len == i + copy_len) ? FAST_CONV_NR - slice_len : 0;

                    memcpy(inpbuf_k + i,
                           inptr_k,
                           copy_len * sizeof(inpbuf_k[0]));

                    memset(inpbuf_k + i + copy_len,
                           0, di_z * sizeof(inpbuf_k[0]));

                    i += copy_len;
                    x0 = 0;
                    xi = xi_0;
                    yi++;
                }
            }
        }
    }
    else if (s1d1)
    {
        /*
         slower, but still fast branch for sy=sx=1, dy=dx=1.
         in this case we copy data from input tensors by chunks and
         interleave the data in inpbuf with 0's
         (that correspond to the padding elements) when necessary
         */
        int y0 = yx0 / W0, x0 = yx0 % W0;
        for (int c = 0; c < Cg; c++)
        {
            float *inpbuf_c = inpbuf + c * (FAST_CONV_NR * ksize);
            const float *inptr_c = inptr + c * inp_planesize;

            for (int k = 0; k < ksize; k++)
            {
                int x0_tmp = x0;

                int xi_0 = yxtab[k * 2 + 1] - pad_left;

                int yi = y0 + yxtab[k * 2] - pad_top, xi = x0_tmp + xi_0;
                float *inpbuf_k = inpbuf_c + k * FAST_CONV_NR;

                int i = 0;
                for (; i < slice_len;) {
                    int copyLen = std::min(slice_len - i, W0 - x0_tmp);

                    int di_z = (i + copyLen == slice_len) ? FAST_CONV_NR - slice_len
                                                          : 0; // The final padding.
                    // pad_top or pad bottom
                    if (yi < 0 || yi > Hi - 1)
                    {
                        memset(inpbuf_k + i,
                               0, (copyLen + di_z) * sizeof(inpbuf_k[0]));
                        i += copyLen + di_z;
                    }
                    else
                    {
                        int x_pad_left = 0, x_pad_right = 0;

                        // pad_left
                        if (xi < 0)
                        {
                            x_pad_left = std::min(-xi, copyLen);
                            xi = 0;
                            copyLen -= x_pad_left;
                        }

                        memset(inpbuf_k + i,
                               0, x_pad_left * sizeof(inpbuf_k[0]));
                        i += x_pad_left;

                        // pad right
                        if (xi + copyLen > Wi)
                        {
                            if (xi > Wi)
                            {
                                x_pad_right = copyLen;
                                copyLen = 0;
                            }
                            else
                            {
                                x_pad_right = std::min(xi + copyLen - Wi, copyLen);
                                copyLen -= x_pad_right;
                            }
                        }

                        CV_Assert(copyLen >= 0);

                        const float *inptr_k = inptr_c + yi * Wi + xi;
                        memcpy(inpbuf_k + i,
                               inptr_k,
                               copyLen * sizeof(inpbuf_k[0]));

                        i += copyLen;

                        // pad_right and the final padding.
                        memset(inpbuf_k + i,
                               0, (di_z + x_pad_right) * sizeof(inpbuf_k[0]));
                        i += x_pad_right + di_z;
                    }

                    x0_tmp = 0;
                    xi = xi_0;
                    yi++;
                }
            }
        }
    }
    else
    {
        int y0_ = yx0 / W0, x0_ = yx0 - y0_ * W0;
        for (int k = 0; k < ksize; k++)
        {
            int dy = yxtab[k * 2], dx = yxtab[k * 2 + 1];
            int i = 0, y0 = y0_, x0 = x0_;
            for (; i < FAST_CONV_NR;)
            {
                float *inpbuf_ki = inpbuf + k * FAST_CONV_NR + i;
                int yi = y0 * stride_y + dy - pad_top;
                int xi = x0 * stride_x + dx - pad_left;

                if ((unsigned) yi < (unsigned) Hi &&
                    (unsigned) xi < (unsigned) Wi)
                {
                    const float *inptr_ki = inptr + yi * Wi + xi;
                    if (i + 4 <= FAST_CONV_NR && x0 + 4 <= W0 && xi + stride_x * 4 <= Wi)
                    {
                        if (stride_x == 2) {
                            for (int c = 0; c < Cg; c++, inpbuf_ki += FAST_CONV_NR *
                                                                      ksize, inptr_ki += inp_planesize)
                            {
                                float t0 = inptr_ki[0], t1 = inptr_ki[2];
                                float t2 = inptr_ki[4], t3 = inptr_ki[6];
                                inpbuf_ki[0] = t0;
                                inpbuf_ki[1] = t1;
                                inpbuf_ki[2] = t2;
                                inpbuf_ki[3] = t3;
                            }
                        }
                        else
                        {
                            for (int c = 0; c < Cg; c++, inpbuf_ki += FAST_CONV_NR *
                                                                      ksize, inptr_ki += inp_planesize)
                            {
                                float t0 = inptr_ki[0], t1 = inptr_ki[stride_x];
                                float t2 = inptr_ki[stride_x * 2], t3 = inptr_ki[stride_x * 3];
                                inpbuf_ki[0] = t0;
                                inpbuf_ki[1] = t1;
                                inpbuf_ki[2] = t2;
                                inpbuf_ki[3] = t3;
                            }
                        }
                        i += 4;
                        x0 += 4;
                    }
                    else
                    {
                        for (int c = 0; c < Cg; c++, inpbuf_ki += FAST_CONV_NR *
                                                                  ksize, inptr_ki += inp_planesize)
                            *inpbuf_ki = *inptr_ki;
                        i++;
                        x0++;
                    }
                }
                else
                {
                    for (int c = 0; c < Cg; c++, inpbuf_ki += FAST_CONV_NR * ksize)
                        inpbuf_ki[0] = 0.f;
                    i++;
                    x0++;
                }
                int mask = x0 >= W0;
                y0 += mask;
                x0 &= mask - 1;
            }
        }
    }
}

static void matMulCompute(float* outptr0, float* inpbuf_task, float* cbuf, const Ptr<FastConv2d>& conv, int HkWkCg,
                          int k0, int k1, int yx0, int yx1, size_t out_planesize, int g, int Kg, int Kg_aligned,
                          bool partial0, ActivationLayer*& activ, float minval, float maxval, bool ifMinMaxAct)
{
    int outstep0 = out_planesize;

    for (int k = k0; k < k1; k += FAST_CONV_MR, outptr0 += outstep0 * FAST_CONV_MR)
    {
        int dk = Kg - k < FAST_CONV_MR ? Kg - k : FAST_CONV_MR;
        bool partial = partial0 || dk < FAST_CONV_MR;
        float *outptr = outptr0;

        int outstep = outstep0;
        if (partial)
        {
            outptr = cbuf;
            outstep = FAST_CONV_NR;
        }


#if CV_TRY_AVX2
        if (conv->useAVX2)
            opt_AVX2::convBlock_AVX2( HkWkCg, conv->weightsBuf.data() + (g * Kg_aligned + k) * HkWkCg,
                                  inpbuf_task, outptr, outstep, conv->biasBuf.data() + Kg * g + k,
                                  minval, maxval, ifMinMaxAct);
        else
#endif
#if CV_TRY_NEON
        if (conv->useNEON)
            opt_NEON::convBlock_NEON(HkWkCg, conv->weightsBuf.data() + (g * Kg_aligned + k) * HkWkCg,
                                 inpbuf_task, outptr, outstep, conv->biasBuf.data() + Kg * g + k,
                                 minval, maxval, ifMinMaxAct);
        else
#endif
            convBlock(HkWkCg, conv->weightsBuf.data() + (g * Kg_aligned + k) * HkWkCg,
                            inpbuf_task, outptr, outstep, conv->biasBuf.data() + Kg * g + k,
                            minval, maxval, ifMinMaxAct);

        // activation
        if (activ)
            activ->forwardSlice(outptr, outptr, yx1 - yx0, outstep, Kg * g + k,
                                Kg * g + k + dk);

        if (partial)
        {
            for (int i = 0; i < dk; i++)
                memcpy(outptr0 + i * outstep0, cbuf + i * FAST_CONV_NR,
                       (yx1 - yx0) * sizeof(cbuf[0]));
        }
    }
}

void runFastConv2d(InputArray _input, OutputArray _output,
                   const Ptr<FastConv2d>& conv, int ntasks, const Ptr<ActivationLayer>& actLayer)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);
    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    ActivationLayer* activ = 0;
    float minval = -FLT_MAX, maxval = FLT_MAX;
    bool ifMinMaxAct = false;
    if (actLayer)
    {
        Ptr<ReLULayer> activ_relu = actLayer.dynamicCast<ReLULayer>();
        Ptr<ReLU6Layer> activ_relu6 = actLayer.dynamicCast<ReLU6Layer>();

        if (!activ_relu.empty())
        {
            if (activ_relu->negativeSlope == 0.0f)
            {
                minval = 0.0f;
                ifMinMaxAct = true;
                activ = nullptr;
            }
            else // Leaky ReLU
            {
                activ = actLayer.get();
            }
        }
        else if (!activ_relu6.empty())
        {
            minval = activ_relu6->minValue;
            maxval = activ_relu6->maxValue;

            ifMinMaxAct = true;
            activ = nullptr;
        }
        else
            activ = actLayer.get();
    }
    else
        activ = nullptr;

    if (conv->ngroups  > 1 && conv->ngroups == conv->K && conv->ngroups == conv->C)
    {
        return runDepthwise(input, output, conv, minval, maxval, activ, ifMinMaxAct);
    }

#if CV_NEON
    if ( conv->ifWinograd63
         && inputShape[2] > 12 && inputShape[3] > 12
         && inputShape[2] < 120 && inputShape[3] < 120 )
    {
        // In general, for winograd branch, more cores will give better performance.
        int maxNumThread = std::max(getNumThreads(), 1);
        if (runWinograd63(input, output, conv, maxNumThread, minval, maxval, activ, ifMinMaxAct))
            return;
    }
#endif

    float* inp = input.ptr<float>();
    float* out = output.ptr<float>();

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->K, Hk = conv->Hk, Wk = conv->Wk;
    int H0 = outputShape[2], W0 = outputShape[3], ngroups = conv->ngroups;         // ngroups
    int Cg = C/ngroups, Kg = K/ngroups;
    int Kg_nblocks = (Kg + FAST_CONV_MR-1)/FAST_CONV_MR, Kg_aligned = Kg_nblocks*FAST_CONV_MR; // align to MR

    const size_t inp_planesize = (size_t)Hi*Wi;
    const size_t out_planesize = (size_t)H0*W0;

    int pad_top = conv->pad_top, pad_bottom = conv->pad_bottom;
    int pad_left = conv->pad_left;
    int pad_right = conv->pad_right;

    int stride_y = conv->stride_y, stride_x = conv->stride_x;
    int dilation_y = conv->dilation_y, dilation_x = conv->dilation_x;

    int ksize = Hk * Wk;
    bool s1d1 = stride_x == 1 && stride_y == 1 && dilation_x == 1 && dilation_y == 1;
    bool s1d1p0 = s1d1 && pad_top == 0 && pad_left ==0 && pad_bottom == 0 && pad_right == 0;
    bool fast_1x1 = stride_x == 1 && stride_y == 1 && ksize == 1;
    int HkWkCg = Hk*Wk*Cg;

    enum { VEC_ALIGN = 8, DFT_TYPE = CV_32F };
    size_t taskbufsize = FAST_CONV_NR*HkWkCg; // input buffer
    size_t taskbufsizeOutput = FAST_CONV_NR * FAST_CONV_MR;
    size_t inputbufsize = 0;
    size_t outbufsize = ntasks * taskbufsizeOutput;

    int stripes_per_sample = (out_planesize + FAST_CONV_NR - 1)/FAST_CONV_NR; // align to NR
    size_t hw_task = stripes_per_sample;
    size_t hw_aligned = stripes_per_sample * FAST_CONV_NR;

    bool separatedLoop = false;

    if (stripes_per_sample < 4 * ntasks)
    {
        // If stripes_per_sample is small, we parallelize on K (output channel).
        stripes_per_sample = 1;

        // Separated Parallelloop could save much time in packing input data. But it may cost more memory, we use it when batch size is 1.
        if (N == 1)
        {
            separatedLoop = true;
            inputbufsize = ngroups * hw_aligned * HkWkCg;
        }

        if (!separatedLoop)
        {
            inputbufsize = taskbufsize * ntasks;
        }
    }
    else
    {
        // If stripes_per_sample is big, we parallelize on H0*W0.
        Kg_nblocks = 1;
        inputbufsize = taskbufsize * ntasks;
    }

    int Kstripes = Kg_nblocks*stripes_per_sample;
    int nsubtasks = N*ngroups*Kstripes;

    AutoBuffer<float> inpbuf_all_, outputbuf_;
    inputbufsize = alignSize(inputbufsize, VEC_ALIGN);
    inpbuf_all_.allocate(inputbufsize + VEC_ALIGN);
    float* inpbuf_all = alignPtr(inpbuf_all_.data(), (int)(VEC_ALIGN*sizeof(float)));

    outbufsize = alignSize(outbufsize, VEC_ALIGN);
    outputbuf_.allocate(outbufsize + VEC_ALIGN);
    float* output_buf = alignPtr(outputbuf_.data(), (int)(VEC_ALIGN*sizeof(float)));

    std::vector<int> ofstab_(Hk*Wk*3, 0);
    int* ofstab = ofstab_.data();
    int* yxtab = ofstab + Hk*Wk;

    for (int y = 0; y < Hk; y++)
        for( int x = 0; x < Wk; x++)
        {
            int k = y*Wk + x;
            int dy = y*dilation_y, dx = x*dilation_x;
            yxtab[k*2] = dy;
            yxtab[k*2+1] = dx;
            ofstab[k] = dy*Wi + dx;
        }

    if (ksize == 1)
    {
        CV_Assert(pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0);
        CV_Assert(stride_x != 1 || stride_y != 1 || (H0 == Hi && W0 == Wi));
    }

    if (separatedLoop)
    {
        // For now this branch only handles batch size = 1. Maybe we could support batch size < 10 in the future.
        // Pack Input data
        parallel_for_(Range(0, ngroups * hw_task), [&](const Range& r0)
        {
            for (int nhwi = r0.start; nhwi < r0.end; nhwi++)
            {
                int g = nhwi/hw_task;
                int hw_i = nhwi % hw_task;
                int hw0 = hw_i * FAST_CONV_NR;
                float* inpbuf = inpbuf_all + g * hw_aligned * HkWkCg + hw0 * HkWkCg;
                const float* inptr = inp + g * Cg * inp_planesize;
                bool partial0 = hw0 + FAST_CONV_NR > out_planesize? true: false;
                int slice_len = FAST_CONV_NR;

                if (partial0)
                    slice_len = out_planesize - hw0;

                packInput(inpbuf, inptr, yxtab, ksize, Cg, Hi, Wi, W0, pad_top, pad_left, stride_x, stride_y,
                          hw0, slice_len, fast_1x1, partial0, s1d1p0, s1d1);
            }
        });

        // Compute
        parallel_for_(Range(0, ntasks), [&](const Range& r0)
        {
            for (int task_id = r0.start; task_id < r0.end; task_id++)
            {
                float *cbuf = output_buf + task_id * taskbufsizeOutput;
                int ngs0 = (int) ((size_t) nsubtasks * task_id / ntasks);
                int ngs1 = (int) ((size_t) nsubtasks * (task_id + 1) / ntasks);
                for (int subtask = ngs0; subtask < ngs1;)
                {
                    int ng = subtask / Kstripes;
                    int kyx0 = subtask - ng * Kstripes;
                    int kyx1 = kyx0 + (ngs1 - subtask);
                    int n = ng / ngroups, g = ng - n * ngroups;

                    CV_Assert(n <= 1);

                    kyx1 = kyx1 <= Kstripes ? kyx1 : Kstripes; // Guarantee that maximum kyx1 is Kstripes.
                    subtask += kyx1 - kyx0;

                    int k0 = kyx0 * FAST_CONV_MR;
                    int k1 = kyx1 * FAST_CONV_MR;
                    k1 = k1 <= Kg ? k1 : Kg;


                    for (int yx0 = 0; yx0 < out_planesize; yx0 += FAST_CONV_NR)
                    {
                        float* inpbuf_task = inpbuf_all + g * hw_aligned * HkWkCg + yx0 * HkWkCg;
                        int yx1 = yx0 + FAST_CONV_NR;
                        yx1 = yx1 <= out_planesize ? yx1 : out_planesize;
                        int slice_len = yx1 - yx0;
                        bool partial0 = slice_len < FAST_CONV_NR;

                        int outstep0 = out_planesize;
                        size_t outofs = ((n * ngroups + g) * Kg + k0) * outstep0 + yx0;
                        float *outptr0 = out + outofs;

                        matMulCompute(outptr0, inpbuf_task, cbuf, conv, HkWkCg, k0, k1, yx0, yx1, out_planesize, g,
                                      Kg, Kg_aligned, partial0, activ, minval, maxval, ifMinMaxAct);
                    }
                }
            }
        });
    }
    else
    {
        parallel_for_(Range(0, ntasks), [&](const Range &r0) {
            for (int task_id = r0.start; task_id < r0.end; task_id++) {
                float *inpbuf_task = &inpbuf_all[taskbufsize * task_id];
                float *cbuf = output_buf + task_id * taskbufsizeOutput;
                int ngs0 = (int) ((size_t) nsubtasks * task_id / ntasks);
                int ngs1 = (int) ((size_t) nsubtasks * (task_id + 1) / ntasks);

                for (int subtask = ngs0; subtask < ngs1;)
                {
                    int ng = subtask / Kstripes;
                    int kyx0 = subtask - ng * Kstripes;
                    int kyx1 = kyx0 + (ngs1 - subtask);
                    int n = ng / ngroups, g = ng - n * ngroups;
                    size_t inp_plane_ofs = (size_t) (n * ngroups + g) * Cg * inp_planesize;
                    kyx1 = kyx1 <= Kstripes ? kyx1 : Kstripes; // Guarantee that maximum kyx1 is Kstripes.
                    subtask += kyx1 - kyx0;
                    int k0, k1;
                    int yx0, yx_limit;

                    if (stripes_per_sample == 1)
                    {
                        k0 = kyx0 * FAST_CONV_MR;
                        k1 = kyx1 * FAST_CONV_MR;
                        k1 = k1 <= Kg ? k1 : Kg;
                        yx0 = 0;
                        yx_limit = out_planesize;
                    }
                    else
                    {
                        k0 = 0;
                        k1 = Kg;
                        yx0 = kyx0 * FAST_CONV_NR;
                        yx_limit = kyx1 * FAST_CONV_NR;
                        yx_limit = yx_limit < out_planesize ? yx_limit : out_planesize;
                    }

                    for (; yx0 < yx_limit; yx0 += FAST_CONV_NR)
                    {
                        float *inpbuf = inpbuf_task;
                        const float *inptr = inp + inp_plane_ofs;
                        int yx1 = yx0 + FAST_CONV_NR;
                        yx1 = yx1 <= yx_limit ? yx1 : yx_limit;
                        int slice_len = yx1 - yx0;
                        bool partial0 = slice_len < FAST_CONV_NR;
                        packInput(inpbuf, inptr, yxtab, ksize, Cg, Hi, Wi, W0, pad_top, pad_left, stride_x, stride_y,
                                     yx0, slice_len, fast_1x1, partial0, s1d1p0, s1d1);

                        // 2. do convolution, compute Kg x (yx1 - yx0) part of the output tensor
                        int outstep0 = out_planesize;
                        size_t outofs = ((n * ngroups + g) * Kg + k0) * outstep0 + yx0;
                        float *outptr0 = out + outofs;

                        matMulCompute(outptr0, inpbuf_task, cbuf, conv, HkWkCg, k0, k1, yx0, yx1, out_planesize, g,
                                      Kg, Kg_aligned, partial0, activ, minval, maxval, ifMinMaxAct);
                    }
                }
            }
        });
    }
}

}} // namespace cv::dnn