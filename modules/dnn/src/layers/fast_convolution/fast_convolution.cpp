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
        InputArray _weightsMat,
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
    Mat weightsMat = _weightsMat.getMat();
    auto wShape = shape(weightsMat);
    const size_t wstep = weightsMat.step1();

#if CV_NEON // For now, winograd is ARM platform only.
    if (ngroups == 1 && Hk ==3 && Wk == 3 && stride_x == 1 && stride_y == 1 &&
        dilation_x == 1 && dilation_y ==1 && K >= 16 && C >= 16)
        conv->ifWinograd63 = true;
#else
    conv->ifWinograd63 = false;
#endif

    float *srcWeights = (float *)weightsMat.data;
    if (ngroups > 1 && ngroups == K && ngroups == C)
    {
        // for depth-wise convolutions on NCHW data we just preserve the weights in KCHW layout,
        // but add some padding to make the weights array layout more SIMD-friendly
        int ksize = Hk*Wk;

        // this code aims to let memory fit with vector size.
        int padded_ksize = ((ksize + FAST_VEC_NLANES-1) / FAST_VEC_NLANES) * FAST_VEC_NLANES;
        int nweights = C*padded_ksize;
        conv->weightsBuf.reserve(nweights);
        float* weightsBufPtr = conv->weightsBuf.data();
        memset(weightsBufPtr, 0, nweights*sizeof(weightsBufPtr[0]));
        for(int c = 0; c < C; c++)
        {
            for (int k = 0; k < ksize; k++)
                weightsBufPtr[c*padded_ksize + k] = srcWeights[c*wstep + k];
        }
    }
    else
    {
        // The weights are packed as
        // ngroups x (ceil((K/ngroups)/CONV_MR)*CONV_MR) x (Cg*Hk*Wk) x CONV_MR tensor
        int Kg = K/ngroups, Cg = max(C/ngroups, 1);
        int numStripsMR = (Kg + CONV_MR - 1) / CONV_MR;
        int Kg_aligned = numStripsMR * CONV_MR;
        int HkWkCg = Hk*Wk*Cg;
        size_t nweights = ngroups*Kg_aligned*HkWkCg;
        conv->weightsBuf.reserve(nweights);
        float* weightsBufPtr = conv->weightsBuf.data();
        memset(weightsBufPtr, 0, nweights*sizeof(weightsBufPtr[0]));

        // Pack the weight.
        parallel_for_(Range(0, ngroups * numStripsMR), [&](const Range& r0){
        for (int gsi = r0.start; gsi < r0.end; gsi++)
        {
            int g = gsi / numStripsMR;
            int si = gsi - g * numStripsMR;

            int startK = si * CONV_MR;
            CV_Assert(startK < Kg_aligned);

            float* packed_wptr = weightsBufPtr + HkWkCg * (startK + g * Kg_aligned);
            int dk = Kg - startK < CONV_MR ? Kg - startK : CONV_MR; // check if we need zero padding.

            int k_idx = g*Kg + startK;
            for(int yx = 0; yx < Hk*Wk; yx++) {
                for(int c = 0; c < Cg; c++, packed_wptr += CONV_MR)
                {
                    const float* wptr = srcWeights + wstep * k_idx + c*Hk*Wk + yx;
                    int k = 0;
                    for(; k < dk; k++, wptr += wstep)
                        packed_wptr[k] = *wptr;
                    for(; k < CONV_MR; k++)
                        packed_wptr[k] = 0.f;
                }
            }
        }});

        // Prepare Weight for Winograd F(6x6, 3x3)
        if (conv->ifWinograd63)
        {
            initWinograd63(conv, weightsMat, K, C);
        }
    }

    // store bias; append some zero's to make sure that
    // we can always read MR elements starting from any valid index
    {
        int k = 0, nbias = K + CONV_MR - 1;
        conv->biasBuf.reserve(nbias);
        float* biasBufPtr = conv->biasBuf.data();
        for(; k < K; k++)
            biasBufPtr[k] = srcBias ? srcBias[k] : 0.f;
        for(; k < nbias; k++)
            biasBufPtr[k] = 0.f;
    }
    return conv;
}

void runFastConv2d(InputArray _input, OutputArray _output, const Ptr<FastConv2d>& conv, int ntasks,
                   const Ptr<ActivationLayer>& actLayer, bool fusedAdd)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();

    Mat fusedAddMat;
    if (fusedAdd)
        fusedAddMat = _output.getMat();

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
        CV_Assert(fusedAddMat.empty()); // Depthwise-Convolution layer should not be followed by Add layer.
        return runDepthwise(input, output, conv, minval, maxval, activ, ifMinMaxAct);
    }

#if CV_NEON
    if (conv->ifWinograd63
         && inputShape[2] > 12 && inputShape[3] > 12
         && inputShape[2] < 120 && inputShape[3] < 120
         )
    {
        if (runWinograd63(input, fusedAddMat, output, conv, ntasks, minval, maxval, activ, ifMinMaxAct))
            return;
    }
#endif

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->K, Hk = conv->Hk, Wk = conv->Wk;
    int H0 = outputShape[2], W0 = outputShape[3], ngroups = conv->ngroups;
    int Cg = C/ngroups, Kg = K/ngroups;

    const size_t inp_planesize = (size_t)Hi*Wi;
    const size_t out_planesize = (size_t)H0*W0;

    int pad_top = conv->pad_top;
    int pad_left = conv->pad_left;

    int stride_y = conv->stride_y, stride_x = conv->stride_x;
    int dilation_y = conv->dilation_y, dilation_x = conv->dilation_x;

    int ksize = Hk * Wk;
    bool fast_1x1 = stride_x == 1 && stride_y == 1 && ksize == 1;
    int HkWkCg = Hk*Wk*Cg;

    enum { VEC_ALIGN = 8, DFT_TYPE = CV_32F }; // Memory alignment.
    int MAX_STRIPES = 2; // (56 + CONV_NR - 1)/CONV_NR;

    // Friendly to L1 cache
    const int K_BLOCK_SIZE = 32;
    const int C_BLOCK_SIZE = 256;

    int Kg_nblocks = (Kg + CONV_MR-1)/CONV_MR, Kg_aligned = Kg_nblocks * CONV_MR;

    int stripes_per_sample = (out_planesize + CONV_NR - 1) / CONV_NR;

    if (stripes_per_sample < ntasks * 4)
    {
        MAX_STRIPES = 1;
        stripes_per_sample = 1;
    }
    else
        Kg_nblocks = 1;

    int Kstripes = Kg_nblocks*stripes_per_sample;
    int nsubtasks = N*ngroups*Kstripes;

    size_t stripesize = CONV_NR * ksize * Cg;
    size_t taskbufsize = (stripesize + CONV_NR * K_BLOCK_SIZE) * MAX_STRIPES;
    size_t totalbufsize = taskbufsize * ntasks;

    AutoBuffer<float> inpbuf_all_;
    totalbufsize = alignSize(totalbufsize, VEC_ALIGN);
    inpbuf_all_.allocate(totalbufsize + VEC_ALIGN);
    float* inpbuf_all = alignPtr(inpbuf_all_.data(), (int)(VEC_ALIGN*sizeof(inpbuf_all_[0])));

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

    float* inp = input.ptr<float>();
    float* out = output.ptr<float>();
    float* fusedAddPtr0 = fusedAddMat.empty() ? 0 : fusedAddMat.ptr<float>();

    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        float* inpbuf_task = &inpbuf_all[taskbufsize * task_id];
        float* cbuf_task = inpbuf_task + stripesize * MAX_STRIPES;

        int ngs0 = (int)((size_t)nsubtasks * task_id / ntasks);
        int ngs1 = (int)((size_t)nsubtasks * (task_id+1) / ntasks);
        for (int subtask = ngs0; subtask < ngs1; )
        {
            int ng = subtask / Kstripes;
            int kyx0 = subtask - ng * Kstripes;
            int kyx1 = kyx0 + (ngs1 - subtask);
            int n = ng / ngroups, g = ng % ngroups; // ng - n * ngroups;
            size_t inp_plane_ofs = (size_t)(n * ngroups + g) * Cg * inp_planesize;
            kyx1 = kyx1 <= Kstripes ? kyx1 : Kstripes;
            subtask += kyx1 - kyx0;
            int k0, k1;
            int yx0, yx_limit, yx_block_limit = 0;

            if (stripes_per_sample == 1)
            {
                k0 = kyx0 * CONV_MR;
                k1 = kyx1 * CONV_MR;
                k1 = k1 <= Kg ? k1 : Kg;
                yx0 = 0;
                yx_limit = out_planesize;
            }
            else
            {
                k0 = 0;
                k1 = Kg;
                yx0 = kyx0 * CONV_NR;
                yx_limit = kyx1 * CONV_NR;
                yx_limit = yx_limit < out_planesize ? yx_limit : out_planesize;
            }

            for (; yx0 < yx_limit; yx0 = yx_block_limit)
            {
                // step 1. extract part of input tensor and represent it in zigzag form
                yx_block_limit = yx0 + CONV_NR * MAX_STRIPES;
                yx_block_limit = yx_block_limit < yx_limit ? yx_block_limit : yx_limit;

                int nstripes = (yx_block_limit - yx0 + CONV_NR - 1) / CONV_NR;
                int yx0_saved = yx0;

                CV_Assert(nstripes <= MAX_STRIPES);

                for (int stripe = 0; yx0 < yx_block_limit; stripe++, yx0 += CONV_NR)
                {
                    float* inpbuf = inpbuf_task + stripe * stripesize;
                    float* inptr = inp + inp_plane_ofs;

                    /*
                        1. pack the data. Copy the HkxWk CONV_NR-wide slices from
                           each feature plane of the input tensor to the input buffer.
                    */
                    if (fast_1x1)
                    {
                        int slice_len = yx_block_limit - yx0;
                        bool partial = slice_len < CONV_NR;
                        // Superfast branch for 1x1 convolutions with sy=sx=1.
                        // in this case each feature plane can be safely treated
                        // as 1D array, and we just extract next portion
                        // of CONV_NR elements from each feature plane and
                        // put it together.
                        inptr += yx0;
                        if (!partial)
                        {
                            // Make special branch where memcpy() is called with a constant buffer size.
                            // Compilers will likely unroll this loop properly.
                            for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR)
                                memcpy(inpbuf, inptr, CONV_NR*sizeof(inpbuf[0]));
                        }
                        else
                        {
                            for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR)
                            {
                                memcpy(inpbuf, inptr, slice_len * sizeof(inpbuf[0]));
                                memset(inpbuf + slice_len, 0, (CONV_NR - slice_len) * sizeof(inpbuf[0]));
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
                            for (; i < CONV_NR;)
                            {
                                float *inpbuf_ki = inpbuf + k * CONV_NR * Cg + i;
                                int yi = y0 * stride_y + dy - pad_top;
                                int xi = x0 * stride_x + dx - pad_left;

                                if ((unsigned) yi < (unsigned) Hi && (unsigned) xi < (unsigned) Wi)
                                {
                                    const float *inptr_ki = inptr + yi * Wi + xi;
                                    if (i + 8 <= CONV_NR && x0 + 8 <= W0 && xi + stride_x * 8 <= Wi)
                                    {
                                        if (stride_x == 1)
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[1];
                                                float t2 = inptr_ki[2], t3 = inptr_ki[3];
                                                float t4 = inptr_ki[4], t5 = inptr_ki[5];
                                                float t6 = inptr_ki[6], t7 = inptr_ki[7];
                                                inpbuf_ki[0] = t0; inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2; inpbuf_ki[3] = t3;
                                                inpbuf_ki[4] = t4; inpbuf_ki[5] = t5;
                                                inpbuf_ki[6] = t6; inpbuf_ki[7] = t7;
                                            }
                                        }
                                        else
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[stride_x];
                                                float t2 = inptr_ki[stride_x*2], t3 = inptr_ki[stride_x*3];
                                                float t4 = inptr_ki[stride_x*4], t5 = inptr_ki[stride_x*5];
                                                float t6 = inptr_ki[stride_x*6], t7 = inptr_ki[stride_x*7];
                                                inpbuf_ki[0] = t0; inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2; inpbuf_ki[3] = t3;
                                                inpbuf_ki[4] = t4; inpbuf_ki[5] = t5;
                                                inpbuf_ki[6] = t6; inpbuf_ki[7] = t7;
                                            }
                                        }
                                        i += 8;
                                        x0 += 8;
                                    }
                                    else if (i + 4 <= CONV_NR && x0 + 4 <= W0 && xi + stride_x * 4 <= Wi)
                                    {
                                        if (stride_x == 1)
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[1];
                                                float t2 = inptr_ki[2], t3 = inptr_ki[3];
                                                inpbuf_ki[0] = t0; inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2; inpbuf_ki[3] = t3;
                                            }
                                        }
                                        else
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[stride_x];
                                                float t2 = inptr_ki[stride_x*2], t3 = inptr_ki[stride_x*3];
                                                inpbuf_ki[0] = t0; inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2; inpbuf_ki[3] = t3;
                                            }
                                        }
                                        i += 4;
                                        x0 += 4;
                                    }
                                    else
                                    {
                                        for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            *inpbuf_ki = *inptr_ki;
                                        i++;
                                        x0++;
                                    }
                                }
                                else
                                {
                                    for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR)
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

                yx0 = yx0_saved;
                float* weights = conv->weightsBuf.data() + g * Kg_aligned * HkWkCg;
                const float* biasptr = conv->biasBuf.data() + Kg * g;
                int ldc = nstripes * CONV_NR;

                // 2. do convolution, compute Kg x (yx_block_limit - yx0) part of the output tensor
                for (int k0_block = k0; k0_block < k1; k0_block += K_BLOCK_SIZE)
                {
                    int k1_block = k0_block + K_BLOCK_SIZE < k1 ? k0_block + K_BLOCK_SIZE : k1;
                    for (int c0 = 0; c0 < HkWkCg; c0 += C_BLOCK_SIZE)
                    {
                        int c1 = c0 + C_BLOCK_SIZE < HkWkCg ? c0 + C_BLOCK_SIZE : HkWkCg;
                        for (int stripe = 0; stripe < nstripes; stripe++)
                        {
                            float* wptr = weights + k0_block*HkWkCg + c0*CONV_MR;
                            const float* inptr = inpbuf_task + stripe*stripesize + c0 * CONV_NR;
                            float* cptr = cbuf_task + stripe * CONV_NR;
                            for (int k = k0_block; k < k1_block; k += CONV_MR,
                                    wptr += HkWkCg * CONV_MR, cptr += CONV_MR * ldc)
                            {
#if CV_TRY_AVX2
                                if (conv->useAVX2)
                                    opt_AVX2::convBlock_AVX2(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0);
                                else
#endif
#if CV_TRY_NEON
                                if (conv->useNEON)
                                    opt_NEON::convBlock_NEON(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0);
                                else
#endif
                                    convBlock(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0);
                            }
                        }
                    }

                    size_t outofs = ((n*ngroups + g) * Kg + k0_block) * out_planesize + yx0;
                    int out_width = yx_block_limit - yx0;
                    const float* cptr = cbuf_task;

                    float* outptr = out + outofs;
                    const float* pbptr = fusedAddPtr0 ? fusedAddPtr0 + outofs : 0;

                    for (int k = k0_block; k < k1_block; k++,
                            cptr += ldc, outptr += out_planesize,
                            pbptr += (pbptr ? out_planesize : 0))
                    {
                        float biasval = biasptr[k];
                        int j = 0;
#if CV_SIMD128
                        v_float32x4 vbias = v_setall_f32(biasval), vmax = v_setall_f32(maxval), vmin = v_setall_f32(minval);
                        if (pbptr)
                        {
                            for (; j + 7 < out_width; j += 8)
                            {
                                v_float32x4 v0 = v_load(cptr + j) + vbias;
                                v_float32x4 v1 = v_load(cptr + j + 4) + vbias;

                                v0 += v_load(pbptr + j);
                                v1 += v_load(pbptr + j + 4);

                                if (ifMinMaxAct)
                                {
                                    v0 = v_min(v_max(v0, vmin), vmax);
                                    v1 = v_min(v_max(v1, vmin), vmax);
                                }

                                v_store(outptr + j, v0);
                                v_store(outptr + j + 4, v1);
                            }
                        }
                        else
                        {
                            for (; j + 7 < out_width; j += 8)
                            {
                                v_float32x4 v0 = v_load(cptr + j) + vbias;
                                v_float32x4 v1 = v_load(cptr + j + 4) + vbias;

                                if (ifMinMaxAct)
                                {
                                    v0 = v_min(v_max(v0, vmin), vmax);
                                    v1 = v_min(v_max(v1, vmin), vmax);
                                }

                                v_store(outptr + j, v0);
                                v_store(outptr + j + 4, v1);
                            }
                        }
#endif
                        if (pbptr) {
                            for (; j < out_width; j++)
                            {
                                float v = cptr[j] + biasval;
                                v += pbptr[j];
                                if (ifMinMaxAct)
                                    v = std::min(std::max(v, minval), maxval);
                                outptr[j] = v;
                            }
                        }
                        else
                        {
                            for (; j < out_width; j++)
                            {
                                float v = cptr[j] + biasval;

                                if (ifMinMaxAct)
                                    v = std::min(std::max(v, minval), maxval);
                                outptr[j] = v;
                            }
                        }

                        if (activ)
                            activ->forwardSlice(outptr, outptr, out_width, out_planesize, Kg * g + k, Kg * g + k + 1);
                    }
                }
            }
        }
    }
    });
}
}} // namespace cv::dnn
