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
enum { VEC_ALIGN = 32, DFT_TYPE = CV_32F }; // Memory alignment.
Ptr<FastConv> initFastConv(
        InputArray _weightsMat,
        float* srcBias,
        int ngroups,
        int K, int C,
        const std::vector<size_t>& kernel_size,
        const std::vector<size_t>& strides,
        const std::vector<size_t>& dilations,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end,
        int conv_dim,
        bool useWinograd)
{
    Ptr<FastConv> conv = makePtr<FastConv>();

    CV_Assert(ngroups > 0 && K > 0 && C > 0 && K % ngroups == 0);

    // Weight shape, [K, C, Dk, Hk, Wk] for Conv3D, [K, C, Hk, Wk] for Conv2D, [K, C, Wk] for Conv1D.
    int Dk = conv_dim == CONV_3D ? (int)kernel_size[0] : 1;
    int Hk = conv_dim == CONV_1D ? 1 : (int)kernel_size[kernel_size.size() - 2];
    int Wk = (int)kernel_size.back();
    int karea = Wk*Hk*Dk;

    conv->pad_front = conv_dim == CONV_3D ? (int)pads_begin[0] : 0;
    conv->pad_top = conv_dim == CONV_1D ? 0 : (int)pads_begin[pads_begin.size() - 2];
    conv->pad_left = (int)pads_begin.back();

    conv->pad_behind = conv_dim == CONV_3D ? (int)pads_end[0] : 0;
    conv->pad_bottom = conv_dim == CONV_1D ? 0 : (int)pads_end[pads_end.size() - 2];
    conv->pad_right = (int)pads_end.back();

    int stride_d = conv_dim == CONV_3D ? (int)strides[0] : 0;
    int stride_h = conv_dim == CONV_1D ? 0 : (int)strides[strides.size() - 2];
    int stride_w = (int)strides.back();

    int dilation_d = conv_dim == CONV_3D ? (int)dilations[0] : 1;
    int dilation_h = conv_dim == CONV_1D ? 1 : (int)dilations[dilations.size() - 2];
    int dilation_w = (int)dilations.back();

    CV_Assert(Dk > 0 && Hk > 0 && Wk > 0);
    CV_Assert(stride_d >= 0 && stride_h >= 0 && stride_w > 0);
    CV_Assert(dilation_d > 0 && dilation_h > 0 && dilation_w > 0);

    conv->K = K; conv->C = C; conv->Hk = Hk; conv->Wk = Wk, conv->Dk = Dk;

    conv->stride_d = stride_d;
    conv->stride_h = stride_h;
    conv->stride_w = stride_w;

    conv->dilation_d = dilation_d;
    conv->dilation_h = dilation_h;
    conv->dilation_w = dilation_w;
    conv->conv_dim = conv_dim;
    conv->ngroups = ngroups;

    bool ifRunDepthWise = ngroups > 1 && ngroups == K && ngroups == C;
    bool ifRunDepthWiseRemain = false; // It's for big padding or big kernel or Conv3D depth-wise convolution.

    if (ifRunDepthWise)
    {
        if (conv_dim == CONV_1D)
        {
            ifRunDepthWise &= Hk == 1 && Wk == 3 && (stride_w == 1 || (stride_w == 2 && dilation_w == 1))
                              && max(stride_w, dilation_w) >= conv->pad_left && conv->pad_left <= 1;
        }
        else if (conv_dim == CONV_2D)
        {
            ifRunDepthWise &= Hk == 3 && Wk == 3 && ((stride_w == 1) || (stride_w == 2 && dilation_w == 1)) &&
                              max(stride_w, dilation_w) >= conv->pad_left && max(stride_h, dilation_h) >= conv->pad_top
                              && conv->pad_left <= 1 && conv->pad_top <= 1;
        }

        if (!ifRunDepthWise || conv_dim == CONV_3D)
        {
            ifRunDepthWise = false;
            ifRunDepthWiseRemain = true;
        }
    }

    conv->conv_type = ifRunDepthWise && conv_dim != CONV_3D ? _FX_CONV_TYPE_DEPTHWISE :
            useWinograd && (conv_dim == CONV_2D && (conv->useSIMD128 || conv->useAVX2 || conv->useNEON) &&
            Hk == 3 && Wk == 3 && dilation_h == 1 && dilation_w == 1 && stride_h == 1 && stride_w == 1) ?
            _FX_CONV_TYPE_WINOGRAD3X3 :
            (ifRunDepthWiseRemain ? _FX_CONV_TYPE_DEPTHWISE_REMAIN : _FX_CONV_TYPE_GENERIC);

#if !(CV_NEON || CV_SIMD128 || CV_TRY_AVX2)
    if (conv->conv_type == _FX_CONV_TYPE_WINOGRAD3X3) // Disabel Winograd when CV_NEON, CV_SIMD128 and CV_TRY_AVX2 are not available.
        conv->conv_type = _FX_CONV_TYPE_GENERIC;
#endif

    Mat weightsMat = _weightsMat.getMat();
    auto wShape = shape(weightsMat);
    const size_t wstep = weightsMat.step1();

    float *srcWeights = (float *)weightsMat.data;
    if (conv->conv_type == _FX_CONV_TYPE_DEPTHWISE || conv->conv_type == _FX_CONV_TYPE_DEPTHWISE_REMAIN)
    {
        // Handle the Conv1D, Conv2D and Conv3D depth-wise.
        // for depth-wise convolutions on NCHW data we just preserve the weights in KCHW layout,
        // but add some padding to make the weights array layout more SIMD-friendly
        int ksize = karea;

        // TODO: simplify the following code with std::copy.
        // this code aims to let memory fit with vector size.
        int padded_ksize = ((ksize + VEC_ALIGN-1) / VEC_ALIGN) * VEC_ALIGN;
        int nweights = C*padded_ksize;
        conv->weightsBuf.reserve(nweights + VEC_ALIGN);
        conv->weightsBufPtr = alignPtr(conv->weightsBuf.data(), VEC_ALIGN);
        memset(conv->weightsBufPtr, 0, nweights*sizeof(conv->weightsBufPtr[0]));
        auto weightsBufPtr = conv->weightsBufPtr;
        parallel_for_(Range(0, C), [&](const Range& r0){
        for(int c = r0.start; c < r0.end; c++)
        {
            for (int k = 0; k < ksize; k++)
                weightsBufPtr[c*padded_ksize + k] = srcWeights[c*wstep + k];
        }});
    }
    else if(conv->conv_type == _FX_CONV_TYPE_WINOGRAD3X3) // winograd
    {
        static const float ktm[8][3] = {
                {1.0f,      0.0f,      0.0f},
                {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                {1.0f / 90, 1.0f / 45, 2.0f / 45},
                {1.0f / 90, -1.0f / 45, 2.0f / 45},
                {32.f/45, 16.f/45, 8.f/45},
                {32.f/45, -16.f/45, 8.f/45},
                {0.0f, 0.0f, 1.0f}
        };

        // the weights are packed as 6-dim tensor:
        // ngroups * ceil((K/ngroups)/KBLOCK) * (W*W/ATOM_SIZE) * (C/ngroups) * KBLOCK * ATOM_SIZE,
        // where W is the size of Winograd-transformed kernel (8x8),
        // ATOM_SIZE is number of lanes in SIMD register (4 for NEON and FP32),
        // KBLOCK is some platform-dependent constant dependent on the number of SIMD registers.
        int ksize = _FX_WINO_KSIZE * _FX_WINO_KSIZE;
        int Cg = C/ngroups;
        int Kg = K/ngroups;
        int Kg_nblocks = (Kg + _FX_WINO_KBLOCK - 1)/_FX_WINO_KBLOCK;
        size_t nweights = ngroups*Kg_nblocks*Cg*_FX_WINO_KBLOCK*_FX_WINO_AREA;
        conv->weightsWinoBuf.reserve(nweights + VEC_ALIGN);
        conv->weightsWinoBufPtr = alignPtr(conv->weightsWinoBuf.data(), VEC_ALIGN);
        float* wptrWino = conv->weightsWinoBufPtr;
        memset(wptrWino, 0, nweights * sizeof(wptrWino[0]));

        parallel_for_(Range(0, K), [&](const Range& r0){
        float kernelTm[_FX_WINO_AREA];
        for (int k = r0.start; k < r0.end; k++)
        {
            int g = k / Kg;
            int k_ = k - g*Kg;
            int ki = k_ / _FX_WINO_KBLOCK;
            int dk = k_ - ki*_FX_WINO_KBLOCK;

            for (int c = 0; c < Cg; c++)
            {
                // wstep = Hk*Wk*Cg
                const float *kernel0 = srcWeights + k * wstep + c * ksize;

                // transform kernel, transposed
                const float *k0 = kernel0;
                const float *k1 = kernel0 + 3;
                const float *k2 = kernel0 + 6;

                // h
                float tmp[8][3];
                for (int i = 0; i < 8; i++)
                {
                    tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                    tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                    tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                }

                // v
                for (int j = 0; j < 8; j++)
                {
                    float *tmpp = &tmp[j][0];

                    for (int i = 0; i < 8; i++)
                        kernelTm[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }

                // repack the data.
                float* wptr = wptrWino + (g*Kg_nblocks + ki) * Cg *_FX_WINO_KBLOCK*_FX_WINO_AREA +
                              (c*_FX_WINO_KBLOCK + dk)*_FX_WINO_ATOM_F32;
                for (int i = 0; i < _FX_WINO_NATOMS_F32; i++,
                        wptr += Cg * _FX_WINO_KBLOCK * _FX_WINO_ATOM_F32)
                {
                    CV_Assert(conv->weightsWinoBufPtr <= wptr && wptr + _FX_WINO_ATOM_F32 <= conv->weightsWinoBufPtr + nweights);
                    memcpy(wptr, kernelTm + i * _FX_WINO_ATOM_F32, _FX_WINO_ATOM_F32*sizeof (wptr[0]));
                }
            }
        }});
    }
    else if (conv->conv_type == _FX_CONV_TYPE_GENERIC)
    {
        // The weights are packed as
        // ngroups x (ceil((K/ngroups)/CONV_MR)*CONV_MR) x (Cg*Hk*Wk*Dk) x CONV_MR tensor
        int Kg = K/ngroups, Cg = max(C/ngroups, 1);
        int numStripsMR = (Kg + CONV_MR - 1) / CONV_MR;
        int Kg_aligned = numStripsMR * CONV_MR;
        int DkHkWkCg = Dk*Hk*Wk*Cg;
        size_t nweights = ngroups*Kg_aligned*DkHkWkCg;
        conv->weightsBuf.reserve(nweights + VEC_ALIGN);
        conv->weightsBufPtr = alignPtr(conv->weightsBuf.data(), VEC_ALIGN);
        float* weightsBufPtr = conv->weightsBufPtr;
        memset(weightsBufPtr, 0, nweights*sizeof(weightsBufPtr[0]));

        // Pack the weight.
        parallel_for_(Range(0, ngroups * numStripsMR), [&](const Range& r0){
        for (int gsi = r0.start; gsi < r0.end; gsi++)
        {
            int g = gsi / numStripsMR;
            int si = gsi - g * numStripsMR;

            int startK = si * CONV_MR;
            CV_Assert(startK < Kg_aligned);

            float* packed_wptr = weightsBufPtr + DkHkWkCg * (startK + g * Kg_aligned);
            int dk = Kg - startK < CONV_MR ? Kg - startK : CONV_MR; // check if we need zero padding.

            int k_idx = g*Kg + startK;
            for(int hwd = 0; hwd < Hk*Wk*Dk; hwd++) {
                for(int c = 0; c < Cg; c++, packed_wptr += CONV_MR)
                {
                    const float* wptr = srcWeights + wstep * k_idx + c*Hk*Wk*Dk + hwd;
                    int k = 0;
                    for(; k < dk; k++, wptr += wstep)
                        packed_wptr[k] = *wptr;
                    for(; k < CONV_MR; k++)
                        packed_wptr[k] = 0.f;
                }
            }
        }});
    }
    else
        CV_Error(CV_StsUnsupportedFormat, "Unknown convolution type.");

    // store bias; append some zero's to make sure that
    // we can always read MR elements starting from any valid index
    {
        int k = 0, nbias = K + VEC_ALIGN;
        conv->biasBuf.reserve(nbias);
        float* biasBufPtr = conv->biasBuf.data();
        for(; k < K; k++)
            biasBufPtr[k] = srcBias ? srcBias[k] : 0.f;
        for(; k < nbias; k++)
            biasBufPtr[k] = 0.f;
    }
    return conv;
}

static inline void packData8(float*& inpbuf, float*& inptrIn, int& in_w, int& x0, int& s0, const int* ofstab,
                      const int stride_w, const int ksize)
{
    float* inpbufC = inpbuf + s0;
    float* inptrInC = inptrIn;

    if (stride_w == 1)
        for (int k = 0; k < ksize; k++)
        {
            int k1 = ofstab[k];
            float v0 = inptrInC[k1];
            float v1 = inptrInC[k1 + 1];
            float v2 = inptrInC[k1 + 2];
            float v3 = inptrInC[k1 + 3];
            float v4 = inptrInC[k1 + 4];
            float v5 = inptrInC[k1 + 5];
            float v6 = inptrInC[k1 + 6];
            float v7 = inptrInC[k1 + 7];

            inpbufC[k*CONV_NR] = v0;
            inpbufC[k*CONV_NR+1] = v1;
            inpbufC[k*CONV_NR+2] = v2;
            inpbufC[k*CONV_NR+3] = v3;
            inpbufC[k*CONV_NR+4] = v4;
            inpbufC[k*CONV_NR+5] = v5;
            inpbufC[k*CONV_NR+6] = v6;
            inpbufC[k*CONV_NR+7] = v7;
        }
    else
        for (int k = 0; k < ksize; k++)
        {
            int k1 = ofstab[k];
            float v0 = inptrInC[k1];
            float v1 = inptrInC[k1 + stride_w];
            float v2 = inptrInC[k1 + 2*stride_w];
            float v3 = inptrInC[k1 + 3*stride_w];
            float v4 = inptrInC[k1 + 4*stride_w];
            float v5 = inptrInC[k1 + 5*stride_w];
            float v6 = inptrInC[k1 + 6*stride_w];
            float v7 = inptrInC[k1 + 7*stride_w];

            inpbufC[k*CONV_NR] = v0;
            inpbufC[k*CONV_NR+1] = v1;
            inpbufC[k*CONV_NR+2] = v2;
            inpbufC[k*CONV_NR+3] = v3;
            inpbufC[k*CONV_NR+4] = v4;
            inpbufC[k*CONV_NR+5] = v5;
            inpbufC[k*CONV_NR+6] = v6;
            inpbufC[k*CONV_NR+7] = v7;
        }
    x0+=7;
    s0+=7;
    inptrIn += 7*stride_w;
    in_w += 7*stride_w;
}

static inline void packData2(float*& inpbuf, float*& inptrIn, int& in_w, int& x0, int& s0, const int* ofstab,
                      const int stride_w, const int ksize)
{
    float* inpbufC = inpbuf + s0;
    float* inptrInC = inptrIn;

    for (int k = 0; k < ksize; k++)
    {
        int k1 = ofstab[k];
        float v0 = inptrInC[k1];
        float v1 = inptrInC[k1 + stride_w];
        inpbufC[k*CONV_NR] = v0;
        inpbufC[k*CONV_NR+1] = v1;
    }

    x0++;
    s0++;
    inptrIn += stride_w;
    in_w += stride_w;
}

void runFastConv(InputArray _input, OutputArray _output, const Ptr<FastConv>& conv, int ntasks,
                   const Ptr<ActivationLayer>& actLayer, const std::vector<float>& reluslope, bool fusedAdd)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    int conv_dim = conv->conv_dim;

    CV_Assert_N(input.dims == output.dims,
                input.size[0] == output.size[0],
                conv->C == input.size[1],
                conv->K == output.size[1],
                input.type() == output.type(),
                input.isContinuous(),
                output.isContinuous());

    Mat fusedAddMat;
    if (fusedAdd)
    {
        CV_Assert(conv->conv_dim != CONV_3D && "Conv3D does not support Conv+Add fusion optimization!");
        fusedAddMat = _output.getMat();
    }

    if (conv->conv_type == _FX_CONV_TYPE_DEPTHWISE)
    {
        // Depthwise-Convolution layer should not be followed by Add layer.
        CV_Assert(fusedAddMat.empty() && (conv_dim == CONV_1D || conv_dim == CONV_2D));
        return runDepthwise(input, output, conv,actLayer.get(), reluslope);
    }

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);

    CV_Assert(inputShape.size() == outputShape.size());

    ActivationLayer* activ = nullptr;
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

    if (conv->conv_type == _FX_CONV_TYPE_WINOGRAD3X3) // winograd
    {
        CV_Assert(conv->weightsWinoBufPtr && input.dims == 4 && conv_dim == CONV_2D);
        if (runWinograd63(input, fusedAddMat, output, conv, ntasks, minval, maxval, activ, ifMinMaxAct))
            return;
    }

    int N = inputShape[0], C = inputShape[1];

    // input shape: [N, C, D, H, W] for Conv3D, [N, C, H, W] for Conv2D, [N, C, W] for Conv1D.
    int Di = conv_dim == CONV_3D ? inputShape[2] : 1;
    int Hi = conv_dim == CONV_1D ? 1 : inputShape[inputShape.size() - 2];
    int Wi = inputShape[inputShape.size() - 1];

    int ngroups = conv->ngroups;
    int K = conv->K, Dk = conv->Dk, Hk = conv->Hk, Wk = conv->Wk;

    int D0 = conv_dim == CONV_3D ? outputShape[2] : 1;
    int H0 = conv_dim == CONV_1D ? 1 : outputShape[outputShape.size() - 2];
    int W0 = outputShape[outputShape.size() - 1];

    int Cg = C/ngroups, Kg = K/ngroups;

    const size_t inp_planesize = (size_t)Di*Hi*Wi;
    const size_t out_planesize = (size_t)D0*H0*W0;

    int pad_front = conv->pad_front;
    int pad_top = conv->pad_top;
    int pad_left = conv->pad_left;

    int stride_d = conv->stride_d, stride_h = conv->stride_h, stride_w = conv->stride_w;
    int dilation_d = conv->dilation_d, dilation_h = conv->dilation_h, dilation_w = conv->dilation_w;

    int ksize = Dk*Hk*Wk;
    bool fast_1x1 = ksize == 1 && stride_d == 1 && stride_w == 1 && stride_h == 1 &&
                    pad_front == 0 && pad_top == 0 && pad_left == 0;
    int DkHkWkCg = Dk*Hk*Wk*Cg;

    std::vector<int> ofstab_(Hk*Wk*Dk*4, 0);
    int* ofstab = ofstab_.data();
    int* dhwTab = ofstab + Hk*Wk*Dk;
    int padded_ksize = ((ksize + VEC_ALIGN-1) / VEC_ALIGN) * VEC_ALIGN;

    if (conv_dim == CONV_1D)
    {
        for( int w = 0; w < Wk; w++)
        {
            int dw = w*dilation_w;
            dhwTab[w*3+2] = dw;
            ofstab[w] = dw;
        }
    }
    else if (conv_dim == CONV_2D)
    {
        for (int h = 0; h < Hk; h++)
            for( int w = 0; w < Wk; w++)
            {
                int k = h*Wk + w;
                int dh = h*dilation_h, dw = w*dilation_w;
                dhwTab[k*3+1] = dh;
                dhwTab[k*3+2] = dw;
                ofstab[k] = dh*Wi + dw;
            }
    }
    else
    {
        for (int d = 0; d < Dk; d++)
            for (int h = 0; h < Hk; h++)
            {
                for (int w = 0; w < Wk; w++)
                {
                    int k = d*Hk*Wk + h*Wk + w;
                    int dd = d*dilation_d, dh = h*dilation_h, dw = w*dilation_w;
                    dhwTab[k*3] = dd;
                    dhwTab[k*3+1] = dh;
                    dhwTab[k*3+2] = dw;
                    ofstab[k] = dd*Hi*Wi + dh*Wi + dw;
                }
            }
    }

    int MAX_STRIPES = (56 + CONV_NR - 1)/CONV_NR;

    // Friendly to L1 cache
    const int K_BLOCK_SIZE = conv->conv_type == _FX_CONV_TYPE_DEPTHWISE_REMAIN ? 1 : 32;
    const int C_BLOCK_SIZE = 256;

    int Kg_nblocks = (Kg + CONV_MR-1)/CONV_MR, Kg_aligned = Kg_nblocks * CONV_MR;

    int stripes_per_sample = ((int)out_planesize + CONV_NR - 1) / CONV_NR;

    if (stripes_per_sample < ntasks * 4 && conv->conv_type != _FX_CONV_TYPE_DEPTHWISE_REMAIN)
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
            int kzyx0 = subtask - ng * Kstripes;
            int kzyx1 = kzyx0 + (ngs1 - subtask);
            int n = ng / ngroups, g = ng % ngroups; // ng - n * ngroups;
            size_t inp_plane_ofs = (size_t)(n * ngroups + g) * Cg * inp_planesize;
            kzyx1 = kzyx1 <= Kstripes ? kzyx1 : Kstripes;
            subtask += kzyx1 - kzyx0;
            int k0, k1;
            int zyx0, zyx_limit, zyx_block_limit = 0;

            if (stripes_per_sample == 1 && conv->conv_type != _FX_CONV_TYPE_DEPTHWISE_REMAIN)
            {
                k0 = kzyx0 * CONV_MR;
                k1 = kzyx1 * CONV_MR;
                k1 = k1 <= Kg ? k1 : Kg;
                zyx0 = 0;
                zyx_limit = (int)out_planesize;
            }
            else
            {
                k0 = 0;
                k1 = Kg;
                zyx0 = kzyx0 * CONV_NR;
                zyx_limit = kzyx1 * CONV_NR;
                zyx_limit = zyx_limit < out_planesize ? zyx_limit : (int)out_planesize;
            }

            for (; zyx0 < zyx_limit; zyx0 = zyx_block_limit)
            {
                // step 1. extract part of input tensor and represent it in zigzag form
                zyx_block_limit = zyx0 + CONV_NR * MAX_STRIPES;
                zyx_block_limit = zyx_block_limit < zyx_limit ? zyx_block_limit : zyx_limit;

                int nstripes = (zyx_block_limit - zyx0 + CONV_NR - 1) / CONV_NR;
                int zyx0_saved = zyx0;

                CV_Assert(nstripes <= MAX_STRIPES);

                for (int stripe = 0; zyx0 < zyx_block_limit; stripe++, zyx0 += CONV_NR)
                {
                    float *inpbuf = inpbuf_task + stripe * stripesize;
                    float *inptr = inp + inp_plane_ofs;

                    /*
                        1. pack the data. Copy the HkxWk CONV_NR-wide slices from
                           each feature plane of the input tensor to the input buffer.
                    */
                    if (fast_1x1)
                    {
                        int slice_len = zyx_block_limit - zyx0;
                        bool partial = slice_len < CONV_NR;
                        // Superfast branch for 1x1 convolutions with sy=sx=1.
                        // in this case each feature plane can be safely treated
                        // as 1D array, and we just extract next portion
                        // of CONV_NR elements from each feature plane and
                        // put it together.
                        inptr += zyx0;
                        if (!partial)
                        {
                            // Make special branch where memcpy() is called with a constant buffer size.
                            // Compilers will likely unroll this loop properly.
                            for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR)
                                memcpy(inpbuf, inptr, CONV_NR * sizeof(inpbuf[0]));
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
                    else if (conv->conv_type == _FX_CONV_TYPE_DEPTHWISE_REMAIN)
                    {
                        CV_Assert(Cg == 1);
                        const int HW0 = H0 * W0;
                        const int HWi = Hi * Wi;
                        int slice_len = std::min(zyx_block_limit - zyx0, CONV_NR);

                        // here some non-continuous sub-row of the row will not be
                        // filled from the tensor; we need to make sure that the uncovered
                        // elements are explicitly set to 0's. the easiest way is to
                        // set all the elements to 0's before the loop.
                        memset(inpbuf, 0, stripesize*sizeof(inpbuf[0]));

                        int z0 = zyx0 / HW0, yx0 = zyx0 - z0 * HW0;
                        int y0 = yx0 / W0, x0 = yx0 - y0 * W0;

                        if (conv_dim == CONV_1D)
                        {
                            for (int slice_i = 0; slice_i < slice_len; y0++, x0=0)
                            {
                                int delta = std::min(slice_len - slice_i, W0 - x0);
                                int x1 = x0 + delta;

                                int in_w = x0 * stride_w - pad_left;
                                float* inptrIn = inptr + in_w;

                                int s0 = slice_i;

                                for (; x0 < x1; x0++, s0++, inptrIn += stride_w, in_w += stride_w)
                                {
                                    // Pack 8
                                    if (x0 + 8 <= x1 && 0 <= in_w &&
                                        in_w + stride_w*8 <= Wi - (Wk-1)*dilation_w)
                                    {
                                        packData8(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize);
                                    }
                                    else if (x0 + 2 <= x1 && 0 <= in_w &&
                                        in_w + stride_w*2 <= Wi - (Wk-1)*dilation_w)
                                    {
                                        packData2(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize);
                                    }
                                    else
                                    {
                                        int w0 = std::max(0, (-in_w + dilation_w-1)/dilation_w);
                                        int w1 = std::min(Wk, (Wi - in_w + dilation_w-1)/dilation_w);

                                        float* inpbufC = inpbuf + s0;
                                        float* inptrInC = inptrIn;
                                        for (int w = w0; w < w1; w++)
                                        {
                                            int imgofs = w*dilation_w;
                                            inpbufC[w*CONV_NR] = inptrInC[imgofs];
                                        }
                                    }
                                }
                                slice_i += delta;
                            }
                        }
                        else if (conv_dim == CONV_2D)
                        {
                            for (int slice_i = 0; slice_i < slice_len; y0++, x0=0)
                            {
                                int delta = std::min(slice_len - slice_i, W0 - x0);
                                int x1 = x0 + delta;

                                int in_h = y0 * stride_h - pad_top;
                                int in_w = x0 * stride_w - pad_left;

                                float* inptrIn = inptr + in_h*Wi + in_w;

                                bool ok_i = 0 <= in_h && in_h < Hi - (Hk-1)*dilation_h;
                                int h0 = std::max(0, (-in_h + dilation_h-1)/dilation_h);
                                int h1 = std::min(Hk, (Hi - in_h + dilation_h-1)/dilation_h);

                                int s0 = slice_i;
                                for (; x0 < x1; x0++, s0++, inptrIn += stride_w, in_w += stride_w)
                                {
                                    // Pack 8
                                    if (ok_i && x0 + 8 <= x1 && 0 <= in_w &&
                                        in_w + stride_w*8 <= Wi - (Wk-1)*dilation_w)
                                    {
                                        packData8(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize);
                                    }
                                    else if (ok_i && x0 + 2 <= x1 && 0 <= in_w &&
                                            in_w + stride_w*2 <= Wi - (Wk-1)*dilation_w)
                                    {
                                        packData2(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize);
                                    }
                                    else
                                    {
                                        int w0 = std::max(0, (-in_w + dilation_w-1)/dilation_w);
                                        int w1 = std::min(Wk, (Wi - in_w + dilation_w-1)/dilation_w);

                                        float* inpbufC = inpbuf + s0;
                                        float* inptrInC = inptrIn;

                                        for (int h = h0; h < h1; h++)
                                        {
                                            for (int w = w0; w < w1; w++)
                                            {
                                                int imgofs = h*(dilation_h*Wi) + w*dilation_w;
                                                inpbufC[(h*Wk + w)*CONV_NR] = inptrInC[imgofs];
                                            }
                                        }
                                    }
                                }
                                slice_i += delta;
                            }
                        }
                        else if (conv_dim == CONV_3D)
                        {
                            for (int slice_i = 0; slice_i < slice_len; z0 += (y0+1)/H0, y0 = (y0+1)%H0, x0=0)
                            {
                                int delta = std::min(slice_len - slice_i, W0 - x0);
                                int x1 = x0 + delta;

                                int in_d = z0 * stride_d - pad_front;
                                int in_h = y0 * stride_h - pad_top;
                                int in_w = x0 * stride_w - pad_left;

                                float* inptrIn = inptr + in_d*HWi + in_h*Wi + in_w;

                                int d0 = std::max(0, (-in_d + dilation_d - 1) / dilation_d);
                                int d1 = std::min(Dk, (Di - in_d + dilation_d - 1) / dilation_d);

                                bool ok_i = 0 <= in_d && in_d < Di - (Dk-1)*dilation_d &&
                                        0 <= in_h && in_h < Hi - (Hk-1)*dilation_h;
                                int h0 = std::max(0, (-in_h + dilation_h-1)/dilation_h);
                                int h1 = std::min(Hk, (Hi - in_h + dilation_h-1)/dilation_h);

                                int s0 = slice_i;
                                for (; x0 < x1; x0++, s0++, inptrIn += stride_w, in_w += stride_w)
                                {
                                    // Pack 8
                                    if (ok_i && x0 + 8 <= x1 && 0 <= in_w &&
                                        in_w + stride_w*8 <= Wi - (Wk-1)*dilation_w)
                                    {
                                        packData8(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize);
                                    }
                                    else if (ok_i && x0 + 2 <= x1 && 0 <= in_w &&
                                        in_w + stride_w*2 <= Wi - (Wk-1)*dilation_w)
                                    {
                                        packData2(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize);
                                    }
                                    else
                                    {
                                        int w0 = std::max(0, (-in_w + dilation_w-1)/dilation_w);
                                        int w1 = std::min(Wk, (Wi - in_w + dilation_w-1)/dilation_w);

                                        float* inpbufC = inpbuf + s0;
                                        float* inptrInC = inptrIn;

                                        for ( int d = d0; d < d1; d++)
                                        {
                                            for (int h = h0; h < h1; h++)
                                            {
                                                for (int w = w0; w < w1; w++)
                                                {
                                                    int imgofs = d*dilation_d*HWi + h*(dilation_h*Wi) + w*dilation_w;
                                                    inpbufC[((d*Hk + h)*Wk + w)*CONV_NR] = inptrInC[imgofs];
                                                }
                                            }
                                        }
                                    }
                                }
                                slice_i += delta;
                            }
                        }
                    }
                    else
                    {
                        const int HW0 = H0 * W0;
                        const int HWi = Hi * Wi;
                        int z0_ = zyx0 / HW0, yx0 = zyx0 - z0_ * HW0;
                        int y0_ = yx0 / W0, x0_ = yx0 - y0_ * W0;
                        for (int k = 0; k < ksize; k++)
                        {
                            int dz = dhwTab[k * 3], dy = dhwTab[k * 3 + 1], dx = dhwTab[k * 3 + 2];
                            int i = 0, z0 = z0_, y0 = y0_, x0 = x0_;
                            for (; i < CONV_NR;)
                            {
                                float *inpbuf_ki = inpbuf + k * CONV_NR * Cg + i;
                                int zi = z0 * stride_d + dz - pad_front;
                                int yi = y0 * stride_h + dy - pad_top;
                                int xi = x0 * stride_w + dx - pad_left;

                                if ((unsigned) zi < (unsigned) Di && (unsigned) yi < (unsigned) Hi &&
                                    (unsigned) xi < (unsigned) Wi)
                                {
                                    const float *inptr_ki = inptr + zi * HWi + yi * Wi + xi;
                                    if (i + 8 <= CONV_NR && x0 + 8 <= W0 && xi + stride_w * 8 <= Wi)
                                    {
                                        if (stride_w == 1)
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[1];
                                                float t2 = inptr_ki[2], t3 = inptr_ki[3];
                                                float t4 = inptr_ki[4], t5 = inptr_ki[5];
                                                float t6 = inptr_ki[6], t7 = inptr_ki[7];
                                                inpbuf_ki[0] = t0;
                                                inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2;
                                                inpbuf_ki[3] = t3;
                                                inpbuf_ki[4] = t4;
                                                inpbuf_ki[5] = t5;
                                                inpbuf_ki[6] = t6;
                                                inpbuf_ki[7] = t7;
                                            }
                                        }
                                        else if (stride_w == 2)
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[2];
                                                float t2 = inptr_ki[4], t3 = inptr_ki[6];
                                                float t4 = inptr_ki[8], t5 = inptr_ki[10];
                                                float t6 = inptr_ki[12], t7 = inptr_ki[14];
                                                inpbuf_ki[0] = t0;
                                                inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2;
                                                inpbuf_ki[3] = t3;
                                                inpbuf_ki[4] = t4;
                                                inpbuf_ki[5] = t5;
                                                inpbuf_ki[6] = t6;
                                                inpbuf_ki[7] = t7;
                                            }
                                        }
                                        else
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[stride_w];
                                                float t2 = inptr_ki[stride_w * 2], t3 = inptr_ki[stride_w * 3];
                                                float t4 = inptr_ki[stride_w * 4], t5 = inptr_ki[stride_w * 5];
                                                float t6 = inptr_ki[stride_w * 6], t7 = inptr_ki[stride_w * 7];
                                                inpbuf_ki[0] = t0;
                                                inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2;
                                                inpbuf_ki[3] = t3;
                                                inpbuf_ki[4] = t4;
                                                inpbuf_ki[5] = t5;
                                                inpbuf_ki[6] = t6;
                                                inpbuf_ki[7] = t7;
                                            }
                                        }
                                        i += 8;
                                        x0 += 8;
                                    }
                                    else if (i + 4 <= CONV_NR && x0 + 4 <= W0 && xi + stride_w * 4 <= Wi)
                                    {
                                        if (stride_w == 1)
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[1];
                                                float t2 = inptr_ki[2], t3 = inptr_ki[3];
                                                inpbuf_ki[0] = t0;
                                                inpbuf_ki[1] = t1;
                                                inpbuf_ki[2] = t2;
                                                inpbuf_ki[3] = t3;
                                            }
                                        }
                                        else
                                        {
                                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                            {
                                                float t0 = inptr_ki[0], t1 = inptr_ki[stride_w];
                                                float t2 = inptr_ki[stride_w * 2], t3 = inptr_ki[stride_w * 3];
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

                                mask = y0 >= H0;
                                z0 += mask;
                                y0 &= mask - 1;
                            }
                        }
                    }
                }

                zyx0 = zyx0_saved;

                // spacial branch for depth-wise convolution implemented using generic convolution.
                // In this case, CONV_MR is 1, and CONV_NR is the same.
                if (conv->conv_type == _FX_CONV_TYPE_DEPTHWISE_REMAIN)
                {
                    size_t outofs = (n * ngroups + g) * out_planesize + zyx0;
                    float *cptr0 = cbuf_task;
                    float *weights = conv->weightsBufPtr + g * padded_ksize;
                    int out_width = zyx_block_limit - zyx0;
                    float *outptr = out + outofs;
                    const float biasVal = *(conv->biasBuf.data() + g);
                    for (int stripe = 0; stripe < nstripes; stripe++)
                    {
                        const float *inptr = inpbuf_task + stripe * stripesize;
                        const int outLen = std::min(out_width - stripe * CONV_NR, CONV_NR);
                        bool ifBuffer = outLen < CONV_NR;
                        float *cptr = outptr + stripe * CONV_NR;
                        if (ifBuffer)
                        {
                            memcpy(cptr0, cptr, outLen * sizeof(cptr[0]));
                            cptr = cptr0;
                        }
#if CV_TRY_AVX2
                        if (conv->useAVX2 && outLen > CONV_NR/3)
                                opt_AVX2::convBlockMR1(DkHkWkCg, weights, inptr, cptr, biasVal, fusedAdd, minval, maxval, ifMinMaxAct);
                        else
#endif
                        convBlockMR1(DkHkWkCg, weights, inptr, cptr, biasVal, fusedAdd, minval, maxval, ifMinMaxAct, outLen);

                        if (ifBuffer)
                        {
                            memcpy(outptr + stripe * CONV_NR, cptr, outLen * sizeof(cptr[0]));
                        }
                    }
                    if (activ)
                        activ->forwardSlice(outptr, outptr, out_width, out_planesize, g, g + 1);
                    continue;
                }

                float *weights = conv->weightsBufPtr + g * Kg_aligned * DkHkWkCg;
                const float *biasptr = conv->biasBuf.data() + Kg * g;
                int ldc = nstripes * CONV_NR;

                // 2. do convolution, compute Kg x (zyx_block_limit - zyx0) part of the output tensor
                int out_width = zyx_block_limit - zyx0;
                for (int k0_block = k0; k0_block < k1; k0_block += K_BLOCK_SIZE)
                {
                    int k1_block = k0_block + K_BLOCK_SIZE < k1 ? k0_block + K_BLOCK_SIZE : k1;
                    for (int c0 = 0; c0 < DkHkWkCg; c0 += C_BLOCK_SIZE)
                    {
                        int c1 = c0 + C_BLOCK_SIZE < DkHkWkCg ? c0 + C_BLOCK_SIZE : DkHkWkCg;
                        for (int stripe = 0; stripe < nstripes; stripe++)
                        {
                            const int outLen = std::min(out_width - stripe * CONV_NR, CONV_NR);

#if CV_TRY_AVX2 || CV_TRY_NEON
                            // The possible CONV_NR is 28, 24, 12, so the possible CONV_NR/3 is 9, 8, 4.
                            bool runOpt = outLen > std::min(8, CONV_NR/3);
#endif
                            float *wptr = weights + k0_block * DkHkWkCg + c0 * CONV_MR;
                            const float *inptr = inpbuf_task + stripe * stripesize + c0 * CONV_NR;
                            float *cptr = cbuf_task + stripe * CONV_NR;
                            for (int k = k0_block; k < k1_block; k += CONV_MR,
                                    wptr += DkHkWkCg * CONV_MR, cptr += CONV_MR * ldc)
                            {
#if CV_TRY_AVX2
                                if (conv->useAVX2 && runOpt)
                                    opt_AVX2::convBlock_AVX2(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0);
                                else
#endif
#if CV_TRY_NEON
                                if (conv->useNEON && runOpt)
                                    opt_NEON::convBlock_NEON(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0);
                                else
#endif
                                // The possible outLen range is 24 or 8~1.
                                convBlock(c1 - c0, wptr, inptr, cptr, ldc, c0 == 0, outLen);
                            }
                        }
                    }

                    size_t outofs = ((n * ngroups + g) * Kg + k0_block) * out_planesize + zyx0;
                    const float *cptr = cbuf_task;

                    float *outptr = out + outofs;
                    const float *pbptr = fusedAddPtr0 ? fusedAddPtr0 + outofs : 0;

                    for (int k = k0_block; k < k1_block; k++,
                            cptr += ldc, outptr += out_planesize,
                            pbptr += (pbptr ? out_planesize : 0)) {
                        float biasval = biasptr[k];
                        int j = 0;
#if CV_SIMD128
                        v_float32x4 vbias = v_setall_f32(biasval);
                        v_float32x4 vmax = v_setall_f32(maxval);
                        v_float32x4 vmin = v_setall_f32(minval);

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
                            for (; j < out_width; j++) {
                                float v = cptr[j] + biasval;
                                v += pbptr[j];
                                if (ifMinMaxAct)
                                    v = std::min(std::max(v, minval), maxval);
                                outptr[j] = v;
                            }
                        } else {
                            for (; j < out_width; j++) {
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
