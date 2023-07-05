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
#include "convolution.hpp"

#include "conv_block.simd.hpp"
#include "layers/cpu_kernels/conv_block.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv { namespace dnn {
enum { VEC_ALIGN = 32, DFT_TYPE = CV_32F }; // Memory alignment.

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int outLen,
               const int convMR, const int convNR);
void convBlockMR1(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
                  const float minval, const float maxval, bool ifMinMaxAct, const int outLen, const int convNR);

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
        const bool _useFP16,
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

    int stride_d = conv_dim == CONV_3D ? (int)strides[0] : 1;
    int stride_h = conv_dim == CONV_1D ? 1 : (int)strides[strides.size() - 2];
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

    conv->conv_type = ifRunDepthWise && conv_dim != CONV_3D ? CONV_TYPE_DEPTHWISE :
            useWinograd && (conv_dim == CONV_2D && (conv->useSIMD128 || conv->useAVX || conv->useAVX2 || conv->useNEON) &&
            Hk == 3 && Wk == 3 && dilation_h == 1 && dilation_w == 1 && stride_h == 1 && stride_w == 1) ?
            CONV_TYPE_WINOGRAD3X3 :
            (ifRunDepthWiseRemain ? CONV_TYPE_DEPTHWISE_REMAIN : CONV_TYPE_GENERIC);

#if !(CV_NEON || CV_SIMD128 || CV_TRY_AVX || CV_TRY_AVX2)
    if (conv->conv_type == CONV_TYPE_WINOGRAD3X3) // Disabel Winograd when CV_NEON, CV_SIMD128 ,CV_TRY_AVX and CV_TRY_AVX2 are not available.
        conv->conv_type = CONV_TYPE_GENERIC;
#endif

    Mat weightsMat = _weightsMat.getMat();
    auto wShape = shape(weightsMat);
    const size_t wstep = weightsMat.step1();

    conv->useFP16 = false;
#ifdef CONV_ARM_FP16
    // TODO: add FP16 support for Winograd.
    if (_useFP16 && (conv->conv_type == CONV_TYPE_GENERIC || conv->conv_type == CONV_TYPE_DEPTHWISE_REMAIN))
        conv->useFP16 = true;
#endif

    float *srcWeights = (float *)weightsMat.data;
    if (conv->conv_type == CONV_TYPE_DEPTHWISE || conv->conv_type == CONV_TYPE_DEPTHWISE_REMAIN)
    {
        // Handle the Conv1D, Conv2D and Conv3D depth-wise.
        // for depth-wise convolutions on NCHW data we just preserve the weights in KCHW layout,
        // but add some padding to make the weights array layout more SIMD-friendly
        int ksize = karea;

        // TODO: simplify the following code with std::copy.
        // this code aims to let memory fit with vector size.
        int padded_ksize = ((ksize + VEC_ALIGN-1) / VEC_ALIGN) * VEC_ALIGN;
        int nweights = C * padded_ksize;

#ifdef CONV_ARM_FP16
        if (conv->useFP16)
        {
            conv->weightsBuf_FP16.resize(nweights + VEC_ALIGN);
            conv->weightsBufPtr_FP16 = alignPtr(conv->weightsBuf_FP16.data(), VEC_ALIGN * sizeof(float16_t ));
            memset(conv->weightsBufPtr_FP16, 0, nweights * sizeof(float16_t ));
            auto weightsBufPtr_FP16 = conv->weightsBufPtr_FP16;

            parallel_for_(Range(0, C), [&](const Range& r0){
            for(int c = r0.start; c < r0.end; c++)
            {
                for (int k = 0; k < ksize; k++)
                    weightsBufPtr_FP16[c*padded_ksize + k] = (float16_t)srcWeights[c*wstep + k];
            }});
        }
        else
#endif
        {
            conv->weightsBuf.resize(nweights + VEC_ALIGN);
            conv->weightsBufPtr = alignPtr(conv->weightsBuf.data(), VEC_ALIGN * sizeof(float ));
            memset(conv->weightsBufPtr, 0, nweights*sizeof(float ));
            auto weightsBufPtr = conv->weightsBufPtr;

            parallel_for_(Range(0, C), [&](const Range& r0){
            for(int c = r0.start; c < r0.end; c++)
            {
                for (int k = 0; k < ksize; k++)
                    weightsBufPtr[c*padded_ksize + k] = srcWeights[c*wstep + k];
            }});
        }
    }
    else if(conv->conv_type == CONV_TYPE_WINOGRAD3X3) // winograd
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

        const int CONV_WINO_KBLOCK = 4;

#if CV_TRY_AVX || CV_TRY_AVX2
        const int CONV_WINO_ATOM_F32 = (conv->useAVX || conv->useAVX2) ? 8 : 4;
#else
        const int CONV_WINO_ATOM_F32 = 4;
#endif
        const int CONV_WINO_NATOMS_F32 = CONV_WINO_AREA / CONV_WINO_ATOM_F32; // for AVX2, it is 8, otherwise, it's 16.

#ifdef CONV_ARM_FP16
        // FP 16
        const int CONV_WINO_ATOM_F16 = CONV_WINO_ATOM_F32 * 2;
        const int CONV_WINO_NATOMS_F16 = CONV_WINO_AREA / CONV_WINO_ATOM_F16;
#endif

        // the weights are packed as 6-dim tensor:
        // ngroups * ceil((K/ngroups)/KBLOCK) * (W*W/ATOM_SIZE) * (C/ngroups) * KBLOCK * ATOM_SIZE,
        // where W is the size of Winograd-transformed kernel (8x8),
        // ATOM_SIZE is number of lanes in SIMD register (4 for NEON and FP32),
        // KBLOCK is some platform-dependent constant dependent on the number of SIMD registers.
        int ksize = CONV_WINO_KSIZE * CONV_WINO_KSIZE;
        int Cg = C/ngroups;
        int Kg = K/ngroups;
        int Kg_nblocks = (Kg + CONV_WINO_KBLOCK - 1)/CONV_WINO_KBLOCK;
        size_t nweights = ngroups*Kg_nblocks*Cg*CONV_WINO_KBLOCK*CONV_WINO_AREA;

        float* wptrWino = nullptr;
#ifdef CONV_ARM_FP16
        float16_t* wptrWino_FP16 = nullptr;
        if (conv->useFP16)
        {
            conv->weightsWinoBuf_FP16.resize(nweights + VEC_ALIGN);
            conv->weightsWinoBufPtr_FP16 = alignPtr(conv->weightsWinoBuf_FP16.data(), VEC_ALIGN);
            wptrWino_FP16 = conv->weightsWinoBufPtr_FP16;
            memset(wptrWino_FP16, 0, nweights * sizeof(wptrWino_FP16[0]));
        }
        else
#endif
        {
            conv->weightsWinoBuf.resize(nweights + VEC_ALIGN);
            conv->weightsWinoBufPtr = alignPtr(conv->weightsWinoBuf.data(), VEC_ALIGN);
            wptrWino = conv->weightsWinoBufPtr;
            memset(wptrWino, 0, nweights * sizeof(wptrWino[0]));
        }

        parallel_for_(Range(0, K), [&](const Range& r0){
        float kernelTm[CONV_WINO_AREA];
        for (int k = r0.start; k < r0.end; k++)
        {
            int g = k / Kg;
            int k_ = k - g*Kg;
            int ki = k_ / CONV_WINO_KBLOCK;
            int dk = k_ - ki*CONV_WINO_KBLOCK;

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
#ifdef CONV_ARM_FP16
                if (conv->useFP16)
                {
                    float16_t* wptr = wptrWino_FP16 + (g*Kg_nblocks + ki) * Cg *CONV_WINO_KBLOCK*CONV_WINO_AREA +
                                  (c*CONV_WINO_KBLOCK + dk)*CONV_WINO_ATOM_F16;
                    for (int i = 0; i < CONV_WINO_NATOMS_F16; i++,
                            wptr += Cg * CONV_WINO_KBLOCK * CONV_WINO_ATOM_F16)
                    {
                        CV_Assert(conv->weightsWinoBufPtr_FP16 <= wptr && wptr + CONV_WINO_ATOM_F16 <= conv->weightsWinoBufPtr_FP16 + nweights);
                        for (int j = 0; j < CONV_WINO_ATOM_F16; j++)
                        {
                            wptr[j] = (float16_t)kernelTm[i * CONV_WINO_ATOM_F16 + j];
                        }
                    }
                }
                else
#endif
                {
                    float* wptr = wptrWino + (g*Kg_nblocks + ki) * Cg *CONV_WINO_KBLOCK*CONV_WINO_AREA +
                                  (c*CONV_WINO_KBLOCK + dk)*CONV_WINO_ATOM_F32;
                    for (int i = 0; i < CONV_WINO_NATOMS_F32; i++,
                            wptr += Cg * CONV_WINO_KBLOCK * CONV_WINO_ATOM_F32)
                    {
                        CV_Assert(conv->weightsWinoBufPtr <= wptr && wptr + CONV_WINO_ATOM_F32 <= conv->weightsWinoBufPtr + nweights);
                        memcpy(wptr, kernelTm + i * CONV_WINO_ATOM_F32, CONV_WINO_ATOM_F32*sizeof (wptr[0]));
                    }
                }
            }
        }
        });
    }
    else if (conv->conv_type == CONV_TYPE_GENERIC)
    {
        // The weights are packed as
        // ngroups x (ceil((K/ngroups)/CONV_MR)*CONV_MR) x (Cg*Hk*Wk*Dk) x CONV_MR tensor
        int Kg = K/ngroups, Cg = max(C/ngroups, 1);
        int DkHkWkCg = Dk*Hk*Wk*Cg;

        int numStripsMR = (Kg + CONV_MR_FP32 - 1) / CONV_MR_FP32;
        int Kg_aligned = numStripsMR * CONV_MR_FP32;
        size_t nweights = ngroups*Kg_aligned*DkHkWkCg;

        float* weightsBufPtr = nullptr;

#ifdef CONV_ARM_FP16
        int numStripsMR_FP16 = (Kg + CONV_MR_FP16 - 1) / CONV_MR_FP16;
        int Kg_aligned_FP16 = numStripsMR_FP16 * CONV_MR_FP16;
        size_t nweights_FP16 = ngroups * Kg_aligned_FP16 * DkHkWkCg;

        float16_t* weightsBufPtr_FP16 = nullptr;
        if (conv->useFP16)
        {
            conv->weightsBuf_FP16.resize(nweights_FP16 + VEC_ALIGN);
            conv->weightsBufPtr_FP16 = alignPtr(conv->weightsBuf_FP16.data(), VEC_ALIGN);
            weightsBufPtr_FP16 = conv->weightsBufPtr_FP16;
            memset(weightsBufPtr_FP16, 0, nweights_FP16*sizeof(weightsBufPtr_FP16[0]));
        }
        else
#endif
        {
            conv->weightsBuf.resize(nweights + VEC_ALIGN);
            conv->weightsBufPtr = alignPtr(conv->weightsBuf.data(), VEC_ALIGN);
            weightsBufPtr = conv->weightsBufPtr;
            memset(weightsBufPtr, 0, nweights*sizeof(weightsBufPtr[0]));
        }

        // Pack the weight.
#ifdef CONV_ARM_FP16
        if (conv->useFP16)
        {
            parallel_for_(Range(0, ngroups * numStripsMR_FP16), [&](const Range& r0){
            for (int gsi = r0.start; gsi < r0.end; gsi++)
            {
                int g = gsi / numStripsMR_FP16;
                int si = gsi - g * numStripsMR_FP16;

                int startK = si * CONV_MR_FP16;
                CV_Assert(startK < Kg_aligned_FP16);

                float16_t* packed_wptr = weightsBufPtr_FP16 + DkHkWkCg * (startK + g * Kg_aligned_FP16);
                int dk = Kg - startK < CONV_MR_FP16 ? Kg - startK : CONV_MR_FP16; // check if we need zero padding.

                int k_idx = g*Kg + startK;
                for(int hwd = 0; hwd < Hk*Wk*Dk; hwd++)
                {
                    for(int c = 0; c < Cg; c++, packed_wptr += CONV_MR_FP16)
                    {
                        const float* wptr = srcWeights + wstep * k_idx + c*Hk*Wk*Dk + hwd;
                        int k = 0;
                        for(; k < dk; k++, wptr += wstep)
                            packed_wptr[k] = (float16_t)(*wptr);
                        for(; k < CONV_MR_FP16; k++)
                            packed_wptr[k] = (float16_t)0.f;
                    }
                }
            }});
        }
        else
#endif
        {
            parallel_for_(Range(0, ngroups * numStripsMR), [&](const Range& r0){
            for (int gsi = r0.start; gsi < r0.end; gsi++)
            {
                int g = gsi / numStripsMR;
                int si = gsi - g * numStripsMR;

                int startK = si * CONV_MR_FP32;
                CV_Assert(startK < Kg_aligned);

                float* packed_wptr = weightsBufPtr + DkHkWkCg * (startK + g * Kg_aligned);
                int dk = Kg - startK < CONV_MR_FP32 ? Kg - startK : CONV_MR_FP32; // check if we need zero padding.

                int k_idx = g*Kg + startK;
                for(int hwd = 0; hwd < Hk*Wk*Dk; hwd++)
                {
                    for(int c = 0; c < Cg; c++, packed_wptr += CONV_MR_FP32)
                    {
                        const float* wptr = srcWeights + wstep * k_idx + c*Hk*Wk*Dk + hwd;
                        int k = 0;
                        for(; k < dk; k++, wptr += wstep)
                            packed_wptr[k] = *wptr;
                        for(; k < CONV_MR_FP32; k++)
                            packed_wptr[k] = 0.f;
                    }
                }
            }});
        }
    }
    else
        CV_Error(CV_StsUnsupportedFormat, "Unknown convolution type.");

    // store bias; append some zero's to make sure that
    // we can always read MR elements starting from any valid index
    {
        int k = 0, nbias = K + VEC_ALIGN;
        conv->biasBuf.resize(nbias);
        float* biasBufPtr = conv->biasBuf.data();
        for(; k < K; k++)
            biasBufPtr[k] = srcBias ? srcBias[k] : 0.f;
        for(; k < nbias; k++)
            biasBufPtr[k] = 0.f;
    }
    return conv;
}

static inline void packData8(char*& inpbuf, float*& inptrIn, int& in_w, int& x0, int& s0, const int* ofstab,
                      const int stride_w, const int ksize, const int esz)
{
    char * inpbufC = inpbuf + s0 * esz;
    float* inptrInC = (float* )inptrIn;

#ifdef CONV_ARM_FP16
    float16_t* inpbufC_FP16 = (float16_t *)inpbufC;
    if (esz == sizeof(float16_t))
    {
        if (stride_w == 1)
        {
            for (int k = 0; k < ksize; k++)
            {
                int k1 = ofstab[k];

                float32x4_t v0 = vld1q_f32(inptrInC + k1);
                float32x4_t v1 = vld1q_f32(inptrInC + k1 + 4);
                vst1q_f16((__fp16*)inpbufC_FP16 + k * CONV_NR_FP16, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
            }
        }
        else
        {
            for (int k = 0; k < ksize; k++)
            {
                int k1 = ofstab[k];
                float32x4_t v0, v1;

                v0[0] = inptrInC[k1];
                v0[1] = inptrInC[k1 + stride_w];
                v0[2] = inptrInC[k1 + 2*stride_w];
                v0[3] = inptrInC[k1 + 3*stride_w];
                v1[0] = inptrInC[k1 + 4*stride_w];
                v1[1] = inptrInC[k1 + 5*stride_w];
                v1[2] = inptrInC[k1 + 6*stride_w];
                v1[3] = inptrInC[k1 + 7*stride_w];

                vst1q_f16((__fp16*)inpbufC_FP16 + k * CONV_NR_FP16, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
            }
        }
    }
    else // float 32
#endif
    {
        CV_Assert(esz == sizeof(float ));
        float* inpbufC_FP32 = (float* )inpbufC;
        if (stride_w == 1)
            for (int k = 0; k < ksize; k++)
            {
                int k1 = ofstab[k];
#if CV_SIMD256
                vx_store(inpbufC_FP32 + k*CONV_NR_FP32, vx_load(inptrInC + k1));
#elif CV_SIMD128
                v_float32x4 vv0 = v_load(inptrInC + k1);
                v_float32x4 vv1 = v_load(inptrInC + k1 + 4);
                v_store(inpbufC_FP32 + k*CONV_NR_FP32, vv0);
                v_store(inpbufC_FP32 + k*CONV_NR_FP32 + 4, vv1);
#else
                float v0 = inptrInC[k1];
                float v1 = inptrInC[k1 + 1];
                float v2 = inptrInC[k1 + 2];
                float v3 = inptrInC[k1 + 3];
                float v4 = inptrInC[k1 + 4];
                float v5 = inptrInC[k1 + 5];
                float v6 = inptrInC[k1 + 6];
                float v7 = inptrInC[k1 + 7];

                inpbufC_FP32[k*CONV_NR_FP32] = v0;
                inpbufC_FP32[k*CONV_NR_FP32+1] = v1;
                inpbufC_FP32[k*CONV_NR_FP32+2] = v2;
                inpbufC_FP32[k*CONV_NR_FP32+3] = v3;
                inpbufC_FP32[k*CONV_NR_FP32+4] = v4;
                inpbufC_FP32[k*CONV_NR_FP32+5] = v5;
                inpbufC_FP32[k*CONV_NR_FP32+6] = v6;
                inpbufC_FP32[k*CONV_NR_FP32+7] = v7;
#endif
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

                inpbufC_FP32[k*CONV_NR_FP32] = v0;
                inpbufC_FP32[k*CONV_NR_FP32+1] = v1;
                inpbufC_FP32[k*CONV_NR_FP32+2] = v2;
                inpbufC_FP32[k*CONV_NR_FP32+3] = v3;
                inpbufC_FP32[k*CONV_NR_FP32+4] = v4;
                inpbufC_FP32[k*CONV_NR_FP32+5] = v5;
                inpbufC_FP32[k*CONV_NR_FP32+6] = v6;
                inpbufC_FP32[k*CONV_NR_FP32+7] = v7;
            }
    }
    x0+=7;
    s0+=7;
    inptrIn += 7*stride_w;
    in_w += 7*stride_w;
}

static inline void packData2(char *& inpbuf, float*& inptrIn, int& in_w, int& x0, int& s0, const int* ofstab,
                      const int stride_w, const int ksize, const int esz)
{
    char* inpbufC = inpbuf + s0 * esz;
    float* inptrInC = inptrIn;

#ifdef CONV_ARM_FP16
    float16_t* inpbufC_FP16 = (float16_t *)inpbufC;
    if (esz == sizeof(float16_t))
    {
        for (int k = 0; k < ksize; k++)
        {
            int k1 = ofstab[k];
            float v0 = inptrInC[k1];
            float v1 = inptrInC[k1 + stride_w];
            inpbufC_FP16[k*CONV_NR_FP16] = (float16_t)v0;
            inpbufC_FP16[k*CONV_NR_FP16+1] = (float16_t)v1;
        }
    } else
#endif
    {
        float * inpbufC_FP32 = (float *)inpbufC;
        for (int k = 0; k < ksize; k++)
        {
            int k1 = ofstab[k];
            float v0 = inptrInC[k1];
            float v1 = inptrInC[k1 + stride_w];
            inpbufC_FP32[k*CONV_NR_FP32] = v0;
            inpbufC_FP32[k*CONV_NR_FP32+1] = v1;
        }
    }

    x0++;
    s0++;
    inptrIn += stride_w;
    in_w += stride_w;
}

#ifdef CONV_ARM_FP16
// Fast convert float 32 to float16
static inline void _cvt32f16f( const float* src, float16_t* dst, int len)
{
    int j = 0;
    const int VECSZ = 4;
    __fp16* dst_FP16 = (__fp16 *)dst;
    if (len > VECSZ * 4)
    {
        const int VECSZ4 = 4 * VECSZ;
        for( ; j + VECSZ4 < len; j += VECSZ4)
        {

            float32x4_t v0 = vld1q_f32(src + j);
            float32x4_t v1 = vld1q_f32(src + j + 4);
            float32x4_t v2 = vld1q_f32(src + j + 8);
            float32x4_t v3 = vld1q_f32(src + j + 12);

            vst1q_f16(dst_FP16 + j, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
            vst1q_f16(dst_FP16 + j + 8, vcombine_f16(vcvt_f16_f32(v2), vcvt_f16_f32(v3)));
        }
    }

    for( ; j < len; j += VECSZ )
    {
        if( j > len - VECSZ )
        {
            if( j == 0 )
                break;
            j = len - VECSZ;
        }

        float16x4_t hv = vcvt_f16_f32(vld1q_f32(src + j));
        vst1_f16(dst_FP16 + j, hv);
    }
    for( ; j < len; j++ )
        dst[j] = float16_t(src[j]);
}
#endif

static inline void packInputData(char* inpbuf_task, float* inp, const int* ofstab, const int* dhwTab, int zyx0, int zyx_limit,
                                 int ksize, int stride_d, int stride_h, int stride_w, int pad_front, int pad_top, int pad_left,
                                 int Dk, int Hk, int Wk, int dilation_d, int dilation_h, int dilation_w, int Di, int Hi, int Wi,
                                 int H0, int W0, int Cg, int stripesize, int inp_plane_ofs, int inp_planesize, int conv_dim, int conv_type,
                                 const int CONV_NR, const int esz,  bool fast_1x1, bool useFP16)
{
    for (int stripe = 0; zyx0 < zyx_limit; stripe++, zyx0 += CONV_NR)
    {
        char *inpbuf = inpbuf_task + stripe * stripesize * esz;
        float *inptr = inp + inp_plane_ofs;

        /*
            1. pack the data. Copy the HkxWk CONV_NR-wide slices from
               each feature plane of the input tensor to the input buffer.
        */
        if (fast_1x1)
        {
            int slice_len = zyx_limit - zyx0;
            bool partial = slice_len < CONV_NR;
            const int CONV_NR_esz = CONV_NR * esz;
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
#ifdef CONV_ARM_FP16
                if (useFP16)
                {
                    for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR_esz)
                        _cvt32f16f(inptr, (float16_t *)inpbuf, CONV_NR);
                }
                else
#endif
                    for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR_esz)
                        memcpy(inpbuf, inptr, CONV_NR_esz);
            }
            else
            {
#ifdef CONV_ARM_FP16
                if (useFP16)
                {
                    for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR_esz)
                    {
                        _cvt32f16f(inptr, (float16_t *)inpbuf, slice_len);
                        memset(inpbuf + slice_len * esz, 0, (CONV_NR - slice_len) * esz);
                    }
                }
                else
#endif
                for (int c = 0; c < Cg; c++, inptr += inp_planesize, inpbuf += CONV_NR_esz)
                {
                    memcpy(inpbuf, inptr, slice_len * esz);
                    memset(inpbuf + slice_len * esz, 0, (CONV_NR - slice_len) * esz);
                }
            }
        }
        else if (conv_type == CONV_TYPE_DEPTHWISE_REMAIN)
        {
            CV_Assert(Cg == 1);
            const int HW0 = H0 * W0;
            const int HWi = Hi * Wi;
            int slice_len = std::min(zyx_limit - zyx0, CONV_NR);

            // here some non-continuous sub-row of the row will not be
            // filled from the tensor; we need to make sure that the uncovered
            // elements are explicitly set to 0's. the easiest way is to
            // set all the elements to 0's before the loop.
            memset(inpbuf, 0, stripesize * esz);

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
                            packData8(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize, esz);
                        }
                        else if (x0 + 2 <= x1 && 0 <= in_w &&
                                 in_w + stride_w*2 <= Wi - (Wk-1)*dilation_w)
                        {
                            packData2(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize, esz);
                        }
                        else
                        {
                            int w0 = std::max(0, (-in_w + dilation_w-1)/dilation_w);
                            int w1 = std::min(Wk, (Wi - in_w + dilation_w-1)/dilation_w);
                            const float* inptrInC = inptrIn;
#ifdef CONV_ARM_FP16
                            if (useFP16)
                            {
                                float16_t* inpbufC = (float16_t *)inpbuf + s0;
                                for (int w = w0; w < w1; w++)
                                {
                                    int imgofs = w*dilation_w;
                                    inpbufC[w*CONV_NR] = (float16_t)inptrInC[imgofs];
                                }
                            }
                            else
#endif
                            {
                                float* inpbufC = (float *)inpbuf + s0;
                                for (int w = w0; w < w1; w++)
                                {
                                    int imgofs = w*dilation_w;
                                    inpbufC[w*CONV_NR] = inptrInC[imgofs];
                                }
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
                            packData8(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize, esz);
                        }
                        else if (ok_i && x0 + 2 <= x1 && 0 <= in_w &&
                                 in_w + stride_w*2 <= Wi - (Wk-1)*dilation_w)
                        {
                            packData2(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize, esz);
                        }
                        else
                        {
                            int w0 = std::max(0, (-in_w + dilation_w-1)/dilation_w);
                            int w1 = std::min(Wk, (Wi - in_w + dilation_w-1)/dilation_w);

                            const float* inptrInC = inptrIn;
#ifdef CONV_ARM_FP16
                            if (useFP16)
                            {
                                float16_t* inpbufC = (float16_t *)inpbuf + s0;

                                for (int h = h0; h < h1; h++)
                                {
                                    for (int w = w0; w < w1; w++)
                                    {
                                        int imgofs = h*(dilation_h*Wi) + w*dilation_w;
                                        inpbufC[(h*Wk + w)*CONV_NR] = (float16_t)inptrInC[imgofs];
                                    }
                                }
                            }
                            else
#endif
                            {
                                float* inpbufC = (float *)inpbuf + s0;

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
                            packData8(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize, esz);
                        }
                        else if (ok_i && x0 + 2 <= x1 && 0 <= in_w &&
                                 in_w + stride_w*2 <= Wi - (Wk-1)*dilation_w)
                        {
                            packData2(inpbuf, inptrIn, in_w, x0, s0, ofstab, stride_w, ksize, esz);
                        }
                        else
                        {
                            int w0 = std::max(0, (-in_w + dilation_w-1)/dilation_w);
                            int w1 = std::min(Wk, (Wi - in_w + dilation_w-1)/dilation_w);
                            const float* inptrInC = inptrIn;
#ifdef CONV_ARM_FP16
                            if (useFP16)
                            {
                                float16_t* inpbufC = (float16_t* )inpbuf + s0;

                                for ( int d = d0; d < d1; d++)
                                {
                                    for (int h = h0; h < h1; h++)
                                    {
                                        for (int w = w0; w < w1; w++)
                                        {
                                            int imgofs = d*dilation_d*HWi + h*(dilation_h*Wi) + w*dilation_w;
                                            inpbufC[((d*Hk + h)*Wk + w)*CONV_NR] = (float16_t)inptrInC[imgofs];
                                        }
                                    }
                                }
                            }
                            else
#endif
                            {
                                float* inpbufC = (float* )inpbuf + s0;

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
                    float* inpbuf_ki = (float* )inpbuf + k * CONV_NR * Cg + i;
#ifdef CONV_ARM_FP16
                    float16_t * inpbuf_ki_FP16 = (float16_t *)inpbuf + k * CONV_NR * Cg + i;
#endif

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
#ifdef CONV_ARM_FP16
                                if (useFP16)
                                {
                                    for (int c = 0; c < Cg; c++, inpbuf_ki_FP16 += CONV_NR, inptr_ki += inp_planesize)
                                    {
                                        float32x4_t v0 = vld1q_f32(inptr_ki);
                                        float32x4_t v1 = vld1q_f32(inptr_ki + 4);

                                        vst1q_f16((__fp16* )inpbuf_ki_FP16, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
                                    }
                                }
                                else
#endif
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
#ifdef CONV_ARM_FP16
                                if (useFP16)
                                {
                                    for (int c = 0; c < Cg; c++, inpbuf_ki_FP16 += CONV_NR, inptr_ki += inp_planesize)
                                    {
                                        float32x4_t v0, v1;
                                        v0[0] = inptr_ki[0], v0[1] = inptr_ki[2];
                                        v0[2] = inptr_ki[4], v0[3] = inptr_ki[6];
                                        v1[0] = inptr_ki[8], v1[1] = inptr_ki[10];
                                        v1[2] = inptr_ki[12], v1[3] = inptr_ki[14];
                                        vst1q_f16((__fp16* )inpbuf_ki_FP16, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
                                    }
                                }
                                else
#endif
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
#ifdef CONV_ARM_FP16
                                if (useFP16)
                                {
                                    for (int c = 0; c < Cg; c++, inpbuf_ki_FP16 += CONV_NR, inptr_ki += inp_planesize)
                                    {
                                        float32x4_t v0, v1;

                                        v0[0] = inptr_ki[0], v0[1] = inptr_ki[stride_w];
                                        v0[2] = inptr_ki[stride_w * 2], v0[3] = inptr_ki[stride_w * 3];
                                        v1[0] = inptr_ki[stride_w * 4], v1[1] = inptr_ki[stride_w * 5];
                                        v1[2] = inptr_ki[stride_w * 6], v1[3] = inptr_ki[stride_w * 7];
                                        vst1q_f16((__fp16* )inpbuf_ki_FP16, vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
                                    }
                                }
                                else
#endif
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
#ifdef CONV_ARM_FP16
                                if (useFP16)
                                {
                                    for (int c = 0; c < Cg; c++, inpbuf_ki_FP16 += CONV_NR, inptr_ki += inp_planesize)
                                    {
                                        float32x4_t v0 = vld1q_f32(inptr_ki);
                                        vst1_f16((__fp16* )inpbuf_ki_FP16, vcvt_f16_f32(v0));
                                    }
                                }
                                else
#endif
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
#ifdef CONV_ARM_FP16
                                if (useFP16)
                                {
                                    for (int c = 0; c < Cg; c++, inpbuf_ki_FP16 += CONV_NR, inptr_ki += inp_planesize)
                                    {
                                        float32x4_t v0;
                                        v0[0] = inptr_ki[0], v0[1] = inptr_ki[stride_w];
                                        v0[2] = inptr_ki[stride_w * 2], v0[3] = inptr_ki[stride_w * 3];
                                        vst1_f16((__fp16* )inpbuf_ki_FP16, vcvt_f16_f32(v0));
                                    }
                                }
                                else
#endif
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
#ifdef CONV_ARM_FP16
                            if (useFP16)
                            {
                                for (int c = 0; c < Cg; c++, inpbuf_ki_FP16 += CONV_NR, inptr_ki += inp_planesize)
                                    inpbuf_ki_FP16[0] = (float16_t)(*inptr_ki);
                            }
                            else
#endif
                            for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR, inptr_ki += inp_planesize)
                                *inpbuf_ki = *inptr_ki;
                            i++;
                            x0++;
                        }
                    }
                    else
                    {
#ifdef CONV_ARM_FP16
                        if (useFP16)
                        {
                            for (int c = 0; c < Cg; c++, inpbuf_ki_FP16 += CONV_NR)
                                inpbuf_ki_FP16[0] = (float16_t)0.f;
                        }
                        else
#endif
                        for (int c = 0; c < Cg; c++, inpbuf_ki += CONV_NR)
                            inpbuf_ki[0] = 0.f;
                        i++;
                        x0++;
                    }

                    int mask = x0 >= W0;
                    y0 += mask;
                    x0 &= mask - 1;

                    mask = y0 >= H0; // Only Conv 3D need jump at z0 dimension
                    if (mask && conv_dim != CONV_3D)
                        break;

                    z0 += mask;
                    y0 &= mask - 1;
                }
            }
        }
    }
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

    const bool useFP16 = conv->useFP16;
    Mat fusedAddMat;
    if (fusedAdd)
    {
        CV_Assert(conv->conv_dim != CONV_3D && "Conv3D does not support Conv+Add fusion optimization!");
        fusedAddMat = _output.getMat();
    }

    if (conv->conv_type == CONV_TYPE_DEPTHWISE)
    {
        // Depthwise-Convolution layer should not be followed by Add layer.
        CV_Assert((conv_dim == CONV_1D || conv_dim == CONV_2D) && !useFP16);
        return runDepthwise(input, output, conv, actLayer.get(), reluslope, fusedAdd);
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

    // TODO: support FP16 for winograd.
    if (conv->conv_type == CONV_TYPE_WINOGRAD3X3) // winograd
    {
        CV_Assert(conv->weightsWinoBufPtr && input.dims == 4 && conv_dim == CONV_2D && !useFP16);
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
    bool fast_1x1 = ksize == 1 && stride_d == 1 && stride_w == 1 && stride_h == 1
            && pad_front == 0 && pad_left == 0 && pad_top == 0;
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

    int CONV_NR = CONV_NR_FP32;
    int CONV_MR = CONV_MR_FP32;
    int esz = sizeof(float );

#ifdef CONV_ARM_FP16
    if (useFP16)
    {
        // works at FP 16.
        CONV_NR = CONV_NR_FP16;
        CONV_MR = CONV_MR_FP16;
        esz = sizeof(float16_t);
    }
#endif

    int MAX_STRIPES = conv->conv_type == CONV_TYPE_DEPTHWISE_REMAIN ? 1 : (56 + CONV_NR - 1)/CONV_NR;

    // Friendly to L1 cache
    const int K_BLOCK_SIZE = conv->conv_type == CONV_TYPE_DEPTHWISE_REMAIN ? 1 : 32;
    const int C_BLOCK_SIZE = 256;

    int Kg_nblocks = (Kg + CONV_MR-1)/CONV_MR;
    int Kg_aligned = Kg_nblocks * CONV_MR;

    int stripes_per_plane0 = ((int)out_planesize + CONV_NR - 1) / CONV_NR;
    int stripes_per_plane = stripes_per_plane0;

    if (stripes_per_plane < ntasks * 4 || conv->conv_type == CONV_TYPE_DEPTHWISE_REMAIN)
    {
        MAX_STRIPES = 1;
        stripes_per_plane = 1;
    }
    else
        Kg_nblocks = 1;

    bool separateIm2col = fast_1x1 || stripes_per_plane == 1;

    int Kstripes = Kg_nblocks * stripes_per_plane;
    int nsubtasks = N * ngroups * Kstripes;

    size_t stripesize = alignSize(CONV_NR * ksize * Cg, VEC_ALIGN);
    size_t cbufsize = alignSize(CONV_NR * K_BLOCK_SIZE * MAX_STRIPES, VEC_ALIGN);

    size_t taskbufsize = cbufsize * sizeof(float );

    if (!separateIm2col)
        taskbufsize += MAX_STRIPES * stripesize * esz;

    size_t totalbufsize_base = taskbufsize * ntasks;
    size_t totalbufsize = totalbufsize_base;
    if (separateIm2col)
        totalbufsize += N * ngroups * stripes_per_plane0 * stripesize * esz;

    AutoBuffer<char> inpbuf_all_;
    char* inpbuf_all = nullptr;

    inpbuf_all_.allocate(totalbufsize + VEC_ALIGN * sizeof(float ));
    inpbuf_all = alignPtr(inpbuf_all_.data(), (int)(VEC_ALIGN * sizeof(float )));
    char* inpbuf_all_0 = inpbuf_all + totalbufsize_base;

    float* inp = input.ptr<float>();
    float* out = output.ptr<float>();
    float* fusedAddPtr0 = fusedAddMat.empty() ? 0 : fusedAddMat.ptr<float>();

    // In the case of 1x1 convolution we first reorder the whole input tensor.
    // In general, im2row results in Hk*Wk-x unrolling factor
    // (e.g. 3*3=9x unrolling for 3x3 convolution), thus for 1x1 convolution
    // the reordered tensor will take as much space as the original tensor.
    if (separateIm2col)
    {
        // the optional phase 1. im2row
        parallel_for_(Range(0, ntasks), [&](const Range& r0) {
        for (int task_id = r0.start; task_id < r0.end; task_id++)
        {
            if (fast_1x1)
            {
                int nc0 = task_id*N*C/ntasks, nc1 = (task_id+1)*N*C/ntasks, dc = 0;
                for (; nc0 < nc1; nc0 += dc)
                {
                    int n = nc0/C, c0 = nc0 - n*C;
                    int g = c0 / Cg;
                    c0 -= g*Cg;
                    dc = Cg - c0 <= nc1 - nc0 ? Cg - c0 : nc1 - nc0;

                    float * inptr_ = inp + (size_t)nc0*inp_planesize;
                    char* inpbuf_ = inpbuf_all_0 + ((n*ngroups + g)*stripes_per_plane0*stripesize + c0*CONV_NR)*esz;

                    packInputData(inpbuf_, inptr_, ofstab, dhwTab, 0, out_planesize, ksize, stride_d, stride_h,
                                  stride_w, pad_front, pad_top, pad_left, Dk, Hk, Wk, dilation_d, dilation_h, dilation_w,
                                  Di, Hi, Wi, H0, W0, dc, stripesize, 0, inp_planesize, conv->conv_dim,
                                  conv->conv_type, CONV_NR, esz, fast_1x1, useFP16);
                }
            }
            else
            {
                const int allTasks = N * ngroups * stripes_per_plane0;
                int ngs0 = task_id*allTasks/ntasks, ngs1 = (task_id+1)*allTasks/ntasks, ds = 0;

                for (; ngs0 < ngs1; ngs0 += ds)
                {
                    int n = ngs0 / (ngroups * stripes_per_plane0), gs0 = ngs0 - n*ngroups*stripes_per_plane0;
                    int g = gs0 / stripes_per_plane0, s0 = gs0 - g*stripes_per_plane0;

                    ds = stripes_per_plane0 - s0 <= ngs1 - ngs0 ? stripes_per_plane0 - s0 : ngs1 - ngs0;

                    int zyx = s0 * CONV_NR;
                    int zyx_limit = (s0 + ds) * CONV_NR < out_planesize ? (s0 + ds) * CONV_NR : out_planesize;

                    float * inptr_ = inp + (size_t)(n * ngroups + g) * Cg * inp_planesize;
                    char* inpbuf_ = inpbuf_all_0 + ((n * ngroups + g) * stripes_per_plane0 * stripesize + s0 * stripesize) * esz;

                    packInputData(inpbuf_, inptr_, ofstab, dhwTab, zyx, zyx_limit, ksize, stride_d, stride_h,
                                  stride_w, pad_front, pad_top, pad_left, Dk, Hk, Wk, dilation_d, dilation_h, dilation_w,
                                  Di, Hi, Wi, H0, W0, Cg, stripesize, 0, inp_planesize, conv->conv_dim,
                                  conv->conv_type, CONV_NR, esz, fast_1x1, useFP16);
                }
            }
        }
        });
    }

    // Compute
    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        float * cbuf_task = (float *)(inpbuf_all + taskbufsize * task_id);
        char * inpbuf_task = (char*)(cbuf_task + cbufsize);

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

            if (stripes_per_plane == 1 || conv->conv_type == CONV_TYPE_DEPTHWISE_REMAIN)
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

                CV_Assert(nstripes <= MAX_STRIPES);

                if (!separateIm2col)
                {
                    packInputData(inpbuf_task, inp, ofstab, dhwTab, zyx0, zyx_block_limit, ksize, stride_d, stride_h,
                                  stride_w, pad_front, pad_top, pad_left, Dk, Hk, Wk, dilation_d, dilation_h, dilation_w,
                                  Di, Hi, Wi, H0, W0, Cg, stripesize, inp_plane_ofs, inp_planesize, conv->conv_dim,
                                  conv->conv_type, CONV_NR, esz, fast_1x1, useFP16);
                }

                char *weights = nullptr;
#ifdef CONV_ARM_FP16
                if (useFP16)
                {
                    CV_Assert(!conv->weightsBuf_FP16.empty());
                    weights = (char *)conv->weightsBufPtr_FP16;
                }
                else
#endif
                {
                    CV_Assert(!conv->weightsBuf.empty());
                    weights = (char *)conv->weightsBufPtr;
                }
                // optional branch, only for depth-wise convolution which was implemented by generic convolution.
                // In this case, CONV_MR is 1, and CONV_NR remains the same.
                if (conv->conv_type == CONV_TYPE_DEPTHWISE_REMAIN)
                {
                    CV_Assert(weights);
                    size_t outofs = (n * ngroups + g) * out_planesize + zyx0;
                    float *cptr0 = cbuf_task;
                    weights += g * padded_ksize * esz;

                    int out_width = zyx_block_limit - zyx0;
                    float *outptr = out + outofs;
                    const float biasVal = *(conv->biasBuf.data() + g);
                    const char *inptr_ = separateIm2col ? inpbuf_all_0 + (ng * stripes_per_plane0 + zyx0 / CONV_NR) * stripesize * esz :
                                         inpbuf_task;

                    for (int stripe = 0; stripe < nstripes; stripe++)
                    {
                        const char *inptr = inptr_ + stripe * stripesize * esz;
                        const int outLen = std::min(out_width - stripe * CONV_NR, CONV_NR);
                        bool ifBuffer = outLen < CONV_NR;
                        float *cptr = outptr + stripe * CONV_NR;
                        if (ifBuffer)
                        {
                            memcpy(cptr0, cptr, outLen * sizeof(float ));
                            cptr = cptr0;
                        }
#if CV_NEON && CV_NEON_AARCH64
                        if (conv->useNEON)
                        {
#ifdef CONV_ARM_FP16
                            if (useFP16)
                            {
                                opt_NEON::convBlockMR1_FP16(DkHkWkCg, weights, inptr, cptr, biasVal, fusedAdd, minval, maxval, ifMinMaxAct, outLen, CONV_NR);
                            }
                            else
#endif
                            opt_NEON::convBlockMR1_F32(DkHkWkCg, (const float *)weights, (const float *)inptr, cptr, biasVal, fusedAdd, minval, maxval, ifMinMaxAct, outLen, CONV_NR);
                        }
                        else
#endif
                        convBlockMR1(DkHkWkCg, (const float *)weights, (const float *)inptr, cptr, biasVal, fusedAdd, minval, maxval, ifMinMaxAct, outLen, CONV_NR);

                        if (ifBuffer)
                        {
                            memcpy(outptr + stripe * CONV_NR, cptr, outLen * sizeof(float ));
                        }
                    }
                    if (activ)
                        activ->forwardSlice(outptr, outptr, out_width, out_planesize, g, g + 1);
                    continue;
                }

                CV_Assert(weights);
                weights += g * Kg_aligned * DkHkWkCg * esz;

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
                        const char *inptr = separateIm2col ? inpbuf_all_0 + (ng * stripes_per_plane0 + zyx0 / CONV_NR) * stripesize * esz :
                                            inpbuf_task;
                        inptr += (c0 * CONV_NR) * esz;
                        for (int stripe = 0; stripe < nstripes; stripe++, inptr += stripesize * esz)
                        {
                            const int outLen = std::min(out_width - stripe * CONV_NR, CONV_NR);

                            char *wptr = weights + (k0_block * DkHkWkCg + c0 * CONV_MR) * esz;
                            float *cptr = cbuf_task + stripe * CONV_NR;
                            float16_t* cptr_f16 = (float16_t*)cbuf_task + stripe*CONV_NR;
                            for (int k = k0_block; k < k1_block; k += CONV_MR,
                                    wptr += DkHkWkCg * CONV_MR * esz, cptr += CONV_MR * ldc, cptr_f16 += CONV_MR * ldc)
                            {
#if CV_TRY_AVX2
                                if (conv->useAVX2)
                                    opt_AVX2::convBlock(c1 - c0, (const float *)wptr, (const float *)inptr, cptr, ldc, c0 == 0, outLen, CONV_MR, CONV_NR);
                                else
#endif
#if CV_TRY_AVX
                                if (conv->useAVX)
                                    opt_AVX::convBlock(c1 - c0, (const float *)wptr, (const float *)inptr, cptr, ldc, c0 == 0, outLen, CONV_MR, CONV_NR);
                                else
#endif
#if CV_NEON
                                if (conv->useNEON)
                                {
#ifdef CONV_ARM_FP16
                                    if (useFP16)
                                    {
                                        opt_NEON::convBlock_FP16(c1 - c0, wptr, inptr, (char *)cptr_f16, ldc, c0 == 0, outLen, CONV_MR, CONV_NR);
                                    }
                                    else
#endif
                                    opt_NEON::convBlock(c1 - c0, (const float *)wptr, (const float *)inptr, cptr, ldc, c0 == 0, outLen, CONV_MR, CONV_NR);
                                }
                                else
#endif
                                // The possible outLen range is 24 or 8~1.
                                convBlock(c1 - c0, (const float *)wptr, (const float *)inptr, cptr, ldc, c0 == 0, outLen, CONV_MR, CONV_NR);
                            }
                        }
                    }

                    size_t outofs = ((n * ngroups + g) * Kg + k0_block) * out_planesize + zyx0;
                    const float *cptr = cbuf_task;
                    const float16_t *cptr_fp16 = (const float16_t *)cbuf_task;
                    float *outptr = out + outofs;
                    const float *pbptr = fusedAddPtr0 ? fusedAddPtr0 + outofs : 0;

                    for (int k = k0_block; k < k1_block; k++,
                            cptr += ldc, cptr_fp16 += ldc, outptr += out_planesize,
                            pbptr += (pbptr ? out_planesize : 0))
                    {
                        float biasval = biasptr[k];
                        int j = 0;

#ifdef CONV_ARM_FP16
                        if (useFP16)
                        {
                            float32x4_t vbias = vdupq_n_f32(biasval);
                            float32x4_t vmax = vdupq_n_f32(maxval);
                            float32x4_t vmin = vdupq_n_f32(minval);
                            if (pbptr)
                            {
                                for (; j + 7 < out_width; j += 8)
                                {
                                    float32x4_t v0 = vcvt_f32_f16(vld1_f16((const __fp16 *)cptr_fp16 + j)) + vbias;
                                    float32x4_t v1 = vcvt_f32_f16(vld1_f16((const __fp16 *)cptr_fp16 + + j + 4)) + vbias;

                                    v0 += vld1q_f32(pbptr + j);
                                    v1 += vld1q_f32(pbptr + j + 4);

                                    if (ifMinMaxAct)
                                    {
                                        v0 = vminq_f32(vmaxq_f32(v0, vmin), vmax);
                                        v1 = vminq_f32(vmaxq_f32(v1, vmin), vmax);
                                    }

                                    vst1q_f32(outptr + j, v0);
                                    vst1q_f32(outptr + j + 4, v1);
                                }
                            }
                            else
                            {
                                for (; j + 7 < out_width; j += 8)
                                {
                                    float32x4_t v0 = vcvt_f32_f16(vld1_f16((const __fp16 *)cptr_fp16 + j)) + vbias;
                                    float32x4_t v1 = vcvt_f32_f16(vld1_f16((const __fp16 *)cptr_fp16 + j + 4)) + vbias;

                                    if (ifMinMaxAct)
                                    {
                                        v0 = vminq_f32(vmaxq_f32(v0, vmin), vmax);
                                        v1 = vminq_f32(vmaxq_f32(v1, vmin), vmax);
                                    }

                                    vst1q_f32(outptr + j, v0);
                                    vst1q_f32(outptr + j + 4, v1);
                                }
                            }

                            if (pbptr)
                            {
                                for (; j < out_width; j++)
                                {
                                    float v = (float )cptr_fp16[j] + biasval;
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
                                    float v = (float )cptr_fp16[j] + biasval;

                                    if (ifMinMaxAct)
                                        v = std::min(std::max(v, minval), maxval);
                                    outptr[j] = v;
                                }
                            }
                        }
                        else
#endif
                        {
#if CV_SIMD128
                            v_float32x4 vbias = v_setall_f32(biasval);
                            v_float32x4 vmax = v_setall_f32(maxval);
                            v_float32x4 vmin = v_setall_f32(minval);

                            if (pbptr)
                            {
                                for (; j + 7 < out_width; j += 8)
                                {
                                    v_float32x4 v0 = v_add(v_load(cptr + j), vbias);
                                    v_float32x4 v1 = v_add(v_load(cptr + j + 4), vbias);

                                    v0 = v_add(v0, v_load(pbptr + j));
                                    v1 = v_add(v1, v_load(pbptr + j + 4));

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
                                    v_float32x4 v0 = v_add(v_load(cptr + j), vbias);
                                    v_float32x4 v1 = v_add(v_load(cptr + j + 4), vbias);

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
                            if (pbptr)
                            {
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


/****************************************************************************************\
                                    SIMD and no-SIMD code for convBlock
\****************************************************************************************/

static inline void convBlockMR1NoSIMD(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
                               const float minval, const float maxval, bool ifMinMaxAct, const int outLen, const int convNR)
{
    std::vector<float> cbuffer(outLen, 0);
    float* cbuf = cbuffer.data();
    for( int p = 0; p < np; p++ )
    {
        float ai = a[p];
        for( int j = 0; j < outLen; j++ )
            cbuf[j] += b[convNR*p + j] * ai;
    }

    if (init_c)
    {
        for(int j = 0; j < outLen; j++)
        {
            c[j] += cbuf[j] + bias;
            if (ifMinMaxAct)
                c[j] = std::min(std::max(c[j], minval), maxval);
        }
    }
    else
    {
        for(int j = 0; j < outLen; j++)
        {
            c[j] = cbuf[j] + bias;
            if (ifMinMaxAct)
                c[j] = std::min(std::max(c[j], minval), maxval);
        }
    }
}

#if CV_SIMD128
static inline void convBlockMR1x24(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
                            const float minval, const float maxval, bool ifMinMaxAct, const int convNR)
{
    CV_Assert(convNR == 24);
    v_float32x4 c0  = v_setall_f32(bias), c1 = c0, c2 = c0;
    v_float32x4 c3 = c0, c4 = c0, c5 = c0;

    for (int p = 0; p < np; p++, a++, b += convNR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4), b2 = v_load(b + 8);
        v_float32x4 b3 = v_load(b + 12), b4 = v_load(b + 16), b5 = v_load(b + 20);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b1, a0, c1);
        c2 = v_fma(b2, a0, c2);
        c3 = v_fma(b3, a0, c3);
        c4 = v_fma(b4, a0, c4);
        c5 = v_fma(b5, a0, c5);
    }

    if (init_c)
    {
        c0 = v_add(c0, v_load(c));
        c1 = v_add(c1, v_load(c + 4));
        c2 = v_add(c2, v_load(c + 8));
        c3 = v_add(c3, v_load(c + 12));
        c4 = v_add(c4, v_load(c + 16));
        c5 = v_add(c5, v_load(c + 20));
    }

    if (ifMinMaxAct)
    {
        v_float32x4 vmax = v_setall_f32(maxval), vmin = v_setall_f32(minval);
        c0 = v_min(v_max(c0, vmin), vmax);
        c1 = v_min(v_max(c1, vmin), vmax);
        c2 = v_min(v_max(c2, vmin), vmax);
        c3 = v_min(v_max(c3, vmin), vmax);
        c4 = v_min(v_max(c4, vmin), vmax);
        c5 = v_min(v_max(c5, vmin), vmax);
    }

    v_store(c, c0);
    v_store(c + 4, c1);
    v_store(c + 8, c2);
    v_store(c + 12, c3);
    v_store(c + 16, c4);
    v_store(c + 20, c5);
}

static inline void convBlockMR1x12(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
                            const float minval, const float maxval, bool ifMinMaxAct, const int convNR)
{
    CV_Assert(convNR == 12);
    v_float32x4 c0  = v_setall_f32(bias), c1 = c0, c2 = c0;
    for (int p = 0; p < np; p++, a++, b += convNR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4), b2 = v_load(b + 8);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b1, a0, c1);
        c2 = v_fma(b2, a0, c2);
    }

    if (init_c)
    {
        c0 = v_add(c0, v_load(c));
        c1 = v_add(c1, v_load(c + 4));
        c2 = v_add(c2, v_load(c + 8));
    }

    if (ifMinMaxAct)
    {
        v_float32x4 vmax = v_setall_f32(maxval), vmin = v_setall_f32(minval);
        c0 = v_min(v_max(c0, vmin), vmax);
        c1 = v_min(v_max(c1, vmin), vmax);
        c2 = v_min(v_max(c2, vmin), vmax);
    }

    v_store(c, c0);
    v_store(c + 4, c1);
    v_store(c + 8, c2);
}
#endif

void convBlockMR1(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
                  const float minval, const float maxval, bool ifMinMaxAct, const int outLen, const int convNR)
{
#if CV_SIMD128
    // The outLen represents the valid output value in CONV_NR length.
    // When outLen is very small, we use the no-SIMD branch.
    const int convNRby3 = convNR/3;
    if (outLen > convNRby3)
    {
        if (convNR == 24)
            convBlockMR1x24(np, a, b, c, bias, init_c, minval, maxval, ifMinMaxAct, convNR);
        else if (convNR == 12)
            convBlockMR1x12(np, a, b, c, bias, init_c, minval, maxval, ifMinMaxAct, convNR);
        else
            convBlockMR1NoSIMD(np, a, b, c, bias, init_c, minval, maxval, ifMinMaxAct, outLen, convNR);
    }
     else
        convBlockMR1NoSIMD(np, a, b, c, bias, init_c, minval, maxval, ifMinMaxAct, outLen, convNR);
#else
    convBlockMR1NoSIMD(np, a, b, c, bias, init_c, minval, maxval, ifMinMaxAct, outLen, convNR);
#endif
}

#if CV_SIMD128
static inline void convBlock4x24(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR)
{
    v_float32x4 c0  = v_setzero_f32(), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0;
    v_float32x4 c6  = v_setzero_f32(), c7 = c6, c8 = c6, c9 = c6, c10 = c6, c11 = c6;
    v_float32x4 c12 = v_setzero_f32(), c13 = c12, c14 = c12, c15 = c12, c16 = c12, c17 = c12;
    v_float32x4 c18 = v_setzero_f32(), c19 = c18, c20 = c18, c21 = c18, c22 = c18, c23 = c18;

    for (int p = 0; p < np; p++, a += convMR, b += convNR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4), b2 = v_load(b + 8);
        v_float32x4 b3 = v_load(b + 12), b4 = v_load(b + 16), b5 = v_load(b + 20);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b1, a0, c1);
        c2 = v_fma(b2, a0, c2);
        c3 = v_fma(b3, a0, c3);
        c4 = v_fma(b4, a0, c4);
        c5 = v_fma(b5, a0, c5);

        a0  = v_setall_f32(a[1]);
        c6  = v_fma(b0, a0, c6);
        c7  = v_fma(b1, a0, c7);
        c8  = v_fma(b2, a0, c8);
        c9  = v_fma(b3, a0, c9);
        c10 = v_fma(b4, a0, c10);
        c11 = v_fma(b5, a0, c11);

        a0 = v_setall_f32(a[2]);
        c12 = v_fma(b0, a0, c12);
        c13 = v_fma(b1, a0, c13);
        c14 = v_fma(b2, a0, c14);
        c15 = v_fma(b3, a0, c15);
        c16 = v_fma(b4, a0, c16);
        c17 = v_fma(b5, a0, c17);

        a0 = v_setall_f32(a[3]);
        c18 = v_fma(b0, a0, c18);
        c19 = v_fma(b1, a0, c19);
        c20 = v_fma(b2, a0, c20);
        c21 = v_fma(b3, a0, c21);
        c22 = v_fma(b4, a0, c22);
        c23 = v_fma(b5, a0, c23);
    }

    if (!init_c)
    {
        c0 = v_add(c0, v_load(c));
        c1 = v_add(c1, v_load(c + 4));
        c2 = v_add(c2, v_load(c + 8));
        c3 = v_add(c3, v_load(c + 12));
        c4 = v_add(c4, v_load(c + 16));
        c5 = v_add(c5, v_load(c + 20));

        c6  = v_add(c6 , v_load(c + ldc));
        c7  = v_add(c7 , v_load(c + ldc + 4));
        c8  = v_add(c8 , v_load(c + ldc + 8));
        c9  = v_add(c9 , v_load(c + ldc + 12));
        c10 = v_add(c10, v_load(c + ldc + 16));
        c11 = v_add(c11, v_load(c + ldc + 20));

        c12 = v_add(c12, v_load(c + ldc*2));
        c13 = v_add(c13, v_load(c + ldc*2 + 4));
        c14 = v_add(c14, v_load(c + ldc*2 + 8));
        c15 = v_add(c15, v_load(c + ldc*2 + 12));
        c16 = v_add(c16, v_load(c + ldc*2 + 16));
        c17 = v_add(c17, v_load(c + ldc*2 + 20));

        c18 = v_add(c18, v_load(c + ldc*3));
        c19 = v_add(c19, v_load(c + ldc*3 + 4));
        c20 = v_add(c20, v_load(c + ldc*3 + 8));
        c21 = v_add(c21, v_load(c + ldc*3 + 12));
        c22 = v_add(c22, v_load(c + ldc*3 + 16));
        c23 = v_add(c23, v_load(c + ldc*3 + 20));
    }

    v_store(c, c0);
    v_store(c + 4, c1);
    v_store(c + 8, c2);
    v_store(c + 12, c3);
    v_store(c + 16, c4);
    v_store(c + 20, c5);

    v_store(c + ldc, c6);
    v_store(c + ldc + 4, c7);
    v_store(c + ldc + 8, c8);
    v_store(c + ldc + 12, c9);
    v_store(c + ldc + 16, c10);
    v_store(c + ldc + 20, c11);

    v_store(c + ldc * 2, c12);
    v_store(c + ldc * 2 + 4, c13);
    v_store(c + ldc * 2 + 8, c14);
    v_store(c + ldc * 2 + 12, c15);
    v_store(c + ldc * 2 + 16, c16);
    v_store(c + ldc * 2 + 20, c17);

    v_store(c + ldc * 3, c18);
    v_store(c + ldc * 3 + 4, c19);
    v_store(c + ldc * 3 + 8, c20);
    v_store(c + ldc * 3 + 12, c21);
    v_store(c + ldc * 3 + 16, c22);
    v_store(c + ldc * 3 + 20, c23);
}

static inline void convBlock4x8(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR)
{
    CV_Assert(convNR >= 4);
    v_float32x4 c0  = v_setzero_f32(), c1 = c0, c2 = c0, c3 = c0;
    v_float32x4 c4 = c0, c5 = c0, c6 = c0, c7 = c0;

    for (int p = 0; p < np; p++, a += convMR, b += convNR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 a1 = v_setall_f32(a[1]);
        v_float32x4 a2 = v_setall_f32(a[2]);
        v_float32x4 a3 = v_setall_f32(a[3]);

        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b1, a0, c1);

        c2 = v_fma(b0, a1, c2);
        c3 = v_fma(b1, a1, c3);

        c4 = v_fma(b0, a2, c4);
        c5 = v_fma(b1, a2, c5);

        c6  = v_fma(b0, a3, c6);
        c7  = v_fma(b1, a3, c7);
    }

    if (!init_c)
    {
        c0 = v_add(c0, v_load(c));
        c1 = v_add(c1, v_load(c + 4));

        c2 = v_add(c2, v_load(c + ldc));
        c3 = v_add(c3, v_load(c + ldc + 4));

        c4 = v_add(c4, v_load(c + ldc*2));
        c5 = v_add(c5, v_load(c + ldc*2 + 4));

        c6 = v_add(c6, v_load(c + ldc*3));
        c7 = v_add(c7, v_load(c + ldc*3 + 4));
    }

    v_store(c, c0);
    v_store(c + 4, c1);
    v_store(c + ldc, c2);
    v_store(c + ldc + 4, c3);
    v_store(c + ldc * 2, c4);
    v_store(c + ldc * 2 + 4, c5);
    v_store(c + ldc * 3, c6);
    v_store(c + ldc * 3 + 4, c7);
}

static inline void convBlock4x4(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR)
{
    CV_Assert(convNR >= 4);
    v_float32x4 c0  = v_setzero_f32(), c1 = c0, c2 = c0, c3 = c0;

    for (int p = 0; p < np; p++, a += convMR, b += convNR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 a1 = v_setall_f32(a[1]);
        v_float32x4 a2 = v_setall_f32(a[2]);
        v_float32x4 a3 = v_setall_f32(a[3]);

        v_float32x4 b0 = v_load(b);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b0, a1, c1);
        c2 = v_fma(b0, a2, c2);
        c3 = v_fma(b0, a3, c3);
    }

    if (!init_c)
    {
        c0 = v_add(c0, v_load(c));
        c1 = v_add(c1, v_load(c + ldc));
        c2 = v_add(c2, v_load(c + ldc*2));
        c3 = v_add(c3, v_load(c + ldc*3));
    }

    v_store(c, c0);
    v_store(c + ldc, c1);
    v_store(c + ldc * 2, c2);
    v_store(c + ldc * 3, c3);
}
#endif

static inline void convBlockNoSIMD(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int outLen,
                            const int convMR, const int convNR)
{
    std::vector<float> cbuffer(convMR * outLen, 0);
    float* cbuf = cbuffer.data();
    for( int p = 0; p < np; p++ )
    {
        for( int i = 0; i < convMR; i++ )
        {
            float ai = a[convMR*p + i];
            for( int j = 0; j < outLen; j++ )
                cbuf[i * outLen+j] += b[convNR*p + j] * ai;
        }
    }

    if (!init_c)
    {
        for(int i = 0; i < convMR; i++)
        {
            for(int j = 0; j < outLen; j++)
                c[i*ldc + j] += cbuf[i*outLen + j];
        }
    }
    else
    {
        for(int i = 0; i < convMR; i++)
        {
            for(int j = 0; j < outLen; j++)
                c[i*ldc + j] = cbuf[i*outLen + j];
        }
    }
}

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int outLen,
               const int convMR, const int convNR)
{
    // The possible outLen range is [24, 8~1].
#if CV_SIMD128
    CV_Assert(convMR == 4);
    if (outLen > 8 && convNR == 24)
    {
        convBlock4x24(np, a, b, c, ldc, init_c, convMR, convNR);
        return;
    }

    if (outLen <= 8 && outLen > 4)
    {
        convBlock4x8(np, a, b, c, ldc, init_c, convMR, convNR);
        return;
    }

    if (outLen <= 4 && outLen > 1)
    {
        convBlock4x4(np, a, b, c, ldc, init_c, convMR, convNR);
        return;
    }
    convBlockNoSIMD(np, a, b, c, ldc, init_c, outLen, convMR, convNR);
#else
    convBlockNoSIMD(np, a, b, c, ldc, init_c, outLen, convMR, convNR);
#endif
}

}} // namespace cv::dnn
