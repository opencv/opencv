// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpConv_Winograd.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "convolution.hpp"

#include "conv_winograd_f63.simd.hpp"
#include "layers/cpu_kernels/conv_winograd_f63.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv { namespace dnn {

#if CV_NEON || CV_SIMD128 || CV_TRY_AVX2
enum { VEC_ALIGN = 32, DFT_TYPE = CV_32F }; // Memory alignment.

void winofunc_accum_f32(const float* inwptr, const float* wptr, float* outbuf, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32);

/*Input transform*/
void winofunc_BtXB_8x8_f32(const float* inptr, int inpstep,
                          float* outptr, int Cg, const int winoIblock, const int winoAtomF32);

/*Output transform*/
void winofunc_AtXA_8x8_f32(const float* inptr, int inpstep, float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct);

int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv>& conv,
                  int ntasks, float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();
    Mat fusedAddMat = _fusedAddMat.getMat();

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);
    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->K;
    int H0 = outputShape[2], W0 = outputShape[3];

    int pad_top = conv->pad_top;
    int pad_left = conv->pad_left;

    int ngroups = conv->ngroups, Cg = C/ngroups, Kg = K/ngroups;

    const int CONV_WINO_KBLOCK = 4;
#if (CV_NEON && CV_NEON_AARCH64)
    const int CONV_WINO_IBLOCK = 6;
#elif  CV_TRY_AVX || CV_TRY_AVX2
    const int CONV_WINO_IBLOCK = (conv->useAVX || conv->useAVX2) ? 6 : 3;
#else
    const int CONV_WINO_IBLOCK = 3;
#endif

#if CV_TRY_AVX || CV_TRY_AVX2
    const int CONV_WINO_ATOM_F32 = (conv->useAVX || conv->useAVX2) ? 8 : 4;
#else
    const int CONV_WINO_ATOM_F32 = 4;
#endif
    const int CONV_WINO_NATOMS_F32 = CONV_WINO_AREA / CONV_WINO_ATOM_F32; // for AVX2, it is 8, otherwise, it's 16.

    int Kg_nblocks = (Kg + CONV_WINO_KBLOCK - 1)/CONV_WINO_KBLOCK;
    const size_t inp_planesize = (size_t)Hi*Wi;
    const size_t out_planesize = (size_t)H0*W0;

    int blocks_per_row = (W0+CONV_WINO_STEP-1)/CONV_WINO_STEP;
    int blocks_per_plane = ((H0+CONV_WINO_STEP-1)/CONV_WINO_STEP)*blocks_per_row;
    int blocks_per_plane_aligned = ((blocks_per_plane +
                                     CONV_WINO_IBLOCK-1)/CONV_WINO_IBLOCK)*CONV_WINO_IBLOCK;

    size_t totalbufsize = (size_t)N*C*blocks_per_plane_aligned*CONV_WINO_AREA;

    AutoBuffer<float> _buf;
    _buf.allocate(totalbufsize + VEC_ALIGN);
    float* wbuf_all = alignPtr(_buf.data(), VEC_ALIGN);

    float* inp = input.ptr<float>();
    float* out = output.ptr<float>();

    float* fusedAddPtr = fusedAddMat.empty() ? nullptr : fusedAddMat.ptr<float>();

    // Phase 1. compute forward Winograd transforms for all input blocks,
    // all input planes, all samples in the batch.
    // [TODO]: maybe, if there are too many input channels, it makes sense to
    // transform only part of input channels at once and then compute the partial
    // accumulated sums (i.e. update the output buffers several times,
    // rather than compute them in one pass).
    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        int nc0 = (N*C)*task_id/ntasks;
        int nc1 = (N*C)*(task_id+1)/ntasks;
        for(; nc0 < nc1; nc0++)
        {
            int n = nc0 / C;
            int c = nc0 - n*C;
            int g = c / Cg;
            c -= g*Cg;
            for (int block_id = 0; block_id < blocks_per_plane; block_id += CONV_WINO_IBLOCK)
            {
                for (int db = 0; db < CONV_WINO_IBLOCK; db++)
                {
                    size_t inwofs = ((n*ngroups + g)*blocks_per_plane_aligned +
                                     block_id)*Cg*CONV_WINO_AREA +
                                    (c*CONV_WINO_IBLOCK + db)*CONV_WINO_ATOM_F32;
                    float* inwptr = (float*)wbuf_all + inwofs;

                    if (block_id + db < blocks_per_plane)
                    {
                        int y0 = (block_id + db) / blocks_per_row;
                        int x0 = (block_id + db) - y0 * blocks_per_row;
                        y0 = y0*CONV_WINO_STEP - pad_top;
                        x0 = x0*CONV_WINO_STEP - pad_left;
                        bool partial = y0 < 0 || y0 + CONV_WINO_SIZE > Hi ||
                                       x0 < 0 || x0 + CONV_WINO_SIZE > Wi;
                        int dx1 = 0, dx2 = CONV_WINO_SIZE, dy1 = 0, dy2 = CONV_WINO_SIZE;
                        int inpstep = Wi;

                        float inpbuf[CONV_WINO_AREA];
                        float* inptr0 = (float*)inp + nc0*inp_planesize + y0*Wi + x0;
                        float* inptr = inptr0;

                        if (partial)
                        {
                            memset(inpbuf, 0, sizeof(inpbuf));
                            dy1 = -y0 > 0 ? -y0 : 0;
                            dy2 = Hi - y0 < CONV_WINO_SIZE ? Hi - y0 : CONV_WINO_SIZE;

                            if (dy2 < dy1) {dy2 = dy1 = 0;}
                            dx1 = -x0 > 0 ? -x0 : 0;
                            dx2 = Wi - x0 < CONV_WINO_SIZE ? Wi - x0 : CONV_WINO_SIZE;

                            if (dx2 < dx1) {dx2 = dx1 = 0;}
                            inptr0 -= y0*Wi + x0;

                            if (dx1 < dx2 && dy1 < dy2)
                            {
                                for(int dy = dy1; dy < dy2; dy++)
                                    memcpy(&inpbuf[dy*CONV_WINO_SIZE + dx1],
                                           inptr0 + (y0+dy)*Wi + (x0+dx1),
                                           (dx2-dx1)*sizeof(inpbuf[0]));
                            }

                            inptr = inpbuf;
                            inpstep = CONV_WINO_SIZE;
                        }
#if CV_TRY_AVX2
                        if (conv->useAVX2)
                            opt_AVX2::winofunc_BtXB_8x8_f32(inptr, inpstep, inwptr, Cg, CONV_WINO_IBLOCK, CONV_WINO_ATOM_F32);
                        else
#endif
#if CV_TRY_AVX
                        if (conv->useAVX)
                            opt_AVX::winofunc_BtXB_8x8_f32(inptr, inpstep, inwptr, Cg, CONV_WINO_IBLOCK, CONV_WINO_ATOM_F32);
                        else
#endif
#if CV_NEON && CV_NEON_AARCH64
                        if (conv->useNEON)
                            opt_NEON::winofunc_BtXB_8x8_f32(inptr, inpstep, inwptr, Cg, CONV_WINO_IBLOCK, CONV_WINO_ATOM_F32);
                        else
#endif
                        winofunc_BtXB_8x8_f32(inptr, inpstep, inwptr, Cg, CONV_WINO_IBLOCK, CONV_WINO_ATOM_F32);
                    }
                    else
                    {
                        for (int i = 0; i < CONV_WINO_NATOMS_F32; i++, inwptr += CONV_WINO_IBLOCK*CONV_WINO_ATOM_F32)
                            memset(inwptr, 0, CONV_WINO_ATOM_F32*sizeof(inwptr[0]));
                    }
                }
            }
        }
    }});

    // Phase 2. compute elemwise-weighted sums of transformed blocks,
    // apply inverse Winograd transforms to the sums,
    // add bias, apply activation function if any and store the results.
    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        size_t out_wbuf_size = CONV_WINO_AREA*CONV_WINO_KBLOCK*CONV_WINO_IBLOCK;
        size_t outbuf_size = CONV_WINO_AREA;
        AutoBuffer<float> out_wbuf_, outbuf_;
        out_wbuf_.allocate(out_wbuf_size + VEC_ALIGN);
        float* out_wbuf = alignPtr(out_wbuf_.data(), VEC_ALIGN);
        outbuf_.allocate(outbuf_size + VEC_ALIGN);
        float* outbuf = alignPtr(outbuf_.data(), VEC_ALIGN);

        memset(out_wbuf, 0, out_wbuf_size * sizeof(float));
        memset(outbuf, 0, outbuf_size * sizeof(float));

        int ngk0 = (int)(((int64_t)N*Kg_nblocks*ngroups)*task_id/ntasks);
        int ngk1 = (int)(((int64_t)N*Kg_nblocks*ngroups)*(task_id+1)/ntasks);

        for(; ngk0 < ngk1; ngk0++)
        {
            int n = ngk0 / (Kg_nblocks*ngroups);
            int gk0 = ngk0 % (Kg_nblocks*ngroups);
            int g = gk0 / Kg_nblocks;
            int k0 = (gk0 % Kg_nblocks)*CONV_WINO_KBLOCK;
            int k1 = k0 + CONV_WINO_KBLOCK <= Kg ? k0 + CONV_WINO_KBLOCK : Kg;

            for (int block_id0 = 0; block_id0 < blocks_per_plane; block_id0 += CONV_WINO_IBLOCK)
            {
                int block_id1 = block_id0 + CONV_WINO_IBLOCK;
                block_id1 = block_id1 < blocks_per_plane ? block_id1 : blocks_per_plane;
                size_t inwofs = ((n*ngroups + g)*blocks_per_plane_aligned + block_id0)*Cg*CONV_WINO_AREA;
                size_t wofs = (g*Kg_nblocks*CONV_WINO_KBLOCK + k0)*Cg*CONV_WINO_AREA;

                float* inwptr = wbuf_all + inwofs;
                const float* wptr = conv->weightsWinoBufPtr + wofs;

#if CV_TRY_AVX2
                if (conv->useAVX2)
                    opt_AVX2::winofunc_accum_f32(inwptr, wptr, out_wbuf, Cg, block_id1 - block_id0, CONV_WINO_IBLOCK,
                                       CONV_WINO_KBLOCK, CONV_WINO_ATOM_F32, CONV_WINO_NATOMS_F32);
                else
#endif
#if CV_TRY_AVX
                if (conv->useAVX)
                    opt_AVX::winofunc_accum_f32(inwptr, wptr, out_wbuf, Cg, block_id1 - block_id0, CONV_WINO_IBLOCK,
                                       CONV_WINO_KBLOCK, CONV_WINO_ATOM_F32, CONV_WINO_NATOMS_F32);
                else
#endif
#if CV_NEON && CV_NEON_AARCH64
                if (conv->useNEON)
                    opt_NEON::winofunc_accum_f32(inwptr, wptr, out_wbuf, Cg, block_id1 - block_id0, CONV_WINO_IBLOCK,
                                       CONV_WINO_KBLOCK, CONV_WINO_ATOM_F32, CONV_WINO_NATOMS_F32);
                else
#endif

                winofunc_accum_f32(inwptr, wptr, out_wbuf, Cg, block_id1 - block_id0, CONV_WINO_IBLOCK,
                                       CONV_WINO_KBLOCK, CONV_WINO_ATOM_F32, CONV_WINO_NATOMS_F32);
                for (int k = k0; k < k1; k++)
                {
                    float biasv = conv->biasBuf[g*Kg + k];
                    for (int block_id = block_id0; block_id < block_id1; block_id++)
                    {
                        int y0 = block_id / blocks_per_row;
                        int x0 = block_id - y0 * blocks_per_row;
                        y0 = y0*CONV_WINO_STEP;
                        x0 = x0*CONV_WINO_STEP;
                        int dy1 = H0 - y0;
                        if (dy1 > CONV_WINO_STEP) dy1 = CONV_WINO_STEP;
                        int dx1 = W0 - x0;
                        if (dx1 > CONV_WINO_STEP) dx1 = CONV_WINO_STEP;
                        assert(dx1 > 0 && dy1 > 0);
                        bool partial = activ || dy1 < CONV_WINO_STEP || dx1 < CONV_WINO_STEP;
                        size_t outofs = (n*K + g*Kg + k)*out_planesize + y0*W0 + x0;
                        int outstep = W0;

                        float* outptr0 = (float*)out + outofs;
                        float* pbptr0 = fusedAddPtr ? fusedAddPtr + outofs : nullptr;
                        float *outptr = outptr0, *bpptr = pbptr0;

                        if (partial)
                        {
                            outptr = outbuf;
                            outstep = CONV_WINO_SIZE;
                            if (pbptr0)
                            {
                                bpptr = outbuf;
                                for (int y = 0; y < dy1; y++)
                                    memcpy(outbuf + y*CONV_WINO_SIZE, pbptr0 + y*W0,
                                           dx1*sizeof(pbptr0[0]));
                            }
                        }
#if CV_TRY_AVX2
                        if (conv->useAVX2)
                            opt_AVX::winofunc_AtXA_8x8_f32(out_wbuf + ((k - k0)*CONV_WINO_IBLOCK + (block_id - block_id0))*CONV_WINO_AREA, CONV_WINO_SIZE,
                                                                bpptr, outstep, outptr, outstep, biasv, minval, maxval, ifMinMaxAct);
                        else
#endif
#if CV_TRY_AVX
                        if (conv->useAVX)
                            opt_AVX::winofunc_AtXA_8x8_f32(out_wbuf + ((k - k0)*CONV_WINO_IBLOCK + (block_id - block_id0))*CONV_WINO_AREA, CONV_WINO_SIZE,
                                                                bpptr, outstep, outptr, outstep, biasv, minval, maxval, ifMinMaxAct);
                        else
#endif
#if CV_NEON && CV_NEON_AARCH64
                        if (conv->useNEON)
                            // NEON optimization is only for ARMv8 device, and for ARMv7 device, we use the Universal intrinsics.
                            opt_NEON::winofunc_AtXA_8x8_f32(out_wbuf + ((k - k0)*CONV_WINO_IBLOCK + (block_id - block_id0))*CONV_WINO_AREA, CONV_WINO_SIZE,
                                                                bpptr, outstep, outptr, outstep, biasv, minval, maxval, ifMinMaxAct);
                        else
#endif
                        winofunc_AtXA_8x8_f32(out_wbuf + ((k - k0)*CONV_WINO_IBLOCK + (block_id - block_id0))*CONV_WINO_AREA, CONV_WINO_SIZE,
                                                  bpptr, outstep, outptr, outstep, biasv, minval, maxval, ifMinMaxAct);
                        if (partial)
                        {
                            if (activ)
                                activ->forwardSlice(outptr, outptr, CONV_WINO_SIZE*CONV_WINO_STEP, 0, g*Kg + k, g*Kg + k + 1);
                            for (int y = 0; y < dy1; y++)
                                memcpy(outptr0 + y*W0, outptr + y*CONV_WINO_SIZE,dx1*sizeof(outptr0[0]));
                        }
                    }
                }
            }
        }
    }});
    return 1;
}

/****************************************************************************************\
                                    SIMD for winograd function
\****************************************************************************************/

#if CV_SIMD128

void winofunc_accum_f32(const float* inwptr, const float* wptr, float* outbuf, int Cg, int iblock,
                            const int winoIblock, const int winoKblock, const int winoAtomF32, const int winoNatomF32)
{
#if 1
    CV_Assert(winoIblock == 3 && winoKblock == 4 && winoAtomF32 == 4);
    for (int atom_id = 0; atom_id < winoNatomF32; atom_id++,
            outbuf += winoAtomF32)
    {
        v_float32x4 s00 = v_setzero_f32(), s01 = s00, s02 = s00;
        v_float32x4 s10 = v_setzero_f32(), s11 = s00, s12 = s00;
        v_float32x4 s20 = v_setzero_f32(), s21 = s00, s22 = s00;
        v_float32x4 s30 = v_setzero_f32(), s31 = s00, s32 = s00;

        for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                     wptr += winoKblock*winoAtomF32)
        {
            v_float32x4 x0, x1, x2;
            x0 = v_load(inwptr);
            x1 = v_load(inwptr + 4);
            x2 = v_load(inwptr + 8);

            v_float32x4 w0 = v_load(wptr);
            s00 = v_fma(w0, x0, s00);
            s01 = v_fma(w0, x1, s01);
            s02 = v_fma(w0, x2, s02);

            w0 = v_load(wptr + 4);
            s10 = v_fma(w0, x0, s10);
            s11 = v_fma(w0, x1, s11);
            s12 = v_fma(w0, x2, s12);

            w0 = v_load(wptr + 8);
            s20 = v_fma(w0, x0, s20);
            s21 = v_fma(w0, x1, s21);
            s22 = v_fma(w0, x2, s22);

            w0 = v_load(wptr + 12);
            s30 = v_fma(w0, x0, s30);
            s31 = v_fma(w0, x1, s31);
            s32 = v_fma(w0, x2, s32);
        }

        v_store(outbuf, s00);
        v_store(outbuf + 1*64, s01);
        v_store(outbuf + 2*64, s02);
        v_store(outbuf + 3*64, s10);
        v_store(outbuf + 4*64, s11);
        v_store(outbuf + 5*64, s12);
        v_store(outbuf + 6*64, s20);
        v_store(outbuf + 7*64, s21);
        v_store(outbuf + 8*64, s22);
        v_store(outbuf + 9*64, s30);
        v_store(outbuf + 10*64, s31);
        v_store(outbuf + 11*64, s32);
    }
#else
    // Naive C++ code, the code should never be run here.
    for (int atom_id = 0; atom_id < winoNatomF32;
                atom_id++, outbuf += winoAtomF32)
    {
        float sumbuf[winoIblock*winoKblock*winoAtomF32];
        memset(sumbuf, 0, sizeof(sumbuf));
        for (int c = 0; c < Cg; c++, inwptr += winoIblock*winoAtomF32,
                                     wptr += winoKblock*winoAtomF32)
        {
            for (int i = 0; i < winoKblock; i++)
            {
                for (int j = 0; j < winoIblock; j++)
                {
                    int i_ = i*winoAtomF32;
                    int j_ = j*winoAtomF32;
                    int ij_ = i_*winoIblock + j_;
                    float s0 = inwptr[j_ + 0]*wptr[i_ + 0];
                    float s1 = inwptr[j_ + 1]*wptr[i_ + 1];
                    float s2 = inwptr[j_ + 2]*wptr[i_ + 2];
                    float s3 = inwptr[j_ + 3]*wptr[i_ + 3];
                    sumbuf[ij_ + 0] += s0;
                    sumbuf[ij_ + 1] += s1;
                    sumbuf[ij_ + 2] += s2;
                    sumbuf[ij_ + 3] += s3;
                }
            }
        }
        for (int ij = 0; ij < winoKblock*winoIblock; ij++)
        {
            int ij_ = ij*winoAtomF32;
            int ij_out = ij*CONV_WINO_AREA;
            outbuf[ij_out + 0] = sumbuf[ij_ + 0];
            outbuf[ij_out + 1] = sumbuf[ij_ + 1];
            outbuf[ij_out + 2] = sumbuf[ij_ + 2];
            outbuf[ij_out + 3] = sumbuf[ij_ + 3];
        }
    }
#endif
}

/*Input transform*/
void winofunc_BtXB_8x8_f32(const float* inptr, int inpstep,
                          float* outptr, int Cg, const int winoIblock, const int winoAtomF32)
{
    CV_Assert(winoIblock == 3 && winoAtomF32 == 4);
    v_float32x4 x00 = v_load(inptr), x01 = v_load(inptr + 4);
    v_float32x4 x10 = v_load(inptr + inpstep), x11 = v_load(inptr + inpstep + 4);
    v_float32x4 x20 = v_load(inptr + inpstep*2), x21 = v_load(inptr + inpstep*2 + 4);
    v_float32x4 x30 = v_load(inptr + inpstep*3), x31 = v_load(inptr + inpstep*3 + 4);
    v_float32x4 x40 = v_load(inptr + inpstep*4), x41 = v_load(inptr + inpstep*4 + 4);
    v_float32x4 x50 = v_load(inptr + inpstep*5), x51 = v_load(inptr + inpstep*5 + 4);
    v_float32x4 x60 = v_load(inptr + inpstep*6), x61 = v_load(inptr + inpstep*6 + 4);
    v_float32x4 x70 = v_load(inptr + inpstep*7), x71 = v_load(inptr + inpstep*7 + 4);

    v_float32x4 z00, z01, z10, z11, z20, z21, z30, z31, z40, z41, z50, z51, z60, z61, z70, z71;

    {
        /* Y[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*X */
        /* Y[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*X */
        v_float32x4 q5_25 = v_setall_f32(5.25f), t00, t01, t10, t11;
        t00 = x40 - x20;
        t01 = x41 - x21;
        t10 = x30 - x50;
        t11 = x31 - x51;
        v_float32x4 y00 = v_fma(t00, q5_25, x00 - x60);
        v_float32x4 y01 = v_fma(t01, q5_25, x01 - x61);
        v_float32x4 y70 = v_fma(t10, q5_25, x70 - x10);
        v_float32x4 y71 = v_fma(t11, q5_25, x71 - x11);

        /* Y[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*X */
        /* Y[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*X */
        v_float32x4 qm4_25 = v_setall_f32(-4.25f);
        t00 = v_fma(x30, qm4_25, x10 + x50);
        t01 = v_fma(x31, qm4_25, x11 + x51);
        t10 = v_fma(x40, qm4_25, x20 + x60);
        t11 = v_fma(x41, qm4_25, x21 + x61);

        v_float32x4 y10 = t00 + t10, y11 = t01 + t11;
        v_float32x4 y20 = t10 - t00, y21 = t11 - t01;

        /* Y[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*X */
        /* Y[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*X */
        v_float32x4 q0_5 = v_setall_f32(0.5f), q0_25 = v_setall_f32(0.25f);
        v_float32x4 qm2_5 = v_setall_f32(-2.5f), qm1_25 = v_setall_f32(-1.25f);
        t00 = v_fma(x10, q0_5, x50 + x50);
        t01 = v_fma(x11, q0_5, x51 + x51);
        t10 = v_fma(x20, q0_25, x60);
        t11 = v_fma(x21, q0_25, x61);
        t00 = v_fma(x30, qm2_5, t00);
        t01 = v_fma(x31, qm2_5, t01);
        t10 = v_fma(x40, qm1_25, t10);
        t11 = v_fma(x41, qm1_25, t11);

        v_float32x4 y30 = t00 + t10, y31 = t01 + t11;
        v_float32x4 y40 = t10 - t00, y41 = t11 - t01;

        /* Y[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*X */
        /* Y[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*X */
        v_float32x4 q4 = v_setall_f32(4.f), qm5 = v_setall_f32(-5.f);
        t00 = v_fma(x50, q0_5, x10 + x10);
        t01 = v_fma(x51, q0_5, x11 + x11);
        t10 = v_fma(x20, q4   , x60);
        t11 = v_fma(x21, q4   , x61);
        t00 = v_fma(x30, qm2_5, t00);
        t01 = v_fma(x31, qm2_5, t01);
        t10 = v_fma(x40, qm5  , t10);
        t11 = v_fma(x41, qm5  , t11);

        v_float32x4 y50 = t00 + t10, y51 = t01 + t11;
        v_float32x4 y60 = t10 - t00, y61 = t11 - t01;

        /* transpose 8x8 matrix with v_transpose4x4 */

        v_float32x4 y000, y100, y200, y300, y010, y110, y210, y310, y400, y500, y600, y700, y410, y510, y610, y710;
        v_transpose4x4(y00, y10, y20, y30, y000, y100, y200, y300);
        v_transpose4x4(y01, y11, y21, y31, y010, y110, y210, y310);
        v_transpose4x4(y40, y50, y60, y70, y400, y500, y600, y700);
        v_transpose4x4(y41, y51, y61, y71, y410, y510, y610, y710);

        /* Z[0] = [1.f, 0.f, -5.25f, 0.f, 5.25f, 0.f, -1.f, 0.f]*Y */
        /* Z[7] = [0.f, -1.f, 0.f, 5.25f, 0.f, -5.25f, 0.f, 1.f]*Y */
        t00 = y010 - y200;
        t01 = y410 - y600;
        t10 = y300 - y110;
        t11 = y700 - y510;
        z00 = v_fma(t00, q5_25, y000 - y210);
        z01 = v_fma(t01, q5_25, y400 - y610);
        z70 = v_fma(t10, q5_25, y310 - y100);
        z71 = v_fma(t11, q5_25, y710 - y500);

        /* Z[1] = [0.f, 1.f, 1.f, -4.25f, -4.25f, 1.f, 1.f, 0.f]*Y */
        /* Z[2] = [0.f, -1.f, 1.f, 4.25f, -4.25f, -1.f, 1.f, 0.f]*Y */
        t00 = v_fma(y300, qm4_25, y100 + y110);
        t01 = v_fma(y700, qm4_25, y500 + y510);
        t10 = v_fma(y010, qm4_25, y200 + y210);
        t11 = v_fma(y410, qm4_25, y600 + y610);

        z10 = t00 + t10; z11 = t01 + t11;
        z20 = t10 - t00; z21 = t11 - t01;

        /* Z[3] = [0.f, 0.5f, 0.25f, -2.5f, -1.25f, 2.f, 1.f, 0.f]*Y */
        /* Z[4] = [0.f, -0.5f, 0.25f, 2.5f, -1.25f, -2.f, 1.f, 0.f]*Y */
        t00 = v_fma(y100, q0_5, y110 + y110);
        t01 = v_fma(y500, q0_5, y510 + y510);
        t10 = v_fma(y200, q0_25, y210);
        t11 = v_fma(y600, q0_25, y610);
        t00 = v_fma(y300, qm2_5, t00);
        t01 = v_fma(y700, qm2_5, t01);
        t10 = v_fma(y010, qm1_25, t10);
        t11 = v_fma(y410, qm1_25, t11);

        z30 = t00 + t10; z31 = t01 + t11;
        z40 = t10 - t00; z41 = t11 - t01;

        /* Z[5] = [0.f, 2.f, 4.f, -2.5f, -5.f, 0.5f, 1.f, 0.f]*Y */
        /* Z[6] = [0.f, -2.f, 4.f, 2.5f, -5.f, -0.5f, 1.f, 0.f]*Y */
        t00 = v_fma(y110, q0_5, y100 + y100);
        t01 = v_fma(y510, q0_5, y500 + y500);
        t10 = v_fma(y200, q4, y210);
        t11 = v_fma(y600, q4, y610);
        t00 = v_fma(y300, qm2_5, t00);
        t01 = v_fma(y700, qm2_5, t01);
        t10 = v_fma(y010, qm5, t10);
        t11 = v_fma(y410, qm5, t11);

        z50 = t00 + t10; z51 = t01 + t11;
        z60 = t10 - t00; z61 = t11 - t01;
    }

    const int outstep = winoIblock*winoAtomF32*Cg;

    v_store(outptr, z00);
    v_store(outptr + outstep, z01);
    v_store(outptr + outstep*2, z10);
    v_store(outptr + outstep*3, z11);
    v_store(outptr + outstep*4, z20);
    v_store(outptr + outstep*5, z21);
    v_store(outptr + outstep*6, z30);
    v_store(outptr + outstep*7, z31);
    v_store(outptr + outstep*8, z40);
    v_store(outptr + outstep*9, z41);
    v_store(outptr + outstep*10, z50);
    v_store(outptr + outstep*11, z51);
    v_store(outptr + outstep*12, z60);
    v_store(outptr + outstep*13, z61);
    v_store(outptr + outstep*14, z70);
    v_store(outptr + outstep*15, z71);
}

/*Output transform*/
/*  Inverse Winograd 8x8 transform:
    out = (A'*inp*A)', where
    inp is input 8x8 FP32 matrix,
    A' is
    [1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
     0.f, 1.f, -1.f, 2.f, -2.f, 0.5f, -0.5f, 0.f,
     0.f, 1.f, 1.f, 4.f, 4.f, 0.25f, 0.25f, 0.f,
     0.f, 1.f, -1.f, 8.f, -8.f, 0.125f, -0.125f, 0.f,
     0.f, 1.f, 1.f, 16.f, 16.f, 1.f/16, 1.f/16, 0.f,
     0.f, 1.f, -1.f, 32.f, -32.f, 1.f/32, -1.f/32, 1.f]

    inp is pre-loaded into xij registers,
    out will be stored in zij, where (0<=i<=7 for x, 0<=i<=5 for z), 0<=j<=1.

    After the inverse transform is done, we add bias,
    optionally add results from the earlier tensors (by-pass),
    optionally apply activation function and then
    store the final results.

    That is, after both forward and then inverse transformation,
    we get non-transposed result.
    Of course, for the correct work of Winograd-based convolution,
    the Winograd-transformed weights should also be transposed.
    init_conv() (see OpConv.fx) takes care of that.
*/
void winofunc_AtXA_8x8_f32(const float* inptr, int inpstep,
                          float* bpptr, int bpstep, float* outptr, int outstep,
                          float bias, float minval, float maxval, bool ifMinMaxAct)
{
    v_float32x4 x00 = v_load(inptr), x01 = v_load(inptr + 4);
    v_float32x4 x10 = v_load(inptr + inpstep), x11 = v_load(inptr + inpstep + 4);
    v_float32x4 x20 = v_load(inptr + inpstep*2), x21 = v_load(inptr + inpstep*2 + 4);
    v_float32x4 x30 = v_load(inptr + inpstep*3), x31 = v_load(inptr + inpstep*3 + 4);
    v_float32x4 x40 = v_load(inptr + inpstep*4), x41 = v_load(inptr + inpstep*4 + 4);
    v_float32x4 x50 = v_load(inptr + inpstep*5), x51 = v_load(inptr + inpstep*5 + 4);
    v_float32x4 x60 = v_load(inptr + inpstep*6), x61 = v_load(inptr + inpstep*6 + 4);
    v_float32x4 x70 = v_load(inptr + inpstep*7), x71 = v_load(inptr + inpstep*7 + 4);
    v_float32x4 z00, z01, z10, z11, z20, z21, z30, z31, z40, z41, z50, z51;

    {
        v_float32x4 s12_0, s12_1, s34_0, s34_1, s56_0, s56_1;
        s12_0 = x10 + x20; s12_1 = x11 + x21;
        s34_0 = x30 + x40; s34_1 = x31 + x41;
        s56_0 = x50 + x60; s56_1 = x51 + x61;

        v_float32x4 y00 = x00 + s12_0 + s34_0 + s56_0;
        v_float32x4 y01 = x01 + s12_1 + s34_1 + s56_1;

        v_float32x4 a0 = v_setall_f32(0.25f), a1 = v_setall_f32(4.0f);
        v_float32x4 y20 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y21 = v_fma(s56_1, a0 ,v_fma(s34_1, a1, s12_1) );

        a0 = v_setall_f32(1.f/16), a1 = v_setall_f32(16.0f);
        v_float32x4 y40 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y41 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        s12_0 = x10 - x20; s12_1 = x11 - x21;
        s34_0 = x30 - x40; s34_1 = x31 - x41;
        s56_0 = x50 - x60; s56_1 = x51 - x61;

        a0 = v_setall_f32(1.f/32), a1 = v_setall_f32(32.f);
        v_float32x4 y50 = v_fma(s56_0, a0, v_fma(s34_0, a1, x70 + s12_0));
        v_float32x4 y51 = v_fma(s56_1, a0, v_fma(s34_1, a1, x71 + s12_1));

        a0 = v_setall_f32(0.5f), a1 = v_setall_f32(2.f);
        v_float32x4 y10 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y11 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(0.125f), a1 = v_setall_f32(8.f);
        v_float32x4 y30 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        v_float32x4 y31 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        v_float32x4 y60 = v_setall_f32(0.f), y61 = y60, y70 = y60, y71 = y60;

        /* transpose 8x8 matrix with v_transpose4x4 */

        v_float32x4 y000, y100, y200, y300, y010, y110, y210, y310, y400, y500, y600, y700, y410, y510, y610, y710;
        v_transpose4x4(y00, y10, y20, y30, y000, y100, y200, y300);
        v_transpose4x4(y01, y11, y21, y31, y010, y110, y210, y310);
        v_transpose4x4(y40, y50, y60, y70, y400, y500, y600, y700);
        v_transpose4x4(y41, y51, y61, y71, y410, y510, y610, y710);

        s12_0 = y100 + y200; s12_1 = y500 + y600;
        s34_0 = y300 + y010; s34_1 = y700 + y410;
        s56_0 = y110 + y210; s56_1 = y510 + y610;

        z00 = y000 + s12_0 + s34_0 + s56_0;
        z01 = y400 + s12_1 + s34_1 + s56_1;

        a0 = v_setall_f32(0.25f), a1 = v_setall_f32(4.0f);
        z20 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z21 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(1.f/16), a1 = v_setall_f32(16.0f);
        z40 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z41 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        s12_0 = y100 - y200; s12_1 = y500 - y600;
        s34_0 = y300 - y010; s34_1 = y700 - y410;
        s56_0 = y110 - y210; s56_1 = y510 - y610;

        a0 = v_setall_f32(1.f/32), a1 = v_setall_f32(32.0f);
        z50 = v_fma(s56_0, a0, v_fma(s34_0, a1, y310 + s12_0));
        z51 = v_fma(s56_1, a0, v_fma(s34_1, a1, y710 + s12_1));
        a0 = v_setall_f32(0.5f), a1 = v_setall_f32(2.0f);
        z10 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z11 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        a0 = v_setall_f32(0.125f), a1 = v_setall_f32(8.0f);
        z30 = v_fma(s56_0, a0, v_fma(s34_0, a1, s12_0));
        z31 = v_fma(s56_1, a0, v_fma(s34_1, a1, s12_1));

        v_float32x4 vbias = v_setall_f32(bias);
        z00 += vbias;
        z01 += vbias;
        z10 += vbias;
        z11 += vbias;
        z20 += vbias;
        z21 += vbias;
        z30 += vbias;
        z31 += vbias;
        z40 += vbias;
        z41 += vbias;
        z50 += vbias;
        z51 += vbias;
    }

    if (bpptr)
    {
        z00 += v_load(bpptr);
        z01 += v_load_low(bpptr + 4);
        z10 += v_load(bpptr + bpstep);
        z11 += v_load_low(bpptr + bpstep + 4);
        z20 += v_load(bpptr + bpstep*2);
        z21 += v_load_low(bpptr + bpstep*2 + 4);
        z30 += v_load(bpptr + bpstep*3);
        z31 += v_load_low(bpptr + bpstep*3 + 4);
        z40 += v_load(bpptr + bpstep*4);
        z41 += v_load_low(bpptr + bpstep*4 + 4);
        z50 += v_load(bpptr + bpstep*5);
        z51 += v_load_low(bpptr + bpstep*5 + 4);
    }

    if (ifMinMaxAct)
    {
        v_float32x4 vmax = v_setall_f32(maxval);
        v_float32x4 vmin = v_setall_f32(minval);

        z00 = v_min(v_max(z00, vmin), vmax);
        z01 = v_min(v_max(z01, vmin), vmax);
        z10 = v_min(v_max(z10, vmin), vmax);
        z11 = v_min(v_max(z11, vmin), vmax);
        z20 = v_min(v_max(z20, vmin), vmax);
        z21 = v_min(v_max(z21, vmin), vmax);
        z30 = v_min(v_max(z30, vmin), vmax);
        z31 = v_min(v_max(z31, vmin), vmax);
        z40 = v_min(v_max(z40, vmin), vmax);
        z41 = v_min(v_max(z41, vmin), vmax);
        z50 = v_min(v_max(z50, vmin), vmax);
        z51 = v_min(v_max(z51, vmin), vmax);
    }

    v_store(outptr, z00);
    v_store_low(outptr + 4, z01);
    v_store(outptr + outstep, z10);
    v_store_low(outptr + outstep + 4, z11);
    v_store(outptr + outstep*2, z20);
    v_store_low(outptr + outstep*2 + 4, z21);
    v_store(outptr + outstep*3, z30);
    v_store_low(outptr + outstep*3 + 4, z31);
    v_store(outptr + outstep*4, z40);
    v_store_low(outptr + outstep*4 + 4, z41);
    v_store(outptr + outstep*5, z50);
    v_store_low(outptr + outstep*5 + 4, z51);
}
#endif

#else
int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv>& conv,
                  int ntasks, float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct)
{
    return 0;
}
#endif

}} // namespace cv::dnn
