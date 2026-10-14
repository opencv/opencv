// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// Output columns [n_start, n_end) of Y = A @ dequant(B).T.
// wcol is caller-owned scratch of ceil(K/block_size)*block_size floats.
void matMulNBitsStripe_(const float* a, const uchar* bq, const float* sc, const uchar* zp,
                        float* y, int M, int N, int K, int block_size, int bits,
                        int n_start, int n_end, float* wcol);

// Below this per-block byte count the staged kernel is faster than the fused one.
int matMulNBitsGemvMinBlobBytes_();

// Deinterleave A into the even/odd halves the fused path indexes by byte.
void matMulNBitsSplitA_(const float* a, float* a_even, float* a_odd, int pairs);

// Fused 4-bit M=1 GEMV: dequant stays in registers, no column is materialized.
void matMulNBitsGemv_(const float* a_even, const float* a_odd, const uchar* bq,
                      const float* sc, const uchar* zp, float* y, int K, int block_size,
                      int n_start, int n_end);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// Dequantize one block into w[0..block_size-1]. BITS is compile-time so the per-byte
// unpack unrolls and the shift folds to a constant (the hot half of the kernel).
template<int BITS>
static inline void dequantBlock(const uchar* packed, int blob_bytes, float scale, int zp,
                                float* w) {
    const int EPB = 8 / BITS;            // elements per byte: 4-bit:2  8-bit:1
    const int QMASK = (1 << BITS) - 1;
    for (int j = 0; j < blob_bytes; j++) {
        const uchar byte = packed[j];
        for (int e = 0; e < EPB; e++)     // LSB-first within the byte
            w[j * EPB + e] = (((byte >> (e * BITS)) & QMASK) - zp) * scale;
    }
}

// Zero points are packed per column exactly like the weights, 8/BITS to a byte.
template<int BITS>
static inline int zeroPoint(const uchar* zp_n, int blk) {
    const int EPB = 8 / BITS;
    if (!zp_n)
        return 1 << (BITS - 1);
    return (zp_n[blk / EPB] >> ((blk % EPB) * BITS)) & ((1 << BITS) - 1);
}

template<int BITS>
static void runStripe(const float* a, const uchar* bq, const float* sc, const uchar* zp,
                      float* y, int M, int N, int K, int block_size,
                      int n_start, int n_end, float* wcol) {
    const int n_blk = (K + block_size - 1) / block_size;
    const int blob_bytes = block_size * BITS / 8;   // packed weights per block
    const int zp_stride = (n_blk + 8 / BITS - 1) / (8 / BITS);

    for (int n = n_start; n < n_end; n++) {
        const uchar* bq_n = bq + static_cast<size_t>(n) * n_blk * blob_bytes;
        const float* sc_n = sc + static_cast<size_t>(n) * n_blk;
        const uchar* zp_n = zp ? zp + static_cast<size_t>(n) * zp_stride : nullptr;
        // Dequant the whole column once so it amortizes across the M rows.
        for (int blk = 0; blk < n_blk; blk++)
            dequantBlock<BITS>(bq_n + static_cast<size_t>(blk) * blob_bytes, blob_bytes,
                               sc_n[blk], zeroPoint<BITS>(zp_n, blk), wcol + blk * block_size);

        const float* w = wcol;
        for (int m = 0; m < M; m++) {
            const float* arow = a + static_cast<size_t>(m) * K;
            float acc = 0.f;
            int t = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vlanes = VTraits<v_float32>::vlanes();
            v_float32 vacc0 = vx_setzero_f32(), vacc1 = vx_setzero_f32();
            v_float32 vacc2 = vx_setzero_f32(), vacc3 = vx_setzero_f32();
            for (; t <= K - 4 * vlanes; t += 4 * vlanes) {
                vacc0 = v_fma(vx_load(arow + t),              vx_load(w + t),              vacc0);
                vacc1 = v_fma(vx_load(arow + t + vlanes),     vx_load(w + t + vlanes),     vacc1);
                vacc2 = v_fma(vx_load(arow + t + 2 * vlanes), vx_load(w + t + 2 * vlanes), vacc2);
                vacc3 = v_fma(vx_load(arow + t + 3 * vlanes), vx_load(w + t + 3 * vlanes), vacc3);
            }
            v_float32 vacc = v_add(v_add(vacc0, vacc1), v_add(vacc2, vacc3));
            for (; t <= K - vlanes; t += vlanes)
                vacc = v_fma(vx_load(arow + t), vx_load(w + t), vacc);
            acc = v_reduce_sum(vacc);
#endif
            for (; t < K; t++)
                acc += arow[t] * w[t];
            y[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

void matMulNBitsStripe_(const float* a, const uchar* bq, const float* sc, const uchar* zp,
                        float* y, int M, int N, int K, int block_size, int bits,
                        int n_start, int n_end, float* wcol) {
    switch (bits) {                 // bits in {4,8} enforced in the layer constructor
        case 4:  runStripe<4>(a, bq, sc, zp, y, M, N, K, block_size, n_start, n_end, wcol); break;
        default: runStripe<8>(a, bq, sc, zp, y, M, N, K, block_size, n_start, n_end, wcol); break;
    }
}

int matMulNBitsGemvMinBlobBytes_() {
#if (CV_SIMD || CV_SIMD_SCALABLE)
    return VTraits<v_uint32>::vlanes();
#else
    return INT_MAX;                  // no vector path: the caller keeps the staged kernel
#endif
}

void matMulNBitsSplitA_(const float* a, float* a_even, float* a_odd, int pairs) {
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    for (; i <= pairs - vlanes; i += vlanes) {
        v_float32 e, o;
        v_load_deinterleave(a + 2 * i, e, o);
        vx_store(a_even + i, e);
        vx_store(a_odd + i, o);
    }
#endif
    for (; i < pairs; i++) {
        a_even[i] = a[2 * i];
        a_odd[i] = a[2 * i + 1];
    }
}

void matMulNBitsGemv_(const float* a_even, const float* a_odd, const uchar* bq,
                      const float* sc, const uchar* zp, float* y, int K, int block_size,
                      int n_start, int n_end) {
    const int blob_bytes = block_size / 2;                  // 4-bit: two weights per byte
    const int n_blk = (K + block_size - 1) / block_size;
    const int pairs = K / 2;                                // bytes holding a live weight pair
    const int zp_stride = (n_blk + 1) / 2;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_uint32>::vlanes();
    const v_uint32 nibble = vx_setall_u32(0xF);
#endif

    for (int n = n_start; n < n_end; n++) {
        const uchar* bq_n = bq + static_cast<size_t>(n) * n_blk * blob_bytes;
        const float* sc_n = sc + static_cast<size_t>(n) * n_blk;
        const uchar* zp_n = zp ? zp + static_cast<size_t>(n) * zp_stride : nullptr;
        float acc = 0.f;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 vacc_e = vx_setzero_f32(), vacc_o = vx_setzero_f32();
#endif
        for (int blk = 0; blk < n_blk; blk++) {
            const int b0 = blk * blob_bytes;
            // Clamped, not blob_bytes: a partial last block owns fewer bytes than it spans.
            const int b1 = std::min(b0 + blob_bytes, pairs);
            if (b0 >= b1)
                break;
            const float scale = sc_n[blk];
            const int zpv = zeroPoint<4>(zp_n, blk);
            int j = b0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const v_float32 vs = vx_setall_f32(scale);
            const v_int32 zp_v = vx_setall_s32(zpv);
            for (; j <= b1 - vlanes; j += vlanes) {
                // One packed byte per 32-bit lane, so the nibble masks and the shift
                // stay at a width where v_shr exists and v_sub does not saturate.
                v_uint32 q = vx_load_expand_q(bq_n + j);
                v_int32 qe = v_reinterpret_as_s32(v_and(q, nibble));
                v_int32 qo = v_reinterpret_as_s32(v_and(v_shr<4>(q), nibble));
                v_float32 we = v_mul(v_cvt_f32(v_sub(qe, zp_v)), vs);
                v_float32 wo = v_mul(v_cvt_f32(v_sub(qo, zp_v)), vs);
                vacc_e = v_fma(we, vx_load(a_even + j), vacc_e);
                vacc_o = v_fma(wo, vx_load(a_odd + j), vacc_o);
            }
#endif
            for (; j < b1; j++) {
                const uchar byte = bq_n[j];
                acc += (static_cast<int>(byte & 0xF) - zpv) * scale * a_even[j];
                acc += (static_cast<int>(byte >> 4) - zpv) * scale * a_odd[j];
            }
        }
#if (CV_SIMD || CV_SIMD_SCALABLE)
        acc += v_reduce_sum(v_add(vacc_e, vacc_o));
#endif
        y[n] = acc;
    }
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::dnn::

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
