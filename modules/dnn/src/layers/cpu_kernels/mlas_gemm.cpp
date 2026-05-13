// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "mlas_gemm.hpp"

#ifdef HAVE_MLAS

#include "mlas.h"
#include <vector>

namespace cv { namespace dnn {

bool mlasAvailable() {
    static const bool ok = []() {
        const size_t a = MlasGetPreferredBufferAlignment();
        return a > 0 && a <= 256;
    }();
    return ok;
}

bool mlasSgemm(bool trans_a, bool trans_b,
               int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc)
{
    if (!mlasAvailable()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;

    MLAS_SGEMM_DATA_PARAMS data;
    data.A = A;
    data.lda = static_cast<size_t>(lda);
    data.B = B;
    data.ldb = static_cast<size_t>(ldb);
    data.C = C;
    data.ldc = static_cast<size_t>(ldc);
    data.alpha = alpha;
    data.beta = beta;
    data.BIsPacked = false;

    MlasGemm(trans_a ? CblasTrans : CblasNoTrans,
             trans_b ? CblasTrans : CblasNoTrans,
             static_cast<size_t>(M),
             static_cast<size_t>(N),
             static_cast<size_t>(K),
             data,
             /*ThreadPool=*/nullptr,
             /*BackendKernelSelectorConfig=*/nullptr);
    return true;
}

bool mlasSgemmBatch(size_t batch,
                    const size_t* A_offsets,
                    const size_t* B_offsets,
                    const size_t* C_offsets,
                    bool trans_a, bool trans_b,
                    int M, int N, int K,
                    float alpha,
                    const float* A_base, int lda,
                    const float* B_base, int ldb,
                    float beta,
                    float* C_base, int ldc)
{
    if (!mlasAvailable()) return false;
    if (batch == 0 || M <= 0 || N <= 0 || K <= 0) return false;

    std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch);
    for (size_t i = 0; i < batch; i++) {
        data[i].A = A_base + A_offsets[i];
        data[i].lda = static_cast<size_t>(lda);
        data[i].B = B_base + B_offsets[i];
        data[i].ldb = static_cast<size_t>(ldb);
        data[i].C = C_base + C_offsets[i];
        data[i].ldc = static_cast<size_t>(ldc);
        data[i].alpha = alpha;
        data[i].beta = beta;
        data[i].BIsPacked = false;
    }

    MlasGemmBatch(trans_a ? CblasTrans : CblasNoTrans,
                  trans_b ? CblasTrans : CblasNoTrans,
                  static_cast<size_t>(M),
                  static_cast<size_t>(N),
                  static_cast<size_t>(K),
                  data.data(),
                  batch,
                  /*ThreadPool=*/nullptr,
                  /*BackendKernelSelectorConfig=*/nullptr);
    return true;
}

size_t mlasSgemmPackBSize(bool trans_a, bool trans_b, int N, int K)
{
    if (!mlasAvailable()) return 0;
    if (N <= 0 || K <= 0) return 0;
    return MlasGemmPackBSize(trans_a ? CblasTrans : CblasNoTrans,
                             trans_b ? CblasTrans : CblasNoTrans,
                             static_cast<size_t>(N),
                             static_cast<size_t>(K),
                             /*BackendKernelSelectorConfig=*/nullptr);
}

bool mlasSgemmPackB(bool trans_a, bool trans_b, int N, int K,
                    const float* B, int ldb, void* packed_B)
{
    if (!mlasAvailable()) return false;
    if (N <= 0 || K <= 0 || B == nullptr || packed_B == nullptr) return false;
    MlasGemmPackB(trans_a ? CblasTrans : CblasNoTrans,
                  trans_b ? CblasTrans : CblasNoTrans,
                  static_cast<size_t>(N),
                  static_cast<size_t>(K),
                  B, static_cast<size_t>(ldb),
                  packed_B,
                  /*BackendKernelSelectorConfig=*/nullptr);
    return true;
}

bool mlasSgemmPacked(bool trans_a, bool trans_b,
                     int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const void* packed_B,
                     float beta,
                     float* C, int ldc)
{
    if (!mlasAvailable()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;

    MLAS_SGEMM_DATA_PARAMS data;
    data.A = A;
    data.lda = static_cast<size_t>(lda);
    data.B = static_cast<const float*>(packed_B);
    data.ldb = 0;  // ignored when BIsPacked
    data.C = C;
    data.ldc = static_cast<size_t>(ldc);
    data.alpha = alpha;
    data.beta = beta;
    data.BIsPacked = true;

    MlasGemm(trans_a ? CblasTrans : CblasNoTrans,
             trans_b ? CblasTrans : CblasNoTrans,
             static_cast<size_t>(M),
             static_cast<size_t>(N),
             static_cast<size_t>(K),
             data,
             /*ThreadPool=*/nullptr,
             /*BackendKernelSelectorConfig=*/nullptr);
    return true;
}

size_t mlasFlashAttentionBufferBytesPerThread(int q_block_size,
                                              int kv_block_size,
                                              int v_head_size)
{
    if (q_block_size <= 0 || kv_block_size <= 0 || v_head_size <= 0) return 0;
    // flashattn.cpp lays out the per-thread scratch as:
    //   l[q_block_size] + m[q_block_size]
    //   + intermediate[q_block_size * kv_block_size]
    //   + temp_output[q_block_size * v_head_size]
    const size_t q  = static_cast<size_t>(q_block_size);
    const size_t kv = static_cast<size_t>(kv_block_size);
    const size_t vd = static_cast<size_t>(v_head_size);
    return (q * (2 + kv + vd)) * sizeof(float);
}

bool mlasFlashAttention(const float* query, const float* key, const float* value,
                        float* output,
                        int batch_size, int num_heads,
                        int q_seq_len, int kv_seq_len,
                        int qk_head_size, int v_head_size,
                        float scale,
                        int q_block_size, int kv_block_size,
                        void* scratch, int thread_count)
{
    if (!mlasAvailable()) return false;
    if (batch_size <= 0 || num_heads <= 0) return false;
    if (q_seq_len <= 0 || kv_seq_len <= 0) return false;
    if (qk_head_size <= 0 || v_head_size <= 0) return false;
    if (q_block_size <= 0 || kv_block_size <= 0) return false;
    if (thread_count <= 0 || scratch == nullptr) return false;
    if (query == nullptr || key == nullptr || value == nullptr || output == nullptr)
        return false;

    MlasFlashAttentionThreadedArgs args;
    args.batch_size            = batch_size;
    args.num_heads             = num_heads;
    args.q_sequence_length     = q_seq_len;
    args.kv_sequence_length    = kv_seq_len;
    args.qk_head_size          = qk_head_size;
    args.v_head_size           = v_head_size;
    args.q_block_size          = q_block_size;
    args.kv_block_size         = kv_block_size;
    args.scale                 = scale;
    args.thread_count          = thread_count;
    args.buffer                = static_cast<float*>(scratch);
    args.buffer_size_per_thread = mlasFlashAttentionBufferBytesPerThread(
                                      q_block_size, kv_block_size, v_head_size);
    args.query                 = query;
    args.key                   = key;
    args.value                 = value;
    args.output                = output;

    MlasFlashAttention(&args, /*ThreadPool=*/nullptr);
    return true;
}

}}  // cv::dnn

#endif  // HAVE_MLAS
