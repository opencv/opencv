// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_MLAS_GEMM_HPP
#define OPENCV_DNN_MLAS_GEMM_HPP

#include <cstddef>

namespace cv { namespace dnn {

#ifdef HAVE_MLAS

// True if MLAS is usable on this host. False signals callers to fall back.
bool mlasAvailable();

// Row-major SGEMM: C := alpha * op(A) * op(B) + beta * C, op(X) = X or X^T.
// Returns false if MLAS is unavailable or M/N/K <= 0.
bool mlasSgemm(bool trans_a, bool trans_b,
               int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc);

// Batched SGEMM with per-batch element offsets into A_base/B_base/C_base.
// M/N/K and leading dims are shared across the batch.
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
                    float* C_base, int ldc);

// Pack B once, reuse across many mlasSgemmPacked() calls. Returns the
// required buffer size in bytes; caller allocates and passes to mlasSgemmPackB.
size_t mlasSgemmPackBSize(bool trans_a, bool trans_b, int N, int K);

bool mlasSgemmPackB(bool trans_a, bool trans_b, int N, int K,
                    const float* B, int ldb, void* packed_B);

// mlasSgemm with a pre-packed B from mlasSgemmPackB.
bool mlasSgemmPacked(bool trans_a, bool trans_b,
                     int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const void* packed_B,
                     float beta,
                     float* C, int ldc);

// Scratch-buffer size (in bytes) per worker thread for mlasFlashAttention.
// The caller must allocate `thread_count * this` bytes for the scratch
// pointer. Returns 0 if any argument is non-positive.
size_t mlasFlashAttentionBufferBytesPerThread(int q_block_size,
                                              int kv_block_size,
                                              int v_head_size);

// Multi-head attention via MLAS flash-attention. Computes
//   output[b, i, h, :] = softmax(scale * Q[b,h,i,:] @ K[b,h,:,:]^T) @ V[b,h,:,:]
// fused into one tiled kernel without materializing the q_seq x kv_seq
// attention matrix.
//
// Layouts (row-major contiguous, FP32):
//   query : [batch, num_heads, q_seq_len,  qk_head_size]
//   key   : [batch, num_heads, kv_seq_len, qk_head_size]
//   value : [batch, num_heads, kv_seq_len, v_head_size]
//   output: [batch, q_seq_len, num_heads,  v_head_size]   (heads *after* seq)
//
// scale         - usually 1 / sqrt(qk_head_size).
// q_block_size  - tile size along the q sequence (e.g. 256).
// kv_block_size - tile size along the kv sequence (e.g. 256).
// scratch       - caller-owned buffer of at least
//                 thread_count * mlasFlashAttentionBufferBytesPerThread(...).
// thread_count  - number of MLAS workers to fan out across (typically
//                 cv::getNumThreads()).
//
// Returns false if MLAS is unavailable or arguments are invalid.
bool mlasFlashAttention(const float* query, const float* key, const float* value,
                        float* output,
                        int batch_size, int num_heads,
                        int q_seq_len, int kv_seq_len,
                        int qk_head_size, int v_head_size,
                        float scale,
                        int q_block_size, int kv_block_size,
                        void* scratch, int thread_count);

#else  // HAVE_MLAS

inline bool mlasAvailable() { return false; }

inline bool mlasSgemm(bool, bool, int, int, int, float,
                      const float*, int, const float*, int,
                      float, float*, int) { return false; }

inline bool mlasSgemmBatch(size_t, const size_t*, const size_t*, const size_t*,
                           bool, bool, int, int, int, float,
                           const float*, int, const float*, int,
                           float, float*, int) { return false; }

inline size_t mlasSgemmPackBSize(bool, bool, int, int) { return 0; }
inline bool mlasSgemmPackB(bool, bool, int, int, const float*, int, void*) { return false; }
inline bool mlasSgemmPacked(bool, bool, int, int, int, float,
                            const float*, int, const void*, float, float*, int) { return false; }

inline size_t mlasFlashAttentionBufferBytesPerThread(int, int, int) { return 0; }
inline bool mlasFlashAttention(const float*, const float*, const float*, float*,
                               int, int, int, int, int, int, float,
                               int, int, void*, int) { return false; }

#endif  // HAVE_MLAS

}}  // cv::dnn

#endif  // OPENCV_DNN_MLAS_GEMM_HPP
