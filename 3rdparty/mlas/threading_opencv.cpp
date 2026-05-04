// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// OpenCV-backed implementation of MLAS's threading primitives.
//
// Replaces lib/threading.cpp from upstream MLAS (which, when built with
// BUILD_MLAS_NO_ONNXRUNTIME and a nullptr ThreadPool, runs everything in a
// serial for-loop). MLAS internally calls MlasGetMaximumThreadCount() to
// pick a partition count; for the standalone build that returns 1, so even
// with a parallel `MlasTrySimpleParallel` MLAS would still emit a single
// iteration. We patch both halves:
//
//   1. MlasGetMaximumThreadCount() in mlasi.h returns cv::getNumThreads()
//      when MLAS_OPENCV_THREADING is defined (see the small patch in
//      mlasi.h flagged with that define).
//   2. The three threaded entry points below dispatch to cv::parallel_for_.

#include "lib/mlasi.h"
#include "lib/qgemm.h"   // for MLAS_GEMM_QUANT_DISPATCH definition (stub below)

#include <opencv2/core/utility.hpp>

// MLAS_GEMM_ONLY stub: mlasi.h's MLAS_PLATFORM struct uses
//   GemmS8S8Dispatch{&MlasGemmQuantDispatchDefault}
// as in-class initializers. The real definition lives in qgemm_kernel_default.cpp
// which we don't compile in the SGEMM-only build. Provide a zero-initialized
// instance so mlasi.h links — it's never read because we never call MlasQgemm.
extern "C++" const MLAS_GEMM_QUANT_DISPATCH MlasGemmQuantDispatchDefault{};

extern "C" int opencv_dnn_mlas_max_threads()
{
    int n = cv::getNumThreads();
    return n > 0 ? n : 1;
}

void
MlasExecuteThreaded(
    MLAS_THREADED_ROUTINE* ThreadedRoutine,
    void* Context,
    ptrdiff_t Iterations,
    MLAS_THREADPOOL* /*ThreadPool*/)
{
    if (Iterations <= 0) return;
    if (Iterations == 1) { ThreadedRoutine(Context, 0); return; }

    cv::parallel_for_(cv::Range(0, static_cast<int>(Iterations)),
                      [&](const cv::Range& r) {
        for (int tid = r.start; tid < r.end; tid++) {
            ThreadedRoutine(Context, static_cast<ptrdiff_t>(tid));
        }
    });
}

void
MlasTrySimpleParallel(
    MLAS_THREADPOOL* /*ThreadPool*/,
    const std::ptrdiff_t Iterations,
    const std::function<void(std::ptrdiff_t tid)>& Work)
{
    if (Iterations <= 0) return;
    if (Iterations == 1) { Work(0); return; }

    cv::parallel_for_(cv::Range(0, static_cast<int>(Iterations)),
                      [&](const cv::Range& r) {
        for (int tid = r.start; tid < r.end; tid++) {
            Work(static_cast<std::ptrdiff_t>(tid));
        }
    });
}

void
MlasTryBatchParallel(
    MLAS_THREADPOOL* /*ThreadPool*/,
    const std::ptrdiff_t Iterations,
    const std::function<void(std::ptrdiff_t tid)>& Work)
{
    // MLAS only calls this for "non-performance-critical" small batches, but
    // there is no reason to serialize it on a 24-core host either.
    if (Iterations <= 0) return;
    if (Iterations == 1) { Work(0); return; }

    cv::parallel_for_(cv::Range(0, static_cast<int>(Iterations)),
                      [&](const cv::Range& r) {
        for (int tid = r.start; tid < r.end; tid++) {
            Work(static_cast<std::ptrdiff_t>(tid));
        }
    });
}
