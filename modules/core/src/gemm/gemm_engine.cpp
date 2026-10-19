// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "gemm_engine.hpp"
#include "gemm_kernels.simd.hpp"
#include "gemm_sme.hpp"

namespace cv {
namespace gemm_engine {

// C = beta*C for an m x n block; beta = 0 clears it, so NaNs in C do not leak through.
static void scaleC(float* C, size_t ldc, int m, int n, float beta)
{
    if (beta == 1.f)
        return;
    for (int i = 0; i < m; i++)
    {
        float* c = C + i * ldc;
        if (beta == 0.f)
            memset(c, 0, n * sizeof(float));
        else
            for (int j = 0; j < n; j++)
                c[j] *= beta;
    }
}

bool smeEnabled()
{
#if defined(CV_GEMM_HAVE_SME)
    static const bool enabled = checkHardwareSupport(CPU_SME) &&
                                utils::getConfigurationParameterBool("OPENCV_CORE_GEMM_SME", true) &&
                                sme::vectorLength() >= 4;
    return enabled;
#else
    return false;
#endif
}

int smeVectorLength()
{
#if defined(CV_GEMM_HAVE_SME)
    static const int vl = smeEnabled() ? sme::vectorLength() : 0;
    return vl;
#else
    return 0;
#endif
}

#if CV_GEMM_ENGINE_NEON
using namespace CV_CPU_OPTIMIZATION_NAMESPACE;

// Path limits, measured on Apple M5 against MLAS.
enum
{
    SMALL_M_MAX = 32,        // rows up to this go to the direct-A kernels
    GEMM_ROWS_OUTER_BYTES = 1 << 16, // B blocks up to this size are swept with one A panel held in L1
    SME_MIN_M = 8,           // fewer rows waste most of an SME tile
    SME_MIN_MACS = 1 << 18,  // below this the streaming-mode switch and packing dominate
    SME_TASK_MACS = 1 << 21, // smallest SME task
    SME_KC = 512,            // k block when k and the tiles are large
    SME_BIG = 256
};

struct Problem
{
    int batch;
    const size_t *aOfs, *bOfs, *cOfs;
    int m, n, k;
    float alpha, beta;
    const float* A;
    size_t lda0, lda1;
    BMatrix B;
    float* C;
    size_t ldc;
    int nthreads;
};

template<typename F>
static void runTasks(int tasks, int nthreads, const F& body)
{
    if (nthreads > 1 && tasks > 1)
        parallel_for_(Range(0, tasks), body, tasks);
    else
        body(Range(0, tasks));
}

// Packs columns [j0, j0+nc) and rows [p0, p0+kc) of B into 12-column panels.
static void packB12(const Problem& P, int b, int j0, int nc, int p0, int kc, float* dst)
{
    const BMatrix& B = P.B;
    if (!B.panelWidth)
    {
        packBPanels(nc, kc, B.ptr + P.bOfs[b] + p0 * B.ld0 + j0 * B.ld1, B.ld0, B.ld1, GEMM_NR, dst);
        return;
    }
    // Panels of another width (packed for SME, read without it): gather element by element.
    const int W = B.panelWidth;
    for (int q = 0; q < nc; q += GEMM_NR, dst += (size_t)GEMM_NR * kc)
        for (int p = 0; p < kc; p++)
            for (int c = 0; c < GEMM_NR; c++)
            {
                const int j = j0 + q + c;
                dst[(size_t)p * GEMM_NR + c] = q + c < nc ? B.ptr[(size_t)(j / W) * W * P.k + (size_t)(p0 + p) * W + j % W] : 0.f;
            }
}

// Large M: packed 8x12 blocks. Each task owns a row block and a run of 12-column panels;
// A and B are packed once and shared when several tasks read them.
static void gemmBlocked(const Problem& P)
{
    const int m = P.m, n = P.n, k = P.k;
    const int target = P.nthreads > 1 ? 4 * P.nthreads : 1;
    const double macs = (double)m * n * k;
    const int nPanels = (n + GEMM_NR - 1) / GEMM_NR;
    int mBlocks = (m + GEMM_MC - 1) / GEMM_MC;
    int MC = roundUp((m + mBlocks - 1) / mBlocks, GEMM_MR);

    // About 4 tasks per thread, none below GEMM_TASK_MACS or narrower than 2 panels.
    auto colTasks = [&](int mb) {
        const int want = (target + P.batch * mb - 1) / (P.batch * mb);
        const int byWork = (int)std::max(1., macs / mb / GEMM_TASK_MACS);
        return std::max(1, std::min(std::min(want, byWork), (nPanels + 1) / 2));
    };
    int tN = colTasks(mBlocks);
    while (P.batch * mBlocks * tN < target && MC > 16 && macs / (2. * mBlocks * tN) >= GEMM_TASK_MACS)
    {
        MC = roundUp(MC / 2, GEMM_MR);
        mBlocks = (m + MC - 1) / MC;
        tN = colTasks(mBlocks);
    }
    const int panelsPerTask = (nPanels + tN - 1) / tN;
    tN = (nPanels + panelsPerTask - 1) / panelsPerTask;

    const int KC = std::min(k, (int)GEMM_KC);
    // Rows a packed A block takes: 8-row panels, the last one 4 rows when it fits.
    auto paddedRows = [](int rows) { const int r = rows % GEMM_MR; return rows - r + (r == 0 ? 0 : r <= 4 ? 4 : 8); };
    // An operand read by several tasks is packed once (whole k) and shared.
    const bool shareA = P.batch == 1 && tN > 1;
    const bool shareB = !P.B.panelWidth && P.batch == 1 && mBlocks > 1;
    const size_t aShared = shareA ? (size_t)paddedRows(MC) * mBlocks * k : 0;
    AutoBuffer<float> shared(aShared + (shareB ? (size_t)roundUp(n, GEMM_NR) * k : 0));
    float* sharedA = shared.data();
    float* sharedB = sharedA + aShared;
    BMatrix Bsh = P.B;
    if (shareA || shareB)
    {
        const int kBlocks = (k + KC - 1) / KC, aUnits = shareA ? mBlocks * kBlocks : 0;
        runTasks(aUnits + (shareB ? nPanels : 0), P.nthreads, [&](const Range& r) {
            for (int q = r.start; q < r.end; q++)
            {
                if (q < aUnits)
                {
                    const int mb = q / kBlocks, p0 = q % kBlocks * KC;
                    const int i0 = mb * MC, mc = std::min(MC, m - i0);
                    packA8(mc, std::min(KC, k - p0), P.A + P.aOfs[0] + i0 * P.lda0 + p0 * P.lda1, P.lda0, P.lda1,
                           sharedA + (size_t)mb * paddedRows(MC) * k + (size_t)p0 * paddedRows(mc));
                }
                else
                {
                    const int j0 = (q - aUnits) * GEMM_NR;
                    packPanel(std::min((int)GEMM_NR, n - j0), k, P.B.ptr + P.bOfs[0] + j0 * P.B.ld1, P.B.ld0, P.B.ld1,
                              GEMM_NR, sharedB + (size_t)j0 * k);
                }
            }
        });
        if (shareB)
            Bsh = BMatrix{ sharedB, 0, 0, GEMM_NR };
    }
    const bool direct = Bsh.panelWidth == GEMM_NR;
    const int perProblem = mBlocks * tN;
    const size_t aSize = shareA ? 0 : (size_t)MC * KC, bSize = direct ? 0 : (size_t)panelsPerTask * GEMM_NR * KC;

    runTasks(P.batch * perProblem, P.nthreads, [&](const Range& r) {
        AutoBuffer<float> buf(aSize + bSize);
        float* packedB = buf.data() + aSize;
        for (int t = r.start; t < r.end; t++)
        {
            const int b = t / perProblem, rem = t - b * perProblem;
            const int mb = rem / tN, i0 = mb * MC, j0 = rem % tN * panelsPerTask * GEMM_NR;
            const int mc = std::min(MC, m - i0), nc = std::min(panelsPerTask * GEMM_NR, n - j0);
            const float* a = P.A + P.aOfs[b] + i0 * P.lda0;
            float* c = P.C + P.cOfs[b] + i0 * P.ldc + j0;
            for (int p0 = 0; p0 < k; p0 += KC)
            {
                const int kc = std::min(KC, k - p0);
                const float* packedA = buf.data();
                if (shareA)
                    packedA = sharedA + (size_t)mb * paddedRows(MC) * k + (size_t)p0 * paddedRows(mc);
                else
                    packA8(mc, kc, a + p0 * P.lda1, P.lda0, P.lda1, buf.data());
                const float* bp = packedB;
                size_t bStride = (size_t)GEMM_NR * kc;
                if (direct)
                {
                    bp = Bsh.ptr + (size_t)j0 * k + (size_t)p0 * GEMM_NR;
                    bStride = (size_t)GEMM_NR * k;
                }
                else
                    packB12(P, b, j0, nc, p0, kc, packedB);
                // First k block writes C (zero mode), later blocks add to it.
                macroKernel(mc, nc, kc, packedA, bp, bStride, P.alpha, p0 == 0 ? P.beta : 1.f, c, P.ldc,
                            (size_t)nc * kc * sizeof(float) <= GEMM_ROWS_OUTER_BYTES);
            }
        }
    });
}

// Small M: A is read in place (no packing), B as stored when possible. Tasks split the columns.
static void gemmSmall(const Problem& P)
{
    const int m = P.m, n = P.n, k = P.k;
    const BMatrix& B = P.B;
    const int W = B.panelWidth;
    enum Kind { PANELS, ROWMAJOR, DOT, REPACK };
    const Kind kind = W ? PANELS : B.ld1 == 1 ? ROWMAJOR : (B.ld0 == 1 && m <= 4) ? DOT : REPACK;
    const bool wide = m <= 2;  // one or two rows: 32 columns per block use whole cache lines
    const int rowCols = wide ? 32 : 16;
    const int unit = kind == PANELS ? W : kind == ROWMAJOR ? rowCols : kind == DOT ? 4 : (int)GEMM_NR;
    const int NV = kind == ROWMAJOR ? rowCols / 4 : kind == REPACK ? 3
                 : W % 32 == 0 && wide ? 8 : W % 16 == 0 ? 4 : W % 12 == 0 ? 3 : 2;
    const int maxRows = kind == DOT ? 4 : smallMaxRows(NV);
    const int passes = (m + maxRows - 1) / maxRows;

    const int nUnits = (n + unit - 1) / unit;
    const int target = P.nthreads > 1 ? 4 * P.nthreads : 1;
    const int byWork = (int)std::max(1., (double)m * n * k / GEMM_SMALL_TASK_MACS);
    int tN = std::max(1, std::min(std::min((target + P.batch - 1) / P.batch, byWork), nUnits));
    const int unitsPerTask = (nUnits + tN - 1) / tN;
    tN = (nUnits + unitsPerTask - 1) / unitsPerTask;

    const bool copyA = P.lda1 != 1;
    const size_t aSize = copyA ? (size_t)m * k : 0;
    const size_t bSize = kind == ROWMAJOR ? (size_t)rowCols * k : kind == REPACK ? (size_t)unitsPerTask * GEMM_NR * k : 0;

    runTasks(P.batch * tN, P.nthreads, [&](const Range& r) {
        AutoBuffer<float> buf(aSize + bSize);
        float* abuf = buf.data();
        float* bbuf = abuf + aSize;
        int lastB = -1;
        for (int t = r.start; t < r.end; t++)
        {
            const int b = t / tN;
            const int j0 = t % tN * unitsPerTask * unit, nc = std::min(unitsPerTask * unit, n - j0);
            const float* A = P.A + P.aOfs[b];
            size_t lda = P.lda0;
            if (copyA)
            {
                // Transposed A: make rows contiguous once per problem.
                if (b != lastB)
                    for (int i = 0; i < m; i++)
                        for (int p = 0; p < k; p++)
                            abuf[(size_t)i * k + p] = A[i * P.lda0 + p * P.lda1];
                A = abuf;
                lda = k;
            }
            lastB = b;
            float* C = P.C + P.cOfs[b];
            const float* Bp = B.ptr + (W ? 0 : P.bOfs[b]);
            if (kind == REPACK)
                packBPanels(nc, k, Bp + j0 * B.ld1, B.ld0, B.ld1, GEMM_NR, bbuf);

            for (int j = j0; j < j0 + nc; )
            {
                // One column block: pointer, row stride and width.
                const float* bb;
                size_t ldb;
                int cols;
                if (kind == PANELS)
                {
                    const int sub = 4 * NV, inPanel = j % W;
                    bb = Bp + (size_t)(j / W) * W * k + inPanel;
                    ldb = W;
                    cols = std::min(std::min(sub, W - inPanel), n - j);
                }
                else if (kind == ROWMAJOR)
                {
                    cols = std::min(rowCols, n - j);
                    bb = Bp + j;
                    ldb = B.ld0;
                    if (cols < rowCols)
                    {
                        // Tail columns: copy into a zero-padded block so loads stay in bounds.
                        for (int p = 0; p < k; p++)
                            for (int c = 0; c < rowCols; c++)
                                bbuf[(size_t)p * rowCols + c] = c < cols ? Bp[p * B.ld0 + j + c] : 0.f;
                        bb = bbuf;
                        ldb = rowCols;
                    }
                }
                else if (kind == REPACK)
                {
                    cols = std::min((int)GEMM_NR, n - j);
                    bb = bbuf + (size_t)(j - j0) * k;
                    ldb = GEMM_NR;
                }
                else
                {
                    cols = std::min(4, n - j);
                    for (int r0 = 0; r0 < m; r0 += 4)
                        dotBlockRows(std::min(4, m - r0), k, A + r0 * lda, lda, Bp + j * B.ld1, B.ld1,
                                     C + r0 * P.ldc + j, P.ldc, P.alpha, P.beta, cols);
                    j += cols;
                    continue;
                }
                // Rows in balanced passes, the B block stays hot in cache between them.
                for (int q = 0, r0 = 0; q < passes; q++)
                {
                    const int mr = (m - r0) / (passes - q);
                    if (NV == 8)
                        smallBlockRows<8>(mr, k, A + r0 * lda, lda, bb, ldb, C + r0 * P.ldc + j, P.ldc, P.alpha, P.beta, cols);
                    else if (NV == 4)
                        smallBlockRows<4>(mr, k, A + r0 * lda, lda, bb, ldb, C + r0 * P.ldc + j, P.ldc, P.alpha, P.beta, cols);
                    else if (NV == 3)
                        smallBlockRows<3>(mr, k, A + r0 * lda, lda, bb, ldb, C + r0 * P.ldc + j, P.ldc, P.alpha, P.beta, cols);
                    else
                        smallBlockRows<2>(mr, k, A + r0 * lda, lda, bb, ldb, C + r0 * P.ldc + j, P.ldc, P.alpha, P.beta, cols);
                    r0 += mr;
                }
                j += cols;
            }
        }
    });
}

#if defined(CV_GEMM_HAVE_SME)
// SME: near-square tiles, about 2 per thread. A and B are packed outside streaming mode,
// then one streaming call runs all tiles of a task for each k block.
static void gemmSME(const Problem& P)
{
    const int m = P.m, n = P.n, k = P.k;
    const int vl = smeVectorLength(), W = 2 * vl;
    const int R = m <= vl ? vl : W;            // short A uses the VL x 4VL kernel
    const int colUnit = R == vl ? 2 * W : W;
    // ZA holds the whole k unless both sides are large, where k blocks keep the panels in cache.
    const int KC = k <= 3 * SME_KC || m < SME_BIG || n < SME_BIG ? k : (int)SME_KC;
    const bool direct = P.B.panelWidth == W;

    const int rowUnits = (m + R - 1) / R, colUnits = (n + colUnit - 1) / colUnit;
    const int target = P.nthreads > 1 ? 2 * P.nthreads : 1;
    int rb = rowUnits, cb = colUnits;
    auto tasks = [&]() { return P.batch * ((rowUnits + rb - 1) / rb) * ((colUnits + cb - 1) / cb); };
    while (tasks() < target && (rb > 1 || cb > 1) && (double)rb * R * cb * colUnit * k >= 2. * SME_TASK_MACS)
    {
        if ((rb * R >= cb * colUnit && rb > 1) || cb == 1)
            rb = (rb + 1) / 2;
        else
            cb = (cb + 1) / 2;
    }
    const int rowsPerTask = rb * R, colsPerTask = cb * colUnit;
    const int tM = (m + rowsPerTask - 1) / rowsPerTask, tN = (n + colsPerTask - 1) / colsPerTask;
    const int perProblem = tM * tN;
    // An operand read by several tasks is packed once per k block instead of once per task.
    const bool shareA = P.batch == 1 && tN > 1, shareB = !direct && P.batch == 1 && tM > 1;
    const int aPanels = shareA ? (m + R - 1) / R : 0, bPanels = shareB ? (n + W - 1) / W : 0;
    AutoBuffer<float> shared((size_t)aPanels * R * KC + (size_t)bPanels * W * KC);
    float* sharedA = shared.data();
    float* sharedB = sharedA + (size_t)aPanels * R * KC;
    const size_t aSize = shareA ? 0 : (size_t)rowsPerTask * KC, bSize = direct || shareB ? 0 : (size_t)colsPerTask * KC;

    for (int p0 = 0; p0 < k; p0 += KC)
    {
        const int kc = std::min(KC, k - p0);
        const float beta = p0 == 0 ? P.beta : 1.f;
        if (aPanels + bPanels > 0)
        {
            runTasks(aPanels + bPanels, P.nthreads, [&](const Range& r) {
                for (int q = r.start; q < r.end; q++)
                {
                    if (q < aPanels)
                    {
                        const int i0 = q * R;
                        packPanel(std::min(R, m - i0), kc, P.A + P.aOfs[0] + i0 * P.lda0 + p0 * P.lda1,
                                  P.lda1, P.lda0, R, sharedA + (size_t)i0 * kc);
                    }
                    else
                    {
                        const int j0 = (q - aPanels) * W;
                        packPanel(std::min(W, n - j0), kc, P.B.ptr + P.bOfs[0] + p0 * P.B.ld0 + j0 * P.B.ld1,
                                  P.B.ld0, P.B.ld1, W, sharedB + (size_t)j0 * kc);
                    }
                }
            });
        }
        runTasks(P.batch * perProblem, P.nthreads, [&](const Range& r) {
            AutoBuffer<float> buf(aSize + bSize);
            float* abuf = buf.data();
            float* bbuf = abuf + aSize;
            for (int t = r.start; t < r.end; t++)
            {
                const int b = t / perProblem, rem = t - b * perProblem;
                const int i0 = rem / tN * rowsPerTask, j0 = rem % tN * colsPerTask;
                const int mc = std::min(rowsPerTask, m - i0), nc = std::min(colsPerTask, n - j0);
                const float* ap = sharedA + (size_t)i0 * kc;
                if (!shareA)
                {
                    const float* a = P.A + P.aOfs[b] + i0 * P.lda0 + p0 * P.lda1;
                    for (int ii = 0; ii < mc; ii += R)
                        packPanel(std::min(R, mc - ii), kc, a + ii * P.lda0, P.lda1, P.lda0, R, abuf + (size_t)ii * kc);
                    ap = abuf;
                }
                const float* bp;
                size_t bStride = (size_t)W * kc;
                if (direct)
                {
                    bp = P.B.ptr + (size_t)j0 * k + (size_t)p0 * W;
                    bStride = (size_t)W * k;
                }
                else if (shareB)
                    bp = sharedB + (size_t)j0 * kc;
                else
                {
                    packBPanels(nc, kc, P.B.ptr + P.bOfs[b] + p0 * P.B.ld0 + j0 * P.B.ld1, P.B.ld0, P.B.ld1, W, bbuf);
                    bp = bbuf;
                }
                sme::gemmPanels(mc, nc, kc, R, ap, (size_t)R * kc, bp, bStride, P.alpha, beta,
                                P.C + P.cOfs[b] + i0 * P.ldc + j0, P.ldc);
            }
        });
    }
}
#endif

void gemmBatch(int batch, const size_t* A_offsets, const size_t* B_offsets, const size_t* C_offsets,
               int m, int n, int k, float alpha, const float* A, size_t lda0, size_t lda1,
               const BMatrix& B, float beta, float* C, size_t ldc, bool multiThread, bool sme)
{
    const Problem P = { batch, A_offsets, B_offsets, C_offsets, m, n, k, alpha, beta, A, lda0, lda1,
                        B, C, ldc, multiThread ? std::max(getNumThreads(), 1) : 1 };
#if defined(CV_GEMM_HAVE_SME)
    const int vl = sme ? smeVectorLength() : 0;
    if (vl && m >= SME_MIN_M && (B.panelWidth == 0 || B.panelWidth == 2 * vl) && (double)m * n * k >= SME_MIN_MACS)
    {
        gemmSME(P);
        return;
    }
#else
    CV_UNUSED(sme);
#endif
    if (m <= SMALL_M_MAX)
        gemmSmall(P);
    else
        gemmBlocked(P);
}

// Below these sizes the old cv::gemm code is as fast and accumulates in double (measured on Apple M5).
static bool engineWorthIt(int m, int n, int k)
{
    return m >= 6 && n >= 8 && k >= 8 && (int64)m * n * k >= 1024;
}
#endif // CV_GEMM_ENGINE_NEON

bool gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
             float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags)
{
#if CV_GEMM_ENGINE_NEON
    static const bool enabled = utils::getConfigurationParameterBool("OPENCV_CORE_GEMM_ENGINE", true);
    const bool transA = (flags & CV_HAL_GEMM_1_T) != 0, transB = (flags & CV_HAL_GEMM_2_T) != 0;
    const bool transC = (flags & CV_HAL_GEMM_3_T) != 0;
    const int m = transA ? n_a : m_a, k = transA ? m_a : n_a, n = n_d;
    const size_t esz = sizeof(float);
    if (!enabled || !engineWorthIt(m, n, k) || dst == src1 || dst == src2 ||
        (src1_step | src2_step | src3_step | dst_step) % esz != 0)
        return false;

    const bool haveC = src3 && src3_step && beta != 0.f;
    if (haveC && src3 == dst && transC)
        return false;

    const size_t lda = src1_step / esz, ldb = src2_step / esz, ldd = dst_step / esz, ldc = src3_step / esz;
    if (haveC && src3 != dst)
    {
        // D = beta*op(C) first, then the product is added on top of it.
        parallel_for_(Range(0, m), [&](const Range& r) {
            for (int i = r.start; i < r.end; i++)
            {
                float* d = dst + i * ldd;
                if (transC)
                    for (int j = 0; j < n; j++)
                        d[j] = beta * src3[j * ldc + i];
                else
                    for (int j = 0; j < n; j++)
                        d[j] = beta * src3[i * ldc + j];
            }
        }, (double)m * n * (1.0 / 65536));
        beta = 1.f;
    }
    const size_t zero = 0;
    const BMatrix B = { src2, transB ? 1 : ldb, transB ? ldb : 1, 0 };
    gemmBatch(1, &zero, &zero, &zero, m, n, k, alpha, src1, transA ? 1 : lda, transA ? lda : 1,
              B, haveC ? beta : 0.f, dst, ldd, true, smeEnabled());
    return true;
#else
    CV_UNUSED(src1); CV_UNUSED(src1_step); CV_UNUSED(src2); CV_UNUSED(src2_step);
    CV_UNUSED(alpha); CV_UNUSED(src3); CV_UNUSED(src3_step); CV_UNUSED(beta);
    CV_UNUSED(dst); CV_UNUSED(dst_step); CV_UNUSED(m_a); CV_UNUSED(n_a); CV_UNUSED(n_d); CV_UNUSED(flags);
    return false;
#endif
}

} // namespace gemm_engine

namespace hal {

using namespace gemm_engine;

bool gemm32fUsesSME()
{
    return smeEnabled();
}

size_t gemm32fPackBSize(int n, int k, int flags)
{
    CV_INSTRUMENT_REGION();
    CV_Assert(n > 0 && k > 0);
    size_t halSize = 0;
    int res = cv_hal_gemm32f_packb_size(n, k, flags, &halSize);
    if (res != CV_HAL_ERROR_OK && res != CV_HAL_ERROR_NOT_IMPLEMENTED)
        CV_Error_(cv::Error::StsInternal, ("HAL implementation gemm32fPackBSize returned %d", res));
    if (res != CV_HAL_ERROR_OK)
        halSize = 0;
    // Room for either format, so packing can still fall back if the HAL declines.
    return (size_t)PACKED_HEADER_SIZE + std::max(halSize, packedPayloadSize(n, k));
}

void gemm32fPackB(const float* B, size_t B_step, int n, int k, int flags, void* packedB)
{
    CV_INSTRUMENT_REGION();
    CV_Assert(B && packedB && n > 0 && k > 0 && B_step % sizeof(float) == 0);
    PackedBHeader* hdr = (PackedBHeader*)packedB;
    int res = cv_hal_gemm32f_packb(B, B_step, n, k, flags, (uchar*)packedB + PACKED_HEADER_SIZE);
    if (res == CV_HAL_ERROR_OK)
    {
        hdr->magic = PACKED_MAGIC;
        hdr->format = PACKED_HAL;
        hdr->n = n;
        hdr->k = k;
        hdr->width = 0;
        hdr->sme = 0;
        return;
    }
    if (res != CV_HAL_ERROR_NOT_IMPLEMENTED)
        CV_Error_(cv::Error::StsInternal, ("HAL implementation gemm32fPackB returned %d", res));

    const size_t ldb = B_step / sizeof(float);
    const bool transB = (flags & CV_HAL_GEMM_2_T) != 0;
    packB(n, k, B, transB ? 1 : ldb, transB ? ldb : 1, packedB, (flags & CV_HAL_GEMM_SINGLE_THREAD) == 0);
}

void gemm32fPacked(const float* A, size_t A_step, const void* packedB,
                   float* C, size_t C_step, int m, int n, int k,
                   float alpha, float beta, int flags)
{
    CV_INSTRUMENT_REGION();
    const PackedBHeader& hdr = checkPackedB(packedB, n, k);
    CV_Assert(m >= 0 && A_step % sizeof(float) == 0 && C_step % sizeof(float) == 0);
    if (m == 0)
        return;
    CV_Assert(A && C);
    const void* payload = (const uchar*)packedB + PACKED_HEADER_SIZE;

    if (hdr.format == PACKED_HAL)
    {
        CALL_HAL(gemm32fPacked, cv_hal_gemm32f_packed, A, A_step, payload, C, C_step, m, n, k, alpha, beta, flags);
        CV_Error(cv::Error::StsBadArg, "packedB was packed by a HAL that cannot compute this product");
    }

    const bool transA = (flags & CV_HAL_GEMM_1_T) != 0;
#if CV_GEMM_ENGINE_NEON
    if (hdr.format == PACKED_PANELS)
    {
        const size_t lda = A_step / sizeof(float), zero = 0;
        const BMatrix B = { (const float*)payload, 0, 0, hdr.width };
        gemmBatch(1, &zero, &zero, &zero, m, n, k, alpha, A, transA ? 1 : lda, transA ? lda : 1, B, beta,
                  C, C_step / sizeof(float), (flags & CV_HAL_GEMM_SINGLE_THREAD) == 0, hdr.sme && smeEnabled());
        return;
    }
#endif
    CV_CheckEQ((int)hdr.format, (int)PACKED_PLAIN, "packedB format is not supported by this build");
    hal::gemm32f(A, A_step, (const float*)payload, n * sizeof(float), alpha, C, C_step, beta, C, C_step,
                 transA ? k : m, transA ? m : k, n, flags & CV_HAL_GEMM_1_T);
}

void gemm32fBatch(int batch,
                  const float* A, const size_t* A_offsets, size_t A_step,
                  const float* B, const size_t* B_offsets, size_t B_step,
                  float* C, const size_t* C_offsets, size_t C_step,
                  int m, int n, int k, float alpha, float beta, int flags)
{
    CV_INSTRUMENT_REGION();
    CV_Assert(batch >= 0 && m >= 0 && n >= 0 && k >= 0);
    CV_Assert((A_step | B_step | C_step) % sizeof(float) == 0);
    if (batch == 0 || m == 0 || n == 0)
        return;
    CV_Assert(C && C_offsets);
    if (k == 0)
    {
        for (int i = 0; i < batch; i++)
            scaleC(C + C_offsets[i], C_step / sizeof(float), m, n, beta);
        return;
    }
    CV_Assert(A && A_offsets && B && B_offsets);

    CALL_HAL(gemm32fBatch, cv_hal_gemm32f_batch, batch, A, A_offsets, A_step, B, B_offsets, B_step,
             C, C_offsets, C_step, m, n, k, alpha, beta, flags);

    const bool transA = (flags & CV_HAL_GEMM_1_T) != 0;
#if CV_GEMM_ENGINE_NEON
    const bool transB = (flags & CV_HAL_GEMM_2_T) != 0;
    const size_t lda = A_step / sizeof(float), ldb = B_step / sizeof(float);
    const BMatrix Bm = { B, transB ? 1 : ldb, transB ? ldb : 1, 0 };
    gemmBatch(batch, A_offsets, B_offsets, C_offsets, m, n, k, alpha,
              A, transA ? 1 : lda, transA ? lda : 1, Bm, beta, C, C_step / sizeof(float),
              (flags & CV_HAL_GEMM_SINGLE_THREAD) == 0, smeEnabled());
#else
    for (int i = 0; i < batch; i++)
    {
        float* c = C + C_offsets[i];
        hal::gemm32f(A + A_offsets[i], A_step, B + B_offsets[i], B_step, alpha, c, C_step, beta, c, C_step,
                     transA ? k : m, transA ? m : k, n, flags & (CV_HAL_GEMM_1_T | CV_HAL_GEMM_2_T));
    }
#endif
}

} // namespace hal
} // namespace cv
