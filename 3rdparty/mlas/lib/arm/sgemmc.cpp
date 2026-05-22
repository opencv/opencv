/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemmc.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#include "mlasi.h"

template<bool ZeroMode, bool ProcessTwoRows>
size_t
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    float32x4_t Row0Block0;
    float32x4_t Row0Block1;
    float32x4_t Row0Block2;
    float32x4_t Row0Block3;

    float32x4_t Row1Block0;
    float32x4_t Row1Block1;
    float32x4_t Row1Block2;
    float32x4_t Row1Block3;

#if defined(_WIN32)

    if (!ProcessTwoRows) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

#endif

    do {

        float32x4_t BElements0;
        float32x4_t BElements1;
        float32x4_t BElements2;
        float32x4_t BElements3;

        float32x2_t Row0AElements;
        float32x2_t Row1AElements;

        //
        // Clear the block accumulators.
        //

        Row0Block0 = vdupq_n_f32(0.0f);
        Row0Block1 = vdupq_n_f32(0.0f);
        Row0Block2 = vdupq_n_f32(0.0f);
        Row0Block3 = vdupq_n_f32(0.0f);

        if (ProcessTwoRows) {
            Row1Block0 = vdupq_n_f32(0.0f);
            Row1Block1 = vdupq_n_f32(0.0f);
            Row1Block2 = vdupq_n_f32(0.0f);
            Row1Block3 = vdupq_n_f32(0.0f);
        }

        //
        // Compute the 16x1 or 16x2 output block.
        //

        const float* a = A;
        size_t k = CountK;

        while (k >= 2) {

            Row0AElements = vld1_f32(a);

            if (ProcessTwoRows) {
                Row1AElements = vld1_f32(a + lda);
            }

            BElements0 = vld1q_f32(B + 0);
            BElements1 = vld1q_f32(B + 4);
            BElements2 = vld1q_f32(B + 8);
            BElements3 = vld1q_f32(B + 12);

            Row0Block0 = vmlaq_lane_f32(Row0Block0, BElements0, Row0AElements, 0);
            Row0Block1 = vmlaq_lane_f32(Row0Block1, BElements1, Row0AElements, 0);
            Row0Block2 = vmlaq_lane_f32(Row0Block2, BElements2, Row0AElements, 0);
            Row0Block3 = vmlaq_lane_f32(Row0Block3, BElements3, Row0AElements, 0);

            if (ProcessTwoRows) {
                Row1Block0 = vmlaq_lane_f32(Row1Block0, BElements0, Row1AElements, 0);
                Row1Block1 = vmlaq_lane_f32(Row1Block1, BElements1, Row1AElements, 0);
                Row1Block2 = vmlaq_lane_f32(Row1Block2, BElements2, Row1AElements, 0);
                Row1Block3 = vmlaq_lane_f32(Row1Block3, BElements3, Row1AElements, 0);
            }

            BElements0 = vld1q_f32(B + 16);
            BElements1 = vld1q_f32(B + 20);
            BElements2 = vld1q_f32(B + 24);
            BElements3 = vld1q_f32(B + 28);

            Row0Block0 = vmlaq_lane_f32(Row0Block0, BElements0, Row0AElements, 1);
            Row0Block1 = vmlaq_lane_f32(Row0Block1, BElements1, Row0AElements, 1);
            Row0Block2 = vmlaq_lane_f32(Row0Block2, BElements2, Row0AElements, 1);
            Row0Block3 = vmlaq_lane_f32(Row0Block3, BElements3, Row0AElements, 1);

            if (ProcessTwoRows) {
                Row1Block0 = vmlaq_lane_f32(Row1Block0, BElements0, Row1AElements, 1);
                Row1Block1 = vmlaq_lane_f32(Row1Block1, BElements1, Row1AElements, 1);
                Row1Block2 = vmlaq_lane_f32(Row1Block2, BElements2, Row1AElements, 1);
                Row1Block3 = vmlaq_lane_f32(Row1Block3, BElements3, Row1AElements, 1);
            }

            a += 2;
            B += 32;
            k -= 2;
        }

        if (k > 0) {

            Row0AElements = vld1_dup_f32(a);

            if (ProcessTwoRows) {
                Row1AElements = vld1_dup_f32(a + lda);
            }

            BElements0 = vld1q_f32(B + 0);
            BElements1 = vld1q_f32(B + 4);
            BElements2 = vld1q_f32(B + 8);
            BElements3 = vld1q_f32(B + 12);

            Row0Block0 = vmlaq_lane_f32(Row0Block0, BElements0, Row0AElements, 0);
            Row0Block1 = vmlaq_lane_f32(Row0Block1, BElements1, Row0AElements, 0);
            Row0Block2 = vmlaq_lane_f32(Row0Block2, BElements2, Row0AElements, 0);
            Row0Block3 = vmlaq_lane_f32(Row0Block3, BElements3, Row0AElements, 0);

            if (ProcessTwoRows) {
                Row1Block0 = vmlaq_lane_f32(Row1Block0, BElements0, Row1AElements, 0);
                Row1Block1 = vmlaq_lane_f32(Row1Block1, BElements1, Row1AElements, 0);
                Row1Block2 = vmlaq_lane_f32(Row1Block2, BElements2, Row1AElements, 0);
                Row1Block3 = vmlaq_lane_f32(Row1Block3, BElements3, Row1AElements, 0);
            }

            B += 16;
        }

        //
        // Multiply by the alpha value.
        //

        Row0Block0 = vmulq_n_f32(Row0Block0, alpha);
        Row0Block1 = vmulq_n_f32(Row0Block1, alpha);
        Row0Block2 = vmulq_n_f32(Row0Block2, alpha);
        Row0Block3 = vmulq_n_f32(Row0Block3, alpha);

        if (ProcessTwoRows) {
            Row1Block0 = vmulq_n_f32(Row1Block0, alpha);
            Row1Block1 = vmulq_n_f32(Row1Block1, alpha);
            Row1Block2 = vmulq_n_f32(Row1Block2, alpha);
            Row1Block3 = vmulq_n_f32(Row1Block3, alpha);
        }

        if (CountN >= 16) {

            //
            // Store the entire output block.
            //

            if (!ZeroMode) {
                Row0Block0 = vaddq_f32(Row0Block0, vld1q_f32(C));
                Row0Block1 = vaddq_f32(Row0Block1, vld1q_f32(C + 4));
                Row0Block2 = vaddq_f32(Row0Block2, vld1q_f32(C + 8));
                Row0Block3 = vaddq_f32(Row0Block3, vld1q_f32(C + 12));
            }

            vst1q_f32(C, Row0Block0);
            vst1q_f32(C + 4, Row0Block1);
            vst1q_f32(C + 8, Row0Block2);
            vst1q_f32(C + 12, Row0Block3);

            if (ProcessTwoRows) {

                if (!ZeroMode) {
                    Row1Block0 = vaddq_f32(Row1Block0, vld1q_f32(C + ldc));
                    Row1Block1 = vaddq_f32(Row1Block1, vld1q_f32(C + ldc + 4));
                    Row1Block2 = vaddq_f32(Row1Block2, vld1q_f32(C + ldc + 8));
                    Row1Block3 = vaddq_f32(Row1Block3, vld1q_f32(C + ldc + 12));
                }

                vst1q_f32(C + ldc, Row1Block0);
                vst1q_f32(C + ldc + 4, Row1Block1);
                vst1q_f32(C + ldc + 8, Row1Block2);
                vst1q_f32(C + ldc + 12, Row1Block3);
            }

        } else {

            //
            // Store the partial output block.
            //

            if ((CountN & 8) != 0) {

                if (!ZeroMode) {
                    Row0Block0 = vaddq_f32(Row0Block0, vld1q_f32(C));
                    Row0Block1 = vaddq_f32(Row0Block1, vld1q_f32(C + 4));
                }

                vst1q_f32(C, Row0Block0);
                vst1q_f32(C + 4, Row0Block1);
                Row0Block0 = Row0Block2;
                Row0Block1 = Row0Block3;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block0 = vaddq_f32(Row1Block0, vld1q_f32(C + ldc));
                        Row1Block1 = vaddq_f32(Row1Block1, vld1q_f32(C + ldc + 4));
                    }

                    vst1q_f32(C + ldc, Row1Block0);
                    vst1q_f32(C + ldc + 4, Row1Block1);
                    Row1Block0 = Row1Block2;
                    Row1Block1 = Row1Block3;
                }

                C += 8;
            }

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Row0Block0 = vaddq_f32(Row0Block0, vld1q_f32(C));
                }

                vst1q_f32(C, Row0Block0);
                Row0Block0 = Row0Block1;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block0 = vaddq_f32(Row1Block0, vld1q_f32(C + ldc));
                    }

                    vst1q_f32(C + ldc, Row1Block0);
                    Row1Block0 = Row1Block1;
                }

                C += 4;
            }

            float32x2_t Row0Block0High;
            float32x2_t Row0Block0Low;

            float32x2_t Row1Block0High;
            float32x2_t Row1Block0Low;

            Row0Block0High = vget_high_f32(Row0Block0);
            Row0Block0Low = vget_low_f32(Row0Block0);

            if (ProcessTwoRows) {
                Row1Block0High = vget_high_f32(Row1Block0);
                Row1Block0Low = vget_low_f32(Row1Block0);
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Row0Block0Low = vadd_f32(Row0Block0Low, vld1_f32(C));
                }

                vst1_f32(C, Row0Block0Low);
                Row0Block0Low = Row0Block0High;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block0Low = vadd_f32(Row1Block0Low, vld1_f32(C + ldc));
                    }

                    vst1_f32(C + ldc, Row1Block0Low);
                    Row1Block0Low = Row1Block0High;
                }

                C += 2;
            }

            if ((CountN & 1) != 0) {

                if (!ZeroMode) {
                    Row0Block0Low = vadd_f32(Row0Block0Low, vld1_dup_f32(C));
                }

                vst1_lane_f32(C, Row0Block0Low, 0);

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block0Low = vadd_f32(Row1Block0Low, vld1_dup_f32(C + ldc));
                    }

                    vst1_lane_f32(C + ldc, Row1Block0Low, 0);
                }
            }

            break;
        }

        C += 16;
        CountN -= 16;

    } while (CountN > 0);

    return ProcessTwoRows ? 2 : 1;
}

template<bool ZeroMode>
size_t
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;

    if (CountM >= 2) {
        RowsHandled = MlasSgemmKernel<ZeroMode, true>(A, B, C, CountK, CountN, lda, ldc, alpha);
    } else {
        RowsHandled = MlasSgemmKernel<ZeroMode, false>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    return RowsHandled;
}

size_t
MLASCALL
MlasSgemmKernelZero(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    return MlasSgemmKernel<true>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

size_t
MLASCALL
MlasSgemmKernelAdd(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    return MlasSgemmKernel<false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}
