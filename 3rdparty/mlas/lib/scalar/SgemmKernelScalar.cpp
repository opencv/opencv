/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelScalar.cpp

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
        MlasSgemmCopyPackB or MlasSgemmTransposePackB with a packing width
        of 16.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scaler multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    float Row0Block00;
    float Row0Block01;
    float Row0Block02;
    float Row0Block03;

    float Row1Block00;
    float Row1Block01;
    float Row1Block02;
    float Row1Block03;

#if defined(_WIN32)

    if (!ProcessTwoRows) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

#endif

    int countb = 0;

    do {

        float BElements00;
        float BElements01;
        float BElements02;
        float BElements03;

        float Row0AElements0;
        float Row0AElements1;
        float Row1AElements0;
        float Row1AElements1;

        //
        // Clear the block accumulators.
        //

        Row0Block00 = 0.0f;
        Row0Block01 = 0.0f;
        Row0Block02 = 0.0f;
        Row0Block03 = 0.0f;

        if (ProcessTwoRows) {
            Row1Block00 = 0.0f;
            Row1Block01 = 0.0f;
            Row1Block02 = 0.0f;
            Row1Block03 = 0.0f;
        }

        //
        // Compute the 4x1 or 4x2 output block.
        //

        const float* a = A;
        const float* b = B;
        size_t k = CountK;

        while (k >= 2) {

            Row0AElements0 = a[0];
            Row0AElements1 = a[1];

            if (ProcessTwoRows) {
                Row1AElements0 = a[lda];
                Row1AElements1 = a[lda + 1];
            }

            BElements00 = b[0];
            BElements01 = b[1];
            BElements02 = b[2];
            BElements03 = b[3];
            Row0Block00 = Row0Block00 + BElements00 * Row0AElements0;
            Row0Block01 = Row0Block01 + BElements01 * Row0AElements0;
            Row0Block02 = Row0Block02 + BElements02 * Row0AElements0;
            Row0Block03 = Row0Block03 + BElements03 * Row0AElements0;

            if (ProcessTwoRows) {
                Row1Block00 = Row1Block00 + BElements00 * Row1AElements0;
                Row1Block01 = Row1Block01 + BElements01 * Row1AElements0;
                Row1Block02 = Row1Block02 + BElements02 * Row1AElements0;
                Row1Block03 = Row1Block03 + BElements03 * Row1AElements0;
            }

            BElements00 = b[16];
            BElements01 = b[17];
            BElements02 = b[18];
            BElements03 = b[19];
            Row0Block00 = Row0Block00 + BElements00 * Row0AElements1;
            Row0Block01 = Row0Block01 + BElements01 * Row0AElements1;
            Row0Block02 = Row0Block02 + BElements02 * Row0AElements1;
            Row0Block03 = Row0Block03 + BElements03 * Row0AElements1;

            if (ProcessTwoRows) {
                Row1Block00 = Row1Block00 + BElements00 * Row1AElements1;
                Row1Block01 = Row1Block01 + BElements01 * Row1AElements1;
                Row1Block02 = Row1Block02 + BElements02 * Row1AElements1;
                Row1Block03 = Row1Block03 + BElements03 * Row1AElements1;
            }

            a += 2;
            b += 32;
            k -= 2;
        }

        if (k > 0) {

            Row0AElements0 = a[0];

            if (ProcessTwoRows) {
                Row1AElements0 = a[lda];
            }

            BElements00 = b[0];
            BElements01 = b[1];
            BElements02 = b[2];
            BElements03 = b[3];
            Row0Block00 = Row0Block00 + BElements00 * Row0AElements0;
            Row0Block01 = Row0Block01 + BElements01 * Row0AElements0;
            Row0Block02 = Row0Block02 + BElements02 * Row0AElements0;
            Row0Block03 = Row0Block03 + BElements03 * Row0AElements0;

            if (ProcessTwoRows) {
                Row1Block00 = Row1Block00 + BElements00 * Row1AElements0;
                Row1Block01 = Row1Block01 + BElements01 * Row1AElements0;
                Row1Block02 = Row1Block02 + BElements02 * Row1AElements0;
                Row1Block03 = Row1Block03 + BElements03 * Row1AElements0;
            }
        }

        //
        // Multiply by the alpha value.
        //

        Row0Block00 = Row0Block00 * alpha;
        Row0Block01 = Row0Block01 * alpha;
        Row0Block02 = Row0Block02 * alpha;
        Row0Block03 = Row0Block03 * alpha;

        if (ProcessTwoRows) {
            Row1Block00 = Row1Block00 * alpha;
            Row1Block01 = Row1Block01 * alpha;
            Row1Block02 = Row1Block02 * alpha;
            Row1Block03 = Row1Block03 * alpha;
        }

        if (CountN >= 4) {

            //
            // Store the entire output block.
            //

            if (!ZeroMode) {
                Row0Block00 = Row0Block00 + C[0];
                Row0Block01 = Row0Block01 + C[1];
                Row0Block02 = Row0Block02 + C[2];
                Row0Block03 = Row0Block03 + C[3];
            }

            C[0] = Row0Block00;
            C[1] = Row0Block01;
            C[2] = Row0Block02;
            C[3] = Row0Block03;

            if (ProcessTwoRows) {

                if (!ZeroMode) {
                    Row1Block00 = Row1Block00 + C[ldc];
                    Row1Block01 = Row1Block01 + C[ldc + 1];
                    Row1Block02 = Row1Block02 + C[ldc + 2];
                    Row1Block03 = Row1Block03 + C[ldc + 3];
                }

                C[ldc] = Row1Block00;
                C[ldc + 1] = Row1Block01;
                C[ldc + 2] = Row1Block02;
                C[ldc + 3] = Row1Block03;
            }

        } else {

            //
            // Store the partial output block.
            //
            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Row0Block00 = Row0Block00 + C[0];
                    Row0Block01 = Row0Block01 + C[1];
                }

                C[0] = Row0Block00;
                C[1] = Row0Block01;
                Row0Block00 = Row0Block02;
                Row0Block01 = Row0Block03;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block00 = Row1Block00 + C[ldc];
                        Row1Block01 = Row1Block01 + C[ldc + 1];
                    }

                    C[ldc] = Row1Block00;
                    C[ldc + 1] = Row1Block01;
                    Row1Block00 = Row1Block02;
                    Row1Block01 = Row1Block03;
                }

                C += 2;
            }

            if ((CountN & 1) != 0) {

                if (!ZeroMode) {
                    Row0Block00 = Row0Block00 + C[0];
                }

                C[0] = Row0Block00;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block00 = Row1Block00 + C[ldc];
                    }

                    C[ldc] = Row1Block00;
                }
            }

            break;
        }

        B += 4;
        C += 4;
        CountN -= 4;

        countb = (countb + 1) % 4;
        if (countb == 0) {
            B += CountK * 16 - 16;
        }
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

    alpha - Supplies the scaler multiplier (see SGEMM definition).

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

    alpha - Supplies the scaler multiplier (see SGEMM definition).

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

    alpha - Supplies the scaler multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    return MlasSgemmKernel<false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}
