/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemvKernelScalar.cpp

Abstract:

    This module implements the kernels for the single precision matrix/vector
    multiply operation (SGEMV).

--*/

#include "mlasi.h"

void
MLASCALL 
MlasGemvFloatKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows. This handles the special case of M=1.

    The elements in matrix B are not transposed.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldb - Supplies the first dimension of matrix B.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    None.

--*/
{
    if (ZeroMode && CountK > 0) {
        float* c = C;
        const float* b = B;
        const float A0 = A[0];
        auto N = CountN;
        constexpr size_t kWidth = 4;
        for (; N >= kWidth; N -= kWidth) {
            c[0] = A0 * b[0];
            c[1] = A0 * b[1];
            c[2] = A0 * b[2];
            c[3] = A0 * b[3];
            c += kWidth;
            b += kWidth;
        }

        for (; N > 0; N--) {
            c[0] = A0 * b[0];
            c++;
            b++;
        }
        A++;
        B += ldb;

        CountK--;
    }

    for (; CountK >= 4; CountK -= 4) {
        float* c = C;
        const float* b = B;
        const float* b2 = B + ldb * 2;

        const float A0 = A[0];
        const float A1 = A[1];
        const float A2 = A[2];
        const float A3 = A[3];

        constexpr size_t kWidth = 4;
        auto N = CountN;
        for (; N >= kWidth; N -= kWidth) {
            float c0 = c[0] + A0 * b[0];
            float c1 = c[1] + A0 * b[1];
            float c2 = c[2] + A0 * b[2];
            float c3 = c[3] + A0 * b[3];

            c0 += A1 * b[ldb + 0];
            c1 += A1 * b[ldb + 1];
            c2 += A1 * b[ldb + 2];
            c3 += A1 * b[ldb + 3];

            c0 += A2 * b2[0];
            c1 += A2 * b2[1];
            c2 += A2 * b2[2];
            c3 += A2 * b2[3];

            c0 += A3 * b2[ldb + 0];
            c1 += A3 * b2[ldb + 1];
            c2 += A3 * b2[ldb + 2];
            c3 += A3 * b2[ldb + 3];

            c[0] = c0;
            c[1] = c1;
            c[2] = c2;
            c[3] = c3;

            c += kWidth;
            b += kWidth;
            b2 += kWidth;
        }

        for (; N > 0; N--) {
            c[0] += A0 * b[0] + A1 * b[ldb] + A2 * b2[0] + A3 * b2[ldb];
            c++;
            b++;
            b2++;
        }

        B += 4 * ldb;
        A += 4;
    }

    for (; CountK > 0; CountK--) {
        float* c = C;
        const float* b = B;
        const float A0 = A[0];
        constexpr size_t kWidth = 4;
        auto N = CountN;
        for (; N >= kWidth; N -= kWidth) {
            c[0] += A0 * b[0];
            c[1] += A0 * b[1];
            c[2] += A0 * b[2];
            c[3] += A0 * b[3];

            c += kWidth;
            b += kWidth;
        }

        for (; N > 0; N--) {
            c[0] += A0 * b[0];
            c++;
            b++;
        }
        B += ldb;
        A++;
    }
}
