/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelPower.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/
#include "SgemmKernelpower.h"

size_t
MLASCALL
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
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

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;

    MLAS_FLOAT32X4 AlphaBroadcast = MlasBroadcastFloat32x4(alpha);

    if (CountM >= 4) {
        RowsHandled = MlasSgemmProcessCount<4>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 2) {
        RowsHandled = MlasSgemmProcessCount<2>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else {
        RowsHandled = MlasSgemmProcessCount<1>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    }

    return RowsHandled;
}
