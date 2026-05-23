/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelZVECTOR.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#include "SgemmKernelZVECTOR.h"

#include <vecintrin.h>

struct MlasSgemmBroadcastAElementsZVECTOR
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 ABroadcast[RowCount],
        const float* A,
        size_t lda
        )
    {
        ABroadcast[0][Row] = A [Row * lda];
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasSgemmComputeAElements(
    MLAS_FLOAT32X4 AElements[RowCount],
    MLAS_FLOAT32X4 ABroadcast[RowCount]
    )
{
        const __vector unsigned char mask0 = { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23 };
        const __vector unsigned char mask3 = { 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
        const __vector unsigned char mask_even = { 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27 };
        const __vector unsigned char mask_odd = { 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31 };

        __vector float a1,a2;

        a1 = vec_perm(AElements[0], AElements[1], mask_even);
        a2 = vec_perm(AElements[2], AElements[3], mask_even);
        ABroadcast[0] = vec_perm(a1, a2, mask0);
        ABroadcast[2] = vec_perm(a1, a2, mask3);
        a1 = vec_perm(AElements[0], AElements[1], mask_odd);
        a2 = vec_perm(AElements[2], AElements[3], mask_odd);
        ABroadcast[1] = vec_perm(a1, a2, mask0);
        ABroadcast[3] = vec_perm(a1, a2, mask3);
}
template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasSgemmComputeBlockZVECTOR(
    MLAS_FLOAT32X4 acc[32],
    MLAS_FLOAT32X4 ABroadcast,
    MLAS_FLOAT32X4 A2Broadcast,
    const float* B,
    size_t CountM
    )
{

    MLAS_FLOAT32X4 AElements[8];

    AElements[0] = vec_splats(ABroadcast[0]);
    AElements[1] = vec_splats(ABroadcast[1]);
    AElements[2] = vec_splats(ABroadcast[2]);
    AElements[3] = vec_splats(ABroadcast[3]);

    if (CountM == 8) {
        AElements[4] = vec_splats(A2Broadcast[0]);
        AElements[5] = vec_splats(A2Broadcast[1]);
        AElements[6] = vec_splats(A2Broadcast[2]);
        AElements[7] = vec_splats(A2Broadcast[3]);
    }

    MLAS_FLOAT32X4 BElements[4];

    BElements[0] = MlasLoadFloat32x4(B);
    BElements[1] = MlasLoadFloat32x4(B + 4);
    BElements[2] = MlasLoadFloat32x4(B + 8);
    BElements[3] = MlasLoadFloat32x4(B + 12);

    acc[0]  = __builtin_s390_vfmasb(AElements[0], BElements[0], acc[0]);
    acc[1]  = __builtin_s390_vfmasb(AElements[1], BElements[0], acc[1]);
    acc[2]  = __builtin_s390_vfmasb(AElements[2], BElements[0], acc[2]);
    acc[3]  = __builtin_s390_vfmasb(AElements[3], BElements[0], acc[3]);

    acc[4]  = __builtin_s390_vfmasb(AElements[0], BElements[1], acc[4]);
    acc[5]  = __builtin_s390_vfmasb(AElements[1], BElements[1], acc[5]);
    acc[6]  = __builtin_s390_vfmasb(AElements[2], BElements[1], acc[6]);
    acc[7]  = __builtin_s390_vfmasb(AElements[3], BElements[1], acc[7]);

    acc[8]  = __builtin_s390_vfmasb(AElements[0], BElements[2], acc[8]);
    acc[9]  = __builtin_s390_vfmasb(AElements[1], BElements[2], acc[9]);
    acc[10] = __builtin_s390_vfmasb(AElements[2], BElements[2], acc[10]);
    acc[11] = __builtin_s390_vfmasb(AElements[3], BElements[2], acc[11]);

    acc[12] = __builtin_s390_vfmasb(AElements[0], BElements[3], acc[12]);
    acc[13] = __builtin_s390_vfmasb(AElements[1], BElements[3], acc[13]);
    acc[14] = __builtin_s390_vfmasb(AElements[2], BElements[3], acc[14]);
    acc[15] = __builtin_s390_vfmasb(AElements[3], BElements[3], acc[15]);

    if (CountM == 8) {
        acc[16] = __builtin_s390_vfmasb(AElements[4], BElements[0], acc[16]);
        acc[17] = __builtin_s390_vfmasb(AElements[5], BElements[0], acc[17]);
        acc[18] = __builtin_s390_vfmasb(AElements[6], BElements[0], acc[18]);
        acc[19] = __builtin_s390_vfmasb(AElements[7], BElements[0], acc[19]);

        acc[20] = __builtin_s390_vfmasb(AElements[4], BElements[1], acc[20]);
        acc[21] = __builtin_s390_vfmasb(AElements[5], BElements[1], acc[21]);
        acc[22] = __builtin_s390_vfmasb(AElements[6], BElements[1], acc[22]);
        acc[23] = __builtin_s390_vfmasb(AElements[7], BElements[1], acc[23]);

        acc[24] = __builtin_s390_vfmasb(AElements[4], BElements[2], acc[24]);
        acc[25] = __builtin_s390_vfmasb(AElements[5], BElements[2], acc[25]);
        acc[26] = __builtin_s390_vfmasb(AElements[6], BElements[2], acc[26]);
        acc[27] = __builtin_s390_vfmasb(AElements[7], BElements[2], acc[27]);

        acc[28] = __builtin_s390_vfmasb(AElements[4], BElements[3], acc[28]);
        acc[29] = __builtin_s390_vfmasb(AElements[5], BElements[3], acc[29]);
        acc[30] = __builtin_s390_vfmasb(AElements[6], BElements[3], acc[30]);
        acc[31] = __builtin_s390_vfmasb(AElements[7], BElements[3], acc[31]);
    }
}
template<size_t VectorCount>
struct MlasSgemmStoreVectorZVECTOR
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Result[4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        MLAS_FLOAT32X4 *rowC;
        if (ZeroMode) {
            rowC = reinterpret_cast<MLAS_FLOAT32X4 *>(&C[Row * ldc + VectorCount]);
            rowC[0] = Result[Row] * AlphaBroadcast;
        } else {
            rowC = reinterpret_cast<MLAS_FLOAT32X4 *>(&C[Row * ldc + VectorCount]);
            rowC[0] += Result[Row] * AlphaBroadcast;
        }
    }
};

struct MlasSgemmMultiplyAlphaTrailingZVECTOR
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[RowCount],
        MLAS_FLOAT32X4 AlphaBroadcast
        )
    {
        Accumulators[Row] = MlasMultiplyFloat32x4(Accumulators[Row], AlphaBroadcast);
    }
};
template<unsigned Lane>
struct MlasSgemmStoreScalarZVECTOR
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[RowCount],
        float* C,
        size_t ldc,
        bool ZeroMode
        )
    {
        float* c = C + Row * ldc + Lane;
        float Value = Accumulators[Row][Lane];
        if (!ZeroMode) {
            Value += *c;
        }

        *c = Value;
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
size_t
MlasSgemmZVECTORProcessCount(
    const float* A,
    const float* B,
    float* C,
    size_t CountM,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    MLAS_FLOAT32X4 AlphaBroadcast,
    bool ZeroMode
    )
{
    do {

        const float* a = A;
        size_t k = CountK;

        MLAS_FLOAT32X4 AElements[RowCount];
        MLAS_FLOAT32X4 ABroadcast[RowCount] = { 0 };
        MLAS_FLOAT32X4 A2Broadcast[RowCount] = { 0 };
        MLAS_FLOAT32X4 acc[32] = { 0 };
        MLAS_FLOAT32X4 Accumulators[2][RowCount] = {{0}};

        //
        // Compute the output block.
        //
        while (k >= 4) {

            MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, a, lda);
            MlasSgemmComputeAElements<RowCount>(AElements, ABroadcast);
            if (CountM == 8) {
                MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, a + ( lda * 4), lda);
                MlasSgemmComputeAElements<RowCount>(AElements, A2Broadcast);
            }
            MlasSgemmComputeBlockZVECTOR<RowCount>(&acc[0], ABroadcast[0], A2Broadcast[0], B, CountM);
            MlasSgemmComputeBlockZVECTOR<RowCount>(&acc[0], ABroadcast[1], A2Broadcast[1], B+16, CountM);
            MlasSgemmComputeBlockZVECTOR<RowCount>(&acc[0], ABroadcast[2], A2Broadcast[2], B+32, CountM);
            MlasSgemmComputeBlockZVECTOR<RowCount>(&acc[0], ABroadcast[3], A2Broadcast[3], B+48, CountM);
            B += 16 * 4;
            a += 4;
            k -= 4;
        }

        while (k > 0) {
            MlasLoopUnroll<RowCount, MlasSgemmBroadcastAElementsZVECTOR>()(ABroadcast, a, lda);
            if (CountM == 8)  {
                MlasLoopUnroll<RowCount, MlasSgemmBroadcastAElementsZVECTOR>()(A2Broadcast, a + (lda * 4), lda);
            }
            MlasSgemmComputeBlockZVECTOR<RowCount>(&acc[0], ABroadcast[0], A2Broadcast[0], B, CountM);
            a += 1;
            B += 16;
            k -= 1;
        }
        if (CountN >= 16) {

            //
            // Store the entire output block.
            //
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc, C, ldc, AlphaBroadcast, ZeroMode);
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<4>>()(acc + 4, C, ldc, AlphaBroadcast, ZeroMode);
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<8>>()(acc + 8, C, ldc, AlphaBroadcast, ZeroMode);
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<12>>()(acc + 12, C, ldc, AlphaBroadcast, ZeroMode);
            if (CountM == 8) {
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc + 16, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<4>>()(acc + 20, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<8>>()(acc + 24, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<12>>()(acc + 28, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
            }
        } else {

            //
            // Store the partial output block.
            //

            if (CountN >= 12) {
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc, C, ldc, AlphaBroadcast, ZeroMode);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<4>>()(acc + 4, C, ldc, AlphaBroadcast, ZeroMode);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<8>>()(acc + 8, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc + 16, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<4>>()(acc + 20, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<8>>()(acc + 24, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 12 > 0) {
                        for (size_t i = 0; i < 4; ++i) {
                            Accumulators[1][i] = acc[i + 28];
                        }
                    }
                }
                if (CountN - 12 > 0) {
                    for (size_t i = 0; i < 4; ++i) {
                        Accumulators[0][i] = acc[i + 12];
                    }
                }
            } else if (CountN >= 8) {
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc, C, ldc, AlphaBroadcast, ZeroMode);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<4>>()(acc + 4, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc + 16, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<4>>()(acc + 20, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 8 > 0) {
                        for (size_t i = 0; i < 4; ++i) {
                            Accumulators[1][i] = acc[i + 24];
                        }
                    }
                }
                if (CountN - 8 > 0) {
                    for (size_t i = 0; i < 4; ++i) {
                        Accumulators[0][i] = acc[i + 8];
                    }
                }
            } else if (CountN >= 4) {
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorZVECTOR<0>>()(acc + 16, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 4 > 0) {
                        for (size_t i = 0; i < 4; ++i) {
                            Accumulators[1][i] = acc[i + 20];
                        }
                    }
                }
                if (CountN - 4 > 0) {
                    for (size_t i = 0; i < 4; ++i) {
                        Accumulators[0][i] = acc[i + 4];
                    }
                }
            } else {
                for (size_t i = 0; i < 4; ++i) {
                    Accumulators[0][i] = acc[i];
                }

                if (CountM == 8) {
                    for (size_t i = 0; i < 4; ++i) {
                        Accumulators[1][i] = acc[i + 16];
                    }
                }
           }

            //
            // Store the remaining unaligned columns.
            //

            C += (CountN & ~3);
            CountN &= 3;

            if (CountN > 0) {

                MlasLoopUnroll<RowCount, MlasSgemmMultiplyAlphaTrailingZVECTOR>()(Accumulators[0], AlphaBroadcast);
                MlasLoopUnroll<RowCount, MlasSgemmStoreScalarZVECTOR<0>>()(Accumulators[0], C, ldc, ZeroMode);
                if (CountM == 8) {
                    MlasLoopUnroll<RowCount, MlasSgemmMultiplyAlphaTrailingZVECTOR>()(Accumulators[1], AlphaBroadcast);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalarZVECTOR<0>>()(Accumulators[1], C + (ldc*4), ldc, ZeroMode);
                }
                if (CountN >= 2) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalarZVECTOR<1>>()(Accumulators[0], C, ldc, ZeroMode);
                    if (CountM == 8)  {
                        MlasLoopUnroll<RowCount, MlasSgemmStoreScalarZVECTOR<1>>()(Accumulators[1], C + (ldc*4), ldc, ZeroMode);
                    }
                }
                if (CountN >= 3) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalarZVECTOR<2>>()(Accumulators[0], C, ldc, ZeroMode);
                    if (CountM == 8)  {
                        MlasLoopUnroll<RowCount, MlasSgemmStoreScalarZVECTOR<2>>()(Accumulators[1], C + (ldc*4), ldc, ZeroMode);
                    }
                }
            }

            break;
        }

        C += 16;
        CountN -= 16;

    } while (CountN > 0);

    return CountM;
}

size_t
MLASCALL
MlasSgemmKernelZVECTOR(
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

    if (CountM >= 8) {
        RowsHandled = MlasSgemmZVECTORProcessCount<4>(A, B, C, 8 ,CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 4) {
        RowsHandled = MlasSgemmZVECTORProcessCount<4>(A, B, C, 4, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 2) {
        RowsHandled = MlasSgemmProcessCount<2>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else {
        RowsHandled = MlasSgemmProcessCount<1>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    }

    return RowsHandled;
}
