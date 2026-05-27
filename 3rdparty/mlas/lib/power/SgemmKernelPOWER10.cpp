/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelPower.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#define PREFETCH_ADDR(addr) \
    asm volatile("dcbt 0, %0" ::"r"(addr) : "memory");

#include "SgemmKernelpower.h"
extern "C" void
PackAKernelPOWER10(__vector float* D, const float* A, size_t lda, size_t k, size_t RowCount);
struct MlasSgemmBroadcastAElementsMMA
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
        ABroadcast[0] = vec_insert(A[Row * lda], ABroadcast[0], Row);
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
        __vector float a1,a2;
        a1 = vec_mergee (AElements[0], AElements[1]);
        a2 = vec_mergee (AElements[2], AElements[3]);
        ABroadcast[0] =vec_xxpermdi(a1,a2,0);
        ABroadcast[2] =vec_xxpermdi(a1,a2,3);
        a1 = vec_mergeo (AElements[0], AElements[1]);
        a2 = vec_mergeo (AElements[2], AElements[3]);
        ABroadcast[1] =vec_xxpermdi(a1,a2,0);
        ABroadcast[3] =vec_xxpermdi(a1,a2,3);
}
template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasSgemmComputeBlockMMA(
    __vector_quad acc[8],
    MLAS_FLOAT32X4 ABroadcast,
    MLAS_FLOAT32X4 A2Broadcast,
    const float* B,
    size_t CountM
    )
{
    MLAS_FLOAT32X4 BElements[4];
    typedef __vector unsigned char  vec_t;

    BElements[0] = MlasLoadFloat32x4(B);
    BElements[1] = MlasLoadFloat32x4(B + 4);
    BElements[2] = MlasLoadFloat32x4(B + 8);
    BElements[3] = MlasLoadFloat32x4(B + 12);
   __builtin_mma_xvf32gerpp (&acc[0], reinterpret_cast<vec_t>(ABroadcast), reinterpret_cast<vec_t>(BElements[0]));
   __builtin_mma_xvf32gerpp (&acc[1], reinterpret_cast<vec_t>(ABroadcast), reinterpret_cast<vec_t>(BElements[1]));
   __builtin_mma_xvf32gerpp (&acc[2], reinterpret_cast<vec_t>(ABroadcast), reinterpret_cast<vec_t>(BElements[2]));
   __builtin_mma_xvf32gerpp (&acc[3], reinterpret_cast<vec_t>(ABroadcast), reinterpret_cast<vec_t>(BElements[3]));
   if (CountM == 8) {
       __builtin_mma_xvf32gerpp (&acc[4], reinterpret_cast<vec_t>(A2Broadcast), reinterpret_cast<vec_t>(BElements[0]));
       __builtin_mma_xvf32gerpp (&acc[5], reinterpret_cast<vec_t>(A2Broadcast), reinterpret_cast<vec_t>(BElements[1]));
       __builtin_mma_xvf32gerpp (&acc[6], reinterpret_cast<vec_t>(A2Broadcast), reinterpret_cast<vec_t>(BElements[2]));
       __builtin_mma_xvf32gerpp (&acc[7], reinterpret_cast<vec_t>(A2Broadcast), reinterpret_cast<vec_t>(BElements[3]));
   }
}
template<size_t VectorCount>
struct MlasSgemmStoreVectorMMA
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

struct MlasSgemmMultiplyAlphaTrailingMMA
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
struct MlasSgemmStoreScalarMMA
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

template <size_t RowCount>
MLAS_FORCEINLINE
    size_t
    MlasSgemmMMAProcessCount(
        __vector float* Pa,
        const float* B,
        float* C,
        size_t CountM,
        size_t CountK,
        size_t CountN,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
    )
{
    do {
        __vector float* pa1 = Pa;
        size_t k = CountK;

        MLAS_FLOAT32X4 Accumulators[2][RowCount] = {{0}};
        MLAS_FLOAT32X4 Result[RowCount];
        MLAS_FLOAT32X4 ABroadcast[RowCount] = {0};
        __vector_quad acc[8];

        //
        // Clear the block accumulators.
        //
        __builtin_mma_xxsetaccz(&acc[0]);
        __builtin_mma_xxsetaccz(&acc[1]);
        __builtin_mma_xxsetaccz(&acc[2]);
        __builtin_mma_xxsetaccz(&acc[3]);
        __builtin_mma_xxsetaccz(&acc[4]);
        __builtin_mma_xxsetaccz(&acc[5]);
        __builtin_mma_xxsetaccz(&acc[6]);
        __builtin_mma_xxsetaccz(&acc[7]);

        //
        // Compute the output block.
        //
        while (k >= 8) {
            if (CountM == 8) {
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[0], pa1[4], B, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[1], pa1[5], B + 16, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[2], pa1[6], B + 32, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[3], pa1[7], B + 48, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[8], pa1[12], B + 64, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[9], pa1[13], B + 80, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[10], pa1[14], B + 96, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[11], pa1[15], B + 112, CountM);
                B += 128;
                pa1 += 16;
                k -= 8;
            } else {
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[0], ABroadcast[0], B, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[1], ABroadcast[1], B + 16, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[2], ABroadcast[2], B + 32, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[3], ABroadcast[3], B + 48, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[4], ABroadcast[0], B + 64, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[5], ABroadcast[1], B + 80, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[6], ABroadcast[2], B + 96, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[7], ABroadcast[3], B + 112, CountM);
                B += 128;
                pa1 += 8;
                k -= 8;
            }
        }

        while (k >= 4) {
            if (CountM == 8) {
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[0], pa1[4], B, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[1], pa1[5], B + 16, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[2], pa1[6], B + 32, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[3], pa1[7], B + 48, CountM);
                B += 16 * 4;
                pa1 += 8;
                k -= 4;
            } else {
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[0], ABroadcast[0], B, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[1], ABroadcast[1], B + 16, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[2], ABroadcast[2], B + 32, CountM);
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[3], ABroadcast[3], B + 48, CountM);
                B += 16 * 4;
                pa1 += 4;
                k -= 4;
            }
        }
        while (k > 0) {
            if (CountM == 8) {
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[0], pa1[1], B, CountM);
                pa1 += 2;
            } else {
                MlasSgemmComputeBlockMMA<RowCount>(&acc[0], pa1[0], ABroadcast[0], B, CountM);
                pa1 += 1;
            }
            B += 16;
            k -= 1;
        }
        if (CountN >= 16) {

            //
            // Store the entire output block.
            //
            __builtin_mma_disassemble_acc (Result, &acc[0]);
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            __builtin_mma_disassemble_acc (Result, &acc[1]);
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<4>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            __builtin_mma_disassemble_acc (Result, &acc[2]);
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<8>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            __builtin_mma_disassemble_acc (Result, &acc[3]);
            MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<12>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            if (CountM == 8) {
                __builtin_mma_disassemble_acc (Result, &acc[4]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[5]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<4>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[6]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<8>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[7]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<12>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
            }
        } else {

            //
            // Store the partial output block.
            //

            if (CountN >= 12) {
                __builtin_mma_disassemble_acc (Result, &acc[0]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[1]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<4>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[2]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<8>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Result, &acc[4]);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    __builtin_mma_disassemble_acc (Result, &acc[5]);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<4>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    __builtin_mma_disassemble_acc (Result, &acc[6]);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<8>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 12 > 0) {
                        __builtin_mma_disassemble_acc (Accumulators[1], &acc[7]);
                    }
                }
                if (CountN - 12 > 0) {
                    __builtin_mma_disassemble_acc (Accumulators[0], &acc[3]);
                }
            } else if (CountN >= 8) {
                __builtin_mma_disassemble_acc (Result, &acc[0]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[1]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<4>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Result, &acc[4]);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    __builtin_mma_disassemble_acc (Result, &acc[5]);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<4>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 8 > 0) {
                        __builtin_mma_disassemble_acc (Accumulators[1], &acc[6]);
                    }
                }
                if (CountN - 8 > 0) {
                    __builtin_mma_disassemble_acc (Accumulators[0], &acc[2]);
                }
            } else if (CountN >= 4) {
                __builtin_mma_disassemble_acc (Result, &acc[0]);
                MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Result, &acc[4]);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 4 > 0) {
                        __builtin_mma_disassemble_acc (Accumulators[1], &acc[5]);
                    }
                }
                if (CountN - 4 > 0) {
                    __builtin_mma_disassemble_acc (Accumulators[0], &acc[1]);
                }
            } else {
                __builtin_mma_disassemble_acc (Accumulators[0], &acc[0]);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Accumulators[1], &acc[4]);
                }
           }

            //
            // Store the remaining unaligned columns.
            //

            C += (CountN & ~3);
            CountN &= 3;

            if (CountN > 0) {

                MlasLoopUnroll<RowCount, MlasSgemmMultiplyAlphaTrailingMMA>()(Accumulators[0], AlphaBroadcast);
                MlasLoopUnroll<RowCount, MlasSgemmStoreScalarMMA<0>>()(Accumulators[0], C, ldc, ZeroMode);
                if (CountM == 8) {
                    MlasLoopUnroll<RowCount, MlasSgemmMultiplyAlphaTrailingMMA>()(Accumulators[1], AlphaBroadcast);
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalarMMA<0>>()(Accumulators[1], C + (ldc*4), ldc, ZeroMode);
                }
                if (CountN >= 2) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalarMMA<1>>()(Accumulators[0], C, ldc, ZeroMode);
                    if (CountM == 8)  {
                        MlasLoopUnroll<RowCount, MlasSgemmStoreScalarMMA<1>>()(Accumulators[1], C + (ldc*4), ldc, ZeroMode);
                    }
                }
                if (CountN >= 3) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalarMMA<2>>()(Accumulators[0], C, ldc, ZeroMode);
                    if (CountM == 8)  {
                        MlasLoopUnroll<RowCount, MlasSgemmStoreScalarMMA<2>>()(Accumulators[1], C + (ldc*4), ldc, ZeroMode);
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

template <size_t RowCount>
MLAS_FORCEINLINE void
MlasSgemmPackA(
    __vector float* D,
    const float* A,
    size_t lda,
    size_t k
)
{
    __vector float a1, a2;
    const float* a = A;
    MLAS_FLOAT32X4 AElements[RowCount] = {};
    MLAS_FLOAT32X4 A2Elements[RowCount] = {};
    while (k >= 16)

    {
        PREFETCH_ADDR(a);
        PREFETCH_ADDR(a + lda);
        PREFETCH_ADDR(a + 2 * lda);
        PREFETCH_ADDR(a + 3 * lda);
        PREFETCH_ADDR(a + 4 * lda);
        PREFETCH_ADDR(a + 5 * lda);
        PREFETCH_ADDR(a + 6 * lda);
        PREFETCH_ADDR(a + 7 * lda);
        MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a, lda);
        a1 = vec_mergee(AElements[0], AElements[1]);
        a2 = vec_mergee(AElements[2], AElements[3]);
        D[0] = vec_xxpermdi(a1, a2, 0);
        D[2] = vec_xxpermdi(a1, a2, 3);
        a1 = vec_mergeo(AElements[0], AElements[1]);
        a2 = vec_mergeo(AElements[2], AElements[3]);
        D[1] = vec_xxpermdi(a1, a2, 0);
        D[3] = vec_xxpermdi(a1, a2, 3);
        if (RowCount == 8) {
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 4, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[8] = vec_xxpermdi(a1, a2, 0);
            D[10] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[9] = vec_xxpermdi(a1, a2, 0);
            D[11] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 8, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[16] = vec_xxpermdi(a1, a2, 0);
            D[18] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[17] = vec_xxpermdi(a1, a2, 0);
            D[19] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 12, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[24] = vec_xxpermdi(a1, a2, 0);
            D[26] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[25] = vec_xxpermdi(a1, a2, 0);
            D[27] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(A2Elements, a + (lda * 4), lda);
            a1 = vec_mergee(A2Elements[0], A2Elements[1]);
            a2 = vec_mergee(A2Elements[2], A2Elements[3]);
            D[4] = vec_xxpermdi(a1, a2, 0);
            D[6] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(A2Elements[0], A2Elements[1]);
            a2 = vec_mergeo(A2Elements[2], A2Elements[3]);
            D[5] = vec_xxpermdi(a1, a2, 0);
            D[7] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(A2Elements, (a + 4) + (lda * 4), lda);
            a1 = vec_mergee(A2Elements[0], A2Elements[1]);
            a2 = vec_mergee(A2Elements[2], A2Elements[3]);
            D[12] = vec_xxpermdi(a1, a2, 0);
            D[14] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(A2Elements[0], A2Elements[1]);
            a2 = vec_mergeo(A2Elements[2], A2Elements[3]);
            D[13] = vec_xxpermdi(a1, a2, 0);
            D[15] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(A2Elements, (a + 8) + (lda * 4), lda);
            a1 = vec_mergee(A2Elements[0], A2Elements[1]);
            a2 = vec_mergee(A2Elements[2], A2Elements[3]);
            D[20] = vec_xxpermdi(a1, a2, 0);
            D[22] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(A2Elements[0], A2Elements[1]);
            a2 = vec_mergeo(A2Elements[2], A2Elements[3]);
            D[21] = vec_xxpermdi(a1, a2, 0);
            D[23] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(A2Elements, (a + 12) + (lda * 4), lda);
            a1 = vec_mergee(A2Elements[0], A2Elements[1]);
            a2 = vec_mergee(A2Elements[2], A2Elements[3]);
            D[28] = vec_xxpermdi(a1, a2, 0);
            D[30] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(A2Elements[0], A2Elements[1]);
            a2 = vec_mergeo(A2Elements[2], A2Elements[3]);
            D[29] = vec_xxpermdi(a1, a2, 0);
            D[31] = vec_xxpermdi(a1, a2, 3);
            D += 32;
        } else {
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 4, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[4] = vec_xxpermdi(a1, a2, 0);
            D[6] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[5] = vec_xxpermdi(a1, a2, 0);
            D[7] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 8, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[8] = vec_xxpermdi(a1, a2, 0);
            D[10] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[9] = vec_xxpermdi(a1, a2, 0);
            D[11] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 12, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[12] = vec_xxpermdi(a1, a2, 0);
            D[14] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[13] = vec_xxpermdi(a1, a2, 0);
            D[15] = vec_xxpermdi(a1, a2, 3);
            D += 16;
        }
        k -= 16;
        a += 16;
    }

    while (k >= 8) {
        MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a, lda);
        a1 = vec_mergee(AElements[0], AElements[1]);
        a2 = vec_mergee(AElements[2], AElements[3]);
        D[0] = vec_xxpermdi(a1, a2, 0);
        D[2] = vec_xxpermdi(a1, a2, 3);
        a1 = vec_mergeo(AElements[0], AElements[1]);
        a2 = vec_mergeo(AElements[2], AElements[3]);
        D[1] = vec_xxpermdi(a1, a2, 0);
        D[3] = vec_xxpermdi(a1, a2, 3);
        if (RowCount == 8) {
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 4, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[8] = vec_xxpermdi(a1, a2, 0);
            D[10] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[9] = vec_xxpermdi(a1, a2, 0);
            D[11] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(A2Elements, a + (lda * 4), lda);
            a1 = vec_mergee(A2Elements[0], A2Elements[1]);
            a2 = vec_mergee(A2Elements[2], A2Elements[3]);
            D[4] = vec_xxpermdi(a1, a2, 0);
            D[6] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(A2Elements[0], A2Elements[1]);
            a2 = vec_mergeo(A2Elements[2], A2Elements[3]);
            D[5] = vec_xxpermdi(a1, a2, 0);
            D[7] = vec_xxpermdi(a1, a2, 3);
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(A2Elements, (a + 4) + (lda * 4), lda);
            a1 = vec_mergee(A2Elements[0], A2Elements[1]);
            a2 = vec_mergee(A2Elements[2], A2Elements[3]);
            D[12] = vec_xxpermdi(a1, a2, 0);
            D[14] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(A2Elements[0], A2Elements[1]);
            a2 = vec_mergeo(A2Elements[2], A2Elements[3]);
            D[13] = vec_xxpermdi(a1, a2, 0);
            D[15] = vec_xxpermdi(a1, a2, 3);
            D += 16;
        } else {
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a + 4, lda);
            a1 = vec_mergee(AElements[0], AElements[1]);
            a2 = vec_mergee(AElements[2], AElements[3]);
            D[4] = vec_xxpermdi(a1, a2, 0);
            D[6] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(AElements[0], AElements[1]);
            a2 = vec_mergeo(AElements[2], AElements[3]);
            D[5] = vec_xxpermdi(a1, a2, 0);
            D[7] = vec_xxpermdi(a1, a2, 3);
            D += 8;
        }
        a += 8;
        k -= 8;
    }

    while (k >= 4) {
        MlasLoopUnroll<4, MlasFgemmLoadAElements>()(AElements, a, lda);
        a1 = vec_mergee(AElements[0], AElements[1]);
        a2 = vec_mergee(AElements[2], AElements[3]);
        D[0] = vec_xxpermdi(a1, a2, 0);
        D[2] = vec_xxpermdi(a1, a2, 3);
        a1 = vec_mergeo(AElements[0], AElements[1]);
        a2 = vec_mergeo(AElements[2], AElements[3]);
        D[1] = vec_xxpermdi(a1, a2, 0);
        D[3] = vec_xxpermdi(a1, a2, 3);
        if (RowCount == 8) {
            MlasLoopUnroll<4, MlasFgemmLoadAElements>()(A2Elements, a + (lda * 4), lda);
            a1 = vec_mergee(A2Elements[0], A2Elements[1]);
            a2 = vec_mergee(A2Elements[2], A2Elements[3]);
            D[4] = vec_xxpermdi(a1, a2, 0);
            D[6] = vec_xxpermdi(a1, a2, 3);
            a1 = vec_mergeo(A2Elements[0], A2Elements[1]);
            a2 = vec_mergeo(A2Elements[2], A2Elements[3]);
            D[5] = vec_xxpermdi(a1, a2, 0);
            D[7] = vec_xxpermdi(a1, a2, 3);
            D += 8;
        } else
            D += 4;
        a += 4;
        k -= 4;
    }

    /* When k is less than 4, copy a single element from each row. */
    while (k > 0) {
        MlasLoopUnroll<4, MlasSgemmBroadcastAElementsMMA>()(AElements, a, lda);
        D[0] = AElements[0];
        if (RowCount == 8) {
            MlasLoopUnroll<4, MlasSgemmBroadcastAElementsMMA>()(A2Elements, a + (lda * 4), lda);
            D[1] = A2Elements[0];
            D += 2;
        } else {
            D += 1;
        }
        a += 1;
        k -= 1;
    }
}

size_t
MLASCALL
MlasSgemmKernelPOWER10(
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
    size_t index = CountK * 2;

    MLAS_FLOAT32X4* PackA =
        reinterpret_cast<MLAS_FLOAT32X4*>(alloca(sizeof(MLAS_FLOAT32X4) * index));
    MLAS_FLOAT32X4 AlphaBroadcast = MlasBroadcastFloat32x4(alpha);

    if (CountM >= 8) {
#ifdef _AIX
        MlasSgemmPackA<8>(PackA, A, lda, CountK);
#else
        if (CountK >= 16 && !(CountK % 16)) {
            PackAKernelPOWER10(PackA, A, lda, CountK, 8);
        } else {
            MlasSgemmPackA<8>(PackA, A, lda, CountK);
        }
#endif

        RowsHandled = MlasSgemmMMAProcessCount<4>(PackA, B, C, 8, CountK, CountN, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 4) {
        memset(PackA + CountK, 0, sizeof(MLAS_FLOAT32X4) * CountK);
#ifdef _AIX
        MlasSgemmPackA<4>(PackA, A, lda, CountK);
#else
        if (CountK >= 16 && !(CountK % 16)) {
            PackAKernelPOWER10(PackA, A, lda, CountK, 4);
        } else {
            MlasSgemmPackA<4>(PackA, A, lda, CountK);
        }
#endif
        RowsHandled = MlasSgemmMMAProcessCount<4>(PackA, B, C, 4, CountK, CountN, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 2) {
        RowsHandled = MlasSgemmProcessCount<2>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else {
        RowsHandled = MlasSgemmProcessCount<1>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    }
    return RowsHandled;
}
