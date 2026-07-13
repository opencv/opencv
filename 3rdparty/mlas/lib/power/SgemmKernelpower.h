/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelpower.h

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#include "FgemmKernelpower.h"

template<size_t RowCount>
MLAS_FORCEINLINE
size_t
MlasSgemmProcessCount(
    const float* A,
    const float* B,
    float* C,
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

        MLAS_FLOAT32X4 Accumulators[RowCount][4];
        MLAS_FLOAT32X4 AElements[RowCount];
        MLAS_FLOAT32X4 ABroadcast[RowCount];

        //
        // Clear the block accumulators.
        //

        MlasLoopUnroll<RowCount, MlasFgemmZeroAccumulators>()(Accumulators);

        //
        // Compute the output block.
        //

        while (k >= 4) {

            MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, a, lda);

            MlasLoopUnroll<RowCount, MlasFgemmSplatAElements<0>>()(AElements, ABroadcast);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            MlasLoopUnroll<RowCount, MlasFgemmSplatAElements<1>>()(AElements, ABroadcast);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 16);

            MlasLoopUnroll<RowCount, MlasFgemmSplatAElements<2>>()(AElements, ABroadcast);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 32);

            MlasLoopUnroll<RowCount, MlasFgemmSplatAElements<3>>()(AElements, ABroadcast);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 48);

            a += 4;
            B += 16 * 4;
            k -= 4;
        }

        while (k > 0) {

            MlasLoopUnroll<RowCount, MlasFgemmBroadcastAElements>()(ABroadcast, a, lda);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            a += 1;
            B += 16;
            k -= 1;
        }

        if (CountN >= 16) {

            //
            // Store the entire output block.
            //

            MlasLoopUnroll<RowCount, MlasFgemmStoreVector<4>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);

        } else {

            //
            // Store the partial output block.
            //

            if (CountN >= 12) {
                MlasLoopUnroll<RowCount, MlasFgemmStoreVector<3>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 8) {
                MlasLoopUnroll<RowCount, MlasFgemmStoreVector<2>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 4) {
                MlasLoopUnroll<RowCount, MlasFgemmStoreVector<1>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            }

            //
            // Store the remaining unaligned columns.
            //

            C += (CountN & ~3);
            CountN &= 3;

            if (CountN > 0) {

                MlasLoopUnroll<RowCount, MlasFgemmMultiplyAlphaTrailing>()(Accumulators, AlphaBroadcast);

                MlasLoopUnroll<RowCount, MlasFgemmStoreScalar<0>>()(Accumulators, C, ldc, ZeroMode);

                if (CountN >= 2) {
                    MlasLoopUnroll<RowCount, MlasFgemmStoreScalar<1>>()(Accumulators, C, ldc, ZeroMode);
                }

                if (CountN >= 3) {
                    MlasLoopUnroll<RowCount, MlasFgemmStoreScalar<2>>()(Accumulators, C, ldc, ZeroMode);
                }
            }

            break;
        }

        C += 16;
        CountN -= 16;

    } while (CountN > 0);

    return RowCount;
}

