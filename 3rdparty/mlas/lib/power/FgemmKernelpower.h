/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelPower.h

Abstract:

    This module implements the kernels for the single/double precision matrix/matrix
    multiply operation (DGEMM/SGEMM).

--*/

#include "mlasi.h"
#if defined(SINGLE)
#define MLAS_FLOATTYPE MLAS_FLOAT32X4
#define MLAS_GEMMTYPE float 
#define MLAS_LOAD_FLOAT MlasLoadFloat32x4
#define MLAS_ZERO_FLOAT MlasZeroFloat32x4
#define MLAS_STORE_FLOAT MlasStoreFloat32x4
#define MLAS_EXTRACT_FLOAT MlasExtractLaneFloat32x4
#define MLAS_MUL_FLOAT MlasMultiplyFloat32x4
#define MLAS_MULADD_FLOAT MlasMultiplyAddFloat32x4
#define MLAS_BROADCAST_FLOAT MlasBroadcastFloat32x4
#else
#define MLAS_FLOATTYPE MLAS_FLOAT64X2
#define MLAS_GEMMTYPE double
#define MLAS_LOAD_FLOAT MlasLoadFloat64x2
#define MLAS_ZERO_FLOAT MlasZeroFloat64x2
#define MLAS_STORE_FLOAT MlasStoreFloat64x2
#define MLAS_EXTRACT_FLOAT MlasExtractLaneFloat64x2
#define MLAS_MUL_FLOAT MlasMultiplyFloat64x2
#define MLAS_MULADD_FLOAT MlasMultiplyAddFloat64x2
#define MLAS_BROADCAST_FLOAT MlasBroadcastFloat64x2
#endif
//
// Templates to ensure that a loop is unrolled.
//

template<size_t Count, size_t Index>
struct MlasLoopUnrollStep
{
    template<typename IterationType, typename... IterationArgs>
    MLAS_FORCEINLINE
    static
    void
    Step(
        IterationArgs&&... Arguments
        )
    {
        IterationType::template Iteration<Count, Index>(Arguments...);
        MlasLoopUnrollStep<Count, Index + 1>::template Step<IterationType>(Arguments...);
    }
};

template<size_t Count>
struct MlasLoopUnrollStep<Count, Count>
{
    template<typename IterationType, typename... IterationArgs>
    MLAS_FORCEINLINE
    static
    void
    Step(
        IterationArgs&&...
        )
    {
        // Terminate the loop.
    }
};

template<size_t Count, typename IteratorType>
struct MlasLoopUnroll
{
    template<typename... IterationArgs>
    MLAS_FORCEINLINE
    void
    operator()(
        IterationArgs&&... Arguments
        )
    {
        MlasLoopUnrollStep<Count, 0>::template Step<IteratorType>(Arguments...);
    }
};

//
// Templates used with loop unrolling to perform an action on one row of the
// output.
//

struct MlasFgemmZeroAccumulators
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[RowCount][4]
        )
    {
        Accumulators[Row][0] = MLAS_ZERO_FLOAT();
        Accumulators[Row][1] = MLAS_ZERO_FLOAT();
        Accumulators[Row][2] = MLAS_ZERO_FLOAT();
        Accumulators[Row][3] = MLAS_ZERO_FLOAT();
    }
};

struct MlasFgemmLoadAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE AElements[RowCount],
        const MLAS_GEMMTYPE* A,
        size_t lda
        )
    {
        AElements[Row] = MLAS_LOAD_FLOAT(A + Row * lda);
    }
};

struct MlasFgemmBroadcastAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE ABroadcast[RowCount],
        const MLAS_GEMMTYPE* A,
        size_t lda
        )
    {
        ABroadcast[Row] = MLAS_BROADCAST_FLOAT(A + Row * lda);
    }
};

template<unsigned Lane>
struct MlasFgemmSplatAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE AElements[RowCount],
        MLAS_FLOATTYPE ABroadcast[RowCount]
        )
    {
        ABroadcast[Row] = vec_splat(AElements[Row], Lane);
    }
};

struct MlasFgemmMultiplyAddRow
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[RowCount][4],
        MLAS_FLOATTYPE ABroadcast[RowCount],
        MLAS_FLOATTYPE BElements[4]
        )
    {
        Accumulators[Row][0] = MLAS_MULADD_FLOAT(ABroadcast[Row], BElements[0], Accumulators[Row][0]);
        Accumulators[Row][1] = MLAS_MULADD_FLOAT(ABroadcast[Row], BElements[1], Accumulators[Row][1]);
        Accumulators[Row][2] = MLAS_MULADD_FLOAT(ABroadcast[Row], BElements[2], Accumulators[Row][2]);
        Accumulators[Row][3] = MLAS_MULADD_FLOAT(ABroadcast[Row], BElements[3], Accumulators[Row][3]);
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasFgemmComputeBlock(
    MLAS_FLOATTYPE Accumulators[RowCount][4],
    MLAS_FLOATTYPE ABroadcast[RowCount],
    const MLAS_GEMMTYPE* B
    )
{
    MLAS_FLOATTYPE BElements[4];
#if defined(SINGLE)
    BElements[0] = MLAS_LOAD_FLOAT(B);
    BElements[1] = MLAS_LOAD_FLOAT(B + 4);
    BElements[2] = MLAS_LOAD_FLOAT(B + 8);
    BElements[3] = MLAS_LOAD_FLOAT(B + 12);
#else
    BElements[0] = MLAS_LOAD_FLOAT(B);
    BElements[1] = MLAS_LOAD_FLOAT(B + 2);
    BElements[2] = MLAS_LOAD_FLOAT(B + 4);
    BElements[3] = MLAS_LOAD_FLOAT(B + 6);
#endif

    MlasLoopUnroll<RowCount, MlasFgemmMultiplyAddRow>()(Accumulators, ABroadcast, BElements);
}

struct MlasFgemmMultiplyAlphaRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[4],
        MLAS_FLOATTYPE AlphaBroadcast
        )
    {
        Accumulators[Index] = MLAS_MUL_FLOAT(Accumulators[Index], AlphaBroadcast);
    }
};

struct MlasFgemmMultiplyAlphaAddRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[4],
        MLAS_FLOATTYPE AlphaBroadcast,
        const MLAS_GEMMTYPE* C
        )
    {
#if defined(SINGLE)
        Accumulators[Index] = MLAS_MULADD_FLOAT(Accumulators[Index],
            AlphaBroadcast, MLAS_LOAD_FLOAT(C + Index * 4));
#else
        Accumulators[Index] = MLAS_MULADD_FLOAT(Accumulators[Index],
            AlphaBroadcast, MLAS_LOAD_FLOAT(C + Index * 2));
#endif
    }
};

struct MlasFgemmStoreRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[4],
        MLAS_GEMMTYPE* C
        )
    {
#if defined(SINGLE)
        MLAS_STORE_FLOAT(C + Index * 4, Accumulators[Index]);
#else
        MLAS_STORE_FLOAT(C + Index * 2, Accumulators[Index]);
#endif
    }
};

template<size_t VectorCount>
struct MlasFgemmStoreVector
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[RowCount][4],
        MLAS_GEMMTYPE* C,
        size_t ldc,
        MLAS_FLOATTYPE AlphaBroadcast,
        bool ZeroMode
        )
    {
        MLAS_GEMMTYPE* c = C + Row * ldc;

        if (ZeroMode) {
            MlasLoopUnroll<VectorCount, MlasFgemmMultiplyAlphaRow>()(Accumulators[Row], AlphaBroadcast);
        } else {
            MlasLoopUnroll<VectorCount, MlasFgemmMultiplyAlphaAddRow>()(Accumulators[Row], AlphaBroadcast, c);
        }

        MlasLoopUnroll<VectorCount, MlasFgemmStoreRow>()(Accumulators[Row], c);

        //
        // Shift down any unaligned elements to the bottom for further processing.
        //

        if (VectorCount < 4) {
            Accumulators[Row][0] = Accumulators[Row][VectorCount];
        }
    }
};

struct MlasFgemmMultiplyAlphaTrailing
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[RowCount][4],
        MLAS_FLOATTYPE AlphaBroadcast
        )
    {
        Accumulators[Row][0] = MLAS_MUL_FLOAT(Accumulators[Row][0], AlphaBroadcast);
    }
};

template<unsigned Lane>
struct MlasFgemmStoreScalar
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOATTYPE Accumulators[RowCount][4],
        MLAS_GEMMTYPE* C,
        size_t ldc,
        bool ZeroMode
        )
    {
        MLAS_GEMMTYPE* c = C + Row * ldc + Lane;
        MLAS_GEMMTYPE Value = MLAS_EXTRACT_FLOAT<Lane>(Accumulators[Row][0]);

        if (!ZeroMode) {
            Value += *c;
        }

        *c = Value;
    }
};

