/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.h

Abstract:

    This module defines the set of template functions to implement a kernel of
    quantized integer matrix/matrix multiply operation (QGEMM).

    To implement a new kernel, template functions below need to be specialized:
        MlasGemmQuantFixupZeroPointA
        MlasGemmQuantFixupZeroPointB
        MlasGemmQuantCopyPackA
        MlasGemmQuantCopyPackB
        MlasGemmQuantKernel
    Specialization of MlasGemmQuantTryGemvKernel is optional.

    MlasGemmQuantOperation and MlasGemmQuantPackedOperation are shared kernel drivers.
    MlasGemmQuantScaleSumBuffer is a helper function.

    It also includes the dispatcher logics.

--*/

#pragma once

#include "mlasi.h"

#include <sstream>
#include <string>
#include <cstdlib>

//
// Define the default striding parameters used for the quantized integer
// matrix/matrix multiply operation.
//

struct MLAS_GEMM_QUANT_STRIDES {
    size_t M;
    size_t N;
    size_t K;
};

template<typename KernelType>
MLAS_FORCEINLINE
bool
MlasGemmQuantTryGemvKernel(
    const uint8_t* A,
    const uint8_t* B,
    size_t ldb,
    int32_t* C,
    size_t CountK,
    size_t CountN,
    bool AIsSigned,
    bool BIsSigned
)
{
    MLAS_UNREFERENCED_PARAMETER(A);
    MLAS_UNREFERENCED_PARAMETER(B);
    MLAS_UNREFERENCED_PARAMETER(ldb);
    MLAS_UNREFERENCED_PARAMETER(C);
    MLAS_UNREFERENCED_PARAMETER(CountK);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    return false;
}

template <typename KernelType>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA(
    int32_t ZeroPointA,
    bool AIsSigned)
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    return ZeroPointA;
}

template<typename KernelType>
int32_t constexpr
MlasGemmQuantFixupZeroPointB(
    int32_t ZeroPointB,
    bool BIsSigned
)
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    return ZeroPointB;
}

template<typename KernelType>
MLAS_FORCEINLINE
void
MlasGemmQuantFixupZeroPointB(
    const uint8_t* PackedZeroPointB,
    int32_t* ZeroPointBBuffer,
    size_t N,
    bool BIsSigned
)
{
    int32_t ZeroPointB;

    for (size_t n = 0; n < N; n++) {

        ZeroPointB = typename KernelType::OffsetBType(PackedZeroPointB[n]);
        ZeroPointB = MlasGemmQuantFixupZeroPointB<KernelType>(ZeroPointB, BIsSigned);

        ZeroPointBBuffer[n] = -ZeroPointB;
    }

    //
    // Fill the misaligned slots of the zero point buffer with zeros to guard
    // against tools that check for uninitialized data usage.
    //

    size_t AlignedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    for (size_t n = N; n < AlignedN; n++) {
        ZeroPointBBuffer[n] = 0;
    }
}

template<typename KernelType>
void
MlasGemmQuantCopyPackA(
    typename KernelType::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
);

template<typename KernelType>
void
MlasGemmQuantCopyPackB(
    typename KernelType::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
);

template<typename KernelType>
size_t
MlasGemmQuantKernel(
    const typename KernelType::PackedAType* A,
    const typename KernelType::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
);

/**
 * @brief Usually a wrapper of assembly/intrinsic kernel
 *        of symmetric quant gemm
 * @tparam KernelType
 * @param A                   Left hand side matrix
 * @param B                   Prepacked right hand side matrix
 * @param C                   Result matrix
 * @param PackedCountK        Number of packed rows from B
 * @param CountM              Number of rows to process
 * @param CountN              Number of columns to process
 * @param ldc                 Row stride of C
 * @param lda                 Row stride of A
 * @param ColumnSumVector     Column sum of B scaled by zero point A
 * @return                    Number of rows processed
*/
template<typename KernelType>
size_t
MlasSymmQGemmKernel(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    size_t lda,
    const int32_t* ColumnSumVector
);

inline
void
MlasGemmQuantScaleSumBuffer(
    int32_t* Output,
    const int32_t* Input,
    size_t N,
    int32_t Scale
)
{
    for (size_t n = 0; n < N; n++) {
        Output[n] = Input[n] * Scale;
    }
}


MLAS_FORCEINLINE
void
MlasGemmQuantScaleSumBuffer(
    int32_t* SumBuffer,
    size_t N,
    int32_t Scale
)
{
    return MlasGemmQuantScaleSumBuffer(SumBuffer, SumBuffer, N, Scale);
}

template<typename KernelType>
MLAS_FORCEINLINE
void
MlasGemmQuantThreadInit()
{
    constexpr MLAS_GEMM_QUANT_STRIDES Strides = KernelType::Strides;
    constexpr size_t packASize =
        UpAlignSize(Strides.M * Strides.K * sizeof(typename KernelType::PackedAType));
    constexpr size_t packBSize =
        UpAlignSize(Strides.N * Strides.K * sizeof(typename KernelType::PackedBType));
    constexpr size_t rowSumSize = UpAlignSize(Strides.M * sizeof(int32_t));
    constexpr size_t colSumSize = UpAlignSize(Strides.N * sizeof(int32_t));
    constexpr size_t zpbSize = UpAlignSize(Strides.N * sizeof(int32_t));

    constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides = KernelType::PackedStrides;
    constexpr size_t packedASize =
        UpAlignSize(PackedStrides.M * PackedStrides.K * sizeof(typename KernelType::PackedAType));

    constexpr size_t bufsize = std::max(packASize + packBSize, packedASize) + rowSumSize + colSumSize + zpbSize;

    MlasThreadedBufAlloc(bufsize);
}

template<typename KernelType>
void
MlasGemmQuantOperation(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    RangeStartM - Supplies the starting row index to output.

    RangeCountM - Supplies the number of rows to output.

    RangeStartN - Supplies the starting column index to output.

    RangeCountN - Supplies the number of columns to output.

Return Value:

    None.

--*/
{
    constexpr MLAS_GEMM_QUANT_STRIDES Strides = KernelType::Strides;
    constexpr size_t packASize =
        UpAlignSize(Strides.M * Strides.K * sizeof(typename KernelType::PackedAType));
    constexpr size_t packBSize =
        UpAlignSize(Strides.N * Strides.K * sizeof(typename KernelType::PackedBType));
    constexpr size_t rowSumSize = UpAlignSize(Strides.M * sizeof(int32_t));
    constexpr size_t colSumSize = UpAlignSize(Strides.N * sizeof(int32_t));

    MlasGemmQuantThreadInit<KernelType>();

    uint8_t* p = ThreadedBufHolder.get();
    typename KernelType::PackedAType* PanelA =
        reinterpret_cast<typename KernelType::PackedAType*>(p);
    p += packASize;
    typename KernelType::PackedBType* PanelB =
        reinterpret_cast<typename KernelType::PackedBType*>(p);
    p += packBSize;
    int32_t* RowSumBuffer = reinterpret_cast<int32_t*>(p);
    p += rowSumSize;
    int32_t* ColumnSumBuffer = reinterpret_cast<int32_t*>(p);
    p += colSumSize;
    int32_t* ZeroPointBBuffer = reinterpret_cast<int32_t*>(p);


    const size_t K = Shape->K;

    const size_t lda = Data->lda;
    const size_t ldb = Data->ldb;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* B = (const uint8_t*)Data->B + RangeStartN;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ?
        Data->ZeroPointB + RangeStartN : nullptr;
    bool IsAccumulateMode = Shape->IsAccumulateMode;

    int32_t ZeroPointA = typename KernelType::OffsetAType(Data->ZeroPointA);
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Try to use a GEMV kernel if supported by this kernel type.
    //

    if ((RangeCountM == 1) &&
        (ZeroPointA == 0) && (PackedZeroPointB == nullptr) && (ZeroPointB == 0) &&
        (Data->OutputProcessor == nullptr)) {
        if (MlasGemmQuantTryGemvKernel<KernelType>(A, B, ldb, C, K, RangeCountN, Shape->AIsSigned, Shape->BIsSigned)) {
            return;
        }
    }

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix A if the
    // kernel requires opposite-signed data.
    //

    ZeroPointA = MlasGemmQuantFixupZeroPointA<KernelType>(ZeroPointA, Shape->AIsSigned);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix B if the
    // data is the opposite format of the kernel implementation. This value is
    // ignored if per-column zero point offsets are used instead.
    //

    ZeroPointB = MlasGemmQuantFixupZeroPointB<KernelType>(ZeroPointB, Shape->BIsSigned);

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {

            CountN = std::min(RangeCountN - n, Strides.N);

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmQuantFixupZeroPointB<KernelType>(
                    PackedZeroPointB + n,
                    ZeroPointBBuffer,
                    CountN,
                    Shape->BIsSigned);
            }

            //
            // Copy a panel of matrix B to a local packed buffer.
            //

            MlasGemmQuantCopyPackB<KernelType>(
                PanelB,
                B + n,
                ldb,
                CountN,
                CountK,
                ColumnSumBuffer,
                Shape->BIsSigned);

            MlasGemmQuantScaleSumBuffer(ColumnSumBuffer, CountN, -ZeroPointA);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {

                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmQuantCopyPackA<KernelType>(
                    PanelA,
                    A + m * lda,
                    lda,
                    CountM,
                    CountK,
                    RowSumBuffer,
                    Shape->AIsSigned);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmQuantScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0) && !IsAccumulateMode;
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled = MlasGemmQuantKernel<KernelType>(
                        pa,
                        PanelB,
                        c,
                        PackedCountK,
                        RowsRemaining,
                        CountN,
                        ldc,
                        RowSums,
                        ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr,
                        ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        B += CountK * ldb;
    }
}


template<typename KernelType>
void
MlasGemmQuantPackedOperation(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    Shape - Supplies the structure containing the GEMM input and output shapes.

    Data  - Supplies the structure containing the GEMM input and output data layout

    RangeStartM - Supplies the starting row index to output.

    RangeCountM - Supplies the number of rows to output.

    RangeStartN - Supplies the starting column index to output.

    RangeCountN - Supplies the number of columns to output.

Return Value:

    None.

--*/
{
    constexpr MLAS_GEMM_QUANT_STRIDES Strides = KernelType::PackedStrides;
    constexpr size_t packASize =
        UpAlignSize(Strides.M * Strides.K * sizeof(typename KernelType::PackedAType));
    constexpr size_t rowSumSize = UpAlignSize(Strides.M * sizeof(int32_t));
    constexpr size_t colSumSize = UpAlignSize(Strides.N * sizeof(int32_t));

    MlasGemmQuantThreadInit<KernelType>();

    uint8_t* p = ThreadedBufHolder.get();
    typename KernelType::PackedAType* PanelA =
        reinterpret_cast<typename KernelType::PackedAType*>(p);
    p += packASize;
    int32_t* RowSumBuffer = reinterpret_cast<int32_t*>(p);
    p += rowSumSize;
    int32_t* ColumnSumBuffer = reinterpret_cast<int32_t*>(p);
    p += colSumSize;
    int32_t* ZeroPointBBuffer = reinterpret_cast<int32_t*>(p);

    const size_t K = Shape->K;

    const size_t lda = Data->lda;
    const size_t ldc = Data->ldc;

    const uint8_t* A = Data->A + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)Data->B;
    int32_t* C = Data->C + RangeStartM * ldc + RangeStartN;
    const uint8_t* PackedZeroPointB = Data->PerColumnZeroPoints ?
        Data->ZeroPointB + RangeStartN : nullptr;
    bool IsAccumulateMode = Shape->IsAccumulateMode;

    int32_t ZeroPointA = typename KernelType::OffsetAType(Data->ZeroPointA);
    int32_t ZeroPointB = typename KernelType::OffsetBType(*Data->ZeroPointB);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix A if the
    // kernel requires signed data.
    //

    ZeroPointA = MlasGemmQuantFixupZeroPointA<KernelType>(ZeroPointA, Shape->AIsSigned);

    //
    // Fixup the sign bit of the per-matrix zero point offset of matrix B if the
    // data is the opposite format of the kernel implementation. This value is
    // ignored if per-column zero point offsets are used instead.
    //

    ZeroPointB = MlasGemmQuantFixupZeroPointB<KernelType>(ZeroPointB, Shape->BIsSigned);

    //
    // Extract the pointer to the column sum buffer from the packed matrix.
    //

    const size_t AlignedN =
        (Shape->N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const int32_t* PackedColumnSumBuffer = (const int32_t*)PackedB;
    PackedB = (const uint8_t*)(PackedColumnSumBuffer + AlignedN);
    PackedColumnSumBuffer += RangeStartN;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        if (k > 0) {
            std::fill_n(ColumnSumBuffer, Strides.N, 0);
        }

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {

            CountN = std::min(RangeCountN - n, Strides.N);

            if (k == 0) {
                MlasGemmQuantScaleSumBuffer(ColumnSumBuffer, PackedColumnSumBuffer + n,
                    CountN, -ZeroPointA);
            }

            //
            // Fixup the sign bit of the per-column zero point offsets of matrix B
            // if the data is the opposite format of the kernel implementation.
            //

            if (PackedZeroPointB != nullptr) {
                MlasGemmQuantFixupZeroPointB<KernelType>(
                    PackedZeroPointB + n,
                    ZeroPointBBuffer,
                    CountN,
                    Shape->BIsSigned);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const uint8_t* b = PackedB + (RangeStartN + n) *
                KernelType::PackedK * PackedCountK;
            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {

                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                MlasGemmQuantCopyPackA<KernelType>(
                    PanelA,
                    A + m * lda,
                    lda,
                    CountM,
                    CountK,
                    RowSumBuffer,
                    Shape->AIsSigned);

                //
                // Apply the global depth value constant without the ZeroPointB scaling from:
                //
                //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
                //              ==>
                //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
                //
                // The ZeroPointB term is factored out and either applied below for per-matrix
                // quantization or inside the kernel for per-column quantization.
                //

                for (size_t mm = 0; mm < CountM; mm++) {
                    RowSumBuffer[mm] -= int32_t(CountK) * ZeroPointA;
                }

                //
                // Scale the row sums by the per-matrix zero point offset of matrix B.
                //

                if (PackedZeroPointB == nullptr) {
                    MlasGemmQuantScaleSumBuffer(RowSumBuffer, CountM, -ZeroPointB);
                }

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0) && !IsAccumulateMode;
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled = MlasGemmQuantKernel<KernelType>(
                        pa,
                        b,
                        c,
                        PackedCountK,
                        RowsRemaining,
                        CountN,
                        ldc,
                        RowSums,
                        ColumnSumBuffer,
                        (PackedZeroPointB != nullptr) ? ZeroPointBBuffer : nullptr,
                        ZeroMode);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        PackedB = (const uint8_t*)PackedB + AlignedN * CountK;
    }
}

/**
 * @brief Operation for Quantized GEMM where B is symmetrically
 *          quantized and packed matrix
 * @param Shape
 * @param Data
 * @param RangeStartM
 * @param RangeCountM
 * @param RangeStartN
 * @param RangeCountN
*/
template<typename KernelType>
void
MlasSymmQGemmPackedOperation(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
    const MLAS_SYMM_QGEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
{

    const size_t K = Shape->K;

    const size_t lda = Data->lda;
    const size_t ldc = Data->ldc;

    const int8_t* PanelA = (const int8_t*)(Data->A) + RangeStartM * lda;
    const int8_t* PackedB = (const int8_t*)Data->B;
    int32_t* C = (int32_t*)(Data->C) + RangeStartM * ldc + RangeStartN;

    //
    // Extract the pointer to the column sum buffer from the packed matrix.
    //
    const size_t AlignedN =
        (Shape->N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const int32_t* PackedColumnSumBuffer = (const int32_t*)PackedB;
    PackedB = (const int8_t*)(PackedColumnSumBuffer + AlignedN);
    PackedColumnSumBuffer += RangeStartN;

    const size_t PackedCountK = (K + KernelType::PackedK - 1) / KernelType::PackedK;

    //
    // Apply the global depth value constant without the ZeroPointB scaling from:
    //
    //     (A[i] - ZeroPointA) * (B[i] - ZeroPointB)
    //              ==>
    //     A[i] * B[i] - A[i] * ZeroPointB - B[i] * ZeroPointA + ZeroPointA * ZeroPointB
    //
    // ZeroPointB is zero, which makes this much simpler
    //

    const int8_t* b = PackedB + RangeStartN * KernelType::PackedK * PackedCountK;
    int32_t* c = C;

    auto pa = PanelA;
    size_t RowsRemaining = RangeCountM;

    while (RowsRemaining > 0) {
        size_t RowsHandled = MlasSymmQGemmKernel<KernelType>(
            pa, b, c, PackedCountK, RowsRemaining, RangeCountN, ldc, lda, PackedColumnSumBuffer);

        c += ldc * RowsHandled;
        pa += lda * RowsHandled;
        RowsRemaining -= RowsHandled;
    }
}


//
// Quantized integer matrix/matrix dispatch structure.
//

typedef
void
(MLAS_GEMM_QUANT_OPERATION)(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );

typedef
void
(MLAS_SYMM_QGEMM_OPERATION)(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS* Shape,
    const MLAS_SYMM_QGEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );

typedef
void
(MLAS_GEMM_QUANT_COPY_PACKB_ROUTINE)(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    );

struct MLAS_GEMM_QUANT_DISPATCH {
    MLAS_GEMM_QUANT_OPERATION* Operation;
    MLAS_GEMM_QUANT_OPERATION* PackedOperation;
    MLAS_GEMM_QUANT_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    size_t PackedK;
    size_t PackedStrideK;
    size_t StrideM;
};

struct MLAS_SYMM_QGEMM_DISPATCH {
    MLAS_SYMM_QGEMM_OPERATION* LitOperation; /// running on little cores with narrow memory load
    MLAS_SYMM_QGEMM_OPERATION* BigOperation; /// running on big cores with wider memory load
    MLAS_GEMM_QUANT_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    size_t StrideM; /**< num of rows processed by kernel at a time */
    size_t PackedK;
};

MLAS_FORCEINLINE
const MLAS_GEMM_QUANT_DISPATCH*
MlasGemmQuantGetDispatch(
    bool AIsSigned,
    bool BIsSigned
)
{
    const MLAS_GEMM_QUANT_DISPATCH* GemmQuantDispatch = &MlasGemmQuantDispatchDefault;

#if !defined(FORCE_GENERIC_ALGORITHMS)
#if defined(MLAS_TARGET_AMD64_IX86)
    if (AIsSigned) {
        GemmQuantDispatch =
            BIsSigned ? GetMlasPlatform().GemmS8S8Dispatch : GetMlasPlatform().GemmS8U8Dispatch;
    } else {
        GemmQuantDispatch =
            BIsSigned ? GetMlasPlatform().GemmU8S8Dispatch : GetMlasPlatform().GemmU8U8Dispatch;
    }
#elif defined(MLAS_TARGET_ARM64)
    if(BIsSigned) {
        GemmQuantDispatch = AIsSigned ? GetMlasPlatform().GemmS8S8Dispatch : GetMlasPlatform().GemmU8S8Dispatch;
    } else if(!AIsSigned) {
        GemmQuantDispatch = GetMlasPlatform().GemmU8U8Dispatch;
    }
#elif defined(MLAS_TARGET_ARM64EC) || (defined(MLAS_TARGET_ARM) && !defined(_MSC_VER))
    if(BIsSigned || !AIsSigned) {
        GemmQuantDispatch = &MlasGemmU8X8DispatchNeon;
    }
#elif defined(MLAS_TARGET_WASM_RELAXED_SIMD)
    if (!AIsSigned) {
        if (HasUSDot()) {
          GemmQuantDispatch = &MlasGemmU8X8DispatchWasmRelaxedSimd;
        } else {
          GemmQuantDispatch = &MlasGemmU8X8DispatchWasmSimd;
        }
    }
#elif defined(MLAS_TARGET_WASM_SIMD)
    if (!AIsSigned) {
        GemmQuantDispatch = &MlasGemmU8X8DispatchWasmSimd;
    }
#elif defined(MLAS_TARGET_POWER) && (defined(__linux__)  || defined(_AIX)) && defined(POWER10) && \
    ((defined(__GNUC__) && ((__GNUC__ > 10) || (__GNUC__== 10 && __GNUC_MINOR__ >= 2))) || \
    (defined(__clang__) && (__clang_major__ >= 12)))
    if (GetMlasPlatform().GemmU8X8Dispatch == &MlasGemm8X8DispatchPOWER10) {
        GemmQuantDispatch = GetMlasPlatform().GemmU8X8Dispatch;
    }
#elif defined(MLAS_TARGET_LARCH64)
    if (AIsSigned) {
        GemmQuantDispatch =
            BIsSigned ? GetMlasPlatform().GemmS8S8Dispatch : GetMlasPlatform().GemmS8U8Dispatch;
    } else { // !AIsSigned
        GemmQuantDispatch =
            BIsSigned ? GetMlasPlatform().GemmU8S8Dispatch : GetMlasPlatform().GemmU8U8Dispatch;
    }
#elif defined(MLAS_TARGET_S390X)
    if (GetMlasPlatform().GemmU8X8Dispatch == &MlasGemm8X8DispatchZVECTOR) {
        GemmQuantDispatch = GetMlasPlatform().GemmU8X8Dispatch;
    }
#endif
#endif // !defined(FORCE_GENERIC_ALGORITHMS)

    if (nullptr == GemmQuantDispatch) {
        std::stringstream ss;
        ss << "Quant GEMM format: AIsSigned(" << AIsSigned << "), BIsSigned(" << BIsSigned
           << ") is not supported on this device";
        MLAS_THROW_EX(std::invalid_argument, ss.str());
    }

    return GemmQuantDispatch;
}
