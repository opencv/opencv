/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm.cpp

Abstract:

    This module implements the single precision matrix/matrix multiply
    operation (SGEMM).

--*/

#include "mlasi.h"

//
// Define the number of rows from matrix A to transpose to a local buffer.
//
// N.B. AVX processes a maximum of 4 rows, FMA3 processes a maximum of 6
// rows, and AVX512F processes a maximum of 12 rows.
//

#define MLAS_SGEMM_TRANSA_ROWS              12

//
// Define the parameters to execute segments of a SGEMM operation on worker
// threads.
//

void
MlasSgemmMultiplyBeta(
    float* C,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    float beta
    )
/*++

Routine Description:

    This routine multiplies all elements of the output matrix by the beta
    scalar value.

Arguments:

    C - Supplies the address of matrix C.

    CountM - Supplies the number of rows from matrix C.

    CountN - Supplies the number of columns from matrix C.

    ldc - Supplies the first dimension of matrix C.

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 BetaBroadcast = MlasBroadcastFloat32x4(beta);

    while (CountM-- > 0) {

        float* c = C;
        size_t n = CountN;

        while (n >= 4) {
            MlasStoreFloat32x4(c, MlasMultiplyFloat32x4(MlasLoadFloat32x4(c), BetaBroadcast));
            c += 4;
            n -= 4;
        }

        while (n > 0) {
#if defined(MLAS_SSE2_INTRINSICS)
            _mm_store_ss(c, _mm_mul_ss(_mm_load_ss(c), BetaBroadcast));
#else
            *c = *c * beta;
#endif
            c += 1;
            n -= 1;
        }

        C += ldc;
    }
}

void
MlasSgemmTransposeA(
    float* D,
    const float* A,
    size_t lda,
    size_t CountY,
    size_t CountX
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    buffer.

Arguments:

    D - Supplies the address of the destination buffer.

    A - Supplies the address of the source matrix.

    lda - Supplies the number of elements per row of the source matrix.

    CountY - Supplies the number of columns of the source matrix to transpose.

    CountX - Supplies the number of rows of the source matrix to transpose.

Return Value:

    None.

--*/
{
    size_t ldd = CountX;

    //
    // Transpose elements from matrix A into the destination buffer 4 columns
    // at a time.
    //

    while (CountX >= 4) {

        float* d = D;
        const float* a = A;
        size_t y = CountY;

        do {

            float t0 = a[0];
            float t1 = a[lda];
            float t2 = a[lda * 2];
            float t3 = a[lda * 3];

            d[0] = t0;
            d[1] = t1;
            d[2] = t2;
            d[3] = t3;

            d += ldd;
            a += 1;
            y--;

        } while (y > 0);

        D += 4;
        A += lda * 4;
        CountX -= 4;
    }

    //
    // Transpose elements from matrix A into the destination buffer for the
    // remaining columns.
    //

    if (CountX >= 2) {

        float* d = D;
        const float* a = A;
        size_t y = CountY;

        do {

            float t0 = a[0];
            float t1 = a[lda];

            d[0] = t0;
            d[1] = t1;

            d += ldd;
            a += 1;
            y--;

        } while (y > 0);

        D += 2;
        A += lda * 2;
        CountX -= 2;
    }

    if (CountX >= 1) {

        float* d = D;
        const float* a = A;
        size_t y = CountY;

        do {

            d[0] = a[0];

            d += ldd;
            a += 1;
            y--;

        } while (y > 0);
    }
}

#if !defined(MLAS_TARGET_WASM_SCALAR)

void
MlasSgemmCopyPackB(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

    Columns of 16 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SGEMM kernels. Any remaining
    columns less than 16 elements wide are zero-padded.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountX - Supplies the number of columns of the source matrix to copy.

    CountY - Supplies the number of rows of the source matrix to copy.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV) && !defined(FORCE_GENERIC_ALGORITHMS)
    if (GetMlasPlatform().GemmFloatKernel != nullptr) {
        MlasSgemmCopyPackBRvv(D, B, ldb, CountX, CountY);
        return;
    }
#endif

    //
    // Copy data from matrix B into the destination buffer 16 columns at a
    // time.
    //

    while (CountX >= 16) {

        const float* b = B;
        size_t y = CountY;

        do {

#if defined(MLAS_NEON_INTRINSICS)
            vst4q_f32(D, vld4q_f32(b));
#else
            MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);
            MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&b[4]);
            MLAS_FLOAT32X4 t2 = MlasLoadFloat32x4(&b[8]);
            MLAS_FLOAT32X4 t3 = MlasLoadFloat32x4(&b[12]);

            MlasStoreAlignedFloat32x4(&D[0], t0);
            MlasStoreAlignedFloat32x4(&D[4], t1);
            MlasStoreAlignedFloat32x4(&D[8], t2);
            MlasStoreAlignedFloat32x4(&D[12], t3);
#endif

            D += 16;
            b += ldb;
            y--;

        } while (y > 0);

        B += 16;
        CountX -= 16;
    }

    //
    // Special case the handling of the remaining columns less than 16 elements
    // wide.
    //

    if (CountX > 0) {

        MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

#if defined(MLAS_NEON_INTRINSICS)
        float32x4x4_t ZeroFloat32x4x4 = { ZeroFloat32x4, ZeroFloat32x4, ZeroFloat32x4, ZeroFloat32x4 };
#endif

        size_t y = CountY;

        do {

            float* d = D;
            const float* b = B;

#if defined(MLAS_NEON_INTRINSICS)
            vst4q_f32(d, ZeroFloat32x4x4);
#else
            MlasStoreAlignedFloat32x4(d, ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(d + 4, ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(d + 8, ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(d + 12, ZeroFloat32x4);
#endif

            if ((CountX & 8) != 0) {

                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(b);
                MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(b + 4);

                MlasStoreAlignedFloat32x4(d, t0);
                MlasStoreAlignedFloat32x4(d + 4, t1);

                d += 8;
                b += 8;
            }

            if ((CountX & 4) != 0) {

                MlasStoreAlignedFloat32x4(d, MlasLoadFloat32x4(b));

                d += 4;
                b += 4;
            }

            if ((CountX & 2) != 0) {

                float t0 = b[0];
                float t1 = b[1];

                d[0] = t0;
                d[1] = t1;

                d += 2;
                b += 2;
            }

            if ((CountX & 1) != 0) {
                d[0] = b[0];
            }

            D += 16;
            B += ldb;
            y--;

        } while (y > 0);
    }
}

template<unsigned N>
inline
void
MlasSgemmTransposePackBNx4(
    float* D,
    const float* B,
    size_t ldb
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    4 columns of N rows from the source matrix are transposed to N columns of 4
    rows in the destination packed buffer.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

Return Value:

    None.

--*/
{
    for (unsigned n = 0; n < N / 4; n++) {

        MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&B[ldb * 0]);
        MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&B[ldb * 1]);
        MLAS_FLOAT32X4 t2 = MlasLoadFloat32x4(&B[ldb * 2]);
        MLAS_FLOAT32X4 t3 = MlasLoadFloat32x4(&B[ldb * 3]);

#if defined(MLAS_NEON_INTRINSICS)
        float32x4x2_t z0 = vzipq_f32(t0, t2);
        float32x4x2_t z1 = vzipq_f32(t1, t3);
        float32x4x2_t o0 = vzipq_f32(z0.val[0], z1.val[0]);
        float32x4x2_t o1 = vzipq_f32(z0.val[1], z1.val[1]);
        t0 = o0.val[0];
        t1 = o0.val[1];
        t2 = o1.val[0];
        t3 = o1.val[1];
#else
        MLAS_FLOAT32X4 z0 = MlasInterleaveLowFloat32x4(t0, t2);
        MLAS_FLOAT32X4 z1 = MlasInterleaveHighFloat32x4(t0, t2);
        MLAS_FLOAT32X4 z2 = MlasInterleaveLowFloat32x4(t1, t3);
        MLAS_FLOAT32X4 z3 = MlasInterleaveHighFloat32x4(t1, t3);
        t0 = MlasInterleaveLowFloat32x4(z0, z2);
        t1 = MlasInterleaveHighFloat32x4(z0, z2);
        t2 = MlasInterleaveLowFloat32x4(z1, z3);
        t3 = MlasInterleaveHighFloat32x4(z1, z3);
#endif

        MlasStoreAlignedFloat32x4(&D[0], t0);
        MlasStoreAlignedFloat32x4(&D[16], t1);
        MlasStoreAlignedFloat32x4(&D[32], t2);
        MlasStoreAlignedFloat32x4(&D[48], t3);

        D += 4;
        B += ldb * 4;
    }
}

void
MlasSgemmTransposePackB(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountY,
    size_t CountX
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    Columns of 16 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SGEMM kernels. Any remaining
    columns less than 16 elements wide are zero-padded.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountY - Supplies the number of rows of the source matrix to transpose.

    CountX - Supplies the number of columns of the source matrix to transpose.

Return Value:

    None.

--*/
{
    //
    // Transpose elements from matrix B into the packed buffer 16 rows at a
    // time.
    //

    while (CountY >= 16) {

        const float* b = B;
        size_t x = CountX;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)

        MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE* SgemmTransposePackB16x4Routine =
            GetMlasPlatform().TransposePackB16x4Routine;

        while (x >= 4) {

            SgemmTransposePackB16x4Routine(&D[0], &b[0], ldb);

            D += 16 * 4;
            b += 4;
            x -= 4;
        }

#else

        while (x >= 4) {

            MlasSgemmTransposePackBNx4<16>(&D[0], &b[0], ldb);

            D += 16 * 4;
            b += 4;
            x -= 4;
        }

#endif

        while (x > 0) {

            float t0 = b[0];
            float t1 = b[ldb];
            float t2 = b[ldb * 2];
            float t3 = b[ldb * 3];
            float t4 = b[ldb * 4];
            float t5 = b[ldb * 5];
            float t6 = b[ldb * 6];
            float t7 = b[ldb * 7];
            float t8 = b[ldb * 8];
            float t9 = b[ldb * 9];
            float t10 = b[ldb * 10];
            float t11 = b[ldb * 11];
            float t12 = b[ldb * 12];
            float t13 = b[ldb * 13];
            float t14 = b[ldb * 14];
            float t15 = b[ldb * 15];

            D[0] = t0;
            D[1] = t1;
            D[2] = t2;
            D[3] = t3;
            D[4] = t4;
            D[5] = t5;
            D[6] = t6;
            D[7] = t7;
            D[8] = t8;
            D[9] = t9;
            D[10] = t10;
            D[11] = t11;
            D[12] = t12;
            D[13] = t13;
            D[14] = t14;
            D[15] = t15;

            D += 16;
            b += 1;
            x--;
        }

        B += ldb * 16;
        CountY -= 16;
    }

    //
    // Special case the handling of the less than 16 remaining rows.
    //

    if (CountY > 0) {

        MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

        size_t x = CountX;

        //
        // Transpose 4 columns at a time.
        //

        while (x >= 4) {

            float* d = D;
            const float* b = B;

            if ((CountY & 8) != 0) {

                MlasSgemmTransposePackBNx4<8>(&d[0], &b[0], ldb);

                d += 8;
                b += ldb * 8;

            } else {

                MlasStoreAlignedFloat32x4(&d[8], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[12], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[24], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[28], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[40], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[44], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[56], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[60], ZeroFloat32x4);
            }

            if ((CountY & 4) != 0) {

                MlasSgemmTransposePackBNx4<4>(&d[0], &b[0], ldb);

                d += 4;
                b += ldb * 4;

            } else {

                MlasStoreAlignedFloat32x4(&d[4], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[20], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[36], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[52], ZeroFloat32x4);
            }

            MlasStoreAlignedFloat32x4(&d[0], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[16], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[32], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[48], ZeroFloat32x4);

            if ((CountY & 2) != 0) {

                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);
                MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&b[ldb]);

#if defined(MLAS_SSE2_INTRINSICS)
                __m128 v0 = _mm_unpacklo_ps(t0, t1);
                __m128 v1 = _mm_unpackhi_ps(t0, t1);
                _mm_storel_pi((__m64*)&d[0], v0);
                _mm_storeh_pi((__m64*)&d[16], v0);
                _mm_storel_pi((__m64*)&d[32], v1);
                _mm_storeh_pi((__m64*)&d[48], v1);
#else
                MlasStoreLaneFloat32x4<0>(&d[0], t0);
                MlasStoreLaneFloat32x4<0>(&d[1], t1);
                MlasStoreLaneFloat32x4<1>(&d[16], t0);
                MlasStoreLaneFloat32x4<1>(&d[17], t1);
                MlasStoreLaneFloat32x4<2>(&d[32], t0);
                MlasStoreLaneFloat32x4<2>(&d[33], t1);
                MlasStoreLaneFloat32x4<3>(&d[48], t0);
                MlasStoreLaneFloat32x4<3>(&d[49], t1);
#endif

                d += 2;
                b += ldb * 2;
            }

            if ((CountY & 1) != 0) {

#if defined(MLAS_NEON_INTRINSICS)
                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);

                MlasStoreLaneFloat32x4<0>(&d[0], t0);
                MlasStoreLaneFloat32x4<1>(&d[16], t0);
                MlasStoreLaneFloat32x4<2>(&d[32], t0);
                MlasStoreLaneFloat32x4<3>(&d[48], t0);
#else
                d[0] = b[0];
                d[16] = b[1];
                d[32] = b[2];
                d[48] = b[3];
#endif
            }

            D += 16 * 4;
            B += 4;
            x -= 4;
        }

        //
        // Transpose the remaining columns.
        //

        while (x > 0) {

            float* d = D;
            const float* b = B;

            if ((CountY & 8) != 0) {

                float t0 = b[0];
                float t1 = b[ldb];
                float t2 = b[ldb * 2];
                float t3 = b[ldb * 3];
                float t4 = b[ldb * 4];
                float t5 = b[ldb * 5];
                float t6 = b[ldb * 6];
                float t7 = b[ldb * 7];

                d[0] = t0;
                d[1] = t1;
                d[2] = t2;
                d[3] = t3;
                d[4] = t4;
                d[5] = t5;
                d[6] = t6;
                d[7] = t7;

                d += 8;
                b += ldb * 8;

            } else {

                MlasStoreAlignedFloat32x4(&d[8], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[12], ZeroFloat32x4);
            }

            if ((CountY & 4) != 0) {

                float t0 = b[0];
                float t1 = b[ldb];
                float t2 = b[ldb * 2];
                float t3 = b[ldb * 3];

                d[0] = t0;
                d[1] = t1;
                d[2] = t2;
                d[3] = t3;

                d += 4;
                b += ldb * 4;

            } else {

                MlasStoreAlignedFloat32x4(&d[4], ZeroFloat32x4);
            }

            MlasStoreAlignedFloat32x4(d, ZeroFloat32x4);

            if ((CountY & 2) != 0) {

                float t0 = b[0];
                float t1 = b[ldb];

                d[0] = t0;
                d[1] = t1;

                d += 2;
                b += ldb * 2;
            }

            if ((CountY & 1) != 0) {
                d[0] = b[0];
            }

            D += 16;
            B += 1;
            x--;
        }
    }
}

#else //defined(MLAS_TARGET_WASM_SCALAR)

void
MlasSgemmCopyPackB(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

    Columns of 16 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SGEMM kernels. Any remaining
    columns less than 16 elements wide are zero-padded.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountX - Supplies the number of columns of the source matrix to copy.

    CountY - Supplies the number of rows of the source matrix to copy.

Return Value:

    None.

--*/
{
    //
    // Copy data from matrix B into the destination buffer 16 columns at a
    // time.
    //

    while (CountX >= 16) {

        const float* b = B;
        size_t y = CountY;

        do {

            std::copy_n(b, 16, D);

            D += 16;
            b += ldb;
            y--;

        } while (y > 0);

        B += 16;
        CountX -= 16;
    }

    //
    // Special case the handling of the remaining columns less than 16 elements
    // wide.
    //

    if (CountX > 0) {

        size_t y = CountY;

        do {

            std::fill_n(D, 16, 0.0f);
            std::copy_n(B, CountX, D);

            D += 16;
            B += ldb;
            y--;

        } while (y > 0);
    }
}

void
MlasSgemmTransposePackB(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountY,
    size_t CountX
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    Columns of 16 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SGEMM kernels. Any remaining
    columns less than 16 elements wide are zero-padded.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountY - Supplies the number of rows of the source matrix to transpose.

    CountX - Supplies the number of columns of the source matrix to transpose.

Return Value:

    None.

--*/
{
    //
    // Transpose elements from matrix B into the packed buffer 16 rows at a
    // time.
    //

    while (CountY >= 16) {

        const float* b = B;
        size_t x = CountX;

        while (x >= 4) {

            for (size_t row = 0; row < 16; row++) {
                D[0 * 16 + row] = b[row * ldb + 0];
                D[1 * 16 + row] = b[row * ldb + 1];
                D[2 * 16 + row] = b[row * ldb + 2];
                D[3 * 16 + row] = b[row * ldb + 3];
            }

            D += 16 * 4;
            b += 4;
            x -= 4;
        }

        while (x > 0) {

            for (size_t row = 0; row < 16; row++) {
                D[row] = b[row * ldb];
            }

            D += 16;
            b += 1;
            x--;
        }

        B += ldb * 16;
        CountY -= 16;
    }

    //
    // Special case the handling of the less than 16 remaining rows.
    //

    if (CountY > 0) {

        size_t x = CountX;

        //
        // Transpose 4 columns at a time.
        //

        while (x >= 4) {

            std::fill_n(D, 16 * 4, 0.0f);

            for (size_t row = 0; row < CountY; row++) {
                D[0 * 16 + row] = B[row * ldb + 0];
                D[1 * 16 + row] = B[row * ldb + 1];
                D[2 * 16 + row] = B[row * ldb + 2];
                D[3 * 16 + row] = B[row * ldb + 3];
            }

            D += 16 * 4;
            B += 4;
            x -= 4;
        }

        //
        // Transpose the remaining columns.
        //

        while (x > 0) {

            std::fill_n(D, 16, 0.0f);

            for (size_t row = 0; row < CountY; row++) {
                D[row] = B[row * ldb];
            }

            D += 16;
            B += 1;
            x--;
        }
    }
}

#endif

MLAS_FORCEINLINE
float*
MlasSgemmKernelLoop(
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

    This routine steps through the rows of the input and output matrices calling
    the kernel until all rows have been processed.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the number of rows from matrix A and matrix C to iterate
        over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the next address of matrix C.

--*/
{
    while (CountM > 0) {

        size_t RowsHandled;

#if (defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_POWER) || defined(MLAS_TARGET_S390X) || defined(MLAS_TARGET_LARCH64)) && !defined(FORCE_GENERIC_ALGORITHMS)
        RowsHandled = GetMlasPlatform().GemmFloatKernel(A, B, C, CountK, CountM, CountN, lda, ldc, alpha, ZeroMode);
#elif defined(MLAS_TARGET_RISCV64) && !defined(FORCE_GENERIC_ALGORITHMS)
        if (GetMlasPlatform().GemmFloatKernel != nullptr) {
            RowsHandled = GetMlasPlatform().GemmFloatKernel(A, B, C, CountK, CountM, CountN, lda, ldc, alpha, ZeroMode);
        } else if (ZeroMode) {
            RowsHandled = MlasSgemmKernelZero(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        } else {
            RowsHandled = MlasSgemmKernelAdd(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        }
#else
        if (ZeroMode) {
            RowsHandled = MlasSgemmKernelZero(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        } else {
            RowsHandled = MlasSgemmKernelAdd(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        }
#endif

        C += ldc * RowsHandled;
        A += lda * RowsHandled;
        CountM -= RowsHandled;
    }

    return C;
}

void
MlasSgemmOperation(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
    )
/*++

Routine Description:

    This routine implements the single precision matrix/matrix multiply
    operation (SGEMM).

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/
{
    float PanelA[MLAS_SGEMM_TRANSA_ROWS * MLAS_SGEMM_STRIDEK];
    MLAS_DECLSPEC_ALIGN(float PanelB[MLAS_SGEMM_STRIDEN * MLAS_SGEMM_STRIDEK], 16 * sizeof(float));

    //
    // Handle the special case of K equals zero. Apply the beta multiplier to
    // the output matrix and exit.
    //

    if (K == 0) {
        MlasSgemmMultiplyBeta(C, M, N, ldc, beta);
        return;
    }

    //
    // Handle the special case of a small M. The data from matrix B is not
    // referenced multiple times, so using a local packed buffer is a wasted
    // memory copy.
    //

    if (M == 1 && TransA == CblasNoTrans && alpha == 1.0f && (beta == 0.0f || beta == 1.0f)) {

#if !defined(FORCE_GENERIC_ALGORITHMS)
#if defined(MLAS_TARGET_AMD64)

        MLAS_SGEMM_KERNEL_M1_ROUTINE* SgemmKernelM1Routine;

        if (TransB == CblasNoTrans) {
            SgemmKernelM1Routine = GetMlasPlatform().KernelM1Routine;
        } else {
            SgemmKernelM1Routine = GetMlasPlatform().KernelM1TransposeBRoutine;
        }

        if (SgemmKernelM1Routine != nullptr) {
            SgemmKernelM1Routine(A, B, C, K, N, ldb, beta);
            return;
        }

#elif defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_WASM)

        if (TransB == CblasNoTrans) {
            MlasGemvFloatKernel(A, B, C, K, N, ldb, (beta == 0.0f));
            return;
        }

#endif
#endif // !defined(FORCE_GENERIC_ALGORITHMS)

    }

    //
    // Handle the case when both B and C are column-vectors that are contiguous in memory.
    // Because transposition of such vectors doesn't change their layout, and
    // Transpose(A*B) = Transpose(B) * Transpose(A), we can apply the same 'small-M'
    // optimization as above, with A and B flipped.
    //

    if (N == 1 && ldb == 1 && ldc == 1 && alpha == 1.0f && (beta == 0.0f || beta == 1.0f)) {

#if defined(MLAS_TARGET_AMD64) && !defined(FORCE_GENERIC_ALGORITHMS)

        MLAS_SGEMM_KERNEL_M1_ROUTINE* SgemmKernelM1Routine;

        if (TransA == CblasNoTrans) {
            SgemmKernelM1Routine = GetMlasPlatform().KernelM1TransposeBRoutine;
        } else {
            SgemmKernelM1Routine = GetMlasPlatform().KernelM1Routine;
        }

        if (SgemmKernelM1Routine != nullptr) {
            SgemmKernelM1Routine(B, A, C, K, M, lda, beta);
            return;
        }

#endif

    }

    //
    // Compute the strides to step through slices of the input matrices.
    //
    // Expand the N stride if K is small or expand the K stride if N is small
    // for better utilization of the B panel. Avoid changing the K stride if
    // the A panel needs to be used for transposing.
    //

    size_t StrideN = MLAS_SGEMM_STRIDEN;
    size_t StrideK = MLAS_SGEMM_STRIDEK;

    if (N >= K) {

        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }

    } else if (TransA == CblasNoTrans) {

        while (StrideN > 16 && StrideN / 2 >= N) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < N; n += CountN) {

        CountN = std::min(N - n, StrideN);

        //
        // Multiply the output matrix by beta as needed.
        //

        if (beta != 0.0f && beta != 1.0f) {
            MlasSgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        //
        // Step through each slice of matrix B along the K dimension.
        //

        size_t CountK;
        bool ZeroMode = (beta == 0.0f);

        for (size_t k = 0; k < K; k += CountK) {

            CountK = std::min(K - k, StrideK);

            //
            // Copy or transpose a panel of matrix B to a local packed buffer.
            //

            if (TransB == CblasNoTrans) {
                MlasSgemmCopyPackB(PanelB, B + n + k * ldb, ldb, CountN, CountK);
            } else {
                MlasSgemmTransposePackB(PanelB, B + k + n * ldb, ldb, CountN, CountK);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            float* c = C + n;

            if (TransA == CblasNoTrans) {

                MlasSgemmKernelLoop(A + k, PanelB, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);

            } else {

                const float* a = A + k * lda;
                size_t RowsRemaining = M;

                while (RowsRemaining > 0) {

                    //
                    // Transpose elements from matrix A into a local buffer.
                    //

                    size_t RowsTransposed = std::min(RowsRemaining, size_t(MLAS_SGEMM_TRANSA_ROWS));

                    MlasSgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);

                    RowsRemaining -= RowsTransposed;
                    a += RowsTransposed;

                    //
                    // Step through the rows of the local buffer.
                    //

                    c = MlasSgemmKernelLoop(PanelA, PanelB, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha, ZeroMode);
                }
            }

            ZeroMode = false;
        }
    }
}

void
MlasSgemmPackedOperation(
    CBLAS_TRANSPOSE TransA,
    size_t M,
    size_t RangeStartN,
    size_t RangeCountN,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const void* PackedB,
    size_t AlignedN,
    float beta,
    float* C,
    size_t ldc
    )
/*++

Routine Description:

    This routine implements the single precision matrix/matrix multiply
    operation (SGEMM).

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    M - Supplies the number of rows of matrix A and matrix C.

    RangeStartN - Supplies the starting column from packed matrix B.

    RangeCountN - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    PackedB - Supplies the address of packed matrix B.

    AlignedN - Supplies the total number of aligned columns for packed matrix B.

    ldb - Supplies the first dimension of matrix B.

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/
{
    float PanelA[MLAS_SGEMM_TRANSA_ROWS * MLAS_SGEMM_PACKED_STRIDEK];

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < RangeCountN; n += CountN) {

        const size_t SliceStartN = RangeStartN + n;

        CountN = std::min(RangeCountN - n, size_t(MLAS_SGEMM_PACKED_STRIDEN));

        //
        // Multiply the output matrix by beta as needed.
        //

        if (beta != 0.0f && beta != 1.0f) {
            MlasSgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        //
        // Step through each slice of matrix B along the K dimension.
        //

        size_t CountK;
        bool ZeroMode = (beta == 0.0f);

        for (size_t k = 0; k < K; k += CountK) {

            CountK = std::min(K - k, size_t(MLAS_SGEMM_PACKED_STRIDEK));

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const float* pb = (const float*)PackedB + AlignedN * k + CountK * SliceStartN;
            float* c = C + n;

            if (TransA == CblasNoTrans) {

                MlasSgemmKernelLoop(A + k, pb, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);

            } else {

                const float* a = A + k * lda;
                size_t RowsRemaining = M;

                while (RowsRemaining > 0) {

                    //
                    // Transpose elements from matrix A into a local buffer.
                    //

                    size_t RowsTransposed = std::min(RowsRemaining, size_t(MLAS_SGEMM_TRANSA_ROWS));

                    MlasSgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);

                    RowsRemaining -= RowsTransposed;
                    a += RowsTransposed;

                    //
                    // Step through the rows of the local buffer.
                    //

                    c = MlasSgemmKernelLoop(PanelA, pb, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha, ZeroMode);
                }
            }

            ZeroMode = false;
        }
    }
}

void
MlasSgemmThreaded(
    const ptrdiff_t ThreadCountM,
    const ptrdiff_t ThreadCountN,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const size_t M,
    const size_t N,
    const size_t K,

    const MLAS_SGEMM_DATA_PARAMS* DataParams,
    ptrdiff_t ThreadId
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    SGEMM operation.

Arguments:

    ThreadCountM - Supplies the total thread partition on the M dimension.

    ThreadCountN - Supplies the total thread partition on the N dimension.

    TransA - Supplies the transpose operation on A matrix

    TransB - Supplies the transpose operation on B matrix

    M, N, K - Supplies the shape of the multiplication

    DataParams - Supplies the data position and layout of the matrices

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{

    const ptrdiff_t ThreadIdM = ThreadId / ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    size_t RangeStartM;
    size_t RangeCountM;

    MlasPartitionWork(ThreadIdM, ThreadCountM, M, &RangeStartM, &RangeCountM);

    //
    // Partition the operation along the N dimension.
    //

    size_t RangeStartN;
    size_t RangeCountN;

    const size_t BlockedN = (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) /
        MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &RangeStartN,
        &RangeCountN);

    RangeStartN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    //
    // Dispatch the partitioned operation.
    //

    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const float* A = DataParams->A + RangeStartM * ((TransA == CblasNoTrans) ? lda : 1);
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    if (DataParams->BIsPacked) {

        MlasSgemmPackedOperation(TransA, RangeCountM, RangeStartN, RangeCountN,
            K, DataParams->alpha, A, lda, DataParams->B,
            BlockedN * MLAS_SGEMM_STRIDEN_THREAD_ALIGN, DataParams->beta, C, ldc);

    } else {

        const size_t ldb = DataParams->ldb;

        const float* B = (const float*)DataParams->B + RangeStartN * ((TransB == CblasNoTrans) ? 1 : ldb);

        MlasSgemmOperation(TransA, TransB, RangeCountM, RangeCountN, K,
            DataParams->alpha, A, lda, B, ldb, DataParams->beta, C, ldc);
    }
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
void
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    // Override
    if ((!BackendKernelSelectorConfig || BackendKernelSelectorConfig->use_kleidiai) &&
        GetMlasPlatform().MlasSGemmBatchOverride != nullptr &&
        // TODO: Remove once KAI supports transposing for A
        TransA != CBLAS_TRANSPOSE::CblasTrans &&
        GetMlasPlatform().MlasSGemmBatchOverride(TransA, TransB, M, N, K, Data, BatchSize, ThreadPool)){
        return;
    }
    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    //
    // Segment the operation across multiple threads.
    //
    // N.B. Currently, the operation is segmented as a 1D partition, which
    // works okay for operations involving skinny matrices.
    //

    ptrdiff_t ThreadsPerGemm = (TargetThreadCount + BatchSize - 1) / BatchSize;
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;

    if (N > M) {

        const size_t BlockedN = (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) /
            MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

        if (size_t(ThreadsPerGemm) > BlockedN) {
            ThreadsPerGemm = ptrdiff_t(BlockedN);
        }

        ThreadCountM = 1;
        ThreadCountN = ThreadsPerGemm;

    } else {

        if (size_t(ThreadsPerGemm) > M) {
            ThreadsPerGemm = ptrdiff_t(M);
        }

        ThreadCountM = ThreadsPerGemm;
        ThreadCountN = 1;
    }

    MlasTrySimpleParallel(ThreadPool,
        ThreadsPerGemm * static_cast<ptrdiff_t>(BatchSize),
        [=](ptrdiff_t tid)
    {
        ptrdiff_t GemmIdx = tid / ThreadsPerGemm;
        ptrdiff_t ThreadIdx = tid % ThreadsPerGemm;
        MlasSgemmThreaded(ThreadCountM, ThreadCountN,
            TransA, TransB, M, N, K, &(Data[GemmIdx]), ThreadIdx);
    });
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

size_t
MLASCALL
MlasGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
/*++

Routine Description:

    This routine computes the length in bytes for the packed matrix B buffer.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

    BackendKernelSelectorConfig - Supplies the backend kernel selector
                                  configuration options, else nullptr if the
                                  default configuration should be used.
Return Value:

    Returns the size in bytes for the packed matrix B buffer.

--*/
{
    //
    // Compute the number of bytes required to hold the packed buffer.
    //
    // KleidiAI or other override
    #if defined(USE_KLEIDIAI)
    if ((!BackendKernelSelectorConfig || BackendKernelSelectorConfig->use_kleidiai) &&
        GetMlasPlatform().MlasSGemmPackBSizeOverride != nullptr &&
        // TODO: Remove once KAI supports transposing for A
        TransA != CBLAS_TRANSPOSE::CblasTrans) {
        size_t bytes_required;
        //TODO pass status by reference to indicate success/fail
        bytes_required = GetMlasPlatform().MlasSGemmPackBSizeOverride(TransA, TransB, N, K);
        if (bytes_required != 0){// If ArmKleidiAI::MlasGemmPackBSize ran to completion
            return bytes_required;
        }
    }
    #endif
    MLAS_UNREFERENCED_PARAMETER(TransA);
    MLAS_UNREFERENCED_PARAMETER(TransB);
    MLAS_UNREFERENCED_PARAMETER(BackendKernelSelectorConfig);



    const size_t AlignedN =
        (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1);

    const size_t BytesRequired = AlignedN * K * sizeof(float);
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired = (BytesRequired + BufferAlignment - 1) &
        ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

void
MLASCALL
MlasGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
/*++

Routine Description:

    This routine packs the contents of matrix B to the destination buffer. The
    destination buffer should be sized based on MlasGemmPackBSize(). For best
    performance, the destination buffer should be aligned to the value returned
    from MlasGetPreferredBufferAlignment().

Arguments:

    TransB - Supplies the transpose operation for matrix B.

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    None.

--*/
{
#if defined(USE_KLEIDIAI)
    if ((!BackendKernelSelectorConfig || BackendKernelSelectorConfig->use_kleidiai) &&
        GetMlasPlatform().MlasSGemmPackBOverride != nullptr  &&
        // TODO: Remove once KAI supports transposing for A
        TransA != CBLAS_TRANSPOSE::CblasTrans    &&
        GetMlasPlatform().MlasSGemmPackBOverride(TransA, TransB, N, K, B, ldb, PackedB)){
         return;
    }
#endif
    MLAS_UNREFERENCED_PARAMETER(TransA);
    MLAS_UNREFERENCED_PARAMETER(BackendKernelSelectorConfig);


    const size_t AlignedN =
        (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1);

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, size_t(MLAS_SGEMM_PACKED_STRIDEK));

        if (TransB == CblasNoTrans) {
            MlasSgemmCopyPackB((float*)PackedB, B + k * ldb, ldb, N, CountK);
        } else {
            MlasSgemmTransposePackB((float*)PackedB, B + k, ldb, N, CountK);
        }

        PackedB = (float*)PackedB + AlignedN * CountK;
    }
}
