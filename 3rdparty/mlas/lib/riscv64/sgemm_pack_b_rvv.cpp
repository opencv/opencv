/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm_pack_b_rvv.cpp

Abstract:

    This module implements an RVV packing helper for the single precision
    matrix/matrix multiply operation (SGEMM) on riscv64.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

namespace {

// Keep MLAS packing in 16-column tiles, but let RVV decide the actual chunk
// size at runtime via vsetvl so the same code works across different VLENs.
constexpr size_t kPackedCountN = 16;

MLAS_FORCEINLINE
void
MlasStoreZeroPaddedBlock(
    float* D,
    const float* B,
    size_t CountX
    )
{
    size_t remaining = kPackedCountN;
    size_t offset = 0;

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(remaining);
        __riscv_vse32_v_f32m4(D + offset, __riscv_vfmv_v_f_f32m4(0.0f, vl), vl);
        offset += vl;
        remaining -= vl;
    }

    remaining = CountX;
    offset = 0;

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(remaining);
        __riscv_vse32_v_f32m4(D + offset, __riscv_vle32_v_f32m4(B + offset, vl), vl);
        offset += vl;
        remaining -= vl;
    }
}

MLAS_FORCEINLINE
void
MlasStoreFullBlock(
    float* D,
    const float* B
    )
{
    size_t remaining = kPackedCountN;
    size_t offset = 0;

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(remaining);
        __riscv_vse32_v_f32m4(D + offset, __riscv_vle32_v_f32m4(B + offset, vl), vl);
        offset += vl;
        remaining -= vl;
    }
}

}  // namespace

void
MlasSgemmCopyPackBRvv(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
{
    while (CountX >= kPackedCountN) {
        const float* b = B;
        size_t y = CountY;

        do {
            MlasStoreFullBlock(D, b);
            D += kPackedCountN;
            b += ldb;
            y--;
        } while (y > 0);

        B += kPackedCountN;
        CountX -= kPackedCountN;
    }

    if (CountX > 0) {
        size_t y = CountY;

        do {
            MlasStoreZeroPaddedBlock(D, B, CountX);
            D += kPackedCountN;
            B += ldb;
            y--;
        } while (y > 0);
    }
}

#endif  // defined(MLAS_USE_RVV)
