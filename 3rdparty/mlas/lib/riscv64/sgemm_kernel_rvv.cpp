/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm_kernel_rvv.cpp

Abstract:

    This module implements an RVV kernel for the single precision matrix/matrix
    multiply operation (SGEMM) on riscv64.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

namespace {

// The packed B layout stays 16 columns wide to match MLAS, but each tile is
// consumed in runtime-sized RVV chunks so the kernel is not tied to a fixed
// VLEN such as 128 or 256 bits.
constexpr size_t kPackedCountN = 16;

template<bool ZeroMode, bool AlphaIsOne>
MLAS_FORCEINLINE
void
MlasStoreAccumulatorRvv(
    float* C,
    vfloat32m4_t Accumulator,
    size_t vl,
    float alpha
    )
{
#if defined(_WIN32)

    if constexpr (AlphaIsOne) {
        UNREFERENCED_PARAMETER(alpha);
    }

#endif

    if constexpr (!AlphaIsOne) {
        Accumulator = __riscv_vfmul_vf_f32m4(Accumulator, alpha, vl);
    }

    if constexpr (!ZeroMode) {
        Accumulator = __riscv_vfadd_vv_f32m4(Accumulator, __riscv_vle32_v_f32m4(C, vl), vl);
    }

    __riscv_vse32_v_f32m4(C, Accumulator, vl);
}

template<bool ZeroMode, bool AlphaIsOne, size_t Rows>
MLAS_FORCEINLINE
size_t
MlasSgemmKernelRvv(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
{
    static_assert(Rows >= 1 && Rows <= 4, "unsupported RVV SGEMM tile height");

#if defined(_WIN32)

    if constexpr (Rows == 1) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

    if constexpr (AlphaIsOne) {
        UNREFERENCED_PARAMETER(alpha);
    }

#endif

    const float* packed_b_block = B;
    float* c_block = C;
    size_t remaining_n_total = CountN;

    do {
        const size_t count_n_block = remaining_n_total >= kPackedCountN ? kPackedCountN : remaining_n_total;
        size_t remaining_n_block = count_n_block;
        size_t column_offset = 0;
        float* c = c_block;

        while (remaining_n_block > 0) {
            // Split a packed 16-column tile into however many lanes the current
            // machine exposes for e32,m4. This keeps the kernel VLEN-agnostic.
            const size_t vl = __riscv_vsetvl_e32m4(remaining_n_block);
            vfloat32m4_t row0_block = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t row1_block;
            vfloat32m4_t row2_block;
            vfloat32m4_t row3_block;

            if constexpr (Rows >= 2) {
                row1_block = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            }
            if constexpr (Rows >= 3) {
                row2_block = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            }
            if constexpr (Rows >= 4) {
                row3_block = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            }

            const float* a = A;
            const float* b = packed_b_block + column_offset;
            size_t k = CountK;

            while (k >= 2) {
                const float row0_a0 = a[0];
                const float row0_a1 = a[1];
                vfloat32m4_t b_elements = __riscv_vle32_v_f32m4(b, vl);
                row0_block = __riscv_vfmacc_vf_f32m4(row0_block, row0_a0, b_elements, vl);

                if constexpr (Rows >= 2) {
                    row1_block = __riscv_vfmacc_vf_f32m4(row1_block, a[lda], b_elements, vl);
                }
                if constexpr (Rows >= 3) {
                    row2_block = __riscv_vfmacc_vf_f32m4(row2_block, a[lda * 2], b_elements, vl);
                }
                if constexpr (Rows >= 4) {
                    row3_block = __riscv_vfmacc_vf_f32m4(row3_block, a[lda * 3], b_elements, vl);
                }

                b_elements = __riscv_vle32_v_f32m4(b + kPackedCountN, vl);
                row0_block = __riscv_vfmacc_vf_f32m4(row0_block, row0_a1, b_elements, vl);

                if constexpr (Rows >= 2) {
                    row1_block = __riscv_vfmacc_vf_f32m4(row1_block, a[lda + 1], b_elements, vl);
                }
                if constexpr (Rows >= 3) {
                    row2_block = __riscv_vfmacc_vf_f32m4(row2_block, a[lda * 2 + 1], b_elements, vl);
                }
                if constexpr (Rows >= 4) {
                    row3_block = __riscv_vfmacc_vf_f32m4(row3_block, a[lda * 3 + 1], b_elements, vl);
                }

                a += 2;
                b += kPackedCountN * 2;
                k -= 2;
            }

            if (k > 0) {
                vfloat32m4_t b_elements = __riscv_vle32_v_f32m4(b, vl);
                row0_block = __riscv_vfmacc_vf_f32m4(row0_block, a[0], b_elements, vl);

                if constexpr (Rows >= 2) {
                    row1_block = __riscv_vfmacc_vf_f32m4(row1_block, a[lda], b_elements, vl);
                }
                if constexpr (Rows >= 3) {
                    row2_block = __riscv_vfmacc_vf_f32m4(row2_block, a[lda * 2], b_elements, vl);
                }
                if constexpr (Rows >= 4) {
                    row3_block = __riscv_vfmacc_vf_f32m4(row3_block, a[lda * 3], b_elements, vl);
                }
            }

            MlasStoreAccumulatorRvv<ZeroMode, AlphaIsOne>(c, row0_block, vl, alpha);

            if constexpr (Rows >= 2) {
                MlasStoreAccumulatorRvv<ZeroMode, AlphaIsOne>(c + ldc, row1_block, vl, alpha);
            }
            if constexpr (Rows >= 3) {
                MlasStoreAccumulatorRvv<ZeroMode, AlphaIsOne>(c + ldc * 2, row2_block, vl, alpha);
            }
            if constexpr (Rows >= 4) {
                MlasStoreAccumulatorRvv<ZeroMode, AlphaIsOne>(c + ldc * 3, row3_block, vl, alpha);
            }

            c += vl;
            column_offset += vl;
            remaining_n_block -= vl;
        }

        c_block += count_n_block;
        packed_b_block += CountK * kPackedCountN;
        remaining_n_total -= count_n_block;

    } while (remaining_n_total > 0);

    return Rows;
}

template<bool ZeroMode, bool AlphaIsOne>
MLAS_FORCEINLINE
size_t
MlasGemmFloatKernelRvvDispatchRows(
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
{
    if (CountM >= 4) {
        return MlasSgemmKernelRvv<ZeroMode, AlphaIsOne, 4>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    if (CountM == 3) {
        return MlasSgemmKernelRvv<ZeroMode, AlphaIsOne, 3>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    if (CountM >= 2) {
        return MlasSgemmKernelRvv<ZeroMode, AlphaIsOne, 2>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    return MlasSgemmKernelRvv<ZeroMode, AlphaIsOne, 1>(A, B, C, CountK, CountN, lda, ldc, alpha);
}

template<bool ZeroMode>
MLAS_FORCEINLINE
size_t
MlasGemmFloatKernelRvvDispatch(
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
{
    if (alpha == 1.0f) {
        return MlasGemmFloatKernelRvvDispatchRows<ZeroMode, true>(
            A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
    }

    return MlasGemmFloatKernelRvvDispatchRows<ZeroMode, false>(
        A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

}  // namespace

size_t
MLASCALL
MlasGemmFloatKernelRvv(
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
{
    if (ZeroMode) {
        return MlasGemmFloatKernelRvvDispatch<true>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
    }

    return MlasGemmFloatKernelRvvDispatch<false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

#endif  // defined(MLAS_USE_RVV)
