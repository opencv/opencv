/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmTransposePackB16x4Avx.s

Abstract:

    This module implements routines for packing buffers for the single precision
    matrix/matrix multiply operation (SGEMM).

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

/*++

Macro Description:

    4 columns of 8 rows from the source matrix are transposed to 8 columns of 4
    rows in the destination packed buffer.

Arguments:

    StoreOffset - Supplies the relative byte offset into the destination packed
        buffer.

Implicit Arguments:

    rdi - Supplies the address of the destination packed buffer.

    rsi - Supplies the address of the source matrix.

    rdx - Supplies the number of elements per row of the source matrix.

--*/

        .macro TransposePackB8x4BlockAvx StoreOffset

//
// Load 4 columns from 8 rows of the source matrix into the lower and upper
// halves of 4 YMM registers.
//

        lea     rax,[rsi+rdx*2]
        vmovups xmm0,XMMWORD PTR [rsi]
        vmovups xmm1,XMMWORD PTR [rsi+rdx]
        lea     rsi,[rax+rdx*2]
        vmovups xmm2,XMMWORD PTR [rax]
        vmovups xmm3,XMMWORD PTR [rax+rdx]
        lea     rax,[rsi+rdx*2]
        vinsertf128 ymm0,ymm0,XMMWORD PTR [rsi],1
        vinsertf128 ymm1,ymm1,XMMWORD PTR [rsi+rdx],1
        vinsertf128 ymm2,ymm2,XMMWORD PTR [rax],1
        vinsertf128 ymm3,ymm3,XMMWORD PTR [rax+rdx],1

//
// Transpose the lower and upper halves of the 4 YMM registers as two 4x4
// matrices and store the output to the destination packed buffer.
//

        vunpcklps ymm4,ymm0,ymm1
        vunpckhps ymm5,ymm0,ymm1
        vunpcklps ymm0,ymm2,ymm3
        vunpckhps ymm1,ymm2,ymm3
        vunpcklpd ymm2,ymm4,ymm0
        vunpckhpd ymm3,ymm4,ymm0
        vmovaps YMMWORD PTR [rdi+16*4*0+\StoreOffset\()],ymm2
        vmovaps YMMWORD PTR [rdi+16*4*1+\StoreOffset\()],ymm3
        vunpcklpd ymm0,ymm5,ymm1
        vunpckhpd ymm4,ymm5,ymm1
        vmovaps YMMWORD PTR [rdi+16*4*2+\StoreOffset\()],ymm0
        vmovaps YMMWORD PTR [rdi+16*4*3+\StoreOffset\()],ymm4

        .endm

/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    4 columns of 16 rows from the source matrix are transposed to 16 columns of 4
    rows in the destination packed buffer.

Arguments:

    D (rdi) - Supplies the address of the destination packed buffer.

    B (rsi) - Supplies the address of the source matrix.

    ldb (rdx) - Supplies the number of elements per row of the source matrix.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasSgemmTransposePackB16x4Avx

        shl     rdx,2                       # convert ldb to bytes
        TransposePackB8x4BlockAvx 0*4
        lea     rsi,[rax+rdx*2]
        TransposePackB8x4BlockAvx 8*4
        vzeroupper
        ret

        .end
