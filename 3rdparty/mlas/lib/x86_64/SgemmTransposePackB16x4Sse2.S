/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmTransposePackB16x4Sse2.s

Abstract:

    This module implements routines for packing buffers for the single precision
    matrix/matrix multiply operation (SGEMM).

    This implementation uses SSE2 instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

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

        FUNCTION_ENTRY MlasSgemmTransposePackB16x4Sse

        shl     rdx,2                       # convert ldb to bytes
        mov     ecx,4                       # transpose four 4x4 blocks

.LTransposeBlockLoop:
        lea     rax,[rsi+rdx*2]
        movups  xmm0,XMMWORD PTR [rsi]
        movups  xmm1,XMMWORD PTR [rsi+rdx]
        movups  xmm2,XMMWORD PTR [rax]
        movups  xmm3,XMMWORD PTR [rax+rdx]
        movaps  xmm4,xmm0
        unpcklps xmm4,xmm1
        unpckhps xmm0,xmm1
        movaps  xmm5,xmm2
        unpcklps xmm5,xmm3
        unpckhps xmm2,xmm3
        movaps  xmm1,xmm4
        unpcklpd xmm1,xmm5
        unpckhpd xmm4,xmm5
        movaps  xmm3,xmm0
        unpcklpd xmm3,xmm2
        unpckhpd xmm0,xmm2
        movaps  XMMWORD PTR [rdi+16*4*0],xmm1
        movaps  XMMWORD PTR [rdi+16*4*1],xmm4
        movaps  XMMWORD PTR [rdi+16*4*2],xmm3
        movaps  XMMWORD PTR [rdi+16*4*3],xmm0
        add     rdi,4*4
        lea     rsi,[rax+rdx*2]
        dec     ecx
        jnz     .LTransposeBlockLoop
        ret

        .end
