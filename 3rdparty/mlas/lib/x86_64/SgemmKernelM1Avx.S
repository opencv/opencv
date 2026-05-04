/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelM1Avx.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM). This handles the special case of M=1.

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows. This handles the special case of M=1.

    The elements in matrix B are not transposed.

Arguments:

    A (rdi) - Supplies the address of matrix A.

    B (rsi) - Supplies the address of matrix B.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountN (r8) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldb (r9) - Supplies the first dimension of matrix B.

    Beta (xmm0) - Supplies the scalar beta multiplier (see SGEMM definition).

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasSgemmKernelM1Avx

        push    rbx
        shl     r9,2                        # convert ldb to bytes
        mov     r10,rdx
        mov     r11,rsi

//
// Compute the initial results mask for zeroing or accumulate mode.
//

        vxorps  xmm1,xmm1,xmm1
        vcmpeqss xmm0,xmm1,xmm0
        vshufps xmm0,xmm0,xmm0,0
        vinsertf128 ymm0,ymm0,xmm0,1

//
// Compute the conditional load/store mask for an unaligned CountN.
//

        mov     eax,r8d
        and     eax,7
        vmovd   xmm7,eax
        vshufps xmm7,xmm7,xmm7,0
        vpcmpgtd xmm6,xmm7,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip+16]
        vpcmpgtd xmm7,xmm7,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vinsertf128 ymm7,ymm7,xmm6,1

//
// Process 4 rows of the matrices in a loop.
//

        sub     rcx,4
        jb      .LProcessRemainingCountK

.LProcessRowLoop4:
        vbroadcastss ymm2,DWORD PTR [rdi]
        mov     rax,r8                      # reload CountN
        vbroadcastss ymm3,DWORD PTR [rdi+4]
        mov     rsi,r11                     # reload matrix B
        vbroadcastss ymm4,DWORD PTR [rdi+8]
        mov     rdx,r10                     # reload matrix C
        vbroadcastss ymm5,DWORD PTR [rdi+12]
        add     rdi,4*4                     # advance matrix A by 4 columns
        lea     r11,[rsi+r9*4]              # advance matrix B by 4 rows
        sub     rax,16
        jb      .LProcessRemainingCountN4

.LProcessColumnLoop4:
        lea     rbx,[rsi+r9*2]              # compute matrix B plus 2 rows
        vmulps  ymm1,ymm2,YMMWORD PTR [rsi]
        vmulps  ymm6,ymm2,YMMWORD PTR [rsi+32]
        vmulps  ymm8,ymm3,YMMWORD PTR [rsi+r9]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm3,YMMWORD PTR [rsi+r9+32]
        vaddps  ymm6,ymm6,ymm8
        vmulps  ymm8,ymm4,YMMWORD PTR [rbx]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm4,YMMWORD PTR [rbx+32]
        vaddps  ymm6,ymm6,ymm8
        vmulps  ymm8,ymm5,YMMWORD PTR [rbx+r9]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm5,YMMWORD PTR [rbx+r9+32]
        vaddps  ymm6,ymm6,ymm8
        vandnps ymm8,ymm0,YMMWORD PTR [rdx]
        vaddps  ymm1,ymm1,ymm8
        vandnps ymm8,ymm0,YMMWORD PTR [rdx+32]
        vaddps  ymm6,ymm6,ymm8
        vmovups YMMWORD PTR [rdx],ymm1
        vmovups YMMWORD PTR [rdx+32],ymm6
        add     rsi,16*4                    # advance matrix B by 16 columns
        add     rdx,16*4                    # advance matrix C by 16 columns
        sub     rax,16
        jae     .LProcessColumnLoop4

.LProcessRemainingCountN4:
        test    al,15                       # test for unaligned columns
        jz      .LProcessedRemainingCountN4
        test    al,8                        # CountN >= 8?
        jz      .LProcessRemainingCountNSmall4
        lea     rbx,[rsi+r9*2]              # compute matrix B plus 2 rows
        vmulps  ymm1,ymm2,YMMWORD PTR [rsi]
        vmulps  ymm8,ymm3,YMMWORD PTR [rsi+r9]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm4,YMMWORD PTR [rbx]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm5,YMMWORD PTR [rbx+r9]
        vaddps  ymm1,ymm1,ymm8
        vandnps ymm8,ymm0,YMMWORD PTR [rdx]
        vaddps  ymm1,ymm1,ymm8
        vmovups YMMWORD PTR [rdx],ymm1
        add     rsi,8*4                     # advance matrix B by 8 columns
        add     rdx,8*4                     # advance matrix C by 8 columns
        test    al,7
        jz      .LProcessedRemainingCountN4

.LProcessRemainingCountNSmall4:
        lea     rbx,[rsi+r9*2]              # compute matrix B plus 2 rows
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm1,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi+r9]
        vmulps  ymm8,ymm3,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rbx]
        vmulps  ymm8,ymm4,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rbx+r9]
        vmulps  ymm8,ymm5,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vandnps ymm6,ymm0,ymm6
        vaddps  ymm1,ymm1,ymm6
        vmaskmovps YMMWORD PTR [rdx],ymm7,ymm1

.LProcessedRemainingCountN4:
        vxorps  xmm0,xmm0,xmm0              # switch to accumulate mode
        sub     rcx,4
        jae     .LProcessRowLoop4

.LProcessRemainingCountK:
        test    cl,2
        jnz     .LProcessRowLoop2
        test    cl,1
        jnz     .LProcessRowLoop1

.LExitKernel:
        vzeroupper
        pop     rbx
        ret

//
// Process 2 rows of the matrices.
//

.LProcessRowLoop2:
        vbroadcastss ymm2,DWORD PTR [rdi]
        mov     rax,r8                      # reload CountN
        vbroadcastss ymm3,DWORD PTR [rdi+4]
        mov     rsi,r11                     # reload matrix B
        mov     rdx,r10                     # reload matrix C
        add     rdi,2*4                     # advance matrix A by 2 columns
        lea     r11,[rsi+r9*2]              # advance matrix B by 2 rows
        sub     rax,8
        jb      .LProcessRemainingCountN2

.LProcessColumnLoop2:
        vmulps  ymm1,ymm2,YMMWORD PTR [rsi]
        vmulps  ymm8,ymm3,YMMWORD PTR [rsi+r9]
        vaddps  ymm1,ymm1,ymm8
        vandnps ymm6,ymm0,YMMWORD PTR [rdx]
        vaddps  ymm1,ymm1,ymm6
        vmovups YMMWORD PTR [rdx],ymm1
        add     rsi,8*4                     # advance matrix B by 8 columns
        add     rdx,8*4                     # advance matrix C by 8 columns
        sub     rax,8
        jae     .LProcessColumnLoop2

.LProcessRemainingCountN2:
        test    al,7                        # test for unaligned columns
        jz      .LProcessedRemainingCountN2
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm1,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi+r9]
        vmulps  ymm8,ymm3,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vandnps ymm6,ymm0,ymm6
        vaddps  ymm1,ymm1,ymm6
        vmaskmovps YMMWORD PTR [rdx],ymm7,ymm1

.LProcessedRemainingCountN2:
        test    cl,1
        jz      .LExitKernel
        vxorps  xmm0,xmm0,xmm0              # switch to accumulate mode

//
// Process 1 row of the matrices.
//

.LProcessRowLoop1:
        vbroadcastss ymm2,DWORD PTR [rdi]
        mov     rax,r8                      # reload CountN
        mov     rsi,r11                     # reload matrix B
        mov     rdx,r10                     # reload matrix C
        sub     rax,8
        jb      .LProcessRemainingCountN1

.LProcessColumnLoop1:
        vmulps  ymm1,ymm2,YMMWORD PTR [rsi]
        vandnps ymm6,ymm0,YMMWORD PTR [rdx]
        vaddps  ymm1,ymm1,ymm6
        vmovups YMMWORD PTR [rdx],ymm1
        add     rsi,8*4                     # advance matrix B by 8 columns
        add     rdx,8*4                     # advance matrix C by 8 columns
        sub     rax,8
        jae     .LProcessColumnLoop1

.LProcessRemainingCountN1:
        test    al,7                        # test for unaligned columns
        jz      .LExitKernel
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm1,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vandnps ymm6,ymm0,ymm6
        vaddps  ymm1,ymm1,ymm6
        vmaskmovps YMMWORD PTR [rdx],ymm7,ymm1
        jmp     .LExitKernel

        .end
