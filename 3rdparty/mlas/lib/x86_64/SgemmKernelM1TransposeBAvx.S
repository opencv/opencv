/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelM1TransposeBAvx.s

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

    The elements in matrix B are transposed.

Arguments:

    A (rdi) - Supplies the address of matrix A.

    B (rsi) - Supplies the address of matrix B. The elements are transposed.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from matrix A and the number
        of columns from matrix B to iterate over.

    CountN (r8) - Supplies the number of rows from matrix B and the number of
        columns from matrix C to iterate over.

    ldb (r9) - Supplies the first dimension of matrix B.

    Beta (xmm0) - Supplies the scalar beta multiplier (see SGEMM definition).

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasSgemmKernelM1TransposeBAvx

        push    rbx
        shl     r9,2                        # convert ldb to bytes
        mov     r10,rdi
        mov     r11,rsi

//
// Compute the results mask for zeroing or accumulate mode.
//

        vxorps  xmm1,xmm1,xmm1
        vcmpeqss xmm0,xmm1,xmm0
        vshufps xmm0,xmm0,xmm0,0

//
// Compute the conditional load/store mask for an unaligned CountK.
//

        mov     eax,ecx
        and     eax,7
        vmovd   xmm7,eax
        vshufps xmm7,xmm7,xmm7,0
        vpcmpgtd xmm6,xmm7,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip+16]
        vpcmpgtd xmm7,xmm7,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vinsertf128 ymm7,ymm7,xmm6,1

//
// Process 4 rows of the matrices in a loop.
//

        sub     r8,4
        jb      .LProcessRemainingCountN

.LProcessRowLoop4:
        vxorps  xmm2,xmm2,xmm2              # clear row accumulators
        vxorps  xmm3,xmm3,xmm3
        vxorps  xmm4,xmm4,xmm4
        vxorps  xmm5,xmm5,xmm5
        mov     rdi,r10                     # reload matrix A
        mov     rsi,r11                     # reload matrix B
        mov     rax,rcx                     # reload CountK
        lea     r11,[rsi+r9*4]              # advance matrix B by 4 rows
        sub     rax,8
        jb      .LProcessRemainingCountK4

.LProcessColumnLoop4:
        lea     rbx,[rsi+r9*2]              # compute matrix B plus 2 rows
        vmovups ymm1,YMMWORD PTR [rdi]
        vmulps  ymm6,ymm1,YMMWORD PTR [rsi]
        vaddps  ymm2,ymm2,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rsi+r9]
        vaddps  ymm3,ymm3,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rbx]
        vaddps  ymm4,ymm4,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rbx+r9]
        vaddps  ymm5,ymm5,ymm6
        add     rdi,8*4                     # advance matrix A by 8 columns
        add     rsi,8*4                     # advance matrix B by 8 columns
        sub     rax,8
        jae     .LProcessColumnLoop4

.LProcessRemainingCountK4:
        test    al,7                        # test for unaligned columns
        jz      .LOutput4x1Block
        lea     rbx,[rsi+r9*2]              # compute matrix B plus 2 rows
        vmaskmovps ymm1,ymm7,YMMWORD PTR [rdi]
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm2,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi+r9]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm3,ymm3,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rbx]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm4,ymm4,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rbx+r9]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm5,ymm5,ymm6

//
// Reduce and output the row accumulators.
//

.LOutput4x1Block:
        vunpcklps ymm6,ymm2,ymm3            # transpose row accumulators
        vunpckhps ymm1,ymm2,ymm3
        vunpcklps ymm2,ymm4,ymm5
        vunpckhps ymm3,ymm4,ymm5
        vunpcklpd ymm4,ymm6,ymm2
        vunpckhpd ymm5,ymm6,ymm2
        vaddps  ymm4,ymm4,ymm5
        vunpcklpd ymm6,ymm1,ymm3
        vunpckhpd ymm2,ymm1,ymm3
        vaddps  ymm4,ymm4,ymm6
        vaddps  ymm4,ymm4,ymm2
        vextractf128 xmm5,ymm4,1
        vaddps  xmm4,xmm4,xmm5
        vandnps xmm6,xmm0,XMMWORD PTR [rdx]
        vaddps  xmm4,xmm4,xmm6
        vmovups XMMWORD PTR [rdx],xmm4
        add     rdx,4*4                     # advance matrix C by 4 columns
        sub     r8,4
        jae     .LProcessRowLoop4

.LProcessRemainingCountN:
        test    r8d,2
        jnz     .LProcessRowLoop2
        test    r8d,1
        jnz     .LProcessRowLoop1

.LExitKernel:
        vzeroupper
        pop     rbx
        ret

//
// Process 2 rows of the matrices.
//

.LProcessRowLoop2:
        vxorps  xmm2,xmm2,xmm2              # clear row accumulators
        vxorps  xmm3,xmm3,xmm3
        mov     rdi,r10                     # reload matrix A
        mov     rsi,r11                     # reload matrix B
        mov     rax,rcx                     # reload CountK
        lea     r11,[rsi+r9*2]              # advance matrix B by 2 rows
        sub     rax,8
        jb      .LProcessRemainingCountK2

.LProcessColumnLoop2:
        vmovups ymm1,YMMWORD PTR [rdi]
        vmulps  ymm6,ymm1,YMMWORD PTR [rsi]
        vaddps  ymm2,ymm2,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rsi+r9]
        vaddps  ymm3,ymm3,ymm6
        add     rdi,8*4                     # advance matrix A by 8 columns
        add     rsi,8*4                     # advance matrix B by 8 columns
        sub     rax,8
        jae     .LProcessColumnLoop2

.LProcessRemainingCountK2:
        test    al,7                        # test for unaligned columns
        jz      .LOutput2x1Block
        vmaskmovps ymm1,ymm7,YMMWORD PTR [rdi]
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm2,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi+r9]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm3,ymm3,ymm6

//
// Reduce and output the row accumulators.
//

.LOutput2x1Block:
        vunpcklps ymm4,ymm2,ymm3            # reduce row accumulators
        vunpckhps ymm2,ymm2,ymm3
        vaddps  ymm2,ymm2,ymm4
        vextractf128 xmm4,ymm2,1
        vaddps  xmm2,xmm2,xmm4
        vmovhlps xmm4,xmm2,xmm2
        vaddps  xmm2,xmm2,xmm4
        vmovsd  xmm3,QWORD PTR [rdx]
        vandnps xmm3,xmm0,xmm3
        vaddps  xmm2,xmm2,xmm3
        vmovsd  QWORD PTR [rdx],xmm2
        add     rdx,2*4                     # advance matrix C by 2 columns
        test    r8d,1
        jz      .LExitKernel

//
// Process 1 row of the matrices.
//

.LProcessRowLoop1:
        vxorps  xmm2,xmm2,xmm2              # clear row accumulators
        mov     rdi,r10                     # reload matrix A
        mov     rsi,r11                     # reload matrix B
        mov     rax,rcx                     # reload CountK
        sub     rax,8
        jb      .LProcessRemainingCountK1

.LProcessColumnLoop1:
        vmovups ymm1,YMMWORD PTR [rdi]
        vmulps  ymm6,ymm1,YMMWORD PTR [rsi]
        vaddps  ymm2,ymm2,ymm6
        add     rdi,8*4                     # advance matrix A by 8 columns
        add     rsi,8*4                     # advance matrix B by 8 columns
        sub     rax,8
        jae     .LProcessColumnLoop1

.LProcessRemainingCountK1:
        test    al,7                        # test for unaligned columns
        jz      .LOutput1x1Block
        vmaskmovps ymm1,ymm7,YMMWORD PTR [rdi]
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm2,ymm2,ymm6

//
// Reduce and output the row accumulators.
//

.LOutput1x1Block:
        vhaddps ymm2,ymm2,ymm2              # reduce row accumulators
        vhaddps ymm2,ymm2,ymm2
        vextractf128 xmm4,ymm2,1
        vaddss  xmm2,xmm2,xmm4
        vmovss  xmm3,DWORD PTR [rdx]
        vandnps xmm3,xmm0,xmm3
        vaddss  xmm2,xmm2,xmm3
        vmovss  DWORD PTR [rdx],xmm2
        jmp     .LExitKernel

        .end
