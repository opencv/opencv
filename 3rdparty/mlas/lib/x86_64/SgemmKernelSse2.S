/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelSse2.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses SSE2 instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"
#include "FgemmKernelSse2Common.h"

        .intel_syntax noprefix

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    Shuffle - Supplies the shuffle mask to extract the element from matrix A.

Implicit Arguments:

    rsi - Supplies the address into the matrix B data.

    xmm0-xmm1 - Supplies up to four elements loaded from matrix A and matrix A
        plus one row.

    xmm8-xmm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseBy16 RowCount, VectorOffset, Shuffle

        movaps  xmm4,XMMWORD PTR [rsi+\VectorOffset\()]
        movaps  xmm5,XMMWORD PTR [rsi+\VectorOffset\()+16]
        pshufd  xmm2,xmm0,\Shuffle\()
.if \RowCount\() == 2
        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
.endif
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm8,xmm4
        addps   xmm9,xmm5
.if \RowCount\() == 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm12,xmm6
        addps   xmm13,xmm7
.endif
        movaps  xmm4,XMMWORD PTR [rsi+\VectorOffset\()+32]
        movaps  xmm5,XMMWORD PTR [rsi+\VectorOffset\()+48]
.if \RowCount\() == 2
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
.endif
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm10,xmm4
        addps   xmm11,xmm5
.if \RowCount\() == 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm14,xmm6
        addps   xmm15,xmm7
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

    Fallthrough - Supplies a non-blank value if the macro may fall through to
        the ExitKernel label.

Implicit Arguments:

    rdi - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    r11 - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rdx - Supplies the address of matrix C.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    rax - Supplies the length in bytes of a row from matrix C.

    r15 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountM RowCount, Fallthrough

.LProcessNextColumnLoop16xN\@:
        EmitIfCountGE \RowCount\(), 1, "xorps xmm8,xmm8"
        EmitIfCountGE \RowCount\(), 1, "xorps xmm9,xmm9"
        EmitIfCountGE \RowCount\(), 1, "xorps xmm10,xmm10"
        EmitIfCountGE \RowCount\(), 1, "xorps xmm11,xmm11"
        EmitIfCountGE \RowCount\(), 2, "xorps xmm12,xmm12"
        EmitIfCountGE \RowCount\(), 2, "xorps xmm13,xmm13"
        EmitIfCountGE \RowCount\(), 2, "xorps xmm14,xmm14"
        EmitIfCountGE \RowCount\(), 2, "xorps xmm15,xmm15"
        mov     rbp,rcx                     # reload CountK
        sub     rbp,4
        jb      .LProcessRemaining16xNBlocks\@

.LCompute16xNBlockBy4Loop\@:
        EmitIfCountGE \RowCount\(), 1, "movups xmm0,XMMWORD PTR [rdi]"
        EmitIfCountGE \RowCount\(), 2, "movups xmm1,XMMWORD PTR [rdi+r10]"
        ComputeBlockSseBy16 2, 0, 0x00
        ComputeBlockSseBy16 2, 16*4, 0x55
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        ComputeBlockSseBy16 2, 0, 0xAA
        ComputeBlockSseBy16 2, 16*4, 0xFF
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        add     rdi,4*4                     # advance matrix A by 4 columns
        sub     rbp,4
        jae     .LCompute16xNBlockBy4Loop\@

.LProcessRemaining16xNBlocks\@:
        add     rbp,4                       # correct for over-subtract above
        jz      .LOutput16xNBlock\@

.LCompute16xNBlockBy1Loop\@:
        EmitIfCountGE \RowCount\(), 1, "movss xmm0,[rdi]"
        EmitIfCountGE \RowCount\(), 2, "movss xmm1,[rdi+r10]"
        ComputeBlockSseBy16 2, 0, 0x00
        add     rsi,16*4                    # advance matrix B by 16 columns
        add     rdi,4                       # advance matrix A by 1 column
        dec     rbp
        jne     .LCompute16xNBlockBy1Loop\@

.LOutput16xNBlock\@:
        movss   xmm2,.LFgemmKernelFrame_alpha[rsp]
        shufps  xmm2,xmm2,0
        EmitIfCountGE \RowCount\(), 1, "mulps xmm8,xmm2"
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 1, "mulps xmm9,xmm2"
        EmitIfCountGE \RowCount\(), 1, "mulps xmm10,xmm2"
        EmitIfCountGE \RowCount\(), 1, "mulps xmm11,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulps xmm12,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulps xmm13,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulps xmm14,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulps xmm15,xmm2"
        sub     r9,16
        jb      .LOutputPartial16xNBlock\@
        AccumulateAndStoreBlock \RowCount\(), 4
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        test    r9,r9
        jnz     .LProcessNextColumnLoop16xN\@
        jmp     .LExitKernel

//
// Output a partial 16xN block to the matrix.
//

.LOutputPartial16xNBlock\@:
        add     r9,16                       # correct for over-subtract above
        cmp     r9,4
        jb      .LOutputPartialLessThan4xNBlock\@
        cmp     r9,8
        jb      .LOutputPartialLessThan8xNBlock\@
        cmp     r9,12
        jb      .LOutputPartialLessThan12xNBlock\@
        AccumulateAndStoreBlock \RowCount\(), 3
        and     r9d,3                       # check if remaining count is small
        jz      .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "movaps xmm8,xmm11"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "movaps xmm12,xmm15"
        add     rdx,12*4                    # advance matrix C by 12 columns
        jmp     .LOutputPartialLessThan4xNBlock\@

.LOutputPartialLessThan12xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 2
        and     r9d,3                       # check if remaining count is small
        jz      .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "movaps xmm8,xmm10"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "movaps xmm12,xmm14"
        add     rdx,8*4                     # advance matrix C by 8 columns
        jmp     .LOutputPartialLessThan4xNBlock\@

.LOutputPartialLessThan8xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 1
        and     r9d,3                       # check if remaining count is small
        jz      .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "movaps xmm8,xmm9"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "movaps xmm12,xmm13"
        add     rdx,4*4                     # advance matrix C by 4 columns

.LOutputPartialLessThan4xNBlock\@:
        test    r9d,2
        jz      .LOutputPartial1xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput2xN\@
        EmitIfCountGE \RowCount\(), 1, "movsd xmm0,QWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "movsd xmm1,QWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 1, "addps xmm8,xmm0"
        EmitIfCountGE \RowCount\(), 2, "addps xmm12,xmm1"

.LSkipAccumulateOutput2xN\@:
        EmitIfCountGE \RowCount\(), 1, "movsd QWORD PTR [rdx],xmm8"
        EmitIfCountGE \RowCount\(), 2, "movsd QWORD PTR [rdx+rax],xmm12"
        test    r9d,1                       # check if remaining count is odd
        jz      .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "movhlps xmm8,xmm8"
                                            # shift third element down
        EmitIfCountGE \RowCount\(), 2, "movhlps xmm12,xmm12"
        add     rdx,2*4                     # advance matrix C by 2 columns

.LOutputPartial1xNBlock\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput1xN\@
        EmitIfCountGE \RowCount\(), 1, "addss xmm8,[rdx]"
        EmitIfCountGE \RowCount\(), 2, "addss xmm12,[rdx+rax]"

.LSkipAccumulateOutput1xN\@:
        EmitIfCountGE \RowCount\(), 1, "movss [rdx],xmm8"
        EmitIfCountGE \RowCount\(), 2, "movss [rdx+rax],xmm12"
.ifb \Fallthrough\()
        jmp     .LExitKernel
.endif

        .endm

//
// Generate the GEMM kernel.
//

FgemmKernelSse2Function MlasGemmFloatKernelSse

        .end
