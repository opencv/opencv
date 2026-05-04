/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelAvxCommon.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses AVX instructions.

--*/

/*++

Macro Description:

    This macro multiplies and accumulates for 2 YMMWORDs by N rows of the output
    matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

    PrefetchOffset - Optionally supplies the byte offset from matrix B to
        prefetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm8-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy16 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.if \RowCount\() == 1
        vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]
        vmulpf  ymm4,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vaddpf  ymm8,ymm8,ymm4
        vmulpf  ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()+32]
        vaddpf  ymm9,ymm9,ymm5
.else
        vmovapf ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vmovapf ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm4,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm8,ymm8,ymm4"
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm5,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm9,ymm9,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastsf ymm3,[rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm6,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm10,ymm10,ymm6"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm7,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm11,ymm11,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastsf ymm3,[rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm4,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm12,ymm12,ymm4"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm5,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm13,ymm13,ymm5"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastsf ymm3,[rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm6,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm14,ymm14,ymm6"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm7,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm15,ymm15,ymm7"
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for 1 YMMWORD by N rows of the output
    matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

    PrefetchOffset - Optionally supplies the byte offset from matrix B to
        prefetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm8-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy8 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.if \RowCount\() == 1
        vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]
        vmulpf  ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vaddpf  ymm9,ymm9,ymm5
.else
        vmovapf ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm5,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm9,ymm9,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastsf ymm3,[rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm7,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm11,ymm11,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastsf ymm3,[rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm5,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm13,ymm13,ymm5"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastsf ymm3,[rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm7,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm15,ymm15,ymm7"
.endif

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ComputeBlock - Supplies the macro to compute a single block.

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxLoop ComputeBlock, RowCount

.if \RowCount\() > 2
        lea     rbx,[rdi+r10*2]             # compute matrix A plus 2 rows
.endif
        ComputeBlockLoop \ComputeBlock\(), \RowCount\(), \RowCount\() > 2
.if \RowCount\() > 2
        lea     rbx,[rdx+rax*2]             # compute matrix C plus 2 rows
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

        cmp     r9,.LFgemmYmmElementCount
        jbe     .LProcessRemainingCountN\@

.LProcessNextColumnLoop2xN\@:
        EmitIfCountGE \RowCount\(), 1, "vxorpf xmm8,xmm8,xmm8"
        EmitIfCountGE \RowCount\(), 1, "vxorpf xmm9,xmm9,xmm9"
        EmitIfCountGE \RowCount\(), 2, "vxorpf xmm10,xmm10,xmm10"
        EmitIfCountGE \RowCount\(), 2, "vxorpf xmm11,xmm11,xmm11"
        EmitIfCountGE \RowCount\(), 3, "vxorpf xmm12,xmm12,xmm12"
        EmitIfCountGE \RowCount\(), 3, "vxorpf xmm13,xmm13,xmm13"
        EmitIfCountGE \RowCount\(), 4, "vxorpf xmm14,xmm14,xmm14"
        EmitIfCountGE \RowCount\(), 4, "vxorpf xmm15,xmm15,xmm15"
        ComputeBlockAvxLoop ComputeBlockAvxBy16, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm8,ymm8,ymm2"
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm10,ymm10,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm12,ymm12,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm14,ymm14,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm15,ymm15,ymm2"
        sub     r9,2*.LFgemmYmmElementCount
        jb      .LOutputMasked2xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStore2xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm8,ymm8,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm9,ymm9,YMMWORD PTR [rdx+32]"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm10,ymm10,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm11,ymm11,YMMWORD PTR [rdx+rax+32]"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm12,ymm12,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm13,ymm13,YMMWORD PTR [rbx+32]"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm14,ymm14,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm15,ymm15,YMMWORD PTR [rbx+rax+32]"

.LStore2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx],ymm8"
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx+32],ymm9"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax],ymm10"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax+32],ymm11"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rbx],ymm12"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rbx+32],ymm13"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx+rax],ymm14"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx+rax+32],ymm15"
        add     rdx,2*32                    # advance matrix C by 2 YMMWORDs
        mov     rdi,r11                     # reload matrix A
        cmp     r9,.LFgemmYmmElementCount
        ja      .LProcessNextColumnLoop2xN\@
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN\@:
        EmitIfCountGE \RowCount\(), 1, "vxorpf xmm9,xmm9,xmm9"
        EmitIfCountGE \RowCount\(), 2, "vxorpf xmm11,xmm11,xmm11"
        EmitIfCountGE \RowCount\(), 3, "vxorpf xmm13,xmm13,xmm13"
        EmitIfCountGE \RowCount\(), 4, "vxorpf xmm15,xmm15,xmm15"
        ComputeBlockAvxLoop ComputeBlockAvxBy8, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm15,ymm15,ymm2"
        cmp     r9,.LFgemmYmmElementCount
        jb      .LOutputMasked1xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStore1xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm9,ymm9,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm11,ymm11,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm13,ymm13,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm15,ymm15,YMMWORD PTR [rbx+rax]"

.LStore1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx],ymm9"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax],ymm11"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rbx],ymm13"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx+rax],ymm15"
        jmp     .LExitKernel

.LOutputMasked2xNBlock\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStoreMasked2xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm8,ymm8,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm10,ymm10,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm12,ymm12,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm14,ymm14,YMMWORD PTR [rbx+rax]"

.LStoreMasked2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx],ymm8"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax],ymm10"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rbx],ymm12"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx+rax],ymm14"
        add     rdx,32                      # advance matrix C by YMMWORD
.if \RowCount\() > 2
        add     rbx,32                      # advance matrix C plus 2 rows by YMMWORD
.endif
        add     r9,.LFgemmYmmElementCount   # correct for over-subtract above

.LOutputMasked1xNBlock\@:
        neg     r9
        lea     rdi,C_UNDERSCORE(MlasMaskMoveTableAvx)[rip+8*4]
        vmovdqu ymm0,YMMWORD PTR [rdi+r9*.LFgemmElementSize]
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStoreMasked1xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vmaskmovpf ymm8,ymm0,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovpf ymm10,ymm0,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovpf ymm12,ymm0,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovpf ymm14,ymm0,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 1, "vaddpf ymm9,ymm9,ymm8"
        EmitIfCountGE \RowCount\(), 2, "vaddpf ymm11,ymm11,ymm10"
        EmitIfCountGE \RowCount\(), 3, "vaddpf ymm13,ymm13,ymm12"
        EmitIfCountGE \RowCount\(), 4, "vaddpf ymm15,ymm15,ymm14"

.LStoreMasked1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmaskmovpf YMMWORD PTR [rdx],ymm0,ymm9"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovpf YMMWORD PTR [rdx+rax],ymm0,ymm11"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovpf YMMWORD PTR [rbx],ymm0,ymm13"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovpf YMMWORD PTR [rbx+rax],ymm0,ymm15"
.ifb \Fallthrough\()
        jmp     .LExitKernel
.endif

        .endm

/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelAvxFunction FunctionName

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    Alpha (xmm0) - Supplies the scalar alpha multiplier (see GEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY \FunctionName\()

        push    rbp
        push    rbx
        push    r15
        mov     r11,rdi
        mov     r10,.LFgemmKernelFrame_lda[rsp]
        shl     r10,.LFgemmElementShift     # convert lda to bytes
        mov     rax,.LFgemmKernelFrame_ldc[rsp]
        shl     rax,.LFgemmElementShift     # convert ldc to bytes
        movzx   r15,BYTE PTR .LFgemmKernelFrame_ZeroMode[rsp]
        vmovsf  .LFgemmKernelFrame_alpha[rsp],xmm0
        vbroadcastsf ymm2,.LFgemmKernelFrame_alpha[rsp]

//
// Process 4 rows of the matrices.
//

        cmp     r8,4
        jb      .LProcessCountMLessThan4
        mov     r8d,4                      # return 4 rows handled
        ProcessCountM 4, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        vzeroupper
        mov     eax,r8d
        pop     r15
        pop     rbx
        pop     rbp
        ret

//
// Process 2 rows of the matrices.
//

.LProcessCountMLessThan4:
        cmp     r8,2
        jb      .LProcessCountMLessThan2
        mov     r8d,2                       # return 2 rows handled
        ProcessCountM 2

//
// Process 1 row of the matrices.
//

.LProcessCountMLessThan2:
        ProcessCountM 1

        .endm
