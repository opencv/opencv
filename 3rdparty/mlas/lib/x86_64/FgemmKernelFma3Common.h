/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelFma3Common.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses AVX fused multiply/add instructions.

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

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3By2 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
.endif
.if \RowCount\() == 1
        vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]
        vfmadd231pf ymm4,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231pf ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()+32]
.else
        vmovapf ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vmovapf ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231pf ymm4,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231pf ymm5,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastsf ymm3,[rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231pf ymm6,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231pf ymm7,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastsf ymm3,[rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231pf ymm8,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231pf ymm9,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastsf ymm3,[rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231pf ymm10,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231pf ymm11,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 5, "vbroadcastsf ymm3,[rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231pf ymm12,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231pf ymm13,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 6, "vbroadcastsf ymm3,[rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231pf ymm14,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231pf ymm15,ymm3,ymm1"
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

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3By1 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
.endif
.if \RowCount\() == 1
        vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]
        vfmadd231pf ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
.else
        vmovapf ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastsf ymm3,[rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231pf ymm5,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastsf ymm3,[rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231pf ymm7,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastsf ymm3,[rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231pf ymm9,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastsf ymm3,[rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231pf ymm11,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 5, "vbroadcastsf ymm3,[rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231pf ymm13,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 6, "vbroadcastsf ymm3,[rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231pf ymm15,ymm3,ymm0"
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

        .macro ComputeBlockFma3Loop ComputeBlock, RowCount

.if \RowCount\() > 3
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif
        ComputeBlockLoop \ComputeBlock\(), \RowCount\(), \RowCount\() > 3
        vbroadcastsf ymm2,[rsp+.LFgemmKernelFrame_alpha]
.if \RowCount\() > 3
        lea     rbx,[rax*2+rax]
        add     rbx,rdx                     # compute matrix C plus 3 rows
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

    Fallthrough - Supplies a non-blank value if the macro may fall through to
        the ExitKernelAndZeroUpper label.

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
        ComputeBlockFma3Loop ComputeBlockFma3By2, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "prefetcht0 [rdx+64]"
        EmitIfCountGE \RowCount\(), 2, "prefetcht0 [rdx+rax+64]"
        EmitIfCountGE \RowCount\(), 3, "prefetcht0 [rdx+rax*2+64]"
        EmitIfCountGE \RowCount\(), 4, "prefetcht0 [rbx+64]"
        EmitIfCountGE \RowCount\(), 5, "prefetcht0 [rbx+rax+64]"
        EmitIfCountGE \RowCount\(), 6, "prefetcht0 [rbx+rax*2+64]"
        sub     r9,2*.LFgemmYmmElementCount
        jb      .LOutputMasked2xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha2xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf ymm4,ymm2,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf ymm5,ymm2,YMMWORD PTR [rdx+32]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf ymm6,ymm2,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf ymm7,ymm2,YMMWORD PTR [rdx+rax+32]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf ymm8,ymm2,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf ymm9,ymm2,YMMWORD PTR [rdx+rax*2+32]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf ymm10,ymm2,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf ymm11,ymm2,YMMWORD PTR [rbx+32]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf ymm12,ymm2,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf ymm13,ymm2,YMMWORD PTR [rbx+rax+32]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf ymm14,ymm2,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf ymm15,ymm2,YMMWORD PTR [rbx+rax*2+32]"
        jmp     .LStore2xNBlock\@

.LMultiplyAlpha2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm4,ymm4,ymm2"
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm5,ymm5,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm6,ymm6,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm7,ymm7,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm8,ymm8,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm10,ymm10,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulpf ymm12,ymm12,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulpf ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulpf ymm14,ymm14,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulpf ymm15,ymm15,ymm2"

.LStore2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx+32],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax+32],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rdx+rax*2+32],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx+32],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovupf YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 5, "vmovupf YMMWORD PTR [rbx+rax+32],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovupf YMMWORD PTR [rbx+rax*2],ymm14"
        EmitIfCountGE \RowCount\(), 6, "vmovupf YMMWORD PTR [rbx+rax*2+32],ymm15"
        add     rdx,2*32                    # advance matrix C by 2 YMMWORDs
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,.LFgemmYmmElementCount
        ja      .LProcessNextColumnLoop2xN\@
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN\@:
        ComputeBlockFma3Loop ComputeBlockFma3By1, \RowCount\()
        cmp     r9,.LFgemmYmmElementCount
        jb      .LOutputMasked1xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha1xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf ymm5,ymm2,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf ymm7,ymm2,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf ymm9,ymm2,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf ymm11,ymm2,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf ymm13,ymm2,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf ymm15,ymm2,YMMWORD PTR [rbx+rax*2]"
        jmp     .LStore1xNBlock\@

.LMultiplyAlpha1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm5,ymm5,ymm2"
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm7,ymm7,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulpf ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulpf ymm15,ymm15,ymm2"

.LStore1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rdx+rax*2],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovupf YMMWORD PTR [rbx+rax],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovupf YMMWORD PTR [rbx+rax*2],ymm15"
        jmp     .LExitKernelAndZeroUpper

.LOutputMasked2xNBlock\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlphaMasked2xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf ymm4,ymm2,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf ymm6,ymm2,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf ymm8,ymm2,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf ymm10,ymm2,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf ymm12,ymm2,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf ymm14,ymm2,YMMWORD PTR [rbx+rax*2]"
        jmp     .LStoreMasked2xNBlock\@

.LMultiplyAlphaMasked2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm4,ymm4,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm6,ymm6,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm8,ymm8,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm10,ymm10,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulpf ymm12,ymm12,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulpf ymm14,ymm14,ymm2"

.LStoreMasked2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 2, "vmovupf YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 3, "vmovupf YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 4, "vmovupf YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 5, "vmovupf YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 6, "vmovupf YMMWORD PTR [rbx+rax*2],ymm14"
        add     rdx,32                      # advance matrix C by YMMWORD
.if \RowCount\() > 3
        add     rbx,32                      # advance matrix C plus 3 rows by YMMWORD
.endif
        add     r9,.LFgemmYmmElementCount   # correct for over-subtract above

.LOutputMasked1xNBlock\@:
        neg     r9
        lea     rdi,C_UNDERSCORE(MlasMaskMoveTableAvx)[rip+8*4]
        vmovdqu ymm0,YMMWORD PTR [rdi+r9*.LFgemmElementSize]
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlphaMasked1xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vmaskmovpf ymm4,ymm0,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovpf ymm6,ymm0,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovpf ymm8,ymm0,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovpf ymm10,ymm0,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vmaskmovpf ymm12,ymm0,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vmaskmovpf ymm14,ymm0,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf ymm5,ymm2,ymm4"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf ymm7,ymm2,ymm6"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf ymm9,ymm2,ymm8"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf ymm11,ymm2,ymm10"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf ymm13,ymm2,ymm12"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf ymm15,ymm2,ymm14"
        jmp     .LStoreMasked1xNBlock\@

.LMultiplyAlphaMasked1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf ymm5,ymm5,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulpf ymm7,ymm7,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulpf ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulpf ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulpf ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulpf ymm15,ymm15,ymm2"

.LStoreMasked1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmaskmovpf YMMWORD PTR [rdx],ymm0,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovpf YMMWORD PTR [rdx+rax],ymm0,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovpf YMMWORD PTR [rdx+rax*2],ymm0,ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovpf YMMWORD PTR [rbx],ymm0,ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmaskmovpf YMMWORD PTR [rbx+rax],ymm0,ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmaskmovpf YMMWORD PTR [rbx+rax*2],ymm0,ymm15"
.ifb \Fallthrough\()
        jmp     .LExitKernelAndZeroUpper
.endif

        .endm

/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelFma3Function FunctionName

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
        vzeroall

//
// Process CountM rows of the matrices.
//

        cmp     r8,5
        ja      .LProcessCountM6
        je      .LProcessCountM5
        cmp     r8,3
        ja      .LProcessCountM4
        je      .LProcessCountM3
        cmp     r8,1
        je      .LProcessCountM1

.LProcessCountM2:
        ProcessCountM 2

.LProcessCountM4:
        ProcessCountM 4

.LProcessCountM6:
        mov     r8d,6                       # return 6 rows handled
        ProcessCountM 6, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernelAndZeroUpper:
        vzeroupper

.LExitKernel:
        mov     eax,r8d
        pop     r15
        pop     rbx
        pop     rbp
        ret

.LProcessCountM1:
        ProcessCountM 1

.LProcessCountM3:
        ProcessCountM 3

.LProcessCountM5:
        ProcessCountM 5

        .endm
