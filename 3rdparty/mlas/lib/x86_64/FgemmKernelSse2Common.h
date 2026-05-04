/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelSse2Common.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses SSE2 instructions.

--*/

/*++

Macro Description:

    This stores the block accumulators to the output matrix with an optional
    accumulation of the existing contents of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorCount - Supplies the number of vector columns to process.

Implicit Arguments:

    rax - Supplies the length in bytes of a row from matrix C.

    rdx - Supplies the address of matrix C.

    r15 - Stores the ZeroMode argument from the stack frame.

    xmm8-xmm15 - Supplies the block accumulators.

--*/

        .macro AccumulateAndStoreBlock RowCount, VectorCount

        test    r15b,r15b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput\@
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 1, "movupf xmm0,XMMWORD PTR [rdx]"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 2, "movupf xmm1,XMMWORD PTR [rdx+16]"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 3, "movupf xmm2,XMMWORD PTR [rdx+32]"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 4, "movupf xmm3,XMMWORD PTR [rdx+48]"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 1, "movupf xmm4,XMMWORD PTR [rdx+rax]"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "movupf xmm5,XMMWORD PTR [rdx+rax+16]"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "movupf xmm6,XMMWORD PTR [rdx+rax+32]"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "movupf xmm7,XMMWORD PTR [rdx+rax+48]"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 1, "addpf xmm8,xmm0"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 2, "addpf xmm9,xmm1"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 3, "addpf xmm10,xmm2"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 4, "addpf xmm11,xmm3"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 1, "addpf xmm12,xmm4"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "addpf xmm13,xmm5"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "addpf xmm14,xmm6"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "addpf xmm15,xmm7"

.LSkipAccumulateOutput\@:
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 1, "movupf XMMWORD PTR [rdx],xmm8"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 2, "movupf XMMWORD PTR [rdx+16],xmm9"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 3, "movupf XMMWORD PTR [rdx+32],xmm10"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 4, "movupf XMMWORD PTR [rdx+48],xmm11"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 1, "movupf XMMWORD PTR [rdx+rax],xmm12"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "movupf XMMWORD PTR [rdx+rax+16],xmm13"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "movupf XMMWORD PTR [rdx+rax+32],xmm14"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "movupf XMMWORD PTR [rdx+rax+48],xmm15"

        .endm

/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelSse2Function FunctionName

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
        movsf   .LFgemmKernelFrame_alpha[rsp],xmm0

//
// Process CountM rows of the matrices.
//

        cmp     r8,2
        jb      .LProcessCountM1
        mov     r8d,2                       # return 2 rows handled
        ProcessCountM 2, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d
        pop     r15
        pop     rbx
        pop     rbp
        ret

//
// Process 1 row of the matrices.
//

.LProcessCountM1:
        ProcessCountM 1

        .endm
