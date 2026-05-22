/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the floating
    point matrix/matrix multiply operation (SGEMM and DGEMM).

--*/

//
// Stack frame layout for the floating point kernels.
//

        .equ    .LFgemmKernelFrame_SavedR12, -32
        .equ    .LFgemmKernelFrame_SavedR13, -24
        .equ    .LFgemmKernelFrame_SavedR14, -16
        .equ    .LFgemmKernelFrame_alpha, -8
        .equ    .LFgemmKernelFrame_SavedR15, 0
        .equ    .LFgemmKernelFrame_SavedRbx, 8
        .equ    .LFgemmKernelFrame_SavedRbp, 16
        .equ    .LFgemmKernelFrame_ReturnAddress, 24
        .equ    .LFgemmKernelFrame_lda, 32
        .equ    .LFgemmKernelFrame_ldc, 40
        .equ    .LFgemmKernelFrame_ZeroMode, 48

//
// Define the number of elements per vector register.
//

        .equ    .LFgemmXmmElementCount, 16 / .LFgemmElementSize
        .equ    .LFgemmYmmElementCount, 32 / .LFgemmElementSize
        .equ    .LFgemmZmmElementCount, 64 / .LFgemmElementSize

//
// Define the typed instruction template.
//

#define FGEMM_TYPED_INSTRUCTION(Untyped, Typed) \
        .macro Untyped Operand:vararg; Typed \Operand\(); .endm;

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ComputeBlock - Supplies the macro to compute a single block.

    RowCount - Supplies the number of rows to process.

    AdvanceMatrixAPlusRows - Supplies a non-zero value if the data pointer
        in rbx should also be advanced as part of the loop.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoop ComputeBlock, RowCount, AdvanceMatrixAPlusRows

        mov     rbp,rcx                     # reload CountK
        sub     rbp,4
        jb      .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        \ComputeBlock\() \RowCount\(), 0, .LFgemmElementSize*0, 64*4
        \ComputeBlock\() \RowCount\(), 2*32, .LFgemmElementSize*1, 64*4
        add_immed rsi,2*2*32                # advance matrix B by 128 bytes
        \ComputeBlock\() \RowCount\(), 0, .LFgemmElementSize*2, 64*4
        \ComputeBlock\() \RowCount\(), 2*32, .LFgemmElementSize*3, 64*4
        add_immed rsi,2*2*32                # advance matrix B by 128 bytes
        add     rdi,4*.LFgemmElementSize    # advance matrix A by 4 elements
.if \RowCount\() > 3
        add     rbx,4*.LFgemmElementSize    # advance matrix A plus rows by 4 elements
.if \RowCount\() == 12
        add     r13,4*.LFgemmElementSize
        add     r14,4*.LFgemmElementSize
.endif
.endif
        sub     rbp,4
        jae     .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        add     rbp,4                       # correct for over-subtract above
        jz      .LOutputBlock\@

.LComputeBlockBy1Loop\@:
        \ComputeBlock\() \RowCount\(), 0, 0
        add     rsi,2*32                    # advance matrix B by 64 bytes
        add     rdi,.LFgemmElementSize      # advance matrix A by 1 element
.if \RowCount\() > 3
        add     rbx,.LFgemmElementSize      # advance matrix A plus rows by 1 element
.if \RowCount\() == 12
        add     r13,.LFgemmElementSize
        add     r14,.LFgemmElementSize
.endif
.endif
        dec     rbp
        jne     .LComputeBlockBy1Loop\@

.LOutputBlock\@:

        .endm
