/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the floating
    point matrix/matrix multiply operation (SGEMM and DGEMM).

--*/

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

    a0 - Supplies the address into the matrix A data.

    t7 - Supplies the address into the matrix A data plus 3 rows.

    a1 - Supplies the address into the matrix B data.

    a3 - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    vr4-vr15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoop ComputeBlock, RowCount, AdvanceMatrixAPlusRows

        move     $t8, $a3                     # reload CountK
        li.d    $s0, 4
        blt     $t8, $s0, .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        \ComputeBlock\() \RowCount\(), 0, LFgemmElementSize*0, 64*4
        \ComputeBlock\() \RowCount\(), 2*32, LFgemmElementSize*1, 64*4
        addi.d $a1, $a1, 2*2*32                # advance matrix B by 128 bytes
        \ComputeBlock\() \RowCount\(), 0, LFgemmElementSize*2, 64*4
        \ComputeBlock\() \RowCount\(), 2*32, LFgemmElementSize*3, 64*4
        addi.d  $a1, $a1, 2*2*32                # advance matrix B by 128 bytes
        addi.d  $a0, $a0, 4*LFgemmElementSize    # advance matrix A by 4 elements
.if \RowCount\() > 3
        addi.d     $t7, $t7, 4*LFgemmElementSize    # advance matrix A plus rows by 4 elements
.if \RowCount\() == 12
        addi.d     $t3, $t3, 4*LFgemmElementSize
        addi.d     $t4,, $t4, 4*LFgemmElementSize
.endif
.endif
        addi.d     $t8, $t8, -4
        li.d        $s0, 4
        bge     $t8, $s0, .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        beqz    $t8,      .LOutputBlock\@

.LComputeBlockBy1Loop\@:
        \ComputeBlock\() \RowCount\(), 0, 0
        addi.d     $a1, $a1, 2*32                    # advance matrix B by 64 bytes
        addi.d     $a0, $a0, LFgemmElementSize      # advance matrix A by 1 element
.if \RowCount\() > 3
        addi.d     $t7, $t7, LFgemmElementSize      # advance matrix A plus rows by 1 element
.if \RowCount\() == 12
        addi.d     $t3, $t3, LFgemmElementSize
        addi.d     $t4, $t4, LFgemmElementSize
.endif
.endif
        addi.d     $t8, $t8, -1
        bnez    $t8,     .LComputeBlockBy1Loop\@

.LOutputBlock\@:

        .endm
