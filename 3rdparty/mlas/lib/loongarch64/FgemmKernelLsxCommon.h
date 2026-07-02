/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelLsxCommon.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses Lsx instructions.

--*/

#include "FgemmKernelCommon.h"
/*++

Macro Description:

    This stores the block accumulators to the output matrix with an optional
    accumulation of the existing contents of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorCount - Supplies the number of vector columns to process.

Implicit Arguments:

    t5 - Supplies the length in bytes of a row from matrix C.

    a2 - Supplies the address of matrix C.

    s3 - Stores the ZeroMode argument from the stack frame.

    vr8-vr15 - Supplies the block accumulators.

--*/

        .macro AccumulateAndStoreBlock RowCount, VectorCount

        and    $s0, $t5,$t5                   # ZeroMode?
        bnez    $s0 , .LSkipAccumulateOutput\@
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 1, "vld $vr0, $a2, 0"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 2, "vld $vr1, $a2, 16"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 3, "vld $vr2, $a2, 32"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 4, "vld $vr3, $a2, 48"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 1, "vldx $vr4, $a2, $t6"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "addi.d $s0, $t6, 16"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "vldx $vr5, $a2, $s0"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "addi.d $s0, $t6, 32"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "vldx $vr6, $a2, $s0"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "addi.d $s0, $t6, 48"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "vldx $vr7, $a2, $s0"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 1, "vfadd $vr8, $vr8, $vr0"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 2, "vfadd $vr9, $vr9, $vr1"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 3, "vfadd $vr10,$vr10,$vr2"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 4, "vfadd $vr11,$vr11,$vr3"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 1, "vfadd $vr12,$vr12,$vr4"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "vfadd $vr13,$vr13,$vr5"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "vfadd $vr14,$vr14,$vr6"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "vfadd $vr15,$vr15,$vr7"

.LSkipAccumulateOutput\@:
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 1, "vst $vr8, $a2, 0"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 2, "vst $vr9,  $a2, 16"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 3, "vst $vr10, $a2, 32"
        EmitIfCount2GE \RowCount\(), 1, \VectorCount\(), 4, "vst $vr11, $a2, 48"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 1, "vstx $vr12, $a2, $t6"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "addi.d $s0, $t6, 16"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 2, "vstx $vr13, $a2, $s0"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "addi.d $s0, $t6, 32"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 3, "vstx $vr14, $a2, $s0"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "addi.d $s0, $t6, 48"
        EmitIfCount2GE \RowCount\(), 2, \VectorCount\(), 4, "vstx $vr15, $a2, $s0"

        .endm
/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelLsxFunction FunctionName

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (a0) - Supplies the address of matrix A.

    B (a1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (a2) - Supplies the address of matrix C.

    CountK (a3) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (a4) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (a5) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda (a6) Supplies the first dimension of matrix A.

    ldc (a7) Supplies the first dimension of matrix C.

    Alpha (f0) - Supplies the scalar alpha multiplier (see GEMM definition).

    ZeroMode (sp 0) - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

FUNCTION_ENTRY \FunctionName\()
    addi.d  $sp, $sp, -64
    st.d    $t5, $sp, 0
    st.d    $s0, $sp, 1*8
    st.d    $s1, $sp, 2*8
    st.d    $s2, $sp, 3*8
    st.d    $s3, $sp, 4*8
    move    $t1, $a0
    slli.d  $t0, $a6, 2   //convert lda to bytes
    slli.d  $t6, $a7, 2   //convert ldc to bytes
    ld.d    $t5, $sp, 64
    fmov.s    $f24, $f0     //f0 destroyed by lsx

    li.d    $s0, 2
    blt     $a4, $s0, .LProcessCountM1

    li.d    $a4, 2
    ProcessCountM 2, Fallthrough

.LExitKernel:
    ld.d    $t5, $sp, 0
    ld.d    $s0, $sp, 1*8
    ld.d    $s1, $sp, 2*8
    ld.d    $s2, $sp, 3*8
    ld.d    $s3, $sp, 4*8
    addi.d  $sp, $sp, 64
    move    $a0, $a4
    jr      $ra

.LProcessCountM1:
    ProcessCountM 1
    .endm
