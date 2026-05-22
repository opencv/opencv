/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelLsx.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses Lsx instructions.

--*/

#include "asmmacro.h"
#include "FgemmKernelLsxCommon.h"

FGEMM_TYPED_INSTRUCTION(vfadd, vfadd.s)

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    Shuffle - Supplies the shuffle mask to extract the element from matrix A.

Implicit Arguments:

    a1 - Supplies the address into the matrix B data.

    vr0-vr1 - Supplies up to four elements loaded from matrix A and matrix A
        plus one row.

    vr8-vr15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseBy16 RowCount, VectorOffset, Shuffle
        vld     $vr4, $a1, \VectorOffset
        vld     $vr5, $a1, \VectorOffset + 16
        vreplvei.w   $vr2, $vr0, \Shuffle
.if \RowCount\() == 2
        vreplvei.w   $vr3, $vr1, \Shuffle
        vmove   $vr6, $vr4
        vmove   $vr7, $vr5
.endif
        vfmadd.s $vr8, $vr4, $vr2, $vr8
        vfmadd.s $vr9, $vr5, $vr2, $vr9
.if \RowCount\() == 2
        vfmadd.s $vr12, $vr6, $vr3, $vr12
        vfmadd.s $vr13, $vr7, $vr3, $vr13
.endif
        vld     $vr4, $a1,  \VectorOffset + 32
        vld     $vr5, $a1,  \VectorOffset + 48
.if \RowCount\() == 2
        vmove   $vr6, $vr4
        vmove   $vr7, $vr5
.endif
        vfmadd.s $vr10, $vr4, $vr2, $vr10
        vfmadd.s $vr11, $vr5, $vr2, $vr11
.if \RowCount\() == 2
        vfmadd.s $vr14, $vr6, $vr3, $vr14
        vfmadd.s $vr15, $vr7, $vr3, $vr15
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

    a0 - Supplies the address of matrix A.

    a1 - Supplies the address of matrix B.

    t8 - Supplies the address of matrix A.

    a5 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    a2 - Supplies the address of matrix C.

    a3 - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    t7 - Supplies the length in bytes of a row from matrix A.

    t5 - Supplies the length in bytes of a row from matrix C.

    s3 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountM RowCount, Fallthrough
.LProcessNextColumnLoop16xN\@:
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr8, $vr8,$vr8"
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr9, $vr9,$vr9"
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr10, $vr10,$vr10"
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr11, $vr11,$vr11"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr12, $vr12,$vr12"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr13, $vr13,$vr13"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr14, $vr14,$vr14"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr15, $vr15,$vr15"
        move    $t8, $a3
        li.d    $s0, 4
        blt     $t8, $s0, .LProcessRemaining16xNBlocks\@
.LCompute16xNBlockBy4Loop\@:
        EmitIfCountGE \RowCount\(), 1, "vld $vr0, $a0, 0"
        EmitIfCountGE \RowCount\(), 2, "vldx $vr1, $a0, $t0"    #second line of A
        ComputeBlockSseBy16 2, 0, 0x0
        ComputeBlockSseBy16 2, 16*4, 0x1
        addi.d  $a1, $a1, 32*4                 # advance matrix B by 32 columns
        ComputeBlockSseBy16 2, 0, 0x2
        ComputeBlockSseBy16 2, 16*4, 0x3
        addi.d  $a1, $a1, 32*4                 # advance matrix B by 32 columns
        addi.d  $a0, $a0, 4*4                   # advance matrix A by 4 columns
        addi.d  $t8, $t8, -4
        li.d    $s0, 4                          #check matrix A remaining less than 4
        bge     $t8, $s0, .LCompute16xNBlockBy4Loop\@

.LProcessRemaining16xNBlocks\@:
        beqz    $t8, .LOutput16xNBlock\@

.LCompute16xNBlockBy1Loop\@:
        EmitIfCountGE \RowCount\(), 1, "ld.w $s0, $a0, 0"
        EmitIfCountGE \RowCount\(), 1, "vinsgr2vr.w $vr0, $s0, 0"
        EmitIfCountGE \RowCount\(), 2, "ldx.w $s0,$a0, $t0"
        EmitIfCountGE \RowCount\(), 2, "vinsgr2vr.w $vr1,$s0, 0"
        ComputeBlockSseBy16 2, 0, 0x00
        addi.d  $a1, $a1, 16*4      #advance matrix B by 16 columns
        addi.d  $a0, $a0, 1*4       #advance matrix A by 1 column
        addi.d  $t8, $t8, -1
        bnez    $t8, .LCompute16xNBlockBy1Loop\@

.LOutput16xNBlock\@:
        movfr2gr.s      $s0,  $f24
        vreplgr2vr.w    $vr2, $s0
        EmitIfCountGE \RowCount\(), 1, "vfmul.s $vr8,$vr8,$vr2"
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 1, "vfmul.s $vr9,$vr9,$vr2"
        EmitIfCountGE \RowCount\(), 1, "vfmul.s $vr10,$vr10,$vr2"
        EmitIfCountGE \RowCount\(), 1, "vfmul.s $vr11,$vr11,$vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.s $vr12,$vr12,$vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.s $vr13,$vr13,$vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.s $vr14,$vr14,$vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.s $vr15,$vr15,$vr2"
        li.d    $s0, 16
        blt     $a5, $s0, .LOutputPartial16xNBlock\@
        sub.d   $a5, $a5, $s0
        AccumulateAndStoreBlock \RowCount\(), 4
        addi.d  $a2, $a2, 16*4          # advance matrix C by 16 columns
        move    $a0, $t1                # reload matrix A
        bnez    $a5, .LProcessNextColumnLoop16xN\@
        b       .LExitKernel

//
// Output a partial 16xN block to the matrix.
//

.LOutputPartial16xNBlock\@:
        li.d    $s0, 4
        blt     $a5, $s0, .LOutputPartialLessThan4xNBlock\@
        li.d    $s0, 8
        blt     $a5, $s0, .LOutputPartialLessThan8xNBlock\@
        li.d    $s0, 12
        blt     $a5, $s0, .LOutputPartialLessThan12xNBlock\@
        AccumulateAndStoreBlock \RowCount\(), 3
        andi  $a5, $a5, 3
        beqz    $a5, .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "vmove $vr8, $vr11"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "vmove $vr12, $vr15"
        addi.d  $a2, $a2,12*4                    # advance matrix C by 12 columns
        b     .LOutputPartialLessThan4xNBlock\@

.LOutputPartialLessThan12xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 2
        andi  $a5, $a5, 3
        beqz    $a5, .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "vmove $vr8, $vr10"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "vmove $vr12, $vr14"
        addi.d  $a2, $a2,8*4                    # advance matrix C by 8 columns
        b     .LOutputPartialLessThan4xNBlock\@

.LOutputPartialLessThan8xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 1
        andi  $a5, $a5, 3
        beqz    $a5, .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "vmove $vr8, $vr9"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "vmove $vr12, $vr13"
        addi.d  $a2, $a2, 4*4                     # advance matrix C by 4 columns

.LOutputPartialLessThan4xNBlock\@:
        andi  $s0, $a5, 2
        beqz    $s0, .LOutputPartial1xNBlock\@
        and     $s0,  $t5, $t5       # ZeroMode?
        bnez    $s0, .LSkipAccumulateOutput2xN\@
        EmitIfCountGE \RowCount\(), 1, "vxor.v  $vr0, $vr0, $vr0"
        EmitIfCountGE \RowCount\(), 1, "ld.d    $s0, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "vinsgr2vr.d     $vr0, $s0, 0"
        EmitIfCountGE \RowCount\(), 2, "vxor.v  $vr1, $vr1, $vr1"
        EmitIfCountGE \RowCount\(), 2, "ldx.d   $s0, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "vinsgr2vr.d     $vr1, $s0, 0"
        EmitIfCountGE \RowCount\(), 1, "vfadd.s $vr8, $vr8, $vr0"
        EmitIfCountGE \RowCount\(), 2, "vfadd.s $vr12, $vr12, $vr1"

.LSkipAccumulateOutput2xN\@:
        EmitIfCountGE \RowCount\(), 1, "vstelm.d    $vr8, $a2, 0, 0"
        EmitIfCountGE \RowCount\(), 2, "vpickve2gr.d    $s0, $vr12, 0"
        EmitIfCountGE \RowCount\(), 2, "stx.d    $s0, $a2, $t6"
        andi     $s0, $a5, 1
        beqz    $s0, .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "vpermi.w $vr8, $vr8, 0xee"
                                            # shift third element down
        EmitIfCountGE \RowCount\(), 2, "vpermi.w $vr12, $vr12, 0xee"
        addi.d     $a2, $a2, 2*4                     # advance matrix C by 2 columns

.LOutputPartial1xNBlock\@:
        and    $s0, $t5, $t5                   # ZeroMode?
        bnez    $s0, .LSkipAccumulateOutput1xN\@

        EmitIfCountGE \RowCount\(), 1, "fld.s $f16, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "fadd.s $f8, $f16, $f8"
        EmitIfCountGE \RowCount\(), 2, "fldx.s $f17, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "fadd.s $f12, $f12, $f17"

.LSkipAccumulateOutput1xN\@:
        EmitIfCountGE \RowCount\(), 1, "fst.s $f8, $a2, 0"
        EmitIfCountGE \RowCount\(), 2, "fstx.s $f12, $a2, $t6"
.ifb \Fallthrough\()
        b     .LExitKernel
.endif
        .endm

//
// Generate the GEMM kernel.
//

FgemmKernelLsxFunction MlasGemmFloatKernelLSX

        .end
