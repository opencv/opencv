
/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelLasxCommon.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses LASX instructions.

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

    a0 - Supplies the address into the matrix A data.

    t7 - Supplies the address into the matrix A data plus 2 rows.

    a1 - Supplies the address into the matrix B data.

    t0 - Supplies the length in bytes of a row from matrix A.

    xr8-xr15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLasxBy16 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.if \RowCount\() == 1
    xvldrepl.w	$xr3, $a0, \BroadcastOffset\()
	xvld	$xr4, $a1, \VectorOffset\()
	xvfmadd	$xr8, $xr4, $xr3, $xr8
	xvld	$xr5, $a1, \VectorOffset\()+32
	xvfmadd	$xr9, $xr5, $xr3, $xr9
.else
	xvld	$xr0, $a1, \VectorOffset\()
	xvld	$xr1, $a1, \VectorOffset\()+32
        EmitIfCountGE \RowCount\(), 1, "xvldrepl $xr3,$a0, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 1, "xvfmadd $xr8, $xr3, $xr0, $xr8"
        EmitIfCountGE \RowCount\(), 1, "xvfmadd $xr9, $xr3, $xr1, $xr9"
        EmitIfCountGE \RowCount\(), 2, "add.d $s0,$a0, $t0"
        EmitIfCountGE \RowCount\(), 2, "xvldrepl $xr3,$s0, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 2, "xvfmadd $xr10, $xr3, $xr0, $xr10"
        EmitIfCountGE \RowCount\(), 2, "xvfmadd $xr11, $xr3, $xr1, $xr11"

        EmitIfCountGE \RowCount\(), 3, "xvldrepl $xr3,$t7, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 3, "xvfmadd $xr12, $xr3, $xr0, $xr12"
        EmitIfCountGE \RowCount\(), 3, "xvfmadd $xr13, $xr3, $xr1, $xr13"
        EmitIfCountGE \RowCount\(), 4, "add.d $s0,$t7, $t0"
        EmitIfCountGE \RowCount\(), 4, "xvldrepl $xr3,$s0, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 4, "xvfmadd $xr14, $xr3, $xr0, $xr14"
        EmitIfCountGE \RowCount\(), 4, "xvfmadd $xr15, $xr3, $xr1, $xr15"
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

    a0 - Supplies the address into the matrix A data.

    t7 - Supplies the address into the matrix A data plus 2 rows.

    a1 - Supplies the address into the matrix B data.

    t0 - Supplies the length in bytes of a row from matrix A.

    xr8-xr15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLasxBy8 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.if \RowCount\() == 1
    xvldrepl.w	$xr3, $a0, \BroadcastOffset\()
	xvld	$xr5, $a1, \VectorOffset\()
	xvfmadd.s	$xr9, $xr5, $xr3, $xr9
.else
	xvld	$xr0, $a1, \VectorOffset\()
        EmitIfCountGE \RowCount\(), 1, "xvldrepl $xr3, $a0, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 1, "xvfmadd $xr9, $xr3, $xr0, $xr9"

        EmitIfCountGE \RowCount\(), 2, "add.d $s0, $a0, $t0"
        EmitIfCountGE \RowCount\(), 2, "xvldrepl $xr3, $s0, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 2, "xvfmadd $xr11, $xr3, $xr0, $xr11"
        EmitIfCountGE \RowCount\(), 3, "xvldrepl $xr3, $t7, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 3, "xvfmadd $xr13, $xr3, $xr0, $xr13"
        EmitIfCountGE \RowCount\(), 4, "add.d $s0, $t7, $t0"
        EmitIfCountGE \RowCount\(), 4, "xvldrepl $xr3, $s0, \BroadcastOffset\()"
        EmitIfCountGE \RowCount\(), 4, "xvfmadd $xr15, $xr3, $xr0, $xr15"
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

    a0 - Supplies the address into the matrix A data.

    a1 - Supplies the address into the matrix B data.

    a3 - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    t0 - Supplies the length in bytes of a row from matrix A.

    vr4-vr15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLasxLoop ComputeBlock, RowCount

.if \RowCount\() > 2
        # compute matrix A plus 2 rows
	slli.d	$s0, $t0, 1
	add.d	$t7, $a0, $s0
.endif
        ComputeBlockLoop \ComputeBlock\(), \RowCount\(), \RowCount\() > 2
.if \RowCount\() > 2
        # compute matrix C plus 2 rows
	slli.d	$s0, $t6, 1
	add.d	$t7, $a2, $s0
.endif

        .endm

    .macro store_n  src, num, dst
    move    $s2,    \num\()
    beqz    $s2, .Lstore_exit\@
    xvstelm.w   \src\(), \dst\(), 0, 0
    addi.d  $s2, $s2, -1
    beqz    $s2, .Lstore_exit\@

    xvstelm.w   \src\(), \dst\(), 4, 1
    addi.d  $s2, $s2, -1
    beqz    $s2, .Lstore_exit\@

    xvstelm.w   \src\(), \dst\(), 8, 2
    addi.d  $s2, $s2, -1
    beqz    $s2, .Lstore_exit\@

    xvstelm.w   \src\(), \dst\(), 12, 3
    addi.d  $s2, $s2, -1
    beqz    $s2, .Lstore_exit\@

    xvstelm.w   \src\(), \dst\(), 16, 4
    addi.d  $s2, $s2, -1
    beqz    $s2, .Lstore_exit\@

    xvstelm.w   \src\(), \dst\(), 20, 5
    addi.d  $s2, $s2, -1
    beqz    $s2, .Lstore_exit\@

    xvstelm.w   \src\(), \dst\(), 24, 6
    addi.d  $s2, $s2, -1
    beqz    $s2, .Lstore_exit\@

.Lstore_exit\@:
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

    t1 - Supplies the address of matrix A.

    a5 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    a2 - Supplies the address of matrix C.

    a3 - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    t0 - Supplies the length in bytes of a row from matrix A.

    t6 - Supplies the length in bytes of a row from matrix C.

    t5 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountM RowCount, Fallthrough

	ori	$s1, $r0, LFgemmYmmElementCount
	bgeu	$s1, $a5, .LProcessRemainingCountN\@

.LProcessNextColumnLoop2xN\@:
        EmitIfCountGE \RowCount\(), 1, "xvxor.v $xr8, $xr8, $xr8"
        EmitIfCountGE \RowCount\(), 1, "xvxor.v $xr9, $xr9, $xr9"
        EmitIfCountGE \RowCount\(), 2, "xvxor.v $xr10, $xr10, $xr10"
        EmitIfCountGE \RowCount\(), 2, "xvxor.v $xr11, $xr11, $xr11"
        EmitIfCountGE \RowCount\(), 3, "xvxor.v $xr12, $xr12, $xr12"
        EmitIfCountGE \RowCount\(), 3, "xvxor.v $xr13, $xr13, $xr13"
        EmitIfCountGE \RowCount\(), 4, "xvxor.v $xr14, $xr14, $xr14"
        EmitIfCountGE \RowCount\(), 4, "xvxor.v $xr15, $xr15, $xr15"

        ComputeBlockLasxLoop ComputeBlockLasxBy16, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "xvfmul $xr8, $xr8, $xr2"
        EmitIfCountGE \RowCount\(), 1, "xvfmul $xr9, $xr9, $xr2"
        EmitIfCountGE \RowCount\(), 2, "xvfmul $xr10, $xr10, $xr2"
        EmitIfCountGE \RowCount\(), 2, "xvfmul $xr11, $xr11, $xr2"
        EmitIfCountGE \RowCount\(), 3, "xvfmul $xr12, $xr12, $xr2"
        EmitIfCountGE \RowCount\(), 3, "xvfmul $xr13, $xr13, $xr2"
        EmitIfCountGE \RowCount\(), 4, "xvfmul $xr14, $xr14, $xr2"
        EmitIfCountGE \RowCount\(), 4, "xvfmul $xr15, $xr15, $xr2"

	sub.d	$a5, $a5, $s1
	sub.d	$a5, $a5, $s1
	blt	$a5, $zero, .LOutputMasked2xNBlock\@
	andi	$s0, $t5, 0xff # ZeroMode?
	bnez	$s0, .LStore2xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "xvld $xr16, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "xvfadd $xr8, $xr8, $xr16"
        EmitIfCountGE \RowCount\(), 1, "xvld $xr16, $a2, 0x20"
        EmitIfCountGE \RowCount\(), 1, "xvfadd $xr9, $xr9, $xr16"
        EmitIfCountGE \RowCount\(), 2, "xvldx $xr16, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "xvfadd $xr10, $xr10, $xr16"
        EmitIfCountGE \RowCount\(), 2, "add.d $s0, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "xvld $xr16, $s0, 0x20"
        EmitIfCountGE \RowCount\(), 2, "xvfadd $xr11, $xr11, $xr16"
        EmitIfCountGE \RowCount\(), 3, "xvld $xr16, $t7, 0"
        EmitIfCountGE \RowCount\(), 3, "xvfadd $xr12, $xr12, $xr16"
        EmitIfCountGE \RowCount\(), 3, "xvld $xr16, $t7, 0x20"
        EmitIfCountGE \RowCount\(), 3, "xvfadd $xr13, $xr13, $xr16"
        EmitIfCountGE \RowCount\(), 4, "xvldx $xr16, $t7, $t6"
        EmitIfCountGE \RowCount\(), 4, "xvfadd $xr14, $xr14, $xr16"
        EmitIfCountGE \RowCount\(), 4, "add.d $s0, $t7, $t6"
        EmitIfCountGE \RowCount\(), 4, "xvld $xr16, $s0, 0x20"
        EmitIfCountGE \RowCount\(), 4, "xvfadd $xr15, $xr15, $xr16"

.LStore2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "xvst $xr8, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "xvst $xr9, $a2, 0x20"
        EmitIfCountGE \RowCount\(), 2, "xvstx $xr10, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "add.d $s0, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "xvst $xr11, $s0, 0x20"
        EmitIfCountGE \RowCount\(), 3, "xvst $xr12, $t7, 0"
        EmitIfCountGE \RowCount\(), 3, "xvst $xr13, $t7, 0x20"
        EmitIfCountGE \RowCount\(), 4, "xvstx $xr14, $t7, $t6"
        EmitIfCountGE \RowCount\(), 4, "add.d $s0, $t7, $t6"
        EmitIfCountGE \RowCount\(), 4, "xvst $xr15, $s0, 0x20"

	addi.d	$a2, $a2, 0x40     # advance matrix C by 2 XRWORDs
	move	$a0, $t1	   # reload matrix A
	bltu	$s1, $a5, .LProcessNextColumnLoop2xN\@
	beqz	$a5, .LExitKernel

.LProcessRemainingCountN\@:
        EmitIfCountGE \RowCount\(), 1, "xvxor.v $xr9, $xr9, $xr9"
        EmitIfCountGE \RowCount\(), 2, "xvxor.v $xr11, $xr11, $xr11"
        EmitIfCountGE \RowCount\(), 3, "xvxor.v $xr13, $xr13, $xr13"
        EmitIfCountGE \RowCount\(), 4, "xvxor.v $xr15, $xr15, $xr15"


        ComputeBlockLasxLoop ComputeBlockLasxBy8, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "xvfmul $xr9, $xr9, $xr2"
        EmitIfCountGE \RowCount\(), 2, "xvfmul $xr11, $xr11, $xr2"
        EmitIfCountGE \RowCount\(), 3, "xvfmul $xr13, $xr13, $xr2"
        EmitIfCountGE \RowCount\(), 4, "xvfmul $xr15, $xr15, $xr2"
	bltu	$a5, $s1, .LOutputMasked1xNBlock\@
	andi	$s0, $t5, 0xff # ZeroMode?
	bnez	$s0, .LStore1xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "xvld  $xr16, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "xvfadd  $xr9, $xr9, $xr16"
        EmitIfCountGE \RowCount\(), 2, "xvldx  $xr16, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "xvfadd  $xr11, $xr11, $xr16"
        EmitIfCountGE \RowCount\(), 3, "xvld  $xr16, $t7, 0"
        EmitIfCountGE \RowCount\(), 3, "xvfadd  $xr13, $xr13, $xr16"
        EmitIfCountGE \RowCount\(), 4, "xvldx  $xr16, $t7, $t6"
        EmitIfCountGE \RowCount\(), 4, "xvfadd  $xr15, $xr15, $xr16"

.LStore1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "xvst $xr9, $a2, 0"
        EmitIfCountGE \RowCount\(), 2, "xvstx $xr11, $a2, $t6"
        EmitIfCountGE \RowCount\(), 3, "xvst $xr13, $t7, 0"
        EmitIfCountGE \RowCount\(), 4, "xvstx $xr15, $t7, $t6"
        b     .LExitKernel

.LOutputMasked2xNBlock\@:
	andi	$s0, $t5, 0xff # ZeroMode?
	bnez	$s0, .LStoreMasked2xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "xvld $xr16, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "xvfadd $xr8, $xr8, $xr16"
        EmitIfCountGE \RowCount\(), 2, "xvldx $xr16, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "xvfadd $xr10, $xr10, $xr16"
        EmitIfCountGE \RowCount\(), 3, "xvld $xr16, $t7, 0"
        EmitIfCountGE \RowCount\(), 3, "xvfadd $xr12, $xr12, $xr16"
        EmitIfCountGE \RowCount\(), 4, "xvldx $xr16, $t7, $t6"
        EmitIfCountGE \RowCount\(), 4, "xvfadd $xr14, $xr14, $xr16"

.LStoreMasked2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "xvst $xr8, $a2, 0"
        EmitIfCountGE \RowCount\(), 2, "xvstx $xr10, $a2, $t6"
        EmitIfCountGE \RowCount\(), 3, "xvst $xr12, $t7, 0"
        EmitIfCountGE \RowCount\(), 4, "xvstx $xr14, $t7, $t6"
	addi.d	$a2, $a2, 0x20              # advance matrix C by YMMWORD
.if \RowCount\() > 2
	addi.d	$t7, $t7, 0x20               # advance matrix C plus 2 rows by YMMWORD

.endif
	addi.d	$a5, $a5, LFgemmYmmElementCount   # correct for over-subtract above


.LOutputMasked1xNBlock\@:

.if \RowCount\() > 2
    slli.d $s0, $t0, 1
    add.d   $t7, $a0, $s0
.endif

.if \RowCount\() == 1
.else
.endif

.if \RowCount\() > 2
    slli.d  $s0, $t6, 1
    add.d   $t7, $a2, $s0
.endif

	sub.d	$a5, $zero, $a5
    la.global	$a0, MlasMaskMoveTableLasx
	ori	$s0, $r0, LFgemmElementSize
	mul.d	$s0, $a5, $s0
    addi.d  $s0, $s0, 8*4
	xvldx	$xr0, $a0, $s0
	andi	$s0, $t5, 0xff

	sub.d	$a5, $zero, $a5

	bnez	$s0, .LStoreMasked1xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "xvld $xr16, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "xvand.v $xr8, $xr16, $xr0"
        EmitIfCountGE \RowCount\(), 2, "xvldx $xr16, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "xvand.v $xr10, $xr16, $xr0"
        EmitIfCountGE \RowCount\(), 3, "xvld $xr16, $t7, 0"
        EmitIfCountGE \RowCount\(), 3, "xvand.v $xr12, $xr16, $xr0"
        EmitIfCountGE \RowCount\(), 4, "xvldx $xr16, $t7, $t6"
        EmitIfCountGE \RowCount\(), 4, "xvand.v $xr14, $xr16, $xr0"

        EmitIfCountGE \RowCount\(), 1, "xvfadd $xr9, $xr9, $xr8"
        EmitIfCountGE \RowCount\(), 2, "xvfadd $xr11, $xr11, $xr10"
        EmitIfCountGE \RowCount\(), 3, "xvfadd $xr13, $xr13, $xr12"
        EmitIfCountGE \RowCount\(), 4, "xvfadd $xr15, $xr15, $xr14"
.LStoreMasked1xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "store_n $xr9, $a5, $a2"

        add.d   $s3, $a2, $t6
        EmitIfCountGE \RowCount\(), 2, "store_n $xr11, $a5, $s3"

        EmitIfCountGE \RowCount\(), 3, "store_n $xr13, $a5, $t7"

        add.d   $s3, $t7, $t6
        EmitIfCountGE \RowCount\(), 4, "store_n $xr15, $a5, $s3"
	    sub.d	$a5, $zero, $a5
.ifb \Fallthrough\()
        b     .LExitKernel
.endif

        .endm

/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelLasxFunction FunctionName

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A a0 - Supplies the address of matrix A.

    B a1 - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C a2 - Supplies the address of matrix C.

    CountK a3 - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM a4 - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN a5 - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda a6 - Supplies the first dimension of matrix A.

    ldc a7 - Supplies the first dimension of matrix C.

    Alpha f0 - Supplies the scalar alpha multiplier (see GEMM definition).

    ZeroMode (sp + 0)- Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY \FunctionName\()

	addi.d	$sp, $sp, -64
	st.d	$ra, $sp, 56
	st.d	$s0, $sp, 0*8
	st.d	$s1, $sp, 1*8
	fst.s	$f0, $sp, 2*8
    fst.d   $f16, $sp,3*8
    st.d    $s2, $sp, 4*8
    st.d    $s3, $sp, 5*8

	move	$t1, $a0
	slli.d	$t0, $a6, 2  # convert lda to bytes
	slli.d	$t6, $a7, 2  # convert ldc to bytes
	ld.d	$t5, $sp, 64 # get zeromode
	fst.s	$f0, $sp, 2*8
	xvldrepl.w	$xr2, $sp, 0x10

//
// Process 4 rows of the matrices.
//

	ori	$s0, $zero, 4
	bltu	$a4, $s0, .LProcessCountMLessThan4
	li.d	$a4, 4	# return 4 rows handled
        ProcessCountM 4, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
    bstrpick.d	$a0, $a4, 31, 0
	ld.d	$s0, $sp, 0
	ld.d	$s1, $sp, 8
    fld.d   $f16, $sp,3*8
    ld.d    $s2, $sp, 4*8
    ld.d    $s3, $sp, 5*8
	ld.d	$ra, $sp, 7*8
	addi.d	$sp, $sp, 64
	jr	$ra

//
// Process 2 rows of the matrices.
//

.LProcessCountMLessThan4:
	ori	$s0, $r0, 2
	bltu	$a4, $s0, .LProcessCountMLessThan2
	li.d	$a4, 2	# return 2 rows handled
        ProcessCountM 2

//
// Process 1 row of the matrices.
//

.LProcessCountMLessThan2:
        ProcessCountM 1

        .endm
