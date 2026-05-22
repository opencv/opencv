/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmTransposePackB16x4LSX.s

Abstract:

    This module implements routines for packing buffers for the single precision
    matrix/matrix multiply operation (SGEMM).

    This implementation uses Lsx instructions.

--*/

#include "asmmacro.h"

        .text

/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    4 columns of 16 rows from the source matrix are transposed to 16 columns of 4
    rows in the destination packed buffer.

Arguments:

    D (a0) - Supplies the address of the destination packed buffer.

    B (a1) - Supplies the address of the source matrix.

    ldb (a2) - Supplies the number of elements per row of the source matrix.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasSgemmTransposePackB16x4LSX
    addi.d  $sp, $sp, -64
    st.d    $s0, $sp, 0*8
    st.d    $s1, $sp, 1*8
	slli.d	$a2, $a2, 2		# convert ldb to bytes
	ori	$a3, $zero, 4		# transpose four 4x4 blocks
	vxor.v	$vr7, $vr7, $vr7
.LTransposeBlockLoop:
	slli.d	$s0, $a2, 1
	add.d	$s1, $a1, $s0
	vld	$vr0, $a1, 0
	vldx	$vr1, $a1, $a2
	vld	$vr2, $s1, 0
	vldx	$vr3, $s1, $a2

	vor.v	$vr4, $vr0, $vr7
	vilvl.w	$vr4, $vr1, $vr4
	vilvh.w	$vr0, $vr1, $vr0
	vor.v	$vr5, $vr2, $vr7
	vilvl.w	$vr5, $vr3, $vr5
	vilvh.w	$vr2, $vr3, $vr2
	vor.v	$vr1, $vr4, $vr7
	vilvl.d	$vr1, $vr5, $vr1
	vilvh.d	$vr4, $vr5, $vr4
	vor.v	$vr3, $vr0, $vr7
	vilvl.d	$vr3, $vr2, $vr3
	vilvh.d	$vr0, $vr2, $vr0
	vst	$vr1, $a0, 0
	vst	$vr4, $a0, 0x40
	vst	$vr3, $a0, 0x80
	vst	$vr0, $a0, 0xc0
	addi.d	$a0, $a0, 0x10
	slli.d	$s0, $a2, 1
	add.d	$a1, $s0, $s1
	addi.d	$a3, $a3, -1
	bnez	$a3, .LTransposeBlockLoop
    ld.d    $s0, $sp, 0*8
    ld.d    $s1, $sp, 1*8
    addi.d  $sp, $sp, 64
	jr	$ra

        .end
