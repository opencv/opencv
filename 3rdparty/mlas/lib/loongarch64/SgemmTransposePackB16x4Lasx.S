/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmTransposePackB16x4Lasx.s

Abstract:

    This module implements routines for packing buffers for the single precision
    matrix/matrix multiply operation (SGEMM).

    This implementation uses Lasx instructions.

--*/

#include "asmmacro.h"

        .text

/*++

Macro Description:

    4 columns of 8 rows from the source matrix are transposed to 8 columns of 4
    rows in the destination packed buffer.

Arguments:

    StoreOffset - Supplies the relative byte offset into the destination packed
        buffer.

Implicit Arguments:

    a0 - Supplies the address of the destination packed buffer.

    a1 - Supplies the address of the source matrix.

    a2 - Supplies the number of elements per row of the source matrix.

--*/

        .macro TransposePackB8x4BlockLasx StoreOffset

//
// Load 4 columns from 8 rows of the source matrix into the lower and upper
// halves of 4 XR registers.
//

	add.d	$t0, $a2, $a2
	add.d	$t6, $a1, $t0
	vld	$vr0, $a1, 0
	vldx	$vr1, $a1, $a2
	add.d	$t0, $a2, $a2
	add.d	$a1, $t6, $t0
	vld	$vr2, $t6, 0
	vldx	$vr3, $t6, $a2
	add.d	$t0, $a2, $a2
	add.d	$t6, $a1, $t0

	vld	$vr4, $a1, 0
	xvpermi.q	$xr0, $xr4, 0x2
	vldx	$vr5, $a1, $a2
	xvpermi.q	$xr1, $xr5, 0x2
	vld	$vr4, $t6, 0
	xvpermi.q	$xr2, $xr4, 0x2
	vldx	$vr5, $t6, $a2
	xvpermi.q	$xr3, $xr5, 0x2

//
// Transpose the lower and upper halves of the 4 XR registers as two 4x4
// matrices and store the output to the destination packed buffer.
//

	xvilvl.w	$xr4, $xr1, $xr0
	xvilvh.w	$xr5, $xr1, $xr0
	xvilvl.w	$xr0, $xr3, $xr2
	xvilvh.w	$xr1, $xr3, $xr2
	xvilvl.d	$xr2, $xr0, $xr4
	xvilvh.d	$xr3, $xr0, $xr4
	xvst	$xr2, $a0, \StoreOffset\()
	xvst	$xr3, $a0, 0x40+\StoreOffset\()
	xvilvl.d	$xr0, $xr1, $xr5
	xvilvh.d	$xr4, $xr1, $xr5
	xvst	$xr0, $a0, 0x80+\StoreOffset\()
	xvst	$xr4, $a0, 0xc0+\StoreOffset\()

        .endm

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

        FUNCTION_ENTRY MlasSgemmTransposePackB16x4Lasx

	slli.d	$a2, $a2, 2                 # convert ldb to bytes
        TransposePackB8x4BlockLasx 0*4
	add.d	$t0, $a2, $a2
	add.d	$a1, $t0, $t6
        TransposePackB8x4BlockLasx 8*4
	jr	$ra

        .end
