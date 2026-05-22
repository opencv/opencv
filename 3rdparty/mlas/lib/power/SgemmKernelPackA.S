/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelPackA.S

Abstract:

    This module implements the POWER10 kernel for packing matrix A for single precision SGEMM.

    This implementation targets power10 using VSX instructions.

--*/
/*++
Routine Description:

    This routine is an inner kernel to pack  matrix A for rows 4 or 8.

Arguments:

    D (r3) - Supplies the address of Packed A.

    A (r4) - Supplies the address of matrix A.

    lda (r5) - LDA.

    k (r6) - Supplies the number of columns from matrix A.

    RowCount (r7) - Supplies the number of rows to process.


Return Value:

    None.
--*/
#include "asmmacro.h"
.text
FUNCTION_ENTRY PackAKernelPOWER10
	slwi	9,5,2
	cmpldi	7,8
	add     8,4,9
	add     10,8,9
	add     11,10,9
	dcbt	0,4
	dcbt	0,8
	dcbt	0,10
	dcbt	0,11
	blt	L_loop
L_Rows8:
	lxvp    32,0(4)
	lxvp    42,32(4)
	addi    4,4,64
	dcbt	0,4
	lxvp    34,0(8) //a+lda
	lxvp    44,32(8) //a+32+lda
	lxvp    36,0(10) //a+2*lda
	lxvp    46,32(10) //a+32+2*lda
	lxvp    38,0(11) //a+3*lda
	lxvp    48,32(11) //a+32+3*lda
	add	7,11,9
	dcbt	0,7
	add	8,7,9
	dcbt	0,8
	add	10,8,9
	dcbt	0,10
	add     11,10,9
	dcbt	0,11
	vmrgow  8,3,1
	vmrgew  18,3,1
	vmrgow  9,7,5
	vmrgew  19,7,5
	xxpermdi        1,41,40,3
	xxpermdi        0,51,50,3
	xxpermdi        3,41,40,0
	xxpermdi        2,51,50,0
	vmrgow  8,2,0
	vmrgow  9,6,4
	vmrgew  18,2,0
	vmrgew  19,6,4
	stxvp   0,0(3)
	xxpermdi        9,41,40,3
	xxpermdi        8,51,50,3
	xxpermdi        11,41,40,0
	xxpermdi        10,51,50,0
	stxvp   2,32(3)
	stxvp   8,128(3)
	stxvp   10,160(3)

	vmrgow  0,13,11
	vmrgow  1,17,15
	vmrgew  18,13,11
	vmrgew  19,17,15
	xxpermdi	5,33,32,3
	xxpermdi        7,33,32,0
	xxpermdi        4,51,50,3
	xxpermdi        6,51,50,0
	vmrgow	0,12,10
	vmrgow  1,16,14
	stxvp   4,256(3)
	stxvp   6,288(3)
	vmrgew  18,12,10
	vmrgew  19,16,14
	xxpermdi        9,33,32,3
	xxpermdi        8,51,50,3
	xxpermdi        11,33,32,0
	xxpermdi        10,51,50,0
	lxvp    32,0(7) //a+4*lda
	lxvp    34,0(8) //a+5*lda
	lxvp    36,0(10) //a+6*lda
	lxvp    38, 0(11) //a+7*lda

	stxvp   8,384(3)
	stxvp   10,416(3)
	lxvp    42,32(7) //a+32+4*lda

	vmrgow  8,3,1
	vmrgew  18,3,1
	vmrgow  9,7,5
	vmrgew  19,7,5
	lxvp    44,32(8) //a+32+5*lda
	lxvp    46,32(10) //a+32+6*lda

	xxpermdi        1,41,40,3
	xxpermdi        0,51,50,3
	xxpermdi        3,41,40,0
	xxpermdi        2,51,50,0
	lxvp    48,32(11) //a+32+7*lda
	add     8,4,9
	dcbt	0,8
	add     10,8,9
	dcbt	0,10
	add     11,10,9
	dcbt	0,11
	stxvp   0,64(3)
	stxvp   2,96(3)
	vmrgow  8,2,0
	vmrgow  9,6,4
	vmrgew  18,2,0
	vmrgew  19,6,4
	vmrgow  0,13,11
	vmrgow  1,17,15

	xxpermdi        9,41,40,3
	xxpermdi        8,51,50,3
	xxpermdi        11,41,40,0
	xxpermdi        10,51,50,0
	vmrgew  18,13,11
	vmrgew  19,17,15
	stxvp   8,192(3)
	stxvp   10,224(3)
	xxpermdi        5,33,32,3
	xxpermdi        4,51,50,3
	xxpermdi        7,33,32,0
	xxpermdi        6,51,50,0
	vmrgow  0,12,10
	vmrgow  1,16,14
	stxvp   4,320(3)
	stxvp   6,352(3)
	vmrgew  18,12,10
	vmrgew  19,16,14
	xxpermdi        9,33,32,3
	xxpermdi        8,51,50,3
	xxpermdi        11,33,32,0
	xxpermdi        10,51,50,0

	stxvp   8,448(3)
	stxvp   10,480(3)
	addi    6,6,-16
	cmpldi  6,16

	addi	3,3,512
	bge     L_Rows8
	b L_exit

L_loop:
	lxvp    32,0(4)
	lxvp    42,32(4)
	addi    4,4,64
	dcbt    0,4
	lxvp    34,0(8) //a+lda
	lxvp    44,32(8) //a+32+lda
	lxvp    36,0(10) //a+2*lda
	lxvp    46,32(10) //a+32+2*lda
	lxvp    38,0(11) //a+3*lda
	lxvp    48,32(11) //a+32+3*lda
	vmrgow  8,3,1
	vmrgew  18,3,1
	vmrgow  9,7,5
	vmrgew  19,7,5

	add     8,4,9
	dcbt    0,8
	add     10,8,9
	dcbt    0,10
	add     11,10,9
	dcbt    0,11

	xxpermdi        1,41,40,3
	xxpermdi        0,51,50,3
	xxpermdi        3,41,40,0
	xxpermdi        2,51,50,0
	vmrgow  8,2,0
	vmrgow  9,6,4
	vmrgew  18,2,0
	vmrgew  19,6,4
	stxvp   0,0(3)

	xxpermdi        9,41,40,3
	xxpermdi        8,51,50,3
	xxpermdi        11,41,40,0
	xxpermdi        10,51,50,0
	stxvp   2,32(3)
	stxvp   8,64(3)
	stxvp   10,96(3)

	vmrgow  0,13,11
	vmrgow  1,17,15
	vmrgew  18,13,11
	vmrgew  19,17,15
	xxpermdi        5,33,32,3
	xxpermdi        7,33,32,0
	xxpermdi        4,51,50,3
	xxpermdi        6,51,50,0
	vmrgow  0,12,10
	vmrgow  1,16,14
	stxvp   4,128(3)
	stxvp   6,160(3)
	vmrgew  18,12,10
	vmrgew  19,16,14
	xxpermdi        9,33,32,3
	xxpermdi        8,51,50,3
	xxpermdi        11,33,32,0
	xxpermdi        10,51,50,0
	stxvp   8,192(3)
	stxvp   10,224(3)

	addi	3,3,256
	addi    6,6,-16
	cmpldi  6,16
	bge     L_loop

L_exit:
	blr
