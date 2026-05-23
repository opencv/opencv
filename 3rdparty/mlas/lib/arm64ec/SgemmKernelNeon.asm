;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelNeon.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;--

#include "kxarm64.h"

        TEXTAREA

;
; ClearRowAccumulators
;
; Generates the code to clear the accumulators for a single row of the output
; block.
;

        MACRO
        ClearRowAccumulators $Columns, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

        movi    $Vec1Reg..16b,#0
        movi    $Vec2Reg..16b,#0
    IF $Columns > 8
        movi    $Vec3Reg..16b,#0
        movi    $Vec4Reg..16b,#0
    ENDIF

        MEND

;
; ClearBlockAccumulators
;
; Generates the code to clear the accumulators for a single row of the output
; block.
;

        MACRO
        ClearBlockAccumulators $Columns, $Rows

        ClearRowAccumulators $Columns, v8, v9, v10, v11
    IF $Rows >= 2
        ClearRowAccumulators $Columns, v12, v13, v14, v15
    ENDIF

        MEND

;
; LoadMatrixAElementsBy4
; LoadMatrixAElementsBy1
;
; Generates the code to load 1 or 4 elements from matrix A.
;

        MACRO
        LoadMatrixAElementsBy4 $Rows

        ldr     v2,[x0],#16
    IF $Rows >= 2
        ldr     v3,[x10],#16
    ENDIF

        MEND

        MACRO
        LoadMatrixAElementsBy1 $Rows

        ldr     s2,[x0],#4
    IF $Rows >= 2
        ldr     s3,[x10],#4
    ENDIF

        MEND

;
; MultiplyAccumulateRow
;
; Generates the code to multiply and accumulate a single row of the output
; block.
;

        MACRO
        MultiplyAccumulateRow $Columns, $MatrixAReg, $Broadcast, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

        fmla    $Vec1Reg..4s,v4.4s,$MatrixAReg..s[$Broadcast]
        fmla    $Vec2Reg..4s,v5.4s,$MatrixAReg..s[$Broadcast]
    IF $Columns > 8
        fmla    $Vec3Reg..4s,v6.4s,$MatrixAReg..s[$Broadcast]
        fmla    $Vec4Reg..4s,v7.4s,$MatrixAReg..s[$Broadcast]
    ENDIF

        MEND

;
; MultiplyAccumulateBlock
;
; Generates the code to multiply and accumulate into the output block.
;

        MACRO
        MultiplyAccumulateBlock $Columns, $Rows, $Broadcast

        MultiplyAccumulateRow $Columns, v2, $Broadcast, v8, v9, v10, v11
    IF $Rows >= 2
        MultiplyAccumulateRow $Columns, v3, $Broadcast, v12, v13, v14, v15
    ENDIF

        MEND

;
; ComputeBlockLoop
;
; Generates the code to loop over K entries of the input matrices to produce
; the output block.
;

        MACRO
        ComputeBlockLoop $Mode, $Columns, $Rows

        ClearBlockAccumulators $Columns, $Rows

    IF $Rows >= 2
        add     x10,x0,x6 lsl #2            ; compute matrix A plus 1 row
    ENDIF

        sub     x9,x3,#4                    ; decrement block count to process
        tbnz    x9,#63,$Mode.ProcessRemaining$Columns.x$Rows.Blocks

$Mode.Compute$Columns.x$Rows.BlockBy4Loop
        LoadMatrixAElementsBy4 $Rows
        ldp     v4,v5,[x1],#64*4
    IF $Columns > 8
        ldp     v6,v7,[x1,#-56*4]
    ENDIF
        MultiplyAccumulateBlock $Columns,$Rows,0
        ldp     v4,v5,[x1,#-48*4]
    IF $Columns > 8
        ldp     v6,v7,[x1,#-40*4]
    ENDIF
        MultiplyAccumulateBlock $Columns,$Rows,1
        ldp     v4,v5,[x1,#-32*4]
    IF $Columns > 8
        ldp     v6,v7,[x1,#-24*4]
    ENDIF
        MultiplyAccumulateBlock $Columns,$Rows,2
        ldp     v4,v5,[x1,#-16*4]
    IF $Columns > 8
        ldp     v6,v7,[x1,#-8*4]
    ENDIF
        MultiplyAccumulateBlock $Columns,$Rows,3
        sub     x9,x9,#4
        tbz     x9,#63,$Mode.Compute$Columns.x$Rows.BlockBy4Loop

$Mode.ProcessRemaining$Columns.x$Rows.Blocks
        add     x9,x9,#4                    ; correct for over-subtract above
        cbz     x9,$Mode.Output$Columns.x$Rows.Block

$Mode.Compute$Columns.x$Rows.BlockBy1Loop
        LoadMatrixAElementsBy1 $Rows
        ldp     v4,v5,[x1],#16*4
    IF $Columns > 8
        ldp     v6,v7,[x1,#-8*4]
    ENDIF
        MultiplyAccumulateBlock $Columns,$Rows,0
        sub     x9,x9,#1
        cbnz    x9,$Mode.Compute$Columns.x$Rows.BlockBy1Loop

$Mode.Output$Columns.x$Rows.Block

        MEND

;
; MultiplyAlphaRow
;
; Generates the code to multiply a single row of the output block by the alpha
; value.
;

        MACRO
        MultiplyAlphaRow $Columns, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

    IF $Columns <= 4
        fmul    $Vec1Reg..4s,$Vec1Reg..4s,v0.s[0]
    ELIF $Columns <= 8
        fmul    $Vec1Reg..4s,$Vec1Reg..4s,v0.s[0]
        fmul    $Vec2Reg..4s,$Vec2Reg..4s,v0.s[0]
    ELIF $Columns <= 12
        fmul    $Vec1Reg..4s,$Vec1Reg..4s,v0.s[0]
        fmul    $Vec2Reg..4s,$Vec2Reg..4s,v0.s[0]
        fmul    $Vec3Reg..4s,$Vec3Reg..4s,v0.s[0]
    ELSE
        fmul    $Vec1Reg..4s,$Vec1Reg..4s,v0.s[0]
        fmul    $Vec2Reg..4s,$Vec2Reg..4s,v0.s[0]
        fmul    $Vec3Reg..4s,$Vec3Reg..4s,v0.s[0]
        fmul    $Vec4Reg..4s,$Vec4Reg..4s,v0.s[0]
    ENDIF

        MEND

;
; MultiplyAlphaBlock
;
; Generates the code to multiply the output block by the alpha value.
;

        MACRO
        MultiplyAlphaBlock $Columns, $Rows

        MultiplyAlphaRow $Columns, v8, v9, v10, v11
    IF $Rows >= 2
        MultiplyAlphaRow $Columns, v12, v13, v14, v15
    ENDIF

        MEND

;
; OutputRow1Element
; OutputRow2Element
; OutputRow4Element
; OutputRow8Element
; OutputRow16Element
;
; Generates the code to store elements to the output block.
;

        MACRO
        OutputRow1Element $Mode, $AddrReg, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

    IF "$Mode"=="Add"
        ld1     {v4.s}[0],[$AddrReg]
        fmla    v4.2s,$Vec1Reg..2s,v0.s[0]
        st1     {v4.s}[0],[$AddrReg]        ; post-increment not needed for last element
    ELSE
        st1     {$Vec1Reg..s}[0],[$AddrReg] ; post-increment not needed for last element
    ENDIF

        MEND

        MACRO
        OutputRow2Element $Mode, $AddrReg, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

    IF "$Mode"=="Add"
        ld1     {v4.2s},[$AddrReg]
        fmla    v4.2s,$Vec1Reg..2s,v0.s[0]
        st1     {v4.2s},[$AddrReg],#2*4
    ELSE
        st1     {$Vec1Reg..2s},[$AddrReg],#2*4
    ENDIF
        dup     $Vec1Reg..4s,$Vec1Reg..s[2] ; shift remaining elements down

        MEND

        MACRO
        OutputRow4Element $Mode, $AddrReg, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

    IF "$Mode"=="Add"
        ld1     {v4.4s},[$AddrReg]
        fmla    v4.4s,$Vec1Reg..4s,v0.s[0]
        st1     {v4.4s},[$AddrReg],#4*4
    ELSE
        st1     {$Vec1Reg..4s},[$AddrReg],#4*4
    ENDIF
        mov     $Vec1Reg..16b,$Vec2Reg..16b ; shift remaining elements down

        MEND

        MACRO
        OutputRow8Element $Mode, $AddrReg, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

    IF "$Mode"=="Add"
        ldp     v4,v5,[$AddrReg]
        fmla    v4.4s,$Vec1Reg..4s,v0.s[0]
        fmla    v5.4s,$Vec2Reg..4s,v0.s[0]
        stp     v4,v5,[$AddrReg],#8*4
    ELSE
        stp     $Vec1Reg.,$Vec2Reg.,[$AddrReg],#8*4
    ENDIF
        mov     $Vec1Reg..16b,$Vec3Reg..16b ; shift remaining elements down
        mov     $Vec2Reg..16b,$Vec4Reg..16b

        MEND

        MACRO
        OutputRow16Element $Mode, $AddrReg, $Vec1Reg, $Vec2Reg, $Vec3Reg, $Vec4Reg

    IF "$Mode"=="Add"
        ldp     v4,v5,[$AddrReg]
        ldp     v6,v7,[$AddrReg,#8*4]
        fmla    v4.4s,$Vec1Reg..4s,v0.s[0]
        fmla    v5.4s,$Vec2Reg..4s,v0.s[0]
        fmla    v6.4s,$Vec3Reg..4s,v0.s[0]
        fmla    v7.4s,$Vec4Reg..4s,v0.s[0]
        stp     v4,v5,[$AddrReg],#16*4
        stp     v6,v7,[$AddrReg,#-8*4]
    ELSE
        stp     $Vec1Reg.,$Vec2Reg.,[$AddrReg],#16*4
        stp     $Vec3Reg.,$Vec4Reg.,[$AddrReg,#-8*4]
    ENDIF

        MEND

;
; OutputBlock
;
; Generates the code to store the output block.
;

        MACRO
        OutputBlock $Mode, $Columns, $Rows

        OutputRow$Columns.Element $Mode, x2, v8, v9, v10, v11
    IF $Rows >= 2
        OutputRow$Columns.Element $Mode, x11, v12, v13, v14, v15
    ENDIF

        MEND

;
; ProcessRows
;
; Generates the code to process a compute and store the output block for a
; fixed number of rows.
;

        MACRO
        ProcessRows $Mode, $Rows

        mov     x4,#$Rows                   ; return number of rows handled
        cmp     x5,#8
        ble     $Mode.ProcessRemainingCountN$Rows

$Mode.ProcessNextColumnLoop16x$Rows
        ComputeBlockLoop $Mode,16,$Rows
    IF "$Mode"=="Zero"
        MultiplyAlphaBlock 16,$Rows
    ENDIF
        sub     x5,x5,#16
        tbnz    x5,#63,$Mode.OutputMasked16x$Rows.Block
        OutputBlock $Mode,16,$Rows
        mov     x0,x8                       ; reload matrix A
        cmp     x5,#8
        bgt     $Mode.ProcessNextColumnLoop16x$Rows
        cbz     x5,$Mode.ExitKernel

$Mode.ProcessRemainingCountN$Rows
        ComputeBlockLoop $Mode,8,$Rows
    IF "$Mode"=="Zero"
        MultiplyAlphaBlock 8,$Rows
    ENDIF

$Mode.OutputMasked16x$Rows.Block
        tbz     x5,#3,$Mode.OutputRemaining7x$Rows.Block
        OutputBlock $Mode,8,$Rows

$Mode.OutputRemaining7x$Rows.Block
        tbz     x5,#2,$Mode.OutputRemaining3x$Rows.Block
        OutputBlock $Mode,4,$Rows

$Mode.OutputRemaining3x$Rows.Block
        tbz     x5,#1,$Mode.OutputRemaining1x$Rows.Block
        OutputBlock $Mode,2,$Rows

$Mode.OutputRemaining1x$Rows.Block
        tbz     x5,#0,$Mode.ExitKernel
        OutputBlock $Mode,1,$Rows

        MEND

        SUBT    "SGEMM kernel"
;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows.
;
; Arguments:
;
;   A (x0) - Supplies the address of matrix A.
;
;   B (x1) - Supplies the address of matrix B. The matrix data has been packed
;       using MlasSgemmCopyPackB or MlasSgemmTransposePackB.
;
;   C (x2) - Supplies the address of matrix C.
;
;   CountK (x3) - Supplies the number of columns from matrix A and the number
;       of rows from matrix B to iterate over.
;
;   CountM (x4) - Supplies the maximum number of rows that can be processed for
;       matrix A and matrix C. The actual number of rows handled for this
;       invocation depends on the kernel implementation.
;
;   CountN (x5) - Supplies the number of columns from matrix B and matrix C to
;       iterate over.
;
;   lda (x6) - Supplies the first dimension of matrix A.
;
;   ldc (x7) - Supplies the first dimension of matrix C.
;
;   Alpha (s0) - Supplies the scalar multiplier (see SGEMM definition).
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

        MACRO
        SgemmKernelNeonFunction $Mode

        NESTED_ENTRY_COMDAT A64NAME(MlasSgemmKernel$Mode)

        PROLOG_SAVE_REG_PAIR d8,d9,#-64!
        PROLOG_SAVE_REG_PAIR d10,d11,#16
        PROLOG_SAVE_REG_PAIR d12,d13,#32
        PROLOG_SAVE_REG_PAIR d14,d15,#48

        add     x11,x2,x7 lsl #2            ; compute matrix C plus 1 row
        mov     x8,x0                       ; save matrix A

;
; Process 2 rows of the matrices.
;

        cmp     x4,#2
        blt     $Mode.ProcessCountMLessThan2
        ProcessRows $Mode,2

;
; Restore non-volatile registers and return.
;

$Mode.ExitKernel
        mov     x0,x4
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN

;
; Process 1 row of the matrices.
;

$Mode.ProcessCountMLessThan2
        ProcessRows $Mode,1
        b       $Mode.ExitKernel

        NESTED_END

        MEND

        SgemmKernelNeonFunction Zero
        SgemmKernelNeonFunction Add

        END
