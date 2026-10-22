;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemvKernelNeon.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/vector
;   multiply operation (SGEMV).
;
;--

#include "kxarm64.h"

        TEXTAREA

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows. This handles the special case of M=1.
;
;   The elements in matrix B are not transposed.
;
; Arguments:
;
;   A (x0) - Supplies the address of matrix A.
;
;   B (x1) - Supplies the address of matrix B.
;
;   C (x2) - Supplies the address of matrix C.
;
;   CountK (x3) - Supplies the number of columns from matrix A and the number
;       of rows from matrix B to iterate over.
;
;   CountN (x4) - Supplies the number of columns from matrix B and matrix C to
;       iterate over.
;
;   ldb (x5) - Supplies the first dimension of matrix B.
;
;   ZeroMode (x6) - Supplies true if the output matrix must be zero initialized,
;       else false if the output matrix is accumulated into.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY MlasGemvFloatKernel

        cmp     x4,#64
        blo     ProcessRemainingCountN
        mov     x14,x0                      ; preserve vector A

;
; Process 64 columns at a time in a loop.
;

ProcessColumnLoopBy64
        ldr     q4,[x1]
        add     x15,x1,#256                 ; compute next matrix B
        ldr     q5,[x1,#16]
        tst     w6,0xFF                     ; ZeroMode?
        mov     x13,x3                      ; reload CountK
        ldr     q6,[x1,#32]
        beq     LoadOutputBy64
        movi    v16.4s,#0
        movi    v17.4s,#0
        movi    v18.4s,#0
        movi    v19.4s,#0
        movi    v20.4s,#0
        movi    v21.4s,#0
        movi    v22.4s,#0
        movi    v23.4s,#0
        movi    v24.4s,#0
        movi    v25.4s,#0
        movi    v26.4s,#0
        movi    v27.4s,#0
        movi    v28.4s,#0
        movi    v29.4s,#0
        movi    v30.4s,#0
        movi    v31.4s,#0
        b       MultiplyAccumulateBy64

LoadOutputBy64
        ldp     q16,q17,[x2]
        ldp     q18,q19,[x2,#32]
        ldp     q20,q21,[x2,#64]
        ldp     q22,q23,[x2,#96]
        ldp     q24,q25,[x2,#128]
        ldp     q26,q27,[x2,#160]
        ldp     q28,q29,[x2,#192]
        ldp     q30,q31,[x2,#224]

MultiplyAccumulateBy64
        ld1r    {v0.4s},[x0]                ; broadcast next vector A element
        add     x0,x0,4                     ; advance vector A by 1 element
        sub     x13,x13,#1                  ; decrement K remaining
        fmla    v16.4s,v4.4s,v0.4s
        ldr     q7,[x1,#48]
        fmla    v17.4s,v5.4s,v0.4s
        ldr     q4,[x1,#64]
        fmla    v18.4s,v6.4s,v0.4s
        ldr     q5,[x1,#80]
        fmla    v19.4s,v7.4s,v0.4s
        ldr     q6,[x1,#96]
        fmla    v20.4s,v4.4s,v0.4s
        ldr     q7,[x1,#112]
        fmla    v21.4s,v5.4s,v0.4s
        ldr     q4,[x1,#128]
        fmla    v22.4s,v6.4s,v0.4s
        ldr     q5,[x1,#144]
        fmla    v23.4s,v7.4s,v0.4s
        ldr     q6,[x1,#160]
        fmla    v24.4s,v4.4s,v0.4s
        ldr     q7,[x1,#176]
        fmla    v25.4s,v5.4s,v0.4s
        ldr     q4,[x1,#192]
        fmla    v26.4s,v6.4s,v0.4s
        ldr     q5,[x1,#208]
        fmla    v27.4s,v7.4s,v0.4s
        ldr     q6,[x1,#224]
        fmla    v28.4s,v4.4s,v0.4s
        ldr     q7,[x1,#240]
        add     x1,x1,x5,lsl #2             ; compute next matrix B row address
        cbz     x13,StoreOutputBy64
        ldr     q4,[x1]                     ; load data for next iteration
        fmla    v29.4s,v5.4s,v0.4s
        ldr     q5,[x1,#16]
        fmla    v30.4s,v6.4s,v0.4s
        ldr     q6,[x1,#32]
        fmla    v31.4s,v7.4s,v0.4s
        b       MultiplyAccumulateBy64

StoreOutputBy64
        stp     q16,q17,[x2]
        fmla    v29.4s,v5.4s,v0.4s          ; finish computing tail vectors
        stp     q18,q19,[x2,#32]
        fmla    v30.4s,v6.4s,v0.4s
        stp     q20,q21,[x2,#64]
        fmla    v31.4s,v7.4s,v0.4s
        stp     q22,q23,[x2,#96]
        sub     x4,x4,#64                   ; subtract 64 columns
        stp     q24,q25,[x2,#128]
        mov     x0,x14                      ; reload vector A
        stp     q26,q27,[x2,#160]
        mov     x1,x15                      ; load next matrix B
        stp     q28,q29,[x2,#192]
        stp     q30,q31,[x2,#224]
        add     x2,x2,#256                  ; advance vector C by 64 columns
        cbz     x4,ExitKernel
        cmp     x4,#64
        bhs     ProcessColumnLoopBy64

;
; Process the remaining 1 to 63 columns.
;

ProcessRemainingCountN
        tst     w6,0xFF                     ; ZeroMode?
        beq     LoadOutputPartial32
        movi    v16.4s,#0
        movi    v17.4s,#0
        movi    v18.4s,#0
        movi    v19.4s,#0
        movi    v20.4s,#0
        movi    v21.4s,#0
        movi    v22.4s,#0
        movi    v23.4s,#0
        movi    v24.4s,#0
        movi    v25.4s,#0
        movi    v26.4s,#0
        movi    v27.4s,#0
        movi    v28.4s,#0
        movi    v29.4s,#0
        movi    v30.4s,#0
        movi    v31.4s,#0                   ; trailing float[2]
        movi    v1.4s,#0                    ; trailing float[1]
        b       ProcessNextPartialRow

LoadOutputPartial32
        mov     x15,x2
        tbz     x4,#5,LoadOutputPartial16
        ldp     q16,q17,[x15],#128
        ldp     q18,q19,[x15,#-96]
        ldp     q20,q21,[x15,#-64]
        ldp     q22,q23,[x15,#-32]

LoadOutputPartial16
        tbz     x4,#4,LoadOutputPartial8
        ldp     q24,q25,[x15],#64
        ldp     q26,q27,[x15,#-32]

LoadOutputPartial8
        tbz     x4,#3,LoadOutputPartial4
        ldp     q28,q29,[x15],#32

LoadOutputPartial4
        tbz     x4,#2,LoadOutputPartial2
        ldr     q30,[x15],#16

LoadOutputPartial2
        tbz     x4,#1,LoadOutputPartial1
        ldr     d31,[x15],#8

LoadOutputPartial1
        tbz     x4,#0,ProcessNextPartialRow
        ldr     s1,[x15]

ProcessNextPartialRow
        ld1r    {v0.4s},[x0]
        add     x0,x0,4
        sub     x3,x3,#1                    ; decrement K remaining
        mov     x15,x1

MultiplyAccumulatePartial32
        tbz     x4,#5,MultiplyAccumulatePartial16
        ldp     q4,q5,[x15],#128
        fmla    v16.4s,v4.4s,v0.4s
        ldp     q6,q7,[x15,#-96]
        fmla    v17.4s,v5.4s,v0.4s
        ldp     q4,q5,[x15,#-64]
        fmla    v18.4s,v6.4s,v0.4s
        fmla    v19.4s,v7.4s,v0.4s
        ldp     q6,q7,[x15,#-32]
        fmla    v20.4s,v4.4s,v0.4s
        fmla    v21.4s,v5.4s,v0.4s
        fmla    v22.4s,v6.4s,v0.4s
        fmla    v23.4s,v7.4s,v0.4s

MultiplyAccumulatePartial16
        tbz     x4,#4,MultiplyAccumulatePartial8
        ldp     q4,q5,[x15],#64
        fmla    v24.4s,v4.4s,v0.4s
        ldp     q6,q7,[x15,#-32]
        fmla    v25.4s,v5.4s,v0.4s
        fmla    v26.4s,v6.4s,v0.4s
        fmla    v27.4s,v7.4s,v0.4s

MultiplyAccumulatePartial8
        tbz     x4,#3,MultiplyAccumulatePartial4
        ldp     q4,q5,[x15],#32
        fmla    v28.4s,v4.4s,v0.4s
        fmla    v29.4s,v5.4s,v0.4s

MultiplyAccumulatePartial4
        tbz     x4,#2,MultiplyAccumulatePartial2
        ldr     q4,[x15],#16
        fmla    v30.4s,v4.4s,v0.4s

MultiplyAccumulatePartial2
        tbz     x4,#1,MultiplyAccumulatePartial1
        ldr     d4,[x15],#8
        fmla    v31.4s,v4.4s,v0.4s

MultiplyAccumulatePartial1
        tbz     x4,#0,AdvancePartialRow
        ldr     s4,[x15]
        fmla    v1.4s,v4.4s,v0.4s

AdvancePartialRow
        add     x1,x1,x5,lsl #2             ; compute next matrix B row address
        cbnz    x3,ProcessNextPartialRow

StoreOutputPartial32
        tbz     x4,#5,StoreOutputPartial16
        stp     q16,q17,[x2],#128
        stp     q18,q19,[x2,#-96]
        stp     q20,q21,[x2,#-64]
        stp     q22,q23,[x2,#-32]

StoreOutputPartial16
        tbz     x4,#4,StoreOutputPartial8
        stp     q24,q25,[x2],#64
        stp     q26,q27,[x2,#-32]

StoreOutputPartial8
        tbz     x4,#3,StoreOutputPartial4
        stp     q28,q29,[x2],#32

StoreOutputPartial4
        tbz     x4,#2,StoreOutputPartial2
        str     q30,[x2],#16

StoreOutputPartial2
        tbz     x4,#1,StoreOutputPartial1
        str     d31,[x2],#8

StoreOutputPartial1
        tbz     x4,#0,ExitKernel
        str     s1,[x2]

ExitKernel
        ret

        LEAF_END MlasGemvFloatKernel

        END
