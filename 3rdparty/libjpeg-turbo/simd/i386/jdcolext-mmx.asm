;
; YCbCr-to-RGB Color Conversion (32-bit MMX)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2016, 2024-2025, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jcolsamp.inc"

; --------------------------------------------------------------------------
;
; Convert some rows of samples to the output colorspace.
;
; GLOBAL(void)
; jsimd_ycc_rgb_convert_mmx(JDIMENSION out_width, JSAMPIMAGE input_buf,
;                           JDIMENSION input_row, JSAMPARRAY output_buf,
;                           int num_rows)

%define out_width(b)   (b) + 8          ; JDIMENSION out_width
%define input_buf(b)   (b) + 12         ; JSAMPIMAGE input_buf
%define input_row(b)   (b) + 16         ; JDIMENSION input_row
%define output_buf(b)  (b) + 20         ; JSAMPARRAY output_buf
%define num_rows(b)    (b) + 24         ; int num_rows

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_MMWORD  ; mmword wk[WK_NUM]
%define WK_NUM        2
%define gotptr        wk(0) - SIZEOF_POINTER  ; void *gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_ycc_rgb_convert_mmx)

EXTN(jsimd_ycc_rgb_convert_mmx):
    push        ebp
    mov         eax, esp                ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_MMWORD)  ; align to 64 bits
    mov         [esp], eax
    mov         ebp, esp                ; ebp = aligned ebp
    lea         esp, [wk(0)]
    PUSHPIC     eax                     ; make a room for GOT address
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    GET_GOT     ebx                     ; get GOT address
    MOVPIC      POINTER [gotptr], ebx   ; save GOT address

    mov         ecx, JDIMENSION [out_width(eax)]  ; num_cols
    test        ecx, ecx
    jz          near .return

    push        ecx

    mov         edi, JSAMPIMAGE [input_buf(eax)]
    mov         ecx, JDIMENSION [input_row(eax)]
    mov         esi, JSAMPARRAY [edi + 0 * SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [edi + 1 * SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [edi + 2 * SIZEOF_JSAMPARRAY]
    lea         esi, [esi + ecx * SIZEOF_JSAMPROW]
    lea         ebx, [ebx + ecx * SIZEOF_JSAMPROW]
    lea         edx, [edx + ecx * SIZEOF_JSAMPROW]

    pop         ecx

    mov         edi, JSAMPARRAY [output_buf(eax)]
    mov         eax, INT [num_rows(eax)]
    test        eax, eax
    jle         near .return
    ALIGNX      16, 7
.rowloop:
    push        eax
    push        edi
    push        edx
    push        ebx
    push        esi
    push        ecx                     ; col

    mov         esi, JSAMPROW [esi]     ; inptr0
    mov         ebx, JSAMPROW [ebx]     ; inptr1
    mov         edx, JSAMPROW [edx]     ; inptr2
    mov         edi, JSAMPROW [edi]     ; outptr
    MOVPIC      eax, POINTER [gotptr]   ; load GOT address (eax)
    ALIGNX      16, 7
.columnloop:

    movq        mm5, MMWORD [ebx]       ; mm5 = Cb(01234567)
    movq        mm1, MMWORD [edx]       ; mm1 = Cr(01234567)

    pcmpeqw     mm4, mm4
    pcmpeqw     mm7, mm7
    psrlw       mm4, BYTE_BIT
    psllw       mm7, 7                 ; mm7 = { 0xFF80 0xFF80 0xFF80 0xFF80 }
    movq        mm0, mm4               ; mm0 = mm4 = { 0xFF 0x00 0xFF 0x00 .. }

    pand        mm4, mm5                ; mm4 = Cb(0246) = CbE
    psrlw       mm5, BYTE_BIT           ; mm5 = Cb(1357) = CbO
    pand        mm0, mm1                ; mm0 = Cr(0246) = CrE
    psrlw       mm1, BYTE_BIT           ; mm1 = Cr(1357) = CrO

    paddw       mm4, mm7
    paddw       mm5, mm7
    paddw       mm0, mm7
    paddw       mm1, mm7

    ; (Original)
    ; R = Y                + 1.40200 * Cr
    ; G = Y - 0.34414 * Cb - 0.71414 * Cr
    ; B = Y + 1.77200 * Cb
    ;
    ; (This implementation)
    ; R = Y                + 0.40200 * Cr + Cr
    ; G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
    ; B = Y - 0.22800 * Cb + Cb + Cb

    movq        mm2, mm4                ; mm2 = CbE
    movq        mm3, mm5                ; mm3 = CbO
    paddw       mm4, mm4                ; mm4 = 2 * CbE
    paddw       mm5, mm5                ; mm5 = 2 * CbO
    movq        mm6, mm0                ; mm6 = CrE
    movq        mm7, mm1                ; mm7 = CrO
    paddw       mm0, mm0                ; mm0 = 2 * CrE
    paddw       mm1, mm1                ; mm1 = 2 * CrO

    pmulhw      mm4, [GOTOFF(eax, PW_MF0228)]
                ; mm4 = (2 * CbE * -FIX(0.22800))
    pmulhw      mm5, [GOTOFF(eax, PW_MF0228)]
                ; mm5 = (2 * CbO * -FIX(0.22800))
    pmulhw      mm0, [GOTOFF(eax, PW_F0402)]
                ; mm0 = (2 * CrE * FIX(0.40200))
    pmulhw      mm1, [GOTOFF(eax, PW_F0402)]
                ; mm1 = (2 * CrO * FIX(0.40200))

    paddw       mm4, [GOTOFF(eax, PW_ONE)]
    paddw       mm5, [GOTOFF(eax, PW_ONE)]
    psraw       mm4, 1                  ; mm4 = (CbE * -FIX(0.22800))
    psraw       mm5, 1                  ; mm5 = (CbO * -FIX(0.22800))
    paddw       mm0, [GOTOFF(eax, PW_ONE)]
    paddw       mm1, [GOTOFF(eax, PW_ONE)]
    psraw       mm0, 1                  ; mm0 = (CrE * FIX(0.40200))
    psraw       mm1, 1                  ; mm1 = (CrO * FIX(0.40200))

    paddw       mm4, mm2
    paddw       mm5, mm3
    paddw       mm4, mm2                ; mm4 = (CbE * FIX(1.77200)) = (B - Y)E
    paddw       mm5, mm3                ; mm5 = (CbO * FIX(1.77200)) = (B - Y)O
    paddw       mm0, mm6                ; mm0 = (CrE * FIX(1.40200)) = (R - Y)E
    paddw       mm1, mm7                ; mm1 = (CrO * FIX(1.40200)) = (R - Y)O

    movq        MMWORD [wk(0)], mm4     ; wk(0) = (B - Y)E
    movq        MMWORD [wk(1)], mm5     ; wk(1) = (B - Y)O

    movq        mm4, mm2
    movq        mm5, mm3
    punpcklwd   mm2, mm6
    punpckhwd   mm4, mm6
    pmaddwd     mm2, [GOTOFF(eax, PW_MF0344_F0285)]
    pmaddwd     mm4, [GOTOFF(eax, PW_MF0344_F0285)]
    punpcklwd   mm3, mm7
    punpckhwd   mm5, mm7
    pmaddwd     mm3, [GOTOFF(eax, PW_MF0344_F0285)]
    pmaddwd     mm5, [GOTOFF(eax, PW_MF0344_F0285)]

    paddd       mm2, [GOTOFF(eax, PD_ONEHALF)]
    paddd       mm4, [GOTOFF(eax, PD_ONEHALF)]
    psrad       mm2, SCALEBITS
    psrad       mm4, SCALEBITS
    paddd       mm3, [GOTOFF(eax, PD_ONEHALF)]
    paddd       mm5, [GOTOFF(eax, PD_ONEHALF)]
    psrad       mm3, SCALEBITS
    psrad       mm5, SCALEBITS

    packssdw    mm2, mm4
                ; mm2 = CbE * -FIX(0.344) + CrE * FIX(0.285)
    packssdw    mm3, mm5
                ; mm3 = CbO * -FIX(0.344) + CrO * FIX(0.285)
    psubw       mm2, mm6
                ; mm2 = CbE * -FIX(0.344) + CrE * -FIX(0.714) = (G - Y)E
    psubw       mm3, mm7
                ; mm3 = CbO * -FIX(0.344) + CrO * -FIX(0.714) = (G - Y)O

    movq        mm5, MMWORD [esi]       ; mm5 = Y(01234567)

    pcmpeqw     mm4, mm4
    psrlw       mm4, BYTE_BIT           ; mm4 = { 0xFF 0x00 0xFF 0x00 .. }
    pand        mm4, mm5                ; mm4 = Y(0246) = YE
    psrlw       mm5, BYTE_BIT           ; mm5 = Y(1357) = YO

    paddw       mm0, mm4           ; mm0 = ((R - Y)E + YE) = RE = (R0 R2 R4 R6)
    paddw       mm1, mm5           ; mm1 = ((R - Y)O + YO) = RO = (R1 R3 R5 R7)
    packuswb    mm0, mm0           ; mm0 = (R0 R2 R4 R6 ** ** ** **)
    packuswb    mm1, mm1           ; mm1 = (R1 R3 R5 R7 ** ** ** **)

    paddw       mm2, mm4           ; mm2 = ((G - Y)E + YE) = GE = (G0 G2 G4 G6)
    paddw       mm3, mm5           ; mm3 = ((G - Y)O + YO) = GO = (G1 G3 G5 G7)
    packuswb    mm2, mm2           ; mm2 = (G0 G2 G4 G6 ** ** ** **)
    packuswb    mm3, mm3           ; mm3 = (G1 G3 G5 G7 ** ** ** **)

    paddw       mm4,  MMWORD [wk(0)]
                ; mm4 = (YE + (B - Y)E) = BE = (B0 B2 B4 B6)
    paddw       mm5,  MMWORD [wk(1)]
                ; mm5 = (YO + (B - Y)O) = BO = (B1 B3 B5 B7)
    packuswb    mm4, mm4                ; mm4 = (B0 B2 B4 B6 ** ** ** **)
    packuswb    mm5, mm5                ; mm5 = (B1 B3 B5 B7 ** ** ** **)

%if RGB_PIXELSIZE == 3  ; ---------------

    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, and C to red, green, and blue.
    ;
    ; mmA = (A0 A2 A4 A6 ** ** ** **) = AE
    ; mmB = (A1 A3 A5 A7 ** ** ** **) = AO
    ; mmC = (B0 B2 B4 B6 ** ** ** **) = BE
    ; mmD = (B1 B3 B5 B7 ** ** ** **) = BO
    ; mmE = (C0 C2 C4 C6 ** ** ** **) = CE
    ; mmF = (C1 C3 C5 C7 ** ** ** **) = CO
    ; mmG = (** ** ** ** ** ** ** **)
    ; mmH = (** ** ** ** ** ** ** **)

    punpcklbw   mmA, mmC                ; mmA = (A0 B0 A2 B2 A4 B4 A6 B6)
    punpcklbw   mmE, mmB                ; mmE = (C0 A1 C2 A3 C4 A5 C6 A7)
    punpcklbw   mmD, mmF                ; mmD = (B1 C1 B3 C3 B5 C5 B7 C7)

    movq        mmG, mmA
    movq        mmH, mmA
    punpcklwd   mmA, mmE                ; mmA = (A0 B0 C0 A1 A2 B2 C2 A3)
    punpckhwd   mmG, mmE                ; mmG = (A4 B4 C4 A5 A6 B6 C6 A7)

    psrlq       mmH, 2 * BYTE_BIT       ; mmH = (A2 B2 A4 B4 A6 B6 -- --)
    psrlq       mmE, 2 * BYTE_BIT       ; mmE = (C2 A3 C4 A5 C6 A7 -- --)

    movq        mmC, mmD
    movq        mmB, mmD
    punpcklwd   mmD, mmH                ; mmD = (B1 C1 A2 B2 B3 C3 A4 B4)
    punpckhwd   mmC, mmH                ; mmC = (B5 C5 A6 B6 B7 C7 -- --)

    psrlq       mmB, 2 * BYTE_BIT       ; mmB = (B3 C3 B5 C5 B7 C7 -- --)

    movq        mmF, mmE
    punpcklwd   mmE, mmB                ; mmE = (C2 A3 B3 C3 C4 A5 B5 C5)
    punpckhwd   mmF, mmB                ; mmF = (C6 A7 B7 C7 -- -- -- --)

    punpckldq   mmA, mmD                ; mmA = (A0 B0 C0 A1 B1 C1 A2 B2)
    punpckldq   mmE, mmG                ; mmE = (C2 A3 B3 C3 A4 B4 C4 A5)
    punpckldq   mmC, mmF                ; mmC = (B5 C5 A6 B6 C6 A7 B7 C7)

    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st16

    movq        MMWORD [edi + 0 * SIZEOF_MMWORD], mmA
    movq        MMWORD [edi + 1 * SIZEOF_MMWORD], mmE
    movq        MMWORD [edi + 2 * SIZEOF_MMWORD], mmC

    sub         ecx, byte SIZEOF_MMWORD
    jz          short .nextrow

    add         esi, byte SIZEOF_MMWORD                  ; inptr0
    add         ebx, byte SIZEOF_MMWORD                  ; inptr1
    add         edx, byte SIZEOF_MMWORD                  ; inptr2
    add         edi, byte RGB_PIXELSIZE * SIZEOF_MMWORD  ; outptr
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st16:
    lea         ecx, [ecx + ecx * 2]    ; imul ecx, RGB_PIXELSIZE
    cmp         ecx, byte 2 * SIZEOF_MMWORD
    jb          short .column_st8
    movq        MMWORD [edi + 0 * SIZEOF_MMWORD], mmA
    movq        MMWORD [edi + 1 * SIZEOF_MMWORD], mmE
    movq        mmA, mmC
    sub         ecx, byte 2 * SIZEOF_MMWORD
    add         edi, byte 2 * SIZEOF_MMWORD
    jmp         short .column_st4
.column_st8:
    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st4
    movq        MMWORD [edi + 0 * SIZEOF_MMWORD], mmA
    movq        mmA, mmE
    sub         ecx, byte SIZEOF_MMWORD
    add         edi, byte SIZEOF_MMWORD
.column_st4:
    movd        eax, mmA
    cmp         ecx, byte SIZEOF_DWORD
    jb          short .column_st2
    mov         dword [edi + 0 * SIZEOF_DWORD], eax
    psrlq       mmA, DWORD_BIT
    movd        eax, mmA
    sub         ecx, byte SIZEOF_DWORD
    add         edi, byte SIZEOF_DWORD
.column_st2:
    cmp         ecx, byte SIZEOF_WORD
    jb          short .column_st1
    mov         word [edi + 0 * SIZEOF_WORD], ax
    shr         eax, WORD_BIT
    sub         ecx, byte SIZEOF_WORD
    add         edi, byte SIZEOF_WORD
.column_st1:
    cmp         ecx, byte SIZEOF_BYTE
    jb          short .nextrow
    mov         byte [edi + 0 * SIZEOF_BYTE], al

%else  ; RGB_PIXELSIZE == 4 ; -----------

%ifdef RGBX_FILLER_0XFF
    pcmpeqb     mm6, mm6                ; mm6 = (X0 X2 X4 X6 ** ** ** **)
    pcmpeqb     mm7, mm7                ; mm7 = (X1 X3 X5 X7 ** ** ** **)
%else
    pxor        mm6, mm6                ; mm6 = (X0 X2 X4 X6 ** ** ** **)
    pxor        mm7, mm7                ; mm7 = (X1 X3 X5 X7 ** ** ** **)
%endif
    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, C, and D to red, green, and blue.
    ;
    ; mmA = (A0 A2 A4 A6 ** ** ** **) = AE
    ; mmB = (A1 A3 A5 A7 ** ** ** **) = AO
    ; mmC = (B0 B2 B4 B6 ** ** ** **) = BE
    ; mmD = (B1 B3 B5 B7 ** ** ** **) = BO
    ; mmE = (C0 C2 C4 C6 ** ** ** **) = CE
    ; mmF = (C1 C3 C5 C7 ** ** ** **) = CO
    ; mmG = (D0 D2 D4 D6 ** ** ** **) = DE
    ; mmH = (D1 D3 D5 D7 ** ** ** **) = DO

    punpcklbw   mmA, mmC                ; mmA = (A0 B0 A2 B2 A4 B4 A6 B6)
    punpcklbw   mmE, mmG                ; mmE = (C0 D0 C2 D2 C4 D4 C6 D6)
    punpcklbw   mmB, mmD                ; mmB = (A1 B1 A3 B3 A5 B5 A7 B7)
    punpcklbw   mmF, mmH                ; mmF = (C1 D1 C3 D3 C5 D5 C7 D7)

    movq        mmC, mmA
    punpcklwd   mmA, mmE                ; mmA = (A0 B0 C0 D0 A2 B2 C2 D2)
    punpckhwd   mmC, mmE                ; mmC = (A4 B4 C4 D4 A6 B6 C6 D6)
    movq        mmG, mmB
    punpcklwd   mmB, mmF                ; mmB = (A1 B1 C1 D1 A3 B3 C3 D3)
    punpckhwd   mmG, mmF                ; mmG = (A5 B5 C5 D5 A7 B7 C7 D7)

    movq        mmD, mmA
    punpckldq   mmA, mmB                ; mmA = (A0 B0 C0 D0 A1 B1 C1 D1)
    punpckhdq   mmD, mmB                ; mmD = (A2 B2 C2 D2 A3 B3 C3 D3)
    movq        mmH, mmC
    punpckldq   mmC, mmG                ; mmC = (A4 B4 C4 D4 A5 B5 C5 D5)
    punpckhdq   mmH, mmG                ; mmH = (A6 B6 C6 D6 A7 B7 C7 D7)

    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st16

    movq        MMWORD [edi + 0 * SIZEOF_MMWORD], mmA
    movq        MMWORD [edi + 1 * SIZEOF_MMWORD], mmD
    movq        MMWORD [edi + 2 * SIZEOF_MMWORD], mmC
    movq        MMWORD [edi + 3 * SIZEOF_MMWORD], mmH

    sub         ecx, byte SIZEOF_MMWORD
    jz          short .nextrow

    add         esi, byte SIZEOF_MMWORD                  ; inptr0
    add         ebx, byte SIZEOF_MMWORD                  ; inptr1
    add         edx, byte SIZEOF_MMWORD                  ; inptr2
    add         edi, byte RGB_PIXELSIZE * SIZEOF_MMWORD  ; outptr
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st16:
    cmp         ecx, byte SIZEOF_MMWORD / 2
    jb          short .column_st8
    movq        MMWORD [edi + 0 * SIZEOF_MMWORD], mmA
    movq        MMWORD [edi + 1 * SIZEOF_MMWORD], mmD
    movq        mmA, mmC
    movq        mmD, mmH
    sub         ecx, byte SIZEOF_MMWORD / 2
    add         edi, byte 2 * SIZEOF_MMWORD
.column_st8:
    cmp         ecx, byte SIZEOF_MMWORD / 4
    jb          short .column_st4
    movq        MMWORD [edi + 0 * SIZEOF_MMWORD], mmA
    movq        mmA, mmD
    sub         ecx, byte SIZEOF_MMWORD / 4
    add         edi, byte 1 * SIZEOF_MMWORD
.column_st4:
    cmp         ecx, byte SIZEOF_MMWORD / 8
    jb          short .nextrow
    movd        dword [edi + 0 * SIZEOF_DWORD], mmA

%endif  ; RGB_PIXELSIZE ; ---------------

    ALIGNX      16, 7

.nextrow:
    pop         ecx
    pop         esi
    pop         ebx
    pop         edx
    pop         edi
    pop         eax

    add         esi, byte SIZEOF_JSAMPROW
    add         ebx, byte SIZEOF_JSAMPROW
    add         edx, byte SIZEOF_JSAMPROW
    add         edi, byte SIZEOF_JSAMPROW  ; output_buf
    dec         eax                        ; num_rows
    jg          near .rowloop

    emms                                ; empty MMX state

.return:
    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    pop         ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
