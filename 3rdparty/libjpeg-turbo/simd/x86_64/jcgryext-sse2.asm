;
; RGB-to-Grayscale Color Conversion (64-bit SSE2)
;
; Copyright (C) 2011, 2016, 2024-2025, D. R. Commander.
; Copyright (C) 2018, Matthias Räncker.
; Copyright (C) 2023, Aliaksiej Kandracienka.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jcolsamp.inc"

; --------------------------------------------------------------------------
;
; Convert some rows of samples to the JPEG colorspace.
;
; GLOBAL(void)
; jsimd_rgb_gray_convert_sse2(JDIMENSION img_width, JSAMPARRAY input_buf,
;                             JSAMPIMAGE output_buf, JDIMENSION output_row,
;                             int num_rows)
;
; r10d = JDIMENSION img_width
; r11 = JSAMPARRAY input_buf
; r12 = JSAMPIMAGE output_buf
; r13d = JDIMENSION output_row
; r14d = int num_rows

%define wk(i)   r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD  ; xmmword wk[WK_NUM]
%define WK_NUM  2

    align       32
    GLOBAL_FUNCTION(jsimd_rgb_gray_convert_sse2)

EXTN(jsimd_rgb_gray_convert_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    push        r15
    and         rsp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    ; Allocate stack space for wk array.  r15 is used to access it.
    mov         r15, rsp
    sub         rsp, byte (SIZEOF_XMMWORD * WK_NUM)
    COLLECT_ARGS 5
    push        rbx

    mov         ecx, r10d
    test        rcx, rcx
    jz          near .return

    push        rcx

    mov         rsi, r12
    mov         ecx, r13d
    mov         rdip, JSAMPARRAY [rsi + 0 * SIZEOF_JSAMPARRAY]
    lea         rdi, [rdi + rcx * SIZEOF_JSAMPROW]

    pop         rcx

    mov         rsi, r11
    mov         eax, r14d
    test        rax, rax
    jle         near .return
.rowloop:
    push        rdi
    push        rsi
    push        rcx                     ; col

    mov         rsip, JSAMPROW [rsi]    ; inptr
    mov         rdip, JSAMPROW [rdi]    ; outptr0

    cmp         rcx, byte SIZEOF_XMMWORD
    jae         near .columnloop

%if RGB_PIXELSIZE == 3  ; ---------------

.column_ld1:
    push        rax
    push        rdx
    lea         rcx, [rcx + rcx * 2]    ; imul ecx, RGB_PIXELSIZE
    test        cl, SIZEOF_BYTE
    jz          short .column_ld2
    sub         rcx, byte SIZEOF_BYTE
    movzx       rax, byte [rsi + rcx]
.column_ld2:
    test        cl, SIZEOF_WORD
    jz          short .column_ld4
    sub         rcx, byte SIZEOF_WORD
    movzx       rdx, word [rsi + rcx]
    shl         rax, WORD_BIT
    or          rax, rdx
.column_ld4:
    movd        xmmA, eax
    pop         rdx
    pop         rax
    test        cl, SIZEOF_DWORD
    jz          short .column_ld8
    sub         rcx, byte SIZEOF_DWORD
    movd        xmmF, XMM_DWORD [rsi + rcx]
    pslldq      xmmA, SIZEOF_DWORD
    por         xmmA, xmmF
.column_ld8:
    test        cl, SIZEOF_MMWORD
    jz          short .column_ld16
    sub         rcx, byte SIZEOF_MMWORD
    movq        xmmB, XMM_MMWORD [rsi + rcx]
    pslldq      xmmA, SIZEOF_MMWORD
    por         xmmA, xmmB
.column_ld16:
    test        cl, SIZEOF_XMMWORD
    jz          short .column_ld32
    movdqa      xmmF, xmmA
    movdqu      xmmA, XMMWORD [rsi + 0 * SIZEOF_XMMWORD]
    mov         rcx, SIZEOF_XMMWORD
    jmp         short .rgb_gray_cnv
.column_ld32:
    test        cl, 2 * SIZEOF_XMMWORD
    mov         rcx, SIZEOF_XMMWORD
    jz          short .rgb_gray_cnv
    movdqa      xmmB, xmmA
    movdqu      xmmA, XMMWORD [rsi + 0 * SIZEOF_XMMWORD]
    movdqu      xmmF, XMMWORD [rsi + 1 * SIZEOF_XMMWORD]
    jmp         short .rgb_gray_cnv

.columnloop:
    movdqu      xmmA, XMMWORD [rsi + 0 * SIZEOF_XMMWORD]
    movdqu      xmmF, XMMWORD [rsi + 1 * SIZEOF_XMMWORD]
    movdqu      xmmB, XMMWORD [rsi + 2 * SIZEOF_XMMWORD]

.rgb_gray_cnv:
    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, and C to red, green, and blue.
    ;
    ; xmmA = (A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4 A5)
    ; xmmF = (B5 C5 A6 B6 C6 A7 B7 C7 A8 B8 C8 A9 B9 C9 Aa Ba)
    ; xmmB = (Ca Ab Bb Cb Ac Bc Cc Ad Bd Cd Ae Be Ce Af Bf Cf)

    movdqa      xmmG, xmmA
    pslldq      xmmA, 8
                ; xmmA = (-- -- -- -- -- -- -- -- A0 B0 C0 A1 B1 C1 A2 B2)
    psrldq      xmmG, 8
                ; xmmG = (C2 A3 B3 C3 A4 B4 C4 A5 -- -- -- -- -- -- -- --)

    punpckhbw   xmmA, xmmF
                ; xmmA = (A0 A8 B0 B8 C0 C8 A1 A9 B1 B9 C1 C9 A2 Aa B2 Ba)
    pslldq      xmmF, 8
                ; xmmF = (-- -- -- -- -- -- -- -- B5 C5 A6 B6 C6 A7 B7 C7)

    punpcklbw   xmmG, xmmB
                ; xmmG = (C2 Ca A3 Ab B3 Bb C3 Cb A4 Ac B4 Bc C4 Cc A5 Ad)
    punpckhbw   xmmF, xmmB
                ; xmmF = (B5 Bd C5 Cd A6 Ae B6 Be C6 Ce A7 Af B7 Bf C7 Cf)

    movdqa      xmmD, xmmA
    pslldq      xmmA, 8
                ; xmmA = (-- -- -- -- -- -- -- -- A0 A8 B0 B8 C0 C8 A1 A9)
    psrldq      xmmD, 8
                ; xmmD = (B1 B9 C1 C9 A2 Aa B2 Ba -- -- -- -- -- -- -- --)

    punpckhbw   xmmA, xmmG
                ; xmmA = (A0 A4 A8 Ac B0 B4 B8 Bc C0 C4 C8 Cc A1 A5 A9 Ad)
    pslldq      xmmG, 8
                ; xmmG = (-- -- -- -- -- -- -- -- C2 Ca A3 Ab B3 Bb C3 Cb)

    punpcklbw   xmmD, xmmF
                ; xmmD = (B1 B5 B9 Bd C1 C5 C9 Cd A2 A6 Aa Ae B2 B6 Ba Be)
    punpckhbw   xmmG, xmmF
                ; xmmG = (C2 C6 Ca Ce A3 A7 Ab Af B3 B7 Bb Bf C3 C7 Cb Cf)

    movdqa      xmmE, xmmA
    pslldq      xmmA, 8
                ; xmmA = (-- -- -- -- -- -- -- -- A0 A4 A8 Ac B0 B4 B8 Bc)
    psrldq      xmmE, 8
                ; xmmE = (C0 C4 C8 Cc A1 A5 A9 Ad -- -- -- -- -- -- -- --)

    punpckhbw   xmmA, xmmD
                ; xmmA = (A0 A2 A4 A6 A8 Aa Ac Ae B0 B2 B4 B6 B8 Ba Bc Be)
    pslldq      xmmD, 8
                ; xmmD = (-- -- -- -- -- -- -- -- B1 B5 B9 Bd C1 C5 C9 Cd)

    punpcklbw   xmmE, xmmG
                ; xmmE = (C0 C2 C4 C6 C8 Ca Cc Ce A1 A3 A5 A7 A9 Ab Ad Af)
    punpckhbw   xmmD, xmmG
                ; xmmD = (B1 B3 B5 B7 B9 Bb Bd Bf C1 C3 C5 C7 C9 Cb Cd Cf)

    pxor        xmmH, xmmH

    movdqa      xmmC, xmmA
    punpcklbw   xmmA, xmmH              ; xmmA = (A0 A2 A4 A6 A8 Aa Ac Ae) = AE
    punpckhbw   xmmC, xmmH              ; xmmC = (B0 B2 B4 B6 B8 Ba Bc Be) = BE

    movdqa      xmmB, xmmE
    punpcklbw   xmmE, xmmH              ; xmmE = (C0 C2 C4 C6 C8 Ca Cc Ce) = CE
    punpckhbw   xmmB, xmmH              ; xmmB = (A1 A3 A5 A7 A9 Ab Ad Af) = AO

    movdqa      xmmF, xmmD
    punpcklbw   xmmD, xmmH              ; xmmD = (B1 B3 B5 B7 B9 Bb Bd Bf) = BO
    punpckhbw   xmmF, xmmH              ; xmmF = (C1 C3 C5 C7 C9 Cb Cd Cf) = CO

%else  ; RGB_PIXELSIZE == 4 ; -----------

.column_ld1:
    test        cl, SIZEOF_XMMWORD / 16
    jz          short .column_ld2
    sub         rcx, byte SIZEOF_XMMWORD / 16
    movd        xmmA, XMM_DWORD [rsi + rcx * RGB_PIXELSIZE]
.column_ld2:
    test        cl, SIZEOF_XMMWORD / 8
    jz          short .column_ld4
    sub         rcx, byte SIZEOF_XMMWORD / 8
    movq        xmmE, XMM_MMWORD [rsi + rcx * RGB_PIXELSIZE]
    pslldq      xmmA, SIZEOF_MMWORD
    por         xmmA, xmmE
.column_ld4:
    test        cl, SIZEOF_XMMWORD / 4
    jz          short .column_ld8
    sub         rcx, byte SIZEOF_XMMWORD / 4
    movdqa      xmmE, xmmA
    movdqu      xmmA, XMMWORD [rsi + rcx * RGB_PIXELSIZE]
.column_ld8:
    test        cl, SIZEOF_XMMWORD / 2
    mov         rcx, SIZEOF_XMMWORD
    jz          short .rgb_gray_cnv
    movdqa      xmmF, xmmA
    movdqa      xmmH, xmmE
    movdqu      xmmA, XMMWORD [rsi + 0 * SIZEOF_XMMWORD]
    movdqu      xmmE, XMMWORD [rsi + 1 * SIZEOF_XMMWORD]
    jmp         short .rgb_gray_cnv

.columnloop:
    movdqu      xmmA, XMMWORD [rsi + 0 * SIZEOF_XMMWORD]
    movdqu      xmmE, XMMWORD [rsi + 1 * SIZEOF_XMMWORD]
    movdqu      xmmF, XMMWORD [rsi + 2 * SIZEOF_XMMWORD]
    movdqu      xmmH, XMMWORD [rsi + 3 * SIZEOF_XMMWORD]

.rgb_gray_cnv:
    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, C, and D to red, green, and blue.
    ;
    ; xmmA = (A0 B0 C0 D0 A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3)
    ; xmmE = (A4 B4 C4 D4 A5 B5 C5 D5 A6 B6 C6 D6 A7 B7 C7 D7)
    ; xmmF = (A8 B8 C8 D8 A9 B9 C9 D9 Aa Ba Ca Da Ab Bb Cb Db)
    ; xmmH = (Ac Bc Cc Dc Ad Bd Cd Dd Ae Be Ce De Af Bf Cf Df)

    movdqa      xmmD, xmmA
    punpcklbw   xmmA, xmmE
                ; xmmA = (A0 A4 B0 B4 C0 C4 D0 D4 A1 A5 B1 B5 C1 C5 D1 D5)
    punpckhbw   xmmD, xmmE
                ; xmmD = (A2 A6 B2 B6 C2 C6 D2 D6 A3 A7 B3 B7 C3 C7 D3 D7)

    movdqa      xmmC, xmmF
    punpcklbw   xmmF, xmmH
                ; xmmF = (A8 Ac B8 Bc C8 Cc D8 Dc A9 Ad B9 Bd C9 Cd D9 Dd)
    punpckhbw   xmmC, xmmH
                ; xmmC = (Aa Ae Ba Be Ca Ce Da De Ab Af Bb Bf Cb Cf Db Df)

    movdqa      xmmB, xmmA
    punpcklwd   xmmA, xmmF
                ; xmmA = (A0 A4 A8 Ac B0 B4 B8 Bc C0 C4 C8 Cc D0 D4 D8 Dc)
    punpckhwd   xmmB, xmmF
                ; xmmB = (A1 A5 A9 Ad B1 B5 B9 Bd C1 C5 C9 Cd D1 D5 D9 Dd)

    movdqa      xmmG, xmmD
    punpcklwd   xmmD, xmmC
                ; xmmD = (A2 A6 Aa Ae B2 B6 Ba Be C2 C6 Ca Ce D2 D6 Da De)
    punpckhwd   xmmG, xmmC
                ; xmmG = (A3 A7 Ab Af B3 B7 Bb Bf C3 C7 Cb Cf D3 D7 Db Df)

    movdqa      xmmE, xmmA
    punpcklbw   xmmA, xmmD
                ; xmmA = (A0 A2 A4 A6 A8 Aa Ac Ae B0 B2 B4 B6 B8 Ba Bc Be)
    punpckhbw   xmmE, xmmD
                ; xmmE = (C0 C2 C4 C6 C8 Ca Cc Ce D0 D2 D4 D6 D8 Da Dc De)

    movdqa      xmmH, xmmB
    punpcklbw   xmmB, xmmG
                ; xmmB = (A1 A3 A5 A7 A9 Ab Ad Af B1 B3 B5 B7 B9 Bb Bd Bf)
    punpckhbw   xmmH, xmmG
                ; xmmH = (C1 C3 C5 C7 C9 Cb Cd Cf D1 D3 D5 D7 D9 Db Dd Df)

    pxor        xmmF, xmmF

    movdqa      xmmC, xmmA
    punpcklbw   xmmA, xmmF              ; xmmA = (A0 A2 A4 A6 A8 Aa Ac Ae) = AE
    punpckhbw   xmmC, xmmF              ; xmmC = (B0 B2 B4 B6 B8 Ba Bc Be) = BE

    movdqa      xmmD, xmmB
    punpcklbw   xmmB, xmmF              ; xmmB = (A1 A3 A5 A7 A9 Ab Ad Af) = AO
    punpckhbw   xmmD, xmmF              ; xmmD = (B1 B3 B5 B7 B9 Bb Bd Bf) = BO

    movdqa      xmmG, xmmE
    punpcklbw   xmmE, xmmF              ; xmmE = (C0 C2 C4 C6 C8 Ca Cc Ce) = CE
    punpckhbw   xmmG, xmmF              ; xmmG = (D0 D2 D4 D6 D8 Da Dc De) = DE

    punpcklbw   xmmF, xmmH
    punpckhbw   xmmH, xmmH
    psrlw       xmmF, BYTE_BIT          ; xmmF = (C1 C3 C5 C7 C9 Cb Cd Cf) = CO
    psrlw       xmmH, BYTE_BIT          ; xmmH = (D1 D3 D5 D7 D9 Db Dd Df) = DO

%endif  ; RGB_PIXELSIZE ; ---------------

    ; xmm0 = (R0 R2 R4 R6 R8 Ra Rc Re) = RE
    ; xmm2 = (G0 G2 G4 G6 G8 Ga Gc Ge) = GE
    ; xmm4 = (B0 B2 B4 B6 B8 Ba Bc Be) = BE
    ; xmm1 = (R1 R3 R5 R7 R9 Rb Rd Rf) = RO
    ; xmm3 = (G1 G3 G5 G7 G9 Gb Gd Gf) = GO
    ; xmm5 = (B1 B3 B5 B7 B9 Bb Bd Bf) = BO
    ;
    ; (Original)
    ; Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    ;
    ; (This implementation)
    ; Y  =  0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G

    movdqa      xmm6, xmm1
    punpcklwd   xmm1, xmm3
    punpckhwd   xmm6, xmm3
    pmaddwd     xmm1, [rel PW_F0299_F0337]
                ; xmm1 = ROL * FIX(0.299) + GOL * FIX(0.337)
    pmaddwd     xmm6, [rel PW_F0299_F0337]
                ; xmm6 = ROH * FIX(0.299) + GOH * FIX(0.337)

    movdqa      xmm7, xmm6
                ; xmm7 = ROH * FIX(0.299) + GOH * FIX(0.337)

    movdqa      xmm6, xmm0
    punpcklwd   xmm0, xmm2
    punpckhwd   xmm6, xmm2
    pmaddwd     xmm0, [rel PW_F0299_F0337]
                ; xmm0 = REL * FIX(0.299) + GEL * FIX(0.337)
    pmaddwd     xmm6, [rel PW_F0299_F0337]
                ; xmm6 = REH * FIX(0.299) + GEH * FIX(0.337)

    movdqa      XMMWORD [wk(0)], xmm0
                ; wk(0) = REL * FIX(0.299) + GEL * FIX(0.337)
    movdqa      XMMWORD [wk(1)], xmm6
                ; wk(1) = REH * FIX(0.299) + GEH * FIX(0.337)

    movdqa      xmm0, xmm5              ; xmm0 = BO
    movdqa      xmm6, xmm4              ; xmm6 = BE

    movdqa      xmm4, xmm0
    punpcklwd   xmm0, xmm3
    punpckhwd   xmm4, xmm3
    pmaddwd     xmm0, [rel PW_F0114_F0250]
                ; xmm0 = BOL * FIX(0.114) + GOL * FIX(0.250)
    pmaddwd     xmm4, [rel PW_F0114_F0250]
                ; xmm4 = BOH * FIX(0.114) + GOH * FIX(0.250)

    movdqa      xmm3, [rel PD_ONEHALF]  ; xmm3 = [PD_ONEHALF]

    paddd       xmm0, xmm1
    paddd       xmm4, xmm7
    paddd       xmm0, xmm3
    paddd       xmm4, xmm3
    psrld       xmm0, SCALEBITS         ; xmm0 = YOL
    psrld       xmm4, SCALEBITS         ; xmm4 = YOH
    packssdw    xmm0, xmm4              ; xmm0 = YO

    movdqa      xmm4, xmm6
    punpcklwd   xmm6, xmm2
    punpckhwd   xmm4, xmm2
    pmaddwd     xmm6, [rel PW_F0114_F0250]
                ; xmm6 = BEL * FIX(0.114) + GEL * FIX(0.250)
    pmaddwd     xmm4, [rel PW_F0114_F0250]
                ; xmm4 = BEH * FIX(0.114) + GEH * FIX(0.250)

    movdqa      xmm2, [rel PD_ONEHALF]  ; xmm2 = [PD_ONEHALF]

    paddd       xmm6, XMMWORD [wk(0)]
    paddd       xmm4, XMMWORD [wk(1)]
    paddd       xmm6, xmm2
    paddd       xmm4, xmm2
    psrld       xmm6, SCALEBITS         ; xmm6 = YEL
    psrld       xmm4, SCALEBITS         ; xmm4 = YEH
    packssdw    xmm6, xmm4              ; xmm6 = YE

    psllw       xmm0, BYTE_BIT
    por         xmm6, xmm0              ; xmm6 = Y
    movdqa      XMMWORD [rdi], xmm6     ; Save Y

    sub         rcx, byte SIZEOF_XMMWORD
    add         rsi, byte RGB_PIXELSIZE * SIZEOF_XMMWORD  ; inptr
    add         rdi, byte SIZEOF_XMMWORD                  ; outptr0
    cmp         rcx, byte SIZEOF_XMMWORD
    jae         near .columnloop
    test        rcx, rcx
    jnz         near .column_ld1

    pop         rcx                     ; col
    pop         rsi
    pop         rdi

    add         rsi, byte SIZEOF_JSAMPROW  ; input_buf
    add         rdi, byte SIZEOF_JSAMPROW
    dec         rax                        ; num_rows
    jg          near .rowloop

.return:
    pop         rbx
    UNCOLLECT_ARGS 5
    lea         rsp, [rbp - 8]
    pop         r15
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
