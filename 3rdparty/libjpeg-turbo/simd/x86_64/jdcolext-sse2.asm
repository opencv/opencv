;
; YCbCr-to-RGB Color Conversion (64-bit SSE2)
;
; Copyright 2009, 2012 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2012, 2016, 2024-2025, D. R. Commander.
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
; Convert some rows of samples to the output colorspace.
;
; GLOBAL(void)
; jsimd_ycc_rgb_convert_sse2(JDIMENSION out_width, JSAMPIMAGE input_buf,
;                            JDIMENSION input_row, JSAMPARRAY output_buf,
;                            int num_rows)
;
; r10d = JDIMENSION out_width
; r11 = JSAMPIMAGE input_buf
; r12d = JDIMENSION input_row
; r13 = JSAMPARRAY output_buf
; r14d = int num_rows

%define wk(i)   r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD  ; xmmword wk[WK_NUM]
%define WK_NUM  2

    align       32
    GLOBAL_FUNCTION(jsimd_ycc_rgb_convert_sse2)

EXTN(jsimd_ycc_rgb_convert_sse2):
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

    mov         ecx, r10d               ; num_cols
    test        rcx, rcx
    jz          near .return

    push        rcx

    mov         rdi, r11
    mov         ecx, r12d
    mov         rsip, JSAMPARRAY [rdi + 0 * SIZEOF_JSAMPARRAY]
    mov         rbxp, JSAMPARRAY [rdi + 1 * SIZEOF_JSAMPARRAY]
    mov         rdxp, JSAMPARRAY [rdi + 2 * SIZEOF_JSAMPARRAY]
    lea         rsi, [rsi + rcx * SIZEOF_JSAMPROW]
    lea         rbx, [rbx + rcx * SIZEOF_JSAMPROW]
    lea         rdx, [rdx + rcx * SIZEOF_JSAMPROW]

    pop         rcx

    mov         rdi, r13
    mov         eax, r14d
    test        rax, rax
    jle         near .return
.rowloop:
    push        rax
    push        rdi
    push        rdx
    push        rbx
    push        rsi
    push        rcx                     ; col

    mov         rsip, JSAMPROW [rsi]    ; inptr0
    mov         rbxp, JSAMPROW [rbx]    ; inptr1
    mov         rdxp, JSAMPROW [rdx]    ; inptr2
    mov         rdip, JSAMPROW [rdi]    ; outptr
.columnloop:

    movdqa      xmm5, XMMWORD [rbx]     ; xmm5 = Cb(0123456789abcdef)
    movdqa      xmm1, XMMWORD [rdx]     ; xmm1 = Cr(0123456789abcdef)

    pcmpeqw     xmm4, xmm4
    pcmpeqw     xmm7, xmm7
    psrlw       xmm4, BYTE_BIT
    psllw       xmm7, 7             ; xmm7 = { 0xFF80 0xFF80 0xFF80 0xFF80 .. }
    movdqa      xmm0, xmm4          ; xmm0 = xmm4 = { 0xFF 0x00 0xFF 0x00 .. }

    pand        xmm4, xmm5              ; xmm4 = Cb(02468ace) = CbE
    psrlw       xmm5, BYTE_BIT          ; xmm5 = Cb(13579bdf) = CbO
    pand        xmm0, xmm1              ; xmm0 = Cr(02468ace) = CrE
    psrlw       xmm1, BYTE_BIT          ; xmm1 = Cr(13579bdf) = CrO

    paddw       xmm4, xmm7
    paddw       xmm5, xmm7
    paddw       xmm0, xmm7
    paddw       xmm1, xmm7

    ; (Original)
    ; R = Y                + 1.40200 * Cr
    ; G = Y - 0.34414 * Cb - 0.71414 * Cr
    ; B = Y + 1.77200 * Cb
    ;
    ; (This implementation)
    ; R = Y                + 0.40200 * Cr + Cr
    ; G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
    ; B = Y - 0.22800 * Cb + Cb + Cb

    movdqa      xmm2, xmm4              ; xmm2 = CbE
    movdqa      xmm3, xmm5              ; xmm3 = CbO
    paddw       xmm4, xmm4              ; xmm4 = 2 * CbE
    paddw       xmm5, xmm5              ; xmm5 = 2 * CbO
    movdqa      xmm6, xmm0              ; xmm6 = CrE
    movdqa      xmm7, xmm1              ; xmm7 = CrO
    paddw       xmm0, xmm0              ; xmm0 = 2 * CrE
    paddw       xmm1, xmm1              ; xmm1 = 2 * CrO

    pmulhw      xmm4, [rel PW_MF0228]   ; xmm4 = (2 * CbE * -FIX(0.22800))
    pmulhw      xmm5, [rel PW_MF0228]   ; xmm5 = (2 * CbO * -FIX(0.22800))
    pmulhw      xmm0, [rel PW_F0402]    ; xmm0 = (2 * CrE * FIX(0.40200))
    pmulhw      xmm1, [rel PW_F0402]    ; xmm1 = (2 * CrO * FIX(0.40200))

    paddw       xmm4, [rel PW_ONE]
    paddw       xmm5, [rel PW_ONE]
    psraw       xmm4, 1                 ; xmm4 = (CbE * -FIX(0.22800))
    psraw       xmm5, 1                 ; xmm5 = (CbO * -FIX(0.22800))
    paddw       xmm0, [rel PW_ONE]
    paddw       xmm1, [rel PW_ONE]
    psraw       xmm0, 1                 ; xmm0 = (CrE * FIX(0.40200))
    psraw       xmm1, 1                 ; xmm1 = (CrO * FIX(0.40200))

    paddw       xmm4, xmm2
    paddw       xmm5, xmm3
    paddw       xmm4, xmm2             ; xmm4 = (CbE * FIX(1.77200)) = (B - Y)E
    paddw       xmm5, xmm3             ; xmm5 = (CbO * FIX(1.77200)) = (B - Y)O
    paddw       xmm0, xmm6             ; xmm0 = (CrE * FIX(1.40200)) = (R - Y)E
    paddw       xmm1, xmm7             ; xmm1 = (CrO * FIX(1.40200)) = (R - Y)O

    movdqa      XMMWORD [wk(0)], xmm4   ; wk(0) = (B - Y)E
    movdqa      XMMWORD [wk(1)], xmm5   ; wk(1) = (B - Y)O

    movdqa      xmm4, xmm2
    movdqa      xmm5, xmm3
    punpcklwd   xmm2, xmm6
    punpckhwd   xmm4, xmm6
    pmaddwd     xmm2, [rel PW_MF0344_F0285]
    pmaddwd     xmm4, [rel PW_MF0344_F0285]
    punpcklwd   xmm3, xmm7
    punpckhwd   xmm5, xmm7
    pmaddwd     xmm3, [rel PW_MF0344_F0285]
    pmaddwd     xmm5, [rel PW_MF0344_F0285]

    paddd       xmm2, [rel PD_ONEHALF]
    paddd       xmm4, [rel PD_ONEHALF]
    psrad       xmm2, SCALEBITS
    psrad       xmm4, SCALEBITS
    paddd       xmm3, [rel PD_ONEHALF]
    paddd       xmm5, [rel PD_ONEHALF]
    psrad       xmm3, SCALEBITS
    psrad       xmm5, SCALEBITS

    packssdw    xmm2, xmm4
                ; xmm2 = CbE * -FIX(0.344) + CrE * FIX(0.285)
    packssdw    xmm3, xmm5
                ; xmm3 = CbO * -FIX(0.344) + CrO * FIX(0.285)
    psubw       xmm2, xmm6
                ; xmm2 = CbE * -FIX(0.344) + CrE * -FIX(0.714) = (G - Y)E
    psubw       xmm3, xmm7
                ; xmm3 = CbO * -FIX(0.344) + CrO * -FIX(0.714) = (G - Y)O

    movdqa      xmm5, XMMWORD [rsi]     ; xmm5 = Y(0123456789abcdef)

    pcmpeqw     xmm4, xmm4
    psrlw       xmm4, BYTE_BIT          ; xmm4 = { 0xFF 0x00 0xFF 0x00 .. }
    pand        xmm4, xmm5              ; xmm4 = Y(02468ace) = YE
    psrlw       xmm5, BYTE_BIT          ; xmm5 = Y(13579bdf) = YO

    paddw       xmm0, xmm4          ; xmm0 = ((R - Y)E + YE) = RE = R(02468ace)
    paddw       xmm1, xmm5          ; xmm1 = ((R - Y)O + YO) = RO = R(13579bdf)
    packuswb    xmm0, xmm0          ; xmm0 = R(02468ace********)
    packuswb    xmm1, xmm1          ; xmm1 = R(13579bdf********)

    paddw       xmm2, xmm4          ; xmm2 = ((G - Y)E + YE) = GE = G(02468ace)
    paddw       xmm3, xmm5          ; xmm3 = ((G - Y)O + YO) = GO = G(13579bdf)
    packuswb    xmm2, xmm2          ; xmm2 = G(02468ace********)
    packuswb    xmm3, xmm3          ; xmm3 = G(13579bdf********)

    paddw       xmm4, XMMWORD [wk(0)]
                ; xmm4 = (YE + (B - Y)E) = BE = B(02468ace)
    paddw       xmm5, XMMWORD [wk(1)]
                ; xmm5 = (YO + (B - Y)O) = BO = B(13579bdf)
    packuswb    xmm4, xmm4              ; xmm4 = B(02468ace********)
    packuswb    xmm5, xmm5              ; xmm5 = B(13579bdf********)

%if RGB_PIXELSIZE == 3  ; ---------------

    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, and C to red, green, and blue.
    ;
    ; xmmA = (A0 A2 A4 A6 A8 Aa Ac Ae) = AE
    ; xmmB = (A1 A3 A5 A7 A9 Ab Ad Af) = AO
    ; xmmC = (B0 B2 B4 B6 B8 Ba Bc Be) = BE
    ; xmmD = (B1 B3 B5 B7 B9 Bb Bd Bf) = BO
    ; xmmE = (C0 C2 C4 C6 C8 Ca Cc Ce) = CE
    ; xmmF = (C1 C3 C5 C7 C9 Cb Cd Cf) = CO
    ; xmmG = (** ** ** ** ** ** ** **)
    ; xmmH = (** ** ** ** ** ** ** **)

    punpcklbw   xmmA, xmmC
                ; xmmA = (A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 Aa Ba Ac Bc Ae Be)
    punpcklbw   xmmE, xmmB
                ; xmmE = (C0 A1 C2 A3 C4 A5 C6 A7 C8 A9 Ca Ab Cc Ad Ce Af)
    punpcklbw   xmmD, xmmF
                ; xmmD = (B1 C1 B3 C3 B5 C5 B7 C7 B9 C9 Bb Cb Bd Cd Bf Cf)

    movdqa      xmmG, xmmA
    movdqa      xmmH, xmmA
    punpcklwd   xmmA, xmmE
                ; xmmA = (A0 B0 C0 A1 A2 B2 C2 A3 A4 B4 C4 A5 A6 B6 C6 A7)
    punpckhwd   xmmG, xmmE
                ; xmmG = (A8 B8 C8 A9 Aa Ba Ca Ab Ac Bc Cc Ad Ae Be Ce Af)

    psrldq      xmmH, 2
                ; xmmH = (A2 B2 A4 B4 A6 B6 A8 B8 Aa Ba Ac Bc Ae Be -- --)
    psrldq      xmmE, 2
                ; xmmE = (C2 A3 C4 A5 C6 A7 C8 A9 Ca Ab Cc Ad Ce Af -- --)

    movdqa      xmmC, xmmD
    movdqa      xmmB, xmmD
    punpcklwd   xmmD, xmmH
                ; xmmD = (B1 C1 A2 B2 B3 C3 A4 B4 B5 C5 A6 B6 B7 C7 A8 B8)
    punpckhwd   xmmC, xmmH
                ; xmmC = (B9 C9 Aa Ba Bb Cb Ac Bc Bd Cd Ae Be Bf Cf -- --)

    psrldq      xmmB, 2
                ; xmmB = (B3 C3 B5 C5 B7 C7 B9 C9 Bb Cb Bd Cd Bf Cf -- --)

    movdqa      xmmF, xmmE
    punpcklwd   xmmE, xmmB
                ; xmmE = (C2 A3 B3 C3 C4 A5 B5 C5 C6 A7 B7 C7 C8 A9 B9 C9)
    punpckhwd   xmmF, xmmB
                ; xmmF = (Ca Ab Bb Cb Cc Ad Bd Cd Ce Af Bf Cf -- -- -- --)

    pshufd      xmmH, xmmA, 0x4E
                ; xmmH = (A4 B4 C4 A5 A6 B6 C6 A7 A0 B0 C0 A1 A2 B2 C2 A3)
    movdqa      xmmB, xmmE
    punpckldq   xmmA, xmmD
                ; xmmA = (A0 B0 C0 A1 B1 C1 A2 B2 A2 B2 C2 A3 B3 C3 A4 B4)
    punpckldq   xmmE, xmmH
                ; xmmE = (C2 A3 B3 C3 A4 B4 C4 A5 C4 A5 B5 C5 A6 B6 C6 A7)
    punpckhdq   xmmD, xmmB
                ; xmmD = (B5 C5 A6 B6 C6 A7 B7 C7 B7 C7 A8 B8 C8 A9 B9 C9)

    pshufd      xmmH, xmmG, 0x4E
                ; xmmH = (Ac Bc Cc Ad Ae Be Ce Af A8 B8 C8 A9 Aa Ba Ca Ab)
    movdqa      xmmB, xmmF
    punpckldq   xmmG, xmmC
                ; xmmG = (A8 B8 C8 A9 B9 C9 Aa Ba Aa Ba Ca Ab Bb Cb Ac Bc)
    punpckldq   xmmF, xmmH
                ; xmmF = (Ca Ab Bb Cb Ac Bc Cc Ad Cc Ad Bd Cd Ae Be Ce Af)
    punpckhdq   xmmC, xmmB
                ; xmmC = (Bd Cd Ae Be Ce Af Bf Cf Bf Cf -- -- -- -- -- --)

    punpcklqdq  xmmA, xmmE
                ; xmmA = (A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4 A5)
    punpcklqdq  xmmD, xmmG
                ; xmmD = (B5 C5 A6 B6 C6 A7 B7 C7 A8 B8 C8 A9 B9 C9 Aa Ba)
    punpcklqdq  xmmF, xmmC
                ; xmmF = (Ca Ab Bb Cb Ac Bc Cc Ad Bd Cd Ae Be Ce Af Bf Cf)

    cmp         rcx, byte SIZEOF_XMMWORD
    jb          short .column_st32

    test        rdi, SIZEOF_XMMWORD - 1
    jnz         short .out1
    ; --(aligned)-------------------
    movntdq     XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    movntdq     XMMWORD [rdi + 1 * SIZEOF_XMMWORD], xmmD
    movntdq     XMMWORD [rdi + 2 * SIZEOF_XMMWORD], xmmF
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    movdqu      XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi + 1 * SIZEOF_XMMWORD], xmmD
    movdqu      XMMWORD [rdi + 2 * SIZEOF_XMMWORD], xmmF
.out0:
    add         rdi, byte RGB_PIXELSIZE * SIZEOF_XMMWORD  ; outptr
    sub         rcx, byte SIZEOF_XMMWORD
    jz          near .nextrow

    add         rsi, byte SIZEOF_XMMWORD  ; inptr0
    add         rbx, byte SIZEOF_XMMWORD  ; inptr1
    add         rdx, byte SIZEOF_XMMWORD  ; inptr2
    jmp         near .columnloop

.column_st32:
    lea         rcx, [rcx + rcx * 2]          ; imul ecx, RGB_PIXELSIZE
    cmp         rcx, byte 2 * SIZEOF_XMMWORD
    jb          short .column_st16
    movdqu      XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi + 1 * SIZEOF_XMMWORD], xmmD
    add         rdi, byte 2 * SIZEOF_XMMWORD  ; outptr
    movdqa      xmmA, xmmF
    sub         rcx, byte 2 * SIZEOF_XMMWORD
    jmp         short .column_st15
.column_st16:
    cmp         rcx, byte SIZEOF_XMMWORD
    jb          short .column_st15
    movdqu      XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    add         rdi, byte SIZEOF_XMMWORD      ; outptr
    movdqa      xmmA, xmmD
    sub         rcx, byte SIZEOF_XMMWORD
.column_st15:
    ; Store the lower 8 bytes of xmmA to the output when it has enough
    ; space.
    cmp         rcx, byte SIZEOF_MMWORD
    jb          short .column_st7
    movq        XMM_MMWORD [rdi], xmmA
    add         rdi, byte SIZEOF_MMWORD
    sub         rcx, byte SIZEOF_MMWORD
    psrldq      xmmA, SIZEOF_MMWORD
.column_st7:
    ; Store the lower 4 bytes of xmmA to the output when it has enough
    ; space.
    cmp         rcx, byte SIZEOF_DWORD
    jb          short .column_st3
    movd        XMM_DWORD [rdi], xmmA
    add         rdi, byte SIZEOF_DWORD
    sub         rcx, byte SIZEOF_DWORD
    psrldq      xmmA, SIZEOF_DWORD
.column_st3:
    ; Store the lower 2 bytes of rax to the output when it has enough
    ; space.
    movd        eax, xmmA
    cmp         rcx, byte SIZEOF_WORD
    jb          short .column_st1
    mov         word [rdi], ax
    add         rdi, byte SIZEOF_WORD
    sub         rcx, byte SIZEOF_WORD
    shr         rax, 16
.column_st1:
    ; Store the lower 1 byte of rax to the output when it has enough
    ; space.
    test        rcx, rcx
    jz          short .nextrow
    mov         byte [rdi], al

%else  ; RGB_PIXELSIZE == 4 ; -----------

%ifdef RGBX_FILLER_0XFF
    pcmpeqb     xmm6, xmm6              ; xmm6 = XE = X(02468ace********)
    pcmpeqb     xmm7, xmm7              ; xmm7 = XO = X(13579bdf********)
%else
    pxor        xmm6, xmm6              ; xmm6 = XE = X(02468ace********)
    pxor        xmm7, xmm7              ; xmm7 = XO = X(13579bdf********)
%endif
    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, C, and D to red, green, and blue.
    ;
    ; xmmA = (A0 A2 A4 A6 A8 Aa Ac Ae) = AE
    ; xmmB = (A1 A3 A5 A7 A9 Ab Ad Af) = AO
    ; xmmC = (B0 B2 B4 B6 B8 Ba Bc Be) = BE
    ; xmmD = (B1 B3 B5 B7 B9 Bb Bd Bf) = BO
    ; xmmE = (C0 C2 C4 C6 C8 Ca Cc Ce) = CE
    ; xmmF = (C1 C3 C5 C7 C9 Cb Cd Cf) = CO
    ; xmmG = (D0 D2 D4 D6 D8 Da Dc De) = DE
    ; xmmH = (D1 D3 D5 D7 D9 Db Dd Df) = DO

    punpcklbw   xmmA, xmmC
                ; xmmA = (A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 Aa Ba Ac Bc Ae Be)
    punpcklbw   xmmE, xmmG
                ; xmmE = (C0 D0 C2 D2 C4 D4 C6 D6 C8 D8 Ca Da Cc Dc Ce De)
    punpcklbw   xmmB, xmmD
                ; xmmB = (A1 B1 A3 B3 A5 B5 A7 B7 A9 B9 Ab Bb Ad Bd Af Bf)
    punpcklbw   xmmF, xmmH
                ; xmmF = (C1 D1 C3 D3 C5 D5 C7 D7 C9 D9 Cb Db Cd Dd Cf Df)

    movdqa      xmmC, xmmA
    punpcklwd   xmmA, xmmE
                ; xmmA = (A0 B0 C0 D0 A2 B2 C2 D2 A4 B4 C4 D4 A6 B6 C6 D6)
    punpckhwd   xmmC, xmmE
                ; xmmC = (A8 B8 C8 D8 Aa Ba Ca Da Ac Bc Cc Dc Ae Be Ce De)
    movdqa      xmmG, xmmB
    punpcklwd   xmmB, xmmF
                ; xmmB = (A1 B1 C1 D1 A3 B3 C3 D3 A5 B5 C5 D5 A7 B7 C7 D7)
    punpckhwd   xmmG, xmmF
                ; xmmG = (A9 B9 C9 D9 Ab Bb Cb Db Ad Bd Cd Dd Af Bf Cf Df)

    movdqa      xmmD, xmmA
    punpckldq   xmmA, xmmB
                ; xmmA = (A0 B0 C0 D0 A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3)
    punpckhdq   xmmD, xmmB
                ; xmmD = (A4 B4 C4 D4 A5 B5 C5 D5 A6 B6 C6 D6 A7 B7 C7 D7)
    movdqa      xmmH, xmmC
    punpckldq   xmmC, xmmG
                ; xmmC = (A8 B8 C8 D8 A9 B9 C9 D9 Aa Ba Ca Da Ab Bb Cb Db)
    punpckhdq   xmmH, xmmG
                ; xmmH = (Ac Bc Cc Dc Ad Bd Cd Dd Ae Be Ce De Af Bf Cf Df)

    cmp         rcx, byte SIZEOF_XMMWORD
    jb          short .column_st32

    test        rdi, SIZEOF_XMMWORD - 1
    jnz         short .out1
    ; --(aligned)-------------------
    movntdq     XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    movntdq     XMMWORD [rdi + 1 * SIZEOF_XMMWORD], xmmD
    movntdq     XMMWORD [rdi + 2 * SIZEOF_XMMWORD], xmmC
    movntdq     XMMWORD [rdi + 3 * SIZEOF_XMMWORD], xmmH
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    movdqu      XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi + 1 * SIZEOF_XMMWORD], xmmD
    movdqu      XMMWORD [rdi + 2 * SIZEOF_XMMWORD], xmmC
    movdqu      XMMWORD [rdi + 3 * SIZEOF_XMMWORD], xmmH
.out0:
    add         rdi, byte RGB_PIXELSIZE * SIZEOF_XMMWORD  ; outptr
    sub         rcx, byte SIZEOF_XMMWORD
    jz          near .nextrow

    add         rsi, byte SIZEOF_XMMWORD  ; inptr0
    add         rbx, byte SIZEOF_XMMWORD  ; inptr1
    add         rdx, byte SIZEOF_XMMWORD  ; inptr2
    jmp         near .columnloop

.column_st32:
    cmp         rcx, byte SIZEOF_XMMWORD / 2
    jb          short .column_st16
    movdqu      XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi + 1 * SIZEOF_XMMWORD], xmmD
    add         rdi, byte 2 * SIZEOF_XMMWORD  ; outptr
    movdqa      xmmA, xmmC
    movdqa      xmmD, xmmH
    sub         rcx, byte SIZEOF_XMMWORD / 2
.column_st16:
    cmp         rcx, byte SIZEOF_XMMWORD / 4
    jb          short .column_st15
    movdqu      XMMWORD [rdi + 0 * SIZEOF_XMMWORD], xmmA
    add         rdi, byte SIZEOF_XMMWORD      ; outptr
    movdqa      xmmA, xmmD
    sub         rcx, byte SIZEOF_XMMWORD / 4
.column_st15:
    ; Store two pixels (8 bytes) of xmmA to the output when it has enough
    ; space.
    cmp         rcx, byte SIZEOF_XMMWORD / 8
    jb          short .column_st7
    movq        MMWORD [rdi], xmmA
    add         rdi, byte SIZEOF_XMMWORD / 8 * 4
    sub         rcx, byte SIZEOF_XMMWORD / 8
    psrldq      xmmA, SIZEOF_XMMWORD / 8 * 4
.column_st7:
    ; Store one pixel (4 bytes) of xmmA to the output when it has enough
    ; space.
    test        rcx, rcx
    jz          short .nextrow
    movd        XMM_DWORD [rdi], xmmA

%endif  ; RGB_PIXELSIZE ; ---------------

.nextrow:
    pop         rcx
    pop         rsi
    pop         rbx
    pop         rdx
    pop         rdi
    pop         rax

    add         rsi, byte SIZEOF_JSAMPROW
    add         rbx, byte SIZEOF_JSAMPROW
    add         rdx, byte SIZEOF_JSAMPROW
    add         rdi, byte SIZEOF_JSAMPROW  ; output_buf
    dec         rax                        ; num_rows
    jg          near .rowloop

    sfence                              ; flush the write buffer

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
