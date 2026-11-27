;
; Merged Upsampling/Color Conversion (32-bit SSE2)
;
; Copyright 2009, 2012 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2012, 2016, 2024-2025, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jcolsamp.inc"

; --------------------------------------------------------------------------
;
; Upsample and color convert for the case of 2:1 horizontal and 1:1 vertical.
;
; GLOBAL(void)
; jsimd_h2v1_merged_upsample_sse2(JDIMENSION output_width,
;                                 JSAMPIMAGE input_buf,
;                                 JDIMENSION in_row_group_ctr,
;                                 JSAMPARRAY output_buf)

%define output_width(b)      (b) + 8    ; JDIMENSION output_width
%define input_buf(b)         (b) + 12   ; JSAMPIMAGE input_buf
%define in_row_group_ctr(b)  (b) + 16   ; JDIMENSION in_row_group_ctr
%define output_buf(b)        (b) + 20   ; JSAMPARRAY output_buf

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_XMMWORD
                      ; xmmword wk[WK_NUM]
%define WK_NUM        3
%define gotptr        wk(0) - SIZEOF_POINTER  ; void *gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_merged_upsample_sse2)

EXTN(jsimd_h2v1_merged_upsample_sse2):
    push        ebp
    mov         eax, esp                ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
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

    mov         ecx, JDIMENSION [output_width(eax)]  ; col
    test        ecx, ecx
    jz          near .return

    push        ecx

    mov         edi, JSAMPIMAGE [input_buf(eax)]
    mov         ecx, JDIMENSION [in_row_group_ctr(eax)]
    mov         esi, JSAMPARRAY [edi + 0 * SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [edi + 1 * SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [edi + 2 * SIZEOF_JSAMPARRAY]
    mov         edi, JSAMPARRAY [output_buf(eax)]
    mov         esi, JSAMPROW [esi + ecx * SIZEOF_JSAMPROW]  ; inptr0
    mov         ebx, JSAMPROW [ebx + ecx * SIZEOF_JSAMPROW]  ; inptr1
    mov         edx, JSAMPROW [edx + ecx * SIZEOF_JSAMPROW]  ; inptr2
    mov         edi, JSAMPROW [edi]                          ; outptr

    pop         ecx                     ; col

    ALIGNX      16, 7
.columnloop:
    MOVPIC      eax, POINTER [gotptr]   ; load GOT address (eax)

    movdqa      xmm6, XMMWORD [ebx]     ; xmm6 = Cb(0123456789abcdef)
    movdqa      xmm7, XMMWORD [edx]     ; xmm7 = Cr(0123456789abcdef)

    pxor        xmm1, xmm1              ; xmm1 = (all 0's)
    pcmpeqw     xmm3, xmm3
    psllw       xmm3, 7             ; xmm3 = { 0xFF80 0xFF80 0xFF80 0xFF80 .. }

    movdqa      xmm4, xmm6
    punpckhbw   xmm6, xmm1              ; xmm6 = Cb(89abcdef) = CbH
    punpcklbw   xmm4, xmm1              ; xmm4 = Cb(01234567) = CbL
    movdqa      xmm0, xmm7
    punpckhbw   xmm7, xmm1              ; xmm7 = Cr(89abcdef) = CrH
    punpcklbw   xmm0, xmm1              ; xmm0 = Cr(01234567) = CrL

    paddw       xmm6, xmm3
    paddw       xmm4, xmm3
    paddw       xmm7, xmm3
    paddw       xmm0, xmm3

    ; (Original)
    ; R = Y                + 1.40200 * Cr
    ; G = Y - 0.34414 * Cb - 0.71414 * Cr
    ; B = Y + 1.77200 * Cb
    ;
    ; (This implementation)
    ; R = Y                + 0.40200 * Cr + Cr
    ; G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
    ; B = Y - 0.22800 * Cb + Cb + Cb

    movdqa      xmm5, xmm6              ; xmm5 = CbH
    movdqa      xmm2, xmm4              ; xmm2 = CbL
    paddw       xmm6, xmm6              ; xmm6 = 2 * CbH
    paddw       xmm4, xmm4              ; xmm4 = 2 * CbL
    movdqa      xmm1, xmm7              ; xmm1 = CrH
    movdqa      xmm3, xmm0              ; xmm3 = CrL
    paddw       xmm7, xmm7              ; xmm7 = 2 * CrH
    paddw       xmm0, xmm0              ; xmm0 = 2 * CrL

    pmulhw      xmm6, [GOTOFF(eax, PW_MF0228)]
                ; xmm6 = (2 * CbH * -FIX(0.22800))
    pmulhw      xmm4, [GOTOFF(eax, PW_MF0228)]
                ; xmm4 = (2 * CbL * -FIX(0.22800))
    pmulhw      xmm7, [GOTOFF(eax, PW_F0402)]
                ; xmm7 = (2 * CrH * FIX(0.40200))
    pmulhw      xmm0, [GOTOFF(eax, PW_F0402)]
                ; xmm0 = (2 * CrL * FIX(0.40200))

    paddw       xmm6, [GOTOFF(eax, PW_ONE)]
    paddw       xmm4, [GOTOFF(eax, PW_ONE)]
    psraw       xmm6, 1                 ; xmm6 = (CbH * -FIX(0.22800))
    psraw       xmm4, 1                 ; xmm4 = (CbL * -FIX(0.22800))
    paddw       xmm7, [GOTOFF(eax, PW_ONE)]
    paddw       xmm0, [GOTOFF(eax, PW_ONE)]
    psraw       xmm7, 1                 ; xmm7 = (CrH * FIX(0.40200))
    psraw       xmm0, 1                 ; xmm0 = (CrL * FIX(0.40200))

    paddw       xmm6, xmm5
    paddw       xmm4, xmm2
    paddw       xmm6, xmm5             ; xmm6 = (CbH * FIX(1.77200)) = (B - Y)H
    paddw       xmm4, xmm2             ; xmm4 = (CbL * FIX(1.77200)) = (B - Y)L
    paddw       xmm7, xmm1             ; xmm7 = (CrH * FIX(1.40200)) = (R - Y)H
    paddw       xmm0, xmm3             ; xmm0 = (CrL * FIX(1.40200)) = (R - Y)L

    movdqa      XMMWORD [wk(0)], xmm6   ; wk(0) = (B - Y)H
    movdqa      XMMWORD [wk(1)], xmm7   ; wk(1) = (R - Y)H

    movdqa      xmm6, xmm5
    movdqa      xmm7, xmm2
    punpcklwd   xmm5, xmm1
    punpckhwd   xmm6, xmm1
    pmaddwd     xmm5, [GOTOFF(eax, PW_MF0344_F0285)]
    pmaddwd     xmm6, [GOTOFF(eax, PW_MF0344_F0285)]
    punpcklwd   xmm2, xmm3
    punpckhwd   xmm7, xmm3
    pmaddwd     xmm2, [GOTOFF(eax, PW_MF0344_F0285)]
    pmaddwd     xmm7, [GOTOFF(eax, PW_MF0344_F0285)]

    paddd       xmm5, [GOTOFF(eax, PD_ONEHALF)]
    paddd       xmm6, [GOTOFF(eax, PD_ONEHALF)]
    psrad       xmm5, SCALEBITS
    psrad       xmm6, SCALEBITS
    paddd       xmm2, [GOTOFF(eax, PD_ONEHALF)]
    paddd       xmm7, [GOTOFF(eax, PD_ONEHALF)]
    psrad       xmm2, SCALEBITS
    psrad       xmm7, SCALEBITS

    packssdw    xmm5, xmm6
                ; xmm5 = CbH * -FIX(0.344) + CrH * FIX(0.285)
    packssdw    xmm2, xmm7
                ; xmm2 = CbL * -FIX(0.344) + CrL * FIX(0.285)
    psubw       xmm5, xmm1
                ; xmm5 = CbH * -FIX(0.344) + CrH * -FIX(0.714) = (G - Y)H
    psubw       xmm2, xmm3
                ; xmm2 = CbL * -FIX(0.344) + CrL * -FIX(0.714) = (G - Y)L

    movdqa      XMMWORD [wk(2)], xmm5   ; wk(2) = (G - Y)H

    mov         al, 2                   ; Yctr
    jmp         short .Yloop_1st
    ALIGNX      16, 7

.Yloop_2nd:
    movdqa      xmm0, XMMWORD [wk(1)]   ; xmm0 = (R - Y)H
    movdqa      xmm2, XMMWORD [wk(2)]   ; xmm2 = (G - Y)H
    movdqa      xmm4, XMMWORD [wk(0)]   ; xmm4 = (B - Y)H
    ALIGNX      16, 7

.Yloop_1st:
    movdqa      xmm7, XMMWORD [esi]     ; xmm7 = Y(0123456789abcdef)

    pcmpeqw     xmm6, xmm6
    psrlw       xmm6, BYTE_BIT          ; xmm6 = { 0xFF 0x00 0xFF 0x00 .. }
    pand        xmm6, xmm7              ; xmm6 = Y(02468ace) = YE
    psrlw       xmm7, BYTE_BIT          ; xmm7 = Y(13579bdf) = YO

    movdqa      xmm1, xmm0              ; xmm1 = xmm0 = (R - Y)(L / H)
    movdqa      xmm3, xmm2              ; xmm3 = xmm2 = (G - Y)(L / H)
    movdqa      xmm5, xmm4              ; xmm5 = xmm4 = (B - Y)(L / H)

    paddw       xmm0, xmm6           ; xmm0 = ((R - Y) + YE) = RE = R(02468ace)
    paddw       xmm1, xmm7           ; xmm1 = ((R - Y) + YO) = RO = R(13579bdf)
    packuswb    xmm0, xmm0           ; xmm0 = R(02468ace********)
    packuswb    xmm1, xmm1           ; xmm1 = R(13579bdf********)

    paddw       xmm2, xmm6           ; xmm2 = ((G - Y) + YE) = GE = G(02468ace)
    paddw       xmm3, xmm7           ; xmm3 = ((G - Y) + YO) = GO = G(13579bdf)
    packuswb    xmm2, xmm2           ; xmm2 = G(02468ace********)
    packuswb    xmm3, xmm3           ; xmm3 = G(13579bdf********)

    paddw       xmm4, xmm6           ; xmm4 = ((B - Y) + YE) = BE = B(02468ace)
    paddw       xmm5, xmm7           ; xmm5 = ((B - Y) + YO) = BO = B(13579bdf)
    packuswb    xmm4, xmm4           ; xmm4 = B(02468ace********)
    packuswb    xmm5, xmm5           ; xmm5 = B(13579bdf********)

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

    cmp         ecx, byte SIZEOF_XMMWORD
    jb          short .column_st32

    test        edi, SIZEOF_XMMWORD - 1
    jnz         short .out1
    ; --(aligned)-------------------
    movntdq     XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    movntdq     XMMWORD [edi + 1 * SIZEOF_XMMWORD], xmmD
    movntdq     XMMWORD [edi + 2 * SIZEOF_XMMWORD], xmmF
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    movdqu      XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [edi + 1 * SIZEOF_XMMWORD], xmmD
    movdqu      XMMWORD [edi + 2 * SIZEOF_XMMWORD], xmmF
.out0:
    add         edi, byte RGB_PIXELSIZE * SIZEOF_XMMWORD  ; outptr
    sub         ecx, byte SIZEOF_XMMWORD
    jz          near .endcolumn

    add         esi, byte SIZEOF_XMMWORD  ; inptr0
    dec         al                        ; Yctr
    jnz         near .Yloop_2nd

    add         ebx, byte SIZEOF_XMMWORD  ; inptr1
    add         edx, byte SIZEOF_XMMWORD  ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st32:
    lea         ecx, [ecx + ecx * 2]          ; imul ecx, RGB_PIXELSIZE
    cmp         ecx, byte 2 * SIZEOF_XMMWORD
    jb          short .column_st16
    movdqu      XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [edi + 1 * SIZEOF_XMMWORD], xmmD
    add         edi, byte 2 * SIZEOF_XMMWORD  ; outptr
    movdqa      xmmA, xmmF
    sub         ecx, byte 2 * SIZEOF_XMMWORD
    jmp         short .column_st15
.column_st16:
    cmp         ecx, byte SIZEOF_XMMWORD
    jb          short .column_st15
    movdqu      XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    add         edi, byte SIZEOF_XMMWORD      ; outptr
    movdqa      xmmA, xmmD
    sub         ecx, byte SIZEOF_XMMWORD
.column_st15:
    ; Store the lower 8 bytes of xmmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st7
    movq        XMM_MMWORD [edi], xmmA
    add         edi, byte SIZEOF_MMWORD
    sub         ecx, byte SIZEOF_MMWORD
    psrldq      xmmA, SIZEOF_MMWORD
.column_st7:
    ; Store the lower 4 bytes of xmmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_DWORD
    jb          short .column_st3
    movd        XMM_DWORD [edi], xmmA
    add         edi, byte SIZEOF_DWORD
    sub         ecx, byte SIZEOF_DWORD
    psrldq      xmmA, SIZEOF_DWORD
.column_st3:
    ; Store the lower 2 bytes of eax to the output when it has enough
    ; space.
    movd        eax, xmmA
    cmp         ecx, byte SIZEOF_WORD
    jb          short .column_st1
    mov         word [edi], ax
    add         edi, byte SIZEOF_WORD
    sub         ecx, byte SIZEOF_WORD
    shr         eax, 16
.column_st1:
    ; Store the lower 1 byte of eax to the output when it has enough
    ; space.
    test        ecx, ecx
    jz          short .endcolumn
    mov         byte [edi], al

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

    cmp         ecx, byte SIZEOF_XMMWORD
    jb          short .column_st32

    test        edi, SIZEOF_XMMWORD - 1
    jnz         short .out1
    ; --(aligned)-------------------
    movntdq     XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    movntdq     XMMWORD [edi + 1 * SIZEOF_XMMWORD], xmmD
    movntdq     XMMWORD [edi + 2 * SIZEOF_XMMWORD], xmmC
    movntdq     XMMWORD [edi + 3 * SIZEOF_XMMWORD], xmmH
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    movdqu      XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [edi + 1 * SIZEOF_XMMWORD], xmmD
    movdqu      XMMWORD [edi + 2 * SIZEOF_XMMWORD], xmmC
    movdqu      XMMWORD [edi + 3 * SIZEOF_XMMWORD], xmmH
.out0:
    add         edi, byte RGB_PIXELSIZE * SIZEOF_XMMWORD  ; outptr
    sub         ecx, byte SIZEOF_XMMWORD
    jz          near .endcolumn

    add         esi, byte SIZEOF_XMMWORD  ; inptr0
    dec         al                        ; Yctr
    jnz         near .Yloop_2nd

    add         ebx, byte SIZEOF_XMMWORD  ; inptr1
    add         edx, byte SIZEOF_XMMWORD  ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st32:
    cmp         ecx, byte SIZEOF_XMMWORD / 2
    jb          short .column_st16
    movdqu      XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [edi + 1 * SIZEOF_XMMWORD], xmmD
    add         edi, byte 2 * SIZEOF_XMMWORD  ; outptr
    movdqa      xmmA, xmmC
    movdqa      xmmD, xmmH
    sub         ecx, byte SIZEOF_XMMWORD / 2
.column_st16:
    cmp         ecx, byte SIZEOF_XMMWORD / 4
    jb          short .column_st15
    movdqu      XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    add         edi, byte SIZEOF_XMMWORD      ; outptr
    movdqa      xmmA, xmmD
    sub         ecx, byte SIZEOF_XMMWORD / 4
.column_st15:
    ; Store two pixels (8 bytes) of xmmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_XMMWORD / 8
    jb          short .column_st7
    movq        XMM_MMWORD [edi], xmmA
    add         edi, byte SIZEOF_XMMWORD / 8 * 4
    sub         ecx, byte SIZEOF_XMMWORD / 8
    psrldq      xmmA, SIZEOF_XMMWORD / 8 * 4
.column_st7:
    ; Store one pixel (4 bytes) of xmmA to the output when it has enough
    ; space.
    test        ecx, ecx
    jz          short .endcolumn
    movd        XMM_DWORD [edi], xmmA

%endif  ; RGB_PIXELSIZE ; ---------------

.endcolumn:
    sfence                              ; flush the write buffer

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

; --------------------------------------------------------------------------
;
; Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
;
; GLOBAL(void)
; jsimd_h2v2_merged_upsample_sse2(JDIMENSION output_width,
;                                 JSAMPIMAGE input_buf,
;                                 JDIMENSION in_row_group_ctr,
;                                 JSAMPARRAY output_buf)

%define output_width(b)      (b) + 8    ; JDIMENSION output_width
%define input_buf(b)         (b) + 12   ; JSAMPIMAGE input_buf
%define in_row_group_ctr(b)  (b) + 16   ; JDIMENSION in_row_group_ctr
%define output_buf(b)        (b) + 20   ; JSAMPARRAY output_buf

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_merged_upsample_sse2)

EXTN(jsimd_h2v2_merged_upsample_sse2):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         eax, POINTER [output_width(ebp)]

    mov         edi, JSAMPIMAGE [input_buf(ebp)]
    mov         ecx, JDIMENSION [in_row_group_ctr(ebp)]
    mov         esi, JSAMPARRAY [edi + 0 * SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [edi + 1 * SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [edi + 2 * SIZEOF_JSAMPARRAY]
    mov         edi, JSAMPARRAY [output_buf(ebp)]
    lea         esi, [esi + ecx * SIZEOF_JSAMPROW]

    push        edx                     ; inptr2
    push        ebx                     ; inptr1
    push        esi                     ; inptr00
    mov         ebx, esp

    push        edi                     ; output_buf (outptr0)
    push        ecx                     ; in_row_group_ctr
    push        ebx                     ; input_buf
    push        eax                     ; output_width

    call        near EXTN(jsimd_h2v1_merged_upsample_sse2)

    add         esi, byte SIZEOF_JSAMPROW  ; inptr01
    add         edi, byte SIZEOF_JSAMPROW  ; outptr1
    mov         POINTER [ebx + 0 * SIZEOF_POINTER], esi
    mov         POINTER [ebx - 1 * SIZEOF_POINTER], edi

    call        near EXTN(jsimd_h2v1_merged_upsample_sse2)

    add         esp, byte 7 * SIZEOF_DWORD

    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    pop         ebx
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
