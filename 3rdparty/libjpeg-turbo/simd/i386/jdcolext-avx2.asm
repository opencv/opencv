;
; YCbCr-to-RGB Color Conversion (32-bit AVX2)
;
; Copyright 2009, 2012 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2012, 2016, 2024-2025, D. R. Commander.
; Copyright (C) 2015, Intel Corporation.
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
; jsimd_ycc_rgb_convert_avx2(JDIMENSION out_width, JSAMPIMAGE input_buf,
;                            JDIMENSION input_row, JSAMPARRAY output_buf,
;                            int num_rows)

%define out_width(b)   (b) + 8          ; JDIMENSION out_width
%define input_buf(b)   (b) + 12         ; JSAMPIMAGE input_buf
%define input_row(b)   (b) + 16         ; JDIMENSION input_row
%define output_buf(b)  (b) + 20         ; JSAMPARRAY output_buf
%define num_rows(b)    (b) + 24         ; int num_rows

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_YMMWORD
                      ; ymmword wk[WK_NUM]
%define WK_NUM        2
%define gotptr        wk(0) - SIZEOF_POINTER  ; void *gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_ycc_rgb_convert_avx2)

EXTN(jsimd_ycc_rgb_convert_avx2):
    push        ebp
    mov         eax, esp                ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_YMMWORD)  ; align to 256 bits
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

    vmovdqu     ymm5, YMMWORD [ebx]
                ; ymm5 = Cb(0123456789abcdefghijklmnopqrstuv)
    vmovdqu     ymm1, YMMWORD [edx]
                ; ymm1 = Cr(0123456789abcdefghijklmnopqrstuv)

    vpcmpeqw    ymm0, ymm0, ymm0
    vpcmpeqw    ymm7, ymm7, ymm7
    vpsrlw      ymm0, ymm0, BYTE_BIT  ; ymm0 = { 0xFF 0x00 0xFF 0x00 .. }
    vpsllw      ymm7, ymm7, 7       ; ymm7 = { 0xFF80 0xFF80 0xFF80 0xFF80 .. }

    vpand       ymm4, ymm0, ymm5        ; ymm4 = Cb(02468acegikmoqsu) = CbE
    vpsrlw      ymm5, ymm5, BYTE_BIT    ; ymm5 = Cb(13579bdfhjlnprtv) = CbO
    vpand       ymm0, ymm0, ymm1        ; ymm0 = Cr(02468acegikmoqsu) = CrE
    vpsrlw      ymm1, ymm1, BYTE_BIT    ; ymm1 = Cr(13579bdfhjlnprtv) = CrO

    vpaddw      ymm2, ymm4, ymm7
    vpaddw      ymm3, ymm5, ymm7
    vpaddw      ymm6, ymm0, ymm7
    vpaddw      ymm7, ymm1, ymm7

    ; (Original)
    ; R = Y                + 1.40200 * Cr
    ; G = Y - 0.34414 * Cb - 0.71414 * Cr
    ; B = Y + 1.77200 * Cb
    ;
    ; (This implementation)
    ; R = Y                + 0.40200 * Cr + Cr
    ; G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
    ; B = Y - 0.22800 * Cb + Cb + Cb

    vpaddw      ymm4, ymm2, ymm2        ; ymm4 = 2 * CbE
    vpaddw      ymm5, ymm3, ymm3        ; ymm5 = 2 * CbO
    vpaddw      ymm0, ymm6, ymm6        ; ymm0 = 2 * CrE
    vpaddw      ymm1, ymm7, ymm7        ; ymm1 = 2 * CrO

    vpmulhw     ymm4, ymm4, [GOTOFF(eax, PW_MF0228)]
                ; ymm4 = (2 * CbE * -FIX(0.22800))
    vpmulhw     ymm5, ymm5, [GOTOFF(eax, PW_MF0228)]
                ; ymm5 = (2 * CbO * -FIX(0.22800))
    vpmulhw     ymm0, ymm0, [GOTOFF(eax, PW_F0402)]
                ; ymm0 = (2 * CrE * FIX(0.40200))
    vpmulhw     ymm1, ymm1, [GOTOFF(eax, PW_F0402)]
                ; ymm1 = (2 * CrO * FIX(0.40200))

    vpaddw      ymm4, ymm4, [GOTOFF(eax, PW_ONE)]
    vpaddw      ymm5, ymm5, [GOTOFF(eax, PW_ONE)]
    vpsraw      ymm4, ymm4, 1           ; ymm4 = (CbE * -FIX(0.22800))
    vpsraw      ymm5, ymm5, 1           ; ymm5 = (CbO * -FIX(0.22800))
    vpaddw      ymm0, ymm0, [GOTOFF(eax, PW_ONE)]
    vpaddw      ymm1, ymm1, [GOTOFF(eax, PW_ONE)]
    vpsraw      ymm0, ymm0, 1           ; ymm0 = (CrE * FIX(0.40200))
    vpsraw      ymm1, ymm1, 1           ; ymm1 = (CrO * FIX(0.40200))

    vpaddw      ymm4, ymm4, ymm2
    vpaddw      ymm5, ymm5, ymm3
    vpaddw      ymm4, ymm4, ymm2       ; ymm4 = (CbE * FIX(1.77200)) = (B - Y)E
    vpaddw      ymm5, ymm5, ymm3       ; ymm5 = (CbO * FIX(1.77200)) = (B - Y)O
    vpaddw      ymm0, ymm0, ymm6       ; ymm0 = (CrE * FIX(1.40200)) = (R - Y)E
    vpaddw      ymm1, ymm1, ymm7       ; ymm1 = (CrO * FIX(1.40200)) = (R - Y)O

    vmovdqa     YMMWORD [wk(0)], ymm4   ; wk(0) = (B - Y)E
    vmovdqa     YMMWORD [wk(1)], ymm5   ; wk(1) = (B - Y)O

    vpunpckhwd  ymm4, ymm2, ymm6
    vpunpcklwd  ymm2, ymm2, ymm6
    vpmaddwd    ymm2, ymm2, [GOTOFF(eax, PW_MF0344_F0285)]
    vpmaddwd    ymm4, ymm4, [GOTOFF(eax, PW_MF0344_F0285)]
    vpunpckhwd  ymm5, ymm3, ymm7
    vpunpcklwd  ymm3, ymm3, ymm7
    vpmaddwd    ymm3, ymm3, [GOTOFF(eax, PW_MF0344_F0285)]
    vpmaddwd    ymm5, ymm5, [GOTOFF(eax, PW_MF0344_F0285)]

    vpaddd      ymm2, ymm2, [GOTOFF(eax, PD_ONEHALF)]
    vpaddd      ymm4, ymm4, [GOTOFF(eax, PD_ONEHALF)]
    vpsrad      ymm2, ymm2, SCALEBITS
    vpsrad      ymm4, ymm4, SCALEBITS
    vpaddd      ymm3, ymm3, [GOTOFF(eax, PD_ONEHALF)]
    vpaddd      ymm5, ymm5, [GOTOFF(eax, PD_ONEHALF)]
    vpsrad      ymm3, ymm3, SCALEBITS
    vpsrad      ymm5, ymm5, SCALEBITS

    vpackssdw   ymm2, ymm2, ymm4
                ; ymm2 = CbE * -FIX(0.344) + CrE * FIX(0.285)
    vpackssdw   ymm3, ymm3, ymm5
                ; ymm3 = CbO * -FIX(0.344) + CrO * FIX(0.285)
    vpsubw      ymm2, ymm2, ymm6
                ; ymm2 = CbE * -FIX(0.344) + CrE * -FIX(0.714) = (G - Y)E
    vpsubw      ymm3, ymm3, ymm7
                ; ymm3 = CbO * -FIX(0.344) + CrO * -FIX(0.714) = (G - Y)O

    vmovdqu     ymm5, YMMWORD [esi]
                ; ymm5 = Y(0123456789abcdefghijklmnopqrstuv)

    vpcmpeqw    ymm4, ymm4, ymm4
    vpsrlw      ymm4, ymm4, BYTE_BIT    ; ymm4 = { 0xFF 0x00 0xFF 0x00 .. }
    vpand       ymm4, ymm4, ymm5        ; ymm4 = Y(02468acegikmoqsu) = YE
    vpsrlw      ymm5, ymm5, BYTE_BIT    ; ymm5 = Y(13579bdfhjlnprtv) = YO

    vpaddw      ymm0, ymm0, ymm4
                ; ymm0 = ((R - Y)E + YE) = RE = R(02468acegikmoqsu)
    vpaddw      ymm1, ymm1, ymm5
                ; ymm1 = ((R - Y)O + YO) = RO = R(13579bdfhjlnprtv)
    vpackuswb   ymm0, ymm0, ymm0
                ; ymm0 = R(02468ace********gikmoqsu********)
    vpackuswb   ymm1, ymm1, ymm1
                ; ymm1 = R(13579bdf********hjlnprtv********)

    vpaddw      ymm2, ymm2, ymm4
                ; ymm2 = ((G - Y)E + YE) = GE = G(02468acegikmoqsu)
    vpaddw      ymm3, ymm3, ymm5
                ; ymm3 = ((G - Y)O + YO) = GO = G(13579bdfhjlnprtv)
    vpackuswb   ymm2, ymm2, ymm2
                ; ymm2 = G(02468ace********gikmoqsu********)
    vpackuswb   ymm3, ymm3, ymm3
                ; ymm3 = G(13579bdf********hjlnprtv********)

    vpaddw      ymm4, ymm4, YMMWORD [wk(0)]
                ; ymm4 = (YE + (B - Y)E) = BE = B(02468acegikmoqsu)
    vpaddw      ymm5, ymm5, YMMWORD [wk(1)]
                ; ymm5 = (YO + (B - Y)O) = BO = B(13579bdfhjlnprtv)
    vpackuswb   ymm4, ymm4, ymm4
                ; ymm4 = B(02468ace********gikmoqsu********)
    vpackuswb   ymm5, ymm5, ymm5
                ; ymm5 = B(13579bdf********hjlnprtv********)

%if RGB_PIXELSIZE == 3  ; ---------------

    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, and C to red, green, and blue.
    ;
    ; ymmA = (A0 A2 A4 A6 A8 Aa Ac Ae Ag Ai Ak Am Ao Aq As Au) = AE
    ; ymmB = (A1 A3 A5 A7 A9 Ab Ad Af Ah Aj Al An Ap Ar At Av) = AO
    ; ymmC = (B0 B2 B4 B6 B8 Ba Bc Be Bg Bi Bk Bm Bo Bq Bs Bu) = BE
    ; ymmD = (B1 B3 B5 B7 B9 Bb Bd Bf Bh Bj Bl Bn Bp Br Bt Bv) = BO
    ; ymmE = (C0 C2 C4 C6 C8 Ca Cc Ce Cg Ci Ck Cm Co Cq Cs Cu) = CE
    ; ymmF = (C1 C3 C5 C7 C9 Cb Cd Cf Ch Cj Cl Cn Cp Cr Ct Cv) = CO
    ; ymmG = (** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **)
    ; ymmH = (** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **)

    vpunpcklbw  ymmA, ymmA, ymmC
                ; ymmA = (A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 Aa Ba Ac Bc Ae Be
                ;         Ag Bg Ai Bi Ak Bk Am Bm Ao Bo Aq Bq As Bs Au Bu)
    vpunpcklbw  ymmE, ymmE, ymmB
                ; ymmE = (C0 A1 C2 A3 C4 A5 C6 A7 C8 A9 Ca Ab Cc Ad Ce Af
                ;         Cg Ah Ci Aj Ck Al Cm An Co Ap Cq Ar Cs At Cu Av)
    vpunpcklbw  ymmD, ymmD, ymmF
                ; ymmD = (B1 C1 B3 C3 B5 C5 B7 C7 B9 C9 Bb Cb Bd Cd Bf Cf
                ;         Bh Ch Bj Cj Bl Cl Bn Cn Bp Cp Br Cr Bt Ct Bv Cv)

    vpsrldq     ymmH, ymmA, 2
                ; ymmH = (A2 B2 A4 B4 A6 B6 A8 B8 Aa Ba Ac Bc Ae Be Ag Bg
                ;         Ai Bi Ak Bk Am Bm Ao Bo Aq Bq As Bs Au Bu -- --)
    vpunpckhwd  ymmG, ymmA, ymmE
                ; ymmG = (A8 B8 C8 A9 Aa Ba Ca Ab Ac Bc Cc Ad Ae Be Ce Af
                ;         Ao Bo Co Ap Aq Bq Cq Ar As Bs Cs At Au Bu Cu Av)
    vpunpcklwd  ymmA, ymmA, ymmE
                ; ymmA = (A0 B0 C0 A1 A2 B2 C2 A3 A4 B4 C4 A5 A6 B6 C6 A7
                ;         Ag Bg Cg Ah Ai Bi Ci Aj Ak Bk Ck Al Am Bm Cm An)

    vpsrldq     ymmE, ymmE, 2
                ; ymmE = (C2 A3 C4 A5 C6 A7 C8 A9 Ca Ab Cc Ad Ce Af Cg Ah
                ;         Ci Aj Ck Al Cm An Co Ap Cq Ar Cs At Cu Av -- --)

    vpsrldq     ymmB, ymmD, 2
                ; ymmB = (B3 C3 B5 C5 B7 C7 B9 C9 Bb Cb Bd Cd Bf Cf Bh Ch
                ;         Bj Cj Bl Cl Bn Cn Bp Cp Br Cr Bt Ct Bv Cv -- --)
    vpunpckhwd  ymmC, ymmD, ymmH
                ; ymmC = (B9 C9 Aa Ba Bb Cb Ac Bc Bd Cd Ae Be Bf Cf Ag Bg
                ;         Bp Cp Aq Bq Br Cr As Bs Bt Ct Au Bu Bv Cv -- --)
    vpunpcklwd  ymmD, ymmD, ymmH
                ; ymmD = (B1 C1 A2 B2 B3 C3 A4 B4 B5 C5 A6 B6 B7 C7 A8 B8
                ;         Bh Ch Ai Bi Bj Cj Ak Bk Bl Cl Am Bm Bn Cn Ao Bo)

    vpunpckhwd  ymmF, ymmE, ymmB
                ; ymmF = (Ca Ab Bb Cb Cc Ad Bd Cd Ce Af Bf Cf Cg Ah Bh Ch
                ;         Cq Ar Br Cr Cs At Bt Ct Cu Av Bv Cv -- -- -- --)
    vpunpcklwd  ymmE, ymmE, ymmB
                ; ymmE = (C2 A3 B3 C3 C4 A5 B5 C5 C6 A7 B7 C7 C8 A9 B9 C9
                ;         Ci Aj Bj Cj Ck Al Bl Cl Cm An Bn Cn Co Ap Bp Cp)

    vpshufd     ymmH, ymmA, 0x4E
                ; ymmH = (A4 B4 C4 A5 A6 B6 C6 A7 A0 B0 C0 A1 A2 B2 C2 A3
                ;         Ak Bk Ck Al Am Bm Cm An Ag Bg Cg Ah Ai Bi Ci Aj)
    vpunpckldq  ymmA, ymmA, ymmD
                ; ymmA = (A0 B0 C0 A1 B1 C1 A2 B2 A2 B2 C2 A3 B3 C3 A4 B4
                ;         Ag Bg Cg Ah Bh Ch Ai Bi Ai Bi Ci Aj Bj Cj Ak Bk)
    vpunpckhdq  ymmD, ymmD, ymmE
                ; ymmD = (B5 C5 A6 B6 C6 A7 B7 C7 B7 C7 A8 B8 C8 A9 B9 C9
                ;         Bl Cl Am Bm Cm An Bn Cn Bn Cn Ao Bo Co Ap Bp Cp)
    vpunpckldq  ymmE, ymmE, ymmH
                ; ymmE = (C2 A3 B3 C3 A4 B4 C4 A5 C4 A5 B5 C5 A6 B6 C6 A7
                ;         Ci Aj Bj Cj Ak Bk Ck Al Ck Al Bl Cl Am Bm Cm An)

    vpshufd     ymmH, ymmG, 0x4E
                ; ymmH = (Ac Bc Cc Ad Ae Be Ce Af A8 B8 C8 A9 Aa Ba Ca Ab
                ;         As Bs Cs At Au Bu Cu Av Ao Bo Co Ap Aq Bq Cq Ar)
    vpunpckldq  ymmG, ymmG, ymmC
                ; ymmG = (A8 B8 C8 A9 B9 C9 Aa Ba Aa Ba Ca Ab Bb Cb Ac Bc
                ;         Ao Bo Co Ap Bp Cp Aq Bq Aq Bq Cq Ar Br Cr As Bs)
    vpunpckhdq  ymmC, ymmC, ymmF
                ; ymmC = (Bd Cd Ae Be Ce Af Bf Cf Bf Cf Ag Bg Cg Ah Bh Ch
                ;         Bt Ct Au Bu Cu Av Bv Cv Bv Cv -- -- -- -- -- --)
    vpunpckldq  ymmF, ymmF, ymmH
                ; ymmF = (Ca Ab Bb Cb Ac Bc Cc Ad Cc Ad Bd Cd Ae Be Ce Af
                ;         Cq Ar Br Cr As Bs Cs At Cs At Bt Ct Au Bu Cu Av)

    vpunpcklqdq ymmH, ymmA, ymmE
                ; ymmH = (A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4 A5
                ;         Ag Bg Cg Ah Bh Ch Ai Bi Ci Aj Bj Cj Ak Bk Ck Al)
    vpunpcklqdq ymmG, ymmD, ymmG
                ; ymmG = (B5 C5 A6 B6 C6 A7 B7 C7 A8 B8 C8 A9 B9 C9 Aa Ba
                ;         Bl Cl Am Bm Cm An Bn Cn Ao Bo Co Ap Bp Cp Aq Bq)
    vpunpcklqdq ymmC, ymmF, ymmC
                ; ymmC = (Ca Ab Bb Cb Ac Bc Cc Ad Bd Cd Ae Be Ce Af Bf Cf
                ;         Cq Ar Br Cr As Bs Cs At Bt Ct Au Bu Cu Av Bv Cv)

    vperm2i128  ymmA, ymmH, ymmG, 0x20
                ; ymmA = (A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4 A5
                ;         B5 C5 A6 B6 C6 A7 B7 C7 A8 B8 C8 A9 B9 C9 Aa Ba)
    vperm2i128  ymmD, ymmC, ymmH, 0x30
                ; ymmD = (Ca Ab Bb Cb Ac Bc Cc Ad Bd Cd Ae Be Ce Af Bf Cf
                ;         Ag Bg Cg Ah Bh Ch Ai Bi Ci Aj Bj Cj Ak Bk Ck Al)
    vperm2i128  ymmF, ymmG, ymmC, 0x31
                ; ymmF = (Bl Cl Am Bm Cm An Bn Cn Ao Bo Co Ap Bp Cp Aq Bq
                ;         Cq Ar Br Cr As Bs Cs At Bt Ct Au Bu Cu Av Bv Cv)

    cmp         ecx, byte SIZEOF_YMMWORD
    jb          short .column_st64

    test        edi, SIZEOF_YMMWORD - 1
    jnz         short .out1
    ; --(aligned)-------------------
    vmovntdq    YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    vmovntdq    YMMWORD [edi + 1 * SIZEOF_YMMWORD], ymmD
    vmovntdq    YMMWORD [edi + 2 * SIZEOF_YMMWORD], ymmF
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    vmovdqu     YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi + 1 * SIZEOF_YMMWORD], ymmD
    vmovdqu     YMMWORD [edi + 2 * SIZEOF_YMMWORD], ymmF
.out0:
    add         edi, byte RGB_PIXELSIZE * SIZEOF_YMMWORD  ; outptr
    sub         ecx, byte SIZEOF_YMMWORD
    jz          near .nextrow

    add         esi, byte SIZEOF_YMMWORD  ; inptr0
    add         ebx, byte SIZEOF_YMMWORD  ; inptr1
    add         edx, byte SIZEOF_YMMWORD  ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st64:
    lea         ecx, [ecx + ecx * 2]          ; imul ecx, RGB_PIXELSIZE
    cmp         ecx, byte 2 * SIZEOF_YMMWORD
    jb          short .column_st32
    vmovdqu     YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi + 1 * SIZEOF_YMMWORD], ymmD
    add         edi, byte 2 * SIZEOF_YMMWORD  ; outptr
    vmovdqa     ymmA, ymmF
    sub         ecx, byte 2 * SIZEOF_YMMWORD
    jmp         short .column_st31
.column_st32:
    cmp         ecx, byte SIZEOF_YMMWORD
    jb          short .column_st31
    vmovdqu     YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    add         edi, byte SIZEOF_YMMWORD      ; outptr
    vmovdqa     ymmA, ymmD
    sub         ecx, byte SIZEOF_YMMWORD
    jmp         short .column_st31
.column_st31:
    cmp         ecx, byte SIZEOF_XMMWORD
    jb          short .column_st15
    vmovdqu     XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    add         edi, byte SIZEOF_XMMWORD      ; outptr
    vperm2i128  ymmA, ymmA, ymmA, 1
    sub         ecx, byte SIZEOF_XMMWORD
.column_st15:
    ; Store the lower 8 bytes of xmmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st7
    vmovq       XMM_MMWORD [edi], xmmA
    add         edi, byte SIZEOF_MMWORD
    sub         ecx, byte SIZEOF_MMWORD
    vpsrldq     xmmA, xmmA, SIZEOF_MMWORD
.column_st7:
    ; Store the lower 4 bytes of xmmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_DWORD
    jb          short .column_st3
    vmovd       XMM_DWORD [edi], xmmA
    add         edi, byte SIZEOF_DWORD
    sub         ecx, byte SIZEOF_DWORD
    vpsrldq     xmmA, xmmA, SIZEOF_DWORD
.column_st3:
    ; Store the lower 2 bytes of eax to the output when it has enough
    ; space.
    vmovd       eax, xmmA
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
    jz          short .nextrow
    mov         byte [edi], al

%else  ; RGB_PIXELSIZE == 4 ; -----------

%ifdef RGBX_FILLER_0XFF
    vpcmpeqb    ymm6, ymm6, ymm6
                ; ymm6 = XE = X(02468ace********gikmoqsu********)
    vpcmpeqb    ymm7, ymm7, ymm7
                ; ymm7 = XO = X(13579bdf********hjlnprtv********)
%else
    vpxor       ymm6, ymm6, ymm6
                ; ymm6 = XE = X(02468ace********gikmoqsu********)
    vpxor       ymm7, ymm7, ymm7
                ; ymm7 = XO = X(13579bdf********hjlnprtv********)
%endif
    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, C, and D to red, green, and blue.
    ;
    ; ymmA = (A0 A2 A4 A6 A8 Aa Ac Ae Ag Ai Ak Am Ao Aq As Au) = AE
    ; ymmB = (A1 A3 A5 A7 A9 Ab Ad Af Ah Aj Al An Ap Ar At Av) = AO
    ; ymmC = (B0 B2 B4 B6 B8 Ba Bc Be Bg Bi Bk Bm Bo Bq Bs Bu) = BE
    ; ymmD = (B1 B3 B5 B7 B9 Bb Bd Bf Bh Bj Bl Bn Bp Br Bt Bv) = BO
    ; ymmE = (C0 C2 C4 C6 C8 Ca Cc Ce Cg Ci Ck Cm Co Cq Cs Cu) = CE
    ; ymmF = (C1 C3 C5 C7 C9 Cb Cd Cf Ch Cj Cl Cn Cp Cr Ct Cv) = CO
    ; ymmG = (D0 D2 D4 D6 D8 Da Dc De Dg Di Dk Dm Do Dq Ds Du) = DE
    ; ymmH = (D1 D3 D5 D7 D9 Db Dd Df Dh Dj Dl Dn Dp Dr Dt Dv) = DO

    vpunpcklbw  ymmA, ymmA, ymmC
                ; ymmA = (A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 Aa Ba Ac Bc Ae Be
                ;         Ag Bg Ai Bi Ak Bk Am Bm Ao Bo Aq Bq As Bs Au Bu)
    vpunpcklbw  ymmE, ymmE, ymmG
                ; ymmE = (C0 D0 C2 D2 C4 D4 C6 D6 C8 D8 Ca Da Cc Dc Ce De
                ;         Cg Dg Ci Di Ck Dk Cm Dm Co Do Cq Dq Cs Ds Cu Du)
    vpunpcklbw  ymmB, ymmB, ymmD
                ; ymmB = (A1 B1 A3 B3 A5 B5 A7 B7 A9 B9 Ab Bb Ad Bd Af Bf
                ;         Ah Bh Aj Bj Al Bl An Bn Ap Bp Ar Br At Bt Av Bv)
    vpunpcklbw  ymmF, ymmF, ymmH
                ; ymmF = (C1 D1 C3 D3 C5 D5 C7 D7 C9 D9 Cb Db Cd Dd Cf Df
                ;         Ch Dh Cj Dj Cl Dl Cn Dn Cp Dp Cr Dr Ct Dt Cv Dv)

    vpunpckhwd  ymmC, ymmA, ymmE
                ; ymmC = (A8 B8 C8 D8 Aa Ba Ca Da Ac Bc Cc Dc Ae Be Ce De
                ;         Ao Bo Co Do Aq Bq Cq Dq As Bs Cs Ds Au Bu Cu Du)
    vpunpcklwd  ymmA, ymmA, ymmE
                ; ymmA = (A0 B0 C0 D0 A2 B2 C2 D2 A4 B4 C4 D4 A6 B6 C6 D6
                ;         Ag Bg Cg Dg Ai Bi Ci Di Ak Bk Ck Dk Am Bm Cm Dm)
    vpunpckhwd  ymmG, ymmB, ymmF
                ; ymmG = (A9 B9 C9 D9 Ab Bb Cb Db Ad Bd Cd Dd Af Bf Cf Df
                ;         Ap Bp Cp Dp Ar Br Cr Dr At Bt Ct Dt Av Bv Cv Dv)
    vpunpcklwd  ymmB, ymmB, ymmF
                ; ymmB = (A1 B1 C1 D1 A3 B3 C3 D3 A5 B5 C5 D5 A7 B7 C7 D7
                ;         Ah Bh Ch Dh Aj Bj Cj Dj Al Bl Cl Dl An Bn Cn Dn)

    vpunpckhdq  ymmE, ymmA, ymmB
                ; ymmE = (A4 B4 C4 D4 A5 B5 C5 D5 A6 B6 C6 D6 A7 B7 C7 D7
                ;         Ak Bk Ck Dk Al Bl Cl Dl Am Bm Cm Dm An Bn Cn Dn)
    vpunpckldq  ymmB, ymmA, ymmB
                ; ymmB = (A0 B0 C0 D0 A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3
                ;         Ag Bg Cg Dg Ah Bh Ch Dh Ai Bi Ci Di Aj Bj Cj Dj)
    vpunpckhdq  ymmF, ymmC, ymmG
                ; ymmF = (Ac Bc Cc Dc Ad Bd Cd Dd Ae Be Ce De Af Bf Cf Df
                ;         As Bs Cs Ds At Bt Ct Dt Au Bu Cu Du Av Bv Cv Dv)
    vpunpckldq  ymmG, ymmC, ymmG
                ; ymmG = (A8 B8 C8 D8 A9 B9 C9 D9 Aa Ba Ca Da Ab Bb Cb Db
                ;         Ao Bo Co Do Ap Bp Cp Dp Aq Bq Cq Dq Ar Br Cr Dr)

    vperm2i128  ymmA, ymmB, ymmE, 0x20
                ; ymmA = (A0 B0 C0 D0 A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3
                ;         A4 B4 C4 D4 A5 B5 C5 D5 A6 B6 C6 D6 A7 B7 C7 D7)
    vperm2i128  ymmD, ymmG, ymmF, 0x20
                ; ymmD = (A8 B8 C8 D8 A9 B9 C9 D9 Aa Ba Ca Da Ab Bb Cb Db
                ;         Ac Bc Cc Dc Ad Bd Cd Dd Ae Be Ce De Af Bf Cf Df)
    vperm2i128  ymmC, ymmB, ymmE, 0x31
                ; ymmC = (Ag Bg Cg Dg Ah Bh Ch Dh Ai Bi Ci Di Aj Bj Cj Dj
                ;         Ak Bk Ck Dk Al Bl Cl Dl Am Bm Cm Dm An Bn Cn Dn)
    vperm2i128  ymmH, ymmG, ymmF, 0x31
                ; ymmH = (Ao Bo Co Do Ap Bp Cp Dp Aq Bq Cq Dq Ar Br Cr Dr
                ;         As Bs Cs Ds At Bt Ct Dt Au Bu Cu Du Av Bv Cv Dv)

    cmp         ecx, byte SIZEOF_YMMWORD
    jb          short .column_st64

    test        edi, SIZEOF_YMMWORD - 1
    jnz         short .out1
    ; --(aligned)-------------------
    vmovntdq    YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    vmovntdq    YMMWORD [edi + 1 * SIZEOF_YMMWORD], ymmD
    vmovntdq    YMMWORD [edi + 2 * SIZEOF_YMMWORD], ymmC
    vmovntdq    YMMWORD [edi + 3 * SIZEOF_YMMWORD], ymmH
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    vmovdqu     YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi + 1 * SIZEOF_YMMWORD], ymmD
    vmovdqu     YMMWORD [edi + 2 * SIZEOF_YMMWORD], ymmC
    vmovdqu     YMMWORD [edi + 3 * SIZEOF_YMMWORD], ymmH
.out0:
    add         edi, RGB_PIXELSIZE * SIZEOF_YMMWORD  ; outptr
    sub         ecx, byte SIZEOF_YMMWORD
    jz          near .nextrow

    add         esi, byte SIZEOF_YMMWORD  ; inptr0
    add         ebx, byte SIZEOF_YMMWORD  ; inptr1
    add         edx, byte SIZEOF_YMMWORD  ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st64:
    cmp         ecx, byte SIZEOF_YMMWORD / 2
    jb          short .column_st32
    vmovdqu     YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi + 1 * SIZEOF_YMMWORD], ymmD
    add         edi, byte 2 * SIZEOF_YMMWORD  ; outptr
    vmovdqa     ymmA, ymmC
    vmovdqa     ymmD, ymmH
    sub         ecx, byte SIZEOF_YMMWORD / 2
.column_st32:
    cmp         ecx, byte SIZEOF_YMMWORD / 4
    jb          short .column_st16
    vmovdqu     YMMWORD [edi + 0 * SIZEOF_YMMWORD], ymmA
    add         edi, byte SIZEOF_YMMWORD      ; outptr
    vmovdqa     ymmA, ymmD
    sub         ecx, byte SIZEOF_YMMWORD / 4
.column_st16:
    cmp         ecx, byte SIZEOF_YMMWORD / 8
    jb          short .column_st15
    vmovdqu     XMMWORD [edi + 0 * SIZEOF_XMMWORD], xmmA
    vperm2i128  ymmA, ymmA, ymmA, 1
    add         edi, byte SIZEOF_XMMWORD      ; outptr
    sub         ecx, byte SIZEOF_YMMWORD / 8
.column_st15:
    ; Store two pixels (8 bytes) of ymmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_YMMWORD / 16
    jb          short .column_st7
    vmovq       MMWORD [edi], xmmA
    add         edi, byte SIZEOF_YMMWORD / 16 * 4
    sub         ecx, byte SIZEOF_YMMWORD / 16
    vpsrldq     xmmA, SIZEOF_YMMWORD / 16 * 4
.column_st7:
    ; Store one pixel (4 bytes) of ymmA to the output when it has enough
    ; space.
    test        ecx, ecx
    jz          short .nextrow
    vmovd       XMM_DWORD [edi], xmmA

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

    sfence                              ; flush the write buffer

.return:
    vzeroupper
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
