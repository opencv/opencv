;
; RGB-to-Grayscale Color Conversion (64-bit AVX2)
;
; Copyright (C) 2011, 2016, 2024-2025, D. R. Commander.
; Copyright (C) 2015, Intel Corporation.
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
; jsimd_rgb_gray_convert_avx2(JDIMENSION img_width, JSAMPARRAY input_buf,
;                             JSAMPIMAGE output_buf, JDIMENSION output_row,
;                             int num_rows)
;
; r10d = JDIMENSION img_width
; r11 = JSAMPARRAY input_buf
; r12 = JSAMPIMAGE output_buf
; r13d = JDIMENSION output_row
; r14d = int num_rows

%define wk(i)   r15 - (WK_NUM - (i)) * SIZEOF_YMMWORD  ; ymmword wk[WK_NUM]
%define WK_NUM  2

    align       32
    GLOBAL_FUNCTION(jsimd_rgb_gray_convert_avx2)

EXTN(jsimd_rgb_gray_convert_avx2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    push        r15
    and         rsp, byte (-SIZEOF_YMMWORD)  ; align to 256 bits
    ; Allocate stack space for wk array.  r15 is used to access it.
    mov         r15, rsp
    sub         rsp, byte (SIZEOF_YMMWORD * WK_NUM)
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

    cmp         rcx, byte SIZEOF_YMMWORD
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
    vmovd       xmmA, eax
    pop         rdx
    pop         rax
    test        cl, SIZEOF_DWORD
    jz          short .column_ld8
    sub         rcx, byte SIZEOF_DWORD
    vmovd       xmmF, XMM_DWORD [rsi + rcx]
    vpslldq     xmmA, xmmA, SIZEOF_DWORD
    vpor        xmmA, xmmA, xmmF
.column_ld8:
    test        cl, SIZEOF_MMWORD
    jz          short .column_ld16
    sub         rcx, byte SIZEOF_MMWORD
    vmovq       xmmB, XMM_MMWORD [rsi + rcx]
    vpslldq     xmmA, xmmA, SIZEOF_MMWORD
    vpor        xmmA, xmmA, xmmB
.column_ld16:
    test        cl, SIZEOF_XMMWORD
    jz          short .column_ld32
    sub         rcx, byte SIZEOF_XMMWORD
    vmovdqu     xmmB, XMM_MMWORD [rsi + rcx]
    vperm2i128  ymmA, ymmA, ymmA, 1
    vpor        ymmA, ymmB
.column_ld32:
    test        cl, SIZEOF_YMMWORD
    jz          short .column_ld64
    sub         rcx, byte SIZEOF_YMMWORD
    vmovdqa     ymmF, ymmA
    vmovdqu     ymmA, YMMWORD [rsi + 0 * SIZEOF_YMMWORD]
.column_ld64:
    test        cl, 2 * SIZEOF_YMMWORD
    mov         rcx, SIZEOF_YMMWORD
    jz          short .rgb_gray_cnv
    vmovdqa     ymmB, ymmA
    vmovdqu     ymmA, YMMWORD [rsi + 0 * SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [rsi + 1 * SIZEOF_YMMWORD]
    jmp         short .rgb_gray_cnv

.columnloop:
    vmovdqu     ymmA, YMMWORD [rsi + 0 * SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [rsi + 1 * SIZEOF_YMMWORD]
    vmovdqu     ymmB, YMMWORD [rsi + 2 * SIZEOF_YMMWORD]

.rgb_gray_cnv:
    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, and C to red, green, and blue.
    ;
    ; ymmA = (A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4 A5
    ;         B5 C5 A6 B6 C6 A7 B7 C7 A8 B8 C8 A9 B9 C9 Aa Ba)
    ; ymmF = (Ca Ab Bb Cb Ac Bc Cc Ad Bd Cd Ae Be Ce Af Bf Cf
    ;         Ag Bg Cg Ah Bh Ch Ai Bi Ci Aj Bj Cj Ak Bk Ck Al)
    ; ymmB = (Bl Cl Am Bm Cm An Bn Cn Ao Bo Co Ap Bp Cp Aq Bq
    ;         Cq Ar Br Cr As Bs Cs At Bt Ct Au Bu Cu Av Bv Cv)

    vmovdqu     ymmC, ymmA
    vinserti128 ymmA, ymmF, xmmA, 0
                ; ymmA = (A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4 A5
                ;         Ag Bg Cg Ah Bh Ch Ai Bi Ci Aj Bj Cj Ak Bk Ck Al)
    vinserti128 ymmC, ymmC, xmmB, 0
                ; ymmC = (Bl Cl Am Bm Cm An Bn Cn Ao Bo Co Ap Bp Cp Aq Bq
                ;         B5 C5 A6 B6 C6 A7 B7 C7 A8 B8 C8 A9 B9 C9 Aa Ba)
    vinserti128 ymmB, ymmB, xmmF, 0
                ; ymmB = (Ca Ab Bb Cb Ac Bc Cc Ad Bd Cd Ae Be Ce Af Bf Cf
                ;         Cq Ar Br Cr As Bs Cs At Bt Ct Au Bu Cu Av Bv Cv)
    vperm2i128  ymmF, ymmC, ymmC, 1
                ; ymmF = (B5 C5 A6 B6 C6 A7 B7 C7 A8 B8 C8 A9 B9 C9 Aa Ba
                ;         Bl Cl Am Bm Cm An Bn Cn Ao Bo Co Ap Bp Cp Aq Bq)

    vmovdqa     ymmG, ymmA
    vpslldq     ymmA, ymmA, 8
                ; ymmA = (-- -- -- -- -- -- -- -- A0 B0 C0 A1 B1 C1 A2 B2
                ;         C2 A3 B3 C3 A4 B4 C4 A5 Ag Bg Cg Ah Bh Ch Ai Bi)
    vpsrldq     ymmG, ymmG, 8
                ; ymmG = (C2 A3 B3 C3 A4 B4 C4 A5 Ag Bg Cg Ah Bh Ch Ai Bi
                ;         Ci Aj Bj Cj Ak Bk Ck Al -- -- -- -- -- -- -- --)

    vpunpckhbw  ymmA, ymmA, ymmF
                ; ymmA = (A0 A8 B0 B8 C0 C8 A1 A9 B1 B9 C1 C9 A2 Aa B2 Ba
                ;         Ag Ao Bg Bo Cg Co Ah Ap Bh Bp Ch Cp Ai Aq Bi Bq)
    vpslldq     ymmF, ymmF, 8
                ; ymmF = (-- -- -- -- -- -- -- -- B5 C5 A6 B6 C6 A7 B7 C7
                ;         A8 B8 C8 A9 B9 C9 Aa Ba Bl Cl Am Bm Cm An Bn Cn)

    vpunpcklbw  ymmG, ymmG, ymmB
                ; ymmG = (C2 Ca A3 Ab B3 Bb C3 Cb A4 Ac B4 Bc C4 Cc A5 Ad
                ;         Ci Cq Aj Ar Bj Br Cj Cr Ak As Bk Bs Ck Cs Al At)
    vpunpckhbw  ymmF, ymmF, ymmB
                ; ymmF = (B5 Bd C5 Cd A6 Ae B6 Be C6 Ce A7 Af B7 Bf C7 Cf
                ;         Bl Bt Cl Ct Am Au Bm Bu Cm Cu An Av Bn Bv Cn Cv)

    vmovdqa     ymmD, ymmA
    vpslldq     ymmA, ymmA, 8
                ; ymmA = (-- -- -- -- -- -- -- -- A0 A8 B0 B8 C0 C8 A1 A9
                ;         B1 B9 C1 C9 A2 Aa B2 Ba Ag Ao Bg Bo Cg Co Ah Ap)
    vpsrldq     ymmD, ymmD, 8
                ; ymmD = (B1 B9 C1 C9 A2 Aa B2 Ba Ag Ao Bg Bo Cg Co Ah Ap
                ;         Bh Bp Ch Cp Ai Aq Bi Bq -- -- -- -- -- -- -- --)

    vpunpckhbw  ymmA, ymmA, ymmG
                ; ymmA = (A0 A4 A8 Ac B0 B4 B8 Bc C0 C4 C8 Cc A1 A5 A9 Ad
                ;         Ag Ak Ao As Bg Bk Bo Bs Cg Ck Co Cs Ah Al Ap At)
    vpslldq     ymmG, ymmG, 8
                ; ymmG = (-- -- -- -- -- -- -- -- C2 Ca A3 Ab B3 Bb C3 Cb
                ;         A4 Ac B4 Bc C4 Cc A5 Ad Ci Cq Aj Ar Bj Br Cj Cr)

    vpunpcklbw  ymmD, ymmD, ymmF
                ; ymmD = (B1 B5 B9 Bd C1 C5 C9 Cd A2 A6 Aa Ae B2 B6 Ba Be
                ;         Bh Bl Bp Bt Ch Cl Cp Ct Ai Am Aq Au Bi Bm Bq Bu)
    vpunpckhbw  ymmG, ymmG, ymmF
                ; ymmG = (C2 C6 Ca Ce A3 A7 Ab Af B3 B7 Bb Bf C3 C7 Cb Cf
                ;         Ci Cm Cq Cu Aj An Ar Av Bj Bn Br Bv Cj Cn Cr Cv)

    vmovdqa     ymmE, ymmA
    vpslldq     ymmA, ymmA, 8
                ; ymmA = (-- -- -- -- -- -- -- -- A0 A4 A8 Ac B0 B4 B8 Bc
                ;         C0 C4 C8 Cc A1 A5 A9 Ad Ag Ak Ao As Bg Bk Bo Bs)
    vpsrldq     ymmE, ymmE, 8
                ; ymmE = (C0 C4 C8 Cc A1 A5 A9 Ad Ag Ak Ao As Bg Bk Bo Bs
                ;         Cg Ck Co Cs Ah Al Ap At -- -- -- -- -- -- -- --)

    vpunpckhbw  ymmA, ymmA, ymmD
                ; ymmA = (A0 A2 A4 A6 A8 Aa Ac Ae B0 B2 B4 B6 B8 Ba Bc Be
                ;         Ag Ai Ak Am Ao Aq As Au Bg Bi Bk Bm Bo Bq Bs Bu)
    vpslldq     ymmD, ymmD, 8
                ; ymmD = (-- -- -- -- -- -- -- -- B1 B5 B9 Bd C1 C5 C9 Cd
                ;         A2 A6 Aa Ae B2 B6 Ba Be Bh Bl Bp Bt Ch Cl Cp Ct)

    vpunpcklbw  ymmE, ymmE, ymmG
                ; ymmE = (C0 C2 C4 C6 C8 Ca Cc Ce A1 A3 A5 A7 A9 Ab Ad Af
                ;         Cg Ci Ck Cm Co Cq Cs Cu Ah Aj Al An Ap Ar At Av)
    vpunpckhbw  ymmD, ymmD, ymmG
                ; ymmD = (B1 B3 B5 B7 B9 Bb Bd Bf C1 C3 C5 C7 C9 Cb Cd Cf
                ;         Bh Bj Bl Bn Bp Br Bt Bv Ch Cj Cl Cn Cp Cr Ct Cv)

    vpxor       ymmH, ymmH, ymmH

    vmovdqa     ymmC, ymmA
    vpunpcklbw  ymmA, ymmA, ymmH
                ; ymmA = (A0 A2 A4 A6 A8 Aa Ac Ae Ag Ai Ak Am Ao Aq As Au) = AE
    vpunpckhbw  ymmC, ymmC, ymmH
                ; ymmC = (B0 B2 B4 B6 B8 Ba Bc Be Bg Bi Bk Bm Bo Bq Bs Bu) = BE

    vmovdqa     ymmB, ymmE
    vpunpcklbw  ymmE, ymmE, ymmH
                ; ymmE = (C0 C2 C4 C6 C8 Ca Cc Ce Cg Ci Ck Cm Co Cq Cs Cu) = CE
    vpunpckhbw  ymmB, ymmB, ymmH
                ; ymmB = (A1 A3 A5 A7 A9 Ab Ad Af Ah Aj Al An Ap Ar At Av) = AO

    vmovdqa     ymmF, ymmD
    vpunpcklbw  ymmD, ymmD, ymmH
                ; ymmD = (B1 B3 B5 B7 B9 Bb Bd Bf Bh Bj Bl Bn Bp Br Bt Bv) = BO
    vpunpckhbw  ymmF, ymmF, ymmH
                ; ymmF = (C1 C3 C5 C7 C9 Cb Cd Cf Ch Cj Cl Cn Cp Cr Ct Cv) = CO

%else  ; RGB_PIXELSIZE == 4 ; -----------

.column_ld1:
    test        cl, SIZEOF_XMMWORD / 16
    jz          short .column_ld2
    sub         rcx, byte SIZEOF_XMMWORD / 16
    vmovd       xmmA, XMM_DWORD [rsi + rcx * RGB_PIXELSIZE]
.column_ld2:
    test        cl, SIZEOF_XMMWORD / 8
    jz          short .column_ld4
    sub         rcx, byte SIZEOF_XMMWORD / 8
    vmovq       xmmF, XMM_MMWORD [rsi + rcx * RGB_PIXELSIZE]
    vpslldq     xmmA, xmmA, SIZEOF_MMWORD
    vpor        xmmA, xmmA, xmmF
.column_ld4:
    test        cl, SIZEOF_XMMWORD / 4
    jz          short .column_ld8
    sub         rcx, byte SIZEOF_XMMWORD / 4
    vmovdqa     xmmF, xmmA
    vperm2i128  ymmF, ymmF, ymmF, 1
    vmovdqu     xmmA, XMMWORD [rsi + rcx * RGB_PIXELSIZE]
    vpor        ymmA, ymmA, ymmF
.column_ld8:
    test        cl, SIZEOF_XMMWORD / 2
    jz          short .column_ld16
    sub         rcx, byte SIZEOF_XMMWORD / 2
    vmovdqa     ymmF, ymmA
    vmovdqu     ymmA, YMMWORD [rsi + rcx * RGB_PIXELSIZE]
.column_ld16:
    test        cl, SIZEOF_XMMWORD
    mov         rcx, SIZEOF_YMMWORD
    jz          short .rgb_gray_cnv
    vmovdqa     ymmE, ymmA
    vmovdqa     ymmH, ymmF
    vmovdqu     ymmA, YMMWORD [rsi + 0 * SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [rsi + 1 * SIZEOF_YMMWORD]
    jmp         short .rgb_gray_cnv

.columnloop:
    vmovdqu     ymmA, YMMWORD [rsi + 0 * SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [rsi + 1 * SIZEOF_YMMWORD]
    vmovdqu     ymmE, YMMWORD [rsi + 2 * SIZEOF_YMMWORD]
    vmovdqu     ymmH, YMMWORD [rsi + 3 * SIZEOF_YMMWORD]

.rgb_gray_cnv:
    ; NOTE: The values of RGB_RED, RGB_GREEN, and RGB_BLUE determine the
    ; mapping of components A, B, C, and D to red, green, and blue.
    ;
    ; ymmA = (A0 B0 C0 D0 A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3
    ;         A4 B4 C4 D4 A5 B5 C5 D5 A6 B6 C6 D6 A7 B7 C7 D7)
    ; ymmF = (A8 B8 C8 D8 A9 B9 C9 D9 Aa Ba Ca Da Ab Bb Cb Db
    ;         Ac Bc Cc Dc Ad Bd Cd Dd Ae Be Ce De Af Bf Cf Df)
    ; ymmE = (Ag Bg Cg Dg Ah Bh Ch Dh Ai Bi Ci Di Aj Bj Cj Dj
    ;         Ak Bk Ck Dk Al Bl Cl Dl Am Bm Cm Dm An Bn Cn Dn)
    ; ymmH = (Ao Bo Co Do Ap Bp Cp Dp Aq Bq Cq Dq Ar Br Cr Dr
    ;         As Bs Cs Ds At Bt Ct Dt Au Bu Cu Du Av Bv Cv Dv)

    vmovdqa     ymmB, ymmA
    vinserti128 ymmA, ymmA, xmmE, 1
                ; ymmA = (A0 B0 C0 D0 A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3
                ;         Ag Bg Cg Dg Ah Bh Ch Dh Ai Bi Ci Di Aj Bj Cj Dj)
    vperm2i128  ymmE, ymmB, ymmE, 0x31
                ; ymmE = (A4 B4 C4 D4 A5 B5 C5 D5 A6 B6 C6 D6 A7 B7 C7 D7
                ;         Ak Bk Ck Dk Al Bl Cl Dl Am Bm Cm Dm An Bn Cn Dn)

    vmovdqa     ymmB, ymmF
    vinserti128 ymmF, ymmF, xmmH, 1
                ; ymmF = (A8 B8 C8 D8 A9 B9 C9 D9 Aa Ba Ca Da Ab Bb Cb Db
                ;         Ao Bo Co Do Ap Bp Cp Dp Aq Bq Cq Dq Ar Br Cr Dr)
    vperm2i128  ymmH, ymmB, ymmH, 0x31
                ; ymmH = (Ac Bc Cc Dc Ad Bd Cd Dd Ae Be Ce De Af Bf Cf Df
                ;         As Bs Cs Ds At Bt Ct Dt Au Bu Cu Du Av Bv Cv Dv)

    vmovdqa     ymmD, ymmA
    vpunpcklbw  ymmA, ymmA, ymmE
                ; ymmA = (A0 A4 B0 B4 C0 C4 D0 D4 A1 A5 B1 B5 C1 C5 D1 D5
                ;         Ag Ak Bg Bk Cg Ck Dg Dk Ah Al Bh Bl Ch Cl Dh Dl)
    vpunpckhbw  ymmD, ymmD, ymmE
                ; ymmD = (A2 A6 B2 B6 C2 C6 D2 D6 A3 A7 B3 B7 C3 C7 D3 D7
                ;         Ai Am Bi Bm Ci Cm Di Dm Aj An Bj Bn Cj Cn Dj Dn)

    vmovdqa     ymmC, ymmF
    vpunpcklbw  ymmF, ymmF, ymmH
                ; ymmF = (A8 Ac B8 Bc C8 Cc D8 Dc A9 Ad B9 Bd C9 Cd D9 Dd
                ;         Ao As Bo Bs Co Cs Do Ds Ap At Bp Bt Cp Ct Dp Dt)
    vpunpckhbw  ymmC, ymmC, ymmH
                ; ymmC = (Aa Ae Ba Be Ca Ce Da De Ab Af Bb Bf Cb Cf Db Df
                ;         Aq Au Bq Bu Cq Cu Dq Du Ar Av Br Bv Cr Cv Dr Dv)

    vmovdqa     ymmB, ymmA
    vpunpcklwd  ymmA, ymmA, ymmF
                ; ymmA = (A0 A4 A8 Ac B0 B4 B8 Bc C0 C4 C8 Cc D0 D4 D8 Dc
                ;         Ag Ak Ao As Bg Bk Bo Bs Cg Ck Co Cs Dg Dk Do Ds)
    vpunpckhwd  ymmB, ymmB, ymmF
                ; ymmB = (A1 A5 A9 Ad B1 B5 B9 Bd C1 C5 C9 Cd D1 D5 D9 Dd
                ;         Ah Al Ap At Bh Bl Bp Bt Ch Cl Cp Ct Dh Dl Dp Dt)

    vmovdqa     ymmG, ymmD
    vpunpcklwd  ymmD, ymmD, ymmC
                ; ymmD = (A2 A6 Aa Ae B2 B6 Ba Be C2 C6 Ca Ce D2 D6 Da De
                ;         Ai Am Aq Au Bi Bm Bq Bu Ci Cm Cq Cu Di Dm Dq Du)
    vpunpckhwd  ymmG, ymmG, ymmC
                ; ymmG = (A3 A7 Ab Af B3 B7 Bb Bf C3 C7 Cb Cf D3 D7 Db Df
                ;         Aj An Ar Av Bj Bn Br Bv Cj Cn Cr Cv Dj Dn Dr Dv)

    vmovdqa     ymmE, ymmA
    vpunpcklbw  ymmA, ymmA, ymmD
                ; ymmA = (A0 A2 A4 A6 A8 Aa Ac Ae B0 B2 B4 B6 B8 Ba Bc Be
                ;         Ag Ai Ak Am Ao Aq As Au Bg Bi Bk Bm Bo Bq Bs Bu)
    vpunpckhbw  ymmE, ymmE, ymmD
                ; ymmE = (C0 C2 C4 C6 C8 Ca Cc Ce D0 D2 D4 D6 D8 Da Dc De
                ;         Cg Ci Ck Cm Co Cq Cs Cu Dg Di Dk Dm Do Dq Ds Du)

    vmovdqa     ymmH, ymmB
    vpunpcklbw  ymmB, ymmB, ymmG
                ; ymmB = (A1 A3 A5 A7 A9 Ab Ad Af B1 B3 B5 B7 B9 Bb Bd Bf
                ;         Ah Aj Al An Ap Ar At Av Bh Bj Bl Bn Bp Br Bt Bv)
    vpunpckhbw  ymmH, ymmH, ymmG
                ; ymmH = (C1 C3 C5 C7 C9 Cb Cd Cf D1 D3 D5 D7 D9 Db Dd Df
                ;         Ch Cj Cl Cn Cp Cr Ct Cv Dh Dj Dl Dn Dp Dr Dt Dv)

    vpxor       ymmF, ymmF, ymmF

    vmovdqa     ymmC, ymmA
    vpunpcklbw  ymmA, ymmA, ymmF
                ; ymmA = (A0 A2 A4 A6 A8 Aa Ac Ae Ag Ai Ak Am Ao Aq As Au) = AE
    vpunpckhbw  ymmC, ymmC, ymmF
                ; ymmC = (B0 B2 B4 B6 B8 Ba Bc Be Bg Bi Bk Bm Bo Bq Bs Bu) = BE

    vmovdqa     ymmD, ymmB
    vpunpcklbw  ymmB, ymmB, ymmF
                ; ymmB = (A1 A3 A5 A7 A9 Ab Ad Af Ah Aj Al An Ap Ar At Av) = AO
    vpunpckhbw  ymmD, ymmD, ymmF
                ; ymmD = (B1 B3 B5 B7 B9 Bb Bd Bf Bh Bj Bl Bn Bp Br Bt Bv) = BO

    vmovdqa     ymmG, ymmE
    vpunpcklbw  ymmE, ymmE, ymmF
                ; ymmE = (C0 C2 C4 C6 C8 Ca Cc Ce Cg Ci Ck Cm Co Cq Cs Cu) = CE
    vpunpckhbw  ymmG, ymmG, ymmF
                ; ymmG = (D0 D2 D4 D6 D8 Da Dc De Dg Di Dk Dm Do Dq Ds Du) = DE

    vpunpcklbw  ymmF, ymmF, ymmH
    vpunpckhbw  ymmH, ymmH, ymmH
    vpsrlw      ymmF, ymmF, BYTE_BIT
                ; ymmF = (C1 C3 C5 C7 C9 Cb Cd Cf Ch Cj Cl Cn Cp Cr Ct Cv) = CO
    vpsrlw      ymmH, ymmH, BYTE_BIT
                ; ymmH = (D1 D3 D5 D7 D9 Db Dd Df Dh Dj Dl Dn Dp Dr Dt Dv) = DO

%endif  ; RGB_PIXELSIZE ; ---------------

    ; ymm0 = (R0 R2 R4 R6 R8 Ra Rc Re Rg Ri Rk Rm Ro Rq Rs Ru) = RE
    ; ymm2 = (G0 G2 G4 G6 G8 Ga Gc Ge Gg Gi Gk Gm Go Gq Gs Gu) = GE
    ; ymm4 = (B0 B2 B4 B6 B8 Ba Bc Be Bg Bi Bk Bm Bo Bq Bs Bu) = BE
    ; ymm1 = (R1 R3 R5 R7 R9 Rb Rd Rf Rh Rj Rl Rn Rp Rr Rt Rv) = RO
    ; ymm3 = (G1 G3 G5 G7 G9 Gb Gd Gf Gh Gj Gl Gn Gp Gr Gt Gv) = GO
    ; ymm5 = (B1 B3 B5 B7 B9 Bb Bd Bf Bh Bj Bl Bn Bp Br Bt Bv) = BO
    ;
    ; (Original)
    ; Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    ;
    ; (This implementation)
    ; Y  =  0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G

    vmovdqa     ymm6, ymm1
    vpunpcklwd  ymm1, ymm1, ymm3
    vpunpckhwd  ymm6, ymm6, ymm3
    vpmaddwd    ymm1, ymm1, [rel PW_F0299_F0337]
                ; ymm1 = ROL * FIX(0.299) + GOL * FIX(0.337)
    vpmaddwd    ymm6, ymm6, [rel PW_F0299_F0337]
                ; ymm6 = ROH * FIX(0.299) + GOH * FIX(0.337)

    vmovdqa     ymm7, ymm6
                ; ymm7 = ROH * FIX(0.299) + GOH * FIX(0.337)

    vmovdqa     ymm6, ymm0
    vpunpcklwd  ymm0, ymm0, ymm2
    vpunpckhwd  ymm6, ymm6, ymm2
    vpmaddwd    ymm0, ymm0, [rel PW_F0299_F0337]
                ; ymm0 = REL * FIX(0.299) + GEL * FIX(0.337)
    vpmaddwd    ymm6, ymm6, [rel PW_F0299_F0337]
                ; ymm6 = REH * FIX(0.299) + GEH * FIX(0.337)

    vmovdqa     YMMWORD [wk(0)], ymm0
                ; wk(0) = REL * FIX(0.299) + GEL * FIX(0.337)
    vmovdqa     YMMWORD [wk(1)], ymm6
                ; wk(1) = REH * FIX(0.299) + GEH * FIX(0.337)

    vmovdqa     ymm0, ymm5              ; ymm0 = BO
    vmovdqa     ymm6, ymm4              ; ymm6 = BE

    vmovdqa     ymm4, ymm0
    vpunpcklwd  ymm0, ymm0, ymm3
    vpunpckhwd  ymm4, ymm4, ymm3
    vpmaddwd    ymm0, ymm0, [rel PW_F0114_F0250]
                ; ymm0 = BOL * FIX(0.114) + GOL * FIX(0.250)
    vpmaddwd    ymm4, ymm4, [rel PW_F0114_F0250]
                ; ymm4 = BOH * FIX(0.114) + GOH * FIX(0.250)

    vmovdqa     ymm3, [rel PD_ONEHALF]  ; ymm3 = [PD_ONEHALF]

    vpaddd      ymm0, ymm0, ymm1
    vpaddd      ymm4, ymm4, ymm7
    vpaddd      ymm0, ymm0, ymm3
    vpaddd      ymm4, ymm4, ymm3
    vpsrld      ymm0, ymm0, SCALEBITS   ; ymm0 = YOL
    vpsrld      ymm4, ymm4, SCALEBITS   ; ymm4 = YOH
    vpackssdw   ymm0, ymm0, ymm4        ; ymm0 = YO

    vmovdqa     ymm4, ymm6
    vpunpcklwd  ymm6, ymm6, ymm2
    vpunpckhwd  ymm4, ymm4, ymm2
    vpmaddwd    ymm6, ymm6, [rel PW_F0114_F0250]
                ; ymm6 = BEL * FIX(0.114) + GEL * FIX(0.250)
    vpmaddwd    ymm4, ymm4, [rel PW_F0114_F0250]
                ; ymm4 = BEH * FIX(0.114) + GEH * FIX(0.250)

    vmovdqa     ymm2, [rel PD_ONEHALF]  ; ymm2 = [PD_ONEHALF]

    vpaddd      ymm6, ymm6, YMMWORD [wk(0)]
    vpaddd      ymm4, ymm4, YMMWORD [wk(1)]
    vpaddd      ymm6, ymm6, ymm2
    vpaddd      ymm4, ymm4, ymm2
    vpsrld      ymm6, ymm6, SCALEBITS   ; ymm6 = YEL
    vpsrld      ymm4, ymm4, SCALEBITS   ; ymm4 = YEH
    vpackssdw   ymm6, ymm6, ymm4        ; ymm6 = YE

    vpsllw      ymm0, ymm0, BYTE_BIT
    vpor        ymm6, ymm6, ymm0        ; ymm6 = Y
    vmovdqu     YMMWORD [rdi], ymm6     ; Save Y

    sub         rcx, byte SIZEOF_YMMWORD
    add         rsi, RGB_PIXELSIZE * SIZEOF_YMMWORD  ; inptr
    add         rdi, byte SIZEOF_YMMWORD             ; outptr0
    cmp         rcx, byte SIZEOF_YMMWORD
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
    vzeroupper
    UNCOLLECT_ARGS 5
    lea         rsp, [rbp - 8]
    pop         r15
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
