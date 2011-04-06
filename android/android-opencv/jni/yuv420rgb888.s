; YUV-> RGB conversion code Copyright (C) 2008 Robin Watts (robin;wss.co.uk).
;
; Licensed under the GPL. If you need it under another license, contact me
; and ask.
;
;  This program is free software ; you can redistribute it and/or modify
;  it under the terms of the GNU General Public License as published by
;  the Free Software Foundation ; either version 2 of the License, or
;  (at your option) any later version.
;
;  This program is distributed in the hope that it will be useful,
;  but WITHOUT ANY WARRANTY ; without even the implied warranty of
;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;  GNU General Public License for more details.
;
;  You should have received a copy of the GNU General Public License
;  along with this program ; if not, write to the Free Software
;  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
;
;
; The algorithm used here is based heavily on one created by Sophie Wilson
; of Acorn/e-14/Broadcomm. Many thanks.
;
; Additional tweaks (in the fast fixup code) are from Paul Gardiner.
;
; The old implementation of YUV -> RGB did:
;
; R = CLAMP((Y-16)*1.164 +           1.596*V)
; G = CLAMP((Y-16)*1.164 - 0.391*U - 0.813*V)
; B = CLAMP((Y-16)*1.164 + 2.018*U          )
;
; We're going to bend that here as follows:
;
; R = CLAMP(y +           1.596*V)
; G = CLAMP(y - 0.383*U - 0.813*V)
; B = CLAMP(y + 1.976*U          )
;
; where y = 0               for       Y <=  16,
;       y = (  Y-16)*1.164, for  16 < Y <= 239,
;       y = (239-16)*1.164, for 239 < Y
;
; i.e. We clamp Y to the 16 to 239 range (which it is supposed to be in
; anyway). We then pick the B_U factor so that B never exceeds 511. We then
; shrink the G_U factor in line with that to avoid a colour shift as much as
; possible.
;
; We're going to use tables to do it faster, but rather than doing it using
; 5 tables as as the above suggests, we're going to do it using just 3.
;
; We do this by working in parallel within a 32 bit word, and using one
; table each for Y U and V.
;
; Source Y values are    0 to 255, so    0.. 260 after scaling
; Source U values are -128 to 127, so  -49.. 49(G), -253..251(B) after
; Source V values are -128 to 127, so -204..203(R), -104..103(G) after
;
; So total summed values:
; -223 <= R <= 481, -173 <= G <= 431, -253 <= B < 511
;
; We need to pack R G and B into a 32 bit word, and because of Bs range we
; need 2 bits above the valid range of B to detect overflow, and another one
; to detect the sense of the overflow. We therefore adopt the following
; representation:
;
; osGGGGGgggggosBBBBBbbbosRRRRRrrr
;
; Each such word breaks down into 3 ranges.
;
; osGGGGGggggg   osBBBBBbbb   osRRRRRrrr
;
; Thus we have 8 bits for each B and R table entry, and 10 bits for G (good
; as G is the most noticable one). The s bit for each represents the sign,
; and o represents the overflow.
;
; For R and B we pack the table by taking the 11 bit representation of their
; values, and toggling bit 10 in the U and V tables.
;
; For the green case we calculate 4*G (thus effectively using 10 bits for the
; valid range) truncate to 12 bits. We toggle bit 11 in the Y table.

; Theorarm library
; Copyright (C) 2009 Robin Watts for Pinknoise Productions Ltd

	AREA	|.text|, CODE, READONLY

	EXPORT	yuv420_2_rgb888
	EXPORT	yuv420_2_rgb888_PROFILE

; void yuv420_2_rgb565
;  uint8_t *dst_ptr
;  uint8_t *y_ptr
;  uint8_t *u_ptr
;  uint8_t *v_ptr
;  int      width
;  int      height
;  int      y_span
;  int      uv_span
;  int      dst_span
;  int     *tables
;  int      dither

CONST_flags
	DCD	0x40080100
yuv420_2_rgb888
	; r0 = dst_ptr
	; r1 = y_ptr
	; r2 = u_ptr
	; r3 = v_ptr
	; <> = width
	; <> = height
	; <> = y_span
	; <> = uv_span
	; <> = dst_span
	; <> = y_table
	; <> = dither
	STMFD	r13!,{r4-r11,r14}

	LDR	r8, [r13,#10*4]		; r8 = height
	LDR	r10,[r13,#11*4]		; r10= y_span
	LDR	r9, [r13,#13*4]		; r9 = dst_span
	LDR	r14,[r13,#14*4]		; r14= y_table
	LDR	r5, CONST_flags
	LDR	r11,[r13,#9*4]		; r11= width
	ADD	r4, r14, #256*4
	SUBS	r8, r8, #1
	BLT	end
	BEQ	trail_row1
yloop1
	SUB	r8, r8, r11,LSL #16	; r8 = height-(width<<16)
	ADDS	r8, r8, #1<<16		; if (width == 1)
	BGE	trail_pair1		;    just do 1 column
xloop1
	LDRB	r11,[r2], #1		; r11 = u  = *u_ptr++
	LDRB	r12,[r3], #1		; r12 = v  = *v_ptr++
	LDRB	r7, [r1, r10]		; r7  = y2 = y_ptr[stride]
	LDRB	r6, [r1], #1		; r6  = y0 = *y_ptr++
	ADD	r12,r12,#512
	LDR	r11,[r4, r11,LSL #2]	; r11 = u  = u_table[u]
	LDR	r12,[r14,r12,LSL #2]	; r12 = v  = v_table[v]
	LDR	r7, [r14,r7, LSL #2]	; r7  = y2 = y_table[y2]
	LDR	r6, [r14,r6, LSL #2]	; r6  = y0 = y_table[y0]
	ADD	r11,r11,r12		; r11 = uv = u+v

	ADD	r7, r7, r11		; r7  = y2 + uv
	ADD	r6, r6, r11		; r6  = y0 + uv
	ANDS	r12,r7, r5
	TSTEQ	r6, r5
	BNE	fix101
return101
	; Store the bottom one first
	ADD	r12,r0, r9
	STRB	r7,[r12],#1		; Store R
	MOV	r7, r7, ROR #22
	STRB	r7,[r12],#1		; Store G
	MOV	r7, r7, ROR #21
	STRB	r7,[r12],#1		; Store B

	; Then store the top one
	STRB	r6,[r0], #1		; Store R
	MOV	r6, r6, ROR #22
	STRB	r6,[r0], #1		; Store G

	LDRB	r7, [r1, r10]		; r7 = y3 = y_ptr[stride]
	LDRB	r12,[r1], #1		; r12= y1 = *y_ptr++
	MOV	r6, r6, ROR #21
	LDR	r7, [r14, r7, LSL #2]	; r7 = y3 = y_table[y2]
	LDR	r12,[r14, r12,LSL #2]	; r12= y1 = y_table[y0]
	STRB	r6,[r0], #1		; Store B

	ADD	r7, r7, r11		; r7  = y3 + uv
	ADD	r6, r12,r11		; r6  = y1 + uv
	ANDS	r12,r7, r5
	TSTEQ	r6, r5
	BNE	fix102
return102
	; Store the bottom one first
	ADD	r12,r0, r9
	STRB	r7,[r12],#1		; Store R
	MOV	r7, r7, ROR #22
	STRB	r7,[r12],#1		; Store G
	MOV	r7, r7, ROR #21
	STRB	r7,[r12],#1		; Store B

	; Then store the top one
	STRB	r6,[r0], #1		; Store R
	MOV	r6, r6, ROR #22
	STRB	r6,[r0], #1		; Store G
	MOV	r6, r6, ROR #21
	STRB	r6,[r0], #1		; Store B

	ADDS	r8, r8, #2<<16
	BLT	xloop1
	MOVS	r8, r8, LSL #16		; Clear the top 16 bits of r8
	MOV	r8, r8, LSR #16		; If the C bit is clear we still have
	BCC	trail_pair1		; 1 more pixel pair to do
end_xloop1
	LDR	r11,[r13,#9*4]		; r11= width
	LDR	r12,[r13,#12*4]		; r12= uv_stride
	ADD	r0, r0, r9, LSL #1
	SUB	r0, r0, r11,LSL #1
	SUB	r0, r0, r11
	ADD	r1, r1, r10,LSL #1
	SUB	r1, r1, r11
	SUB	r2, r2, r11,LSR #1
	SUB	r3, r3, r11,LSR #1
	ADD	r2, r2, r12
	ADD	r3, r3, r12

	SUBS	r8, r8, #2
	BGT	yloop1

	LDMLTFD	r13!,{r4-r11,pc}
trail_row1
	; We have a row of pixels left to do
	SUB	r8, r8, r11,LSL #16	; r8 = height-(width<<16)
	ADDS	r8, r8, #1<<16		; if (width == 1)
	BGE	trail_pix1		;    just do 1 pixel
xloop12
	LDRB	r11,[r2], #1		; r11 = u  = *u_ptr++
	LDRB	r12,[r3], #1		; r12 = v  = *v_ptr++
	LDRB	r6, [r1], #1		; r6  = y0 = *y_ptr++
	LDRB	r7, [r1], #1		; r7  = y1 = *y_ptr++
	ADD	r12,r12,#512
	LDR	r11,[r4, r11,LSL #2]	; r11 = u  = u_table[u]
	LDR	r12,[r14,r12,LSL #2]	; r12 = v  = v_table[v]
	LDR	r7, [r14,r7, LSL #2]	; r7  = y1 = y_table[y1]
	LDR	r6, [r14,r6, LSL #2]	; r6  = y0 = y_table[y0]
	ADD	r11,r11,r12		; r11 = uv = u+v

	ADD	r6, r6, r11		; r6  = y0 + uv
	ADD	r7, r7, r11		; r7  = y1 + uv
	ANDS	r12,r7, r5
	TSTEQ	r6, r5
	BNE	fix104
return104
	; Store the bottom one first
	STRB	r6,[r0], #1		; Store R
	MOV	r6, r6, ROR #22
	STRB	r6,[r0], #1		; Store G
	MOV	r6, r6, ROR #21
	STRB	r6,[r0], #1		; Store B

	; Then store the top one
	STRB	r7,[r0], #1		; Store R
	MOV	r7, r7, ROR #22
	STRB	r7,[r0], #1		; Store G
	MOV	r7, r7, ROR #21
	STRB	r7,[r0], #1		; Store B

	ADDS	r8, r8, #2<<16
	BLT	xloop12
	MOVS	r8, r8, LSL #16		; Clear the top 16 bits of r8
	MOV	r8, r8, LSR #16		; If the C bit is clear we still have
	BCC	trail_pix1		; 1 more pixel pair to do
end
	LDMFD	r13!,{r4-r11,pc}
trail_pix1
	; We have a single extra pixel to do
	LDRB	r11,[r2], #1		; r11 = u  = *u_ptr++
	LDRB	r12,[r3], #1		; r12 = v  = *v_ptr++
	LDRB	r6, [r1], #1		; r6  = y0 = *y_ptr++
	ADD	r12,r12,#512
	LDR	r11,[r4, r11,LSL #2]	; r11 = u  = u_table[u]
	LDR	r12,[r14,r12,LSL #2]	; r12 = v  = v_table[v]
	LDR	r6, [r14,r6, LSL #2]	; r6  = y0 = y_table[y0]
	ADD	r11,r11,r12		; r11 = uv = u+v

	ADD	r6, r6, r11		; r6  = y0 + uv
	ANDS	r12,r6, r5
	BNE	fix105
return105
	STRB	r6,[r0], #1		; Store R
	MOV	r6, r6, ROR #22
	STRB	r6,[r0], #1		; Store G
	MOV	r6, r6, ROR #21
	STRB	r6,[r0], #1		; Store B

	LDMFD	r13!,{r4-r11,pc}

trail_pair1
	; We have a pair of pixels left to do
	LDRB	r11,[r2]		; r11 = u  = *u_ptr++
	LDRB	r12,[r3]		; r12 = v  = *v_ptr++
	LDRB	r7, [r1, r10]		; r7  = y2 = y_ptr[stride]
	LDRB	r6, [r1], #1		; r6  = y0 = *y_ptr++
	ADD	r12,r12,#512
	LDR	r11,[r4, r11,LSL #2]	; r11 = u  = u_table[u]
	LDR	r12,[r14,r12,LSL #2]	; r12 = v  = v_table[v]
	LDR	r7, [r14,r7, LSL #2]	; r7  = y2 = y_table[y2]
	LDR	r6, [r14,r6, LSL #2]	; r6  = y0 = y_table[y0]
	ADD	r11,r11,r12		; r11 = uv = u+v

	ADD	r7, r7, r11		; r7  = y2 + uv
	ADD	r6, r6, r11		; r6  = y0 + uv
	ANDS	r12,r7, r5
	TSTEQ	r6, r5
	BNE	fix103
return103
	; Store the bottom one first
	ADD	r12,r0, r9
	STRB	r7,[r12],#1		; Store R
	MOV	r7, r7, ROR #22
	STRB	r7,[r12],#1		; Store G
	MOV	r7, r7, ROR #21
	STRB	r7,[r12],#1		; Store B

	; Then store the top one
	STRB	r6,[r0], #1		; Store R
	MOV	r6, r6, ROR #22
	STRB	r6,[r0], #1		; Store G
	MOV	r6, r6, ROR #21
	STRB	r6,[r0], #1		; Store B
	B	end_xloop1
fix101
	; r7 and r6 are the values, at least one of which has overflowed
	; r12 = r7 & mask = .s......s......s......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r7, r7, r12		; r7 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r7, LSR #1	; r12 = .o......o......o......
	ADD	r7, r7, r12,LSR #8	; r7  = fixed value

	AND	r12, r6, r5		; r12 = .S......S......S......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r6, r6, r12		; r6 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r6, LSR #1	; r12 = .o......o......o......
	ADD	r6, r6, r12,LSR #8	; r6  = fixed value
	B	return101
fix102
	; r7 and r6 are the values, at least one of which has overflowed
	; r12 = r7 & mask = .s......s......s......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r7, r7, r12		; r7 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r7, LSR #1	; r12 = .o......o......o......
	ADD	r7, r7, r12,LSR #8	; r7  = fixed value

	AND	r12, r6, r5		; r12 = .S......S......S......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS..SSSSS.SSSSSS
	ORR	r6, r6, r12		; r6 |= ..SSSSSS..SSSSS.SSSSSS
	BIC	r12,r5, r6, LSR #1	; r12 = .o......o......o......
	ADD	r6, r6, r12,LSR #8	; r6  = fixed value
	B	return102
fix103
	; r7 and r6 are the values, at least one of which has overflowed
	; r12 = r7 & mask = .s......s......s......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r7, r7, r12		; r7 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r7, LSR #1	; r12 = .o......o......o......
	ADD	r7, r7, r12,LSR #8	; r7  = fixed value

	AND	r12, r6, r5		; r12 = .S......S......S......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r6, r6, r12		; r6 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r6, LSR #1	; r12 = .o......o......o......
	ADD	r6, r6, r12,LSR #8	; r6  = fixed value
	B	return103
fix104
	; r7 and r6 are the values, at least one of which has overflowed
	; r12 = r7 & mask = .s......s......s......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r7, r7, r12		; r7 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r7, LSR #1	; r12 = .o......o......o......
	ADD	r7, r7, r12,LSR #8	; r7  = fixed value

	AND	r12, r6, r5		; r12 = .S......S......S......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r6, r6, r12		; r6 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r6, LSR #1	; r12 = .o......o......o......
	ADD	r6, r6, r12,LSR #8	; r6  = fixed value
	B	return104
fix105
	; r6 is the value, which has has overflowed
	; r12 = r7 & mask = .s......s......s......
	SUB	r12,r12,r12,LSR #8	; r12 = ..SSSSSS.SSSSSS.SSSSSS
	ORR	r6, r6, r12		; r6 |= ..SSSSSS.SSSSSS.SSSSSS
	BIC	r12,r5, r6, LSR #1	; r12 = .o......o......o......
	ADD	r6, r6, r12,LSR #8	; r6  = fixed value
	B	return105

	END
