	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"samplers.cpp"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi
.LCPI0_0:
	.word	0x38d1b717                      # float 9.99999974E-5
	.section	.text._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi,"ax",@progbits
	.globl	_ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi
	.p2align	1
	.type	_ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi,@function
_ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi: # @_ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:
	addi	sp, sp, -736
	.cfi_def_cfa_offset 736
	sd	ra, 728(sp)                     # 8-byte Folded Spill
	sd	s0, 720(sp)                     # 8-byte Folded Spill
	sd	s1, 712(sp)                     # 8-byte Folded Spill
	sd	s2, 704(sp)                     # 8-byte Folded Spill
	sd	s3, 696(sp)                     # 8-byte Folded Spill
	sd	s4, 688(sp)                     # 8-byte Folded Spill
	sd	s5, 680(sp)                     # 8-byte Folded Spill
	sd	s6, 672(sp)                     # 8-byte Folded Spill
	sd	s7, 664(sp)                     # 8-byte Folded Spill
	sd	s8, 656(sp)                     # 8-byte Folded Spill
	sd	s9, 648(sp)                     # 8-byte Folded Spill
	sd	s10, 640(sp)                    # 8-byte Folded Spill
	sd	s11, 632(sp)                    # 8-byte Folded Spill
	fsd	fs0, 624(sp)                    # 8-byte Folded Spill
	fsd	fs1, 616(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	.cfi_offset fs0, -112
	.cfi_offset fs1, -120
	mv	s4, a3
	mv	s3, a2
	fmv.s	fs0, fa1
	fmv.s	fs1, fa0
	mv	s5, a1
	mv	s1, a0
	sext.w	s0, a1
.Lpcrel_hi0:
	auipc	a0, %pcrel_hi(_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE25__cv_trace_location_fn324)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi0)
	addi	a0, sp, 560
	call	_ZN2cv5utils5trace7details6RegionC1ERKNS3_21LocationStaticStorageE@plt
	blez	s0, .LBB0_91
# %bb.1:
	srai	s0, s5, 32
	blez	s0, .LBB0_92
# %bb.2:
.Ltmp3:
	addi	a0, sp, 352
	li	a2, -1
	mv	a1, s1
	call	_ZNK2cv11_InputArray7getMat_Ei@plt
.Ltmp4:
# %bb.3:
.Ltmp6:
	addi	a0, sp, 352
	call	_ZNK2cv3Mat5emptyEv@plt
.Ltmp7:
# %bb.4:
	beqz	a0, .LBB0_7
# %bb.5:
.Ltmp63:
	mv	a0, s3
	call	_ZNK2cv12_OutputArray7releaseEv@plt
.Ltmp64:
# %bb.6:
	addi	a0, sp, 352
	call	_ZN2cv3MatD1Ev@plt
	lw	a0, 568(sp)
	bnez	a0, .LBB0_77
	j	.LBB0_78
.LBB0_7:
	lw	a0, 352(sp)
	andi	s0, a0, 31
	slli	a0, a0, 52
	srli	s6, a0, 57
	addi	s2, s6, 1
	mv	s1, s0
	bltz	s4, .LBB0_9
# %bb.8:
	andi	s1, s4, 31
.LBB0_9:
	andi	a0, s2, 253
	li	a1, 1
	bne	a0, a1, .LBB0_95
# %bb.10:
	slli	a0, s2, 5
	or	a0, a0, s1
	addi	a2, a0, -32
.Ltmp14:
	li	a3, -1
	mv	a0, s3
	mv	a1, s5
	li	a4, 0
	li	a5, 0
	call	_ZNK2cv12_OutputArray7create_ENS_5Size_IiEEiibNS0_9DepthMaskE@plt
.Ltmp15:
# %bb.11:
.Ltmp17:
	addi	a0, sp, 144
	li	a2, -1
	mv	a1, s3
	call	_ZNK2cv11_InputArray7getMat_Ei@plt
.Ltmp18:
# %bb.12:
	or	a0, s1, s0
	beqz	a0, .LBB0_18
# %bb.13:
	xori	a0, s1, 5
	or	a1, s0, a0
	bnez	a1, .LBB0_22
# %bb.14:
	lw	a0, 424(sp)
	li	a1, 3
	bge	a0, a1, .LBB0_104
# %bb.15:
	blez	a0, .LBB0_37
# %bb.16:
	addi	a1, a0, -1
	snez	a1, a1
	slli	a1, a1, 2
	addi	a2, sp, 352
	or	a1, a1, a2
	lw	a2, 84(a1)
	li	a1, 1
	blt	a1, a0, .LBB0_38
.LBB0_17:
	not	a0, a0
	srliw	a6, a0, 31
	j	.LBB0_39
.LBB0_18:
	lw	a0, 424(sp)
	li	a1, 3
	bge	a0, a1, .LBB0_98
# %bb.19:
	blez	a0, .LBB0_27
# %bb.20:
	addi	a1, a0, -1
	snez	a1, a1
	slli	a1, a1, 2
	addi	a2, sp, 352
	or	a1, a1, a2
	lw	a1, 84(a1)
	li	a2, 1
	blt	a2, a0, .LBB0_28
.LBB0_21:
	not	a0, a0
	srliw	s0, a0, 31
	j	.LBB0_29
.LBB0_22:
	xori	a1, s0, 5
	or	a0, a0, a1
	bnez	a0, .LBB0_107
# %bb.23:
	lw	a0, 424(sp)
	li	a1, 3
	bge	a0, a1, .LBB0_110
# %bb.24:
	blez	a0, .LBB0_43
# %bb.25:
	addi	a1, a0, -1
	snez	a1, a1
	slli	a1, a1, 2
	addi	a2, sp, 352
	or	a1, a1, a2
	lw	a1, 84(a1)
	li	a2, 1
	blt	a2, a0, .LBB0_44
.LBB0_26:
	not	a0, a0
	srliw	s0, a0, 31
	j	.LBB0_45
.LBB0_27:
	seqz	a1, a0
	li	a2, 1
	bge	a2, a0, .LBB0_21
.LBB0_28:
	lw	s0, 436(sp)
.LBB0_29:
	lw	a5, 216(sp)
	li	a0, 3
	bge	a5, a0, .LBB0_101
# %bb.30:
	slli	s0, s0, 32
	slli	a1, a1, 32
	srli	a2, a1, 32
	blez	a5, .LBB0_32
# %bb.31:
	addi	a0, a5, -1
	snez	a0, a0
	slli	a0, a0, 2
	addi	a1, sp, 144
	or	a0, a0, a1
	lw	s1, 84(a0)
	j	.LBB0_33
.LBB0_32:
	seqz	s1, a5
.LBB0_33:
	ld	a0, 376(sp)
	ld	a1, 480(sp)
	ld	a3, 168(sp)
	ld	a4, 272(sp)
	li	a6, 1
	or	a2, a2, s0
	blt	a6, a5, .LBB0_35
# %bb.34:
	not	a5, a5
	srliw	a5, a5, 31
	j	.LBB0_36
.LBB0_35:
	lw	a5, 228(sp)
.LBB0_36:
	slli	a5, a5, 32
	slli	s1, s1, 32
	srli	s1, s1, 32
	or	a5, a5, s1
.Ltmp60:
	fmv.s	fa0, fs1
	fmv.s	fa1, fs0
	mv	a6, s2
	call	_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
.Ltmp61:
	j	.LBB0_76
.LBB0_37:
	seqz	a2, a0
	li	a1, 1
	bge	a1, a0, .LBB0_17
.LBB0_38:
	lw	a6, 436(sp)
.LBB0_39:
	lw	a0, 216(sp)
	li	a1, 3
	bge	a0, a1, .LBB0_113
# %bb.40:
	blez	a0, .LBB0_48
# %bb.41:
	addi	a1, a0, -1
	snez	a1, a1
	slli	a1, a1, 2
	addi	a4, sp, 144
	or	a1, a1, a4
	lw	a7, 84(a1)
	li	a1, 1
	blt	a1, a0, .LBB0_49
.LBB0_42:
	not	a0, a0
	srliw	a5, a0, 31
	j	.LBB0_50
.LBB0_43:
	seqz	a1, a0
	li	a2, 1
	bge	a2, a0, .LBB0_26
.LBB0_44:
	lw	s0, 436(sp)
.LBB0_45:
	lw	a5, 216(sp)
	li	a0, 3
	bge	a5, a0, .LBB0_116
# %bb.46:
	slli	s0, s0, 32
	slli	a1, a1, 32
	srli	a2, a1, 32
	blez	a5, .LBB0_71
# %bb.47:
	addi	a0, a5, -1
	snez	a0, a0
	slli	a0, a0, 2
	addi	a1, sp, 144
	or	a0, a0, a1
	lw	s1, 84(a0)
	j	.LBB0_72
.LBB0_48:
	seqz	a7, a0
	li	a1, 1
	bge	a1, a0, .LBB0_42
.LBB0_49:
	lw	a5, 228(sp)
.LBB0_50:
	ld	a0, 376(sp)
	ld	a1, 480(sp)
	ld	s9, 168(sp)
	ld	a4, 272(sp)
	slli	s1, a7, 32
	addiw	a3, a7, -1
	fcvt.s.w	fa5, a3
	lui	a3, 258048
	fmv.w.x	fa4, a3
	fnmsub.s	fa5, fa5, fa4, fs1
	#APP
	fcvt.w.s	t1, fa5, rdn
	#NO_APP
	sext.w	t1, t1
	snez	s0, s6
	slti	a3, t1, 0
	or	a3, a3, s0
	srli	t6, s1, 32
	bnez	a3, .LBB0_70
# %bb.51:
	addiw	s11, a5, -1
	fcvt.s.w	fa3, s11
	fnmsub.s	fa4, fa3, fa4, fs0
	#APP
	fcvt.w.s	s1, fa4, rdn
	#NO_APP
	sext.w	s1, s1
	addw	a3, t1, a7
	slt	a3, a3, a2
	xori	a3, a3, 1
	slti	s0, s1, 0
	or	a3, a3, s0
	bnez	a3, .LBB0_70
# %bb.52:
	addw	a3, s1, a5
	slt	t0, a3, a6
	sgtz	s0, a7
	sgtz	a3, a5
	and	a3, a3, s0
	and	a3, t0, a3
	beqz	a3, .LBB0_70
# %bb.53:
.Lpcrel_hi18:
	auipc	a2, %pcrel_hi(.LCPI0_0)
	flw	fa2, %pcrel_lo(.Lpcrel_hi18)(a2)
	fcvt.s.w	fa3, t1
	fsub.s	fa5, fa5, fa3
	fcvt.s.w	fa3, s1
	flt.s	a2, fa5, fa2
	fsub.s	fa1, fa4, fa3
	bnez	a2, .LBB0_55
# %bb.54:
	fmv.s	fa2, fa5
.LBB0_55:
	lui	a2, 260096
	fmv.w.x	fa5, a2
	fsub.s	fa3, fa5, fa2
	fsub.s	fa4, fa5, fa1
	fmul.s	fa5, fa4, fa3
	fmul.s	fa4, fa2, fa4
	fmul.s	fa3, fa1, fa3
	fmul.s	fa2, fa1, fa2
	slli	a3, s1, 32
	srli	a3, a3, 32
	mul	a2, a1, a3
	slli	a5, t1, 32
	srli	a5, a5, 32
	add	a2, a2, a5
	vsetvli	t0, zero, e32, m1, ta, ma
	subw	t1, a7, t0
	add	ra, a0, a2
	sd	s9, 40(sp)                      # 8-byte Folded Spill
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	bltz	t1, .LBB0_79
# %bb.56:
	li	a6, 0
	vfmv.v.f	v8, fa5
	vfmv.v.f	v9, fa4
	vfmv.v.f	v10, fa3
	vfmv.v.f	v11, fa2
	slli	s2, t0, 2
	andi	s0, a4, -4
	addi	a4, s2, 4
	sd	a4, 80(sp)                      # 8-byte Folded Spill
	not	a4, t0
	add	a4, a4, a7
	sd	a4, 24(sp)                      # 8-byte Folded Spill
	addi	a3, a3, 1
	mul	a3, a1, a3
	add	a5, a5, a3
	add	s4, a5, t0
	addi	a3, s4, 2
	sd	a3, 64(sp)                      # 8-byte Folded Spill
	add	s3, a2, t0
	addi	a3, s3, 2
	sd	a3, 48(sp)                      # 8-byte Folded Spill
	vsetvli	a3, zero, e32, m2, ta, ma
	vfmv.v.f	v12, fa5
	vfmv.v.f	v14, fa3
	vfmv.v.f	v16, fa2
	csrr	t2, vlenb
	srli	t5, t2, 1
	sd	s3, 56(sp)                      # 8-byte Folded Spill
	add	s3, s3, a0
	sd	s4, 72(sp)                      # 8-byte Folded Spill
	add	s4, s4, a0
	add	s5, s9, s2
	sd	t2, 8(sp)                       # 8-byte Folded Spill
	slli	t2, t2, 1
	add	a2, a2, a0
	addi	s7, a2, 1
	add	a0, a0, a5
	addi	s8, a0, 1
	mv	t4, s9
	sd	s0, 88(sp)                      # 8-byte Folded Spill
	j	.LBB0_58
.LBB0_57:                               #   in Loop: Header=BB0_58 Depth=1
	add	ra, ra, a1
	ld	s0, 88(sp)                      # 8-byte Folded Reload
	add	t4, t4, s0
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	addiw	s11, a0, -1
	addi	a6, a6, 1
	add	s3, s3, a1
	add	s4, s4, a1
	add	s5, s5, s0
	add	s7, s7, a1
	add	s8, s8, a1
	beqz	a0, .LBB0_76
.LBB0_58:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_59 Depth 2
                                        #     Child Loop BB0_66 Depth 2
                                        #     Child Loop BB0_69 Depth 2
	li	s10, 0
	li	s1, 0
	sd	s11, 136(sp)                    # 8-byte Folded Spill
	mul	a0, s0, a6
	add	s11, s2, a0
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	mul	a0, a1, a6
	ld	a2, 72(sp)                      # 8-byte Folded Reload
	add	a2, a2, a0
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	ld	a2, 64(sp)                      # 8-byte Folded Reload
	add	a2, a2, a0
	sd	a2, 112(sp)                     # 8-byte Folded Spill
	ld	a2, 56(sp)                      # 8-byte Folded Reload
	add	a2, a2, a0
	sd	a2, 120(sp)                     # 8-byte Folded Spill
	ld	a2, 48(sp)                      # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	mv	a4, s5
	mv	s0, s4
	mv	a5, s3
	mv	a2, t0
.LBB0_59:                               #   Parent Loop BB0_58 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a0, ra, s1
	vsetvli	a3, zero, e8, mf4, ta, ma
	vle8.v	v18, (a0)
	vwcvtu.x.x.v	v19, v18
	vsetvli	zero, zero, e16, mf2, ta, ma
	vwcvtu.x.x.v	v18, v19
	vsetvli	zero, zero, e32, m1, ta, ma
	addi	a3, a0, 1
	vle8.v	v19, (a3)
	vfcvt.f.x.v	v18, v18
	vsetvli	zero, zero, e8, mf4, ta, ma
	vwcvtu.x.x.v	v20, v19
	vsetvli	zero, zero, e16, mf2, ta, ma
	vwcvtu.x.x.v	v19, v20
	vsetvli	zero, zero, e32, m1, ta, ma
	add	a0, a0, a1
	vle8.v	v20, (a0)
	vfcvt.f.x.v	v19, v19
	vsetvli	zero, zero, e8, mf4, ta, ma
	vwcvtu.x.x.v	v21, v20
	vsetvli	zero, zero, e16, mf2, ta, ma
	vwcvtu.x.x.v	v20, v21
	vsetvli	zero, zero, e32, m1, ta, ma
	addi	a0, a0, 1
	vle8.v	v21, (a0)
	mv	t3, a2
	mv	a0, s10
	mv	t6, a5
	mv	s9, s0
	mv	s6, a4
	vfcvt.f.x.v	v20, v20
	vsetvli	zero, zero, e8, mf4, ta, ma
	vwcvtu.x.x.v	v22, v21
	vsetvli	zero, zero, e16, mf2, ta, ma
	vwcvtu.x.x.v	v21, v22
	vsetvli	zero, zero, e32, m1, ta, ma
	vfcvt.f.x.v	v21, v21
	vfmul.vv	v18, v18, v8
	vfmul.vv	v19, v19, v9
	vfadd.vv	v18, v18, v19
	vfmul.vv	v19, v20, v10
	vfadd.vv	v18, v18, v19
	vfmul.vv	v19, v21, v11
	vfadd.vv	v18, v18, v19
	slli	a2, s1, 2
	add	a2, a2, t4
	vse32.v	v18, (a2)
	add	s1, s1, t0
	add	a2, t3, t0
	addi	s10, s10, 1
	add	a5, a5, t0
	add	s0, s0, t0
	add	a4, a4, s2
	bge	t1, s1, .LBB0_59
# %bb.60:                               #   in Loop: Header=BB0_58 Depth=1
	sext.w	s1, s1
	bge	s1, a7, .LBB0_57
# %bb.61:                               #   in Loop: Header=BB0_58 Depth=1
	li	a2, 16
	mv	a3, t5
	bltu	a2, t5, .LBB0_63
# %bb.62:                               #   in Loop: Header=BB0_58 Depth=1
	li	a3, 16
.LBB0_63:                               #   in Loop: Header=BB0_58 Depth=1
	mul	a2, t0, a0
	ld	a5, 24(sp)                      # 8-byte Folded Reload
	subw	a2, a5, a2
	slli	a2, a2, 32
	srli	a2, a2, 32
	addi	a2, a2, 1
	bltu	a2, a3, .LBB0_68
# %bb.64:                               #   in Loop: Header=BB0_58 Depth=1
	sd	a2, 16(sp)                      # 8-byte Folded Spill
	mul	a3, s2, a0
	add	s11, s11, a3
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	add	s11, s11, a4
	ld	a2, 96(sp)                      # 8-byte Folded Reload
	add	a3, a3, a2
	add	a3, a3, a4
	mul	s10, t0, a0
	subw	a4, a5, s10
	slli	a4, a4, 32
	srli	a5, a4, 32
	srli	a4, a4, 30
	add	a3, a3, a4
	ld	a4, 104(sp)                     # 8-byte Folded Reload
	add	a4, a4, s10
	ld	a2, 32(sp)                      # 8-byte Folded Reload
	add	a4, a4, a2
	ld	s0, 112(sp)                     # 8-byte Folded Reload
	add	s0, s0, s10
	add	s0, s0, a2
	add	s0, s0, a5
	ld	s1, 120(sp)                     # 8-byte Folded Reload
	add	s1, s1, s10
	add	s1, s1, a2
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	add	a0, a0, s10
	add	a0, a0, a2
	add	a0, a0, a5
	sltu	a5, s11, s0
	sltu	a4, a4, a3
	and	a4, a4, a5
	sltu	a0, s11, a0
	sltu	a3, s1, a3
	and	a0, a0, a3
	or	a0, a0, a4
	bnez	a0, .LBB0_68
# %bb.65:                               #   in Loop: Header=BB0_58 Depth=1
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	srli	a0, a0, 1
	neg	a0, a0
	ld	a2, 16(sp)                      # 8-byte Folded Reload
	and	a0, a0, a2
	add	t3, t3, a0
	mv	a3, a0
.LBB0_66:                               #   Parent Loop BB0_58 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vsetvli	a4, zero, e16, m1, ta, ma
	vle8.v	v18, (t6)
	addi	a4, t6, 1
	vle8.v	v19, (a4)
	vzext.vf2	v20, v18
	vfwcvt.f.xu.v	v22, v20
	vzext.vf2	v18, v19
	vfwcvt.f.xu.v	v20, v18
	vsetvli	zero, zero, e32, m2, ta, ma
	vle8.v	v18, (s9)
	vfmul.vf	v20, v20, fa4
	vfmacc.vv	v20, v12, v22
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	vfwcvt.f.xu.v	v22, v19
	vsetvli	zero, zero, e32, m2, ta, ma
	addi	a4, s9, 1
	vle8.v	v18, (a4)
	vfmadd.vv	v22, v14, v20
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	vfwcvt.f.xu.v	v20, v19
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v20, v16, v22
	vs2r.v	v20, (s6)
	sub	a3, a3, t5
	add	t6, t6, t5
	add	s9, s9, t5
	add	s6, s6, t2
	bnez	a3, .LBB0_66
# %bb.67:                               #   in Loop: Header=BB0_58 Depth=1
	beq	a2, a0, .LBB0_57
.LBB0_68:                               #   in Loop: Header=BB0_58 Depth=1
	slli	a0, t3, 2
	add	a0, a0, t4
	add	a2, s7, t3
	add	a3, s8, t3
.LBB0_69:                               #   Parent Loop BB0_58 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	lbu	a4, -1(a2)
	lbu	a5, 0(a2)
	fcvt.s.wu	fa1, a4
	fcvt.s.wu	fa0, a5
	lbu	a4, -1(a3)
	fmul.s	fa0, fa4, fa0
	lbu	a5, 0(a3)
	fmadd.s	fa1, fa5, fa1, fa0
	fcvt.s.wu	fa0, a4
	fmadd.s	fa1, fa3, fa0, fa1
	fcvt.s.wu	fa0, a5
	fmadd.s	fa1, fa2, fa0, fa1
	fsw	fa1, 0(a0)
	addi	a0, a0, 4
	addi	a2, a2, 1
	addiw	t3, t3, 1
	addi	a3, a3, 1
	blt	t3, a7, .LBB0_69
	j	.LBB0_57
.LBB0_70:
.Ltmp48:
	slli	a2, a2, 32
	slli	a6, a6, 32
	srli	a2, a2, 32
	or	a2, a6, a2
	slli	a5, a5, 32
	or	a5, a5, t6
	mv	a3, s9
	fmv.s	fa0, fs1
	fmv.s	fa1, fs0
	mv	a6, s2
	call	_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
.Ltmp49:
	j	.LBB0_76
.LBB0_71:
	seqz	s1, a5
.LBB0_72:
	ld	a0, 376(sp)
	ld	a1, 480(sp)
	ld	a3, 168(sp)
	ld	a4, 272(sp)
	li	a6, 1
	or	a2, a2, s0
	blt	a6, a5, .LBB0_74
# %bb.73:
	not	a5, a5
	srliw	a5, a5, 31
	j	.LBB0_75
.LBB0_74:
	lw	a5, 228(sp)
.LBB0_75:
	slli	a5, a5, 32
	slli	s1, s1, 32
	srli	s1, s1, 32
	or	a5, a5, s1
.Ltmp36:
	fmv.s	fa0, fs1
	fmv.s	fa1, fs0
	mv	a6, s2
	call	_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
.Ltmp37:
.LBB0_76:
	addi	a0, sp, 144
	call	_ZN2cv3MatD1Ev@plt
	addi	a0, sp, 352
	call	_ZN2cv3MatD1Ev@plt
	lw	a0, 568(sp)
	beqz	a0, .LBB0_78
.LBB0_77:
.Ltmp66:
	addi	a0, sp, 560
	call	_ZN2cv5utils5trace7details6Region7destroyEv@plt
.Ltmp67:
.LBB0_78:
	ld	ra, 728(sp)                     # 8-byte Folded Reload
	ld	s0, 720(sp)                     # 8-byte Folded Reload
	ld	s1, 712(sp)                     # 8-byte Folded Reload
	ld	s2, 704(sp)                     # 8-byte Folded Reload
	ld	s3, 696(sp)                     # 8-byte Folded Reload
	ld	s4, 688(sp)                     # 8-byte Folded Reload
	ld	s5, 680(sp)                     # 8-byte Folded Reload
	ld	s6, 672(sp)                     # 8-byte Folded Reload
	ld	s7, 664(sp)                     # 8-byte Folded Reload
	ld	s8, 656(sp)                     # 8-byte Folded Reload
	ld	s9, 648(sp)                     # 8-byte Folded Reload
	ld	s10, 640(sp)                    # 8-byte Folded Reload
	ld	s11, 632(sp)                    # 8-byte Folded Reload
	fld	fs0, 624(sp)                    # 8-byte Folded Reload
	fld	fs1, 616(sp)                    # 8-byte Folded Reload
	addi	sp, sp, 736
	ret
.LBB0_79:
	srli	t5, a4, 2
	addi	a3, a3, 1
	mul	a6, a1, a3
	add	a6, a6, a5
	add	t2, a6, t6
	csrr	a7, vlenb
	srli	t4, a7, 1
	li	a3, 8
	add	t3, a2, t6
	bltu	a3, t4, .LBB0_81
# %bb.80:
	li	t4, 8
.LBB0_81:
	li	s2, 0
	slli	t0, t6, 2
	andi	t1, a4, -4
	addi	t2, t2, 1
	addi	t3, t3, 1
	vsetvli	a3, zero, e32, m2, ta, ma
	vfmv.v.f	v8, fa5
	vfmv.v.f	v10, fa3
	vfmv.v.f	v12, fa2
	add	s1, a0, a6
	slli	t5, t5, 2
	slli	s6, a7, 1
	add	a2, a2, a0
	addi	s4, a2, 1
	addi	s5, s1, 1
	j	.LBB0_83
.LBB0_82:                               #   in Loop: Header=BB0_83 Depth=1
	add	ra, ra, a1
	add	s9, s9, t5
	addiw	s11, s3, -1
	addi	s2, s2, 1
	add	s1, s1, a1
	add	s4, s4, a1
	add	s5, s5, a1
	beqz	s3, .LBB0_76
.LBB0_83:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_89 Depth 2
                                        #     Child Loop BB0_87 Depth 2
	mv	s3, s11
	bltu	t6, t4, .LBB0_85
# %bb.84:                               #   in Loop: Header=BB0_83 Depth=1
	mul	a0, a1, s2
	add	a2, t3, a0
	ld	a4, 32(sp)                      # 8-byte Folded Reload
	add	a2, a2, a4
	add	a3, t2, a0
	add	a3, a3, a4
	add	a0, a0, a6
	add	a0, a0, a4
	mul	a4, t1, s2
	add	a4, a4, t0
	ld	a5, 40(sp)                      # 8-byte Folded Reload
	add	a4, a4, a5
	sltu	a3, s9, a3
	sltu	a0, a0, a4
	and	a0, a0, a3
	sltu	a2, s9, a2
	sltu	a3, ra, a4
	and	a2, a2, a3
	or	a0, a0, a2
	beqz	a0, .LBB0_88
.LBB0_85:                               #   in Loop: Header=BB0_83 Depth=1
	li	a0, 0
.LBB0_86:                               #   in Loop: Header=BB0_83 Depth=1
	slli	a2, a0, 2
	add	a2, a2, s9
	add	a3, s4, a0
	sub	a4, t6, a0
	add	a0, a0, s5
.LBB0_87:                               #   Parent Loop BB0_83 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	lbu	a5, -1(a3)
	lbu	s0, 0(a3)
	fcvt.s.wu	fa1, a5
	fcvt.s.wu	fa0, s0
	lbu	a5, -1(a0)
	fmul.s	fa0, fa4, fa0
	lbu	s0, 0(a0)
	fmadd.s	fa1, fa5, fa1, fa0
	fcvt.s.wu	fa0, a5
	fmadd.s	fa1, fa3, fa0, fa1
	fcvt.s.wu	fa0, s0
	fmadd.s	fa1, fa2, fa0, fa1
	fsw	fa1, 0(a2)
	addi	a2, a2, 4
	addi	a3, a3, 1
	addi	a4, a4, -1
	addi	a0, a0, 1
	bnez	a4, .LBB0_87
	j	.LBB0_82
.LBB0_88:                               #   in Loop: Header=BB0_83 Depth=1
	li	a3, 0
	srli	a0, a7, 3
	slli	a0, a0, 2
	neg	a0, a0
	and	a0, a0, t6
	srli	a4, a7, 1
	mv	a2, s9
.LBB0_89:                               #   Parent Loop BB0_83 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a5, ra, a3
	vle8.v	v14, (a5)
	vsetvli	zero, zero, e16, m1, ta, ma
	addi	a5, a5, 1
	vle8.v	v15, (a5)
	vzext.vf2	v16, v14
	vfwcvt.f.xu.v	v18, v16
	vzext.vf2	v14, v15
	vfwcvt.f.xu.v	v16, v14
	vsetvli	zero, zero, e32, m2, ta, ma
	add	a5, s1, a3
	vle8.v	v14, (a5)
	vfmul.vf	v16, v16, fa4
	vfmacc.vv	v16, v8, v18
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vfwcvt.f.xu.v	v18, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	addi	a5, a5, 1
	vle8.v	v14, (a5)
	vfmadd.vv	v18, v10, v16
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vfwcvt.f.xu.v	v16, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v16, v12, v18
	vs2r.v	v16, (a2)
	add	a3, a3, a4
	add	a2, a2, s6
	bne	a0, a3, .LBB0_89
# %bb.90:                               #   in Loop: Header=BB0_83 Depth=1
	beq	a0, t6, .LBB0_82
	j	.LBB0_86
.LBB0_91:
.Lpcrel_hi1:
	auipc	a0, %pcrel_hi(_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__326)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi1)
	j	.LBB0_93
.LBB0_92:
.Lpcrel_hi2:
	auipc	a0, %pcrel_hi(_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__327)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi2)
.LBB0_93:
.Ltmp0:
	mv	a0, s0
	li	a1, 0
	call	_ZN2cv6detail17check_failed_autoEiiRKNS0_12CheckContextE@plt
.Ltmp1:
# %bb.94:
.LBB0_95:
.Ltmp8:
.Lpcrel_hi3:
	auipc	a0, %pcrel_hi(.L.str.6)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi3)
	addi	a0, sp, 144
	addi	a2, sp, 584
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp9:
# %bb.96:
.Ltmp11:
.Lpcrel_hi4:
	auipc	a0, %pcrel_hi(.L__func__._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi4)
.Lpcrel_hi5:
	auipc	a0, %pcrel_hi(.L.str.1)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi5)
	li	a0, -215
	addi	a1, sp, 144
	li	a4, 339
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp12:
# %bb.97:
.LBB0_98:
.Ltmp50:
.Lpcrel_hi6:
	auipc	a0, %pcrel_hi(.L.str.9)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi6)
	addi	a0, sp, 584
	addi	a2, sp, 583
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp51:
# %bb.99:
.Ltmp52:
.Lpcrel_hi7:
	auipc	a0, %pcrel_hi(.L__func__._ZNK2cv8MatShapeclEv)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi7)
.Lpcrel_hi8:
	auipc	a0, %pcrel_hi(.L.str.10)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi8)
	li	a0, -215
	addi	a1, sp, 584
	li	a4, 109
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp53:
# %bb.100:
.LBB0_101:
.Ltmp55:
.Lpcrel_hi9:
	auipc	a0, %pcrel_hi(.L.str.9)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi9)
	addi	a0, sp, 584
	addi	a2, sp, 583
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp56:
# %bb.102:
.Ltmp57:
.Lpcrel_hi10:
	auipc	a0, %pcrel_hi(.L__func__._ZNK2cv8MatShapeclEv)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi10)
.Lpcrel_hi11:
	auipc	a0, %pcrel_hi(.L.str.10)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi11)
	li	a0, -215
	addi	a1, sp, 584
	li	a4, 109
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp58:
# %bb.103:
.LBB0_104:
.Ltmp38:
.Lpcrel_hi12:
	auipc	a0, %pcrel_hi(.L.str.9)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi12)
	addi	a0, sp, 584
	addi	a2, sp, 583
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp39:
# %bb.105:
.Ltmp40:
.Lpcrel_hi13:
	auipc	a0, %pcrel_hi(.L__func__._ZNK2cv8MatShapeclEv)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi13)
.Lpcrel_hi14:
	auipc	a0, %pcrel_hi(.L.str.10)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi14)
	li	a0, -215
	addi	a1, sp, 584
	li	a4, 109
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp41:
# %bb.106:
.LBB0_107:
.Ltmp20:
.Lpcrel_hi25:
	auipc	a0, %pcrel_hi(.L.str.8)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi25)
	addi	a0, sp, 584
	addi	a2, sp, 583
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp21:
# %bb.108:
.Ltmp23:
.Lpcrel_hi26:
	auipc	a0, %pcrel_hi(.L__func__._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi26)
.Lpcrel_hi27:
	auipc	a0, %pcrel_hi(.L.str.1)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi27)
	li	a0, -210
	addi	a1, sp, 584
	li	a4, 359
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp24:
# %bb.109:
.LBB0_110:
.Ltmp26:
.Lpcrel_hi19:
	auipc	a0, %pcrel_hi(.L.str.9)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi19)
	addi	a0, sp, 584
	addi	a2, sp, 583
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp27:
# %bb.111:
.Ltmp28:
.Lpcrel_hi20:
	auipc	a0, %pcrel_hi(.L__func__._ZNK2cv8MatShapeclEv)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi20)
.Lpcrel_hi21:
	auipc	a0, %pcrel_hi(.L.str.10)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi21)
	li	a0, -215
	addi	a1, sp, 584
	li	a4, 109
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp29:
# %bb.112:
.LBB0_113:
.Ltmp43:
.Lpcrel_hi15:
	auipc	a0, %pcrel_hi(.L.str.9)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi15)
	addi	a0, sp, 584
	addi	a2, sp, 583
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp44:
# %bb.114:
.Ltmp45:
.Lpcrel_hi16:
	auipc	a0, %pcrel_hi(.L__func__._ZNK2cv8MatShapeclEv)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi16)
.Lpcrel_hi17:
	auipc	a0, %pcrel_hi(.L.str.10)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi17)
	li	a0, -215
	addi	a1, sp, 584
	li	a4, 109
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp46:
# %bb.115:
.LBB0_116:
.Ltmp31:
.Lpcrel_hi22:
	auipc	a0, %pcrel_hi(.L.str.9)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi22)
	addi	a0, sp, 584
	addi	a2, sp, 583
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp32:
# %bb.117:
.Ltmp33:
.Lpcrel_hi23:
	auipc	a0, %pcrel_hi(.L__func__._ZNK2cv8MatShapeclEv)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi23)
.Lpcrel_hi24:
	auipc	a0, %pcrel_hi(.L.str.10)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi24)
	li	a0, -215
	addi	a1, sp, 584
	li	a4, 109
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp34:
# %bb.118:
.LBB0_119:
.Ltmp35:
	j	.LBB0_127
.LBB0_120:
.Ltmp47:
	j	.LBB0_127
.LBB0_121:
.Ltmp30:
	j	.LBB0_127
.LBB0_122:
.Ltmp25:
	j	.LBB0_127
.LBB0_123:
.Ltmp42:
	j	.LBB0_127
.LBB0_124:
.Ltmp22:
	j	.LBB0_136
.LBB0_125:
.Ltmp59:
	j	.LBB0_127
.LBB0_126:
.Ltmp54:
.LBB0_127:
	ld	a1, 584(sp)
	addi	a2, sp, 600
	mv	s0, a0
	beq	a1, a2, .LBB0_137
# %bb.128:
	mv	a0, a1
	call	_ZdlPv@plt
	j	.LBB0_137
.LBB0_129:
.Ltmp13:
	ld	a1, 144(sp)
	addi	a2, sp, 160
	mv	s0, a0
	beq	a1, a2, .LBB0_141
# %bb.130:
	mv	a0, a1
	call	_ZdlPv@plt
	j	.LBB0_141
.LBB0_131:
.Ltmp10:
	j	.LBB0_140
.LBB0_132:
.Ltmp19:
	j	.LBB0_140
.LBB0_133:
.Ltmp16:
	j	.LBB0_140
.LBB0_134:
.Ltmp68:
	call	__clang_call_terminate
.LBB0_135:
.Ltmp62:
.LBB0_136:
	mv	s0, a0
.LBB0_137:
	addi	a0, sp, 144
	call	_ZN2cv3MatD1Ev@plt
	j	.LBB0_141
.LBB0_138:
.Ltmp5:
	j	.LBB0_143
.LBB0_139:
.Ltmp65:
.LBB0_140:
	mv	s0, a0
.LBB0_141:
	addi	a0, sp, 352
	call	_ZN2cv3MatD1Ev@plt
	addi	a0, sp, 560
	call	_ZN2cv5utils5trace7details6RegionD2Ev
	mv	a0, s0
	call	_Unwind_Resume@plt
.LBB0_142:
.Ltmp2:
.LBB0_143:
	mv	s0, a0
	addi	a0, sp, 560
	call	_ZN2cv5utils5trace7details6RegionD2Ev
	mv	a0, s0
	call	_Unwind_Resume@plt
.Lfunc_end0:
	.size	_ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi, .Lfunc_end0-_ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi
	.cfi_endproc
	.section	.gcc_except_table._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	3                               # Call site Encoding = udata4
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.word	.Lfunc_begin0-.Lfunc_begin0     # >> Call Site 1 <<
	.word	.Ltmp3-.Lfunc_begin0            #   Call between .Lfunc_begin0 and .Ltmp3
	.word	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.word	.Ltmp3-.Lfunc_begin0            # >> Call Site 2 <<
	.word	.Ltmp4-.Ltmp3                   #   Call between .Ltmp3 and .Ltmp4
	.word	.Ltmp5-.Lfunc_begin0            #     jumps to .Ltmp5
	.byte	0                               #   On action: cleanup
	.word	.Ltmp6-.Lfunc_begin0            # >> Call Site 3 <<
	.word	.Ltmp64-.Ltmp6                  #   Call between .Ltmp6 and .Ltmp64
	.word	.Ltmp65-.Lfunc_begin0           #     jumps to .Ltmp65
	.byte	0                               #   On action: cleanup
	.word	.Ltmp14-.Lfunc_begin0           # >> Call Site 4 <<
	.word	.Ltmp15-.Ltmp14                 #   Call between .Ltmp14 and .Ltmp15
	.word	.Ltmp16-.Lfunc_begin0           #     jumps to .Ltmp16
	.byte	0                               #   On action: cleanup
	.word	.Ltmp17-.Lfunc_begin0           # >> Call Site 5 <<
	.word	.Ltmp18-.Ltmp17                 #   Call between .Ltmp17 and .Ltmp18
	.word	.Ltmp19-.Lfunc_begin0           #     jumps to .Ltmp19
	.byte	0                               #   On action: cleanup
	.word	.Ltmp60-.Lfunc_begin0           # >> Call Site 6 <<
	.word	.Ltmp37-.Ltmp60                 #   Call between .Ltmp60 and .Ltmp37
	.word	.Ltmp62-.Lfunc_begin0           #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.word	.Ltmp66-.Lfunc_begin0           # >> Call Site 7 <<
	.word	.Ltmp67-.Ltmp66                 #   Call between .Ltmp66 and .Ltmp67
	.word	.Ltmp68-.Lfunc_begin0           #     jumps to .Ltmp68
	.byte	1                               #   On action: 1
	.word	.Ltmp0-.Lfunc_begin0            # >> Call Site 8 <<
	.word	.Ltmp1-.Ltmp0                   #   Call between .Ltmp0 and .Ltmp1
	.word	.Ltmp2-.Lfunc_begin0            #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.word	.Ltmp8-.Lfunc_begin0            # >> Call Site 9 <<
	.word	.Ltmp9-.Ltmp8                   #   Call between .Ltmp8 and .Ltmp9
	.word	.Ltmp10-.Lfunc_begin0           #     jumps to .Ltmp10
	.byte	0                               #   On action: cleanup
	.word	.Ltmp11-.Lfunc_begin0           # >> Call Site 10 <<
	.word	.Ltmp12-.Ltmp11                 #   Call between .Ltmp11 and .Ltmp12
	.word	.Ltmp13-.Lfunc_begin0           #     jumps to .Ltmp13
	.byte	0                               #   On action: cleanup
	.word	.Ltmp50-.Lfunc_begin0           # >> Call Site 11 <<
	.word	.Ltmp51-.Ltmp50                 #   Call between .Ltmp50 and .Ltmp51
	.word	.Ltmp62-.Lfunc_begin0           #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.word	.Ltmp52-.Lfunc_begin0           # >> Call Site 12 <<
	.word	.Ltmp53-.Ltmp52                 #   Call between .Ltmp52 and .Ltmp53
	.word	.Ltmp54-.Lfunc_begin0           #     jumps to .Ltmp54
	.byte	0                               #   On action: cleanup
	.word	.Ltmp55-.Lfunc_begin0           # >> Call Site 13 <<
	.word	.Ltmp56-.Ltmp55                 #   Call between .Ltmp55 and .Ltmp56
	.word	.Ltmp62-.Lfunc_begin0           #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.word	.Ltmp57-.Lfunc_begin0           # >> Call Site 14 <<
	.word	.Ltmp58-.Ltmp57                 #   Call between .Ltmp57 and .Ltmp58
	.word	.Ltmp59-.Lfunc_begin0           #     jumps to .Ltmp59
	.byte	0                               #   On action: cleanup
	.word	.Ltmp38-.Lfunc_begin0           # >> Call Site 15 <<
	.word	.Ltmp39-.Ltmp38                 #   Call between .Ltmp38 and .Ltmp39
	.word	.Ltmp62-.Lfunc_begin0           #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.word	.Ltmp40-.Lfunc_begin0           # >> Call Site 16 <<
	.word	.Ltmp41-.Ltmp40                 #   Call between .Ltmp40 and .Ltmp41
	.word	.Ltmp42-.Lfunc_begin0           #     jumps to .Ltmp42
	.byte	0                               #   On action: cleanup
	.word	.Ltmp20-.Lfunc_begin0           # >> Call Site 17 <<
	.word	.Ltmp21-.Ltmp20                 #   Call between .Ltmp20 and .Ltmp21
	.word	.Ltmp22-.Lfunc_begin0           #     jumps to .Ltmp22
	.byte	0                               #   On action: cleanup
	.word	.Ltmp23-.Lfunc_begin0           # >> Call Site 18 <<
	.word	.Ltmp24-.Ltmp23                 #   Call between .Ltmp23 and .Ltmp24
	.word	.Ltmp25-.Lfunc_begin0           #     jumps to .Ltmp25
	.byte	0                               #   On action: cleanup
	.word	.Ltmp26-.Lfunc_begin0           # >> Call Site 19 <<
	.word	.Ltmp27-.Ltmp26                 #   Call between .Ltmp26 and .Ltmp27
	.word	.Ltmp62-.Lfunc_begin0           #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.word	.Ltmp28-.Lfunc_begin0           # >> Call Site 20 <<
	.word	.Ltmp29-.Ltmp28                 #   Call between .Ltmp28 and .Ltmp29
	.word	.Ltmp30-.Lfunc_begin0           #     jumps to .Ltmp30
	.byte	0                               #   On action: cleanup
	.word	.Ltmp43-.Lfunc_begin0           # >> Call Site 21 <<
	.word	.Ltmp44-.Ltmp43                 #   Call between .Ltmp43 and .Ltmp44
	.word	.Ltmp62-.Lfunc_begin0           #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.word	.Ltmp45-.Lfunc_begin0           # >> Call Site 22 <<
	.word	.Ltmp46-.Ltmp45                 #   Call between .Ltmp45 and .Ltmp46
	.word	.Ltmp47-.Lfunc_begin0           #     jumps to .Ltmp47
	.byte	0                               #   On action: cleanup
	.word	.Ltmp31-.Lfunc_begin0           # >> Call Site 23 <<
	.word	.Ltmp32-.Ltmp31                 #   Call between .Ltmp31 and .Ltmp32
	.word	.Ltmp62-.Lfunc_begin0           #     jumps to .Ltmp62
	.byte	0                               #   On action: cleanup
	.word	.Ltmp33-.Lfunc_begin0           # >> Call Site 24 <<
	.word	.Ltmp34-.Ltmp33                 #   Call between .Ltmp33 and .Ltmp34
	.word	.Ltmp35-.Lfunc_begin0           #     jumps to .Ltmp35
	.byte	0                               #   On action: cleanup
	.word	.Ltmp34-.Lfunc_begin0           # >> Call Site 25 <<
	.word	.Lfunc_end0-.Ltmp34             #   Call between .Ltmp34 and .Lfunc_end0
	.word	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.word	0                               # TypeInfo 1
.Lttbase0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,comdat
	.hidden	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_ # -- Begin function _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.p2align	1
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,@function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_: # @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -48
	.cfi_def_cfa_offset 48
	sd	ra, 40(sp)                      # 8-byte Folded Spill
	sd	s0, 32(sp)                      # 8-byte Folded Spill
	sd	s1, 24(sp)                      # 8-byte Folded Spill
	sd	s2, 16(sp)                      # 8-byte Folded Spill
	sd	s3, 8(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	addi	s3, a0, 16
	sd	s3, 0(a0)
	beqz	a1, .LBB1_8
# %bb.1:
	mv	s2, a1
	mv	s0, a0
	mv	a0, a1
	call	strlen@plt
	mv	s1, a0
	li	a0, 16
	sd	s1, 0(sp)
	bltu	s1, a0, .LBB1_3
# %bb.2:
	mv	a1, sp
	mv	a0, s0
	li	a2, 0
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt
	mv	s3, a0
	sd	a0, 0(s0)
	ld	a0, 0(sp)
	sd	a0, 16(s0)
.LBB1_3:
	beqz	s1, .LBB1_7
# %bb.4:
	li	a0, 1
	bne	s1, a0, .LBB1_6
# %bb.5:
	lbu	a0, 0(s2)
	sb	a0, 0(s3)
	j	.LBB1_7
.LBB1_6:
	mv	a0, s3
	mv	a1, s2
	mv	a2, s1
	call	memcpy@plt
.LBB1_7:
	ld	a0, 0(sp)
	ld	a1, 0(s0)
	sd	a0, 8(s0)
	add	a0, a0, a1
	sb	zero, 0(a0)
	ld	ra, 40(sp)                      # 8-byte Folded Reload
	ld	s0, 32(sp)                      # 8-byte Folded Reload
	ld	s1, 24(sp)                      # 8-byte Folded Reload
	ld	s2, 16(sp)                      # 8-byte Folded Reload
	ld	s3, 8(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 48
	ret
.LBB1_8:
.Lpcrel_hi28:
	auipc	a0, %pcrel_hi(.L.str.12)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi28)
	call	_ZSt19__throw_logic_errorPKc@plt
.Lfunc_end1:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_, .Lfunc_end1-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,"axG",@progbits,_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,comdat
	.hidden	_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi # -- Begin function _ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.weak	_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.p2align	1
	.type	_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,@function
_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi: # @_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -352
	.cfi_def_cfa_offset 352
	sd	ra, 344(sp)                     # 8-byte Folded Spill
	sd	s0, 336(sp)                     # 8-byte Folded Spill
	sd	s1, 328(sp)                     # 8-byte Folded Spill
	sd	s2, 320(sp)                     # 8-byte Folded Spill
	sd	s3, 312(sp)                     # 8-byte Folded Spill
	sd	s4, 304(sp)                     # 8-byte Folded Spill
	sd	s5, 296(sp)                     # 8-byte Folded Spill
	sd	s6, 288(sp)                     # 8-byte Folded Spill
	sd	s7, 280(sp)                     # 8-byte Folded Spill
	sd	s8, 272(sp)                     # 8-byte Folded Spill
	sd	s9, 264(sp)                     # 8-byte Folded Spill
	sd	s10, 256(sp)                    # 8-byte Folded Spill
	sd	s11, 248(sp)                    # 8-byte Folded Spill
	fsd	fs0, 240(sp)                    # 8-byte Folded Spill
	fsd	fs1, 232(sp)                    # 8-byte Folded Spill
	fsd	fs2, 224(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	.cfi_offset fs0, -112
	.cfi_offset fs1, -120
	.cfi_offset fs2, -128
	csrr	a7, vlenb
	li	t0, 24
	mul	a7, a7, t0
	sub	sp, sp, a7
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xe0, 0x02, 0x22, 0x11, 0x18, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 352 + 24 * vlenb
	mv	s0, a6
	mv	s4, a5
	mv	s6, a4
	sd	a3, 200(sp)                     # 8-byte Folded Spill
	mv	a7, a2
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	srli	a2, a5, 32
	addiw	a0, a5, -1
	fcvt.s.w	fa5, a0
	lui	a0, 258048
	fmv.w.x	fa4, a0
	fnmsub.s	fa5, fa5, fa4, fa0
	addiw	a0, a2, -1
	fcvt.s.w	fa3, a0
	fnmsub.s	fa4, fa3, fa4, fa1
	#APP
	fcvt.w.s	a4, fa5, rdn
	#NO_APP
	sext.w	a0, a4
	#APP
	fcvt.w.s	a5, fa4, rdn
	#NO_APP
	sext.w	a3, a5
	fcvt.s.w	fa3, a4
	fsub.s	fa5, fa5, fa3
	fcvt.s.w	fa3, a5
	fsub.s	fs0, fa4, fa3
	lui	a4, 260096
	fmv.w.x	fa4, a4
	fsub.s	fa3, fa4, fa5
	fsub.s	fs1, fa4, fs0
	fmul.s	fa4, fa3, fs1
	lui	a4, 292864
	fmv.w.x	fs2, a4
	fmul.s	fa4, fa4, fs2
	#APP
	fcvt.w.s	a4, fa4, rne
	#NO_APP
	sd	a4, 120(sp)                     # 8-byte Folded Spill
	fmul.s	fa4, fa5, fs1
	fmul.s	fa4, fa4, fs2
	#APP
	fcvt.w.s	s7, fa4, rne
	#NO_APP
	fmul.s	fa4, fa3, fs0
	fmul.s	fa4, fa4, fs2
	#APP
	fcvt.w.s	s8, fa4, rne
	#NO_APP
	fmul.s	fa5, fa5, fs0
	fmul.s	fa5, fa5, fs2
	#APP
	fcvt.w.s	s11, fa5, rne
	#NO_APP
	sext.w	a4, a2
	sd	a4, 184(sp)                     # 8-byte Folded Spill
	sd	a1, 144(sp)                     # 8-byte Folded Spill
	bltz	a0, .LBB2_28
# %bb.1:
	srli	a4, a7, 32
	subw	a5, a7, s4
	slt	a5, a0, a5
	slti	s1, a3, 0
	not	s1, s1
	subw	a4, a4, a2
	slt	a2, a3, a4
	and	a2, a2, a5
	and	a2, a2, s1
	beqz	a2, .LBB2_28
# %bb.2:
	ld	t2, 200(sp)                     # 8-byte Folded Reload
	ld	a2, 184(sp)                     # 8-byte Folded Reload
	blez	a2, .LBB2_75
# %bb.3:
	mulw	a4, s4, s0
	slli	a3, a3, 32
	srli	a3, a3, 32
	mul	a2, a3, a1
	mulw	a5, a0, s0
	add	a7, a2, a5
	ld	a2, 168(sp)                     # 8-byte Folded Reload
	add	a0, a2, a7
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	addiw	t1, a4, -2
	slli	a6, a4, 32
	addi	a3, a3, 1
	mul	s1, a3, a1
	add	a0, s1, a5
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	add	a0, a0, a2
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	mv	a0, t1
	bgtz	t1, .LBB2_5
# %bb.4:
	li	a0, 1
.LBB2_5:
	li	t4, 0
	li	t0, 0
	srli	s4, a6, 32
	ld	t3, 128(sp)                     # 8-byte Folded Reload
	add	a2, t3, s0
	sd	a2, 112(sp)                     # 8-byte Folded Spill
	ld	s2, 152(sp)                     # 8-byte Folded Reload
	add	a2, s2, s0
	sd	a2, 104(sp)                     # 8-byte Folded Spill
	li	a2, -63
	srli	a2, a2, 1
	addi	a2, a2, 30
	and	a2, a2, a0
	addi	a3, a2, 2
	add	s1, s1, s0
	add	a5, a5, s1
	sd	a5, 88(sp)                      # 8-byte Folded Spill
	ld	a2, 136(sp)                     # 8-byte Folded Reload
	add	a5, a3, a2
	sd	a5, 80(sp)                      # 8-byte Folded Spill
	add	a5, a5, s0
	sd	a5, 72(sp)                      # 8-byte Folded Spill
	add	s9, a7, s0
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	add	a3, a3, a7
	sd	a3, 56(sp)                      # 8-byte Folded Spill
	add	a3, a3, s0
	sd	a3, 48(sp)                      # 8-byte Folded Spill
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	srli	a0, a0, 1
	addi	a0, a0, 1
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	andi	a0, a0, -32
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	slli	a0, a0, 1
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	ld	a6, 168(sp)                     # 8-byte Folded Reload
	sd	s9, 64(sp)                      # 8-byte Folded Spill
	add	s9, s9, a6
	add	s0, s0, a2
	add	a6, a6, s0
	addi	s3, t2, 1
	lui	ra, 8
	mv	s5, t2
	sd	a4, 192(sp)                     # 8-byte Folded Spill
	j	.LBB2_7
.LBB2_6:                                #   in Loop: Header=BB2_7 Depth=1
	addiw	t0, t0, 1
	add	s2, s2, a1
	add	s5, s5, s6
	addi	t4, t4, 1
	add	s9, s9, a1
	add	a6, a6, a1
	add	t3, t3, a1
	add	s3, s3, s6
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	beq	t0, a0, .LBB2_75
.LBB2_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_23 Depth 2
                                        #     Child Loop BB2_14 Depth 2
                                        #     Child Loop BB2_26 Depth 2
                                        #     Child Loop BB2_21 Depth 2
	mul	s10, t4, s6
	add	s10, s10, t2
	mul	t5, t4, a1
	li	a0, 2
	bge	a4, a0, .LBB2_9
# %bb.8:                                #   in Loop: Header=BB2_7 Depth=1
	li	a3, 0
	ld	s0, 120(sp)                     # 8-byte Folded Reload
	li	a2, 0
	bge	a2, a4, .LBB2_6
	j	.LBB2_16
.LBB2_9:                                #   in Loop: Header=BB2_7 Depth=1
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	li	a2, 62
	sd	s10, 176(sp)                    # 8-byte Folded Spill
	bgeu	a0, a2, .LBB2_11
# %bb.10:                               #   in Loop: Header=BB2_7 Depth=1
	li	a3, 0
	j	.LBB2_13
.LBB2_11:                               #   in Loop: Header=BB2_7 Depth=1
	ld	a1, 96(sp)                      # 8-byte Folded Reload
	add	a1, a1, s10
	ld	a2, 88(sp)                      # 8-byte Folded Reload
	add	a2, a2, t5
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	add	t2, a0, a2
	ld	a3, 72(sp)                      # 8-byte Folded Reload
	add	a3, a3, t5
	add	a3, a3, a0
	ld	a4, 136(sp)                     # 8-byte Folded Reload
	add	a4, a4, t5
	add	t6, a0, a4
	ld	a5, 80(sp)                      # 8-byte Folded Reload
	add	a5, a5, t5
	add	a5, a5, a0
	ld	s1, 64(sp)                      # 8-byte Folded Reload
	add	s1, s1, t5
	add	a7, a0, s1
	ld	a2, 48(sp)                      # 8-byte Folded Reload
	add	a2, a2, t5
	add	a2, a2, a0
	ld	a4, 56(sp)                      # 8-byte Folded Reload
	add	a4, a4, t5
	add	a4, a4, a0
	sltu	a3, s5, a3
	sltu	s1, t2, a1
	and	a3, a3, s1
	sltu	a5, s5, a5
	sltu	s1, t6, a1
	and	a5, a5, s1
	or	a3, a3, a5
	sltu	a2, s5, a2
	sltu	a5, a7, a1
	and	a2, a2, a5
	sltu	a4, s5, a4
	sltu	a1, s2, a1
	and	a1, a1, a4
	or	a1, a1, a2
	or	a1, a1, a3
	beqz	a1, .LBB2_22
# %bb.12:                               #   in Loop: Header=BB2_7 Depth=1
	li	a3, 0
	ld	a1, 144(sp)                     # 8-byte Folded Reload
.LBB2_13:                               #   in Loop: Header=BB2_7 Depth=1
	mv	s10, s11
	mv	s11, s8
	mv	s8, s7
	ld	s7, 120(sp)                     # 8-byte Folded Reload
.LBB2_14:                               #   Parent Loop BB2_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a2, s2, a3
	lbu	a7, 0(a2)
	add	a4, s9, a3
	lbu	t2, 0(a4)
	add	s1, t3, a3
	lbu	a0, 0(s1)
	add	s0, a6, a3
	lbu	a5, 0(s0)
	mul	a7, s7, a7
	mul	t2, s8, t2
	mul	t6, s11, a0
	mul	a0, s10, a5
	lbu	a2, 1(a2)
	lbu	a4, 1(a4)
	lbu	s1, 1(s1)
	lbu	s0, 1(s0)
	mul	a2, s7, a2
	mul	a4, s8, a4
	mul	s1, s11, s1
	mul	s0, s10, s0
	add	a5, a7, t2
	add	a0, a0, t6
	add	a0, a0, a5
	add	a0, a0, ra
	srli	a0, a0, 16
	add	a5, s3, a3
	sb	a0, -1(a5)
	add	a2, a2, a4
	add	s0, s0, s1
	add	a2, a2, s0
	add	a2, a2, ra
	srli	a2, a2, 16
	addi	a3, a3, 2
	sb	a2, 0(a5)
	bge	t1, a3, .LBB2_14
.LBB2_15:                               #   in Loop: Header=BB2_7 Depth=1
	ld	t2, 200(sp)                     # 8-byte Folded Reload
	mv	s0, s7
	mv	s7, s8
	mv	s8, s11
	mv	s11, s10
	ld	a4, 192(sp)                     # 8-byte Folded Reload
	ld	s10, 176(sp)                    # 8-byte Folded Reload
	sext.w	a2, a3
	bge	a2, a4, .LBB2_6
.LBB2_16:                               #   in Loop: Header=BB2_7 Depth=1
	csrr	a2, vlenb
	slli	a0, a2, 1
	li	a4, 32
	bltu	a4, a0, .LBB2_18
# %bb.17:                               #   in Loop: Header=BB2_7 Depth=1
	li	a0, 32
.LBB2_18:                               #   in Loop: Header=BB2_7 Depth=1
	slli	a1, a3, 32
	srli	t2, a1, 32
	sub	t6, s4, t2
	bltu	t6, a0, .LBB2_20
# %bb.19:                               #   in Loop: Header=BB2_7 Depth=1
	ld	a1, 112(sp)                     # 8-byte Folded Reload
	add	a1, a1, t5
	ld	a3, 128(sp)                     # 8-byte Folded Reload
	add	a3, a3, t5
	ld	a4, 104(sp)                     # 8-byte Folded Reload
	add	a4, a4, t5
	ld	a5, 152(sp)                     # 8-byte Folded Reload
	add	a5, a5, t5
	slli	t5, a2, 1
	add	s10, s10, t2
	add	a1, a1, t2
	sub	a1, s10, a1
	sltu	a1, a1, t5
	add	a3, a3, t2
	sub	a2, s10, a3
	sltu	a2, a2, t5
	or	a1, a1, a2
	add	a4, a4, t2
	sub	a2, s10, a4
	sltu	a2, a2, t5
	add	a5, a5, t2
	sub	a3, s10, a5
	sltu	a3, a3, t5
	or	a2, a2, a3
	or	a1, a1, a2
	beqz	a1, .LBB2_25
.LBB2_20:                               #   in Loop: Header=BB2_7 Depth=1
	mv	s1, t2
	ld	t2, 200(sp)                     # 8-byte Folded Reload
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	ld	a4, 192(sp)                     # 8-byte Folded Reload
.LBB2_21:                               #   Parent Loop BB2_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a0, s2, s1
	lbu	a0, 0(a0)
	add	a2, s9, s1
	lbu	a5, 0(a2)
	add	a2, t3, s1
	lbu	a2, 0(a2)
	add	a3, a6, s1
	lbu	a3, 0(a3)
	mul	a0, s0, a0
	mul	a5, s7, a5
	mul	a2, s8, a2
	mul	a3, s11, a3
	add	a0, a0, a5
	add	a2, a2, a3
	add	a0, a0, a2
	add	a0, a0, ra
	srli	a0, a0, 16
	add	a2, s5, s1
	addi	s1, s1, 1
	sb	a0, 0(a2)
	bne	s4, s1, .LBB2_21
	j	.LBB2_6
.LBB2_22:                               #   in Loop: Header=BB2_7 Depth=1
	ld	s1, 40(sp)                      # 8-byte Folded Reload
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	li	a3, 32
.LBB2_23:                               #   Parent Loop BB2_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a2, s2, a1
	vsetvli	zero, a3, e8, m2, ta, ma
	vlseg2e8.v	v8, (a2)
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, a2, sp
	addi	a2, a2, 224
	csrr	a4, vlenb
	slli	a4, a4, 1
	vs2r.v	v8, (a2)                        # Unknown-size Folded Spill
	add	a2, a2, a4
	vs2r.v	v10, (a2)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e32, m8, ta, ma
	vzext.vf4	v16, v8
	vmul.vx	v8, v16, a0
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, a2, sp
	addi	a2, a2, 224
	vs8r.v	v8, (a2)                        # Unknown-size Folded Spill
	add	a2, s9, a1
	vsetvli	zero, zero, e8, m2, ta, ma
	vlseg2e8.v	v20, (a2)
	vsetvli	zero, zero, e32, m8, ta, ma
	vzext.vf4	v24, v20
	add	a2, t3, a1
	vsetvli	zero, zero, e8, m2, ta, ma
	vlseg2e8.v	v14, (a2)
	vsetvli	zero, zero, e32, m8, ta, ma
	add	a2, a6, a1
	vzext.vf4	v0, v14
	vsetvli	zero, zero, e8, m2, ta, ma
	vlseg2e8.v	v18, (a2)
	vsetvli	zero, zero, e32, m8, ta, ma
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, a2, sp
	addi	a2, a2, 224
	vl8r.v	v8, (a2)                        # Unknown-size Folded Reload
	vmadd.vx	v24, s7, v8
	vmacc.vx	v24, s8, v0
	vzext.vf4	v0, v18
	vmacc.vx	v24, s11, v0
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, a2, sp
	addi	a2, a2, 224
	csrr	a4, vlenb
	slli	a4, a4, 1
	vl2r.v	v8, (a2)                        # Unknown-size Folded Reload
	add	a2, a2, a4
	vl2r.v	v10, (a2)                       # Unknown-size Folded Reload
	vzext.vf4	v0, v10
	vmul.vx	v8, v0, a0
	addi	a2, sp, 224
	vs8r.v	v8, (a2)                        # Unknown-size Folded Spill
	vzext.vf4	v8, v22
	vadd.vx	v24, v24, ra
	vzext.vf4	v0, v16
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, a2, sp
	addi	a2, a2, 224
	vs8r.v	v0, (a2)                        # Unknown-size Folded Spill
	vzext.vf4	v0, v20
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, a2, sp
	addi	a2, a2, 224
	vs8r.v	v0, (a2)                        # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v24, 16
	vsetvli	zero, zero, e8, m2, ta, ma
	vnsrl.wi	v20, v16, 0
	vsetvli	zero, zero, e32, m8, ta, ma
	addi	a2, sp, 224
	vl8r.v	v24, (a2)                       # Unknown-size Folded Reload
	vmadd.vx	v8, s7, v24
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, a2, sp
	addi	a2, a2, 224
	vl8r.v	v24, (a2)                       # Unknown-size Folded Reload
	vmacc.vx	v8, s8, v24
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, a2, sp
	addi	a2, a2, 224
	vl8r.v	v24, (a2)                       # Unknown-size Folded Reload
	vmacc.vx	v8, s11, v24
	vadd.vx	v8, v8, ra
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, zero, e8, m2, ta, ma
	vnsrl.wi	v22, v16, 0
	add	a2, s5, a1
	vsseg2e8.v	v20, (a2)
	addi	s1, s1, -32
	addi	a1, a1, 64
	bnez	s1, .LBB2_23
# %bb.24:                               #   in Loop: Header=BB2_7 Depth=1
	ld	a3, 24(sp)                      # 8-byte Folded Reload
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	mv	s10, s11
	mv	s11, s8
	mv	s8, s7
	mv	s7, a0
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	ld	a2, 40(sp)                      # 8-byte Folded Reload
	bne	a0, a2, .LBB2_14
	j	.LBB2_15
.LBB2_25:                               #   in Loop: Header=BB2_7 Depth=1
	li	a4, 0
	neg	a2, t5
	and	a7, t6, a2
	add	s1, a7, t2
	add	s10, s2, t2
	add	a2, s9, t2
	add	a5, a6, t2
	add	a3, t3, t2
	add	t2, t2, s5
.LBB2_26:                               #   Parent Loop BB2_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s10, a4
	vl2r.v	v16, (a1)
	vsetvli	a1, zero, e32, m8, ta, ma
	vzext.vf4	v8, v16
	add	a1, a2, a4
	vl2r.v	v16, (a1)
	add	a1, a3, a4
	vl2r.v	v18, (a1)
	add	a1, a5, a4
	vl2r.v	v20, (a1)
	vmul.vx	v8, v8, s0
	vzext.vf4	v24, v16
	vzext.vf4	v0, v18
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 224
	vs8r.v	v0, (a0)                        # Unknown-size Folded Spill
	vzext.vf4	v0, v20
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 224
	vs8r.v	v0, (a0)                        # Unknown-size Folded Spill
	vmadd.vx	v24, s7, v8
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 224
	vl8r.v	v8, (a0)                        # Unknown-size Folded Reload
	vmacc.vx	v24, s8, v8
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 224
	vl8r.v	v8, (a0)                        # Unknown-size Folded Reload
	vmacc.vx	v24, s11, v8
	vadd.vx	v8, v24, ra
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, zero, e8, m2, ta, ma
	vnsrl.wi	v8, v16, 0
	add	a1, t2, a4
	add	a4, a4, t5
	vs2r.v	v8, (a1)
	bne	a7, a4, .LBB2_26
# %bb.27:                               #   in Loop: Header=BB2_7 Depth=1
	ld	t2, 200(sp)                     # 8-byte Folded Reload
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	ld	a4, 192(sp)                     # 8-byte Folded Reload
	bne	t6, a7, .LBB2_21
	j	.LBB2_6
.LBB2_28:
	sd	zero, 216(sp)
	sd	zero, 208(sp)
	slli	a3, a3, 32
	slli	a0, a0, 32
	srli	a0, a0, 32
	or	a5, a3, a0
	addi	a6, sp, 208
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	mv	a2, s0
	mv	a3, a7
	mv	a4, s4
	call	_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE
	mv	t2, a0
	ld	a7, 200(sp)                     # 8-byte Folded Reload
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	blez	a0, .LBB2_75
# %bb.29:
	lw	a0, 212(sp)
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	lwu	s2, 208(sp)
	lw	t4, 216(sp)
	lw	a1, 220(sp)
	mulw	a6, s2, s0
	mulw	a0, t4, s0
	sd	s6, 112(sp)                     # 8-byte Folded Spill
	sd	a1, 96(sp)                      # 8-byte Folded Spill
	sd	a6, 192(sp)                     # 8-byte Folded Spill
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	blez	s0, .LBB2_61
# %bb.30:
	li	a4, 0
	li	a5, 0
	fmul.s	fa5, fs1, fs2
	#APP
	fcvt.w.s	s5, fa5, rne
	#NO_APP
	fmul.s	fa5, fs0, fs2
	#APP
	fcvt.w.s	t6, fa5, rne
	#NO_APP
	sext.w	t1, s2
	sext.w	s4, s4
	add	s1, a7, a6
	add	a2, a6, s0
	sd	a2, 40(sp)                      # 8-byte Folded Spill
	sub	a0, a0, a6
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	sub	s3, s4, t4
	csrr	s10, vlenb
	srli	a0, s10, 3
	slli	a3, s10, 1
	mul	a2, s0, a0
	slli	a2, a2, 4
	mul	a0, t4, s0
	add	ra, a7, a0
	lui	t5, 8
	sd	s1, 80(sp)                      # 8-byte Folded Spill
	sd	s7, 64(sp)                      # 8-byte Folded Spill
	sd	s8, 56(sp)                      # 8-byte Folded Spill
	sd	s11, 48(sp)                     # 8-byte Folded Spill
	ld	s7, 120(sp)                     # 8-byte Folded Reload
	ld	s8, 64(sp)                      # 8-byte Folded Reload
	ld	s11, 56(sp)                     # 8-byte Folded Reload
	j	.LBB2_32
.LBB2_31:                               #   in Loop: Header=BB2_32 Depth=1
	addiw	a5, a5, 1
	ld	s6, 112(sp)                     # 8-byte Folded Reload
	ld	a7, 200(sp)                     # 8-byte Folded Reload
	add	a7, a7, s6
	ld	a4, 168(sp)                     # 8-byte Folded Reload
	addi	a4, a4, 1
	ld	ra, 152(sp)                     # 8-byte Folded Reload
	add	ra, ra, s6
	ld	s1, 176(sp)                     # 8-byte Folded Reload
	add	s1, s1, s6
	mv	t2, t0
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	ld	a6, 192(sp)                     # 8-byte Folded Reload
	lui	t5, 8
	beq	a5, a0, .LBB2_75
.LBB2_32:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_34 Depth 2
                                        #       Child Loop BB2_38 Depth 3
                                        #       Child Loop BB2_41 Depth 3
                                        #       Child Loop BB2_45 Depth 3
                                        #       Child Loop BB2_48 Depth 3
                                        #     Child Loop BB2_55 Depth 2
                                        #     Child Loop BB2_58 Depth 2
	sd	s1, 176(sp)                     # 8-byte Folded Spill
	li	s9, 0
	sd	a4, 168(sp)                     # 8-byte Folded Spill
	mul	a0, a4, s6
	ld	a4, 80(sp)                      # 8-byte Folded Reload
	add	a0, a0, a4
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	slt	a0, a5, a0
	not	a0, a0
	sd	a5, 160(sp)                     # 8-byte Folded Spill
	slt	a1, a5, a1
	and	a0, a0, a1
	neg	a0, a0
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	and	a0, a0, a1
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	add	t0, t2, a0
	sd	ra, 152(sp)                     # 8-byte Folded Spill
	sd	a7, 200(sp)                     # 8-byte Folded Spill
	ld	t3, 128(sp)                     # 8-byte Folded Reload
	j	.LBB2_34
.LBB2_33:                               #   in Loop: Header=BB2_34 Depth=2
	addi	s9, s9, 1
	addi	a7, a7, 1
	addi	ra, ra, 1
	beq	s9, s0, .LBB2_49
.LBB2_34:                               #   Parent Loop BB2_32 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_38 Depth 3
                                        #       Child Loop BB2_41 Depth 3
                                        #       Child Loop BB2_45 Depth 3
                                        #       Child Loop BB2_48 Depth 3
	blez	t1, .LBB2_42
# %bb.35:                               #   in Loop: Header=BB2_34 Depth=2
	add	a0, s9, a6
	add	a1, t0, a0
	lbu	a1, 0(a1)
	add	a0, a0, t2
	lbu	a0, 0(a0)
	mul	a1, t6, a1
	mul	a0, s5, a0
	add	a0, a0, a1
	add	a0, a0, t5
	slli	a1, s10, 1
	srliw	a0, a0, 16
	bgeu	s2, a1, .LBB2_37
# %bb.36:                               #   in Loop: Header=BB2_34 Depth=2
	li	a1, 0
	j	.LBB2_40
.LBB2_37:                               #   in Loop: Header=BB2_34 Depth=2
	neg	a1, a1
	and	a1, a1, s2
	vsetvli	a4, zero, e8, m2, ta, ma
	vmv.v.x	v8, a0
	mv	a4, a7
	mv	a5, a1
.LBB2_38:                               #   Parent Loop BB2_32 Depth=1
                                        #     Parent Loop BB2_34 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsse8.v	v8, (a4), s0
	sub	a5, a5, a3
	add	a4, a4, a2
	bnez	a5, .LBB2_38
# %bb.39:                               #   in Loop: Header=BB2_34 Depth=2
	beq	a1, s2, .LBB2_42
.LBB2_40:                               #   in Loop: Header=BB2_34 Depth=2
	mul	a4, s0, a1
	add	a4, a4, a7
	sub	a1, s2, a1
.LBB2_41:                               #   Parent Loop BB2_32 Depth=1
                                        #     Parent Loop BB2_34 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	sb	a0, 0(a4)
	addi	a1, a1, -1
	add	a4, a4, s0
	bnez	a1, .LBB2_41
.LBB2_42:                               #   in Loop: Header=BB2_34 Depth=2
	bge	t4, s4, .LBB2_33
# %bb.43:                               #   in Loop: Header=BB2_34 Depth=2
	add	a0, s9, t3
	add	a1, t0, a0
	lbu	a1, 0(a1)
	add	a0, a0, t2
	lbu	a0, 0(a0)
	mul	a1, t6, a1
	mul	a0, s5, a0
	add	a0, a0, a1
	add	a0, a0, t5
	slli	a4, s10, 1
	srliw	a0, a0, 16
	mv	a1, t4
	bltu	s3, a4, .LBB2_47
# %bb.44:                               #   in Loop: Header=BB2_34 Depth=2
	neg	a1, a4
	and	s1, s3, a1
	add	a1, s1, t4
	vsetvli	a4, zero, e8, m2, ta, ma
	vmv.v.x	v8, a0
	mv	a5, ra
	mv	a4, s1
.LBB2_45:                               #   Parent Loop BB2_32 Depth=1
                                        #     Parent Loop BB2_34 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsse8.v	v8, (a5), s0
	sub	a4, a4, a3
	add	a5, a5, a2
	bnez	a4, .LBB2_45
# %bb.46:                               #   in Loop: Header=BB2_34 Depth=2
	beq	s3, s1, .LBB2_33
.LBB2_47:                               #   in Loop: Header=BB2_34 Depth=2
	mul	a4, s0, a1
	add	a4, a4, a7
	sub	a1, s4, a1
.LBB2_48:                               #   Parent Loop BB2_32 Depth=1
                                        #     Parent Loop BB2_34 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	sb	a0, 0(a4)
	addi	a1, a1, -1
	add	a4, a4, s0
	bnez	a1, .LBB2_48
	j	.LBB2_33
.LBB2_49:                               #   in Loop: Header=BB2_32 Depth=1
	bge	a6, t3, .LBB2_59
# %bb.50:                               #   in Loop: Header=BB2_32 Depth=1
	li	a1, 32
	mv	a0, a3
	bltu	a1, a3, .LBB2_52
# %bb.51:                               #   in Loop: Header=BB2_32 Depth=1
	li	a0, 32
.LBB2_52:                               #   in Loop: Header=BB2_32 Depth=1
	ld	s6, 192(sp)                     # 8-byte Folded Reload
	ld	t5, 48(sp)                      # 8-byte Folded Reload
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	lui	t3, 8
	bltu	a1, a0, .LBB2_57
# %bb.53:                               #   in Loop: Header=BB2_32 Depth=1
	slli	a7, s10, 1
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	ld	ra, 136(sp)                     # 8-byte Folded Reload
	add	a0, a4, ra
	add	a0, a0, t2
	ld	a5, 88(sp)                      # 8-byte Folded Reload
	sub	a0, a5, a0
	sltu	a0, a0, a7
	ld	s6, 192(sp)                     # 8-byte Folded Reload
	add	s9, s6, t2
	add	ra, ra, s9
	sub	a1, a5, ra
	sltu	a1, a1, a7
	or	a1, a1, a0
	add	a6, a4, t2
	sub	a4, a5, a6
	sltu	a4, a4, a7
	sub	a5, a5, s9
	sltu	a5, a5, a7
	or	a4, a4, a5
	or	a1, a1, a4
	bnez	a1, .LBB2_57
# %bb.54:                               #   in Loop: Header=BB2_32 Depth=1
	li	s1, 0
	neg	a1, a7
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	and	a1, a1, a4
	ld	s6, 192(sp)                     # 8-byte Folded Reload
	add	s6, s6, a1
	ld	a5, 136(sp)                     # 8-byte Folded Reload
	add	a5, a5, a6
	ld	a0, 176(sp)                     # 8-byte Folded Reload
.LBB2_55:                               #   Parent Loop BB2_32 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a4, s9, s1
	vl2r.v	v16, (a4)
	vsetvli	a4, zero, e32, m8, ta, ma
	vzext.vf4	v8, v16
	add	a4, a6, s1
	vl2r.v	v16, (a4)
	add	a4, ra, s1
	vl2r.v	v18, (a4)
	add	a4, a5, s1
	vl2r.v	v20, (a4)
	vmul.vx	v8, v8, s7
	vzext.vf4	v24, v16
	vzext.vf4	v0, v18
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, a4, sp
	addi	a4, a4, 224
	vs8r.v	v0, (a4)                        # Unknown-size Folded Spill
	vzext.vf4	v0, v20
	csrr	a4, vlenb
	slli	a4, a4, 4
	add	a4, a4, sp
	addi	a4, a4, 224
	vs8r.v	v0, (a4)                        # Unknown-size Folded Spill
	vmadd.vx	v24, s8, v8
	csrr	a4, vlenb
	slli	a4, a4, 3
	add	a4, a4, sp
	addi	a4, a4, 224
	vl8r.v	v8, (a4)                        # Unknown-size Folded Reload
	vmacc.vx	v24, s11, v8
	csrr	a4, vlenb
	slli	a4, a4, 4
	add	a4, a4, sp
	addi	a4, a4, 224
	vl8r.v	v8, (a4)                        # Unknown-size Folded Reload
	vmacc.vx	v24, t5, v8
	vadd.vx	v8, v24, t3
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, zero, e8, m2, ta, ma
	vnsrl.wi	v8, v16, 0
	add	a4, a0, s1
	add	s1, s1, a7
	vs2r.v	v8, (a4)
	bne	a1, s1, .LBB2_55
# %bb.56:                               #   in Loop: Header=BB2_32 Depth=1
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	beq	a0, a1, .LBB2_59
.LBB2_57:                               #   in Loop: Header=BB2_32 Depth=1
	add	a7, t2, s0
	ld	s9, 136(sp)                     # 8-byte Folded Reload
	add	s9, s9, a7
	ld	a5, 128(sp)                     # 8-byte Folded Reload
	ld	a1, 200(sp)                     # 8-byte Folded Reload
	mv	a6, t0
	mv	s1, t2
.LBB2_58:                               #   Parent Loop BB2_32 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a0, s1, s6
	lbu	ra, 0(a0)
	add	a0, a7, s6
	lbu	a0, 0(a0)
	add	a4, a6, s6
	lbu	a4, 0(a4)
	add	t3, s9, s6
	lbu	t3, 0(t3)
	mul	ra, s7, ra
	mul	a0, s8, a0
	mul	a4, s11, a4
	mul	t3, t5, t3
	add	a0, a0, ra
	add	a4, a4, t3
	lui	t3, 8
	add	a0, a0, a4
	add	a0, a0, t3
	srli	a0, a0, 16
	add	a4, a1, s6
	sb	a0, 0(a4)
	addi	a7, a7, 1
	addi	s9, s9, 1
	addi	s1, s1, 1
	addi	a6, a6, 1
	addi	a5, a5, -1
	addi	a1, a1, 1
	bne	s6, a5, .LBB2_58
.LBB2_59:                               #   in Loop: Header=BB2_32 Depth=1
	ld	a1, 96(sp)                      # 8-byte Folded Reload
	ld	a5, 160(sp)                     # 8-byte Folded Reload
	blt	a5, a1, .LBB2_31
# %bb.60:                               #   in Loop: Header=BB2_32 Depth=1
	mv	t0, t2
	j	.LBB2_31
.LBB2_61:
	bge	a6, a0, .LBB2_75
# %bb.62:
	csrr	t6, vlenb
	sd	t6, 168(sp)                     # 8-byte Folded Spill
	slli	t6, t6, 1
	li	a0, 32
	mv	t1, t6
	bltu	a0, t6, .LBB2_64
# %bb.63:
	li	t1, 32
.LBB2_64:
	li	s3, 0
	li	t3, 0
	add	s5, a7, a6
	add	t4, s0, a6
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	sub	t5, a0, a6
	lui	s9, 8
	sd	s5, 176(sp)                     # 8-byte Folded Spill
	j	.LBB2_66
.LBB2_65:                               #   in Loop: Header=BB2_66 Depth=1
	addiw	t3, t3, 1
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	add	a7, a7, a0
	addi	s3, s3, 1
	add	s5, s5, a0
	mv	t2, s2
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	ld	a6, 192(sp)                     # 8-byte Folded Reload
	beq	t3, a0, .LBB2_75
.LBB2_66:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_69 Depth 2
                                        #     Child Loop BB2_72 Depth 2
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	slt	a0, t3, a0
	not	a0, a0
	slt	a1, t3, a1
	and	a0, a0, a1
	neg	a0, a0
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	and	s4, a0, a1
	add	s2, t2, s4
	mv	s6, a6
	ld	a6, 120(sp)                     # 8-byte Folded Reload
	bltu	t5, t1, .LBB2_71
# %bb.67:                               #   in Loop: Header=BB2_66 Depth=1
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	mul	a0, s3, a0
	ld	a1, 176(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	add	a1, t4, s4
	add	a1, a1, t2
	sub	a1, a0, a1
	sltu	a2, a1, t6
	ld	s6, 192(sp)                     # 8-byte Folded Reload
	add	t0, s6, t2
	add	s10, t0, s4
	sub	a3, a0, s10
	sltu	a3, a3, t6
	add	s1, t4, t2
	sub	a4, a0, s1
	sltu	a4, a4, t6
	or	a3, a3, a4
	or	a2, a2, a3
	sub	a0, a0, t0
	sltu	a0, a0, t6
	or	a2, a2, a0
	bnez	a2, .LBB2_71
# %bb.68:                               #   in Loop: Header=BB2_66 Depth=1
	li	a4, 0
	ld	a5, 168(sp)                     # 8-byte Folded Reload
	slli	a5, a5, 1
	neg	a0, a5
	and	a2, t5, a0
	ld	s6, 192(sp)                     # 8-byte Folded Reload
	add	s6, s6, a2
	add	a3, s1, s4
.LBB2_69:                               #   Parent Loop BB2_66 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, t0, a4
	vl2r.v	v16, (a1)
	vsetvli	a1, zero, e32, m8, ta, ma
	vzext.vf4	v8, v16
	add	a1, s1, a4
	vl2r.v	v16, (a1)
	add	a1, s10, a4
	vl2r.v	v18, (a1)
	add	a1, a3, a4
	vl2r.v	v20, (a1)
	vmul.vx	v8, v8, a6
	vzext.vf4	v24, v16
	vzext.vf4	v0, v18
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 224
	vs8r.v	v0, (a0)                        # Unknown-size Folded Spill
	vzext.vf4	v0, v20
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 224
	vs8r.v	v0, (a0)                        # Unknown-size Folded Spill
	vmadd.vx	v24, s7, v8
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 224
	vl8r.v	v8, (a0)                        # Unknown-size Folded Reload
	vmacc.vx	v24, s8, v8
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 224
	vl8r.v	v8, (a0)                        # Unknown-size Folded Reload
	vmacc.vx	v24, s11, v8
	vadd.vx	v8, v24, s9
	vsetvli	zero, zero, e16, m4, ta, ma
	vnsrl.wi	v16, v8, 16
	vsetvli	zero, zero, e8, m2, ta, ma
	vnsrl.wi	v8, v16, 0
	add	a1, s5, a4
	add	a4, a4, a5
	vs2r.v	v8, (a1)
	bne	a2, a4, .LBB2_69
# %bb.70:                               #   in Loop: Header=BB2_66 Depth=1
	beq	t5, a2, .LBB2_73
.LBB2_71:                               #   in Loop: Header=BB2_66 Depth=1
	add	t0, t2, s0
	add	s4, s4, t0
	ld	a3, 128(sp)                     # 8-byte Folded Reload
	mv	s1, a7
	mv	a1, s2
	mv	a2, t2
.LBB2_72:                               #   Parent Loop BB2_66 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a5, a2, s6
	lbu	s10, 0(a5)
	add	a0, t0, s6
	lbu	ra, 0(a0)
	add	a4, a1, s6
	lbu	a4, 0(a4)
	add	a5, s4, s6
	lbu	a5, 0(a5)
	mul	s10, a6, s10
	mul	a0, s7, ra
	mul	a4, s8, a4
	mul	a5, s11, a5
	add	a0, a0, s10
	add	a4, a4, a5
	add	a0, a0, a4
	add	a0, a0, s9
	srli	a0, a0, 16
	add	a4, s1, s6
	sb	a0, 0(a4)
	addi	t0, t0, 1
	addi	s4, s4, 1
	addi	a2, a2, 1
	addi	a1, a1, 1
	addi	a3, a3, -1
	addi	s1, s1, 1
	bne	s6, a3, .LBB2_72
.LBB2_73:                               #   in Loop: Header=BB2_66 Depth=1
	ld	a1, 96(sp)                      # 8-byte Folded Reload
	blt	t3, a1, .LBB2_65
# %bb.74:                               #   in Loop: Header=BB2_66 Depth=1
	mv	s2, t2
	j	.LBB2_65
.LBB2_75:
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	sp, sp, a0
	ld	ra, 344(sp)                     # 8-byte Folded Reload
	ld	s0, 336(sp)                     # 8-byte Folded Reload
	ld	s1, 328(sp)                     # 8-byte Folded Reload
	ld	s2, 320(sp)                     # 8-byte Folded Reload
	ld	s3, 312(sp)                     # 8-byte Folded Reload
	ld	s4, 304(sp)                     # 8-byte Folded Reload
	ld	s5, 296(sp)                     # 8-byte Folded Reload
	ld	s6, 288(sp)                     # 8-byte Folded Reload
	ld	s7, 280(sp)                     # 8-byte Folded Reload
	ld	s8, 272(sp)                     # 8-byte Folded Reload
	ld	s9, 264(sp)                     # 8-byte Folded Reload
	ld	s10, 256(sp)                    # 8-byte Folded Reload
	ld	s11, 248(sp)                    # 8-byte Folded Reload
	fld	fs0, 240(sp)                    # 8-byte Folded Reload
	fld	fs1, 232(sp)                    # 8-byte Folded Reload
	fld	fs2, 224(sp)                    # 8-byte Folded Reload
	addi	sp, sp, 352
	ret
.Lfunc_end2:
	.size	_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi, .Lfunc_end2-_ZN2cv17getRectSubPix_Cn_IhhiNS_11scale_fixptENS_7cast_8uEEEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,"axG",@progbits,_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,comdat
	.hidden	_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi # -- Begin function _ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.weak	_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.p2align	1
	.type	_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,@function
_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi: # @_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -288
	.cfi_def_cfa_offset 288
	sd	ra, 280(sp)                     # 8-byte Folded Spill
	sd	s0, 272(sp)                     # 8-byte Folded Spill
	sd	s1, 264(sp)                     # 8-byte Folded Spill
	sd	s2, 256(sp)                     # 8-byte Folded Spill
	sd	s3, 248(sp)                     # 8-byte Folded Spill
	sd	s4, 240(sp)                     # 8-byte Folded Spill
	sd	s5, 232(sp)                     # 8-byte Folded Spill
	sd	s6, 224(sp)                     # 8-byte Folded Spill
	sd	s7, 216(sp)                     # 8-byte Folded Spill
	sd	s8, 208(sp)                     # 8-byte Folded Spill
	sd	s9, 200(sp)                     # 8-byte Folded Spill
	sd	s10, 192(sp)                    # 8-byte Folded Spill
	sd	s11, 184(sp)                    # 8-byte Folded Spill
	fsd	fs0, 176(sp)                    # 8-byte Folded Spill
	fsd	fs1, 168(sp)                    # 8-byte Folded Spill
	fsd	fs2, 160(sp)                    # 8-byte Folded Spill
	fsd	fs3, 152(sp)                    # 8-byte Folded Spill
	fsd	fs4, 144(sp)                    # 8-byte Folded Spill
	fsd	fs5, 136(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	.cfi_offset fs0, -112
	.cfi_offset fs1, -120
	.cfi_offset fs2, -128
	.cfi_offset fs3, -136
	.cfi_offset fs4, -144
	.cfi_offset fs5, -152
	mv	s10, a6
	mv	s3, a5
	mv	s7, a4
	mv	s2, a3
	mv	a3, a2
	srli	a2, a5, 32
	addiw	a4, a5, -1
	fcvt.s.w	fa5, a4
	lui	a4, 258048
	fmv.w.x	fa4, a4
	fnmsub.s	fa5, fa5, fa4, fa0
	addiw	a4, a2, -1
	fcvt.s.w	fa3, a4
	fnmsub.s	fa4, fa3, fa4, fa1
	#APP
	fcvt.w.s	a4, fa5, rdn
	#NO_APP
	sext.w	s1, a4
	#APP
	fcvt.w.s	a5, fa4, rdn
	#NO_APP
	sext.w	a6, a5
	fcvt.s.w	fa3, a4
	fsub.s	fa5, fa5, fa3
	fcvt.s.w	fa3, a5
	fsub.s	fs4, fa4, fa3
	lui	a4, 260096
	fmv.w.x	fa4, a4
	fsub.s	fa3, fa4, fa5
	fsub.s	fs5, fa4, fs4
	fmul.s	fs0, fa3, fs5
	fmul.s	fs1, fa5, fs5
	fmul.s	fs2, fa3, fs4
	fmul.s	fs3, fa5, fs4
	srli	a4, a1, 2
	sd	a4, 80(sp)                      # 8-byte Folded Spill
	srli	s5, s7, 2
	sext.w	a4, a2
	sd	a4, 112(sp)                     # 8-byte Folded Spill
	bltz	s1, .LBB3_19
# %bb.1:
	srli	a4, a3, 32
	subw	a5, a3, s3
	slt	a5, s1, a5
	slti	s0, a6, 0
	not	s0, s0
	subw	a4, a4, a2
	slt	a2, a6, a4
	and	a2, a2, a5
	and	a2, a2, s0
	beqz	a2, .LBB3_19
# %bb.2:
	vsetvli	a7, zero, e32, m1, ta, ma
	ld	a2, 112(sp)                     # 8-byte Folded Reload
	blez	a2, .LBB3_66
# %bb.3:
	li	t5, 0
	li	s4, 0
	mulw	s3, s3, s10
	vfmv.v.f	v8, fs0
	vfmv.v.f	v9, fs1
	vfmv.v.f	v10, fs2
	vfmv.v.f	v11, fs3
	slli	a2, a6, 32
	srli	a3, a2, 32
	ld	s0, 80(sp)                      # 8-byte Folded Reload
	mul	a3, s0, a3
	mulw	a4, s1, s10
	add	a5, a3, a4
	slli	a5, a5, 2
	add	t4, a0, a5
	subw	a6, s3, a7
	slli	s1, s3, 32
	srli	t6, s1, 32
	andi	s1, s7, -4
	sd	s1, 64(sp)                      # 8-byte Folded Spill
	andi	a1, a1, -4
	sd	a1, 56(sp)                      # 8-byte Folded Spill
	add	a1, s0, s10
	sd	a1, 104(sp)                     # 8-byte Folded Spill
	vsetvli	a1, zero, e32, m2, ta, ma
	vfmv.v.f	v12, fs0
	vfmv.v.f	v14, fs2
	vfmv.v.f	v16, fs3
	slli	ra, s5, 2
	slli	t1, a7, 2
	slli	s11, s0, 2
	add	a5, a5, s11
	add	t2, a0, a5
	slli	s6, s10, 2
	add	s5, a0, s6
	add	s5, s5, a5
	slli	a3, a3, 2
	slli	a4, a4, 2
	add	a1, s6, a4
	add	a1, a1, a0
	add	t3, a1, a3
	csrr	a1, vlenb
	srli	t0, a1, 1
	sd	a1, 40(sp)                      # 8-byte Folded Spill
	slli	a1, a1, 1
	sd	a1, 96(sp)                      # 8-byte Folded Spill
	add	s6, s6, t4
	srli	a2, a2, 30
	addi	a2, a2, 4
	mul	s8, s0, a2
	add	a0, a0, a4
	add	s8, s8, a0
	sd	t4, 88(sp)                      # 8-byte Folded Spill
	mv	s9, s2
	sd	t6, 72(sp)                      # 8-byte Folded Spill
	sd	ra, 48(sp)                      # 8-byte Folded Spill
	sd	s3, 32(sp)                      # 8-byte Folded Spill
	j	.LBB3_5
.LBB3_4:                                #   in Loop: Header=BB3_5 Depth=1
	addiw	s4, s4, 1
	addi	t5, t5, 1
	add	s9, s9, ra
	add	t4, t4, s11
	add	t2, t2, s11
	add	s5, s5, s11
	add	t3, t3, s11
	add	s6, s6, s11
	add	s8, s8, s11
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	beq	s4, a0, .LBB3_66
.LBB3_5:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_7 Depth 2
                                        #     Child Loop BB3_15 Depth 2
                                        #     Child Loop BB3_18 Depth 2
	li	a1, 0
	bltz	a6, .LBB3_8
# %bb.6:                                #   in Loop: Header=BB3_5 Depth=1
	li	a5, 0
	li	a2, 0
	vsetvli	a3, zero, e32, m1, ta, ma
.LBB3_7:                                #   Parent Loop BB3_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a3, t4, a5
	vle32.v	v18, (a3)
	add	a3, t3, a5
	vle32.v	v19, (a3)
	vfmul.vv	v18, v18, v8
	vfmul.vv	v19, v19, v9
	add	a3, t2, a5
	vle32.v	v20, (a3)
	add	a3, s5, a5
	vle32.v	v21, (a3)
	vfadd.vv	v18, v18, v19
	vfmul.vv	v19, v20, v10
	vfadd.vv	v18, v18, v19
	vfmul.vv	v19, v21, v11
	vfadd.vv	v18, v18, v19
	add	a3, s9, a5
	vse32.v	v18, (a3)
	add	a2, a2, a7
	add	a5, a5, t1
	addw	a1, a1, a7
	bge	a6, a2, .LBB3_7
.LBB3_8:                                #   in Loop: Header=BB3_5 Depth=1
	bge	a1, s3, .LBB3_4
# %bb.9:                                #   in Loop: Header=BB3_5 Depth=1
	li	a3, 24
	mv	a2, t0
	bltu	a3, t0, .LBB3_11
# %bb.10:                               #   in Loop: Header=BB3_5 Depth=1
	li	a2, 24
.LBB3_11:                               #   in Loop: Header=BB3_5 Depth=1
	slli	a1, a1, 32
	srli	s7, a1, 32
	sub	a3, t6, s7
	bltu	a3, a2, .LBB3_17
# %bb.12:                               #   in Loop: Header=BB3_5 Depth=1
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	mul	a1, a0, t5
	add	a1, a1, s2
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	mul	a2, a0, t5
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	add	a2, a2, a0
	slli	ra, s7, 2
	add	a1, a1, ra
	ld	t6, 104(sp)                     # 8-byte Folded Reload
	add	t6, t6, s7
	slli	t6, t6, 2
	add	a4, a2, t6
	sub	a4, a1, a4
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	sltu	a4, a4, a0
	ld	a5, 80(sp)                      # 8-byte Folded Reload
	add	a5, a5, s7
	slli	a5, a5, 2
	add	a5, a5, a2
	sub	a5, a1, a5
	sltu	a5, a5, a0
	or	a4, a4, a5
	add	a5, s10, s7
	slli	a5, a5, 2
	add	a5, a5, a2
	sub	a5, a1, a5
	sltu	a5, a5, a0
	add	a2, a2, ra
	sub	a1, a1, a2
	sltu	a1, a1, a0
	or	a1, a1, a5
	or	a1, a1, a4
	beqz	a1, .LBB3_14
# %bb.13:                               #   in Loop: Header=BB3_5 Depth=1
	ld	t6, 72(sp)                      # 8-byte Folded Reload
	ld	ra, 48(sp)                      # 8-byte Folded Reload
	j	.LBB3_17
.LBB3_14:                               #   in Loop: Header=BB3_5 Depth=1
	li	s3, 0
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	srli	a1, a1, 1
	neg	a1, a1
	and	a1, a1, a3
	add	s7, s7, a1
	add	a2, t4, ra
	add	a5, s6, ra
	add	t6, t6, t4
	add	a4, s8, ra
	add	ra, ra, s9
	vsetvli	s0, zero, e32, m2, ta, ma
	mv	s0, a1
	ld	a0, 96(sp)                      # 8-byte Folded Reload
.LBB3_15:                               #   Parent Loop BB3_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	s1, a2, s3
	vl2re32.v	v18, (s1)
	add	s1, a5, s3
	vl2re32.v	v20, (s1)
	add	s1, a4, s3
	vl2re32.v	v22, (s1)
	add	s1, t6, s3
	vl2re32.v	v24, (s1)
	vfmul.vf	v20, v20, fs1
	vfmacc.vv	v20, v12, v18
	vfmadd.vv	v22, v14, v20
	vfmadd.vv	v24, v16, v22
	add	s1, ra, s3
	vs2r.v	v24, (s1)
	sub	s0, s0, t0
	add	s3, s3, a0
	bnez	s0, .LBB3_15
# %bb.16:                               #   in Loop: Header=BB3_5 Depth=1
	ld	s3, 32(sp)                      # 8-byte Folded Reload
	ld	t6, 72(sp)                      # 8-byte Folded Reload
	ld	ra, 48(sp)                      # 8-byte Folded Reload
	beq	a3, a1, .LBB3_4
.LBB3_17:                               #   in Loop: Header=BB3_5 Depth=1
	li	a0, 0
	slli	s0, s7, 2
	add	a5, s6, s0
	add	a2, s9, s0
	ld	a3, 104(sp)                     # 8-byte Folded Reload
	add	a3, a3, s7
	slli	a3, a3, 2
	add	a4, t4, a3
	add	s1, s8, s0
	add	s0, s0, t4
	sub	a3, t6, s7
.LBB3_18:                               #   Parent Loop BB3_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s0, a0
	flw	fa5, 0(a1)
	add	a1, a5, a0
	flw	fa4, 0(a1)
	add	a1, s1, a0
	flw	fa3, 0(a1)
	add	a1, a4, a0
	flw	fa2, 0(a1)
	fmul.s	fa4, fs1, fa4
	fmadd.s	fa5, fa5, fs0, fa4
	fmadd.s	fa5, fa3, fs2, fa5
	fmadd.s	fa5, fa2, fs3, fa5
	add	a1, a2, a0
	fsw	fa5, 0(a1)
	addi	a3, a3, -1
	addi	a0, a0, 4
	bnez	a3, .LBB3_18
	j	.LBB3_4
.LBB3_19:
	sd	zero, 128(sp)
	sd	zero, 120(sp)
	andi	a1, a1, -4
	slliw	a2, s10, 2
	slli	a6, a6, 32
	slli	s1, s1, 32
	srli	s1, s1, 32
	or	a5, a6, s1
	addi	a6, sp, 120
	mv	a4, s3
	call	_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE
	mv	t2, a0
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	blez	a0, .LBB3_66
# %bb.20:
	lw	a0, 124(sp)
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	lwu	s11, 120(sp)
	lw	t0, 128(sp)
	lw	a1, 132(sp)
	mulw	s8, s11, s10
	mulw	s6, t0, s10
	sd	a1, 64(sp)                      # 8-byte Folded Spill
	blez	s10, .LBB3_52
# %bb.21:
	li	a3, 0
	li	t1, 0
	sext.w	t3, s11
	sext.w	s3, s3
	slli	a0, s8, 2
	sd	a0, 16(sp)                      # 8-byte Folded Spill
	add	a4, a0, s2
	andi	a0, s7, -4
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	add	a0, s8, s10
	slli	a0, a0, 2
	sd	a0, 8(sp)                       # 8-byte Folded Spill
	sub	a0, s6, s8
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	sub	a7, s3, t0
	vsetvli	a0, zero, e32, m2, ta, ma
	vfmv.v.f	v8, fs0
	vfmv.v.f	v10, fs2
	vfmv.v.f	v12, fs3
	csrr	t5, vlenb
	srli	s9, t5, 3
	srli	s1, t5, 1
	slli	s5, s5, 2
	sd	s5, 32(sp)                      # 8-byte Folded Spill
	mul	s0, s10, s9
	slli	s0, s0, 4
	slli	a2, s10, 2
	mul	a0, t0, s10
	slli	a0, a0, 2
	add	t4, s2, a0
	slli	a6, t5, 1
	sd	a4, 48(sp)                      # 8-byte Folded Spill
	sd	a4, 104(sp)                     # 8-byte Folded Spill
	j	.LBB3_23
.LBB3_22:                               #   in Loop: Header=BB3_23 Depth=1
	addiw	t1, t1, 1
	ld	a3, 96(sp)                      # 8-byte Folded Reload
	addi	a3, a3, 1
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	add	s2, s2, a0
	add	t4, t4, a0
	ld	a4, 104(sp)                     # 8-byte Folded Reload
	add	a4, a4, a0
	sd	a4, 104(sp)                     # 8-byte Folded Spill
	mv	t2, t6
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	beq	t1, a0, .LBB3_66
.LBB3_23:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_25 Depth 2
                                        #       Child Loop BB3_29 Depth 3
                                        #       Child Loop BB3_32 Depth 3
                                        #       Child Loop BB3_36 Depth 3
                                        #       Child Loop BB3_39 Depth 3
                                        #     Child Loop BB3_46 Depth 2
                                        #     Child Loop BB3_49 Depth 2
	li	s5, 0
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	mul	a0, a0, a3
	ld	a3, 48(sp)                      # 8-byte Folded Reload
	add	a0, a0, a3
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	slt	a0, t1, a0
	not	a0, a0
	slt	a3, t1, a1
	and	a0, a0, a3
	neg	a0, a0
	ld	a1, 80(sp)                      # 8-byte Folded Reload
	and	s4, a0, a1
	slli	s4, s4, 2
	add	t6, t2, s4
	sd	t4, 88(sp)                      # 8-byte Folded Spill
	mv	s7, s2
	j	.LBB3_25
.LBB3_24:                               #   in Loop: Header=BB3_25 Depth=2
	addi	s5, s5, 1
	addi	s7, s7, 4
	addi	t4, t4, 4
	beq	s5, s10, .LBB3_40
.LBB3_25:                               #   Parent Loop BB3_23 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB3_29 Depth 3
                                        #       Child Loop BB3_32 Depth 3
                                        #       Child Loop BB3_36 Depth 3
                                        #       Child Loop BB3_39 Depth 3
	blez	t3, .LBB3_33
# %bb.26:                               #   in Loop: Header=BB3_25 Depth=2
	add	a0, s5, s8
	slli	a0, a0, 2
	add	a1, t6, a0
	flw	fa5, 0(a1)
	add	a0, a0, t2
	flw	fa4, 0(a0)
	fmul.s	fa5, fs4, fa5
	srli	a0, t5, 1
	fmadd.s	fa5, fa4, fs5, fa5
	bgeu	s11, a0, .LBB3_28
# %bb.27:                               #   in Loop: Header=BB3_25 Depth=2
	li	a0, 0
	j	.LBB3_31
.LBB3_28:                               #   in Loop: Header=BB3_25 Depth=2
	slli	a0, s9, 2
	neg	a0, a0
	and	a0, a0, s11
	vfmv.v.f	v14, fa5
	slli	a4, s10, 2
	mv	a5, s7
	mv	a3, a0
.LBB3_29:                               #   Parent Loop BB3_23 Depth=1
                                        #     Parent Loop BB3_25 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsse32.v	v14, (a5), a4
	sub	a3, a3, s1
	add	a5, a5, s0
	bnez	a3, .LBB3_29
# %bb.30:                               #   in Loop: Header=BB3_25 Depth=2
	beq	a0, s11, .LBB3_33
.LBB3_31:                               #   in Loop: Header=BB3_25 Depth=2
	mul	a1, s10, a0
	add	a1, a1, s5
	slli	a1, a1, 2
	add	a4, s2, a1
	sub	a3, s11, a0
.LBB3_32:                               #   Parent Loop BB3_23 Depth=1
                                        #     Parent Loop BB3_25 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	fsw	fa5, 0(a4)
	addi	a3, a3, -1
	add	a4, a4, a2
	bnez	a3, .LBB3_32
.LBB3_33:                               #   in Loop: Header=BB3_25 Depth=2
	bge	t0, s3, .LBB3_24
# %bb.34:                               #   in Loop: Header=BB3_25 Depth=2
	add	a0, s5, s6
	slli	a0, a0, 2
	add	a1, t6, a0
	flw	fa5, 0(a1)
	add	a0, a0, t2
	flw	fa4, 0(a0)
	fmul.s	fa5, fs4, fa5
	srli	a0, t5, 1
	fmadd.s	fa5, fa4, fs5, fa5
	mv	a3, t0
	bltu	a7, a0, .LBB3_38
# %bb.35:                               #   in Loop: Header=BB3_25 Depth=2
	neg	a0, a0
	and	a4, a7, a0
	add	a3, a4, t0
	vfmv.v.f	v14, fa5
	slli	a5, s10, 2
	mv	ra, t4
	mv	a0, a4
.LBB3_36:                               #   Parent Loop BB3_23 Depth=1
                                        #     Parent Loop BB3_25 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsse32.v	v14, (ra), a5
	sub	a0, a0, s1
	add	ra, ra, s0
	bnez	a0, .LBB3_36
# %bb.37:                               #   in Loop: Header=BB3_25 Depth=2
	beq	a7, a4, .LBB3_24
.LBB3_38:                               #   in Loop: Header=BB3_25 Depth=2
	mul	a0, s10, a3
	add	a0, a0, s5
	slli	a0, a0, 2
	add	a4, s2, a0
	sub	a3, s3, a3
.LBB3_39:                               #   Parent Loop BB3_23 Depth=1
                                        #     Parent Loop BB3_25 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	fsw	fa5, 0(a4)
	addi	a3, a3, -1
	add	a4, a4, a2
	bnez	a3, .LBB3_39
	j	.LBB3_24
.LBB3_40:                               #   in Loop: Header=BB3_23 Depth=1
	ld	t4, 88(sp)                      # 8-byte Folded Reload
	bge	s8, s6, .LBB3_50
# %bb.41:                               #   in Loop: Header=BB3_23 Depth=1
	li	a1, 16
	mv	a0, s1
	bltu	a1, s1, .LBB3_43
# %bb.42:                               #   in Loop: Header=BB3_23 Depth=1
	li	a0, 16
.LBB3_43:                               #   in Loop: Header=BB3_23 Depth=1
	mv	s5, s8
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	bltu	a1, a0, .LBB3_48
# %bb.44:                               #   in Loop: Header=BB3_23 Depth=1
	ld	a4, 8(sp)                       # 8-byte Folded Reload
	add	a4, a4, t2
	add	a0, a4, s4
	ld	a5, 56(sp)                      # 8-byte Folded Reload
	sub	a0, a5, a0
	sltu	a0, a0, a6
	ld	a3, 16(sp)                      # 8-byte Folded Reload
	add	a3, a3, t2
	add	a1, a3, s4
	sub	a1, a5, a1
	sltu	a1, a1, a6
	or	a0, a0, a1
	sub	a1, a5, a4
	sltu	a1, a1, a6
	sub	a5, a5, a3
	sltu	a5, a5, a6
	or	a1, a1, a5
	or	a0, a0, a1
	mv	s5, s8
	bnez	a0, .LBB3_48
# %bb.45:                               #   in Loop: Header=BB3_23 Depth=1
	srli	a0, t5, 1
	neg	a0, a0
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	and	s7, a1, a0
	add	s5, s7, s8
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	mv	a5, s7
.LBB3_46:                               #   Parent Loop BB3_23 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vl2re32.v	v14, (a3)
	vl2re32.v	v16, (a4)
	add	a1, a3, s4
	vl2re32.v	v18, (a1)
	add	a1, a4, s4
	vl2re32.v	v20, (a1)
	vfmul.vf	v16, v16, fs1
	vfmacc.vv	v16, v8, v14
	vfmadd.vv	v18, v10, v16
	vfmadd.vv	v20, v12, v18
	vs2r.v	v20, (a0)
	sub	a5, a5, s1
	add	a3, a3, a6
	add	a4, a4, a6
	add	a0, a0, a6
	bnez	a5, .LBB3_46
# %bb.47:                               #   in Loop: Header=BB3_23 Depth=1
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	beq	a0, s7, .LBB3_50
.LBB3_48:                               #   in Loop: Header=BB3_23 Depth=1
	add	a0, s10, s5
	slli	a0, a0, 2
	add	a3, t2, a0
	slli	a0, s5, 2
	add	a1, t2, a0
	add	a0, a0, s2
	sub	a5, s6, s5
.LBB3_49:                               #   Parent Loop BB3_23 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	flw	fa5, 0(a1)
	flw	fa4, 0(a3)
	add	a4, a1, s4
	flw	fa3, 0(a4)
	add	a4, a3, s4
	flw	fa2, 0(a4)
	fmul.s	fa4, fs1, fa4
	fmadd.s	fa5, fa5, fs0, fa4
	fmadd.s	fa5, fa3, fs2, fa5
	fmadd.s	fa5, fa2, fs3, fa5
	fsw	fa5, 0(a0)
	addi	a3, a3, 4
	addi	a1, a1, 4
	addi	a5, a5, -1
	addi	a0, a0, 4
	bnez	a5, .LBB3_49
.LBB3_50:                               #   in Loop: Header=BB3_23 Depth=1
	ld	a1, 64(sp)                      # 8-byte Folded Reload
	blt	t1, a1, .LBB3_22
# %bb.51:                               #   in Loop: Header=BB3_23 Depth=1
	mv	t6, t2
	j	.LBB3_22
.LBB3_52:
	bge	s8, s6, .LBB3_66
# %bb.53:
	slli	t0, s8, 2
	csrr	a6, vlenb
	srli	s4, a6, 1
	li	a0, 16
	add	t6, s10, s8
	mv	a7, s4
	bltu	a0, s4, .LBB3_55
# %bb.54:
	li	a7, 16
.LBB3_55:
	li	t5, 0
	li	s9, 0
	add	t3, t0, s2
	andi	t1, s7, -4
	slli	t6, t6, 2
	sub	t4, s6, s8
	vsetvli	a0, zero, e32, m2, ta, ma
	vfmv.v.f	v8, fs0
	vfmv.v.f	v10, fs2
	vfmv.v.f	v12, fs3
	slli	s1, a6, 1
	slli	s5, s5, 2
	mv	s3, t3
	j	.LBB3_57
.LBB3_56:                               #   in Loop: Header=BB3_57 Depth=1
	addiw	s9, s9, 1
	addi	t5, t5, 1
	add	s3, s3, s5
	add	s2, s2, s5
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	beq	s9, a0, .LBB3_66
.LBB3_57:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_60 Depth 2
                                        #     Child Loop BB3_63 Depth 2
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	slt	a0, s9, a0
	not	a0, a0
	slt	a1, s9, a1
	and	a0, a0, a1
	neg	a0, a0
	ld	a1, 80(sp)                      # 8-byte Folded Reload
	and	a1, a1, a0
	slli	a5, a1, 2
	mv	a4, s8
	bltu	t4, a7, .LBB3_62
# %bb.58:                               #   in Loop: Header=BB3_57 Depth=1
	mul	a0, t1, t5
	add	a0, a0, t3
	add	s0, t6, t2
	add	a2, s0, a5
	sub	a2, a0, a2
	sltu	a2, a2, s1
	add	a3, t0, t2
	add	a4, a3, a5
	sub	a4, a0, a4
	sltu	a4, a4, s1
	or	a2, a2, a4
	sub	a4, a0, s0
	sltu	a4, a4, s1
	sub	a0, a0, a3
	sltu	a0, a0, s1
	or	a0, a0, a4
	or	a0, a0, a2
	mv	a4, s8
	bnez	a0, .LBB3_62
# %bb.59:                               #   in Loop: Header=BB3_57 Depth=1
	srli	a0, a6, 1
	neg	a0, a0
	and	s7, t4, a0
	add	a4, s7, s8
	mv	a0, s3
	mv	a2, s7
.LBB3_60:                               #   Parent Loop BB3_57 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vl2re32.v	v14, (a3)
	vl2re32.v	v16, (s0)
	add	a1, a3, a5
	vl2re32.v	v18, (a1)
	add	a1, s0, a5
	vl2re32.v	v20, (a1)
	vfmul.vf	v16, v16, fs1
	vfmacc.vv	v16, v8, v14
	vfmadd.vv	v18, v10, v16
	vfmadd.vv	v20, v12, v18
	vs2r.v	v20, (a0)
	sub	a2, a2, s4
	add	a3, a3, s1
	add	s0, s0, s1
	add	a0, a0, s1
	bnez	a2, .LBB3_60
# %bb.61:                               #   in Loop: Header=BB3_57 Depth=1
	beq	t4, s7, .LBB3_64
.LBB3_62:                               #   in Loop: Header=BB3_57 Depth=1
	add	a0, s10, a4
	slli	a0, a0, 2
	add	a3, t2, a0
	slli	s0, a4, 2
	add	a0, t2, s0
	add	s0, s0, s2
	sub	a4, s6, a4
.LBB3_63:                               #   Parent Loop BB3_57 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	flw	fa5, 0(a0)
	flw	fa4, 0(a3)
	add	a1, a0, a5
	flw	fa3, 0(a1)
	add	a1, a3, a5
	flw	fa2, 0(a1)
	fmul.s	fa4, fs1, fa4
	fmadd.s	fa5, fa5, fs0, fa4
	fmadd.s	fa5, fa3, fs2, fa5
	fmadd.s	fa5, fa2, fs3, fa5
	fsw	fa5, 0(s0)
	addi	a3, a3, 4
	addi	a0, a0, 4
	addi	a4, a4, -1
	addi	s0, s0, 4
	bnez	a4, .LBB3_63
.LBB3_64:                               #   in Loop: Header=BB3_57 Depth=1
	ld	a1, 64(sp)                      # 8-byte Folded Reload
	bge	s9, a1, .LBB3_56
# %bb.65:                               #   in Loop: Header=BB3_57 Depth=1
	add	t2, t2, a5
	j	.LBB3_56
.LBB3_66:
	ld	ra, 280(sp)                     # 8-byte Folded Reload
	ld	s0, 272(sp)                     # 8-byte Folded Reload
	ld	s1, 264(sp)                     # 8-byte Folded Reload
	ld	s2, 256(sp)                     # 8-byte Folded Reload
	ld	s3, 248(sp)                     # 8-byte Folded Reload
	ld	s4, 240(sp)                     # 8-byte Folded Reload
	ld	s5, 232(sp)                     # 8-byte Folded Reload
	ld	s6, 224(sp)                     # 8-byte Folded Reload
	ld	s7, 216(sp)                     # 8-byte Folded Reload
	ld	s8, 208(sp)                     # 8-byte Folded Reload
	ld	s9, 200(sp)                     # 8-byte Folded Reload
	ld	s10, 192(sp)                    # 8-byte Folded Reload
	ld	s11, 184(sp)                    # 8-byte Folded Reload
	fld	fs0, 176(sp)                    # 8-byte Folded Reload
	fld	fs1, 168(sp)                    # 8-byte Folded Reload
	fld	fs2, 160(sp)                    # 8-byte Folded Reload
	fld	fs3, 152(sp)                    # 8-byte Folded Reload
	fld	fs4, 144(sp)                    # 8-byte Folded Reload
	fld	fs5, 136(sp)                    # 8-byte Folded Reload
	addi	sp, sp, 288
	ret
.Lfunc_end3:
	.size	_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi, .Lfunc_end3-_ZN2cv17getRectSubPix_Cn_IfffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2cv5utils5trace7details6RegionD2Ev,"axG",@progbits,_ZN2cv5utils5trace7details6RegionD2Ev,comdat
	.hidden	_ZN2cv5utils5trace7details6RegionD2Ev # -- Begin function _ZN2cv5utils5trace7details6RegionD2Ev
	.weak	_ZN2cv5utils5trace7details6RegionD2Ev
	.p2align	1
	.type	_ZN2cv5utils5trace7details6RegionD2Ev,@function
_ZN2cv5utils5trace7details6RegionD2Ev:  # @_ZN2cv5utils5trace7details6RegionD2Ev
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception1
# %bb.0:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sd	ra, 8(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	lw	a1, 8(a0)
	beqz	a1, .LBB4_2
# %bb.1:
.Ltmp69:
	call	_ZN2cv5utils5trace7details6Region7destroyEv@plt
.Ltmp70:
.LBB4_2:
	ld	ra, 8(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 16
	ret
.LBB4_3:
.Ltmp71:
	call	__clang_call_terminate
.Lfunc_end4:
	.size	_ZN2cv5utils5trace7details6RegionD2Ev, .Lfunc_end4-_ZN2cv5utils5trace7details6RegionD2Ev
	.cfi_endproc
	.section	.gcc_except_table._ZN2cv5utils5trace7details6RegionD2Ev,"aG",@progbits,_ZN2cv5utils5trace7details6RegionD2Ev,comdat
	.p2align	2, 0x0
GCC_except_table4:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase1-.Lttbaseref1
.Lttbaseref1:
	.byte	3                               # Call site Encoding = udata4
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.word	.Ltmp69-.Lfunc_begin1           # >> Call Site 1 <<
	.word	.Ltmp70-.Ltmp69                 #   Call between .Ltmp69 and .Ltmp70
	.word	.Ltmp71-.Lfunc_begin1           #     jumps to .Ltmp71
	.byte	1                               #   On action: 1
.Lcst_end1:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.word	0                               # TypeInfo 1
.Lttbase1:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,"axG",@progbits,_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,comdat
	.hidden	_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi # -- Begin function _ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.weak	_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.p2align	1
	.type	_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi,@function
_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi: # @_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -400
	.cfi_def_cfa_offset 400
	sd	ra, 392(sp)                     # 8-byte Folded Spill
	sd	s0, 384(sp)                     # 8-byte Folded Spill
	sd	s1, 376(sp)                     # 8-byte Folded Spill
	sd	s2, 368(sp)                     # 8-byte Folded Spill
	sd	s3, 360(sp)                     # 8-byte Folded Spill
	sd	s4, 352(sp)                     # 8-byte Folded Spill
	sd	s5, 344(sp)                     # 8-byte Folded Spill
	sd	s6, 336(sp)                     # 8-byte Folded Spill
	sd	s7, 328(sp)                     # 8-byte Folded Spill
	sd	s8, 320(sp)                     # 8-byte Folded Spill
	sd	s9, 312(sp)                     # 8-byte Folded Spill
	sd	s10, 304(sp)                    # 8-byte Folded Spill
	sd	s11, 296(sp)                    # 8-byte Folded Spill
	fsd	fs0, 288(sp)                    # 8-byte Folded Spill
	fsd	fs1, 280(sp)                    # 8-byte Folded Spill
	fsd	fs2, 272(sp)                    # 8-byte Folded Spill
	fsd	fs3, 264(sp)                    # 8-byte Folded Spill
	fsd	fs4, 256(sp)                    # 8-byte Folded Spill
	fsd	fs5, 248(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	.cfi_offset fs0, -112
	.cfi_offset fs1, -120
	.cfi_offset fs2, -128
	.cfi_offset fs3, -136
	.cfi_offset fs4, -144
	.cfi_offset fs5, -152
	mv	s5, a6
	mv	s4, a5
	mv	s3, a4
	sd	a3, 192(sp)                     # 8-byte Folded Spill
	mv	a3, a2
	mv	s10, a1
	srli	a2, a5, 32
	addiw	a1, a5, -1
	fcvt.s.w	fa5, a1
	lui	a1, 258048
	fmv.w.x	fa4, a1
	fnmsub.s	fa5, fa5, fa4, fa0
	addiw	a1, a2, -1
	fcvt.s.w	fa3, a1
	fnmsub.s	fa4, fa3, fa4, fa1
	#APP
	fcvt.w.s	a4, fa5, rdn
	#NO_APP
	sext.w	s0, a4
	#APP
	fcvt.w.s	a5, fa4, rdn
	#NO_APP
	sext.w	a1, a5
	fcvt.s.w	fa3, a4
	fsub.s	fa5, fa5, fa3
	fcvt.s.w	fa3, a5
	fsub.s	fs4, fa4, fa3
	lui	a4, 260096
	fmv.w.x	fa4, a4
	fsub.s	fa3, fa4, fa5
	fsub.s	fs5, fa4, fs4
	fmul.s	fs0, fa3, fs5
	fmul.s	fs1, fa5, fs5
	fmul.s	fs2, fa3, fs4
	fmul.s	fs3, fa5, fs4
	srli	s11, s3, 2
	sext.w	a4, a2
	sd	a4, 224(sp)                     # 8-byte Folded Spill
	sd	s10, 184(sp)                    # 8-byte Folded Spill
	bltz	s0, .LBB5_29
# %bb.1:
	srli	a4, a3, 32
	subw	a5, a3, s4
	slt	a5, s0, a5
	slti	s1, a1, 0
	not	s1, s1
	subw	a4, a4, a2
	slt	a2, a1, a4
	and	a2, a2, a5
	and	a2, a2, s1
	beqz	a2, .LBB5_29
# %bb.2:
	ld	a2, 224(sp)                     # 8-byte Folded Reload
	blez	a2, .LBB5_76
# %bb.3:
	mv	a5, s11
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	mulw	a6, s4, s5
	slli	a1, a1, 32
	srli	a1, a1, 32
	mul	t4, a1, s10
	mulw	t5, s0, s5
	add	a0, t4, t5
	addiw	t3, a6, -2
	sd	a6, 144(sp)                     # 8-byte Folded Spill
	slli	a6, a6, 32
	srli	t2, a6, 32
	addi	a1, a1, 1
	mul	s9, a1, s10
	add	s7, s9, s5
	add	a1, s7, t5
	sd	a1, 216(sp)                     # 8-byte Folded Spill
	add	a1, s9, t5
	sd	a1, 208(sp)                     # 8-byte Folded Spill
	add	s6, t4, s5
	add	a1, s6, t5
	sd	a1, 200(sp)                     # 8-byte Folded Spill
	mv	a4, t3
	bgtz	t3, .LBB5_5
# %bb.4:
	li	a4, 1
.LBB5_5:
	li	s8, 0
	li	s4, 0
	ld	a3, 120(sp)                     # 8-byte Folded Reload
	add	s1, a3, a0
	srli	a1, a6, 30
	sd	a1, 104(sp)                     # 8-byte Folded Spill
	andi	a1, s3, -4
	sd	a1, 176(sp)                     # 8-byte Folded Spill
	ld	a6, 216(sp)                     # 8-byte Folded Reload
	add	a1, a6, t2
	sd	a1, 96(sp)                      # 8-byte Folded Spill
	ld	t1, 208(sp)                     # 8-byte Folded Reload
	add	a1, t1, t2
	sd	a1, 88(sp)                      # 8-byte Folded Spill
	ld	a7, 200(sp)                     # 8-byte Folded Reload
	add	a1, a7, t2
	sd	a1, 80(sp)                      # 8-byte Folded Spill
	add	a1, a0, t2
	sd	a1, 72(sp)                      # 8-byte Folded Spill
	slli	a1, a4, 2
	li	a2, -15
	srli	a2, a2, 1
	and	a1, a1, a2
	addi	a1, a1, 8
	sd	a1, 64(sp)                      # 8-byte Folded Spill
	addi	a2, a2, 6
	and	a1, a4, a2
	add	a2, t1, a1
	addi	a2, a2, 2
	sd	a2, 56(sp)                      # 8-byte Folded Spill
	add	a2, a2, s5
	sd	a2, 48(sp)                      # 8-byte Folded Spill
	add	a0, a0, a1
	addi	a0, a0, 2
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	add	a0, a0, s5
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	sd	a4, 128(sp)                     # 8-byte Folded Spill
	srli	a0, a4, 1
	addi	a0, a0, 1
	sd	a0, 16(sp)                      # 8-byte Folded Spill
	andi	a0, a0, -8
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	slli	a0, a0, 1
	sd	a0, 8(sp)                       # 8-byte Folded Spill
	vsetivli	zero, 8, e32, m2, ta, ma
	vfmv.v.f	v8, fs0
	vfmv.v.f	v10, fs2
	vfmv.v.f	v12, fs3
	vsetvli	a0, zero, e32, m2, ta, ma
	vfmv.v.f	v14, fs0
	vfmv.v.f	v16, fs2
	vfmv.v.f	v18, fs3
	add	t4, t4, a3
	add	t6, a3, s6
	add	t0, a3, s7
	add	s11, a3, s9
	mv	s9, a3
	slli	a0, a5, 2
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	add	a7, a7, a3
	ld	s3, 192(sp)                     # 8-byte Folded Reload
	addi	s2, s3, 4
	add	t1, t1, a3
	add	a6, a6, a3
	csrr	s5, vlenb
	sd	s5, 136(sp)                     # 8-byte Folded Spill
	slli	s5, s5, 1
	sd	s5, 0(sp)                       # 8-byte Folded Spill
	ld	s7, 144(sp)                     # 8-byte Folded Reload
	j	.LBB5_7
.LBB5_6:                                #   in Loop: Header=BB5_7 Depth=1
	addiw	s4, s4, 1
	add	s1, s1, s10
	ld	a0, 168(sp)                     # 8-byte Folded Reload
	add	s3, s3, a0
	addi	s8, s8, 1
	add	t4, t4, s10
	add	t6, t6, s10
	add	t0, t0, s10
	add	s11, s11, s10
	add	a7, a7, s10
	add	s2, s2, a0
	add	t1, t1, s10
	add	a6, a6, s10
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	beq	s4, a0, .LBB5_76
.LBB5_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_22 Depth 2
                                        #     Child Loop BB5_13 Depth 2
                                        #     Child Loop BB5_25 Depth 2
                                        #     Child Loop BB5_28 Depth 2
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	mul	s0, a0, s8
	mul	ra, s8, s10
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	add	a0, a0, ra
	add	s6, s9, a0
	ld	a2, 208(sp)                     # 8-byte Folded Reload
	add	a2, a2, ra
	add	a3, s9, a2
	ld	a1, 200(sp)                     # 8-byte Folded Reload
	add	a1, a1, ra
	add	s5, s9, a1
	li	a0, 2
	bge	s7, a0, .LBB5_9
# %bb.8:                                #   in Loop: Header=BB5_7 Depth=1
	li	a0, 0
	li	a1, 0
	bge	a1, s7, .LBB5_6
	j	.LBB5_15
.LBB5_9:                                #   in Loop: Header=BB5_7 Depth=1
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	li	a1, 14
	sd	a3, 160(sp)                     # 8-byte Folded Spill
	sd	s0, 152(sp)                     # 8-byte Folded Spill
	bltu	a0, a1, .LBB5_11
# %bb.10:                               #   in Loop: Header=BB5_7 Depth=1
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	add	a0, a0, s0
	ld	a1, 192(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	ld	a4, 48(sp)                      # 8-byte Folded Reload
	add	a4, a4, ra
	add	a4, a4, s9
	ld	a5, 56(sp)                      # 8-byte Folded Reload
	add	a5, a5, ra
	add	a5, a5, s9
	ld	s0, 32(sp)                      # 8-byte Folded Reload
	add	s0, s0, ra
	add	s0, s0, s9
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	add	a1, a1, ra
	add	a1, a1, s9
	sltu	a4, s3, a4
	sltu	a2, s6, a0
	and	a2, a2, a4
	sltu	a4, s3, a5
	sltu	a5, a3, a0
	and	a4, a4, a5
	or	a2, a2, a4
	sltu	a4, s3, s0
	sltu	a5, s5, a0
	and	a4, a4, a5
	sltu	a1, s3, a1
	sltu	a0, s1, a0
	and	a0, a0, a1
	or	a0, a0, a4
	or	a0, a0, a2
	beqz	a0, .LBB5_21
.LBB5_11:                               #   in Loop: Header=BB5_7 Depth=1
	li	a0, 0
.LBB5_12:                               #   in Loop: Header=BB5_7 Depth=1
	slli	a5, a0, 2
	add	a5, a5, s2
.LBB5_13:                               #   Parent Loop BB5_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, s1, a0
	lbu	a2, 0(a1)
	add	a4, a7, a0
	lbu	s0, 0(a4)
	fcvt.s.wu	fa5, a2
	add	a2, t1, a0
	lbu	a3, 0(a2)
	fcvt.s.wu	fa4, s0
	fmul.s	fa4, fs1, fa4
	fmadd.s	fa5, fa5, fs0, fa4
	fcvt.s.wu	fa4, a3
	add	a3, a6, a0
	lbu	s0, 0(a3)
	fmadd.s	fa5, fa4, fs2, fa5
	lbu	a1, 1(a1)
	lbu	a4, 1(a4)
	fcvt.s.wu	fa4, s0
	fmadd.s	fa5, fa4, fs3, fa5
	fcvt.s.wu	fa4, a1
	fcvt.s.wu	fa3, a4
	lbu	a1, 1(a2)
	fmul.s	fa3, fs1, fa3
	lbu	a2, 1(a3)
	fmadd.s	fa4, fa4, fs0, fa3
	fcvt.s.wu	fa3, a1
	fmadd.s	fa4, fa3, fs2, fa4
	fcvt.s.wu	fa3, a2
	fmadd.s	fa4, fa3, fs3, fa4
	fsw	fa5, -4(a5)
	fsw	fa4, 0(a5)
	addi	a0, a0, 2
	addi	a5, a5, 8
	bge	t3, a0, .LBB5_13
.LBB5_14:                               #   in Loop: Header=BB5_7 Depth=1
	ld	a3, 160(sp)                     # 8-byte Folded Reload
	ld	s0, 152(sp)                     # 8-byte Folded Reload
	sext.w	a1, a0
	bge	a1, s7, .LBB5_6
.LBB5_15:                               #   in Loop: Header=BB5_7 Depth=1
	mv	a5, s6
	ld	a4, 136(sp)                     # 8-byte Folded Reload
	srli	a4, a4, 1
	li	a1, 16
	bltu	a1, a4, .LBB5_17
# %bb.16:                               #   in Loop: Header=BB5_7 Depth=1
	li	a4, 16
.LBB5_17:                               #   in Loop: Header=BB5_7 Depth=1
	slli	a0, a0, 32
	srli	s7, a0, 32
	sub	a0, t2, s7
	bgeu	a0, a4, .LBB5_19
# %bb.18:                               #   in Loop: Header=BB5_7 Depth=1
	mv	a0, s7
	ld	s7, 144(sp)                     # 8-byte Folded Reload
	j	.LBB5_27
.LBB5_19:                               #   in Loop: Header=BB5_7 Depth=1
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	mv	s6, s5
	mv	s5, a3
	ld	a3, 104(sp)                     # 8-byte Folded Reload
	add	a3, a3, s0
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	add	a3, a3, a0
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	add	a0, a0, ra
	add	a0, a0, s9
	ld	a1, 88(sp)                      # 8-byte Folded Reload
	add	a1, a1, ra
	add	a1, a1, s9
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	add	a2, a2, ra
	add	s10, s9, a2
	ld	a2, 72(sp)                      # 8-byte Folded Reload
	add	ra, ra, a2
	add	ra, ra, s9
	slli	s0, s7, 2
	add	s0, s0, s3
	add	a4, a5, s7
	add	a5, s5, s7
	add	a2, s6, s7
	add	s9, s1, s7
	sltu	a0, s0, a0
	sltu	a4, a4, a3
	and	a0, a0, a4
	sltu	a1, s0, a1
	sltu	a4, a5, a3
	and	a1, a1, a4
	or	a0, a0, a1
	sltu	a1, s0, s10
	sltu	a2, a2, a3
	and	a1, a1, a2
	sltu	a2, s0, ra
	sltu	a3, s9, a3
	and	a2, a2, a3
	or	a1, a1, a2
	or	a0, a0, a1
	beqz	a0, .LBB5_24
# %bb.20:                               #   in Loop: Header=BB5_7 Depth=1
	mv	a0, s7
	ld	s10, 184(sp)                    # 8-byte Folded Reload
	ld	s9, 120(sp)                     # 8-byte Folded Reload
	ld	s7, 144(sp)                     # 8-byte Folded Reload
	j	.LBB5_27
.LBB5_21:                               #   in Loop: Header=BB5_7 Depth=1
	mv	a3, s6
	ld	s0, 24(sp)                      # 8-byte Folded Reload
	mv	s10, s3
	mv	s9, s11
	mv	s7, t0
	mv	s6, t6
	mv	a0, t4
.LBB5_22:                               #   Parent Loop BB5_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a1, a0, t5
	vsetivli	zero, 8, e8, mf2, ta, ma
	vlseg2e8.v	v20, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v22, v20
	vfwcvt.f.xu.v	v24, v22
	add	a1, s6, t5
	vsetvli	zero, zero, e8, mf2, ta, ma
	vlseg2e8.v	v22, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v20, v22
	vfwcvt.f.xu.v	v26, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v26, v26, fs1
	vfmacc.vv	v26, v8, v24
	add	a1, s9, t5
	vsetvli	zero, zero, e8, mf2, ta, ma
	vlseg2e8.v	v24, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v20, v24
	vfwcvt.f.xu.v	v28, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v28, v10, v26
	add	a1, s7, t5
	vsetvli	zero, zero, e8, mf2, ta, ma
	vlseg2e8.v	v26, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v20, v26
	vfwcvt.f.xu.v	v2, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v2, v12, v28
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v20, v21
	vfwcvt.f.xu.v	v28, v20
	vzext.vf2	v20, v23
	vfwcvt.f.xu.v	v22, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v20, v22, fs1
	vfmacc.vv	v20, v8, v28
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v22, v25
	vfwcvt.f.xu.v	v24, v22
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v24, v10, v20
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v20, v27
	vfwcvt.f.xu.v	v4, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v4, v12, v24
	vsseg2e32.v	v2, (s10)
	addi	a0, a0, 16
	addi	s6, s6, 16
	addi	s7, s7, 16
	addi	s9, s9, 16
	addi	s0, s0, -8
	addi	s10, s10, 64
	bnez	s0, .LBB5_22
# %bb.23:                               #   in Loop: Header=BB5_7 Depth=1
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	ld	s10, 184(sp)                    # 8-byte Folded Reload
	ld	s9, 120(sp)                     # 8-byte Folded Reload
	ld	s7, 144(sp)                     # 8-byte Folded Reload
	ld	a1, 16(sp)                      # 8-byte Folded Reload
	ld	a2, 24(sp)                      # 8-byte Folded Reload
	mv	s6, a3
	beq	a1, a2, .LBB5_14
	j	.LBB5_12
.LBB5_24:                               #   in Loop: Header=BB5_7 Depth=1
	li	a4, 0
	ld	a2, 136(sp)                     # 8-byte Folded Reload
	srli	s5, a2, 1
	neg	a0, s5
	ld	s6, 112(sp)                     # 8-byte Folded Reload
	and	a3, s6, a0
	add	a0, a3, s7
	add	s10, a7, s7
	add	a1, a6, s7
	add	s7, s7, t1
	vsetvli	a5, zero, e32, m2, ta, ma
	ld	a2, 0(sp)                       # 8-byte Folded Reload
.LBB5_25:                               #   Parent Loop BB5_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a5, s9, a4
	vle8.v	v20, (a5)
	vsetvli	zero, zero, e16, m1, ta, ma
	add	a5, s10, a4
	vle8.v	v21, (a5)
	vzext.vf2	v22, v20
	vfwcvt.f.xu.v	v24, v22
	vzext.vf2	v20, v21
	vfwcvt.f.xu.v	v22, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	add	a5, s7, a4
	vle8.v	v20, (a5)
	vfmul.vf	v22, v22, fs1
	vfmacc.vv	v22, v14, v24
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v21, v20
	vfwcvt.f.xu.v	v24, v21
	vsetvli	zero, zero, e32, m2, ta, ma
	add	a5, a1, a4
	vle8.v	v20, (a5)
	vfmadd.vv	v24, v16, v22
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v21, v20
	vfwcvt.f.xu.v	v22, v21
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v22, v18, v24
	vs2r.v	v22, (s0)
	add	a4, a4, s5
	add	s0, s0, a2
	bne	a3, a4, .LBB5_25
# %bb.26:                               #   in Loop: Header=BB5_7 Depth=1
	ld	s10, 184(sp)                    # 8-byte Folded Reload
	ld	s9, 120(sp)                     # 8-byte Folded Reload
	ld	s7, 144(sp)                     # 8-byte Folded Reload
	beq	s6, a3, .LBB5_6
.LBB5_27:                               #   in Loop: Header=BB5_7 Depth=1
	slli	a1, a0, 2
	add	a1, a1, s3
.LBB5_28:                               #   Parent Loop BB5_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	add	a2, s1, a0
	lbu	a2, 0(a2)
	add	a3, a7, a0
	lbu	a3, 0(a3)
	fcvt.s.wu	fa5, a2
	fcvt.s.wu	fa4, a3
	fmul.s	fa4, fs1, fa4
	add	a2, t1, a0
	lbu	a2, 0(a2)
	add	a3, a6, a0
	lbu	a3, 0(a3)
	fmadd.s	fa5, fa5, fs0, fa4
	fcvt.s.wu	fa4, a2
	fmadd.s	fa5, fa4, fs2, fa5
	fcvt.s.wu	fa4, a3
	fmadd.s	fa5, fa4, fs3, fa5
	fsw	fa5, 0(a1)
	addi	a0, a0, 1
	addi	a1, a1, 4
	bne	t2, a0, .LBB5_28
	j	.LBB5_6
.LBB5_29:
	sd	zero, 240(sp)
	sd	zero, 232(sp)
	slli	a1, a1, 32
	slli	s0, s0, 32
	srli	a5, s0, 32
	or	a5, a5, a1
	addi	a6, sp, 232
	mv	a1, s10
	mv	a2, s5
	mv	a4, s4
	call	_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE
	mv	t2, a0
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	blez	a0, .LBB5_76
# %bb.30:
	lw	a0, 236(sp)
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	lwu	s10, 232(sp)
	lw	t0, 240(sp)
	lw	a3, 244(sp)
	mulw	s7, s10, s5
	mulw	s6, t0, s5
	sd	a3, 168(sp)                     # 8-byte Folded Spill
	blez	s5, .LBB5_62
# %bb.31:
	li	a2, 0
	li	s2, 0
	sext.w	ra, s10
	sext.w	s4, s4
	slli	a5, s7, 2
	andi	a0, s3, -4
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	slli	a0, s6, 2
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	add	a0, s7, s5
	sd	a0, 104(sp)                     # 8-byte Folded Spill
	add	a0, s6, s5
	sd	a0, 96(sp)                      # 8-byte Folded Spill
	sub	a0, s6, s7
	sd	a0, 120(sp)                     # 8-byte Folded Spill
	sub	s8, s4, t0
	vsetvli	a0, zero, e32, m2, ta, ma
	vfmv.v.f	v8, fs0
	vfmv.v.f	v10, fs2
	vfmv.v.f	v12, fs3
	csrr	t4, vlenb
	srli	s3, t4, 3
	srli	a4, t4, 1
	slli	a0, s11, 2
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	mul	a1, s5, s3
	slli	a1, a1, 4
	slli	s0, s5, 2
	mul	a0, t0, s5
	slli	a0, a0, 2
	ld	a7, 192(sp)                     # 8-byte Folded Reload
	add	s1, a7, a0
	sd	a5, 152(sp)                     # 8-byte Folded Spill
	add	a5, a5, a7
	sd	a5, 216(sp)                     # 8-byte Folded Spill
	slli	a0, t4, 1
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	j	.LBB5_33
.LBB5_32:                               #   in Loop: Header=BB5_33 Depth=1
	addiw	s2, s2, 1
	ld	a2, 208(sp)                     # 8-byte Folded Reload
	addi	a2, a2, 1
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	add	a7, a7, a0
	ld	s1, 200(sp)                     # 8-byte Folded Reload
	add	s1, s1, a0
	ld	a5, 216(sp)                     # 8-byte Folded Reload
	add	a5, a5, a0
	sd	a5, 216(sp)                     # 8-byte Folded Spill
	mv	t2, t6
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	beq	s2, a0, .LBB5_76
.LBB5_33:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_35 Depth 2
                                        #       Child Loop BB5_39 Depth 3
                                        #       Child Loop BB5_42 Depth 3
                                        #       Child Loop BB5_46 Depth 3
                                        #       Child Loop BB5_49 Depth 3
                                        #     Child Loop BB5_56 Depth 2
                                        #     Child Loop BB5_59 Depth 2
	li	t1, 0
	sd	a2, 208(sp)                     # 8-byte Folded Spill
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	mul	a0, a0, a2
	ld	a2, 152(sp)                     # 8-byte Folded Reload
	add	a2, a2, a0
	ld	t3, 192(sp)                     # 8-byte Folded Reload
	add	a2, a2, t3
	sd	a2, 160(sp)                     # 8-byte Folded Spill
	ld	a2, 136(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	add	t3, t3, a0
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	slt	a0, s2, a0
	not	a0, a0
	slt	a2, s2, a3
	and	a0, a0, a2
	neg	a0, a0
	ld	a2, 184(sp)                     # 8-byte Folded Reload
	and	a6, a0, a2
	add	t6, t2, a6
	mv	t5, s1
	mv	s9, a7
	j	.LBB5_35
.LBB5_34:                               #   in Loop: Header=BB5_35 Depth=2
	addi	t1, t1, 1
	addi	s9, s9, 4
	addi	t5, t5, 4
	beq	t1, s5, .LBB5_50
.LBB5_35:                               #   Parent Loop BB5_33 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB5_39 Depth 3
                                        #       Child Loop BB5_42 Depth 3
                                        #       Child Loop BB5_46 Depth 3
                                        #       Child Loop BB5_49 Depth 3
	blez	ra, .LBB5_43
# %bb.36:                               #   in Loop: Header=BB5_35 Depth=2
	add	a0, t1, s7
	add	a2, t2, a0
	lbu	a2, 0(a2)
	add	a0, a0, t6
	lbu	a0, 0(a0)
	fcvt.s.wu	fa5, a2
	fcvt.s.wu	fa4, a0
	fmul.s	fa4, fs4, fa4
	srli	a0, t4, 1
	fmadd.s	fa5, fa5, fs5, fa4
	bgeu	s10, a0, .LBB5_38
# %bb.37:                               #   in Loop: Header=BB5_35 Depth=2
	li	a3, 0
	j	.LBB5_41
.LBB5_38:                               #   in Loop: Header=BB5_35 Depth=2
	slli	a0, s3, 2
	neg	a0, a0
	and	a3, a0, s10
	vfmv.v.f	v14, fa5
	slli	a2, s5, 2
	mv	a5, s9
	mv	a0, a3
.LBB5_39:                               #   Parent Loop BB5_33 Depth=1
                                        #     Parent Loop BB5_35 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsse32.v	v14, (a5), a2
	sub	a0, a0, a4
	add	a5, a5, a1
	bnez	a0, .LBB5_39
# %bb.40:                               #   in Loop: Header=BB5_35 Depth=2
	beq	a3, s10, .LBB5_43
.LBB5_41:                               #   in Loop: Header=BB5_35 Depth=2
	mul	a0, s5, a3
	add	a0, a0, t1
	slli	a0, a0, 2
	add	a2, a7, a0
	sub	a0, s10, a3
.LBB5_42:                               #   Parent Loop BB5_33 Depth=1
                                        #     Parent Loop BB5_35 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	fsw	fa5, 0(a2)
	addi	a0, a0, -1
	add	a2, a2, s0
	bnez	a0, .LBB5_42
.LBB5_43:                               #   in Loop: Header=BB5_35 Depth=2
	bge	t0, s4, .LBB5_34
# %bb.44:                               #   in Loop: Header=BB5_35 Depth=2
	add	a0, t1, s6
	add	a2, t2, a0
	lbu	a2, 0(a2)
	add	a0, a0, t6
	lbu	a0, 0(a0)
	fcvt.s.wu	fa5, a2
	fcvt.s.wu	fa4, a0
	fmul.s	fa4, fs4, fa4
	srli	a2, t4, 1
	fmadd.s	fa5, fa5, fs5, fa4
	mv	a0, t0
	bltu	s8, a2, .LBB5_48
# %bb.45:                               #   in Loop: Header=BB5_35 Depth=2
	neg	a0, a2
	and	a2, s8, a0
	add	a0, a2, t0
	vfmv.v.f	v14, fa5
	slli	a5, s5, 2
	mv	s11, t5
	mv	a3, a2
.LBB5_46:                               #   Parent Loop BB5_33 Depth=1
                                        #     Parent Loop BB5_35 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsse32.v	v14, (s11), a5
	sub	a3, a3, a4
	add	s11, s11, a1
	bnez	a3, .LBB5_46
# %bb.47:                               #   in Loop: Header=BB5_35 Depth=2
	beq	s8, a2, .LBB5_34
.LBB5_48:                               #   in Loop: Header=BB5_35 Depth=2
	mul	a2, s5, a0
	add	a2, a2, t1
	slli	a2, a2, 2
	add	a2, a2, a7
	sub	a0, s4, a0
.LBB5_49:                               #   Parent Loop BB5_33 Depth=1
                                        #     Parent Loop BB5_35 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	fsw	fa5, 0(a2)
	addi	a0, a0, -1
	add	a2, a2, s0
	bnez	a0, .LBB5_49
	j	.LBB5_34
.LBB5_50:                               #   in Loop: Header=BB5_33 Depth=1
	sd	s1, 200(sp)                     # 8-byte Folded Spill
	bge	s7, s6, .LBB5_60
# %bb.51:                               #   in Loop: Header=BB5_33 Depth=1
	li	a2, 16
	mv	a0, a4
	bltu	a2, a4, .LBB5_53
# %bb.52:                               #   in Loop: Header=BB5_33 Depth=1
	li	a0, 16
.LBB5_53:                               #   in Loop: Header=BB5_33 Depth=1
	mv	t1, s7
	ld	a2, 120(sp)                     # 8-byte Folded Reload
	bltu	a2, a0, .LBB5_58
# %bb.54:                               #   in Loop: Header=BB5_33 Depth=1
	ld	t1, 104(sp)                     # 8-byte Folded Reload
	add	t1, t1, t2
	add	a0, t1, a6
	ld	a5, 96(sp)                      # 8-byte Folded Reload
	add	a5, a5, t2
	add	s9, a5, a6
	add	a2, t2, s7
	add	a3, a2, a6
	sd	a3, 112(sp)                     # 8-byte Folded Spill
	add	s11, t2, s6
	add	t5, s11, a6
	mv	a3, t3
	ld	t3, 160(sp)                     # 8-byte Folded Reload
	sltu	s1, t3, s9
	sltu	a0, a0, a3
	and	s9, s1, a0
	sltu	s1, t3, t5
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	sltu	a0, a0, a3
	and	a0, a0, s1
	or	t5, s9, a0
	sltu	a5, t3, a5
	sltu	s1, t1, a3
	and	a5, a5, s1
	sltu	a0, t3, s11
	sltu	s1, a2, a3
	and	a3, a0, s1
	or	a3, a3, a5
	or	a0, t5, a3
	mv	t1, s7
	bnez	a0, .LBB5_58
# %bb.55:                               #   in Loop: Header=BB5_33 Depth=1
	srli	a0, t4, 1
	neg	a0, a0
	ld	a3, 120(sp)                     # 8-byte Folded Reload
	and	s9, a3, a0
	add	t1, s9, s7
	add	t3, s5, a6
	ld	a3, 216(sp)                     # 8-byte Folded Reload
	mv	a0, s9
	ld	s1, 88(sp)                      # 8-byte Folded Reload
.LBB5_56:                               #   Parent Loop BB5_33 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vle8.v	v14, (a2)
	vsetvli	zero, zero, e16, m1, ta, ma
	add	a5, a2, s5
	vle8.v	v15, (a5)
	vzext.vf2	v16, v14
	vfwcvt.f.xu.v	v18, v16
	vzext.vf2	v14, v15
	vfwcvt.f.xu.v	v16, v14
	vsetvli	zero, zero, e32, m2, ta, ma
	add	a5, a2, a6
	vle8.v	v14, (a5)
	vfmul.vf	v16, v16, fs1
	vfmacc.vv	v16, v8, v18
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vfwcvt.f.xu.v	v18, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	add	a5, a2, t3
	vle8.v	v14, (a5)
	vfmadd.vv	v18, v10, v16
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vfwcvt.f.xu.v	v16, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v16, v12, v18
	vs2r.v	v16, (a3)
	sub	a0, a0, a4
	add	a2, a2, a4
	add	a3, a3, s1
	bnez	a0, .LBB5_56
# %bb.57:                               #   in Loop: Header=BB5_33 Depth=1
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	beq	a0, s9, .LBB5_60
.LBB5_58:                               #   in Loop: Header=BB5_33 Depth=1
	add	a3, t2, t1
	add	s1, a3, s5
	slli	t3, t1, 2
	add	t3, t3, a7
	sub	a5, s6, t1
.LBB5_59:                               #   Parent Loop BB5_33 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	lbu	t1, 0(a3)
	lbu	a0, 0(s1)
	fcvt.s.wu	fa5, t1
	fcvt.s.wu	fa4, a0
	fmul.s	fa4, fs1, fa4
	add	a0, a3, a6
	lbu	a0, 0(a0)
	add	a2, s1, a6
	lbu	a2, 0(a2)
	fmadd.s	fa5, fa5, fs0, fa4
	fcvt.s.wu	fa4, a0
	fmadd.s	fa5, fa4, fs2, fa5
	fcvt.s.wu	fa4, a2
	fmadd.s	fa5, fa4, fs3, fa5
	fsw	fa5, 0(t3)
	addi	s1, s1, 1
	addi	a3, a3, 1
	addi	a5, a5, -1
	addi	t3, t3, 4
	bnez	a5, .LBB5_59
.LBB5_60:                               #   in Loop: Header=BB5_33 Depth=1
	ld	a3, 168(sp)                     # 8-byte Folded Reload
	blt	s2, a3, .LBB5_32
# %bb.61:                               #   in Loop: Header=BB5_33 Depth=1
	mv	t6, t2
	j	.LBB5_32
.LBB5_62:
	bge	s7, s6, .LBB5_76
# %bb.63:
	csrr	a1, vlenb
	srli	t5, a1, 1
	li	a0, 16
	slli	t0, s7, 2
	mv	t6, t5
	bltu	a0, t5, .LBB5_65
# %bb.64:
	li	t6, 16
.LBB5_65:
	li	s4, 0
	li	t1, 0
	andi	a0, s3, -4
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	slli	a0, s6, 2
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	add	a0, s5, s7
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	add	t4, s5, s6
	sub	a7, s6, s7
	vsetvli	a0, zero, e32, m2, ta, ma
	vfmv.v.f	v8, fs0
	vfmv.v.f	v10, fs2
	vfmv.v.f	v12, fs3
	ld	s9, 192(sp)                     # 8-byte Folded Reload
	add	s8, s9, t0
	slli	s2, s11, 2
	sd	a1, 160(sp)                     # 8-byte Folded Spill
	slli	s10, a1, 1
	j	.LBB5_67
.LBB5_66:                               #   in Loop: Header=BB5_67 Depth=1
	addiw	t1, t1, 1
	addi	s4, s4, 1
	add	s8, s8, s2
	add	s9, s9, s2
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	beq	t1, a0, .LBB5_76
.LBB5_67:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_70 Depth 2
                                        #     Child Loop BB5_73 Depth 2
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	slt	a0, t1, a0
	not	a0, a0
	slt	a1, t1, a3
	and	a0, a0, a1
	neg	a0, a0
	ld	s1, 184(sp)                     # 8-byte Folded Reload
	and	s1, s1, a0
	mv	a6, s7
	bltu	a7, t6, .LBB5_72
# %bb.68:                               #   in Loop: Header=BB5_67 Depth=1
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	mul	a0, a0, s4
	ld	a1, 208(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	ld	a3, 192(sp)                     # 8-byte Folded Reload
	add	a2, a3, a1
	add	a0, a0, t0
	add	a0, a0, a3
	ld	t3, 200(sp)                     # 8-byte Folded Reload
	add	t3, t3, t2
	add	s3, t3, s1
	add	s0, t2, t4
	add	s11, s0, s1
	add	a3, t2, s7
	add	ra, a3, s1
	add	a6, t2, s6
	mv	a1, t0
	add	t0, a6, s1
	sltu	s11, a0, s11
	sltu	a4, s3, a2
	and	s3, s11, a4
	sltu	t0, a0, t0
	sltu	a4, ra, a2
	and	a4, t0, a4
	or	t0, s3, a4
	sltu	s0, a0, s0
	sltu	a4, t3, a2
	and	a4, a4, s0
	sltu	a0, a0, a6
	sltu	a2, a3, a2
	and	a0, a0, a2
	or	a0, a0, a4
	or	a0, t0, a0
	mv	t0, a1
	mv	a6, s7
	bnez	a0, .LBB5_72
# %bb.69:                               #   in Loop: Header=BB5_67 Depth=1
	ld	a0, 160(sp)                     # 8-byte Folded Reload
	srli	a0, a0, 1
	neg	a0, a0
	and	s3, a7, a0
	add	a6, s3, s7
	add	a0, s5, s1
	mv	a2, s8
	mv	s0, s3
.LBB5_70:                               #   Parent Loop BB5_67 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vle8.v	v14, (a3)
	vsetvli	zero, zero, e16, m1, ta, ma
	add	a1, a3, s5
	vle8.v	v15, (a1)
	vzext.vf2	v16, v14
	vfwcvt.f.xu.v	v18, v16
	vzext.vf2	v14, v15
	vfwcvt.f.xu.v	v16, v14
	vsetvli	zero, zero, e32, m2, ta, ma
	add	a1, a3, s1
	vle8.v	v14, (a1)
	vfmul.vf	v16, v16, fs1
	vfmacc.vv	v16, v8, v18
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vfwcvt.f.xu.v	v18, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	add	a1, a3, a0
	vle8.v	v14, (a1)
	vfmadd.vv	v18, v10, v16
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vfwcvt.f.xu.v	v16, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmadd.vv	v16, v12, v18
	vs2r.v	v16, (a2)
	sub	s0, s0, t5
	add	a3, a3, t5
	add	a2, a2, s10
	bnez	s0, .LBB5_70
# %bb.71:                               #   in Loop: Header=BB5_67 Depth=1
	beq	a7, s3, .LBB5_74
.LBB5_72:                               #   in Loop: Header=BB5_67 Depth=1
	add	a4, t2, a6
	add	a0, a4, s5
	slli	a2, a6, 2
	add	a2, a2, s9
	sub	a5, s6, a6
.LBB5_73:                               #   Parent Loop BB5_67 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	lbu	a3, 0(a4)
	lbu	s0, 0(a0)
	fcvt.s.wu	fa5, a3
	fcvt.s.wu	fa4, s0
	fmul.s	fa4, fs1, fa4
	add	a3, a4, s1
	lbu	a3, 0(a3)
	add	a1, a0, s1
	lbu	a1, 0(a1)
	fmadd.s	fa5, fa5, fs0, fa4
	fcvt.s.wu	fa4, a3
	fmadd.s	fa5, fa4, fs2, fa5
	fcvt.s.wu	fa4, a1
	fmadd.s	fa5, fa4, fs3, fa5
	fsw	fa5, 0(a2)
	addi	a0, a0, 1
	addi	a4, a4, 1
	addi	a5, a5, -1
	addi	a2, a2, 4
	bnez	a5, .LBB5_73
.LBB5_74:                               #   in Loop: Header=BB5_67 Depth=1
	ld	a3, 168(sp)                     # 8-byte Folded Reload
	bge	t1, a3, .LBB5_66
# %bb.75:                               #   in Loop: Header=BB5_67 Depth=1
	add	t2, t2, s1
	j	.LBB5_66
.LBB5_76:
	ld	ra, 392(sp)                     # 8-byte Folded Reload
	ld	s0, 384(sp)                     # 8-byte Folded Reload
	ld	s1, 376(sp)                     # 8-byte Folded Reload
	ld	s2, 368(sp)                     # 8-byte Folded Reload
	ld	s3, 360(sp)                     # 8-byte Folded Reload
	ld	s4, 352(sp)                     # 8-byte Folded Reload
	ld	s5, 344(sp)                     # 8-byte Folded Reload
	ld	s6, 336(sp)                     # 8-byte Folded Reload
	ld	s7, 328(sp)                     # 8-byte Folded Reload
	ld	s8, 320(sp)                     # 8-byte Folded Reload
	ld	s9, 312(sp)                     # 8-byte Folded Reload
	ld	s10, 304(sp)                    # 8-byte Folded Reload
	ld	s11, 296(sp)                    # 8-byte Folded Reload
	fld	fs0, 288(sp)                    # 8-byte Folded Reload
	fld	fs1, 280(sp)                    # 8-byte Folded Reload
	fld	fs2, 272(sp)                    # 8-byte Folded Reload
	fld	fs3, 264(sp)                    # 8-byte Folded Reload
	fld	fs4, 256(sp)                    # 8-byte Folded Reload
	fld	fs5, 248(sp)                    # 8-byte Folded Reload
	addi	sp, sp, 400
	ret
.Lfunc_end5:
	.size	_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi, .Lfunc_end5-_ZN2cv17getRectSubPix_Cn_IhffNS_3nopIfEES2_EEvPKT_mNS_5Size_IiEEPT0_mS7_NS_6Point_IfEEi
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE,"ax",@progbits
	.p2align	1                               # -- Begin function _ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE
	.type	_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE,@function
_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE: # @_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception2
# %bb.0:
	addi	sp, sp, -64
	.cfi_def_cfa_offset 64
	sd	ra, 56(sp)                      # 8-byte Folded Spill
	sd	s0, 48(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	sext.w	t1, a5
	bltz	t1, .LBB6_2
# %bb.1:
	li	t0, 0
	mulw	s0, a5, a2
	add	a0, a0, s0
	j	.LBB6_4
.LBB6_2:
	sext.w	a7, a4
	negw	t0, a5
	blt	t0, a7, .LBB6_4
# %bb.3:
	mv	t0, a7
.LBB6_4:
	subw	s0, a3, a4
	bge	t1, s0, .LBB6_6
# %bb.5:
	mv	a7, a4
	j	.LBB6_8
.LBB6_6:
	sext.w	t1, a4
	not	s0, a5
	addw	t2, s0, a3
	sgtz	s0, t2
	neg	s0, s0
	and	a7, s0, t2
	blt	t1, a7, .LBB6_12
# %bb.7:
	mulw	t1, t2, a2
	sraiw	s0, t2, 31
	and	s0, s0, t1
	add	a0, a0, s0
.LBB6_8:
	srli	t2, a3, 32
	srli	t3, a4, 32
	srli	a3, a5, 32
	sext.w	t1, a3
	slti	s0, a5, 0
	mul	a4, a3, a1
	addi	s0, s0, -1
	and	a4, a4, s0
	subw	s0, t2, t3
	add	a0, a0, a4
	blt	t1, s0, .LBB6_11
# %bb.9:
	not	a4, a3
	addw	t3, a4, t2
	bgez	t3, .LBB6_11
# %bb.10:
	mul	a1, t3, a1
	add	a0, a0, a1
	li	t3, 0
.LBB6_11:
	negw	a1, a3
	srai	a5, a5, 63
	and	a1, a1, a5
	sw	t0, 0(a6)
	sw	a1, 4(a6)
	sw	a7, 8(a6)
	mulw	a1, t0, a2
	sub	a0, a0, a1
	sw	t3, 12(a6)
	ld	ra, 56(sp)                      # 8-byte Folded Reload
	ld	s0, 48(sp)                      # 8-byte Folded Reload
	addi	sp, sp, 64
	ret
.LBB6_12:
.Lpcrel_hi29:
	auipc	a0, %pcrel_hi(.L.str.11)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi29)
	addi	a0, sp, 16
	addi	a2, sp, 15
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
.Ltmp72:
.Lpcrel_hi30:
	auipc	a0, %pcrel_hi(.L__func__._ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE)
	addi	a2, a0, %pcrel_lo(.Lpcrel_hi30)
.Lpcrel_hi31:
	auipc	a0, %pcrel_hi(.L.str.1)
	addi	a3, a0, %pcrel_lo(.Lpcrel_hi31)
	li	a0, -215
	addi	a1, sp, 16
	li	a4, 79
	call	_ZN2cv5errorENS_5Error4CodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcSB_i@plt
.Ltmp73:
# %bb.13:
.LBB6_14:
.Ltmp74:
	ld	a1, 16(sp)
	addi	a2, sp, 32
	mv	s0, a0
	beq	a1, a2, .LBB6_16
# %bb.15:
	mv	a0, a1
	call	_ZdlPv@plt
.LBB6_16:
	mv	a0, s0
	call	_Unwind_Resume@plt
.Lfunc_end6:
	.size	_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE, .Lfunc_end6-_ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE
	.cfi_endproc
	.section	.gcc_except_table._ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE,"a",@progbits
	.p2align	2, 0x0
GCC_except_table6:
.Lexception2:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	3                               # Call site Encoding = udata4
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.word	.Lfunc_begin2-.Lfunc_begin2     # >> Call Site 1 <<
	.word	.Ltmp72-.Lfunc_begin2           #   Call between .Lfunc_begin2 and .Ltmp72
	.word	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.word	.Ltmp72-.Lfunc_begin2           # >> Call Site 2 <<
	.word	.Ltmp73-.Ltmp72                 #   Call between .Ltmp72 and .Ltmp73
	.word	.Ltmp74-.Lfunc_begin2           #     jumps to .Ltmp74
	.byte	0                               #   On action: cleanup
	.word	.Ltmp73-.Lfunc_begin2           # >> Call Site 3 <<
	.word	.Lfunc_end6-.Ltmp73             #   Call between .Ltmp73 and .Lfunc_end6
	.word	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end2:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text.__clang_call_terminate,"axG",@progbits,__clang_call_terminate,comdat
	.hidden	__clang_call_terminate          # -- Begin function __clang_call_terminate
	.weak	__clang_call_terminate
	.p2align	1
	.type	__clang_call_terminate,@function
__clang_call_terminate:                 # @__clang_call_terminate
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sd	ra, 8(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	call	__cxa_begin_catch@plt
	call	_ZSt9terminatev@plt
.Lfunc_end7:
	.size	__clang_call_terminate, .Lfunc_end7-__clang_call_terminate
	.cfi_endproc
                                        # -- End function
	.type	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE31__cv_trace_location_extra_fn324,@object # @_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE31__cv_trace_location_extra_fn324
	.section	.bss._ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE31__cv_trace_location_extra_fn324,"aw",@nobits
	.p2align	3, 0x0
_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE31__cv_trace_location_extra_fn324:
	.quad	0
	.size	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE31__cv_trace_location_extra_fn324, 8

	.type	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE25__cv_trace_location_fn324,@object # @_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE25__cv_trace_location_fn324
	.section	.data.rel.ro._ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE25__cv_trace_location_fn324,"aw",@progbits
	.p2align	3, 0x0
_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE25__cv_trace_location_fn324:
	.quad	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE31__cv_trace_location_extra_fn324
	.quad	.L.str
	.quad	.L.str.1
	.word	324                             # 0x144
	.word	1                               # 0x1
	.size	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE25__cv_trace_location_fn324, 32

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"void cv::getRectSubPix(InputArray, Size, Point2f, OutputArray, int)"
	.size	.L.str, 68

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"/home/bianbu/rvv-pr30388-redo-20261024/sampler-v3-20261025/source/modules/imgproc/src/samplers.cpp"
	.size	.L.str.1, 99

	.type	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__326,@object # @_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__326
	.section	.data.rel.ro._ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__326,"aw",@progbits
	.p2align	3, 0x0
_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__326:
	.quad	.L.str
	.quad	.L.str.1
	.word	326                             # 0x146
	.word	6                               # 0x6
	.quad	.L.str.2
	.quad	.L.str.3
	.quad	.L.str.4
	.size	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__326, 48

	.type	.L.str.2,@object                # @.str.2
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.2:
	.zero	1
	.size	.L.str.2, 1

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"patchSize.width"
	.size	.L.str.3, 16

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"0"
	.size	.L.str.4, 2

	.type	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__327,@object # @_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__327
	.section	.data.rel.ro._ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__327,"aw",@progbits
	.p2align	3, 0x0
_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__327:
	.quad	.L.str
	.quad	.L.str.1
	.word	327                             # 0x147
	.word	6                               # 0x6
	.quad	.L.str.2
	.quad	.L.str.5
	.quad	.L.str.4
	.size	_ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__327, 48

	.type	.L.str.5,@object                # @.str.5
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.5:
	.asciz	"patchSize.height"
	.size	.L.str.5, 17

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"cn == 1 || cn == 3"
	.size	.L.str.6, 19

	.type	.L__func__._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi,@object # @__func__._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi
.L__func__._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi:
	.asciz	"getRectSubPix"
	.size	.L__func__._ZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEi, 14

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"Unsupported combination of input and output formats"
	.size	.L.str.8, 52

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"dims <= 2"
	.size	.L.str.9, 10

	.type	.L__func__._ZNK2cv8MatShapeclEv,@object # @__func__._ZNK2cv8MatShapeclEv
.L__func__._ZNK2cv8MatShapeclEv:
	.asciz	"operator()"
	.size	.L__func__._ZNK2cv8MatShapeclEv, 11

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"/home/bianbu/rvv-pr30388-redo-20261024/reference/modules/core/include/opencv2/core/mat.inl.hpp"
	.size	.L.str.10, 95

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"rect.width <= win_size.width"
	.size	.L.str.11, 29

	.type	.L__func__._ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE,@object # @__func__._ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE
.L__func__._ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE:
	.asciz	"adjustRect"
	.size	.L__func__._ZN2cvL10adjustRectEPKhmiNS_5Size_IiEES3_NS_6Point_IiEEPNS_5Rect_IiEE, 11

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"basic_string: construction from null is not valid"
	.size	.L.str.12, 50

	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"Ubuntu clang version 17.0.2 (1~exp1ubuntu2.1-bb1 56277be0dfab53becb87b035a858df5a10632457)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE31__cv_trace_location_extra_fn324
	.addrsig_sym _ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE25__cv_trace_location_fn324
	.addrsig_sym _ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__326
	.addrsig_sym _ZZN2cv13getRectSubPixERKNS_11_InputArrayENS_5Size_IiEENS_6Point_IfEERKNS_12_OutputArrayEiE15__cv_check__327
