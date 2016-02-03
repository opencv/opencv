/*
 * Copyrigth (c) 2006, Mónica Díez, LPI-UVA, Spain
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "opj_includes.h"

/** @defgroup T1_3D T1_3D - Implementation of the tier-1 coding */
/*@{*/

/** @name Local static functions */
/*@{*/

static int t1_3d_getctxno_zc(unsigned int f, int orient);
static int t1_3d_getctxno_sc(unsigned int f);
static int t1_3d_getctxno_mag(unsigned int f, int fsvr);
static int t1_3d_getspb(unsigned int f);
static int t1_3d_getnmsedec_sig(opj_t1_3d_t *t1, int x, int bitpos);
static int t1_3d_getnmsedec_ref(opj_t1_3d_t *t1, int x, int bitpos);
static void t1_3d_updateflags(unsigned int *fp, int s);
/**
Encode significant pass
*/
static void t1_3d_enc_sigpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int bpno, int one, int *nmsedec, char type, int vsc);
/**
Decode significant pass
*/
static void t1_3d_dec_sigpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int oneplushalf, char type, int vsc);
/**
Encode significant pass
*/
static void t1_3d_enc_sigpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, char type, int cblksty);
/**
Decode significant pass
*/
static void t1_3d_dec_sigpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, char type, int cblksty);
/**
Encode refinement pass
*/
static void t1_3d_enc_refpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int bpno, int one, int *nmsedec, char type, int vsc);
/**
Decode refinement pass
*/
static void t1_3d_dec_refpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int poshalf, int neghalf, char type, int vsc);
/**
Encode refinement pass
*/
static void t1_3d_enc_refpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int *nmsedec, char type, int cblksty);
/**
Decode refinement pass
*/
static void t1_3d_dec_refpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, char type, int cblksty);
/**
Encode clean-up pass
*/
static void t1_3d_enc_clnpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int bpno, int one, int *nmsedec, int partial, int vsc);
/**
Decode clean-up pass
*/
static void t1_3d_dec_clnpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int oneplushalf, int partial, int vsc);
/**
Encode clean-up pass
*/
static void t1_3d_enc_clnpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, int cblksty);
/**
Decode clean-up pass
*/
static void t1_3d_dec_clnpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, int cblksty);
/**
Encode 1 code-block
@param t1 T1 handle
@param cblk Code-block coding parameters
@param orient
@param compno Component number
@param level[3]
@param dwtid[3]
@param stepsize
@param cblksty Code-block style
@param numcomps
@param tile
*/
static void t1_3d_encode_cblk(opj_t1_3d_t *t1, opj_tcd_cblk_t * cblk, int orient, int compno,  int level[3], int dwtid[3], double stepsize, int cblksty, int numcomps, opj_tcd_tile_t * tile);
/**
Decode 1 code-block
@param t1 T1 handle
@param cblk Code-block coding parameters
@param orient
@param roishift Region of interest shifting value
@param cblksty Code-block style
*/
static void t1_3d_decode_cblk(opj_t1_3d_t *t1, opj_tcd_cblk_t * cblk, int orient, int roishift, int cblksty);
static int t1_3d_init_ctxno_zc(unsigned int f, int orient);
static int t1_3d_init_ctxno_sc(unsigned int f);
static int t1_3d_init_ctxno_mag(unsigned int f, int f2);
static int t1_3d_init_spb(unsigned int f);
/**
Initialize the look-up tables of the Tier-1 coder/decoder
@param t1 T1 handle
*/
static void t1_3d_init_luts(opj_t1_3d_t *t1);

/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */

static int t1_3d_getctxno_zc(unsigned int f, int orient) {
	return t1_3d_init_ctxno_zc((f & T1_3D_SIG_OTH), orient);
}

static int t1_3d_getctxno_sc(unsigned int f) {
	return t1_3d_init_ctxno_sc((f & T1_3D_SIG_PRIM) | ((f >> 16) & T1_3D_SGN));
	//return t1->lut_ctxno_sc[((f & T1_3D_SIG_PRIM) | ((f >> 16) & T1_3D_SGN)) >> 4];
}

static int t1_3d_getctxno_mag(unsigned int f, int fsvr) {
	return t1_3d_init_ctxno_mag((f & T1_3D_SIG_OTH), fsvr);
}

static int t1_3d_getspb(unsigned int f) {
	return t1_3d_init_spb((f & T1_3D_SIG_PRIM) | ((f >> 16) & T1_3D_SGN));
	//return t1->lut_spb[((f & T1_3D_SIG_PRIM) | ((f >> 16) & T1_3D_SGN)) >> 4];
}

static int t1_3d_getnmsedec_sig(opj_t1_3d_t *t1, int x, int bitpos) {
	if (bitpos > T1_NMSEDEC_FRACBITS) {
		return t1->lut_nmsedec_sig[(x >> (bitpos - T1_NMSEDEC_FRACBITS)) & ((1 << T1_NMSEDEC_BITS) - 1)];
	}
	
	return t1->lut_nmsedec_sig0[x & ((1 << T1_NMSEDEC_BITS) - 1)];
}

static int t1_3d_getnmsedec_ref(opj_t1_3d_t *t1, int x, int bitpos) {
	if (bitpos > T1_NMSEDEC_FRACBITS) {
		return t1->lut_nmsedec_ref[(x >> (bitpos - T1_NMSEDEC_FRACBITS)) & ((1 << T1_NMSEDEC_BITS) - 1)];
	}

    return t1->lut_nmsedec_ref0[x & ((1 << T1_NMSEDEC_BITS) - 1)];
}

static void t1_3d_updateflags(unsigned int *fp, int s) {
	unsigned int *np = fp - (T1_MAXCBLKW + 2);
	unsigned int *sp = fp + (T1_MAXCBLKW + 2);

	unsigned int *bwp = fp + ((T1_MAXCBLKW + 2)*(T1_MAXCBLKH +2));
	unsigned int *bnp = bwp - (T1_MAXCBLKW + 2);
	unsigned int *bsp = bwp + (T1_MAXCBLKW + 2);
	
	unsigned int *fwp = fp - ((T1_MAXCBLKW + 2)*(T1_MAXCBLKH +2));
	unsigned int *fnp = fwp - (T1_MAXCBLKW + 2);
	unsigned int *fsp = fwp + (T1_MAXCBLKW + 2);

	np[-1] |= T1_3D_SIG_SE;
	np[1] |= T1_3D_SIG_SW;
	sp[-1] |= T1_3D_SIG_NE;
	sp[1] |= T1_3D_SIG_NW;
	*np |= T1_3D_SIG_S;
	*sp |= T1_3D_SIG_N;
	fp[-1] |= T1_3D_SIG_E;
	fp[1] |= T1_3D_SIG_W;

	*fwp |= T1_3D_SIG_FC;
	*bwp |= T1_3D_SIG_BC;

	fnp[-1] |= T1_3D_SIG_FSE;
	fnp[1] |= T1_3D_SIG_FSW;
	fsp[-1] |= T1_3D_SIG_FNE;
	fsp[1] |= T1_3D_SIG_FNW;
	*fnp |= T1_3D_SIG_FS;
	*fsp |= T1_3D_SIG_FN;
	fwp[-1] |= T1_3D_SIG_FE;
	fwp[1] |= T1_3D_SIG_FW;

	bnp[-1] |= T1_3D_SIG_BSE;
	bnp[1] |= T1_3D_SIG_BSW;
	bsp[-1] |= T1_3D_SIG_BNE;
	bsp[1] |= T1_3D_SIG_BNW;
	*bnp |= T1_3D_SIG_BS;
	*bsp |= T1_3D_SIG_BN;
	bwp[-1] |= T1_3D_SIG_BE;
	bwp[1] |= T1_3D_SIG_BW;

	if (s) {
		*np |= (T1_3D_SGN_S << 16);
		*sp |= (T1_3D_SGN_N << 16);
		fp[-1] |= (T1_3D_SGN_E << 16);
		fp[1] |= (T1_3D_SGN_W << 16);
		*fwp |= (T1_3D_SGN_F << 16);
		*bwp |= (T1_3D_SGN_B << 16);
	}
}

static void t1_3d_enc_sigpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int bpno, int one, int *nmsedec, char type, int vsc) {
	int v, flagsvr;
	unsigned int flag;

	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) : (*fp);
	flagsvr = (*fsvr);
	if ((flag & T1_3D_SIG_OTH) && !(flagsvr & (T1_3D_SIG | T1_3D_VISIT))) {
		v = int_abs(*dp) & one ? 1 : 0;
		if (type == T1_TYPE_RAW) {	/* BYPASS/LAZY MODE */
			mqc_setcurctx(mqc, t1_3d_getctxno_zc(flag, orient));	/* ESSAI */
			mqc_bypass_enc(mqc, v);
		} else {
			mqc_setcurctx(mqc, t1_3d_getctxno_zc(flag, orient));
			mqc_encode(mqc, v);
		}
		if (v) {
			v = *dp < 0 ? 1 : 0;
			*nmsedec +=	t1_3d_getnmsedec_sig(t1, int_abs(*dp), bpno + T1_NMSEDEC_FRACBITS);
			if (type == T1_TYPE_RAW) {	/* BYPASS/LAZY MODE */
				mqc_setcurctx(mqc, t1_3d_getctxno_sc(flag));	/* ESSAI */
				mqc_bypass_enc(mqc, v);
			} else {
				mqc_setcurctx(mqc, t1_3d_getctxno_sc(flag));
				mqc_encode(mqc, v ^ t1_3d_getspb(flag));
			}
			t1_3d_updateflags(fp, v);
			*fsvr |= T1_3D_SIG;
		}
		*fsvr |= T1_3D_VISIT;
	}
}

static void t1_3d_dec_sigpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int oneplushalf, char type, int vsc) {
	int v, flagsvr;
	unsigned int flag;
	
	opj_raw_t *raw = t1->raw;	/* RAW component */
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) : (*fp);
	flagsvr = (*fsvr);
	if ((flag & T1_3D_SIG_OTH) && !(flagsvr & (T1_3D_SIG | T1_3D_VISIT))) {
		if (type == T1_TYPE_RAW) {
			if (raw_decode(raw)) {
				v = raw_decode(raw);	/* ESSAI */
				*dp = v ? -oneplushalf : oneplushalf;
				t1_3d_updateflags(fp, v);
				*fsvr |= T1_3D_SIG;
			}
		} else {
			mqc_setcurctx(mqc, t1_3d_getctxno_zc(flag, orient));
			if (mqc_decode(mqc)) {
				mqc_setcurctx(mqc, t1_3d_getctxno_sc(flag));
				v = mqc_decode(mqc) ^ t1_3d_getspb(flag);
				*dp = v ? -oneplushalf : oneplushalf;
				t1_3d_updateflags(fp, v);
				*fsvr |= T1_3D_SIG;
			}
		}
		*fsvr |= T1_3D_VISIT;
	}
}				/* VSC and  BYPASS by Antonin */

static void t1_3d_enc_sigpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, char type, int cblksty) {
	int i, j, k, m, one, vsc;
	*nmsedec = 0;
	one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
	for (m = 0; m < l; m++) {
		for (k = 0; k < h; k += 4) {
			for (i = 0; i < w; i++) {
				for (j = k; j < k + 4 && j < h; j++) {
					vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
					t1_3d_enc_sigpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->flagSVR[1 + m][1 + j][1 + i], &t1->data[m][j][i], orient, bpno, one, nmsedec, type, vsc);
				}
			}
		}
	}
}

static void t1_3d_dec_sigpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, char type, int cblksty) {
	int i, j, k, m, one, half, oneplushalf, vsc;
	one = 1 << bpno;
	half = one >> 1;
	oneplushalf = one | half;
	for (m = 0; m < l; m++) {
		for (k = 0; k < h; k += 4) {
			for (i = 0; i < w; i++) {
				for (j = k; j < k + 4 && j < h; j++) {
					vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
					t1_3d_dec_sigpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->flagSVR[1 + m][1 + j][1 + i], &t1->data[m][j][i], orient, oneplushalf, type, vsc);
				}
			}
		}
	}
}				/* VSC and  BYPASS by Antonin */

static void t1_3d_enc_refpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int bpno, int one, int *nmsedec, char type, int vsc) {
	int v, flagsvr;
	unsigned int flag;

	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) : (*fp);
	flagsvr = (*fsvr);
	if ((flagsvr & (T1_3D_SIG | T1_3D_VISIT)) == T1_3D_SIG) {
		*nmsedec += t1_3d_getnmsedec_ref(t1, int_abs(*dp), bpno + T1_NMSEDEC_FRACBITS);
		v = int_abs(*dp) & one ? 1 : 0;
		if (type == T1_TYPE_RAW) {	/* BYPASS/LAZY MODE */
			mqc_setcurctx(mqc, t1_3d_getctxno_mag(flag, flagsvr));	/* ESSAI */
			mqc_bypass_enc(mqc, v);
		} else {
			mqc_setcurctx(mqc, t1_3d_getctxno_mag(flag, flagsvr));
			mqc_encode(mqc, v);
		}
		*fsvr |= T1_3D_REFINE;
	}
}

static void t1_3d_dec_refpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int poshalf, int neghalf, char type, int vsc) {
	int v, t, flagsvr;
	unsigned int flag;

	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	opj_raw_t *raw = t1->raw;	/* RAW component */
	
	flag = vsc ? ((*fp) & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) : (*fp);
	flagsvr = (*fsvr);
	if ((flagsvr & (T1_3D_SIG | T1_3D_VISIT)) == T1_3D_SIG) {
		if (type == T1_TYPE_RAW) {
			mqc_setcurctx(mqc, t1_3d_getctxno_mag(flag, flagsvr));	/* ESSAI */
			v = raw_decode(raw);
		} else {
			mqc_setcurctx(mqc, t1_3d_getctxno_mag(flag, flagsvr));
			v = mqc_decode(mqc);
		}
		t = v ? poshalf : neghalf;
		*dp += *dp < 0 ? -t : t;
		*fsvr |= T1_3D_REFINE;
	}
}				/* VSC and  BYPASS by Antonin  */

static void t1_3d_enc_refpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int *nmsedec, char type, int cblksty) {
	int i, j, k, m, one, vsc;
	*nmsedec = 0;
	one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
	for (m = 0; m < l; m++){
		for (k = 0; k < h; k += 4) {
			for (i = 0; i < w; i++) {
				for (j = k; j < k + 4 && j < h; j++) {
					vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
					t1_3d_enc_refpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->flagSVR[1 + m][1 + j][1 + i], &t1->data[m][j][i], bpno, one, nmsedec, type, vsc);
				}
			}
		}
	}	
}

static void t1_3d_dec_refpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, char type, int cblksty) {
	int i, j, k, m, one, poshalf, neghalf;
	int vsc;
	one = 1 << bpno;
	poshalf = one >> 1;
	neghalf = bpno > 0 ? -poshalf : -1;
	for (m = 0; m < l; m++) {
		for (k = 0; k < h; k += 4) {
			for (i = 0; i < w; i++) {
				for (j = k; j < k + 4 && j < h; j++) {
					vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
					t1_3d_dec_refpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->flagSVR[1 + m][1 + j][1 + i], &t1->data[m][j][i], poshalf, neghalf, type, vsc);
				}
			}
		}
	}
}				/* VSC and  BYPASS by Antonin */

static void t1_3d_enc_clnpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int bpno, int one, int *nmsedec, int partial, int vsc) {
	int v, flagsvr;
	unsigned int flag;

	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) : (*fp);
	flagsvr = (*fsvr);
	if (partial) {
		goto LABEL_PARTIAL;
	}
	if (!(*fsvr & (T1_3D_SIG | T1_3D_VISIT))) {
		mqc_setcurctx(mqc, t1_3d_getctxno_zc(flag, orient));
		v = int_abs(*dp) & one ? 1 : 0;
		mqc_encode(mqc, v);
		if (v) {
LABEL_PARTIAL:
			*nmsedec += t1_3d_getnmsedec_sig(t1, int_abs(*dp), bpno + T1_NMSEDEC_FRACBITS);
			mqc_setcurctx(mqc, t1_3d_getctxno_sc(flag));
			v = *dp < 0 ? 1 : 0;
			mqc_encode(mqc, v ^ t1_3d_getspb(flag));
			t1_3d_updateflags(fp, v);
			*fsvr |= T1_3D_SIG;
		}
	}
	*fsvr &= ~T1_3D_VISIT;
}

static void t1_3d_dec_clnpass_step(opj_t1_3d_t *t1, unsigned int *fp, int *fsvr, int *dp, int orient, int oneplushalf, int partial, int vsc) {
	int v, flagsvr;
	unsigned int flag;

	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) : (*fp);
	flagsvr = (*fsvr);
	if (partial) {
		goto LABEL_PARTIAL;
	}
	if (!(flagsvr & (T1_3D_SIG | T1_3D_VISIT))) {
		mqc_setcurctx(mqc, t1_3d_getctxno_zc(flag, orient));
		if (mqc_decode(mqc)) {
LABEL_PARTIAL:
			mqc_setcurctx(mqc, t1_3d_getctxno_sc(flag));
			v = mqc_decode(mqc) ^ t1_3d_getspb(flag);
			*dp = v ? -oneplushalf : oneplushalf;
			t1_3d_updateflags(fp, v);
			*fsvr |= T1_3D_SIG;
		}
	}
	*fsvr &= ~T1_3D_VISIT;
}				/* VSC and  BYPASS by Antonin */

static void t1_3d_enc_clnpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, int cblksty) {
	int i, j, k, m, one, agg, runlen, vsc;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	*nmsedec = 0;
	one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
	for (m = 0; m < l; m++) {
		for (k = 0; k < h; k += 4) {
			for (i = 0; i < w; i++) {
				if (k + 3 < h) {
					if (cblksty & J3D_CCP_CBLKSTY_VSC) {
						agg = !( ((t1->flagSVR[1 + m][1 + k][1 + i] | (T1_3D_SIG | T1_3D_VISIT)) & (t1->flags[1 + m][1 + k][1 + i] & T1_3D_SIG_OTH))
							||   ((t1->flagSVR[1 + m][1 + k + 1][1 + i] | (T1_3D_SIG | T1_3D_VISIT)) & (t1->flags[1 + m][1 + k + 1][1 + i] & T1_3D_SIG_OTH))
							||   ((t1->flagSVR[1 + m][1 + k + 2][1 + i] | (T1_3D_SIG | T1_3D_VISIT)) & (t1->flags[1 + m][1 + k + 2][1 + i] & T1_3D_SIG_OTH))
							||   ((t1->flagSVR[1 + m][1 + k + 3][1 + i] | (T1_3D_SIG | T1_3D_VISIT)) & ((t1->flags[1 + m][1 + k + 3][1 + i] & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) & (T1_3D_SIG_OTH)))
							);
					} else {
						agg = !(
							((t1->flagSVR[1 + m][1 + k][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 1][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 1][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 2][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 2][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 3][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 3][1 + i] & T1_3D_SIG_OTH))
							);
					}
				} else {
					agg = 0;
				}
				if (agg) {
					for (runlen = 0; runlen < 4; runlen++) {
						if (int_abs(t1->data[m][k + runlen][i]) & one)
							break;
					}
					mqc_setcurctx(mqc, T1_CTXNO_AGG);
					mqc_encode(mqc, runlen != 4);
					if (runlen == 4) {
						continue;
					}
					mqc_setcurctx(mqc, T1_CTXNO_UNI);
					mqc_encode(mqc, runlen >> 1);
					mqc_encode(mqc, runlen & 1);
				} else {
					runlen = 0;
				}
				for (j = k + runlen; j < k + 4 && j < h; j++) {
					vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
					t1_3d_enc_clnpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->flagSVR[1 + m][1 + j][1 + i], &t1->data[m][j][i], orient, bpno, one, nmsedec, agg && (j == k + runlen), vsc);
				}
			}
		}
	}
}

static void t1_3d_dec_clnpass(opj_t1_3d_t *t1, int w, int h, int l, int bpno, int orient, int cblksty) {
	int i, j, k, m, one, half, oneplushalf, agg, runlen, vsc;
	int segsym = cblksty & J3D_CCP_CBLKSTY_SEGSYM;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	one = 1 << bpno;
	half = one >> 1;
	oneplushalf = one | half;
	for (m = 0; m < l; m++) {
		for (k = 0; k < h; k += 4) {
			for (i = 0; i < w; i++) {
				if (k + 3 < h) {
					if (cblksty & J3D_CCP_CBLKSTY_VSC) {
						agg = !(
							((t1->flagSVR[1 + m][1 + k][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 1][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 1][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 2][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 2][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 3][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | ((t1->flags[1 + m][1 + k + 3][1 + i] & (~(T1_3D_SIG_S | T1_3D_SIG_SE | T1_3D_SIG_SW | (T1_3D_SGN_S << 16)))) & (T1_3D_SIG_OTH)))
							);
					} else {
						agg = !(
							((t1->flagSVR[1 + m][1 + k][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 1][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 1][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 2][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 2][1 + i] & T1_3D_SIG_OTH))
							|| ((t1->flagSVR[1 + m][1 + k + 3][1 + i] & (T1_3D_SIG | T1_3D_VISIT)) | (t1->flags[1 + m][1 + k + 3][1 + i] & T1_3D_SIG_OTH))
							);
					}
				} else {
					agg = 0;
				}
				if (agg) {
					mqc_setcurctx(mqc, T1_CTXNO_AGG);
					if (!mqc_decode(mqc)) {
						continue;
					}
					mqc_setcurctx(mqc, T1_CTXNO_UNI);
					runlen = mqc_decode(mqc);
					runlen = (runlen << 1) | mqc_decode(mqc);
				} else {
					runlen = 0;
				}
				for (j = k + runlen; j < k + 4 && j < h; j++) {
					vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
					t1_3d_dec_clnpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->flagSVR[1 + m][1 + j][1 + i], &t1->data[m][j][i], orient, oneplushalf, agg && (j == k + runlen), vsc);
				}
			}
		}
	}
	if (segsym) {
		int v = 0;
		mqc_setcurctx(mqc, T1_CTXNO_UNI);
		v = mqc_decode(mqc);
		v = (v << 1) | mqc_decode(mqc);
		v = (v << 1) | mqc_decode(mqc);
		v = (v << 1) | mqc_decode(mqc);
		/*
		if (v!=0xa) {
			opj_event_msg(t1->cinfo, EVT_WARNING, "Bad segmentation symbol %x\n", v);
		} 
		*/
	}
}				/* VSC and  BYPASS by Antonin */


static void t1_3d_encode_cblk(opj_t1_3d_t *t1, opj_tcd_cblk_t * cblk, int orient, int compno, int level[3], int dwtid[3], double stepsize, int cblksty, int numcomps, opj_tcd_tile_t * tile) {
	int i, j, k;
	int w, h, l;
	int passno;
	int bpno, passtype;
	int max;
	int nmsedec = 0;
	double cumwmsedec = 0;
	char type = T1_TYPE_MQ;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	w = cblk->x1 - cblk->x0;
	h = cblk->y1 - cblk->y0;
	l = cblk->z1 - cblk->z0;

	max = 0;
	for (k = 0; k < l; k++) {
		for (j = 0; j < h; j++) {
			for (i = 0; i < w; i++) {
				max = int_max(max, int_abs(t1->data[k][j][i]));
			}
		}
	}
	for (k = 0; k <= l; k++) {
		for (j = 0; j <= h; j++) {
			for (i = 0; i <= w; i++) {
				t1->flags[k][j][i] = 0;
				t1->flagSVR[k][j][i] = 0;
			}
		}
	}
	
	cblk->numbps = max ? (int_floorlog2(max) + 1) - T1_NMSEDEC_FRACBITS : 0;
	
	bpno = cblk->numbps - 1;
	passtype = 2;
	
	mqc_reset_enc(mqc);
	mqc_init_enc(mqc, cblk->data);
	
	for (passno = 0; bpno >= 0; passno++) {
		opj_tcd_pass_t *pass = &cblk->passes[passno];
		int correction = 3;
		double tmpwmsedec;
		type = ((bpno < (cblk->numbps - 4)) && (passtype < 2) && (cblksty & J3D_CCP_CBLKSTY_LAZY)) ? T1_TYPE_RAW : T1_TYPE_MQ;
		
		switch (passtype) {
			case 0:
				t1_3d_enc_sigpass(t1, w, h, l, bpno, orient, &nmsedec, type, cblksty);
				break;
			case 1:
				t1_3d_enc_refpass(t1, w, h, l, bpno, &nmsedec, type, cblksty);
				break;
			case 2:
				t1_3d_enc_clnpass(t1, w, h, l, bpno, orient, &nmsedec, cblksty);
				/* code switch SEGMARK (i.e. SEGSYM) */
				if (cblksty & J3D_CCP_CBLKSTY_SEGSYM)
					mqc_segmark_enc(mqc);
				break;
		}
		
		/* fixed_quality */
		tmpwmsedec = t1_getwmsedec(nmsedec, compno, level, orient, bpno, stepsize, numcomps, dwtid);
		cumwmsedec += tmpwmsedec;
		tile->distotile += tmpwmsedec;
		
		/* Code switch "RESTART" (i.e. TERMALL) */
		if ((cblksty & J3D_CCP_CBLKSTY_TERMALL)	&& !((passtype == 2) && (bpno - 1 < 0))) {
			if (type == T1_TYPE_RAW) {
				mqc_flush(mqc);
				correction = 1;
				/* correction = mqc_bypass_flush_enc(); */
			} else {			/* correction = mqc_restart_enc(); */
				mqc_flush(mqc);
				correction = 1;
			}
			pass->term = 1;
		} else {
			if (((bpno < (cblk->numbps - 4) && (passtype > 0)) 
				|| ((bpno == (cblk->numbps - 4)) && (passtype == 2))) && (cblksty & J3D_CCP_CBLKSTY_LAZY)) {
				if (type == T1_TYPE_RAW) {
					mqc_flush(mqc);
					correction = 1;
				} else {	
					mqc_flush(mqc);
					correction = 1;
				}
				pass->term = 1;
			} else {
				pass->term = 0;
			}
		}
		
		if (++passtype == 3) {
			passtype = 0;
			bpno--;
		}
		
		if (pass->term && bpno > 0) {
			type = ((bpno < (cblk->numbps - 4)) && (passtype < 2) && (cblksty & J3D_CCP_CBLKSTY_LAZY)) ? T1_TYPE_RAW : T1_TYPE_MQ;
			if (type == T1_TYPE_RAW)
				mqc_bypass_init_enc(mqc);
			else
				mqc_restart_init_enc(mqc);
		}
		
		pass->distortiondec = cumwmsedec;
		pass->rate = mqc_numbytes(mqc) + correction;	/* FIXME */
		pass->len = pass->rate - (passno == 0 ? 0 : cblk->passes[passno - 1].rate);

		/* Code-switch "RESET" */
		if (cblksty & J3D_CCP_CBLKSTY_RESET)
			mqc_reset_enc(mqc);
	}
	
	/* Code switch "ERTERM" (i.e. PTERM) */
	if (cblksty & J3D_CCP_CBLKSTY_PTERM)
		mqc_erterm_enc(mqc);
	else /* Default coding */ if (!(cblksty & J3D_CCP_CBLKSTY_LAZY))
		mqc_flush(mqc);
	
	cblk->totalpasses = passno;
}

static void t1_3d_decode_cblk(opj_t1_3d_t *t1, opj_tcd_cblk_t * cblk, int orient, int roishift, int cblksty) {
	int i, j, k;
	int w, h, l;
	int bpno, passtype;
	int segno, passno;
	char type = T1_TYPE_MQ; /* BYPASS mode */
	opj_raw_t *raw = t1->raw;	/* RAW component */
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	w = cblk->x1 - cblk->x0;
	h = cblk->y1 - cblk->y0;
	l = cblk->z1 - cblk->z0;

    for (k = 0; k < l; k++) {
		for (j = 0; j < h; j++) {
			for (i = 0; i < w; i++) {
				t1->data[k][j][i] = 0;
			}
		}
	}
	
	for (k = 0; k <= l; k++) {
		for (j = 0; j <= h; j++) {
			for (i = 0; i <= w; i++) {
				t1->flags[k][j][i] = 0;
				t1->flagSVR[k][j][i] = 0;
			}
		}
	}

	
	bpno = roishift + cblk->numbps - 1;
	passtype = 2;
	
	mqc_reset_enc(mqc);
	
	for (segno = 0; segno < cblk->numsegs; segno++) {
		opj_tcd_seg_t *seg = &cblk->segs[segno];
		
		/* BYPASS mode */
		type = ((bpno <= (cblk->numbps - 1) - 4) && (passtype < 2) && (cblksty & J3D_CCP_CBLKSTY_LAZY)) ? T1_TYPE_RAW : T1_TYPE_MQ;
		if (type == T1_TYPE_RAW) {
			raw_init_dec(raw, seg->data, seg->len);
		} else {
			mqc_init_dec(mqc, seg->data, seg->len);
		}
		
		for (passno = 0; passno < seg->numpasses; passno++) {
			switch (passtype) {
				case 0:
					t1_3d_dec_sigpass(t1, w, h, l, bpno+1, orient, type, cblksty);
					break;
				case 1:
					t1_3d_dec_refpass(t1, w, h, l, bpno+1, type, cblksty);
					break;
				case 2:
					t1_3d_dec_clnpass(t1, w, h, l, bpno+1, orient, cblksty);
					break;
			}
			
			if ((cblksty & J3D_CCP_CBLKSTY_RESET) && type == T1_TYPE_MQ) {
				mqc_reset_enc(mqc);
			}
			if (++passtype == 3) {
				passtype = 0;
				bpno--;
			}

		}
	}
}

static int t1_3d_init_ctxno_zc(unsigned int f, int orient) {
	unsigned int h, v, c;
	unsigned int d2xy, d2xz, d2yz, d3;
	int n;
	unsigned int hvc, hc, d2, d2xy2yz, d2xy2xz;
	n = 0;
	h = ((f & T1_3D_SIG_W) != 0) + ((f & T1_3D_SIG_E) != 0);
	v = ((f & T1_3D_SIG_N) != 0) + ((f & T1_3D_SIG_S) != 0);
	c = ((f & T1_3D_SIG_FC) != 0) + ((f & T1_3D_SIG_BC) != 0);
	d2xy = ((f & T1_3D_SIG_NW) != 0) + ((f & T1_3D_SIG_NE) != 0) + ((f & T1_3D_SIG_SE) != 0) + ((f & T1_3D_SIG_SW) != 0);
	d2xz = ((f & T1_3D_SIG_FW) != 0) + ((f & T1_3D_SIG_BW) != 0) + ((f & T1_3D_SIG_FE) != 0) + ((f & T1_3D_SIG_BE) != 0);
	d2yz = ((f & T1_3D_SIG_FN) != 0) + ((f & T1_3D_SIG_FS) != 0) + ((f & T1_3D_SIG_BN) != 0) + ((f & T1_3D_SIG_BS) != 0);
    d3 = ((f & T1_3D_SIG_FNW) != 0) + ((f & T1_3D_SIG_FNE) != 0) + ((f & T1_3D_SIG_FSE) != 0) + ((f & T1_3D_SIG_FSW) != 0) 
		+ ((f & T1_3D_SIG_BNW) != 0) + ((f & T1_3D_SIG_BNE) != 0) + ((f & T1_3D_SIG_BSE) != 0) + ((f & T1_3D_SIG_BSW) != 0);
	
	switch (orient) {
		case 0: //LLL
		case 7: //HHH
			hvc = h + v + c;
			d2 = d2xy + d2xz + d2yz;
			if (!hvc) {
				if (!d2) {
                    n = (!d3) ? 0 : 1;
				} else if (d2 == 1) {
					n = (!d3) ? 2 : 3;
				} else {
					n = (!d3) ? 4 : 5;
				}
			} else if (hvc == 1) {
				if (!d2) {
                    n = (!d3) ? 6 : 7;
				} else if (d2 == 1) {
					n = (!d3) ? 8 : 9;
				} else {
					n = 10;
				}
			} else if (hvc == 2) {
				if (!d2) {
                    n = (!d3) ? 11 : 12;
				} else {
					n = 13;
				}
			} else if (hvc == 3) {
				n = 14;
			} else {
				n = 15;
			}
			break;
		//LHL, HLL, LLH
		case 1:
		case 2:
		case 4:
			hc = h + c;
			d2xy2yz = d2xy + d2yz;
            if (!hc) {
				if (!v) {
					if (!d2xy) {
						n = (!d2xy2yz) ? ((!d3) ? 0 : 1) : ((!d3) ? 2 : 3);	
					} else if (d2xy == 1) {
						n = (!d2xy2yz) ? ((!d3) ? 4 : 5) : 6;	
					} else { //>=2
                        n = 7;
					}
				} else {
					n = (v == 1) ? 8 : 9; // =1 or =2
				} 
			} else if (hc == 1) {
				n = (!v) ? ( (!d2xy) ? ( (!d2xy2yz) ? ( (!d3) ? 10 : 11) : (12) ) : (13) ) : (14);
			} else { //if (hc >= 2)
				n = 15;
			}
			break;
		//HLH, HHL, LHH
		case 3:
		case 5:
		case 6:
			hc = h + c;
			d2xy2xz = d2xy + d2xz;
			if (!v) {
				if (!d2xz) {
					if (!hc && !d2xy2xz) {
						n = (!d3) ? 0 : 1;
					} else if (hc == 1) {
						n = (!d2xy2xz) ?  2 : 3;
					} else { //if >= 2
						n = 4;
					}
				} else if ( d2xz>=1 && !hc ) {
					n = 5;
				} else if ( hc>=1 ) {
					n = (d2xz==1) ? 6 : 7;
				} 
			} else if (v == 1) {
				if (!d2xz) {
					n = (!hc) ? 8 : 9;
				} else if (d2xz == 1) {
					n = (!hc) ? 10 : 11;
				} else if (d2xz == 2) {
					n = (!hc) ? 12 : 13;
				} else { // if (d2xz >= 3) {
					n = 14;
				}
			} else if (v == 2) {
				n = 15;
			} 
			break;
	}
	
	return (T1_3D_CTXNO_ZC + n);
}

static int t1_3d_init_ctxno_sc(unsigned int f) {
	int hc, vc, cc;
	int n = 0;

	hc = int_min( ( (f & (T1_3D_SIG_E | T1_3D_SGN_E)) == T1_3D_SIG_E ) 
					+ ( (f & (T1_3D_SIG_W | T1_3D_SGN_W)) == T1_3D_SIG_W ) , 1) 
		- int_min( ( (f & (T1_3D_SIG_E | T1_3D_SGN_E)) == (T1_3D_SIG_E | T1_3D_SGN_E) ) 
					+ ( (f & (T1_3D_SIG_W | T1_3D_SGN_W) ) == (T1_3D_SIG_W | T1_3D_SGN_W)), 1);
	
	vc = int_min(((f & (T1_3D_SIG_N | T1_3D_SGN_N)) == T1_3D_SIG_N) 
					+ ((f & (T1_3D_SIG_S | T1_3D_SGN_S)) == T1_3D_SIG_S), 1) 
		- int_min(((f & (T1_3D_SIG_N | T1_3D_SGN_N)) == (T1_3D_SIG_N | T1_3D_SGN_N)) 
					+ ((f & (T1_3D_SIG_S | T1_3D_SGN_S)) == (T1_3D_SIG_S | T1_3D_SGN_S)), 1);
	
	cc = int_min(((f & (T1_3D_SIG_FC | T1_3D_SGN_F)) == T1_3D_SIG_FC) 
					+ ((f & (T1_3D_SIG_BC | T1_3D_SGN_B)) == T1_3D_SIG_BC), 1) 
		- int_min(((f & (T1_3D_SIG_FC | T1_3D_SGN_F)) == (T1_3D_SIG_FC | T1_3D_SGN_F)) 
					+ ((f & (T1_3D_SIG_BC | T1_3D_SGN_B)) == (T1_3D_SIG_BC | T1_3D_SGN_B)), 1);
	if (hc < 0) {
		hc = -hc;
		vc = -vc;
		cc = -cc;
	}

	if (!hc) {
		if (!vc) 
			n = (!cc) ? 0 : 1;
		else if (vc == -1)
			n = (!cc) ? 1 : ( (cc>0) ? 2 : 4);
		else if (vc == 1)
			n = (!cc) ? 1 : ( (cc<0) ? 2 : 4);
	} else if (hc == 1) {
		if (!vc)
			n = (!cc) ? 1 : ( (cc<0) ? 2 : 4);
		else if (vc == 1)
			n = (!cc) ? 4 : ( (cc>0) ? 5 : 3);
		else if (vc == -1)
			n = (!cc) ? 2 : 3;
	} else if (hc == -1) {
		if (!vc)
			n = (!cc) ? 1 : ( (cc>0) ? 2 : 4);
		else if (vc == 1)
			n = (!cc) ? 2 : 3;
		else if (vc == -1)
			n = (!cc) ? 4 : ( (cc<0) ? 5 : 3);
	}
	
	return (T1_3D_CTXNO_SC + n);
}

static int t1_3d_init_ctxno_mag(unsigned int f, int f2) {
	int n;
	if (!(f2 & T1_3D_REFINE))	//First refinement for this coefficient (no previous refinement)
		n = (f & (T1_3D_SIG_PRIM)) ? 1 : 0;
	else
		n = 2;
	
	return (T1_3D_CTXNO_MAG + n);
}

static int t1_3d_init_spb(unsigned int f) {
	int hc, vc, cc;
	int n = 0;
	
	hc = int_min( ( (f & (T1_3D_SIG_E | T1_3D_SGN_E)) == T1_3D_SIG_E ) 
					+ ( (f & (T1_3D_SIG_W | T1_3D_SGN_W)) == T1_3D_SIG_W ) , 1) 
		- int_min( ( (f & (T1_3D_SIG_E | T1_3D_SGN_E)) == (T1_3D_SIG_E | T1_3D_SGN_E) ) 
					+ ( (f & (T1_3D_SIG_W | T1_3D_SGN_W) ) == (T1_3D_SIG_W | T1_3D_SGN_W)), 1);
	
	vc = int_min(((f & (T1_3D_SIG_N | T1_3D_SGN_N)) == T1_3D_SIG_N) 
					+ ((f & (T1_3D_SIG_S | T1_3D_SGN_S)) == T1_3D_SIG_S), 1) 
		- int_min(((f & (T1_3D_SIG_N | T1_3D_SGN_N)) == (T1_3D_SIG_N | T1_3D_SGN_N)) 
					+ ((f & (T1_3D_SIG_S | T1_3D_SGN_S)) == (T1_3D_SIG_S | T1_3D_SGN_S)), 1);
	
	cc = int_min(((f & (T1_3D_SIG_FC | T1_3D_SGN_F)) == T1_3D_SIG_FC) 
					+ ((f & (T1_3D_SIG_BC | T1_3D_SGN_B)) == T1_3D_SIG_BC), 1) 
		- int_min(((f & (T1_3D_SIG_FC | T1_3D_SGN_F)) == (T1_3D_SIG_FC | T1_3D_SGN_F)) 
					+ ((f & (T1_3D_SIG_BC | T1_3D_SGN_B)) == (T1_3D_SIG_BC | T1_3D_SGN_B)), 1);
	
	n = ((hc + vc + cc) < 0); 
	
	return n;
}

static void t1_3d_init_luts(opj_t1_3d_t *t1) {
	int i;
	double u, v, t;
	/*for (j = 0; j < 4; j++) {
		for (i = 0; i < 256; ++i) {
			t1->lut_ctxno_zc[(j << 8) | i] = t1_3d_init_ctxno_zc(i, j);
		}
	}
	for (i = 0; i < 4096; i++) {
		t1->lut_ctxno_sc[i] = t1_3d_init_ctxno_sc(i << 4);
	}
	for (j = 0; j < 2; j++) {
		for (i = 0; i < 2048; ++i) {
			t1->lut_ctxno_mag[(j << 11) + i] = t1_3d_init_ctxno_mag((j ? T1_3D_REFINE : 0) | i);
		}
	}
	for (i = 0; i < 4096; ++i) {
		t1->lut_spb[i] = t1_3d_init_spb(i << 4);
	}*/
	/* FIXME FIXME FIXME */
	for (i = 0; i < (1 << T1_NMSEDEC_BITS); i++) {
		t = i / pow(2, T1_NMSEDEC_FRACBITS);
		u = t;
		v = t - 1.5;
		t1->lut_nmsedec_sig[i] = 
			int_max(0, 
			(int) (floor((u * u - v * v) * pow(2, T1_NMSEDEC_FRACBITS) + 0.5) / pow(2, T1_NMSEDEC_FRACBITS) * 8192.0));
		t1->lut_nmsedec_sig0[i] =
			int_max(0,
			(int) (floor((u * u) * pow(2, T1_NMSEDEC_FRACBITS) + 0.5) / pow(2, T1_NMSEDEC_FRACBITS) * 8192.0));
		u = t - 1.0;
		if (i & (1 << (T1_NMSEDEC_BITS - 1))) {
			v = t - 1.5;
		} else {
			v = t - 0.5;
		}
		t1->lut_nmsedec_ref[i] =
			int_max(0,
			(int) (floor((u * u - v * v) * pow(2, T1_NMSEDEC_FRACBITS) + 0.5) / pow(2, T1_NMSEDEC_FRACBITS) * 8192.0));
		t1->lut_nmsedec_ref0[i] =
			int_max(0,
			(int) (floor((u * u) * pow(2, T1_NMSEDEC_FRACBITS) + 0.5) / pow(2, T1_NMSEDEC_FRACBITS) * 8192.0));
	}
}

/* ----------------------------------------------------------------------- */

opj_t1_3d_t* t1_3d_create(opj_common_ptr cinfo) {
	opj_t1_3d_t *t1 = (opj_t1_3d_t*)opj_malloc(sizeof(opj_t1_3d_t));
	if(t1) {
		t1->cinfo = cinfo;
		/* create MQC and RAW handles */
		t1->mqc = mqc_create();
		t1->raw = raw_create();
		/* initialize the look-up tables of the Tier-1 coder/decoder */
		t1_3d_init_luts(t1);
	}
	return t1;
}

void t1_3d_destroy(opj_t1_3d_t *t1) {
	if(t1) {
		/* destroy MQC and RAW handles */
		mqc_destroy(t1->mqc);
		raw_destroy(t1->raw);
		opj_free(t1);
	}
}

void t1_3d_encode_cblks(opj_t1_3d_t *t1, opj_tcd_tile_t *tile, opj_tcp_t *tcp) {
	int compno, resno, bandno, precno, cblkno;
	int x, y, z, i, j, k, orient;
	int level[3];
	tile->distotile = 0;		/* fixed_quality */

	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];

		for (resno = 0; resno < tilec->numresolution[0]; resno++) {
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];

			for (bandno = 0; bandno < res->numbands; bandno++) {
				opj_tcd_band_t *band = &res->bands[bandno];

				for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
					opj_tcd_precinct_t *prc = &band->precincts[precno];

					for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
						opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];

						if (band->bandno == 0) {
							x = cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 1) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 2) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 3) {		
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 4) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						} else if (band->bandno == 5) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						} else if (band->bandno == 6) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						} else if (band->bandno == 7) {		
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						}

						if (tcp->tccps[compno].reversible == 1) {
							for (k = 0; k < cblk->z1 - cblk->z0; k++) {
								for (j = 0; j < cblk->y1 - cblk->y0; j++) {
                                    for (i = 0; i < cblk->x1 - cblk->x0; i++) {
                                        t1->data[k][j][i] = 
										tilec->data[(x + i) + (y + j) * (tilec->x1 - tilec->x0) + (z + k) * (tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0)] << T1_NMSEDEC_FRACBITS;
									}
								}
							}
						} else if (tcp->tccps[compno].reversible == 0) {
							for (k = 0; k < cblk->z1 - cblk->z0; k++) {
								for (j = 0; j < cblk->y1 - cblk->y0; j++) {
                                    for (i = 0; i < cblk->x1 - cblk->x0; i++) {
                                        t1->data[k][j][i] = fix_mul(
										tilec->data[(x + i) + (y + j) * (tilec->x1 - tilec->x0) + (z + k) * (tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0)],
										8192 * 8192 / ((int) floor(band->stepsize * 8192))) >> (13 - T1_NMSEDEC_FRACBITS);
									}
								}
							}
						}
						orient = band->bandno;	/* FIXME */
						for (i = 0; i < 3; i++) 
							level[i] = tilec->numresolution[i] - 1 - resno;

						t1_3d_encode_cblk(t1, cblk, orient, compno, level, tcp->tccps[compno].dwtid, band->stepsize, tcp->tccps[compno].cblksty, tile->numcomps, tile);
							
					} /* cblkno */
				} /* precno */
			} /* bandno */
		} /* resno  */
	} /* compno  */
}

void t1_3d_decode_cblks(opj_t1_3d_t *t1, opj_tcd_tile_t *tile, opj_tcp_t *tcp) {
	int compno, resno, bandno, precno, cblkno;
	
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];

		for (resno = 0; resno < tilec->numresolution[0]; resno++) {
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];

			for (bandno = 0; bandno < res->numbands; bandno++) {
				opj_tcd_band_t *band = &res->bands[bandno];

				for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
					opj_tcd_precinct_t *prc = &band->precincts[precno];

					for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
						int x, y, z, i, j, k, orient;
						opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];

						orient = band->bandno;	/* FIXME */

						//fprintf(stdout,"[INFO] t1_3d_decode_cblk(t1, cblk, orient(%d), tcp->tccps[compno].roishift (%d), tcp->tccps[compno].cblksty (%d));\n",orient,tcp->tccps[compno].roishift, tcp->tccps[compno].cblksty);
						t1_3d_decode_cblk(t1, cblk, orient, tcp->tccps[compno].roishift, tcp->tccps[compno].cblksty);

						if (band->bandno == 0) {
							x = cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 1) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 2) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 3) {		
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = cblk->z0 - band->z0;
						} else if (band->bandno == 4) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						} else if (band->bandno == 5) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						} else if (band->bandno == 6) {
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						} else if (band->bandno == 7) {		
							opj_tcd_resolution_t *pres = &tilec->resolutions[resno - 1];
							x = pres->x1 - pres->x0 + cblk->x0 - band->x0;
							y = pres->y1 - pres->y0 + cblk->y0 - band->y0;
							z = pres->z1 - pres->z0 + cblk->z0 - band->z0;
						}
					
						if (tcp->tccps[compno].roishift) {
							int thresh, val, mag;
							thresh = 1 << tcp->tccps[compno].roishift;
							for (k = 0; k < cblk->z1 - cblk->z0; k++) {
								for (j = 0; j < cblk->y1 - cblk->y0; j++) {
									for (i = 0; i < cblk->x1 - cblk->x0; i++) {
										val = t1->data[k][j][i];
										mag = int_abs(val);
										if (mag >= thresh) {
											mag >>= tcp->tccps[compno].roishift;
											t1->data[k][j][i] = val < 0 ? -mag : mag;
										}
									}
								}
							}
						}
						
						if (tcp->tccps[compno].reversible == 1) {
							for (k = 0; k < cblk->z1 - cblk->z0; k++) {
								for (j = 0; j < cblk->y1 - cblk->y0; j++) {
									for (i = 0; i < cblk->x1 - cblk->x0; i++) {
										int tmp = t1->data[k][j][i];
										tilec->data[x + i + (y + j) * (tilec->x1 - tilec->x0) + (z + k) * (tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0)] = tmp/2;
									}
								}
							}
						} else {		/* if (tcp->tccps[compno].reversible == 0) */
							for (k = 0; k < cblk->z1 - cblk->z0; k++) {
								for (j = 0; j < cblk->y1 - cblk->y0; j++) {
									for (i = 0; i < cblk->x1 - cblk->x0; i++) {
										double tmp = (double)(t1->data[k][j][i] * band->stepsize * 4096.0);
										if (t1->data[k][j][i] >> 1 == 0) {
											tilec->data[x + i + (y + j) * (tilec->x1 - tilec->x0) + (z + k) * (tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0)] = 0;
										} else {
											int tmp2 = ((int) (floor(fabs(tmp)))) + ((int) floor(fabs(tmp*2))%2);
											tilec->data[x + i + (y + j) * (tilec->x1 - tilec->x0) + (z + k) * (tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0)] = ((tmp<0)?-tmp2:tmp2);
										}
									}
								}
							}
						}
					} /* cblkno */
				} /* precno */
			} /* bandno */
		} /* resno */
	} /* compno */
}
