/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

/** @defgroup T1 T1 - Implementation of the tier-1 coding */
/*@{*/

/** @name Local static functions */
/*@{*/

static int t1_getctxno_zc(opj_t1_t *t1, int f, int orient);
static int t1_getctxno_sc(opj_t1_t *t1, int f);
static int t1_getctxno_mag(opj_t1_t *t1, int f);
static int t1_getspb(opj_t1_t *t1, int f);
static int t1_getnmsedec_sig(opj_t1_t *t1, int x, int bitpos);
static int t1_getnmsedec_ref(opj_t1_t *t1, int x, int bitpos);
static void t1_updateflags(int *fp, int s);
/**
Encode significant pass
*/
static void t1_enc_sigpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int bpno, int one, int *nmsedec, char type, int vsc);
/**
Decode significant pass
*/
static void t1_dec_sigpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int oneplushalf, char type, int vsc);
/**
Encode significant pass
*/
static void t1_enc_sigpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, char type, int cblksty);
/**
Decode significant pass
*/
static void t1_dec_sigpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, char type, int cblksty);
/**
Encode refinement pass
*/
static void t1_enc_refpass_step(opj_t1_t *t1, int *fp, int *dp, int bpno, int one, int *nmsedec, char type, int vsc);
/**
Decode refinement pass
*/
static void t1_dec_refpass_step(opj_t1_t *t1, int *fp, int *dp, int poshalf, int neghalf, char type, int vsc);
/**
Encode refinement pass
*/
static void t1_enc_refpass(opj_t1_t *t1, int w, int h, int l, int bpno, int *nmsedec, char type, int cblksty);
/**
Decode refinement pass
*/
static void t1_dec_refpass(opj_t1_t *t1, int w, int h, int l, int bpno, char type, int cblksty);
/**
Encode clean-up pass
*/
static void t1_enc_clnpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int bpno, int one, int *nmsedec, int partial, int vsc);
/**
Decode clean-up pass
*/
static void t1_dec_clnpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int oneplushalf, int partial, int vsc);
/**
Encode clean-up pass
*/
static void t1_enc_clnpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, int cblksty);
/**
Decode clean-up pass
*/
static void t1_dec_clnpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, int cblksty);
/**
Encode 1 code-block
@param t1 T1 handle
@param cblk Code-block coding parameters
@param orient
@param compno Component number
@param level
@param dwtid
@param stepsize
@param cblksty Code-block style
@param numcomps
@param tile
*/
static void t1_encode_cblk(opj_t1_t *t1, opj_tcd_cblk_t * cblk, int orient, int compno, int level[3], int dwtid[3], double stepsize, int cblksty, int numcomps, opj_tcd_tile_t * tile);
/**
Decode 1 code-block
@param t1 T1 handle
@param cblk Code-block coding parameters
@param orient
@param roishift Region of interest shifting value
@param cblksty Code-block style
*/
static void t1_decode_cblk(opj_t1_t *t1, opj_tcd_cblk_t * cblk, int orient, int roishift, int cblksty);

static int t1_init_ctxno_zc(int f, int orient);
static int t1_init_ctxno_sc(int f);
static int t1_init_ctxno_mag(int f);
static int t1_init_spb(int f);
/**
Initialize the look-up tables of the Tier-1 coder/decoder
@param t1 T1 handle
*/
static void t1_init_luts(opj_t1_t *t1);

/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */

static int t1_getctxno_zc(opj_t1_t *t1, int f, int orient) {
	return t1->lut_ctxno_zc[(orient << 8) | (f & T1_SIG_OTH)];
}

static int t1_getctxno_sc(opj_t1_t *t1, int f) {
	return t1->lut_ctxno_sc[(f & (T1_SIG_PRIM | T1_SGN)) >> 4];
}

static int t1_getctxno_mag(opj_t1_t *t1, int f) {
	return t1->lut_ctxno_mag[(f & T1_SIG_OTH) | (((f & T1_REFINE) != 0) << 11)];
}

static int t1_getspb(opj_t1_t *t1, int f) {
	return t1->lut_spb[(f & (T1_SIG_PRIM | T1_SGN)) >> 4];
}

static int t1_getnmsedec_sig(opj_t1_t *t1, int x, int bitpos) {
	if (bitpos > T1_NMSEDEC_FRACBITS) {
		return t1->lut_nmsedec_sig[(x >> (bitpos - T1_NMSEDEC_FRACBITS)) & ((1 << T1_NMSEDEC_BITS) - 1)];
	}
	
	return t1->lut_nmsedec_sig0[x & ((1 << T1_NMSEDEC_BITS) - 1)];
}

static int t1_getnmsedec_ref(opj_t1_t *t1, int x, int bitpos) {
	if (bitpos > T1_NMSEDEC_FRACBITS) {
		return t1->lut_nmsedec_ref[(x >> (bitpos - T1_NMSEDEC_FRACBITS)) & ((1 << T1_NMSEDEC_BITS) - 1)];
	}

    return t1->lut_nmsedec_ref0[x & ((1 << T1_NMSEDEC_BITS) - 1)];
}

static void t1_updateflags(int *fp, int s) {
	int *np = fp - (T1_MAXCBLKW + 2);
	int *sp = fp + (T1_MAXCBLKW + 2);
	np[-1] |= T1_SIG_SE;
	np[1] |= T1_SIG_SW;
	sp[-1] |= T1_SIG_NE;
	sp[1] |= T1_SIG_NW;
	*np |= T1_SIG_S;
	*sp |= T1_SIG_N;
	fp[-1] |= T1_SIG_E;
	fp[1] |= T1_SIG_W;
	if (s) {
		*np |= T1_SGN_S;
		*sp |= T1_SGN_N;
		fp[-1] |= T1_SGN_E;
		fp[1] |= T1_SGN_W;
	}
}

static void t1_enc_sigpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int bpno, int one, int *nmsedec, char type, int vsc) {
	int v, flag;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW | T1_SGN_S))) : (*fp);
	if ((flag & T1_SIG_OTH) && !(flag & (T1_SIG | T1_VISIT))) {
		v = int_abs(*dp) & one ? 1 : 0;
		if (type == T1_TYPE_RAW) {	/* BYPASS/LAZY MODE */
			mqc_setcurctx(mqc, t1_getctxno_zc(t1, flag, orient));	/* ESSAI */
			mqc_bypass_enc(mqc, v);
		} else {
			mqc_setcurctx(mqc, t1_getctxno_zc(t1, flag, orient));
			mqc_encode(mqc, v);
		}
		if (v) {
			v = *dp < 0 ? 1 : 0;
			*nmsedec +=	t1_getnmsedec_sig(t1, int_abs(*dp), bpno + T1_NMSEDEC_FRACBITS);
			if (type == T1_TYPE_RAW) {	/* BYPASS/LAZY MODE */
				mqc_setcurctx(mqc, t1_getctxno_sc(t1, flag));	/* ESSAI */
				mqc_bypass_enc(mqc, v);
			} else {
				mqc_setcurctx(mqc, t1_getctxno_sc(t1, flag));
				mqc_encode(mqc, v ^ t1_getspb(t1, flag));
			}
			t1_updateflags(fp, v);
			*fp |= T1_SIG;
		}
		*fp |= T1_VISIT;
	}
}

static void t1_dec_sigpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int oneplushalf, char type, int vsc) {
	int v, flag;
	
	opj_raw_t *raw = t1->raw;	/* RAW component */
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW | T1_SGN_S))) : (*fp);
	if ((flag & T1_SIG_OTH) && !(flag & (T1_SIG | T1_VISIT))) {
		if (type == T1_TYPE_RAW) {
			if (raw_decode(raw)) {
				v = raw_decode(raw);	/* ESSAI */
				*dp = v ? -oneplushalf : oneplushalf;
				t1_updateflags(fp, v);
				*fp |= T1_SIG;
			}
		} else {
			mqc_setcurctx(mqc, t1_getctxno_zc(t1, flag, orient));
			if (mqc_decode(mqc)) {
				mqc_setcurctx(mqc, t1_getctxno_sc(t1, flag));
				v = mqc_decode(mqc) ^ t1_getspb(t1, flag);
				*dp = v ? -oneplushalf : oneplushalf;
				t1_updateflags(fp, v);
				*fp |= T1_SIG;
			}
		}
		*fp |= T1_VISIT;
	}
}				/* VSC and  BYPASS by Antonin */

static void t1_enc_sigpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, char type, int cblksty) {
	int i, j, k, m, one, vsc;
	*nmsedec = 0;
	one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
	for (m = 0; m < l; m++) {
	for (k = 0; k < h; k += 4) {
		for (i = 0; i < w; i++) {
			for (j = k; j < k + 4 && j < h; j++) {
				vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
				t1_enc_sigpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->data[m][j][i], orient, bpno, one, nmsedec, type, vsc);
			}
		}
	}
	}
}

static void t1_dec_sigpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, char type, int cblksty) {
	int i, j, k, m, one, half, oneplushalf, vsc;
	one = 1 << bpno;
	half = one >> 1;
	oneplushalf = one | half;
	for (m = 0; m < l; m++) {
	for (k = 0; k < h; k += 4) {
		for (i = 0; i < w; i++) {
			for (j = k; j < k + 4 && j < h; j++) {
				vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
				t1_dec_sigpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->data[m][j][i], orient, oneplushalf, type, vsc);
			}
		}
	}
	}
}				/* VSC and  BYPASS by Antonin */

static void t1_enc_refpass_step(opj_t1_t *t1, int *fp, int *dp, int bpno, int one, int *nmsedec, char type, int vsc) {
	int v, flag;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW | T1_SGN_S))) : (*fp);
	if ((flag & (T1_SIG | T1_VISIT)) == T1_SIG) {
		*nmsedec += t1_getnmsedec_ref(t1, int_abs(*dp), bpno + T1_NMSEDEC_FRACBITS);
		v = int_abs(*dp) & one ? 1 : 0;
		if (type == T1_TYPE_RAW) {	/* BYPASS/LAZY MODE */
			mqc_setcurctx(mqc, t1_getctxno_mag(t1, flag));	/* ESSAI */
			mqc_bypass_enc(mqc, v);
		} else {
			mqc_setcurctx(mqc, t1_getctxno_mag(t1, flag));
			mqc_encode(mqc, v);
		}
		*fp |= T1_REFINE;
	}
}

static void t1_dec_refpass_step(opj_t1_t *t1, int *fp, int *dp, int poshalf, int neghalf, char type, int vsc) {
	int v, t, flag;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	opj_raw_t *raw = t1->raw;	/* RAW component */
	
	flag = vsc ? ((*fp) & (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW | T1_SGN_S))) : (*fp);
	if ((flag & (T1_SIG | T1_VISIT)) == T1_SIG) {
		if (type == T1_TYPE_RAW) {
			mqc_setcurctx(mqc, t1_getctxno_mag(t1, flag));	/* ESSAI */
			v = raw_decode(raw);
		} else {
			mqc_setcurctx(mqc, t1_getctxno_mag(t1, flag));
			v = mqc_decode(mqc);
		}
		t = v ? poshalf : neghalf;
		*dp += *dp < 0 ? -t : t;
		*fp |= T1_REFINE;
	}
}				/* VSC and  BYPASS by Antonin  */

static void t1_enc_refpass(opj_t1_t *t1, int w, int h, int l, int bpno, int *nmsedec, char type, int cblksty) {
	int i, j, k, m, one, vsc;
	*nmsedec = 0;
	one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
	for (m = 0; m < l; m++) {
		for (k = 0; k < h; k += 4) {
		for (i = 0; i < w; i++) {
			for (j = k; j < k + 4 && j < h; j++) {
				vsc = ((cblksty & J3D_CCP_CBLKSTY_VSC) && (j == k + 3 || j == h - 1)) ? 1 : 0;
				t1_enc_refpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->data[m][j][i], bpno, one, nmsedec, type, vsc);
			}
		}
	}
	}
}

static void t1_dec_refpass(opj_t1_t *t1, int w, int h, int l, int bpno, char type, int cblksty) {
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
				t1_dec_refpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->data[m][j][i], poshalf, neghalf, type, vsc);
			}
		}
	}
	}
}				/* VSC and  BYPASS by Antonin */

static void t1_enc_clnpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int bpno, int one, int *nmsedec, int partial, int vsc) {
	int v, flag;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW | T1_SGN_S))) : (*fp);
	if (partial) {
		goto LABEL_PARTIAL;
	}
	if (!(*fp & (T1_SIG | T1_VISIT))) {
		mqc_setcurctx(mqc, t1_getctxno_zc(t1, flag, orient));
		v = int_abs(*dp) & one ? 1 : 0;
		mqc_encode(mqc, v);
		if (v) {
LABEL_PARTIAL:
			*nmsedec += t1_getnmsedec_sig(t1, int_abs(*dp), bpno + T1_NMSEDEC_FRACBITS);
			mqc_setcurctx(mqc, t1_getctxno_sc(t1, flag));
			v = *dp < 0 ? 1 : 0;
			mqc_encode(mqc, v ^ t1_getspb(t1, flag));
			t1_updateflags(fp, v);
			*fp |= T1_SIG;
		}
	}
	*fp &= ~T1_VISIT;
}

static void t1_dec_clnpass_step(opj_t1_t *t1, int *fp, int *dp, int orient, int oneplushalf, int partial, int vsc) {
	int v, flag;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	flag = vsc ? ((*fp) & (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW | T1_SGN_S))) : (*fp);
	if (partial) {
		goto LABEL_PARTIAL;
	}
	if (!(flag & (T1_SIG | T1_VISIT))) {
		mqc_setcurctx(mqc, t1_getctxno_zc(t1, flag, orient));
		if (mqc_decode(mqc)) {
LABEL_PARTIAL:
			mqc_setcurctx(mqc, t1_getctxno_sc(t1, flag));
			v = mqc_decode(mqc) ^ t1_getspb(t1, flag);
			*dp = v ? -oneplushalf : oneplushalf;
			t1_updateflags(fp, v);
			*fp |= T1_SIG;
		}
	}
	*fp &= ~T1_VISIT;
}				/* VSC and  BYPASS by Antonin */

static void t1_enc_clnpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, int *nmsedec, int cblksty) {
	int i, j, k, m, one, agg, runlen, vsc;
	
	opj_mqc_t *mqc = t1->mqc;	/* MQC component */
	
	*nmsedec = 0;
	one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
	for (m = 0; m < l; m++) {
		for (k = 0; k < h; k += 4) {
			for (i = 0; i < w; i++) {
				if (k + 3 < h) {
					if (cblksty & J3D_CCP_CBLKSTY_VSC) {
						agg = !(t1->flags[1 + m][1 + k][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
							|| t1->flags[1 + m][1 + k + 1][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
							|| t1->flags[1 + m][1 + k + 2][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
							|| (t1->flags[1 + m][1 + k + 3][1 + i] 
							& (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW |	T1_SGN_S))) & (T1_SIG | T1_VISIT | T1_SIG_OTH));
					} else {
						agg = !(t1->flags[1 + m][1 + k][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
							|| t1->flags[1 + m][1 + k + 1][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
							|| t1->flags[1 + m][1 + k + 2][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
							|| t1->flags[1 + m][1 + k + 3][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH));
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
					t1_enc_clnpass_step(t1, &(t1->flags[1 + m][1 + j][1 + i]), &(t1->data[m][j][i]), orient, bpno, one, nmsedec, agg && (j == k + runlen), vsc);
				}
			}
	}
	}
}

static void t1_dec_clnpass(opj_t1_t *t1, int w, int h, int l, int bpno, int orient, int cblksty) {
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
					agg = !(t1->flags[1 + m][1 + k][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
						|| t1->flags[1 + m][1 + k + 1][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
						|| t1->flags[1 + m][1 + k + 2][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
						|| (t1->flags[1 + m][1 + k + 3][1 + i] 
						& (~(T1_SIG_S | T1_SIG_SE | T1_SIG_SW |	T1_SGN_S))) & (T1_SIG | T1_VISIT | T1_SIG_OTH));
				} else {
					agg = !(t1->flags[1 + m][1 + k][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
						|| t1->flags[1 + m][1 + k + 1][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
						|| t1->flags[1 + m][1 + k + 2][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH)
						|| t1->flags[1 + m][1 + k + 3][1 + i] & (T1_SIG | T1_VISIT | T1_SIG_OTH));
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
				t1_dec_clnpass_step(t1, &t1->flags[1 + m][1 + j][1 + i], &t1->data[m][j][i], orient, oneplushalf, agg && (j == k + runlen), vsc);
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


static void t1_encode_cblk(opj_t1_t *t1, opj_tcd_cblk_t * cblk, int orient, int compno, int level[3], int dwtid[3], double stepsize, int cblksty, int numcomps, opj_tcd_tile_t * tile) {
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
		//fprintf(stdout,"passno %d passtype %d w %d h %d l %d bpno %d orient %d type %d cblksty %d\n",passno,passtype,w,h,l,bpno,orient,type,cblksty);

		switch (passtype) {
			case 0:
				t1_enc_sigpass(t1, w, h, l, bpno, orient, &nmsedec, type, cblksty);
				break;
			case 1:
				t1_enc_refpass(t1, w, h, l, bpno, &nmsedec, type, cblksty);
				break;
			case 2:
				//fprintf(stdout,"w %d h %d l %d bpno %d orient %d \n",w,h,l,bpno,orient);
				t1_enc_clnpass(t1, w, h, l, bpno, orient, &nmsedec, cblksty);
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
					/* correction = mqc_bypass_flush_enc(); */
				} else {		/* correction = mqc_restart_enc(); */
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

static void t1_decode_cblk(opj_t1_t *t1, opj_tcd_cblk_t * cblk, int orient, int roishift, int cblksty) {
	int i, j, k, w, h, l;
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
					t1_dec_sigpass(t1, w, h, l, bpno+1, orient, type, cblksty);
					break;
				case 1:
					t1_dec_refpass(t1, w, h, l, bpno+1, type, cblksty);
					break;
				case 2:
					t1_dec_clnpass(t1, w, h, l, bpno+1, orient, cblksty);
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

static int t1_init_ctxno_zc(int f, int orient) {
	int h, v, d, n, t, hv;
	n = 0;
	h = ((f & T1_SIG_W) != 0) + ((f & T1_SIG_E) != 0);
	v = ((f & T1_SIG_N) != 0) + ((f & T1_SIG_S) != 0);
	d = ((f & T1_SIG_NW) != 0) + ((f & T1_SIG_NE) != 0) + ((f & T1_SIG_SE) != 0) + ((f & T1_SIG_SW) != 0);
	
	switch (orient) {
		case 2:
			t = h;
			h = v;
			v = t;
		case 0:
		case 1:
			if (!h) {
				if (!v) {
					if (!d)
						n = 0;
					else if (d == 1)
						n = 1;
					else
						n = 2;
				} else if (v == 1) {
					n = 3;
				} else {
					n = 4;
				}
			} else if (h == 1) {
				if (!v) {
					if (!d)
						n = 5;
					else
						n = 6;
				} else {
					n = 7;
				}
			} else
				n = 8;
			break;
		case 3:
			hv = h + v;
			if (!d) {
				if (!hv) {
					n = 0;
				} else if (hv == 1) {
					n = 1;
				} else {
					n = 2;
				}
			} else if (d == 1) {
				if (!hv) {
					n = 3;
				} else if (hv == 1) {
					n = 4;
				} else {
					n = 5;
				}
			} else if (d == 2) {
				if (!hv) {
					n = 6;
				} else {
					n = 7;
				}
			} else {
				n = 8;
			}
			break;
	}
	
	return (T1_CTXNO_ZC + n);
}

static int t1_init_ctxno_sc(int f) {
	int hc, vc, n;
	n = 0;

	hc = int_min(((f & (T1_SIG_E | T1_SGN_E)) ==
		T1_SIG_E) + ((f & (T1_SIG_W | T1_SGN_W)) == T1_SIG_W),
	       1) - int_min(((f & (T1_SIG_E | T1_SGN_E)) ==
		   (T1_SIG_E | T1_SGN_E)) +
		   ((f & (T1_SIG_W | T1_SGN_W)) ==
		   (T1_SIG_W | T1_SGN_W)), 1);
	
	vc = int_min(((f & (T1_SIG_N | T1_SGN_N)) ==
		T1_SIG_N) + ((f & (T1_SIG_S | T1_SGN_S)) == T1_SIG_S),
	       1) - int_min(((f & (T1_SIG_N | T1_SGN_N)) ==
		   (T1_SIG_N | T1_SGN_N)) +
		   ((f & (T1_SIG_S | T1_SGN_S)) ==
		   (T1_SIG_S | T1_SGN_S)), 1);
	
	if (hc < 0) {
		hc = -hc;
		vc = -vc;
	}
	if (!hc) {
		if (vc == -1)
			n = 1;
		else if (!vc)
			n = 0;
		else
			n = 1;
	} else if (hc == 1) {
		if (vc == -1)
			n = 2;
		else if (!vc)
			n = 3;
		else
			n = 4;
	}
	
	return (T1_CTXNO_SC + n);
}

static int t1_init_ctxno_mag(int f) {
	int n;
	if (!(f & T1_REFINE))
		n = (f & (T1_SIG_OTH)) ? 1 : 0;
	else
		n = 2;
	
	return (T1_CTXNO_MAG + n);
}

static int t1_init_spb(int f) {
	int hc, vc, n;
	
	hc = int_min(((f & (T1_SIG_E | T1_SGN_E)) ==
		T1_SIG_E) + ((f & (T1_SIG_W | T1_SGN_W)) == T1_SIG_W),
	       1) - int_min(((f & (T1_SIG_E | T1_SGN_E)) ==
		   (T1_SIG_E | T1_SGN_E)) +
		   ((f & (T1_SIG_W | T1_SGN_W)) ==
		   (T1_SIG_W | T1_SGN_W)), 1);
	
	vc = int_min(((f & (T1_SIG_N | T1_SGN_N)) ==
		T1_SIG_N) + ((f & (T1_SIG_S | T1_SGN_S)) == T1_SIG_S),
	       1) - int_min(((f & (T1_SIG_N | T1_SGN_N)) ==
		   (T1_SIG_N | T1_SGN_N)) +
		   ((f & (T1_SIG_S | T1_SGN_S)) ==
		   (T1_SIG_S | T1_SGN_S)), 1);
	
	if (!hc && !vc)
		n = 0;
	else
		n = (!(hc > 0 || (!hc && vc > 0)));
	
	return n;
}

static void t1_init_luts(opj_t1_t *t1) {
	int i, j;
	double u, v, t;
	for (j = 0; j < 4; j++) {
		for (i = 0; i < 256; ++i) {
			t1->lut_ctxno_zc[(j << 8) | i] = t1_init_ctxno_zc(i, j);
		}
	}
	for (i = 0; i < 256; i++) {
		t1->lut_ctxno_sc[i] = t1_init_ctxno_sc(i << 4);
	}
	for (j = 0; j < 2; j++) {
		for (i = 0; i < 2048; ++i) {
			t1->lut_ctxno_mag[(j << 11) + i] = t1_init_ctxno_mag((j ? T1_REFINE : 0) | i);
		}
	}
	for (i = 0; i < 256; ++i) {
		t1->lut_spb[i] = t1_init_spb(i << 4);
	}
	/* FIXME FIXME FIXME */
	/* fprintf(stdout,"nmsedec luts:\n"); */
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

opj_t1_t* t1_create(opj_common_ptr cinfo) {
	opj_t1_t *t1 = (opj_t1_t*)opj_malloc(sizeof(opj_t1_t));
	if(t1) {
		t1->cinfo = cinfo;
		/* create MQC and RAW handles */
		t1->mqc = mqc_create();
		t1->raw = raw_create();
		/* initialize the look-up tables of the Tier-1 coder/decoder */
		t1_init_luts(t1);
	}
	return t1;
}

void t1_destroy(opj_t1_t *t1) {
	if(t1) {
		/* destroy MQC and RAW handles */
		mqc_destroy(t1->mqc);
		raw_destroy(t1->raw);
		//opj_free(t1->data);
		//opj_free(t1->flags);
		opj_free(t1);
	}
}

void t1_encode_cblks(opj_t1_t *t1, opj_tcd_tile_t *tile, opj_tcp_t *tcp) {
	int compno, resno, bandno, precno, cblkno;
	int x, y, z, i, j, k, orient;
	int n=0;
	int level[3];
	FILE *fid = NULL;
//	char filename[10];
	tile->distotile = 0;		/* fixed_quality */
	
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];

		for (resno = 0; resno < tilec->numresolution[0]; resno++) {
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];
			
			/* Weighted first order entropy
			sprintf(filename,"res%d.txt",resno);
			if ((fid = fopen(filename,"w")) == 0){
				fprintf(stdout,"Error while opening %s\n", filename);
				exit(1);
			}
			*/
			for (bandno = 0; bandno < res->numbands; bandno++) {
				opj_tcd_band_t *band = &res->bands[bandno];
				for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
					opj_tcd_precinct_t *prc = &band->precincts[precno];

					for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
						opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];

						//fprintf(stdout,"Precno %d Cblkno %d \n",precno,cblkno);
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
//fprintf(fid," %d",t1->data[k][j][i]);
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
						if (orient == 2) {
							orient = 1;
						} else if (orient == 1) {
							orient = 2;
						}
						for (i = 0; i < 3; i++) 
							level[i] = tilec->numresolution[i] - 1 - resno;
						//fprintf(stdout,"t1_encode_cblk(t1, cblk, %d, %d, %d %d %d, %d, %f, %d, %d, tile);\n", orient, compno, level[0], level[1], level[2], tcp->tccps[compno].reversible, band->stepsize, tcp->tccps[compno].cblksty, tile->numcomps);
						t1_encode_cblk(t1, cblk, orient, compno, level, tcp->tccps[compno].dwtid, band->stepsize, tcp->tccps[compno].cblksty, tile->numcomps, tile);
							
					} /* cblkno */
				} /* precno */
//fprintf(fid,"\n");
			} /* bandno */
//fclose(fid);
		} /* resno  */
	} /* compno  */
}

void t1_decode_cblks(opj_t1_t *t1, opj_tcd_tile_t *tile, opj_tcp_t *tcp) {
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
						int x, y, k, i, j, z, orient;
						opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];

						orient = band->bandno;	/* FIXME */
						if (orient == 2) {
							orient = 1;
						} else if (orient == 1) {
							orient = 2;
						}

						t1_decode_cblk(t1, cblk, orient, tcp->tccps[compno].roishift, tcp->tccps[compno].cblksty);

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


/** mod fixed_quality */
double t1_getwmsedec(int nmsedec, int compno, int level[3], int orient, int bpno, double stepsize, int numcomps, int dwtid[3])	{
	double w1, w2, wmsedec;
	
	if (dwtid[0] == 1 || dwtid[1] == 1 || dwtid[2] == 1) {
		w1 = (numcomps > 1) ? mct_getnorm_real(compno) : 1;
	} else {			
		w1 = (numcomps > 1) ? mct_getnorm(compno) : 1;
	}
	w2 = dwt_getnorm(orient, level, dwtid);

	//fprintf(stdout,"nmsedec %d level %d %d %d orient %d bpno %d stepsize %f \n",nmsedec ,level[0],level[1],level[2],orient,bpno,stepsize);
	wmsedec = w1 * w2 * stepsize * (1 << bpno);
	wmsedec *= wmsedec * nmsedec / 8192.0;
	
	return wmsedec;
}
/** mod fixed_quality */
