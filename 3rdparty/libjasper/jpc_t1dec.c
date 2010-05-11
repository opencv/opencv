/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2003 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 * 
 * JasPer License Version 2.0
 * 
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 * 
 * All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 * 
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 * 
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 * 
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 * 
 * __END_OF_JASPER_LICENSE__
 */

/*
 * Tier 1 Decoder
 *
 * $Id: jpc_t1dec.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "jasper/jas_fix.h"
#include "jasper/jas_stream.h"
#include "jasper/jas_math.h"

#include "jpc_bs.h"
#include "jpc_mqdec.h"
#include "jpc_t1dec.h"
#include "jpc_t1cod.h"
#include "jpc_dec.h"

/******************************************************************************\
*
\******************************************************************************/

static int jpc_dec_decodecblk(jpc_dec_t *dec, jpc_dec_tile_t *tile, jpc_dec_tcomp_t *tcomp, jpc_dec_band_t *band,
  jpc_dec_cblk_t *cblk, int dopartial, int maxlyrs);
static int dec_sigpass(jpc_dec_t *dec, jpc_mqdec_t *mqdec, int bitpos, int orient,
  int vcausalflag, jas_matrix_t *flags, jas_matrix_t *data);
static int dec_rawsigpass(jpc_dec_t *dec, jpc_bitstream_t *in, int bitpos,
  int vcausalflag, jas_matrix_t *flags, jas_matrix_t *data);
static int dec_refpass(jpc_dec_t *dec, jpc_mqdec_t *mqdec, int bitpos, int vcausalflag,
  jas_matrix_t *flags, jas_matrix_t *data);
static int dec_rawrefpass(jpc_dec_t *dec, jpc_bitstream_t *in, int bitpos,
  int vcausalflag, jas_matrix_t *flags, jas_matrix_t *data);
static int dec_clnpass(jpc_dec_t *dec, jpc_mqdec_t *mqdec, int bitpos, int orient,
  int vcausalflag, int segsymflag, jas_matrix_t *flags, jas_matrix_t *data);

#if defined(DEBUG)
static long t1dec_cnt = 0;
#endif

#if !defined(DEBUG)
#define	JPC_T1D_GETBIT(mqdec, v, passtypename, symtypename) \
	((v) = jpc_mqdec_getbit(mqdec))
#else
#define	JPC_T1D_GETBIT(mqdec, v, passtypename, symtypename) \
{ \
	(v) = jpc_mqdec_getbit(mqdec); \
	if (jas_getdbglevel() >= 100) { \
		jas_eprintf("index = %ld; passtype = %s; symtype = %s; sym = %d\n", t1dec_cnt, passtypename, symtypename, v); \
		++t1dec_cnt; \
	} \
}
#endif
#define	JPC_T1D_GETBITNOSKEW(mqdec, v, passtypename, symtypename) \
	JPC_T1D_GETBIT(mqdec, v, passtypename, symtypename)

#if !defined(DEBUG)
#define	JPC_T1D_RAWGETBIT(bitstream, v, passtypename, symtypename) \
	((v) = jpc_bitstream_getbit(bitstream))
#else
#define	JPC_T1D_RAWGETBIT(bitstream, v, passtypename, symtypename) \
{ \
	(v) = jpc_bitstream_getbit(bitstream); \
	if (jas_getdbglevel() >= 100) { \
		jas_eprintf("index = %ld; passtype = %s; symtype = %s; sym = %d\n", t1dec_cnt, passtypename, symtypename, v); \
		++t1dec_cnt; \
	} \
}
#endif

/******************************************************************************\
* Code.
\******************************************************************************/

int jpc_dec_decodecblks(jpc_dec_t *dec, jpc_dec_tile_t *tile)
{
	jpc_dec_tcomp_t *tcomp;
	int compcnt;
	jpc_dec_rlvl_t *rlvl;
	int rlvlcnt;
	jpc_dec_band_t *band;
	int bandcnt;
	jpc_dec_prc_t *prc;
	int prccnt;
	jpc_dec_cblk_t *cblk;
	int cblkcnt;

	for (compcnt = dec->numcomps, tcomp = tile->tcomps; compcnt > 0;
	  --compcnt, ++tcomp) {
		for (rlvlcnt = tcomp->numrlvls, rlvl = tcomp->rlvls;
		  rlvlcnt > 0; --rlvlcnt, ++rlvl) {
			if (!rlvl->bands) {
				continue;
			}
			for (bandcnt = rlvl->numbands, band = rlvl->bands;
			  bandcnt > 0; --bandcnt, ++band) {
				if (!band->data) {
					continue;
				}
				for (prccnt = rlvl->numprcs, prc = band->prcs;
				  prccnt > 0; --prccnt, ++prc) {
					if (!prc->cblks) {
						continue;
					}
					for (cblkcnt = prc->numcblks,
					  cblk = prc->cblks; cblkcnt > 0;
					  --cblkcnt, ++cblk) {
						if (jpc_dec_decodecblk(dec, tile, tcomp,
						  band, cblk, 1, JPC_MAXLYRS)) {
							return -1;
						}
					}
				}

			}
		}
	}

	return 0;
}

static int jpc_dec_decodecblk(jpc_dec_t *dec, jpc_dec_tile_t *tile, jpc_dec_tcomp_t *tcomp, jpc_dec_band_t *band,
  jpc_dec_cblk_t *cblk, int dopartial, int maxlyrs)
{
	jpc_dec_seg_t *seg;
	int i;
	int bpno;
	int passtype;
	int ret;
	int compno;
	int filldata;
	int fillmask;
	jpc_dec_ccp_t *ccp;

	compno = tcomp - tile->tcomps;

	if (!cblk->flags) {
		/* Note: matrix is assumed to be zeroed */
		if (!(cblk->flags = jas_matrix_create(jas_matrix_numrows(cblk->data) +
		  2, jas_matrix_numcols(cblk->data) + 2))) {
			return -1;
		}
	}

	seg = cblk->segs.head;
	while (seg && (seg != cblk->curseg || dopartial) && (maxlyrs < 0 ||
	  seg->lyrno < maxlyrs)) {
		assert(seg->numpasses >= seg->maxpasses || dopartial);
		assert(seg->stream);
		jas_stream_rewind(seg->stream);
		jas_stream_setrwcount(seg->stream, 0);
		if (seg->type == JPC_SEG_MQ) {
			if (!cblk->mqdec) {
				if (!(cblk->mqdec = jpc_mqdec_create(JPC_NUMCTXS, 0))) {
					return -1;
				}
				jpc_mqdec_setctxs(cblk->mqdec, JPC_NUMCTXS, jpc_mqctxs);
			}
			jpc_mqdec_setinput(cblk->mqdec, seg->stream);
			jpc_mqdec_init(cblk->mqdec);
		} else {
			assert(seg->type == JPC_SEG_RAW);
			if (!cblk->nulldec) {
				if (!(cblk->nulldec = jpc_bitstream_sopen(seg->stream, "r"))) {
					assert(0);
				}
			}
		}


		for (i = 0; i < seg->numpasses; ++i) {
			if (cblk->numimsbs > band->numbps) {
				ccp = &tile->cp->ccps[compno];
				if (ccp->roishift <= 0) {
					jas_eprintf("warning: corrupt code stream\n");
				} else {
					if (cblk->numimsbs < ccp->roishift - band->numbps) {
						jas_eprintf("warning: corrupt code stream\n");
					}
				}
			}
			bpno = band->roishift + band->numbps - 1 - (cblk->numimsbs +
			  (seg->passno + i - cblk->firstpassno + 2) / 3);
if (bpno < 0) {
	goto premature_exit;
}
#if 1
			passtype = (seg->passno + i + 2) % 3;
#else
			passtype = JPC_PASSTYPE(seg->passno + i + 2);
#endif
			assert(bpno >= 0 && bpno < 31);
			switch (passtype) {
			case JPC_SIGPASS:
				ret = (seg->type == JPC_SEG_MQ) ? dec_sigpass(dec,
				  cblk->mqdec, bpno, band->orient,
				  (tile->cp->ccps[compno].cblkctx & JPC_COX_VSC) != 0,
				  cblk->flags, cblk->data) :
				  dec_rawsigpass(dec, cblk->nulldec, bpno,
				  (tile->cp->ccps[compno].cblkctx & JPC_COX_VSC) != 0,
				  cblk->flags, cblk->data);
				break;
			case JPC_REFPASS:
				ret = (seg->type == JPC_SEG_MQ) ?
				  dec_refpass(dec, cblk->mqdec, bpno,
				  (tile->cp->ccps[compno].cblkctx & JPC_COX_VSC) != 0,
				  cblk->flags, cblk->data) :
				  dec_rawrefpass(dec, cblk->nulldec, bpno,
				  (tile->cp->ccps[compno].cblkctx & JPC_COX_VSC) != 0,
				  cblk->flags, cblk->data);
				break;
			case JPC_CLNPASS:
				assert(seg->type == JPC_SEG_MQ);
				ret = dec_clnpass(dec, cblk->mqdec, bpno,
				  band->orient, (tile->cp->ccps[compno].cblkctx &
				  JPC_COX_VSC) != 0, (tile->cp->ccps[compno].cblkctx &
				  JPC_COX_SEGSYM) != 0, cblk->flags,
				  cblk->data);
				break;
			default:
				ret = -1;
				break;
			}
			/* Do we need to reset after each coding pass? */
			if (tile->cp->ccps[compno].cblkctx & JPC_COX_RESET) {
				jpc_mqdec_setctxs(cblk->mqdec, JPC_NUMCTXS, jpc_mqctxs);
			}

			if (ret) {
				jas_eprintf("coding pass failed passtype=%d segtype=%d\n", passtype, seg->type);
				return -1;
			}

		}

		if (seg->type == JPC_SEG_MQ) {
/* Note: dont destroy mq decoder because context info will be lost */
		} else {
			assert(seg->type == JPC_SEG_RAW);
			if (tile->cp->ccps[compno].cblkctx & JPC_COX_PTERM) {
				fillmask = 0x7f;
				filldata = 0x2a;
			} else {
				fillmask = 0;
				filldata = 0;
			}
			if ((ret = jpc_bitstream_inalign(cblk->nulldec, fillmask,
			  filldata)) < 0) {
				return -1;
			} else if (ret > 0) {
				jas_eprintf("warning: bad termination pattern detected\n");
			}
			jpc_bitstream_close(cblk->nulldec);
			cblk->nulldec = 0;
		}

		cblk->curseg = seg->next;
		jpc_seglist_remove(&cblk->segs, seg);
		jpc_seg_destroy(seg);
		seg = cblk->curseg;
	}

	assert(dopartial ? (!cblk->curseg) : 1);

premature_exit:
	return 0;
}

/******************************************************************************\
* Code for significance pass.
\******************************************************************************/

#define	jpc_sigpass_step(fp, frowstep, dp, bitpos, oneplushalf, orient, mqdec, vcausalflag) \
{ \
	int f; \
	int v; \
	f = *(fp); \
	if ((f & JPC_OTHSIGMSK) && !(f & (JPC_SIG | JPC_VISIT))) { \
		jpc_mqdec_setcurctx((mqdec), JPC_GETZCCTXNO(f, (orient))); \
		JPC_T1D_GETBIT((mqdec), v, "SIG", "ZC"); \
		if (v) { \
			jpc_mqdec_setcurctx((mqdec), JPC_GETSCCTXNO(f)); \
			JPC_T1D_GETBIT((mqdec), v, "SIG", "SC"); \
			v ^= JPC_GETSPB(f); \
			JPC_UPDATEFLAGS4((fp), (frowstep), v, (vcausalflag)); \
			*(fp) |= JPC_SIG; \
			*(dp) = (v) ? (-(oneplushalf)) : (oneplushalf); \
		} \
		*(fp) |= JPC_VISIT; \
	} \
}

static int dec_sigpass(jpc_dec_t *dec, register jpc_mqdec_t *mqdec, int bitpos, int orient,
  int vcausalflag, jas_matrix_t *flags, jas_matrix_t *data)
{
	int i;
	int j;
	int one;
	int half;
	int oneplushalf;
	int vscanlen;
	int width;
	int height;
	jpc_fix_t *fp;
	int frowstep;
	int fstripestep;
	jpc_fix_t *fstripestart;
	jpc_fix_t *fvscanstart;
	jpc_fix_t *dp;
	int drowstep;
	int dstripestep;
	jpc_fix_t *dstripestart;
	jpc_fix_t *dvscanstart;
	int k;

	/* Avoid compiler warning about unused parameters. */
	dec = 0;

	width = jas_matrix_numcols(data);
	height = jas_matrix_numrows(data);
	frowstep = jas_matrix_rowstep(flags);
	drowstep = jas_matrix_rowstep(data);
	fstripestep = frowstep << 2;
	dstripestep = drowstep << 2;

	one = 1 << bitpos;
	half = one >> 1;
	oneplushalf = one | half;

	fstripestart = jas_matrix_getref(flags, 1, 1);
	dstripestart = jas_matrix_getref(data, 0, 0);
	for (i = height; i > 0; i -= 4, fstripestart += fstripestep,
	  dstripestart += dstripestep) {
		fvscanstart = fstripestart;
		dvscanstart = dstripestart;
		vscanlen = JAS_MIN(i, 4);
		for (j = width; j > 0; --j, ++fvscanstart, ++dvscanstart) {
			fp = fvscanstart;
			dp = dvscanstart;
			k = vscanlen;

			/* Process first sample in vertical scan. */
			jpc_sigpass_step(fp, frowstep, dp, bitpos, oneplushalf,
			  orient, mqdec, vcausalflag);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process second sample in vertical scan. */
			jpc_sigpass_step(fp, frowstep, dp, bitpos, oneplushalf,
			  orient, mqdec, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process third sample in vertical scan. */
			jpc_sigpass_step(fp, frowstep, dp, bitpos, oneplushalf,
			  orient, mqdec, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process fourth sample in vertical scan. */
			jpc_sigpass_step(fp, frowstep, dp, bitpos, oneplushalf,
			  orient, mqdec, 0);
		}
	}
	return 0;
}

#define	jpc_rawsigpass_step(fp, frowstep, dp, oneplushalf, in, vcausalflag) \
{ \
	jpc_fix_t f = *(fp); \
	jpc_fix_t v; \
	if ((f & JPC_OTHSIGMSK) && !(f & (JPC_SIG | JPC_VISIT))) { \
		JPC_T1D_RAWGETBIT(in, v, "SIG", "ZC"); \
		if (v < 0) { \
			return -1; \
		} \
		if (v) { \
			JPC_T1D_RAWGETBIT(in, v, "SIG", "SC"); \
			if (v < 0) { \
				return -1; \
			} \
			JPC_UPDATEFLAGS4((fp), (frowstep), v, (vcausalflag)); \
			*(fp) |= JPC_SIG; \
			*(dp) = v ? (-oneplushalf) : (oneplushalf); \
		} \
		*(fp) |= JPC_VISIT; \
	} \
}

static int dec_rawsigpass(jpc_dec_t *dec, jpc_bitstream_t *in, int bitpos, int vcausalflag,
  jas_matrix_t *flags, jas_matrix_t *data)
{
	int i;
	int j;
	int k;
	int one;
	int half;
	int oneplushalf;
	int vscanlen;
	int width;
	int height;
	jpc_fix_t *fp;
	int frowstep;
	int fstripestep;
	jpc_fix_t *fstripestart;
	jpc_fix_t *fvscanstart;
	jpc_fix_t *dp;
	int drowstep;
	int dstripestep;
	jpc_fix_t *dstripestart;
	jpc_fix_t *dvscanstart;

	/* Avoid compiler warning about unused parameters. */
	dec = 0;

	width = jas_matrix_numcols(data);
	height = jas_matrix_numrows(data);
	frowstep = jas_matrix_rowstep(flags);
	drowstep = jas_matrix_rowstep(data);
	fstripestep = frowstep << 2;
	dstripestep = drowstep << 2;

	one = 1 << bitpos;
	half = one >> 1;
	oneplushalf = one | half;

	fstripestart = jas_matrix_getref(flags, 1, 1);
	dstripestart = jas_matrix_getref(data, 0, 0);
	for (i = height; i > 0; i -= 4, fstripestart += fstripestep,
	  dstripestart += dstripestep) {
		fvscanstart = fstripestart;
		dvscanstart = dstripestart;
		vscanlen = JAS_MIN(i, 4);
		for (j = width; j > 0; --j, ++fvscanstart, ++dvscanstart) {
			fp = fvscanstart;
			dp = dvscanstart;
			k = vscanlen;

			/* Process first sample in vertical scan. */
			jpc_rawsigpass_step(fp, frowstep, dp, oneplushalf,
			  in, vcausalflag);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process second sample in vertical scan. */
			jpc_rawsigpass_step(fp, frowstep, dp, oneplushalf,
			  in, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process third sample in vertical scan. */
			jpc_rawsigpass_step(fp, frowstep, dp, oneplushalf,
			  in, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process fourth sample in vertical scan. */
			jpc_rawsigpass_step(fp, frowstep, dp, oneplushalf,
			  in, 0);

		}
	}
	return 0;
}

/******************************************************************************\
* Code for refinement pass.
\******************************************************************************/

#define	jpc_refpass_step(fp, dp, poshalf, neghalf, mqdec, vcausalflag) \
{ \
	int v; \
	int t; \
	if (((*(fp)) & (JPC_SIG | JPC_VISIT)) == JPC_SIG) { \
		jpc_mqdec_setcurctx((mqdec), JPC_GETMAGCTXNO(*(fp))); \
		JPC_T1D_GETBITNOSKEW((mqdec), v, "REF", "MR"); \
		t = (v ? (poshalf) : (neghalf)); \
		*(dp) += (*(dp) < 0) ? (-t) : t; \
		*(fp) |= JPC_REFINE; \
	} \
}

static int dec_refpass(jpc_dec_t *dec, register jpc_mqdec_t *mqdec, int bitpos,
  int vcausalflag, jas_matrix_t *flags, jas_matrix_t *data)
{
	int i;
	int j;
	int vscanlen;
	int width;
	int height;
	int one;
	int poshalf;
	int neghalf;
	jpc_fix_t *fp;
	int frowstep;
	int fstripestep;
	jpc_fix_t *fstripestart;
	jpc_fix_t *fvscanstart;
	jpc_fix_t *dp;
	int drowstep;
	int dstripestep;
	jpc_fix_t *dstripestart;
	jpc_fix_t *dvscanstart;
	int k;

	/* Avoid compiler warning about unused parameters. */
	dec = 0;
	vcausalflag = 0;

	width = jas_matrix_numcols(data);
	height = jas_matrix_numrows(data);
	frowstep = jas_matrix_rowstep(flags);
	drowstep = jas_matrix_rowstep(data);
	fstripestep = frowstep << 2;
	dstripestep = drowstep << 2;

	one = 1 << bitpos;
	poshalf = one >> 1;
	neghalf = (bitpos > 0) ? (-poshalf) : (-1);

	fstripestart = jas_matrix_getref(flags, 1, 1);
	dstripestart = jas_matrix_getref(data, 0, 0);
	for (i = height; i > 0; i -= 4, fstripestart += fstripestep,
	  dstripestart += dstripestep) {
		fvscanstart = fstripestart;
		dvscanstart = dstripestart;
		vscanlen = JAS_MIN(i, 4);
		for (j = width; j > 0; --j, ++fvscanstart, ++dvscanstart) {
			fp = fvscanstart;
			dp = dvscanstart;
			k = vscanlen;

			/* Process first sample in vertical scan. */
			jpc_refpass_step(fp, dp, poshalf, neghalf, mqdec,
			  vcausalflag);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process second sample in vertical scan. */
			jpc_refpass_step(fp, dp, poshalf, neghalf, mqdec, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process third sample in vertical scan. */
			jpc_refpass_step(fp, dp, poshalf, neghalf, mqdec, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process fourth sample in vertical scan. */
			jpc_refpass_step(fp, dp, poshalf, neghalf, mqdec, 0);
		}
	}

	return 0;
}

#define	jpc_rawrefpass_step(fp, dp, poshalf, neghalf, in, vcausalflag) \
{ \
	jpc_fix_t v; \
	jpc_fix_t t; \
	if (((*(fp)) & (JPC_SIG | JPC_VISIT)) == JPC_SIG) { \
		JPC_T1D_RAWGETBIT(in, v, "REF", "MAGREF"); \
		if (v < 0) { \
			return -1; \
		} \
		t = (v ? poshalf : neghalf); \
		*(dp) += (*(dp) < 0) ? (-t) : t; \
		*(fp) |= JPC_REFINE; \
	} \
}

static int dec_rawrefpass(jpc_dec_t *dec, jpc_bitstream_t *in, int bitpos, int vcausalflag,
  jas_matrix_t *flags, jas_matrix_t *data)
{
	int i;
	int j;
	int k;
	int vscanlen;
	int width;
	int height;
	int one;
	int poshalf;
	int neghalf;
	jpc_fix_t *fp;
	int frowstep;
	int fstripestep;
	jpc_fix_t *fstripestart;
	jpc_fix_t *fvscanstart;
	jpc_fix_t *dp;
	int drowstep;
	int dstripestep;
	jpc_fix_t *dstripestart;
	jpc_fix_t *dvscanstart;

	/* Avoid compiler warning about unused parameters. */
	dec = 0;
	vcausalflag = 0;

	width = jas_matrix_numcols(data);
	height = jas_matrix_numrows(data);
	frowstep = jas_matrix_rowstep(flags);
	drowstep = jas_matrix_rowstep(data);
	fstripestep = frowstep << 2;
	dstripestep = drowstep << 2;

	one = 1 << bitpos;
	poshalf = one >> 1;
	neghalf = (bitpos > 0) ? (-poshalf) : (-1);

	fstripestart = jas_matrix_getref(flags, 1, 1);
	dstripestart = jas_matrix_getref(data, 0, 0);
	for (i = height; i > 0; i -= 4, fstripestart += fstripestep,
	  dstripestart += dstripestep) {
		fvscanstart = fstripestart;
		dvscanstart = dstripestart;
		vscanlen = JAS_MIN(i, 4);
		for (j = width; j > 0; --j, ++fvscanstart, ++dvscanstart) {
			fp = fvscanstart;
			dp = dvscanstart;
			k = vscanlen;

			/* Process first sample in vertical scan. */
			jpc_rawrefpass_step(fp, dp, poshalf, neghalf, in,
			  vcausalflag);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process second sample in vertical scan. */
			jpc_rawrefpass_step(fp, dp, poshalf, neghalf, in, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process third sample in vertical scan. */
			jpc_rawrefpass_step(fp, dp, poshalf, neghalf, in, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process fourth sample in vertical scan. */
			jpc_rawrefpass_step(fp, dp, poshalf, neghalf, in, 0);
		}
	}
	return 0;
}

/******************************************************************************\
* Code for cleanup pass.
\******************************************************************************/

#define	jpc_clnpass_step(f, fp, frowstep, dp, oneplushalf, orient, mqdec, flabel, plabel, vcausalflag) \
{ \
	int v; \
flabel \
	if (!((f) & (JPC_SIG | JPC_VISIT))) { \
		jpc_mqdec_setcurctx((mqdec), JPC_GETZCCTXNO((f), (orient))); \
		JPC_T1D_GETBIT((mqdec), v, "CLN", "ZC"); \
		if (v) { \
plabel \
			/* Coefficient is significant. */ \
			jpc_mqdec_setcurctx((mqdec), JPC_GETSCCTXNO(f)); \
			JPC_T1D_GETBIT((mqdec), v, "CLN", "SC"); \
			v ^= JPC_GETSPB(f); \
			*(dp) = (v) ? (-(oneplushalf)) : (oneplushalf); \
			JPC_UPDATEFLAGS4((fp), (frowstep), v, (vcausalflag)); \
			*(fp) |= JPC_SIG; \
		} \
	} \
	/* XXX - Is this correct?  Can aggregation cause some VISIT bits not to be reset?  Check. */ \
	*(fp) &= ~JPC_VISIT; \
}

static int dec_clnpass(jpc_dec_t *dec, register jpc_mqdec_t *mqdec, int bitpos, int orient,
  int vcausalflag, int segsymflag, jas_matrix_t *flags, jas_matrix_t *data)
{
	int i;
	int j;
	int k;
	int vscanlen;
	int v;
	int half;
	int runlen;
	int f;
	int width;
	int height;
	int one;
	int oneplushalf;

	jpc_fix_t *fp;
	int frowstep;
	int fstripestep;
	jpc_fix_t *fstripestart;
	jpc_fix_t *fvscanstart;

	jpc_fix_t *dp;
	int drowstep;
	int dstripestep;
	jpc_fix_t *dstripestart;
	jpc_fix_t *dvscanstart;

	/* Avoid compiler warning about unused parameters. */
	dec = 0;

	one = 1 << bitpos;
	half = one >> 1;
	oneplushalf = one | half;

	width = jas_matrix_numcols(data);
	height = jas_matrix_numrows(data);

	frowstep = jas_matrix_rowstep(flags);
	drowstep = jas_matrix_rowstep(data);
	fstripestep = frowstep << 2;
	dstripestep = drowstep << 2;

	fstripestart = jas_matrix_getref(flags, 1, 1);
	dstripestart = jas_matrix_getref(data, 0, 0);
	for (i = 0; i < height; i += 4, fstripestart += fstripestep,
	  dstripestart += dstripestep) {
		fvscanstart = fstripestart;
		dvscanstart = dstripestart;
		vscanlen = JAS_MIN(4, height - i);
		for (j = width; j > 0; --j, ++fvscanstart, ++dvscanstart) {
			fp = fvscanstart;
			if (vscanlen >= 4 && (!((*fp) & (JPC_SIG | JPC_VISIT |
			  JPC_OTHSIGMSK))) && (fp += frowstep, !((*fp) & (JPC_SIG |
			  JPC_VISIT | JPC_OTHSIGMSK))) && (fp += frowstep, !((*fp) &
			  (JPC_SIG | JPC_VISIT | JPC_OTHSIGMSK))) && (fp += frowstep,
			  !((*fp) & (JPC_SIG | JPC_VISIT | JPC_OTHSIGMSK)))) {

				jpc_mqdec_setcurctx(mqdec, JPC_AGGCTXNO);
				JPC_T1D_GETBIT(mqdec, v, "CLN", "AGG");
				if (!v) {
					continue;
				}
				jpc_mqdec_setcurctx(mqdec, JPC_UCTXNO);
				JPC_T1D_GETBITNOSKEW(mqdec, v, "CLN", "RL");
				runlen = v;
				JPC_T1D_GETBITNOSKEW(mqdec, v, "CLN", "RL");
				runlen = (runlen << 1) | v;
				f = *(fp = fvscanstart + frowstep * runlen);
				dp = dvscanstart + drowstep * runlen;
				k = vscanlen - runlen;
				switch (runlen) {
				case 0:
					goto clnpass_partial0;
					break;
				case 1:
					goto clnpass_partial1;
					break;
				case 2:
					goto clnpass_partial2;
					break;
				case 3:
					goto clnpass_partial3;
					break;
				}
			} else {
				f = *(fp = fvscanstart);
				dp = dvscanstart;
				k = vscanlen;
				goto clnpass_full0;
			}

			/* Process first sample in vertical scan. */
			jpc_clnpass_step(f, fp, frowstep, dp, oneplushalf, orient,
			  mqdec, clnpass_full0:, clnpass_partial0:,
			  vcausalflag);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process second sample in vertical scan. */
			f = *fp;
			jpc_clnpass_step(f, fp, frowstep, dp, oneplushalf, orient,
				mqdec, ;, clnpass_partial1:, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process third sample in vertical scan. */
			f = *fp;
			jpc_clnpass_step(f, fp, frowstep, dp, oneplushalf, orient,
				mqdec, ;, clnpass_partial2:, 0);
			if (--k <= 0) {
				continue;
			}
			fp += frowstep;
			dp += drowstep;

			/* Process fourth sample in vertical scan. */
			f = *fp;
			jpc_clnpass_step(f, fp, frowstep, dp, oneplushalf, orient,
				mqdec, ;, clnpass_partial3:, 0);
		}
	}

	if (segsymflag) {
		int segsymval;
		segsymval = 0;
		jpc_mqdec_setcurctx(mqdec, JPC_UCTXNO);
		JPC_T1D_GETBITNOSKEW(mqdec, v, "CLN", "SEGSYM");
		segsymval = (segsymval << 1) | (v & 1);
		JPC_T1D_GETBITNOSKEW(mqdec, v, "CLN", "SEGSYM");
		segsymval = (segsymval << 1) | (v & 1);
		JPC_T1D_GETBITNOSKEW(mqdec, v, "CLN", "SEGSYM");
		segsymval = (segsymval << 1) | (v & 1);
		JPC_T1D_GETBITNOSKEW(mqdec, v, "CLN", "SEGSYM");
		segsymval = (segsymval << 1) | (v & 1);
		if (segsymval != 0xa) {
			jas_eprintf("warning: bad segmentation symbol\n");
		}
	}

	return 0;
}
