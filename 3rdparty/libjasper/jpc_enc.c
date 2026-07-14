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
 * $Id: jpc_enc.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#include "jasper/jas_types.h"
#include "jasper/jas_string.h"
#include "jasper/jas_malloc.h"
#include "jasper/jas_image.h"
#include "jasper/jas_fix.h"
#include "jasper/jas_tvp.h"
#include "jasper/jas_version.h"
#include "jasper/jas_math.h"
#include "jasper/jas_debug.h"

#include "jpc_flt.h"
#include "jpc_fix.h"
#include "jpc_tagtree.h"
#include "jpc_enc.h"
#include "jpc_cs.h"
#include "jpc_mct.h"
#include "jpc_tsfb.h"
#include "jpc_qmfb.h"
#include "jpc_t1enc.h"
#include "jpc_t2enc.h"
#include "jpc_cod.h"
#include "jpc_math.h"
#include "jpc_util.h"

/******************************************************************************\
*
\******************************************************************************/

#define JPC_POW2(n) \
    (1 << (n))

#define JPC_FLOORTOMULTPOW2(x, n) \
  (((n) > 0) ? ((x) & (~((1 << n) - 1))) : (x))
/* Round to the nearest multiple of the specified power of two in the
  direction of negative infinity. */

#define	JPC_CEILTOMULTPOW2(x, n) \
  (((n) > 0) ? JPC_FLOORTOMULTPOW2(((x) + (1 << (n)) - 1), n) : (x))
/* Round to the nearest multiple of the specified power of two in the
  direction of positive infinity. */

#define	JPC_POW2(n)	\
  (1 << (n))

jpc_enc_tile_t *jpc_enc_tile_create(jpc_enc_cp_t *cp, jas_image_t *image, int tileno);
void jpc_enc_tile_destroy(jpc_enc_tile_t *tile);

static jpc_enc_tcmpt_t *tcmpt_create(jpc_enc_tcmpt_t *tcmpt, jpc_enc_cp_t *cp,
  jas_image_t *image, jpc_enc_tile_t *tile);
static void tcmpt_destroy(jpc_enc_tcmpt_t *tcmpt);
static jpc_enc_rlvl_t *rlvl_create(jpc_enc_rlvl_t *rlvl, jpc_enc_cp_t *cp,
  jpc_enc_tcmpt_t *tcmpt, jpc_tsfb_band_t *bandinfos);
static void rlvl_destroy(jpc_enc_rlvl_t *rlvl);
static jpc_enc_band_t *band_create(jpc_enc_band_t *band, jpc_enc_cp_t *cp,
  jpc_enc_rlvl_t *rlvl, jpc_tsfb_band_t *bandinfos);
static void band_destroy(jpc_enc_band_t *bands);
static jpc_enc_prc_t *prc_create(jpc_enc_prc_t *prc, jpc_enc_cp_t *cp,
  jpc_enc_band_t *band);
static void prc_destroy(jpc_enc_prc_t *prcs);
static jpc_enc_cblk_t *cblk_create(jpc_enc_cblk_t *cblk, jpc_enc_cp_t *cp,
  jpc_enc_prc_t *prc);
static void cblk_destroy(jpc_enc_cblk_t *cblks);
int ratestrtosize(char *s, uint_fast32_t rawsize, uint_fast32_t *size);
static void pass_destroy(jpc_enc_pass_t *pass);
void jpc_enc_dump(jpc_enc_t *enc);

/******************************************************************************\
* Local prototypes.
\******************************************************************************/

int dump_passes(jpc_enc_pass_t *passes, int numpasses, jpc_enc_cblk_t *cblk);
void calcrdslopes(jpc_enc_cblk_t *cblk);
void dump_layeringinfo(jpc_enc_t *enc);
static int jpc_calcssexp(jpc_fix_t stepsize);
static int jpc_calcssmant(jpc_fix_t stepsize);
void jpc_quantize(jas_matrix_t *data, jpc_fix_t stepsize);
static int jpc_enc_encodemainhdr(jpc_enc_t *enc);
static int jpc_enc_encodemainbody(jpc_enc_t *enc);
int jpc_enc_encodetiledata(jpc_enc_t *enc);
jpc_enc_t *jpc_enc_create(jpc_enc_cp_t *cp, jas_stream_t *out, jas_image_t *image);
void jpc_enc_destroy(jpc_enc_t *enc);
static int jpc_enc_encodemainhdr(jpc_enc_t *enc);
static int jpc_enc_encodemainbody(jpc_enc_t *enc);
int jpc_enc_encodetiledata(jpc_enc_t *enc);
int rateallocate(jpc_enc_t *enc, int numlyrs, uint_fast32_t *cumlens);
int setins(int numvalues, jpc_flt_t *values, jpc_flt_t value);
static jpc_enc_cp_t *cp_create(char *optstr, jas_image_t *image);
void jpc_enc_cp_destroy(jpc_enc_cp_t *cp);
static uint_fast32_t jpc_abstorelstepsize(jpc_fix_t absdelta, int scaleexpn);

static uint_fast32_t jpc_abstorelstepsize(jpc_fix_t absdelta, int scaleexpn)
{
    int p;
    uint_fast32_t mant;
    uint_fast32_t expn;
    int n;

    if (absdelta < 0) {
        abort();
    }

    p = jpc_firstone(absdelta) - JPC_FIX_FRACBITS;
    n = 11 - jpc_firstone(absdelta);
    mant = ((n < 0) ? (absdelta >> (-n)) : (absdelta << n)) & 0x7ff;
    expn = scaleexpn - p;
    if (scaleexpn < p) {
        abort();
    }
    return JPC_QCX_EXPN(expn) | JPC_QCX_MANT(mant);
}

typedef enum {
    OPT_DEBUG,
    OPT_IMGAREAOFFX,
    OPT_IMGAREAOFFY,
    OPT_TILEGRDOFFX,
    OPT_TILEGRDOFFY,
    OPT_TILEWIDTH,
    OPT_TILEHEIGHT,
    OPT_PRCWIDTH,
    OPT_PRCHEIGHT,
    OPT_CBLKWIDTH,
    OPT_CBLKHEIGHT,
    OPT_MODE,
    OPT_PRG,
    OPT_NOMCT,
    OPT_MAXRLVLS,
    OPT_SOP,
    OPT_EPH,
    OPT_LAZY,
    OPT_TERMALL,
    OPT_SEGSYM,
    OPT_VCAUSAL,
    OPT_RESET,
    OPT_PTERM,
    OPT_NUMGBITS,
    OPT_RATE,
    OPT_ILYRRATES,
    OPT_JP2OVERHEAD
} optid_t;

jas_taginfo_t encopts[] = {
    {OPT_DEBUG, "debug"},
    {OPT_IMGAREAOFFX, "imgareatlx"},
    {OPT_IMGAREAOFFY, "imgareatly"},
    {OPT_TILEGRDOFFX, "tilegrdtlx"},
    {OPT_TILEGRDOFFY, "tilegrdtly"},
    {OPT_TILEWIDTH, "tilewidth"},
    {OPT_TILEHEIGHT, "tileheight"},
    {OPT_PRCWIDTH, "prcwidth"},
    {OPT_PRCHEIGHT, "prcheight"},
    {OPT_CBLKWIDTH, "cblkwidth"},
    {OPT_CBLKHEIGHT, "cblkheight"},
    {OPT_MODE, "mode"},
    {OPT_PRG, "prg"},
    {OPT_NOMCT, "nomct"},
    {OPT_MAXRLVLS, "numrlvls"},
    {OPT_SOP, "sop"},
    {OPT_EPH, "eph"},
    {OPT_LAZY, "lazy"},
    {OPT_TERMALL, "termall"},
    {OPT_SEGSYM, "segsym"},
    {OPT_VCAUSAL, "vcausal"},
    {OPT_PTERM, "pterm"},
    {OPT_RESET, "resetprob"},
    {OPT_NUMGBITS, "numgbits"},
    {OPT_RATE, "rate"},
    {OPT_ILYRRATES, "ilyrrates"},
    {OPT_JP2OVERHEAD, "_jp2overhead"},
    {-1, 0}
};

typedef enum {
    PO_L = 0,
    PO_R
} poid_t;


jas_taginfo_t prgordtab[] = {
    {JPC_COD_LRCPPRG, "lrcp"},
    {JPC_COD_RLCPPRG, "rlcp"},
    {JPC_COD_RPCLPRG, "rpcl"},
    {JPC_COD_PCRLPRG, "pcrl"},
    {JPC_COD_CPRLPRG, "cprl"},
    {-1, 0}
};

typedef enum {
    MODE_INT,
    MODE_REAL
} modeid_t;

jas_taginfo_t modetab[] = {
    {MODE_INT, "int"},
    {MODE_REAL, "real"},
    {-1, 0}
};

/******************************************************************************\
* The main encoder entry point.
\******************************************************************************/

int jpc_encode(jas_image_t *image, jas_stream_t *out, char *optstr)
{
    jpc_enc_t *enc;
    jpc_enc_cp_t *cp;

    enc = 0;
    cp = 0;

    jpc_initluts();

    if (!(cp = cp_create(optstr, image))) {
        jas_eprintf("invalid JP encoder options\n");
        goto error;
    }

    if (!(enc = jpc_enc_create(cp, out, image))) {
        goto error;
    }
    cp = 0;

    /* Encode the main header. */
    if (jpc_enc_encodemainhdr(enc)) {
        goto error;
    }

    /* Encode the main body.  This constitutes most of the encoding work. */
    if (jpc_enc_encodemainbody(enc)) {
        goto error;
    }

    /* Write EOC marker segment. */
    if (!(enc->mrk = jpc_ms_create(JPC_MS_EOC))) {
        goto error;
    }
    if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
        jas_eprintf("cannot write EOI marker\n");
        goto error;
    }
    jpc_ms_destroy(enc->mrk);
    enc->mrk = 0;

    if (jas_stream_flush(enc->out)) {
        goto error;
    }

    jpc_enc_destroy(enc);

    return 0;

error:
    if (cp) {
        jpc_enc_cp_destroy(cp);
    }
    if (enc) {
        jpc_enc_destroy(enc);
    }
    return -1;
}

/******************************************************************************\
* Option parsing code.
\******************************************************************************/

static jpc_enc_cp_t *cp_create(char *optstr, jas_image_t *image)
{
    jpc_enc_cp_t *cp;
    jas_tvparser_t *tvp;
    int ret;
    int numilyrrates;
    double *ilyrrates;
    int i;
    int tagid;
    jpc_enc_tcp_t *tcp;
    jpc_enc_tccp_t *tccp;
    jpc_enc_ccp_t *ccp;
    int cmptno;
    uint_fast16_t rlvlno;
    uint_fast16_t prcwidthexpn;
    uint_fast16_t prcheightexpn;
    bool enablemct;
    uint_fast32_t jp2overhead;
    uint_fast16_t lyrno;
    uint_fast32_t hsteplcm;
    uint_fast32_t vsteplcm;
    bool mctvalid;

    tvp = 0;
    cp = 0;
    ilyrrates = 0;
    numilyrrates = 0;

    if (!(cp = jas_malloc(sizeof(jpc_enc_cp_t)))) {
        goto error;
    }

    prcwidthexpn = 15;
    prcheightexpn = 15;
    enablemct = true;
    jp2overhead = 0;

    cp->ccps = 0;
    cp->debug = 0;
    cp->imgareatlx = UINT_FAST32_MAX;
    cp->imgareatly = UINT_FAST32_MAX;
    cp->refgrdwidth = 0;
    cp->refgrdheight = 0;
    cp->tilegrdoffx = UINT_FAST32_MAX;
    cp->tilegrdoffy = UINT_FAST32_MAX;
    cp->tilewidth = 0;
    cp->tileheight = 0;
    cp->numcmpts = jas_image_numcmpts(image);

    hsteplcm = 1;
    vsteplcm = 1;
    for (cmptno = 0; cmptno < jas_image_numcmpts(image); ++cmptno) {
        if (jas_image_cmptbrx(image, cmptno) + jas_image_cmpthstep(image, cmptno) <=
          jas_image_brx(image) || jas_image_cmptbry(image, cmptno) +
          jas_image_cmptvstep(image, cmptno) <= jas_image_bry(image)) {
            jas_eprintf("unsupported image type\n");
            goto error;
        }
        /* Note: We ought to be calculating the LCMs here.  Fix some day. */
        hsteplcm *= jas_image_cmpthstep(image, cmptno);
        vsteplcm *= jas_image_cmptvstep(image, cmptno);
    }

    if (!(cp->ccps = jas_alloc2(cp->numcmpts, sizeof(jpc_enc_ccp_t)))) {
        goto error;
    }
    for (cmptno = 0, ccp = cp->ccps; cmptno < JAS_CAST(int, cp->numcmpts); ++cmptno,
      ++ccp) {
        ccp->sampgrdstepx = jas_image_cmpthstep(image, cmptno);
        ccp->sampgrdstepy = jas_image_cmptvstep(image, cmptno);
        /* XXX - this isn't quite correct for more general image */
        ccp->sampgrdsubstepx = 0;
        ccp->sampgrdsubstepx = 0;
        ccp->prec = jas_image_cmptprec(image, cmptno);
        ccp->sgnd = jas_image_cmptsgnd(image, cmptno);
        ccp->numstepsizes = 0;
        memset(ccp->stepsizes, 0, sizeof(ccp->stepsizes));
    }

    cp->rawsize = jas_image_rawsize(image);
    cp->totalsize = UINT_FAST32_MAX;

    tcp = &cp->tcp;
    tcp->csty = 0;
    tcp->intmode = true;
    tcp->prg = JPC_COD_LRCPPRG;
    tcp->numlyrs = 1;
    tcp->ilyrrates = 0;

    tccp = &cp->tccp;
    tccp->csty = 0;
    tccp->maxrlvls = 6;
    tccp->cblkwidthexpn = 6;
    tccp->cblkheightexpn = 6;
    tccp->cblksty = 0;
    tccp->numgbits = 2;

    if (!(tvp = jas_tvparser_create(optstr ? optstr : ""))) {
        goto error;
    }

    while (!(ret = jas_tvparser_next(tvp))) {
        switch (jas_taginfo_nonull(jas_taginfos_lookup(encopts,
          jas_tvparser_gettag(tvp)))->id) {
        case OPT_DEBUG:
            cp->debug = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_IMGAREAOFFX:
            cp->imgareatlx = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_IMGAREAOFFY:
            cp->imgareatly = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_TILEGRDOFFX:
            cp->tilegrdoffx = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_TILEGRDOFFY:
            cp->tilegrdoffy = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_TILEWIDTH:
            cp->tilewidth = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_TILEHEIGHT:
            cp->tileheight = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_PRCWIDTH:
            prcwidthexpn = jpc_floorlog2(atoi(jas_tvparser_getval(tvp)));
            break;
        case OPT_PRCHEIGHT:
            prcheightexpn = jpc_floorlog2(atoi(jas_tvparser_getval(tvp)));
            break;
        case OPT_CBLKWIDTH:
            tccp->cblkwidthexpn =
              jpc_floorlog2(atoi(jas_tvparser_getval(tvp)));
            break;
        case OPT_CBLKHEIGHT:
            tccp->cblkheightexpn =
              jpc_floorlog2(atoi(jas_tvparser_getval(tvp)));
            break;
        case OPT_MODE:
            if ((tagid = jas_taginfo_nonull(jas_taginfos_lookup(modetab,
              jas_tvparser_getval(tvp)))->id) < 0) {
                jas_eprintf("ignoring invalid mode %s\n",
                  jas_tvparser_getval(tvp));
            } else {
                tcp->intmode = (tagid == MODE_INT);
            }
            break;
        case OPT_PRG:
            if ((tagid = jas_taginfo_nonull(jas_taginfos_lookup(prgordtab,
              jas_tvparser_getval(tvp)))->id) < 0) {
                jas_eprintf("ignoring invalid progression order %s\n",
                  jas_tvparser_getval(tvp));
            } else {
                tcp->prg = tagid;
            }
            break;
        case OPT_NOMCT:
            enablemct = false;
            break;
        case OPT_MAXRLVLS:
            tccp->maxrlvls = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_SOP:
            cp->tcp.csty |= JPC_COD_SOP;
            break;
        case OPT_EPH:
            cp->tcp.csty |= JPC_COD_EPH;
            break;
        case OPT_LAZY:
            tccp->cblksty |= JPC_COX_LAZY;
            break;
        case OPT_TERMALL:
            tccp->cblksty |= JPC_COX_TERMALL;
            break;
        case OPT_SEGSYM:
            tccp->cblksty |= JPC_COX_SEGSYM;
            break;
        case OPT_VCAUSAL:
            tccp->cblksty |= JPC_COX_VSC;
            break;
        case OPT_RESET:
            tccp->cblksty |= JPC_COX_RESET;
            break;
        case OPT_PTERM:
            tccp->cblksty |= JPC_COX_PTERM;
            break;
        case OPT_NUMGBITS:
            cp->tccp.numgbits = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_RATE:
            if (ratestrtosize(jas_tvparser_getval(tvp), cp->rawsize,
              &cp->totalsize)) {
                jas_eprintf("ignoring bad rate specifier %s\n",
                  jas_tvparser_getval(tvp));
            }
            break;
        case OPT_ILYRRATES:
            if (jpc_atoaf(jas_tvparser_getval(tvp), &numilyrrates,
              &ilyrrates)) {
                jas_eprintf("warning: invalid intermediate layer rates specifier ignored (%s)\n",
                  jas_tvparser_getval(tvp));
            }
            break;

        case OPT_JP2OVERHEAD:
            jp2overhead = atoi(jas_tvparser_getval(tvp));
            break;
        default:
            jas_eprintf("warning: ignoring invalid option %s\n",
             jas_tvparser_gettag(tvp));
            break;
        }
    }

    jas_tvparser_destroy(tvp);
    tvp = 0;

    if (cp->totalsize != UINT_FAST32_MAX) {
        cp->totalsize = (cp->totalsize > jp2overhead) ?
          (cp->totalsize - jp2overhead) : 0;
    }

    if (cp->imgareatlx == UINT_FAST32_MAX) {
        cp->imgareatlx = 0;
    } else {
        if (hsteplcm != 1) {
            jas_eprintf("warning: overriding imgareatlx value\n");
        }
        cp->imgareatlx *= hsteplcm;
    }
    if (cp->imgareatly == UINT_FAST32_MAX) {
        cp->imgareatly = 0;
    } else {
        if (vsteplcm != 1) {
            jas_eprintf("warning: overriding imgareatly value\n");
        }
        cp->imgareatly *= vsteplcm;
    }
    cp->refgrdwidth = cp->imgareatlx + jas_image_width(image);
    cp->refgrdheight = cp->imgareatly + jas_image_height(image);
    if (cp->tilegrdoffx == UINT_FAST32_MAX) {
        cp->tilegrdoffx = cp->imgareatlx;
    }
    if (cp->tilegrdoffy == UINT_FAST32_MAX) {
        cp->tilegrdoffy = cp->imgareatly;
    }
    if (!cp->tilewidth) {
        cp->tilewidth = cp->refgrdwidth - cp->tilegrdoffx;
    }
    if (!cp->tileheight) {
        cp->tileheight = cp->refgrdheight - cp->tilegrdoffy;
    }

    if (cp->numcmpts == 3) {
        mctvalid = true;
        for (cmptno = 0; cmptno < jas_image_numcmpts(image); ++cmptno) {
            if (jas_image_cmptprec(image, cmptno) != jas_image_cmptprec(image, 0) ||
              jas_image_cmptsgnd(image, cmptno) != jas_image_cmptsgnd(image, 0) ||
              jas_image_cmptwidth(image, cmptno) != jas_image_cmptwidth(image, 0) ||
              jas_image_cmptheight(image, cmptno) != jas_image_cmptheight(image, 0)) {
                mctvalid = false;
            }
        }
    } else {
        mctvalid = false;
    }
    if (mctvalid && enablemct && jas_clrspc_fam(jas_image_clrspc(image)) != JAS_CLRSPC_FAM_RGB) {
        jas_eprintf("warning: color space apparently not RGB\n");
    }
    if (mctvalid && enablemct && jas_clrspc_fam(jas_image_clrspc(image)) == JAS_CLRSPC_FAM_RGB) {
        tcp->mctid = (tcp->intmode) ? (JPC_MCT_RCT) : (JPC_MCT_ICT);
    } else {
        tcp->mctid = JPC_MCT_NONE;
    }
    tccp->qmfbid = (tcp->intmode) ? (JPC_COX_RFT) : (JPC_COX_INS);

    for (rlvlno = 0; rlvlno < tccp->maxrlvls; ++rlvlno) {
        tccp->prcwidthexpns[rlvlno] = prcwidthexpn;
        tccp->prcheightexpns[rlvlno] = prcheightexpn;
    }
    if (prcwidthexpn != 15 || prcheightexpn != 15) {
        tccp->csty |= JPC_COX_PRT;
    }

    /* Ensure that the tile width and height is valid. */
    if (!cp->tilewidth) {
        jas_eprintf("invalid tile width %lu\n", (unsigned long)
          cp->tilewidth);
        goto error;
    }
    if (!cp->tileheight) {
        jas_eprintf("invalid tile height %lu\n", (unsigned long)
          cp->tileheight);
        goto error;
    }

    /* Ensure that the tile grid offset is valid. */
    if (cp->tilegrdoffx > cp->imgareatlx ||
      cp->tilegrdoffy > cp->imgareatly ||
      cp->tilegrdoffx + cp->tilewidth < cp->imgareatlx ||
      cp->tilegrdoffy + cp->tileheight < cp->imgareatly) {
        jas_eprintf("invalid tile grid offset (%lu, %lu)\n",
          (unsigned long) cp->tilegrdoffx, (unsigned long)
          cp->tilegrdoffy);
        goto error;
    }

    cp->numhtiles = JPC_CEILDIV(cp->refgrdwidth - cp->tilegrdoffx,
      cp->tilewidth);
    cp->numvtiles = JPC_CEILDIV(cp->refgrdheight - cp->tilegrdoffy,
      cp->tileheight);
    cp->numtiles = cp->numhtiles * cp->numvtiles;

    if (ilyrrates && numilyrrates > 0) {
        tcp->numlyrs = numilyrrates + 1;
        if (!(tcp->ilyrrates = jas_alloc2((tcp->numlyrs - 1),
          sizeof(jpc_fix_t)))) {
            goto error;
        }
        for (i = 0; i < JAS_CAST(int, tcp->numlyrs - 1); ++i) {
            tcp->ilyrrates[i] = jpc_dbltofix(ilyrrates[i]);
        }
    }

    /* Ensure that the integer mode is used in the case of lossless
      coding. */
    if (cp->totalsize == UINT_FAST32_MAX && (!cp->tcp.intmode)) {
        jas_eprintf("cannot use real mode for lossless coding\n");
        goto error;
    }

    /* Ensure that the precinct width is valid. */
    if (prcwidthexpn > 15) {
        jas_eprintf("invalid precinct width\n");
        goto error;
    }

    /* Ensure that the precinct height is valid. */
    if (prcheightexpn > 15) {
        jas_eprintf("invalid precinct height\n");
        goto error;
    }

    /* Ensure that the code block width is valid. */
    if (cp->tccp.cblkwidthexpn < 2 || cp->tccp.cblkwidthexpn > 12) {
        jas_eprintf("invalid code block width %d\n",
          JPC_POW2(cp->tccp.cblkwidthexpn));
        goto error;
    }

    /* Ensure that the code block height is valid. */
    if (cp->tccp.cblkheightexpn < 2 || cp->tccp.cblkheightexpn > 12) {
        jas_eprintf("invalid code block height %d\n",
          JPC_POW2(cp->tccp.cblkheightexpn));
        goto error;
    }

    /* Ensure that the code block size is not too large. */
    if (cp->tccp.cblkwidthexpn + cp->tccp.cblkheightexpn > 12) {
        jas_eprintf("code block size too large\n");
        goto error;
    }

    /* Ensure that the number of layers is valid. */
    if (cp->tcp.numlyrs > 16384) {
        jas_eprintf("too many layers\n");
        goto error;
    }

    /* There must be at least one resolution level. */
    if (cp->tccp.maxrlvls < 1) {
        jas_eprintf("must be at least one resolution level\n");
        goto error;
    }

    /* Ensure that the number of guard bits is valid. */
    if (cp->tccp.numgbits > 8) {
        jas_eprintf("invalid number of guard bits\n");
        goto error;
    }

    /* Ensure that the rate is within the legal range. */
    if (cp->totalsize != UINT_FAST32_MAX && cp->totalsize > cp->rawsize) {
        jas_eprintf("warning: specified rate is unreasonably large (%lu > %lu)\n", (unsigned long) cp->totalsize, (unsigned long) cp->rawsize);
    }

    /* Ensure that the intermediate layer rates are valid. */
    if (tcp->numlyrs > 1) {
        /* The intermediate layers rates must increase monotonically. */
        for (lyrno = 0; lyrno + 2 < tcp->numlyrs; ++lyrno) {
            if (tcp->ilyrrates[lyrno] >= tcp->ilyrrates[lyrno + 1]) {
                jas_eprintf("intermediate layer rates must increase monotonically\n");
                goto error;
            }
        }
        /* The intermediate layer rates must be less than the overall rate. */
        if (cp->totalsize != UINT_FAST32_MAX) {
            for (lyrno = 0; lyrno < tcp->numlyrs - 1; ++lyrno) {
                if (jpc_fixtodbl(tcp->ilyrrates[lyrno]) > ((double) cp->totalsize)
                  / cp->rawsize) {
                    jas_eprintf("warning: intermediate layer rates must be less than overall rate\n");
                    goto error;
                }
            }
        }
    }

    if (ilyrrates) {
        jas_free(ilyrrates);
    }

    return cp;

error:

    if (ilyrrates) {
        jas_free(ilyrrates);
    }
    if (tvp) {
        jas_tvparser_destroy(tvp);
    }
    if (cp) {
        jpc_enc_cp_destroy(cp);
    }
    return 0;
}

void jpc_enc_cp_destroy(jpc_enc_cp_t *cp)
{
    if (cp->ccps) {
        if (cp->tcp.ilyrrates) {
            jas_free(cp->tcp.ilyrrates);
        }
        jas_free(cp->ccps);
    }
    jas_free(cp);
}

int ratestrtosize(char *s, uint_fast32_t rawsize, uint_fast32_t *size)
{
    char *cp;
    jpc_flt_t f;

    /* Note: This function must not modify output size on failure. */
    if ((cp = strchr(s, 'B'))) {
        *size = atoi(s);
    } else {
        f = atof(s);
        if (f < 0) {
            *size = 0;
        } else if (f > 1.0) {
            *size = rawsize + 1;
        } else {
            *size = f * rawsize;
        }
    }
    return 0;
}

/******************************************************************************\
* Encoder constructor and destructor.
\******************************************************************************/

jpc_enc_t *jpc_enc_create(jpc_enc_cp_t *cp, jas_stream_t *out, jas_image_t *image)
{
    jpc_enc_t *enc;

    enc = 0;

    if (!(enc = jas_malloc(sizeof(jpc_enc_t)))) {
        goto error;
    }

    enc->image = image;
    enc->out = out;
    enc->cp = cp;
    enc->cstate = 0;
    enc->tmpstream = 0;
    enc->mrk = 0;
    enc->curtile = 0;

    if (!(enc->cstate = jpc_cstate_create())) {
        goto error;
    }
    enc->len = 0;
    enc->mainbodysize = 0;

    return enc;

error:

    if (enc) {
        jpc_enc_destroy(enc);
    }
    return 0;
}

void jpc_enc_destroy(jpc_enc_t *enc)
{
    /* The image object (i.e., enc->image) and output stream object
    (i.e., enc->out) are created outside of the encoder.
    Therefore, they must not be destroyed here. */

    if (enc->curtile) {
        jpc_enc_tile_destroy(enc->curtile);
    }
    if (enc->cp) {
        jpc_enc_cp_destroy(enc->cp);
    }
    if (enc->cstate) {
        jpc_cstate_destroy(enc->cstate);
    }
    if (enc->tmpstream) {
        jas_stream_close(enc->tmpstream);
    }

    jas_free(enc);
}

/******************************************************************************\
* Code.
\******************************************************************************/

static int jpc_calcssmant(jpc_fix_t stepsize)
{
    int n;
    int e;
    int m;

    n = jpc_firstone(stepsize);
    e = n - JPC_FIX_FRACBITS;
    if (n >= 11) {
        m = (stepsize >> (n - 11)) & 0x7ff;
    } else {
        m = (stepsize & ((1 << n) - 1)) << (11 - n);
    }
    return m;
}

static int jpc_calcssexp(jpc_fix_t stepsize)
{
    return jpc_firstone(stepsize) - JPC_FIX_FRACBITS;
}

static int jpc_enc_encodemainhdr(jpc_enc_t *enc)
{
    jpc_siz_t *siz;
    jpc_cod_t *cod;
    jpc_qcd_t *qcd;
    int i;
long startoff;
long mainhdrlen;
    jpc_enc_cp_t *cp;
    jpc_qcc_t *qcc;
    jpc_enc_tccp_t *tccp;
    uint_fast16_t cmptno;
    jpc_tsfb_band_t bandinfos[JPC_MAXBANDS];
    jpc_fix_t mctsynweight;
    jpc_enc_tcp_t *tcp;
    jpc_tsfb_t *tsfb;
    jpc_tsfb_band_t *bandinfo;
    uint_fast16_t numbands;
    uint_fast16_t bandno;
    uint_fast16_t rlvlno;
    uint_fast16_t analgain;
    jpc_fix_t absstepsize;
    char buf[1024];
    jpc_com_t *com;

    cp = enc->cp;

startoff = jas_stream_getrwcount(enc->out);

    /* Write SOC marker segment. */
    if (!(enc->mrk = jpc_ms_create(JPC_MS_SOC))) {
        return -1;
    }
    if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
        jas_eprintf("cannot write SOC marker\n");
        return -1;
    }
    jpc_ms_destroy(enc->mrk);
    enc->mrk = 0;

    /* Write SIZ marker segment. */
    if (!(enc->mrk = jpc_ms_create(JPC_MS_SIZ))) {
        return -1;
    }
    siz = &enc->mrk->parms.siz;
    siz->caps = 0;
    siz->xoff = cp->imgareatlx;
    siz->yoff = cp->imgareatly;
    siz->width = cp->refgrdwidth;
    siz->height = cp->refgrdheight;
    siz->tilexoff = cp->tilegrdoffx;
    siz->tileyoff = cp->tilegrdoffy;
    siz->tilewidth = cp->tilewidth;
    siz->tileheight = cp->tileheight;
    siz->numcomps = cp->numcmpts;
    siz->comps = jas_alloc2(siz->numcomps, sizeof(jpc_sizcomp_t));
    assert(siz->comps);
    for (i = 0; i < JAS_CAST(int, cp->numcmpts); ++i) {
        siz->comps[i].prec = cp->ccps[i].prec;
        siz->comps[i].sgnd = cp->ccps[i].sgnd;
        siz->comps[i].hsamp = cp->ccps[i].sampgrdstepx;
        siz->comps[i].vsamp = cp->ccps[i].sampgrdstepy;
    }
    if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
        jas_eprintf("cannot write SIZ marker\n");
        return -1;
    }
    jpc_ms_destroy(enc->mrk);
    enc->mrk = 0;

    if (!(enc->mrk = jpc_ms_create(JPC_MS_COM))) {
        return -1;
    }
    sprintf(buf, "Creator: JasPer Version %s", jas_getversion());
    com = &enc->mrk->parms.com;
    com->len = strlen(buf);
    com->regid = JPC_COM_LATIN;
    if (!(com->data = JAS_CAST(uchar *, jas_strdup(buf)))) {
        abort();
    }
    if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
        jas_eprintf("cannot write COM marker\n");
        return -1;
    }
    jpc_ms_destroy(enc->mrk);
    enc->mrk = 0;

#if 0
    if (!(enc->mrk = jpc_ms_create(JPC_MS_CRG))) {
        return -1;
    }
    crg = &enc->mrk->parms.crg;
    crg->comps = jas_alloc2(crg->numcomps, sizeof(jpc_crgcomp_t));
    if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
        jas_eprintf("cannot write CRG marker\n");
        return -1;
    }
    jpc_ms_destroy(enc->mrk);
    enc->mrk = 0;
#endif

    tcp = &cp->tcp;
    tccp = &cp->tccp;
    for (cmptno = 0; cmptno < cp->numcmpts; ++cmptno) {
        tsfb = jpc_cod_gettsfb(tccp->qmfbid, tccp->maxrlvls - 1);
        jpc_tsfb_getbands(tsfb, 0, 0, 1 << tccp->maxrlvls, 1 << tccp->maxrlvls,
          bandinfos);
        jpc_tsfb_destroy(tsfb);
        mctsynweight = jpc_mct_getsynweight(tcp->mctid, cmptno);
        numbands = 3 * tccp->maxrlvls - 2;
        for (bandno = 0, bandinfo = bandinfos; bandno < numbands;
          ++bandno, ++bandinfo) {
            rlvlno = (bandno) ? ((bandno - 1) / 3 + 1) : 0;
            analgain = JPC_NOMINALGAIN(tccp->qmfbid, tccp->maxrlvls,
              rlvlno, bandinfo->orient);
            if (!tcp->intmode) {
                absstepsize = jpc_fix_div(jpc_inttofix(1 <<
                  (analgain + 1)), bandinfo->synenergywt);
            } else {
                absstepsize = jpc_inttofix(1);
            }
            cp->ccps[cmptno].stepsizes[bandno] =
              jpc_abstorelstepsize(absstepsize,
              cp->ccps[cmptno].prec + analgain);
        }
        cp->ccps[cmptno].numstepsizes = numbands;
    }

    if (!(enc->mrk = jpc_ms_create(JPC_MS_COD))) {
        return -1;
    }
    cod = &enc->mrk->parms.cod;
    cod->csty = cp->tccp.csty | cp->tcp.csty;
    cod->compparms.csty = cp->tccp.csty | cp->tcp.csty;
    cod->compparms.numdlvls = cp->tccp.maxrlvls - 1;
    cod->compparms.numrlvls = cp->tccp.maxrlvls;
    cod->prg = cp->tcp.prg;
    cod->numlyrs = cp->tcp.numlyrs;
    cod->compparms.cblkwidthval = JPC_COX_CBLKSIZEEXPN(cp->tccp.cblkwidthexpn);
    cod->compparms.cblkheightval = JPC_COX_CBLKSIZEEXPN(cp->tccp.cblkheightexpn);
    cod->compparms.cblksty = cp->tccp.cblksty;
    cod->compparms.qmfbid = cp->tccp.qmfbid;
    cod->mctrans = (cp->tcp.mctid != JPC_MCT_NONE);
    if (tccp->csty & JPC_COX_PRT) {
        for (rlvlno = 0; rlvlno < tccp->maxrlvls; ++rlvlno) {
            cod->compparms.rlvls[rlvlno].parwidthval = tccp->prcwidthexpns[rlvlno];
            cod->compparms.rlvls[rlvlno].parheightval = tccp->prcheightexpns[rlvlno];
        }
    }
    if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
        jas_eprintf("cannot write COD marker\n");
        return -1;
    }
    jpc_ms_destroy(enc->mrk);
    enc->mrk = 0;

    if (!(enc->mrk = jpc_ms_create(JPC_MS_QCD))) {
        return -1;
    }
    qcd = &enc->mrk->parms.qcd;
    qcd->compparms.qntsty = (tccp->qmfbid == JPC_COX_INS) ?
      JPC_QCX_SEQNT : JPC_QCX_NOQNT;
    qcd->compparms.numstepsizes = cp->ccps[0].numstepsizes;
    qcd->compparms.numguard = cp->tccp.numgbits;
    qcd->compparms.stepsizes = cp->ccps[0].stepsizes;
    if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
        return -1;
    }
    /* We do not want the step size array to be freed! */
    qcd->compparms.stepsizes = 0;
    jpc_ms_destroy(enc->mrk);
    enc->mrk = 0;

    tccp = &cp->tccp;
    for (cmptno = 1; cmptno < cp->numcmpts; ++cmptno) {
        if (!(enc->mrk = jpc_ms_create(JPC_MS_QCC))) {
            return -1;
        }
        qcc = &enc->mrk->parms.qcc;
        qcc->compno = cmptno;
        qcc->compparms.qntsty = (tccp->qmfbid == JPC_COX_INS) ?
          JPC_QCX_SEQNT : JPC_QCX_NOQNT;
        qcc->compparms.numstepsizes = cp->ccps[cmptno].numstepsizes;
        qcc->compparms.numguard = cp->tccp.numgbits;
        qcc->compparms.stepsizes = cp->ccps[cmptno].stepsizes;
        if (jpc_putms(enc->out, enc->cstate, enc->mrk)) {
            return -1;
        }
        /* We do not want the step size array to be freed! */
        qcc->compparms.stepsizes = 0;
        jpc_ms_destroy(enc->mrk);
        enc->mrk = 0;
    }

#define MAINTLRLEN	2
    mainhdrlen = jas_stream_getrwcount(enc->out) - startoff;
    enc->len += mainhdrlen;
    if (enc->cp->totalsize != UINT_FAST32_MAX) {
        uint_fast32_t overhead;
        overhead = mainhdrlen + MAINTLRLEN;
        enc->mainbodysize = (enc->cp->totalsize >= overhead) ?
          (enc->cp->totalsize - overhead) : 0;
    } else {
        enc->mainbodysize = UINT_FAST32_MAX;
    }

    return 0;
}

static int jpc_enc_encodemainbody(jpc_enc_t *enc)
{
    int tileno;
    int tilex;
    int tiley;
    int i;
    jpc_sot_t *sot;
    jpc_enc_tcmpt_t *comp;
    jpc_enc_tcmpt_t *endcomps;
    jpc_enc_band_t *band;
    jpc_enc_band_t *endbands;
    jpc_enc_rlvl_t *lvl;
    int rlvlno;
    jpc_qcc_t *qcc;
    jpc_cod_t *cod;
    int adjust;
    int j;
    int absbandno;
    long numbytes;
    long tilehdrlen;
    long tilelen;
    jpc_enc_tile_t *tile;
    jpc_enc_cp_t *cp;
    double rho;
    int lyrno;
    int cmptno;
    int samestepsizes;
    jpc_enc_ccp_t *ccps;
    jpc_enc_tccp_t *tccp;
int bandno;
uint_fast32_t x;
uint_fast32_t y;
int mingbits;
int actualnumbps;
jpc_fix_t mxmag;
jpc_fix_t mag;
int numgbits;

    cp = enc->cp;

    /* Avoid compile warnings. */
    numbytes = 0;

    for (tileno = 0; tileno < JAS_CAST(int, cp->numtiles); ++tileno) {
        tilex = tileno % cp->numhtiles;
        tiley = tileno / cp->numhtiles;

        if (!(enc->curtile = jpc_enc_tile_create(enc->cp, enc->image, tileno))) {
            abort();
        }

        tile = enc->curtile;

        if (jas_getdbglevel() >= 10) {
            jpc_enc_dump(enc);
        }

        endcomps = &tile->tcmpts[tile->numtcmpts];
        for (cmptno = 0, comp = tile->tcmpts; cmptno < tile->numtcmpts; ++cmptno, ++comp) {
            if (!cp->ccps[cmptno].sgnd) {
                adjust = 1 << (cp->ccps[cmptno].prec - 1);
                for (i = 0; i < jas_matrix_numrows(comp->data); ++i) {
                    for (j = 0; j < jas_matrix_numcols(comp->data); ++j) {
                        *jas_matrix_getref(comp->data, i, j) -= adjust;
                    }
                }
            }
        }

        if (!tile->intmode) {
                endcomps = &tile->tcmpts[tile->numtcmpts];
                for (comp = tile->tcmpts; comp != endcomps; ++comp) {
                    jas_matrix_asl(comp->data, JPC_FIX_FRACBITS);
                }
        }

        switch (tile->mctid) {
        case JPC_MCT_RCT:
assert(jas_image_numcmpts(enc->image) == 3);
            jpc_rct(tile->tcmpts[0].data, tile->tcmpts[1].data,
              tile->tcmpts[2].data);
            break;
        case JPC_MCT_ICT:
assert(jas_image_numcmpts(enc->image) == 3);
            jpc_ict(tile->tcmpts[0].data, tile->tcmpts[1].data,
              tile->tcmpts[2].data);
            break;
        default:
            break;
        }

        for (i = 0; i < jas_image_numcmpts(enc->image); ++i) {
            comp = &tile->tcmpts[i];
            jpc_tsfb_analyze(comp->tsfb, comp->data);

        }


        endcomps = &tile->tcmpts[tile->numtcmpts];
        for (cmptno = 0, comp = tile->tcmpts; comp != endcomps; ++cmptno, ++comp) {
            mingbits = 0;
            absbandno = 0;
            /* All bands must have a corresponding quantizer step size,
              even if they contain no samples and are never coded. */
            /* Some bands may not be hit by the loop below, so we must
              initialize all of the step sizes to a sane value. */
            memset(comp->stepsizes, 0, sizeof(comp->stepsizes));
            for (rlvlno = 0, lvl = comp->rlvls; rlvlno < comp->numrlvls; ++rlvlno, ++lvl) {
                if (!lvl->bands) {
                    absbandno += rlvlno ? 3 : 1;
                    continue;
                }
                endbands = &lvl->bands[lvl->numbands];
                for (band = lvl->bands; band != endbands; ++band) {
                    if (!band->data) {
                        ++absbandno;
                        continue;
                    }
                    actualnumbps = 0;
                    mxmag = 0;
                    for (y = 0; y < JAS_CAST(uint_fast32_t, jas_matrix_numrows(band->data)); ++y) {
                        for (x = 0; x < JAS_CAST(uint_fast32_t, jas_matrix_numcols(band->data)); ++x) {
                            mag = abs(jas_matrix_get(band->data, y, x));
                            if (mag > mxmag) {
                                mxmag = mag;
                            }
                        }
                    }
                    if (tile->intmode) {
                        actualnumbps = jpc_firstone(mxmag) + 1;
                    } else {
                        actualnumbps = jpc_firstone(mxmag) + 1 - JPC_FIX_FRACBITS;
                    }
                    numgbits = actualnumbps - (cp->ccps[cmptno].prec - 1 +
                      band->analgain);
#if 0
jas_eprintf("%d %d mag=%d actual=%d numgbits=%d\n", cp->ccps[cmptno].prec, band->analgain, mxmag, actualnumbps, numgbits);
#endif
                    if (numgbits > mingbits) {
                        mingbits = numgbits;
                    }
                    if (!tile->intmode) {
                        band->absstepsize = jpc_fix_div(jpc_inttofix(1
                          << (band->analgain + 1)),
                          band->synweight);
                    } else {
                        band->absstepsize = jpc_inttofix(1);
                    }
                    band->stepsize = jpc_abstorelstepsize(
                      band->absstepsize, cp->ccps[cmptno].prec +
                      band->analgain);
                    band->numbps = cp->tccp.numgbits +
                      JPC_QCX_GETEXPN(band->stepsize) - 1;

                    if ((!tile->intmode) && band->data) {
                        jpc_quantize(band->data, band->absstepsize);
                    }

                    comp->stepsizes[absbandno] = band->stepsize;
                    ++absbandno;
                }
            }

            assert(JPC_FIX_FRACBITS >= JPC_NUMEXTRABITS);
            if (!tile->intmode) {
                jas_matrix_divpow2(comp->data, JPC_FIX_FRACBITS - JPC_NUMEXTRABITS);
            } else {
                jas_matrix_asl(comp->data, JPC_NUMEXTRABITS);
            }

#if 0
jas_eprintf("mingbits %d\n", mingbits);
#endif
            if (mingbits > cp->tccp.numgbits) {
                jas_eprintf("error: too few guard bits (need at least %d)\n",
                  mingbits);
                return -1;
            }
        }

        if (!(enc->tmpstream = jas_stream_memopen(0, 0))) {
            jas_eprintf("cannot open tmp file\n");
            return -1;
        }

        /* Write the tile header. */
        if (!(enc->mrk = jpc_ms_create(JPC_MS_SOT))) {
            return -1;
        }
        sot = &enc->mrk->parms.sot;
        sot->len = 0;
        sot->tileno = tileno;
        sot->partno = 0;
        sot->numparts = 1;
        if (jpc_putms(enc->tmpstream, enc->cstate, enc->mrk)) {
            jas_eprintf("cannot write SOT marker\n");
            return -1;
        }
        jpc_ms_destroy(enc->mrk);
        enc->mrk = 0;

/************************************************************************/
/************************************************************************/
/************************************************************************/

        tccp = &cp->tccp;
        for (cmptno = 0; cmptno < JAS_CAST(int, cp->numcmpts); ++cmptno) {
            comp = &tile->tcmpts[cmptno];
            if (comp->numrlvls != tccp->maxrlvls) {
                if (!(enc->mrk = jpc_ms_create(JPC_MS_COD))) {
                    return -1;
                }
/* XXX = this is not really correct. we are using comp #0's precint sizes
and other characteristics */
                comp = &tile->tcmpts[0];
                cod = &enc->mrk->parms.cod;
                cod->compparms.csty = 0;
                cod->compparms.numdlvls = comp->numrlvls - 1;
                cod->prg = tile->prg;
                cod->numlyrs = tile->numlyrs;
                cod->compparms.cblkwidthval = JPC_COX_CBLKSIZEEXPN(comp->cblkwidthexpn);
                cod->compparms.cblkheightval = JPC_COX_CBLKSIZEEXPN(comp->cblkheightexpn);
                cod->compparms.cblksty = comp->cblksty;
                cod->compparms.qmfbid = comp->qmfbid;
                cod->mctrans = (tile->mctid != JPC_MCT_NONE);
                for (i = 0; i < comp->numrlvls; ++i) {
                    cod->compparms.rlvls[i].parwidthval = comp->rlvls[i].prcwidthexpn;
                    cod->compparms.rlvls[i].parheightval = comp->rlvls[i].prcheightexpn;
                }
                if (jpc_putms(enc->tmpstream, enc->cstate, enc->mrk)) {
                    return -1;
                }
                jpc_ms_destroy(enc->mrk);
                enc->mrk = 0;
            }
        }

        for (cmptno = 0, comp = tile->tcmpts; cmptno < JAS_CAST(int, cp->numcmpts); ++cmptno, ++comp) {
            ccps = &cp->ccps[cmptno];
            if (JAS_CAST(int, ccps->numstepsizes) == comp->numstepsizes) {
                samestepsizes = 1;
                for (bandno = 0; bandno < JAS_CAST(int, ccps->numstepsizes); ++bandno) {
                    if (ccps->stepsizes[bandno] != comp->stepsizes[bandno]) {
                        samestepsizes = 0;
                        break;
                    }
                }
            } else {
                samestepsizes = 0;
            }
            if (!samestepsizes) {
                if (!(enc->mrk = jpc_ms_create(JPC_MS_QCC))) {
                    return -1;
                }
                qcc = &enc->mrk->parms.qcc;
                qcc->compno = cmptno;
                qcc->compparms.numguard = cp->tccp.numgbits;
                qcc->compparms.qntsty = (comp->qmfbid == JPC_COX_INS) ?
                  JPC_QCX_SEQNT : JPC_QCX_NOQNT;
                qcc->compparms.numstepsizes = comp->numstepsizes;
                qcc->compparms.stepsizes = comp->stepsizes;
                if (jpc_putms(enc->tmpstream, enc->cstate, enc->mrk)) {
                    return -1;
                }
                qcc->compparms.stepsizes = 0;
                jpc_ms_destroy(enc->mrk);
                enc->mrk = 0;
            }
        }

        /* Write a SOD marker to indicate the end of the tile header. */
        if (!(enc->mrk = jpc_ms_create(JPC_MS_SOD))) {
            return -1;
        }
        if (jpc_putms(enc->tmpstream, enc->cstate, enc->mrk)) {
            jas_eprintf("cannot write SOD marker\n");
            return -1;
        }
        jpc_ms_destroy(enc->mrk);
        enc->mrk = 0;
tilehdrlen = jas_stream_getrwcount(enc->tmpstream);

/************************************************************************/
/************************************************************************/
/************************************************************************/

if (jpc_enc_enccblks(enc)) {
    abort();
    return -1;
}

        cp = enc->cp;
        rho = (double) (tile->brx - tile->tlx) * (tile->bry - tile->tly) /
          ((cp->refgrdwidth - cp->imgareatlx) * (cp->refgrdheight -
          cp->imgareatly));
        tile->rawsize = cp->rawsize * rho;

        for (lyrno = 0; lyrno < tile->numlyrs - 1; ++lyrno) {
            tile->lyrsizes[lyrno] = tile->rawsize * jpc_fixtodbl(
              cp->tcp.ilyrrates[lyrno]);
        }
        tile->lyrsizes[tile->numlyrs - 1] = (cp->totalsize != UINT_FAST32_MAX) ?
          (rho * enc->mainbodysize) : UINT_FAST32_MAX;
        for (lyrno = 0; lyrno < tile->numlyrs; ++lyrno) {
            if (tile->lyrsizes[lyrno] != UINT_FAST32_MAX) {
                if (tilehdrlen <= JAS_CAST(long, tile->lyrsizes[lyrno])) {
                    tile->lyrsizes[lyrno] -= tilehdrlen;
                } else {
                    tile->lyrsizes[lyrno] = 0;
                }
            }
        }

        if (rateallocate(enc, tile->numlyrs, tile->lyrsizes)) {
            return -1;
        }

#if 0
jas_eprintf("ENCODE TILE DATA\n");
#endif
        if (jpc_enc_encodetiledata(enc)) {
            jas_eprintf("dotile failed\n");
            return -1;
        }

/************************************************************************/
/************************************************************************/
/************************************************************************/

/************************************************************************/
/************************************************************************/
/************************************************************************/

        tilelen = jas_stream_tell(enc->tmpstream);

        if (jas_stream_seek(enc->tmpstream, 6, SEEK_SET) < 0) {
            return -1;
        }
        jpc_putuint32(enc->tmpstream, tilelen);

        if (jas_stream_seek(enc->tmpstream, 0, SEEK_SET) < 0) {
            return -1;
        }
        if (jpc_putdata(enc->out, enc->tmpstream, -1)) {
            return -1;
        }
        enc->len += tilelen;

        jas_stream_close(enc->tmpstream);
        enc->tmpstream = 0;

        jpc_enc_tile_destroy(enc->curtile);
        enc->curtile = 0;

    }

    return 0;
}

int jpc_enc_encodetiledata(jpc_enc_t *enc)
{
assert(enc->tmpstream);
    if (jpc_enc_encpkts(enc, enc->tmpstream)) {
        return -1;
    }
    return 0;
}

int dump_passes(jpc_enc_pass_t *passes, int numpasses, jpc_enc_cblk_t *cblk)
{
    jpc_enc_pass_t *pass;
    int i;
    jas_stream_memobj_t *smo;

    smo = cblk->stream->obj_;

    pass = passes;
    for (i = 0; i < numpasses; ++i) {
        jas_eprintf("start=%d end=%d type=%d term=%d lyrno=%d firstchar=%02x size=%ld pos=%ld\n",
          (int)pass->start, (int)pass->end, (int)pass->type, (int)pass->term, (int)pass->lyrno,
          smo->buf_[pass->start], (long)smo->len_, (long)smo->pos_);
#if 0
        jas_memdump(stderr, &smo->buf_[pass->start], pass->end - pass->start);
#endif
        ++pass;
    }
    return 0;
}

void jpc_quantize(jas_matrix_t *data, jpc_fix_t stepsize)
{
    int i;
    int j;
    jpc_fix_t t;

    if (stepsize == jpc_inttofix(1)) {
        return;
    }

    for (i = 0; i < jas_matrix_numrows(data); ++i) {
        for (j = 0; j < jas_matrix_numcols(data); ++j) {
            t = jas_matrix_get(data, i, j);

{
    if (t < 0) {
        t = jpc_fix_neg(jpc_fix_div(jpc_fix_neg(t), stepsize));
    } else {
        t = jpc_fix_div(t, stepsize);
    }
}

            jas_matrix_set(data, i, j, t);
        }
    }
}

void calcrdslopes(jpc_enc_cblk_t *cblk)
{
    jpc_enc_pass_t *endpasses;
    jpc_enc_pass_t *pass0;
    jpc_enc_pass_t *pass1;
    jpc_enc_pass_t *pass2;
    jpc_flt_t slope0;
    jpc_flt_t slope;
    jpc_flt_t dd;
    long dr;

    endpasses = &cblk->passes[cblk->numpasses];
    pass2 = cblk->passes;
    slope0 = 0;
    while (pass2 != endpasses) {
        pass0 = 0;
        for (pass1 = cblk->passes; pass1 != endpasses; ++pass1) {
            dd = pass1->cumwmsedec;
            dr = pass1->end;
            if (pass0) {
                dd -= pass0->cumwmsedec;
                dr -= pass0->end;
            }
            if (dd <= 0) {
                pass1->rdslope = JPC_BADRDSLOPE;
                if (pass1 >= pass2) {
                    pass2 = &pass1[1];
                }
                continue;
            }
            if (pass1 < pass2 && pass1->rdslope <= 0) {
                continue;
            }
            if (!dr) {
                assert(pass0);
                pass0->rdslope = 0;
                break;
            }
            slope = dd / dr;
            if (pass0 && slope >= slope0) {
                pass0->rdslope = 0;
                break;
            }
            pass1->rdslope = slope;
            if (pass1 >= pass2) {
                pass2 = &pass1[1];
            }
            pass0 = pass1;
            slope0 = slope;
        }
    }

#if 0
    for (pass0 = cblk->passes; pass0 != endpasses; ++pass0) {
if (pass0->rdslope > 0.0) {
        jas_eprintf("pass %02d nmsedec=%lf dec=%lf end=%d %lf\n", pass0 - cblk->passes,
          fixtodbl(pass0->nmsedec), pass0->wmsedec, pass0->end, pass0->rdslope);
}
    }
#endif
}

void dump_layeringinfo(jpc_enc_t *enc)
{

    jpc_enc_tcmpt_t *tcmpt;
    int tcmptno;
    jpc_enc_rlvl_t *rlvl;
    int rlvlno;
    jpc_enc_band_t *band;
    int bandno;
    jpc_enc_prc_t *prc;
    int prcno;
    jpc_enc_cblk_t *cblk;
    int cblkno;
    jpc_enc_pass_t *pass;
    int passno;
    int lyrno;
    jpc_enc_tile_t *tile;

    tile = enc->curtile;

    for (lyrno = 0; lyrno < tile->numlyrs; ++lyrno) {
        jas_eprintf("lyrno = %02d\n", lyrno);
        for (tcmptno = 0, tcmpt = tile->tcmpts; tcmptno < tile->numtcmpts;
          ++tcmptno, ++tcmpt) {
            for (rlvlno = 0, rlvl = tcmpt->rlvls; rlvlno < tcmpt->numrlvls;
              ++rlvlno, ++rlvl) {
                if (!rlvl->bands) {
                    continue;
                }
                for (bandno = 0, band = rlvl->bands; bandno < rlvl->numbands;
                  ++bandno, ++band) {
                    if (!band->data) {
                        continue;
                    }
                    for (prcno = 0, prc = band->prcs; prcno < rlvl->numprcs;
                      ++prcno, ++prc) {
                        if (!prc->cblks) {
                            continue;
                        }
                        for (cblkno = 0, cblk = prc->cblks; cblkno <
                          prc->numcblks; ++cblkno, ++cblk) {
                            for (passno = 0, pass = cblk->passes; passno <
                              cblk->numpasses && pass->lyrno == lyrno;
                              ++passno, ++pass) {
                                jas_eprintf("lyrno=%02d cmptno=%02d rlvlno=%02d bandno=%02d prcno=%02d cblkno=%03d passno=%03d\n", lyrno, tcmptno, rlvlno, bandno, prcno, cblkno, passno);
                            }
                        }
                    }
                }
            }
        }
    }
}

int rateallocate(jpc_enc_t *enc, int numlyrs, uint_fast32_t *cumlens)
{
    jpc_flt_t lo;
    jpc_flt_t hi;
    jas_stream_t *out;
    long cumlen;
    int lyrno;
    jpc_flt_t thresh;
    jpc_flt_t goodthresh;
    int success;
    long pos;
    long oldpos;
    int numiters;

    jpc_enc_tcmpt_t *comp;
    jpc_enc_tcmpt_t *endcomps;
    jpc_enc_rlvl_t *lvl;
    jpc_enc_rlvl_t *endlvls;
    jpc_enc_band_t *band;
    jpc_enc_band_t *endbands;
    jpc_enc_cblk_t *cblk;
    jpc_enc_cblk_t *endcblks;
    jpc_enc_pass_t *pass;
    jpc_enc_pass_t *endpasses;
    jpc_enc_pass_t *pass1;
    jpc_flt_t mxrdslope;
    jpc_flt_t mnrdslope;
    jpc_enc_tile_t *tile;
    jpc_enc_prc_t *prc;
    int prcno;

    tile = enc->curtile;

    for (lyrno = 1; lyrno < numlyrs - 1; ++lyrno) {
        if (cumlens[lyrno - 1] > cumlens[lyrno]) {
            abort();
        }
    }

    if (!(out = jas_stream_memopen(0, 0))) {
        return -1;
    }


    /* Find minimum and maximum R-D slope values. */
    mnrdslope = DBL_MAX;
    mxrdslope = 0;
    endcomps = &tile->tcmpts[tile->numtcmpts];
    for (comp = tile->tcmpts; comp != endcomps; ++comp) {
        endlvls = &comp->rlvls[comp->numrlvls];
        for (lvl = comp->rlvls; lvl != endlvls; ++lvl) {
            if (!lvl->bands) {
                continue;
            }
            endbands = &lvl->bands[lvl->numbands];
            for (band = lvl->bands; band != endbands; ++band) {
                if (!band->data) {
                    continue;
                }
                for (prcno = 0, prc = band->prcs; prcno < lvl->numprcs; ++prcno, ++prc) {
                    if (!prc->cblks) {
                        continue;
                    }
                    endcblks = &prc->cblks[prc->numcblks];
                    for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                        calcrdslopes(cblk);
                        endpasses = &cblk->passes[cblk->numpasses];
                        for (pass = cblk->passes; pass != endpasses; ++pass) {
                            if (pass->rdslope > 0) {
                                if (pass->rdslope < mnrdslope) {
                                    mnrdslope = pass->rdslope;
                                }
                                if (pass->rdslope > mxrdslope) {
                                    mxrdslope = pass->rdslope;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
if (jas_getdbglevel()) {
    jas_eprintf("min rdslope = %f max rdslope = %f\n", mnrdslope, mxrdslope);
}

    jpc_init_t2state(enc, 1);

    for (lyrno = 0; lyrno < numlyrs; ++lyrno) {

        lo = mnrdslope;
        hi = mxrdslope;

        success = 0;
        goodthresh = 0;
        numiters = 0;

        do {

            cumlen = cumlens[lyrno];
            if (cumlen == UINT_FAST32_MAX) {
                /* Only the last layer can be free of a rate
                  constraint (e.g., for lossless coding). */
                assert(lyrno == numlyrs - 1);
                goodthresh = -1;
                success = 1;
                break;
            }

            thresh = (lo + hi) / 2;

            /* Save the tier 2 coding state. */
            jpc_save_t2state(enc);
            oldpos = jas_stream_tell(out);
            assert(oldpos >= 0);

            /* Assign all passes with R-D slopes greater than or
              equal to the current threshold to this layer. */
            endcomps = &tile->tcmpts[tile->numtcmpts];
            for (comp = tile->tcmpts; comp != endcomps; ++comp) {
                endlvls = &comp->rlvls[comp->numrlvls];
                for (lvl = comp->rlvls; lvl != endlvls; ++lvl) {
                    if (!lvl->bands) {
                        continue;
                    }
                    endbands = &lvl->bands[lvl->numbands];
                    for (band = lvl->bands; band != endbands; ++band) {
                        if (!band->data) {
                            continue;
                        }
                        for (prcno = 0, prc = band->prcs; prcno < lvl->numprcs; ++prcno, ++prc) {
                            if (!prc->cblks) {
                                continue;
                            }
                            endcblks = &prc->cblks[prc->numcblks];
                            for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                                if (cblk->curpass) {
                                    endpasses = &cblk->passes[cblk->numpasses];
                                    pass1 = cblk->curpass;
                                    for (pass = cblk->curpass; pass != endpasses; ++pass) {
                                        if (pass->rdslope >= thresh) {
                                            pass1 = &pass[1];
                                        }
                                    }
                                    for (pass = cblk->curpass; pass != pass1; ++pass) {
                                        pass->lyrno = lyrno;
                                    }
                                    for (; pass != endpasses; ++pass) {
                                        pass->lyrno = -1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Perform tier 2 coding. */
            endcomps = &tile->tcmpts[tile->numtcmpts];
            for (comp = tile->tcmpts; comp != endcomps; ++comp) {
                endlvls = &comp->rlvls[comp->numrlvls];
                for (lvl = comp->rlvls; lvl != endlvls; ++lvl) {
                    if (!lvl->bands) {
                        continue;
                    }
                    for (prcno = 0; prcno < lvl->numprcs; ++prcno) {
                        if (jpc_enc_encpkt(enc, out, comp - tile->tcmpts, lvl - comp->rlvls, prcno, lyrno)) {
                            return -1;
                        }
                    }
                }
            }

            pos = jas_stream_tell(out);

            /* Check the rate constraint. */
            assert(pos >= 0);
            if (pos > cumlen) {
                /* The rate is too high. */
                lo = thresh;
            } else if (pos <= cumlen) {
                /* The rate is low enough, so try higher. */
                hi = thresh;
                if (!success || thresh < goodthresh) {
                    goodthresh = thresh;
                    success = 1;
                }
            }

            /* Save the tier 2 coding state. */
            jpc_restore_t2state(enc);
            if (jas_stream_seek(out, oldpos, SEEK_SET) < 0) {
                abort();
            }

if (jas_getdbglevel()) {
jas_eprintf("maxlen=%08ld actuallen=%08ld thresh=%f\n", cumlen, pos, thresh);
}

            ++numiters;
        } while (lo < hi - 1e-3 && numiters < 32);

        if (!success) {
            jas_eprintf("warning: empty layer generated\n");
        }

if (jas_getdbglevel()) {
jas_eprintf("success %d goodthresh %f\n", success, goodthresh);
}

        /* Assign all passes with R-D slopes greater than or
          equal to the selected threshold to this layer. */
        endcomps = &tile->tcmpts[tile->numtcmpts];
        for (comp = tile->tcmpts; comp != endcomps; ++comp) {
            endlvls = &comp->rlvls[comp->numrlvls];
            for (lvl = comp->rlvls; lvl != endlvls; ++lvl) {
if (!lvl->bands) {
    continue;
}
                endbands = &lvl->bands[lvl->numbands];
                for (band = lvl->bands; band != endbands; ++band) {
                    if (!band->data) {
                        continue;
                    }
                    for (prcno = 0, prc = band->prcs; prcno < lvl->numprcs; ++prcno, ++prc) {
                        if (!prc->cblks) {
                            continue;
                        }
                        endcblks = &prc->cblks[prc->numcblks];
                        for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                            if (cblk->curpass) {
                                endpasses = &cblk->passes[cblk->numpasses];
                                pass1 = cblk->curpass;
                                if (success) {
                                    for (pass = cblk->curpass; pass != endpasses; ++pass) {
                                        if (pass->rdslope >= goodthresh) {
                                            pass1 = &pass[1];
                                        }
                                    }
                                }
                                for (pass = cblk->curpass; pass != pass1; ++pass) {
                                    pass->lyrno = lyrno;
                                }
                                for (; pass != endpasses; ++pass) {
                                    pass->lyrno = -1;
                                }
                            }
                        }
                    }
                }
            }
        }

        /* Perform tier 2 coding. */
        endcomps = &tile->tcmpts[tile->numtcmpts];
        for (comp = tile->tcmpts; comp != endcomps; ++comp) {
            endlvls = &comp->rlvls[comp->numrlvls];
            for (lvl = comp->rlvls; lvl != endlvls; ++lvl) {
                if (!lvl->bands) {
                    continue;
                }
                for (prcno = 0; prcno < lvl->numprcs; ++prcno) {
                    if (jpc_enc_encpkt(enc, out, comp - tile->tcmpts, lvl - comp->rlvls, prcno, lyrno)) {
                        return -1;
                    }
                }
            }
        }
    }

    if (jas_getdbglevel() >= 5) {
        dump_layeringinfo(enc);
    }

    jas_stream_close(out);

    JAS_DBGLOG(10, ("done doing rateallocation\n"));
#if 0
jas_eprintf("DONE RATE ALLOCATE\n");
#endif

    return 0;
}

/******************************************************************************\
* Tile constructors and destructors.
\******************************************************************************/

jpc_enc_tile_t *jpc_enc_tile_create(jpc_enc_cp_t *cp, jas_image_t *image, int tileno)
{
    jpc_enc_tile_t *tile;
    uint_fast32_t htileno;
    uint_fast32_t vtileno;
    uint_fast16_t lyrno;
    uint_fast16_t cmptno;
    jpc_enc_tcmpt_t *tcmpt;

    if (!(tile = jas_malloc(sizeof(jpc_enc_tile_t)))) {
        goto error;
    }

    /* Initialize a few members used in error recovery. */
    tile->tcmpts = 0;
    tile->lyrsizes = 0;
    tile->numtcmpts = cp->numcmpts;
    tile->pi = 0;

    tile->tileno = tileno;
    htileno = tileno % cp->numhtiles;
    vtileno = tileno / cp->numhtiles;

    /* Calculate the coordinates of the top-left and bottom-right
      corners of the tile. */
    tile->tlx = JAS_MAX(cp->tilegrdoffx + htileno * cp->tilewidth,
      cp->imgareatlx);
    tile->tly = JAS_MAX(cp->tilegrdoffy + vtileno * cp->tileheight,
      cp->imgareatly);
    tile->brx = JAS_MIN(cp->tilegrdoffx + (htileno + 1) * cp->tilewidth,
      cp->refgrdwidth);
    tile->bry = JAS_MIN(cp->tilegrdoffy + (vtileno + 1) * cp->tileheight,
      cp->refgrdheight);

    /* Initialize some tile coding parameters. */
    tile->intmode = cp->tcp.intmode;
    tile->csty = cp->tcp.csty;
    tile->prg = cp->tcp.prg;
    tile->mctid = cp->tcp.mctid;

    tile->numlyrs = cp->tcp.numlyrs;
    if (!(tile->lyrsizes = jas_alloc2(tile->numlyrs,
      sizeof(uint_fast32_t)))) {
        goto error;
    }
    for (lyrno = 0; lyrno < tile->numlyrs; ++lyrno) {
        tile->lyrsizes[lyrno] = 0;
    }

    /* Allocate an array for the per-tile-component information. */
    if (!(tile->tcmpts = jas_alloc2(cp->numcmpts, sizeof(jpc_enc_tcmpt_t)))) {
        goto error;
    }
    /* Initialize a few members critical for error recovery. */
    for (cmptno = 0, tcmpt = tile->tcmpts; cmptno < cp->numcmpts;
      ++cmptno, ++tcmpt) {
        tcmpt->rlvls = 0;
        tcmpt->tsfb = 0;
        tcmpt->data = 0;
    }
    /* Initialize the per-tile-component information. */
    for (cmptno = 0, tcmpt = tile->tcmpts; cmptno < cp->numcmpts;
      ++cmptno, ++tcmpt) {
        if (!tcmpt_create(tcmpt, cp, image, tile)) {
            goto error;
        }
    }

    /* Initialize the synthesis weights for the MCT. */
    switch (tile->mctid) {
    case JPC_MCT_RCT:
        tile->tcmpts[0].synweight = jpc_dbltofix(sqrt(3.0));
        tile->tcmpts[1].synweight = jpc_dbltofix(sqrt(0.6875));
        tile->tcmpts[2].synweight = jpc_dbltofix(sqrt(0.6875));
        break;
    case JPC_MCT_ICT:
        tile->tcmpts[0].synweight = jpc_dbltofix(sqrt(3.0000));
        tile->tcmpts[1].synweight = jpc_dbltofix(sqrt(3.2584));
        tile->tcmpts[2].synweight = jpc_dbltofix(sqrt(2.4755));
        break;
    default:
    case JPC_MCT_NONE:
        for (cmptno = 0, tcmpt = tile->tcmpts; cmptno < cp->numcmpts;
          ++cmptno, ++tcmpt) {
            tcmpt->synweight = JPC_FIX_ONE;
        }
        break;
    }

    if (!(tile->pi = jpc_enc_pi_create(cp, tile))) {
        goto error;
    }

    return tile;

error:

    if (tile) {
        jpc_enc_tile_destroy(tile);
    }
    return 0;
}

void jpc_enc_tile_destroy(jpc_enc_tile_t *tile)
{
    jpc_enc_tcmpt_t *tcmpt;
    uint_fast16_t cmptno;

    if (tile->tcmpts) {
        for (cmptno = 0, tcmpt = tile->tcmpts; cmptno <
          tile->numtcmpts; ++cmptno, ++tcmpt) {
            tcmpt_destroy(tcmpt);
        }
        jas_free(tile->tcmpts);
    }
    if (tile->lyrsizes) {
        jas_free(tile->lyrsizes);
    }
    if (tile->pi) {
        jpc_pi_destroy(tile->pi);
    }
    jas_free(tile);
}

static jpc_enc_tcmpt_t *tcmpt_create(jpc_enc_tcmpt_t *tcmpt, jpc_enc_cp_t *cp,
  jas_image_t *image, jpc_enc_tile_t *tile)
{
    uint_fast16_t cmptno;
    uint_fast16_t rlvlno;
    jpc_enc_rlvl_t *rlvl;
    uint_fast32_t tlx;
    uint_fast32_t tly;
    uint_fast32_t brx;
    uint_fast32_t bry;
    uint_fast32_t cmpttlx;
    uint_fast32_t cmpttly;
    jpc_enc_ccp_t *ccp;
    jpc_tsfb_band_t bandinfos[JPC_MAXBANDS];

    tcmpt->tile = tile;
    tcmpt->tsfb = 0;
    tcmpt->data = 0;
    tcmpt->rlvls = 0;

    /* Deduce the component number. */
    cmptno = tcmpt - tile->tcmpts;

    ccp = &cp->ccps[cmptno];

    /* Compute the coordinates of the top-left and bottom-right
      corners of this tile-component. */
    tlx = JPC_CEILDIV(tile->tlx, ccp->sampgrdstepx);
    tly = JPC_CEILDIV(tile->tly, ccp->sampgrdstepy);
    brx = JPC_CEILDIV(tile->brx, ccp->sampgrdstepx);
    bry = JPC_CEILDIV(tile->bry, ccp->sampgrdstepy);

    /* Create a sequence to hold the tile-component sample data. */
    if (!(tcmpt->data = jas_seq2d_create(tlx, tly, brx, bry))) {
        goto error;
    }

    /* Get the image data associated with this tile-component. */
    cmpttlx = JPC_CEILDIV(cp->imgareatlx, ccp->sampgrdstepx);
    cmpttly = JPC_CEILDIV(cp->imgareatly, ccp->sampgrdstepy);
    if (jas_image_readcmpt(image, cmptno, tlx - cmpttlx, tly - cmpttly,
      brx - tlx, bry - tly, tcmpt->data)) {
        goto error;
    }

    tcmpt->synweight = 0;
    tcmpt->qmfbid = cp->tccp.qmfbid;
    tcmpt->numrlvls = cp->tccp.maxrlvls;
    tcmpt->numbands = 3 * tcmpt->numrlvls - 2;
    if (!(tcmpt->tsfb = jpc_cod_gettsfb(tcmpt->qmfbid, tcmpt->numrlvls - 1))) {
        goto error;
    }

    for (rlvlno = 0; rlvlno < tcmpt->numrlvls; ++rlvlno) {
        tcmpt->prcwidthexpns[rlvlno] = cp->tccp.prcwidthexpns[rlvlno];
        tcmpt->prcheightexpns[rlvlno] = cp->tccp.prcheightexpns[rlvlno];
    }
    tcmpt->cblkwidthexpn = cp->tccp.cblkwidthexpn;
    tcmpt->cblkheightexpn = cp->tccp.cblkheightexpn;
    tcmpt->cblksty = cp->tccp.cblksty;
    tcmpt->csty = cp->tccp.csty;

    tcmpt->numstepsizes = tcmpt->numbands;
    assert(tcmpt->numstepsizes <= JPC_MAXBANDS);
    memset(tcmpt->stepsizes, 0, tcmpt->numstepsizes *
      sizeof(uint_fast16_t));

    /* Retrieve information about the various bands. */
    jpc_tsfb_getbands(tcmpt->tsfb, jas_seq2d_xstart(tcmpt->data),
      jas_seq2d_ystart(tcmpt->data), jas_seq2d_xend(tcmpt->data),
      jas_seq2d_yend(tcmpt->data), bandinfos);

    if (!(tcmpt->rlvls = jas_alloc2(tcmpt->numrlvls, sizeof(jpc_enc_rlvl_t)))) {
        goto error;
    }
    for (rlvlno = 0, rlvl = tcmpt->rlvls; rlvlno < tcmpt->numrlvls;
      ++rlvlno, ++rlvl) {
        rlvl->bands = 0;
        rlvl->tcmpt = tcmpt;
    }
    for (rlvlno = 0, rlvl = tcmpt->rlvls; rlvlno < tcmpt->numrlvls;
      ++rlvlno, ++rlvl) {
        if (!rlvl_create(rlvl, cp, tcmpt, bandinfos)) {
            goto error;
        }
    }

    return tcmpt;

error:

    tcmpt_destroy(tcmpt);
    return 0;

}

static void tcmpt_destroy(jpc_enc_tcmpt_t *tcmpt)
{
    jpc_enc_rlvl_t *rlvl;
    uint_fast16_t rlvlno;

    if (tcmpt->rlvls) {
        for (rlvlno = 0, rlvl = tcmpt->rlvls; rlvlno < tcmpt->numrlvls;
          ++rlvlno, ++rlvl) {
            rlvl_destroy(rlvl);
        }
        jas_free(tcmpt->rlvls);
    }

    if (tcmpt->data) {
        jas_seq2d_destroy(tcmpt->data);
    }
    if (tcmpt->tsfb) {
        jpc_tsfb_destroy(tcmpt->tsfb);
    }
}

static jpc_enc_rlvl_t *rlvl_create(jpc_enc_rlvl_t *rlvl, jpc_enc_cp_t *cp,
  jpc_enc_tcmpt_t *tcmpt, jpc_tsfb_band_t *bandinfos)
{
    uint_fast16_t rlvlno;
    uint_fast32_t tlprctlx;
    uint_fast32_t tlprctly;
    uint_fast32_t brprcbrx;
    uint_fast32_t brprcbry;
    uint_fast16_t bandno;
    jpc_enc_band_t *band;

    /* Deduce the resolution level. */
    rlvlno = rlvl - tcmpt->rlvls;

    /* Initialize members required for error recovery. */
    rlvl->bands = 0;
    rlvl->tcmpt = tcmpt;

    /* Compute the coordinates of the top-left and bottom-right
      corners of the tile-component at this resolution. */
    rlvl->tlx = JPC_CEILDIVPOW2(jas_seq2d_xstart(tcmpt->data), tcmpt->numrlvls -
      1 - rlvlno);
    rlvl->tly = JPC_CEILDIVPOW2(jas_seq2d_ystart(tcmpt->data), tcmpt->numrlvls -
      1 - rlvlno);
    rlvl->brx = JPC_CEILDIVPOW2(jas_seq2d_xend(tcmpt->data), tcmpt->numrlvls -
      1 - rlvlno);
    rlvl->bry = JPC_CEILDIVPOW2(jas_seq2d_yend(tcmpt->data), tcmpt->numrlvls -
      1 - rlvlno);

    if (rlvl->tlx >= rlvl->brx || rlvl->tly >= rlvl->bry) {
        rlvl->numhprcs = 0;
        rlvl->numvprcs = 0;
        rlvl->numprcs = 0;
        return rlvl;
    }

    rlvl->numbands = (!rlvlno) ? 1 : 3;
    rlvl->prcwidthexpn = cp->tccp.prcwidthexpns[rlvlno];
    rlvl->prcheightexpn = cp->tccp.prcheightexpns[rlvlno];
    if (!rlvlno) {
        rlvl->cbgwidthexpn = rlvl->prcwidthexpn;
        rlvl->cbgheightexpn = rlvl->prcheightexpn;
    } else {
        rlvl->cbgwidthexpn = rlvl->prcwidthexpn - 1;
        rlvl->cbgheightexpn = rlvl->prcheightexpn - 1;
    }
    rlvl->cblkwidthexpn = JAS_MIN(cp->tccp.cblkwidthexpn, rlvl->cbgwidthexpn);
    rlvl->cblkheightexpn = JAS_MIN(cp->tccp.cblkheightexpn, rlvl->cbgheightexpn);

    /* Compute the number of precincts. */
    tlprctlx = JPC_FLOORTOMULTPOW2(rlvl->tlx, rlvl->prcwidthexpn);
    tlprctly = JPC_FLOORTOMULTPOW2(rlvl->tly, rlvl->prcheightexpn);
    brprcbrx = JPC_CEILTOMULTPOW2(rlvl->brx, rlvl->prcwidthexpn);
    brprcbry = JPC_CEILTOMULTPOW2(rlvl->bry, rlvl->prcheightexpn);
    rlvl->numhprcs = JPC_FLOORDIVPOW2(brprcbrx - tlprctlx, rlvl->prcwidthexpn);
    rlvl->numvprcs = JPC_FLOORDIVPOW2(brprcbry - tlprctly, rlvl->prcheightexpn);
    rlvl->numprcs = rlvl->numhprcs * rlvl->numvprcs;

    if (!(rlvl->bands = jas_alloc2(rlvl->numbands, sizeof(jpc_enc_band_t)))) {
        goto error;
    }
    for (bandno = 0, band = rlvl->bands; bandno < rlvl->numbands;
      ++bandno, ++band) {
        band->prcs = 0;
        band->data = 0;
        band->rlvl = rlvl;
    }
    for (bandno = 0, band = rlvl->bands; bandno < rlvl->numbands;
      ++bandno, ++band) {
        if (!band_create(band, cp, rlvl, bandinfos)) {
            goto error;
        }
    }

    return rlvl;
error:

    rlvl_destroy(rlvl);
    return 0;
}

static void rlvl_destroy(jpc_enc_rlvl_t *rlvl)
{
    jpc_enc_band_t *band;
    uint_fast16_t bandno;

    if (rlvl->bands) {
        for (bandno = 0, band = rlvl->bands; bandno < rlvl->numbands;
          ++bandno, ++band) {
            band_destroy(band);
        }
        jas_free(rlvl->bands);
    }
}

static jpc_enc_band_t *band_create(jpc_enc_band_t *band, jpc_enc_cp_t *cp,
  jpc_enc_rlvl_t *rlvl, jpc_tsfb_band_t *bandinfos)
{
    uint_fast16_t bandno;
    uint_fast16_t gblbandno;
    uint_fast16_t rlvlno;
    jpc_tsfb_band_t *bandinfo;
    jpc_enc_tcmpt_t *tcmpt;
    uint_fast32_t prcno;
    jpc_enc_prc_t *prc;

    tcmpt = rlvl->tcmpt;
    band->data = 0;
    band->prcs = 0;
    band->rlvl = rlvl;

    /* Deduce the resolution level and band number. */
    rlvlno = rlvl - rlvl->tcmpt->rlvls;
    bandno = band - rlvl->bands;
    gblbandno = (!rlvlno) ? 0 : (3 * (rlvlno - 1) + bandno + 1);

    bandinfo = &bandinfos[gblbandno];

if (bandinfo->xstart != bandinfo->xend && bandinfo->ystart != bandinfo->yend) {
    if (!(band->data = jas_seq2d_create(0, 0, 0, 0))) {
        goto error;
    }
    jas_seq2d_bindsub(band->data, tcmpt->data, bandinfo->locxstart,
      bandinfo->locystart, bandinfo->locxend, bandinfo->locyend);
    jas_seq2d_setshift(band->data, bandinfo->xstart, bandinfo->ystart);
}
    band->orient = bandinfo->orient;
    band->analgain = JPC_NOMINALGAIN(cp->tccp.qmfbid, tcmpt->numrlvls, rlvlno,
      band->orient);
    band->numbps = 0;
    band->absstepsize = 0;
    band->stepsize = 0;
    band->synweight = bandinfo->synenergywt;

if (band->data) {
    if (!(band->prcs = jas_alloc2(rlvl->numprcs, sizeof(jpc_enc_prc_t)))) {
        goto error;
    }
    for (prcno = 0, prc = band->prcs; prcno < rlvl->numprcs; ++prcno,
      ++prc) {
        prc->cblks = 0;
        prc->incltree = 0;
        prc->nlibtree = 0;
        prc->savincltree = 0;
        prc->savnlibtree = 0;
        prc->band = band;
    }
    for (prcno = 0, prc = band->prcs; prcno < rlvl->numprcs; ++prcno,
      ++prc) {
        if (!prc_create(prc, cp, band)) {
            goto error;
        }
    }
}

    return band;

error:
    band_destroy(band);
    return 0;
}

static void band_destroy(jpc_enc_band_t *band)
{
    jpc_enc_prc_t *prc;
    jpc_enc_rlvl_t *rlvl;
    uint_fast32_t prcno;

    if (band->prcs) {
        rlvl = band->rlvl;
        for (prcno = 0, prc = band->prcs; prcno < rlvl->numprcs;
          ++prcno, ++prc) {
            prc_destroy(prc);
        }
        jas_free(band->prcs);
    }
    if (band->data) {
        jas_seq2d_destroy(band->data);
    }
}

static jpc_enc_prc_t *prc_create(jpc_enc_prc_t *prc, jpc_enc_cp_t *cp, jpc_enc_band_t *band)
{
    uint_fast32_t prcno;
    uint_fast32_t prcxind;
    uint_fast32_t prcyind;
    uint_fast32_t cbgtlx;
    uint_fast32_t cbgtly;
    uint_fast32_t tlprctlx;
    uint_fast32_t tlprctly;
    uint_fast32_t tlcbgtlx;
    uint_fast32_t tlcbgtly;
    uint_fast16_t rlvlno;
    jpc_enc_rlvl_t *rlvl;
    uint_fast32_t tlcblktlx;
    uint_fast32_t tlcblktly;
    uint_fast32_t brcblkbrx;
    uint_fast32_t brcblkbry;
    uint_fast32_t cblkno;
    jpc_enc_cblk_t *cblk;
    jpc_enc_tcmpt_t *tcmpt;

    prc->cblks = 0;
    prc->incltree = 0;
    prc->savincltree = 0;
    prc->nlibtree = 0;
    prc->savnlibtree = 0;

    rlvl = band->rlvl;
    tcmpt = rlvl->tcmpt;
rlvlno = rlvl - tcmpt->rlvls;
    prcno = prc - band->prcs;
    prcxind = prcno % rlvl->numhprcs;
    prcyind = prcno / rlvl->numhprcs;
    prc->band = band;

tlprctlx = JPC_FLOORTOMULTPOW2(rlvl->tlx, rlvl->prcwidthexpn);
tlprctly = JPC_FLOORTOMULTPOW2(rlvl->tly, rlvl->prcheightexpn);
if (!rlvlno) {
    tlcbgtlx = tlprctlx;
    tlcbgtly = tlprctly;
} else {
    tlcbgtlx = JPC_CEILDIVPOW2(tlprctlx, 1);
    tlcbgtly = JPC_CEILDIVPOW2(tlprctly, 1);
}

    /* Compute the coordinates of the top-left and bottom-right
      corners of the precinct. */
    cbgtlx = tlcbgtlx + (prcxind << rlvl->cbgwidthexpn);
    cbgtly = tlcbgtly + (prcyind << rlvl->cbgheightexpn);
    prc->tlx = JAS_MAX(jas_seq2d_xstart(band->data), cbgtlx);
    prc->tly = JAS_MAX(jas_seq2d_ystart(band->data), cbgtly);
    prc->brx = JAS_MIN(jas_seq2d_xend(band->data), cbgtlx +
      (1 << rlvl->cbgwidthexpn));
    prc->bry = JAS_MIN(jas_seq2d_yend(band->data), cbgtly +
      (1 << rlvl->cbgheightexpn));

    if (prc->tlx < prc->brx && prc->tly < prc->bry) {
        /* The precinct contains at least one code block. */

        tlcblktlx = JPC_FLOORTOMULTPOW2(prc->tlx, rlvl->cblkwidthexpn);
        tlcblktly = JPC_FLOORTOMULTPOW2(prc->tly, rlvl->cblkheightexpn);
        brcblkbrx = JPC_CEILTOMULTPOW2(prc->brx, rlvl->cblkwidthexpn);
        brcblkbry = JPC_CEILTOMULTPOW2(prc->bry, rlvl->cblkheightexpn);
        prc->numhcblks = JPC_FLOORDIVPOW2(brcblkbrx - tlcblktlx,
          rlvl->cblkwidthexpn);
        prc->numvcblks = JPC_FLOORDIVPOW2(brcblkbry - tlcblktly,
          rlvl->cblkheightexpn);
        prc->numcblks = prc->numhcblks * prc->numvcblks;

        if (!(prc->incltree = jpc_tagtree_create(prc->numhcblks,
          prc->numvcblks))) {
            goto error;
        }
        if (!(prc->nlibtree = jpc_tagtree_create(prc->numhcblks,
          prc->numvcblks))) {
            goto error;
        }
        if (!(prc->savincltree = jpc_tagtree_create(prc->numhcblks,
          prc->numvcblks))) {
            goto error;
        }
        if (!(prc->savnlibtree = jpc_tagtree_create(prc->numhcblks,
          prc->numvcblks))) {
            goto error;
        }

        if (!(prc->cblks = jas_alloc2(prc->numcblks, sizeof(jpc_enc_cblk_t)))) {
            goto error;
        }
        for (cblkno = 0, cblk = prc->cblks; cblkno < prc->numcblks;
          ++cblkno, ++cblk) {
            cblk->passes = 0;
            cblk->stream = 0;
            cblk->mqenc = 0;
            cblk->data = 0;
            cblk->flags = 0;
            cblk->prc = prc;
        }
        for (cblkno = 0, cblk = prc->cblks; cblkno < prc->numcblks;
          ++cblkno, ++cblk) {
            if (!cblk_create(cblk, cp, prc)) {
                goto error;
            }
        }
    } else {
        /* The precinct does not contain any code blocks. */
        prc->tlx = prc->brx;
        prc->tly = prc->bry;
        prc->numcblks = 0;
        prc->numhcblks = 0;
        prc->numvcblks = 0;
        prc->cblks = 0;
        prc->incltree = 0;
        prc->nlibtree = 0;
        prc->savincltree = 0;
        prc->savnlibtree = 0;
    }

    return prc;

error:
    prc_destroy(prc);
    return 0;
}

static void prc_destroy(jpc_enc_prc_t *prc)
{
    jpc_enc_cblk_t *cblk;
    uint_fast32_t cblkno;

    if (prc->cblks) {
        for (cblkno = 0, cblk = prc->cblks; cblkno < prc->numcblks;
          ++cblkno, ++cblk) {
            cblk_destroy(cblk);
        }
        jas_free(prc->cblks);
    }
    if (prc->incltree) {
        jpc_tagtree_destroy(prc->incltree);
    }
    if (prc->nlibtree) {
        jpc_tagtree_destroy(prc->nlibtree);
    }
    if (prc->savincltree) {
        jpc_tagtree_destroy(prc->savincltree);
    }
    if (prc->savnlibtree) {
        jpc_tagtree_destroy(prc->savnlibtree);
    }
}

static jpc_enc_cblk_t *cblk_create(jpc_enc_cblk_t *cblk, jpc_enc_cp_t *cp, jpc_enc_prc_t *prc)
{
    jpc_enc_band_t *band;
    uint_fast32_t cblktlx;
    uint_fast32_t cblktly;
    uint_fast32_t cblkbrx;
    uint_fast32_t cblkbry;
    jpc_enc_rlvl_t *rlvl;
    uint_fast32_t cblkxind;
    uint_fast32_t cblkyind;
    uint_fast32_t cblkno;
    uint_fast32_t tlcblktlx;
    uint_fast32_t tlcblktly;

    cblkno = cblk - prc->cblks;
    cblkxind = cblkno % prc->numhcblks;
    cblkyind = cblkno / prc->numhcblks;
    rlvl = prc->band->rlvl;
    cblk->prc = prc;

    cblk->numpasses = 0;
    cblk->passes = 0;
    cblk->numencpasses = 0;
    cblk->numimsbs = 0;
    cblk->numlenbits = 0;
    cblk->stream = 0;
    cblk->mqenc = 0;
    cblk->flags = 0;
    cblk->numbps = 0;
    cblk->curpass = 0;
    cblk->data = 0;
    cblk->savedcurpass = 0;
    cblk->savednumlenbits = 0;
    cblk->savednumencpasses = 0;

    band = prc->band;
    tlcblktlx = JPC_FLOORTOMULTPOW2(prc->tlx, rlvl->cblkwidthexpn);
    tlcblktly = JPC_FLOORTOMULTPOW2(prc->tly, rlvl->cblkheightexpn);
    cblktlx = JAS_MAX(tlcblktlx + (cblkxind << rlvl->cblkwidthexpn), prc->tlx);
    cblktly = JAS_MAX(tlcblktly + (cblkyind << rlvl->cblkheightexpn), prc->tly);
    cblkbrx = JAS_MIN(tlcblktlx + ((cblkxind + 1) << rlvl->cblkwidthexpn),
      prc->brx);
    cblkbry = JAS_MIN(tlcblktly + ((cblkyind + 1) << rlvl->cblkheightexpn),
      prc->bry);

    assert(cblktlx < cblkbrx && cblktly < cblkbry);
    if (!(cblk->data = jas_seq2d_create(0, 0, 0, 0))) {
        goto error;
    }
    jas_seq2d_bindsub(cblk->data, band->data, cblktlx, cblktly, cblkbrx, cblkbry);

    return cblk;

error:
    cblk_destroy(cblk);
    return 0;
}

static void cblk_destroy(jpc_enc_cblk_t *cblk)
{
    uint_fast16_t passno;
    jpc_enc_pass_t *pass;
    if (cblk->passes) {
        for (passno = 0, pass = cblk->passes; passno < cblk->numpasses;
          ++passno, ++pass) {
            pass_destroy(pass);
        }
        jas_free(cblk->passes);
    }
    if (cblk->stream) {
        jas_stream_close(cblk->stream);
    }
    if (cblk->mqenc) {
        jpc_mqenc_destroy(cblk->mqenc);
    }
    if (cblk->data) {
        jas_seq2d_destroy(cblk->data);
    }
    if (cblk->flags) {
        jas_seq2d_destroy(cblk->flags);
    }
}

static void pass_destroy(jpc_enc_pass_t *pass)
{
    /* XXX - need to free resources here */
}

void jpc_enc_dump(jpc_enc_t *enc)
{
    jpc_enc_tile_t *tile;
    jpc_enc_tcmpt_t *tcmpt;
    jpc_enc_rlvl_t *rlvl;
    jpc_enc_band_t *band;
    jpc_enc_prc_t *prc;
    jpc_enc_cblk_t *cblk;
    uint_fast16_t cmptno;
    uint_fast16_t rlvlno;
    uint_fast16_t bandno;
    uint_fast32_t prcno;
    uint_fast32_t cblkno;

    tile = enc->curtile;

    for (cmptno = 0, tcmpt = tile->tcmpts; cmptno < tile->numtcmpts; ++cmptno,
      ++tcmpt) {
        jas_eprintf("  tcmpt %5d %5d %5d %5d\n", jas_seq2d_xstart(tcmpt->data), jas_seq2d_ystart(tcmpt->data), jas_seq2d_xend(tcmpt->data), jas_seq2d_yend(tcmpt->data));
        for (rlvlno = 0, rlvl = tcmpt->rlvls; rlvlno < tcmpt->numrlvls;
          ++rlvlno, ++rlvl) {
            jas_eprintf("    rlvl %5d %5d %5d %5d\n", rlvl->tlx, rlvl->tly, rlvl->brx, rlvl->bry);
            for (bandno = 0, band = rlvl->bands; bandno < rlvl->numbands;
              ++bandno, ++band) {
                if (!band->data) {
                    continue;
                }
                jas_eprintf("      band %5d %5d %5d %5d\n", jas_seq2d_xstart(band->data), jas_seq2d_ystart(band->data), jas_seq2d_xend(band->data), jas_seq2d_yend(band->data));
                for (prcno = 0, prc = band->prcs; prcno < rlvl->numprcs;
                  ++prcno, ++prc) {
                    jas_eprintf("        prc %5d %5d %5d %5d (%5d %5d)\n", prc->tlx, prc->tly, prc->brx, prc->bry, prc->brx - prc->tlx, prc->bry - prc->tly);
                    if (!prc->cblks) {
                        continue;
                    }
                    for (cblkno = 0, cblk = prc->cblks; cblkno < prc->numcblks;
                      ++cblkno, ++cblk) {
                        jas_eprintf("         cblk %5d %5d %5d %5d\n", jas_seq2d_xstart(cblk->data), jas_seq2d_ystart(cblk->data), jas_seq2d_xend(cblk->data), jas_seq2d_yend(cblk->data));
                    }
                }
            }
        }
    }
}
