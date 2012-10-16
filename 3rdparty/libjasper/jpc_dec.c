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
 * $Id: jpc_dec.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "jasper/jas_types.h"
#include "jasper/jas_math.h"
#include "jasper/jas_tvp.h"
#include "jasper/jas_malloc.h"
#include "jasper/jas_debug.h"

#include "jpc_fix.h"
#include "jpc_dec.h"
#include "jpc_cs.h"
#include "jpc_mct.h"
#include "jpc_t2dec.h"
#include "jpc_t1dec.h"
#include "jpc_math.h"

/******************************************************************************\
*
\******************************************************************************/

#define	JPC_MHSOC	0x0001
  /* In the main header, expecting a SOC marker segment. */
#define	JPC_MHSIZ	0x0002
  /* In the main header, expecting a SIZ marker segment. */
#define	JPC_MH		0x0004
  /* In the main header, expecting "other" marker segments. */
#define	JPC_TPHSOT	0x0008
  /* In a tile-part header, expecting a SOT marker segment. */
#define	JPC_TPH		0x0010
  /* In a tile-part header, expecting "other" marker segments. */
#define	JPC_MT		0x0020
  /* In the main trailer. */

typedef struct {

    uint_fast16_t id;
    /* The marker segment type. */

    int validstates;
    /* The states in which this type of marker segment can be
      validly encountered. */

    int (*action)(jpc_dec_t *dec, jpc_ms_t *ms);
    /* The action to take upon encountering this type of marker segment. */

} jpc_dec_mstabent_t;

/******************************************************************************\
*
\******************************************************************************/

/* COD/COC parameters have been specified. */
#define	JPC_CSET	0x0001
/* QCD/QCC parameters have been specified. */
#define	JPC_QSET	0x0002
/* COD/COC parameters set from a COC marker segment. */
#define	JPC_COC	0x0004
/* QCD/QCC parameters set from a QCC marker segment. */
#define	JPC_QCC	0x0008

/******************************************************************************\
* Local function prototypes.
\******************************************************************************/

static int jpc_dec_dump(jpc_dec_t *dec, FILE *out);

jpc_ppxstab_t *jpc_ppxstab_create(void);
void jpc_ppxstab_destroy(jpc_ppxstab_t *tab);
int jpc_ppxstab_grow(jpc_ppxstab_t *tab, int maxents);
int jpc_ppxstab_insert(jpc_ppxstab_t *tab, jpc_ppxstabent_t *ent);
jpc_streamlist_t *jpc_ppmstabtostreams(jpc_ppxstab_t *tab);
int jpc_pptstabwrite(jas_stream_t *out, jpc_ppxstab_t *tab);
jpc_ppxstabent_t *jpc_ppxstabent_create(void);
void jpc_ppxstabent_destroy(jpc_ppxstabent_t *ent);

int jpc_streamlist_numstreams(jpc_streamlist_t *streamlist);
jpc_streamlist_t *jpc_streamlist_create(void);
int jpc_streamlist_insert(jpc_streamlist_t *streamlist, int streamno,
  jas_stream_t *stream);
jas_stream_t *jpc_streamlist_remove(jpc_streamlist_t *streamlist, int streamno);
void jpc_streamlist_destroy(jpc_streamlist_t *streamlist);
jas_stream_t *jpc_streamlist_get(jpc_streamlist_t *streamlist, int streamno);

static void jpc_dec_cp_resetflags(jpc_dec_cp_t *cp);
static jpc_dec_cp_t *jpc_dec_cp_create(uint_fast16_t numcomps);
static int jpc_dec_cp_isvalid(jpc_dec_cp_t *cp);
static jpc_dec_cp_t *jpc_dec_cp_copy(jpc_dec_cp_t *cp);
static int jpc_dec_cp_setfromcod(jpc_dec_cp_t *cp, jpc_cod_t *cod);
static int jpc_dec_cp_setfromcoc(jpc_dec_cp_t *cp, jpc_coc_t *coc);
static int jpc_dec_cp_setfromcox(jpc_dec_cp_t *cp, jpc_dec_ccp_t *ccp,
  jpc_coxcp_t *compparms, int flags);
static int jpc_dec_cp_setfromqcd(jpc_dec_cp_t *cp, jpc_qcd_t *qcd);
static int jpc_dec_cp_setfromqcc(jpc_dec_cp_t *cp, jpc_qcc_t *qcc);
static int jpc_dec_cp_setfromqcx(jpc_dec_cp_t *cp, jpc_dec_ccp_t *ccp,
  jpc_qcxcp_t *compparms, int flags);
static int jpc_dec_cp_setfromrgn(jpc_dec_cp_t *cp, jpc_rgn_t *rgn);
static int jpc_dec_cp_prepare(jpc_dec_cp_t *cp);
static void jpc_dec_cp_destroy(jpc_dec_cp_t *cp);
static int jpc_dec_cp_setfrompoc(jpc_dec_cp_t *cp, jpc_poc_t *poc, int reset);
static int jpc_pi_addpchgfrompoc(jpc_pi_t *pi, jpc_poc_t *poc);

static int jpc_dec_decode(jpc_dec_t *dec);
static jpc_dec_t *jpc_dec_create(jpc_dec_importopts_t *impopts, jas_stream_t *in);
static void jpc_dec_destroy(jpc_dec_t *dec);
static void jpc_dequantize(jas_matrix_t *x, jpc_fix_t absstepsize);
static void jpc_undo_roi(jas_matrix_t *x, int roishift, int bgshift, int numbps);
static jpc_fix_t jpc_calcabsstepsize(int stepsize, int numbits);
static int jpc_dec_tiledecode(jpc_dec_t *dec, jpc_dec_tile_t *tile);
static int jpc_dec_tileinit(jpc_dec_t *dec, jpc_dec_tile_t *tile);
static int jpc_dec_tilefini(jpc_dec_t *dec, jpc_dec_tile_t *tile);
static int jpc_dec_process_soc(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_sot(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_sod(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_eoc(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_siz(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_cod(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_coc(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_rgn(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_qcd(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_qcc(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_poc(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_ppm(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_ppt(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_com(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_unk(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_process_crg(jpc_dec_t *dec, jpc_ms_t *ms);
static int jpc_dec_parseopts(char *optstr, jpc_dec_importopts_t *opts);

static jpc_dec_mstabent_t *jpc_dec_mstab_lookup(uint_fast16_t id);

/******************************************************************************\
* Global data.
\******************************************************************************/

jpc_dec_mstabent_t jpc_dec_mstab[] = {
    {JPC_MS_SOC, JPC_MHSOC, jpc_dec_process_soc},
    {JPC_MS_SOT, JPC_MH | JPC_TPHSOT, jpc_dec_process_sot},
    {JPC_MS_SOD, JPC_TPH, jpc_dec_process_sod},
    {JPC_MS_EOC, JPC_TPHSOT, jpc_dec_process_eoc},
    {JPC_MS_SIZ, JPC_MHSIZ, jpc_dec_process_siz},
    {JPC_MS_COD, JPC_MH | JPC_TPH, jpc_dec_process_cod},
    {JPC_MS_COC, JPC_MH | JPC_TPH, jpc_dec_process_coc},
    {JPC_MS_RGN, JPC_MH | JPC_TPH, jpc_dec_process_rgn},
    {JPC_MS_QCD, JPC_MH | JPC_TPH, jpc_dec_process_qcd},
    {JPC_MS_QCC, JPC_MH | JPC_TPH, jpc_dec_process_qcc},
    {JPC_MS_POC, JPC_MH | JPC_TPH, jpc_dec_process_poc},
    {JPC_MS_TLM, JPC_MH, 0},
    {JPC_MS_PLM, JPC_MH, 0},
    {JPC_MS_PLT, JPC_TPH, 0},
    {JPC_MS_PPM, JPC_MH, jpc_dec_process_ppm},
    {JPC_MS_PPT, JPC_TPH, jpc_dec_process_ppt},
    {JPC_MS_SOP, 0, 0},
    {JPC_MS_CRG, JPC_MH, jpc_dec_process_crg},
    {JPC_MS_COM, JPC_MH | JPC_TPH, jpc_dec_process_com},
    {0, JPC_MH | JPC_TPH, jpc_dec_process_unk}
};

/******************************************************************************\
* The main entry point for the JPEG-2000 decoder.
\******************************************************************************/

jas_image_t *jpc_decode(jas_stream_t *in, char *optstr)
{
    jpc_dec_importopts_t opts;
    jpc_dec_t *dec;
    jas_image_t *image;

    dec = 0;

    if (jpc_dec_parseopts(optstr, &opts)) {
        goto error;
    }

    jpc_initluts();

    if (!(dec = jpc_dec_create(&opts, in))) {
        goto error;
    }

    /* Do most of the work. */
    if (jpc_dec_decode(dec)) {
        goto error;
    }

    if (jas_image_numcmpts(dec->image) >= 3) {
        jas_image_setclrspc(dec->image, JAS_CLRSPC_SRGB);
        jas_image_setcmpttype(dec->image, 0,
          JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_R));
        jas_image_setcmpttype(dec->image, 1,
          JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_G));
        jas_image_setcmpttype(dec->image, 2,
          JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_B));
    } else {
        jas_image_setclrspc(dec->image, JAS_CLRSPC_SGRAY);
        jas_image_setcmpttype(dec->image, 0,
          JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_GRAY_Y));
    }

    /* Save the return value. */
    image = dec->image;

    /* Stop the image from being discarded. */
    dec->image = 0;

    /* Destroy decoder. */
    jpc_dec_destroy(dec);

    return image;

error:
    if (dec) {
        jpc_dec_destroy(dec);
    }
    return 0;
}

typedef enum {
    OPT_MAXLYRS,
    OPT_MAXPKTS,
    OPT_DEBUG
} optid_t;

jas_taginfo_t decopts[] = {
    {OPT_MAXLYRS, "maxlyrs"},
    {OPT_MAXPKTS, "maxpkts"},
    {OPT_DEBUG, "debug"},
    {-1, 0}
};

static int jpc_dec_parseopts(char *optstr, jpc_dec_importopts_t *opts)
{
    jas_tvparser_t *tvp;

    opts->debug = 0;
    opts->maxlyrs = JPC_MAXLYRS;
    opts->maxpkts = -1;

    if (!(tvp = jas_tvparser_create(optstr ? optstr : ""))) {
        return -1;
    }

    while (!jas_tvparser_next(tvp)) {
        switch (jas_taginfo_nonull(jas_taginfos_lookup(decopts,
          jas_tvparser_gettag(tvp)))->id) {
        case OPT_MAXLYRS:
            opts->maxlyrs = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_DEBUG:
            opts->debug = atoi(jas_tvparser_getval(tvp));
            break;
        case OPT_MAXPKTS:
            opts->maxpkts = atoi(jas_tvparser_getval(tvp));
            break;
        default:
            jas_eprintf("warning: ignoring invalid option %s\n",
              jas_tvparser_gettag(tvp));
            break;
        }
    }

    jas_tvparser_destroy(tvp);

    return 0;
}

/******************************************************************************\
* Code for table-driven code stream decoder.
\******************************************************************************/

static jpc_dec_mstabent_t *jpc_dec_mstab_lookup(uint_fast16_t id)
{
    jpc_dec_mstabent_t *mstabent;
    for (mstabent = jpc_dec_mstab; mstabent->id != 0; ++mstabent) {
        if (mstabent->id == id) {
            break;
        }
    }
    return mstabent;
}

static int jpc_dec_decode(jpc_dec_t *dec)
{
    jpc_ms_t *ms;
    jpc_dec_mstabent_t *mstabent;
    int ret;
    jpc_cstate_t *cstate;

    if (!(cstate = jpc_cstate_create())) {
        return -1;
    }
    dec->cstate = cstate;

    /* Initially, we should expect to encounter a SOC marker segment. */
    dec->state = JPC_MHSOC;

    for (;;) {

        /* Get the next marker segment in the code stream. */
        if (!(ms = jpc_getms(dec->in, cstate))) {
            jas_eprintf("cannot get marker segment\n");
            return -1;
        }

        mstabent = jpc_dec_mstab_lookup(ms->id);
        assert(mstabent);

        /* Ensure that this type of marker segment is permitted
          at this point in the code stream. */
        if (!(dec->state & mstabent->validstates)) {
            jas_eprintf("unexpected marker segment type\n");
            jpc_ms_destroy(ms);
            return -1;
        }

        /* Process the marker segment. */
        if (mstabent->action) {
            ret = (*mstabent->action)(dec, ms);
        } else {
            /* No explicit action is required. */
            ret = 0;
        }

        /* Destroy the marker segment. */
        jpc_ms_destroy(ms);

        if (ret < 0) {
            return -1;
        } else if (ret > 0) {
            break;
        }

    }

    return 0;
}

static int jpc_dec_process_crg(jpc_dec_t *dec, jpc_ms_t *ms)
{
    int cmptno;
    jpc_dec_cmpt_t *cmpt;
    jpc_crg_t *crg;

    crg = &ms->parms.crg;
    for (cmptno = 0, cmpt = dec->cmpts; cmptno < dec->numcomps; ++cmptno,
      ++cmpt) {
        /* Ignore the information in the CRG marker segment for now.
          This information serves no useful purpose for decoding anyhow.
          Some other parts of the code need to be changed if these lines
          are uncommented.
        cmpt->hsubstep = crg->comps[cmptno].hoff;
        cmpt->vsubstep = crg->comps[cmptno].voff;
        */
    }
    return 0;
}

static int jpc_dec_process_soc(jpc_dec_t *dec, jpc_ms_t *ms)
{
    /* Eliminate warnings about unused variables. */
    ms = 0;

    /* We should expect to encounter a SIZ marker segment next. */
    dec->state = JPC_MHSIZ;

    return 0;
}

static int jpc_dec_process_sot(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_dec_tile_t *tile;
    jpc_sot_t *sot = &ms->parms.sot;
    jas_image_cmptparm_t *compinfos;
    jas_image_cmptparm_t *compinfo;
    jpc_dec_cmpt_t *cmpt;
    int cmptno;

    if (dec->state == JPC_MH) {

        compinfos = jas_alloc2(dec->numcomps, sizeof(jas_image_cmptparm_t));
        assert(compinfos);
        for (cmptno = 0, cmpt = dec->cmpts, compinfo = compinfos;
          cmptno < dec->numcomps; ++cmptno, ++cmpt, ++compinfo) {
            compinfo->tlx = 0;
            compinfo->tly = 0;
            compinfo->prec = cmpt->prec;
            compinfo->sgnd = cmpt->sgnd;
            compinfo->width = cmpt->width;
            compinfo->height = cmpt->height;
            compinfo->hstep = cmpt->hstep;
            compinfo->vstep = cmpt->vstep;
        }

        if (!(dec->image = jas_image_create(dec->numcomps, compinfos,
          JAS_CLRSPC_UNKNOWN))) {
            return -1;
        }
        jas_free(compinfos);

        /* Is the packet header information stored in PPM marker segments in
          the main header? */
        if (dec->ppmstab) {
            /* Convert the PPM marker segment data into a collection of streams
              (one stream per tile-part). */
            if (!(dec->pkthdrstreams = jpc_ppmstabtostreams(dec->ppmstab))) {
                abort();
            }
            jpc_ppxstab_destroy(dec->ppmstab);
            dec->ppmstab = 0;
        }
    }

    if (sot->len > 0) {
        dec->curtileendoff = jas_stream_getrwcount(dec->in) - ms->len -
          4 + sot->len;
    } else {
        dec->curtileendoff = 0;
    }

    if (JAS_CAST(int, sot->tileno) > dec->numtiles) {
        jas_eprintf("invalid tile number in SOT marker segment\n");
        return -1;
    }
    /* Set the current tile. */
    dec->curtile = &dec->tiles[sot->tileno];
    tile = dec->curtile;
    /* Ensure that this is the expected part number. */
    if (sot->partno != tile->partno) {
        return -1;
    }
    if (tile->numparts > 0 && sot->partno >= tile->numparts) {
        return -1;
    }
    if (!tile->numparts && sot->numparts > 0) {
        tile->numparts = sot->numparts;
    }

    tile->pptstab = 0;

    switch (tile->state) {
    case JPC_TILE_INIT:
        /* This is the first tile-part for this tile. */
        tile->state = JPC_TILE_ACTIVE;
        assert(!tile->cp);
        if (!(tile->cp = jpc_dec_cp_copy(dec->cp))) {
            return -1;
        }
        jpc_dec_cp_resetflags(dec->cp);
        break;
    default:
        if (sot->numparts == sot->partno - 1) {
            tile->state = JPC_TILE_ACTIVELAST;
        }
        break;
    }

    /* Note: We do not increment the expected tile-part number until
      all processing for this tile-part is complete. */

    /* We should expect to encounter other tile-part header marker
      segments next. */
    dec->state = JPC_TPH;

    return 0;
}

static int jpc_dec_process_sod(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_dec_tile_t *tile;
    int pos;

    /* Eliminate compiler warnings about unused variables. */
    ms = 0;

    if (!(tile = dec->curtile)) {
        return -1;
    }

    if (!tile->partno) {
        if (!jpc_dec_cp_isvalid(tile->cp)) {
            return -1;
        }
        jpc_dec_cp_prepare(tile->cp);
        if (jpc_dec_tileinit(dec, tile)) {
            return -1;
        }
    }

    /* Are packet headers stored in the main header or tile-part header? */
    if (dec->pkthdrstreams) {
        /* Get the stream containing the packet header data for this
          tile-part. */
        if (!(tile->pkthdrstream = jpc_streamlist_remove(dec->pkthdrstreams, 0))) {
            return -1;
        }
    }

    if (tile->pptstab) {
        if (!tile->pkthdrstream) {
            if (!(tile->pkthdrstream = jas_stream_memopen(0, 0))) {
                return -1;
            }
        }
        pos = jas_stream_tell(tile->pkthdrstream);
        jas_stream_seek(tile->pkthdrstream, 0, SEEK_END);
        if (jpc_pptstabwrite(tile->pkthdrstream, tile->pptstab)) {
            return -1;
        }
        jas_stream_seek(tile->pkthdrstream, pos, SEEK_SET);
        jpc_ppxstab_destroy(tile->pptstab);
        tile->pptstab = 0;
    }

    if (jas_getdbglevel() >= 10) {
        jpc_dec_dump(dec, stderr);
    }

    if (jpc_dec_decodepkts(dec, (tile->pkthdrstream) ? tile->pkthdrstream :
      dec->in, dec->in)) {
        jas_eprintf("jpc_dec_decodepkts failed\n");
        return -1;
    }

    /* Gobble any unconsumed tile data. */
    if (dec->curtileendoff > 0) {
        long curoff;
        uint_fast32_t n;
        curoff = jas_stream_getrwcount(dec->in);
        if (curoff < dec->curtileendoff) {
            n = dec->curtileendoff - curoff;
            jas_eprintf("warning: ignoring trailing garbage (%lu bytes)\n",
              (unsigned long) n);

            while (n-- > 0) {
                if (jas_stream_getc(dec->in) == EOF) {
                    jas_eprintf("read error\n");
                    return -1;
                }
            }
        } else if (curoff > dec->curtileendoff) {
            jas_eprintf("warning: not enough tile data (%lu bytes)\n",
              (unsigned long) curoff - dec->curtileendoff);
        }

    }

    if (tile->numparts > 0 && tile->partno == tile->numparts - 1) {
        if (jpc_dec_tiledecode(dec, tile)) {
            return -1;
        }
        jpc_dec_tilefini(dec, tile);
    }

    dec->curtile = 0;

    /* Increment the expected tile-part number. */
    ++tile->partno;

    /* We should expect to encounter a SOT marker segment next. */
    dec->state = JPC_TPHSOT;

    return 0;
}

static int jpc_dec_tileinit(jpc_dec_t *dec, jpc_dec_tile_t *tile)
{
    jpc_dec_tcomp_t *tcomp;
    int compno;
    int rlvlno;
    jpc_dec_rlvl_t *rlvl;
    jpc_dec_band_t *band;
    jpc_dec_prc_t *prc;
    int bndno;
    jpc_tsfb_band_t *bnd;
    int bandno;
    jpc_dec_ccp_t *ccp;
    int prccnt;
    jpc_dec_cblk_t *cblk;
    int cblkcnt;
    uint_fast32_t tlprcxstart;
    uint_fast32_t tlprcystart;
    uint_fast32_t brprcxend;
    uint_fast32_t brprcyend;
    uint_fast32_t tlcbgxstart;
    uint_fast32_t tlcbgystart;
    uint_fast32_t brcbgxend;
    uint_fast32_t brcbgyend;
    uint_fast32_t cbgxstart;
    uint_fast32_t cbgystart;
    uint_fast32_t cbgxend;
    uint_fast32_t cbgyend;
    uint_fast32_t tlcblkxstart;
    uint_fast32_t tlcblkystart;
    uint_fast32_t brcblkxend;
    uint_fast32_t brcblkyend;
    uint_fast32_t cblkxstart;
    uint_fast32_t cblkystart;
    uint_fast32_t cblkxend;
    uint_fast32_t cblkyend;
    uint_fast32_t tmpxstart;
    uint_fast32_t tmpystart;
    uint_fast32_t tmpxend;
    uint_fast32_t tmpyend;
    jpc_dec_cp_t *cp;
    jpc_tsfb_band_t bnds[64];
    jpc_pchg_t *pchg;
    int pchgno;
    jpc_dec_cmpt_t *cmpt;

    cp = tile->cp;
    tile->realmode = 0;
    if (cp->mctid == JPC_MCT_ICT) {
        tile->realmode = 1;
    }

    for (compno = 0, tcomp = tile->tcomps, cmpt = dec->cmpts; compno <
      dec->numcomps; ++compno, ++tcomp, ++cmpt) {
        ccp = &tile->cp->ccps[compno];
        if (ccp->qmfbid == JPC_COX_INS) {
            tile->realmode = 1;
        }
        tcomp->numrlvls = ccp->numrlvls;
        if (!(tcomp->rlvls = jas_alloc2(tcomp->numrlvls,
          sizeof(jpc_dec_rlvl_t)))) {
            return -1;
        }
        if (!(tcomp->data = jas_seq2d_create(JPC_CEILDIV(tile->xstart,
          cmpt->hstep), JPC_CEILDIV(tile->ystart, cmpt->vstep),
          JPC_CEILDIV(tile->xend, cmpt->hstep), JPC_CEILDIV(tile->yend,
          cmpt->vstep)))) {
            return -1;
        }
        if (!(tcomp->tsfb = jpc_cod_gettsfb(ccp->qmfbid,
          tcomp->numrlvls - 1))) {
            return -1;
        }
{
    jpc_tsfb_getbands(tcomp->tsfb, jas_seq2d_xstart(tcomp->data), jas_seq2d_ystart(tcomp->data), jas_seq2d_xend(tcomp->data), jas_seq2d_yend(tcomp->data), bnds);
}
        for (rlvlno = 0, rlvl = tcomp->rlvls; rlvlno < tcomp->numrlvls;
          ++rlvlno, ++rlvl) {
rlvl->bands = 0;
            rlvl->xstart = JPC_CEILDIVPOW2(tcomp->xstart,
              tcomp->numrlvls - 1 - rlvlno);
            rlvl->ystart = JPC_CEILDIVPOW2(tcomp->ystart,
              tcomp->numrlvls - 1 - rlvlno);
            rlvl->xend = JPC_CEILDIVPOW2(tcomp->xend,
              tcomp->numrlvls - 1 - rlvlno);
            rlvl->yend = JPC_CEILDIVPOW2(tcomp->yend,
              tcomp->numrlvls - 1 - rlvlno);
            rlvl->prcwidthexpn = ccp->prcwidthexpns[rlvlno];
            rlvl->prcheightexpn = ccp->prcheightexpns[rlvlno];
            tlprcxstart = JPC_FLOORDIVPOW2(rlvl->xstart,
              rlvl->prcwidthexpn) << rlvl->prcwidthexpn;
            tlprcystart = JPC_FLOORDIVPOW2(rlvl->ystart,
              rlvl->prcheightexpn) << rlvl->prcheightexpn;
            brprcxend = JPC_CEILDIVPOW2(rlvl->xend,
              rlvl->prcwidthexpn) << rlvl->prcwidthexpn;
            brprcyend = JPC_CEILDIVPOW2(rlvl->yend,
              rlvl->prcheightexpn) << rlvl->prcheightexpn;
            rlvl->numhprcs = (brprcxend - tlprcxstart) >>
              rlvl->prcwidthexpn;
            rlvl->numvprcs = (brprcyend - tlprcystart) >>
              rlvl->prcheightexpn;
            rlvl->numprcs = rlvl->numhprcs * rlvl->numvprcs;

            if (rlvl->xstart >= rlvl->xend || rlvl->ystart >= rlvl->yend) {
                rlvl->bands = 0;
                rlvl->numprcs = 0;
                rlvl->numhprcs = 0;
                rlvl->numvprcs = 0;
                continue;
            }
            if (!rlvlno) {
                tlcbgxstart = tlprcxstart;
                tlcbgystart = tlprcystart;
                brcbgxend = brprcxend;
                brcbgyend = brprcyend;
                rlvl->cbgwidthexpn = rlvl->prcwidthexpn;
                rlvl->cbgheightexpn = rlvl->prcheightexpn;
            } else {
                tlcbgxstart = JPC_CEILDIVPOW2(tlprcxstart, 1);
                tlcbgystart = JPC_CEILDIVPOW2(tlprcystart, 1);
                brcbgxend = JPC_CEILDIVPOW2(brprcxend, 1);
                brcbgyend = JPC_CEILDIVPOW2(brprcyend, 1);
                rlvl->cbgwidthexpn = rlvl->prcwidthexpn - 1;
                rlvl->cbgheightexpn = rlvl->prcheightexpn - 1;
            }
            rlvl->cblkwidthexpn = JAS_MIN(ccp->cblkwidthexpn,
              rlvl->cbgwidthexpn);
            rlvl->cblkheightexpn = JAS_MIN(ccp->cblkheightexpn,
              rlvl->cbgheightexpn);

            rlvl->numbands = (!rlvlno) ? 1 : 3;
            if (!(rlvl->bands = jas_alloc2(rlvl->numbands,
              sizeof(jpc_dec_band_t)))) {
                return -1;
            }
            for (bandno = 0, band = rlvl->bands;
              bandno < rlvl->numbands; ++bandno, ++band) {
                bndno = (!rlvlno) ? 0 : (3 * (rlvlno - 1) +
                  bandno + 1);
                bnd = &bnds[bndno];

                band->orient = bnd->orient;
                band->stepsize = ccp->stepsizes[bndno];
                band->analgain = JPC_NOMINALGAIN(ccp->qmfbid,
                  tcomp->numrlvls - 1, rlvlno, band->orient);
                band->absstepsize = jpc_calcabsstepsize(band->stepsize,
                  cmpt->prec + band->analgain);
                band->numbps = ccp->numguardbits +
                  JPC_QCX_GETEXPN(band->stepsize) - 1;
                band->roishift = (ccp->roishift + band->numbps >= JPC_PREC) ?
                  (JPC_PREC - 1 - band->numbps) : ccp->roishift;
                band->data = 0;
                band->prcs = 0;
                if (bnd->xstart == bnd->xend || bnd->ystart == bnd->yend) {
                    continue;
                }
                if (!(band->data = jas_seq2d_create(0, 0, 0, 0))) {
                    return -1;
                }
                jas_seq2d_bindsub(band->data, tcomp->data, bnd->locxstart, bnd->locystart, bnd->locxend, bnd->locyend);
                jas_seq2d_setshift(band->data, bnd->xstart, bnd->ystart);

                assert(rlvl->numprcs);

                if (!(band->prcs = jas_alloc2(rlvl->numprcs, sizeof(jpc_dec_prc_t)))) {
                    return -1;
                }

/************************************************/
    cbgxstart = tlcbgxstart;
    cbgystart = tlcbgystart;
    for (prccnt = rlvl->numprcs, prc = band->prcs;
      prccnt > 0; --prccnt, ++prc) {
        cbgxend = cbgxstart + (1 << rlvl->cbgwidthexpn);
        cbgyend = cbgystart + (1 << rlvl->cbgheightexpn);
        prc->xstart = JAS_MAX(cbgxstart, JAS_CAST(uint_fast32_t, jas_seq2d_xstart(band->data)));
        prc->ystart = JAS_MAX(cbgystart, JAS_CAST(uint_fast32_t, jas_seq2d_ystart(band->data)));
        prc->xend = JAS_MIN(cbgxend, JAS_CAST(uint_fast32_t, jas_seq2d_xend(band->data)));
        prc->yend = JAS_MIN(cbgyend, JAS_CAST(uint_fast32_t, jas_seq2d_yend(band->data)));
        if (prc->xend > prc->xstart && prc->yend > prc->ystart) {
            tlcblkxstart = JPC_FLOORDIVPOW2(prc->xstart,
              rlvl->cblkwidthexpn) << rlvl->cblkwidthexpn;
            tlcblkystart = JPC_FLOORDIVPOW2(prc->ystart,
              rlvl->cblkheightexpn) << rlvl->cblkheightexpn;
            brcblkxend = JPC_CEILDIVPOW2(prc->xend,
              rlvl->cblkwidthexpn) << rlvl->cblkwidthexpn;
            brcblkyend = JPC_CEILDIVPOW2(prc->yend,
              rlvl->cblkheightexpn) << rlvl->cblkheightexpn;
            prc->numhcblks = (brcblkxend - tlcblkxstart) >>
              rlvl->cblkwidthexpn;
            prc->numvcblks = (brcblkyend - tlcblkystart) >>
              rlvl->cblkheightexpn;
            prc->numcblks = prc->numhcblks * prc->numvcblks;
            assert(prc->numcblks > 0);

            if (!(prc->incltagtree = jpc_tagtree_create(prc->numhcblks, prc->numvcblks))) {
                return -1;
            }
            if (!(prc->numimsbstagtree = jpc_tagtree_create(prc->numhcblks, prc->numvcblks))) {
                return -1;
            }
            if (!(prc->cblks = jas_alloc2(prc->numcblks, sizeof(jpc_dec_cblk_t)))) {
                return -1;
            }

            cblkxstart = cbgxstart;
            cblkystart = cbgystart;
            for (cblkcnt = prc->numcblks, cblk = prc->cblks; cblkcnt > 0;) {
                cblkxend = cblkxstart + (1 << rlvl->cblkwidthexpn);
                cblkyend = cblkystart + (1 << rlvl->cblkheightexpn);
                tmpxstart = JAS_MAX(cblkxstart, prc->xstart);
                tmpystart = JAS_MAX(cblkystart, prc->ystart);
                tmpxend = JAS_MIN(cblkxend, prc->xend);
                tmpyend = JAS_MIN(cblkyend, prc->yend);
                if (tmpxend > tmpxstart && tmpyend > tmpystart) {
                    cblk->firstpassno = -1;
                    cblk->mqdec = 0;
                    cblk->nulldec = 0;
                    cblk->flags = 0;
                    cblk->numpasses = 0;
                    cblk->segs.head = 0;
                    cblk->segs.tail = 0;
                    cblk->curseg = 0;
                    cblk->numimsbs = 0;
                    cblk->numlenbits = 3;
                    cblk->flags = 0;
                    if (!(cblk->data = jas_seq2d_create(0, 0, 0, 0))) {
                        return -1;
                    }
                    jas_seq2d_bindsub(cblk->data, band->data, tmpxstart, tmpystart, tmpxend, tmpyend);
                    ++cblk;
                    --cblkcnt;
                }
                cblkxstart += 1 << rlvl->cblkwidthexpn;
                if (cblkxstart >= cbgxend) {
                    cblkxstart = cbgxstart;
                    cblkystart += 1 << rlvl->cblkheightexpn;
                }
            }

        } else {
            prc->cblks = 0;
            prc->incltagtree = 0;
            prc->numimsbstagtree = 0;
        }
        cbgxstart += 1 << rlvl->cbgwidthexpn;
        if (cbgxstart >= brcbgxend) {
            cbgxstart = tlcbgxstart;
            cbgystart += 1 << rlvl->cbgheightexpn;
        }

    }
/********************************************/
            }
        }
    }

if (!(tile->pi = jpc_dec_pi_create(dec, tile)))
{
    return -1;
}

    for (pchgno = 0; pchgno < jpc_pchglist_numpchgs(tile->cp->pchglist);
      ++pchgno) {
        pchg = jpc_pchg_copy(jpc_pchglist_get(tile->cp->pchglist, pchgno));
        assert(pchg);
        jpc_pi_addpchg(tile->pi, pchg);
    }
    jpc_pi_init(tile->pi);

    return 0;
}

static int jpc_dec_tilefini(jpc_dec_t *dec, jpc_dec_tile_t *tile)
{
    jpc_dec_tcomp_t *tcomp;
    int compno;
    int bandno;
    int rlvlno;
    jpc_dec_band_t *band;
    jpc_dec_rlvl_t *rlvl;
    int prcno;
    jpc_dec_prc_t *prc;
    jpc_dec_seg_t *seg;
    jpc_dec_cblk_t *cblk;
    int cblkno;

if (tile->tcomps) {

    for (compno = 0, tcomp = tile->tcomps; compno < dec->numcomps;
      ++compno, ++tcomp) {
        for (rlvlno = 0, rlvl = tcomp->rlvls; rlvlno < tcomp->numrlvls;
          ++rlvlno, ++rlvl) {
if (!rlvl->bands) {
    continue;
}
            for (bandno = 0, band = rlvl->bands; bandno < rlvl->numbands; ++bandno, ++band) {
if (band->prcs) {
                for (prcno = 0, prc = band->prcs; prcno <
                  rlvl->numprcs; ++prcno, ++prc) {
if (!prc->cblks) {
    continue;
}
                    for (cblkno = 0, cblk = prc->cblks; cblkno < prc->numcblks; ++cblkno, ++cblk) {

    while (cblk->segs.head) {
        seg = cblk->segs.head;
        jpc_seglist_remove(&cblk->segs, seg);
        jpc_seg_destroy(seg);
    }
    jas_matrix_destroy(cblk->data);
    if (cblk->mqdec) {
        jpc_mqdec_destroy(cblk->mqdec);
    }
    if (cblk->nulldec) {
        jpc_bitstream_close(cblk->nulldec);
    }
    if (cblk->flags) {
        jas_matrix_destroy(cblk->flags);
    }
                    }
                    if (prc->incltagtree) {
                        jpc_tagtree_destroy(prc->incltagtree);
                    }
                    if (prc->numimsbstagtree) {
                        jpc_tagtree_destroy(prc->numimsbstagtree);
                    }
                    if (prc->cblks) {
                        jas_free(prc->cblks);
                    }
                }
}
                if (band->data) {
                    jas_matrix_destroy(band->data);
                }
                if (band->prcs) {
                    jas_free(band->prcs);
                }
            }
            if (rlvl->bands) {
                jas_free(rlvl->bands);
            }
        }
        if (tcomp->rlvls) {
            jas_free(tcomp->rlvls);
        }
        if (tcomp->data) {
            jas_matrix_destroy(tcomp->data);
        }
        if (tcomp->tsfb) {
            jpc_tsfb_destroy(tcomp->tsfb);
        }
    }
}
    if (tile->cp) {
        jpc_dec_cp_destroy(tile->cp);
        tile->cp = 0;
    }
    if (tile->tcomps) {
        jas_free(tile->tcomps);
        tile->tcomps = 0;
    }
    if (tile->pi) {
        jpc_pi_destroy(tile->pi);
        tile->pi = 0;
    }
    if (tile->pkthdrstream) {
        jas_stream_close(tile->pkthdrstream);
        tile->pkthdrstream = 0;
    }
    if (tile->pptstab) {
        jpc_ppxstab_destroy(tile->pptstab);
        tile->pptstab = 0;
    }

    tile->state = JPC_TILE_DONE;

    return 0;
}

static int jpc_dec_tiledecode(jpc_dec_t *dec, jpc_dec_tile_t *tile)
{
    int i;
    int j;
    jpc_dec_tcomp_t *tcomp;
    jpc_dec_rlvl_t *rlvl;
    jpc_dec_band_t *band;
    int compno;
    int rlvlno;
    int bandno;
    int adjust;
    int v;
    jpc_dec_ccp_t *ccp;
    jpc_dec_cmpt_t *cmpt;

    if (jpc_dec_decodecblks(dec, tile)) {
        jas_eprintf("jpc_dec_decodecblks failed\n");
        return -1;
    }

    /* Perform dequantization. */
    for (compno = 0, tcomp = tile->tcomps; compno < dec->numcomps;
      ++compno, ++tcomp) {
        ccp = &tile->cp->ccps[compno];
        for (rlvlno = 0, rlvl = tcomp->rlvls; rlvlno < tcomp->numrlvls;
          ++rlvlno, ++rlvl) {
            if (!rlvl->bands) {
                continue;
            }
            for (bandno = 0, band = rlvl->bands;
              bandno < rlvl->numbands; ++bandno, ++band) {
                if (!band->data) {
                    continue;
                }
                jpc_undo_roi(band->data, band->roishift, ccp->roishift -
                  band->roishift, band->numbps);
                if (tile->realmode) {
                    jas_matrix_asl(band->data, JPC_FIX_FRACBITS);
                    jpc_dequantize(band->data, band->absstepsize);
                }

            }
        }
    }

    /* Apply an inverse wavelet transform if necessary. */
    for (compno = 0, tcomp = tile->tcomps; compno < dec->numcomps;
      ++compno, ++tcomp) {
        ccp = &tile->cp->ccps[compno];
        jpc_tsfb_synthesize(tcomp->tsfb, tcomp->data);
    }


    /* Apply an inverse intercomponent transform if necessary. */
    switch (tile->cp->mctid) {
    case JPC_MCT_RCT:
        assert(dec->numcomps == 3 || dec->numcomps == 4);
        jpc_irct(tile->tcomps[0].data, tile->tcomps[1].data,
          tile->tcomps[2].data);
        break;
    case JPC_MCT_ICT:
        assert(dec->numcomps == 3 || dec->numcomps == 4);
        jpc_iict(tile->tcomps[0].data, tile->tcomps[1].data,
          tile->tcomps[2].data);
        break;
    }

    /* Perform rounding and convert to integer values. */
    if (tile->realmode) {
        for (compno = 0, tcomp = tile->tcomps; compno < dec->numcomps;
          ++compno, ++tcomp) {
            for (i = 0; i < jas_matrix_numrows(tcomp->data); ++i) {
                for (j = 0; j < jas_matrix_numcols(tcomp->data); ++j) {
                    v = jas_matrix_get(tcomp->data, i, j);
                    v = jpc_fix_round(v);
                    jas_matrix_set(tcomp->data, i, j, jpc_fixtoint(v));
                }
            }
        }
    }

    /* Perform level shift. */
    for (compno = 0, tcomp = tile->tcomps, cmpt = dec->cmpts; compno <
      dec->numcomps; ++compno, ++tcomp, ++cmpt) {
        adjust = cmpt->sgnd ? 0 : (1 << (cmpt->prec - 1));
        for (i = 0; i < jas_matrix_numrows(tcomp->data); ++i) {
            for (j = 0; j < jas_matrix_numcols(tcomp->data); ++j) {
                *jas_matrix_getref(tcomp->data, i, j) += adjust;
            }
        }
    }

    /* Perform clipping. */
    for (compno = 0, tcomp = tile->tcomps, cmpt = dec->cmpts; compno <
      dec->numcomps; ++compno, ++tcomp, ++cmpt) {
        jpc_fix_t mn;
        jpc_fix_t mx;
        mn = cmpt->sgnd ? (-(1 << (cmpt->prec - 1))) : (0);
        mx = cmpt->sgnd ? ((1 << (cmpt->prec - 1)) - 1) : ((1 <<
          cmpt->prec) - 1);
        jas_matrix_clip(tcomp->data, mn, mx);
    }

    /* XXX need to free tsfb struct */

    /* Write the data for each component of the image. */
    for (compno = 0, tcomp = tile->tcomps, cmpt = dec->cmpts; compno <
      dec->numcomps; ++compno, ++tcomp, ++cmpt) {
        if (jas_image_writecmpt(dec->image, compno, tcomp->xstart -
          JPC_CEILDIV(dec->xstart, cmpt->hstep), tcomp->ystart -
          JPC_CEILDIV(dec->ystart, cmpt->vstep), jas_matrix_numcols(
          tcomp->data), jas_matrix_numrows(tcomp->data), tcomp->data)) {
            jas_eprintf("write component failed\n");
            return -4;
        }
    }

    return 0;
}

static int jpc_dec_process_eoc(jpc_dec_t *dec, jpc_ms_t *ms)
{
    int tileno;
    jpc_dec_tile_t *tile;

    /* Eliminate compiler warnings about unused variables. */
    ms = 0;

    for (tileno = 0, tile = dec->tiles; tileno < dec->numtiles; ++tileno,
      ++tile) {
        if (tile->state == JPC_TILE_ACTIVE) {
            if (jpc_dec_tiledecode(dec, tile)) {
                return -1;
            }
        }
        jpc_dec_tilefini(dec, tile);
    }

    /* We are done processing the code stream. */
    dec->state = JPC_MT;

    return 1;
}

static int jpc_dec_process_siz(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_siz_t *siz = &ms->parms.siz;
    int compno;
    int tileno;
    jpc_dec_tile_t *tile;
    jpc_dec_tcomp_t *tcomp;
    int htileno;
    int vtileno;
    jpc_dec_cmpt_t *cmpt;

    dec->xstart = siz->xoff;
    dec->ystart = siz->yoff;
    dec->xend = siz->width;
    dec->yend = siz->height;
    dec->tilewidth = siz->tilewidth;
    dec->tileheight = siz->tileheight;
    dec->tilexoff = siz->tilexoff;
    dec->tileyoff = siz->tileyoff;
    dec->numcomps = siz->numcomps;
    if (!(dec->cp = jpc_dec_cp_create(dec->numcomps))) {
        return -1;
    }

    if (!(dec->cmpts = jas_alloc2(dec->numcomps, sizeof(jpc_dec_cmpt_t)))) {
        return -1;
    }

    for (compno = 0, cmpt = dec->cmpts; compno < dec->numcomps; ++compno,
      ++cmpt) {
        cmpt->prec = siz->comps[compno].prec;
        cmpt->sgnd = siz->comps[compno].sgnd;
        cmpt->hstep = siz->comps[compno].hsamp;
        cmpt->vstep = siz->comps[compno].vsamp;
        cmpt->width = JPC_CEILDIV(dec->xend, cmpt->hstep) -
          JPC_CEILDIV(dec->xstart, cmpt->hstep);
        cmpt->height = JPC_CEILDIV(dec->yend, cmpt->vstep) -
          JPC_CEILDIV(dec->ystart, cmpt->vstep);
        cmpt->hsubstep = 0;
        cmpt->vsubstep = 0;
    }

    dec->image = 0;

    dec->numhtiles = JPC_CEILDIV(dec->xend - dec->tilexoff, dec->tilewidth);
    dec->numvtiles = JPC_CEILDIV(dec->yend - dec->tileyoff, dec->tileheight);
    dec->numtiles = dec->numhtiles * dec->numvtiles;
    if (!(dec->tiles = jas_alloc2(dec->numtiles, sizeof(jpc_dec_tile_t)))) {
        return -1;
    }

    for (tileno = 0, tile = dec->tiles; tileno < dec->numtiles; ++tileno,
      ++tile) {
        htileno = tileno % dec->numhtiles;
        vtileno = tileno / dec->numhtiles;
        tile->realmode = 0;
        tile->state = JPC_TILE_INIT;
        tile->xstart = JAS_MAX(dec->tilexoff + htileno * dec->tilewidth,
          dec->xstart);
        tile->ystart = JAS_MAX(dec->tileyoff + vtileno * dec->tileheight,
          dec->ystart);
        tile->xend = JAS_MIN(dec->tilexoff + (htileno + 1) *
          dec->tilewidth, dec->xend);
        tile->yend = JAS_MIN(dec->tileyoff + (vtileno + 1) *
          dec->tileheight, dec->yend);
        tile->numparts = 0;
        tile->partno = 0;
        tile->pkthdrstream = 0;
        tile->pkthdrstreampos = 0;
        tile->pptstab = 0;
        tile->cp = 0;
        if (!(tile->tcomps = jas_alloc2(dec->numcomps,
          sizeof(jpc_dec_tcomp_t)))) {
            return -1;
        }
        for (compno = 0, cmpt = dec->cmpts, tcomp = tile->tcomps;
          compno < dec->numcomps; ++compno, ++cmpt, ++tcomp) {
            tcomp->rlvls = 0;
            tcomp->data = 0;
            tcomp->xstart = JPC_CEILDIV(tile->xstart, cmpt->hstep);
            tcomp->ystart = JPC_CEILDIV(tile->ystart, cmpt->vstep);
            tcomp->xend = JPC_CEILDIV(tile->xend, cmpt->hstep);
            tcomp->yend = JPC_CEILDIV(tile->yend, cmpt->vstep);
            tcomp->tsfb = 0;
        }
    }

    dec->pkthdrstreams = 0;

    /* We should expect to encounter other main header marker segments
      or an SOT marker segment next. */
    dec->state = JPC_MH;

    return 0;
}

static int jpc_dec_process_cod(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_cod_t *cod = &ms->parms.cod;
    jpc_dec_tile_t *tile;

    switch (dec->state) {
    case JPC_MH:
        jpc_dec_cp_setfromcod(dec->cp, cod);
        break;
    case JPC_TPH:
        if (!(tile = dec->curtile)) {
            return -1;
        }
        if (tile->partno != 0) {
            return -1;
        }
        jpc_dec_cp_setfromcod(tile->cp, cod);
        break;
    }
    return 0;
}

static int jpc_dec_process_coc(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_coc_t *coc = &ms->parms.coc;
    jpc_dec_tile_t *tile;

    if (JAS_CAST(int, coc->compno) > dec->numcomps) {
        jas_eprintf("invalid component number in COC marker segment\n");
        return -1;
    }
    switch (dec->state) {
    case JPC_MH:
        jpc_dec_cp_setfromcoc(dec->cp, coc);
        break;
    case JPC_TPH:
        if (!(tile = dec->curtile)) {
            return -1;
        }
        if (tile->partno > 0) {
            return -1;
        }
        jpc_dec_cp_setfromcoc(tile->cp, coc);
        break;
    }
    return 0;
}

static int jpc_dec_process_rgn(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_rgn_t *rgn = &ms->parms.rgn;
    jpc_dec_tile_t *tile;

    if (JAS_CAST(int, rgn->compno) > dec->numcomps) {
        jas_eprintf("invalid component number in RGN marker segment\n");
        return -1;
    }
    switch (dec->state) {
    case JPC_MH:
        jpc_dec_cp_setfromrgn(dec->cp, rgn);
        break;
    case JPC_TPH:
        if (!(tile = dec->curtile)) {
            return -1;
        }
        if (tile->partno > 0) {
            return -1;
        }
        jpc_dec_cp_setfromrgn(tile->cp, rgn);
        break;
    }

    return 0;
}

static int jpc_dec_process_qcd(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_qcd_t *qcd = &ms->parms.qcd;
    jpc_dec_tile_t *tile;

    switch (dec->state) {
    case JPC_MH:
        jpc_dec_cp_setfromqcd(dec->cp, qcd);
        break;
    case JPC_TPH:
        if (!(tile = dec->curtile)) {
            return -1;
        }
        if (tile->partno > 0) {
            return -1;
        }
        jpc_dec_cp_setfromqcd(tile->cp, qcd);
        break;
    }
    return 0;
}

static int jpc_dec_process_qcc(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_qcc_t *qcc = &ms->parms.qcc;
    jpc_dec_tile_t *tile;

    if (JAS_CAST(int, qcc->compno) > dec->numcomps) {
        jas_eprintf("invalid component number in QCC marker segment\n");
        return -1;
    }
    switch (dec->state) {
    case JPC_MH:
        jpc_dec_cp_setfromqcc(dec->cp, qcc);
        break;
    case JPC_TPH:
        if (!(tile = dec->curtile)) {
            return -1;
        }
        if (tile->partno > 0) {
            return -1;
        }
        jpc_dec_cp_setfromqcc(tile->cp, qcc);
        break;
    }
    return 0;
}

static int jpc_dec_process_poc(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_poc_t *poc = &ms->parms.poc;
    jpc_dec_tile_t *tile;
    switch (dec->state) {
    case JPC_MH:
        if (jpc_dec_cp_setfrompoc(dec->cp, poc, 1)) {
            return -1;
        }
        break;
    case JPC_TPH:
        if (!(tile = dec->curtile)) {
            return -1;
        }
        if (!tile->partno) {
            if (jpc_dec_cp_setfrompoc(tile->cp, poc, (!tile->partno))) {
                return -1;
            }
        } else {
            jpc_pi_addpchgfrompoc(tile->pi, poc);
        }
        break;
    }
    return 0;
}

static int jpc_dec_process_ppm(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_ppm_t *ppm = &ms->parms.ppm;
    jpc_ppxstabent_t *ppmstabent;

    if (!dec->ppmstab) {
        if (!(dec->ppmstab = jpc_ppxstab_create())) {
            return -1;
        }
    }

    if (!(ppmstabent = jpc_ppxstabent_create())) {
        return -1;
    }
    ppmstabent->ind = ppm->ind;
    ppmstabent->data = ppm->data;
    ppm->data = 0;
    ppmstabent->len = ppm->len;
    if (jpc_ppxstab_insert(dec->ppmstab, ppmstabent)) {
        return -1;
    }
    return 0;
}

static int jpc_dec_process_ppt(jpc_dec_t *dec, jpc_ms_t *ms)
{
    jpc_ppt_t *ppt = &ms->parms.ppt;
    jpc_dec_tile_t *tile;
    jpc_ppxstabent_t *pptstabent;

    tile = dec->curtile;
    if (!tile->pptstab) {
        if (!(tile->pptstab = jpc_ppxstab_create())) {
            return -1;
        }
    }
    if (!(pptstabent = jpc_ppxstabent_create())) {
        return -1;
    }
    pptstabent->ind = ppt->ind;
    pptstabent->data = ppt->data;
    ppt->data = 0;
    pptstabent->len = ppt->len;
    if (jpc_ppxstab_insert(tile->pptstab, pptstabent)) {
        return -1;
    }
    return 0;
}

static int jpc_dec_process_com(jpc_dec_t *dec, jpc_ms_t *ms)
{
    /* Eliminate compiler warnings about unused variables. */
    dec = 0;
    ms = 0;

    return 0;
}

static int jpc_dec_process_unk(jpc_dec_t *dec, jpc_ms_t *ms)
{
    /* Eliminate compiler warnings about unused variables. */
    dec = 0;

    jas_eprintf("warning: ignoring unknown marker segment\n");
    jpc_ms_dump(ms, stderr);
    return 0;
}

/******************************************************************************\
*
\******************************************************************************/

static jpc_dec_cp_t *jpc_dec_cp_create(uint_fast16_t numcomps)
{
    jpc_dec_cp_t *cp;
    jpc_dec_ccp_t *ccp;
    int compno;

    if (!(cp = jas_malloc(sizeof(jpc_dec_cp_t)))) {
        return 0;
    }
    cp->flags = 0;
    cp->numcomps = numcomps;
    cp->prgord = 0;
    cp->numlyrs = 0;
    cp->mctid = 0;
    cp->csty = 0;
    if (!(cp->ccps = jas_alloc2(cp->numcomps, sizeof(jpc_dec_ccp_t)))) {
        return 0;
    }
    if (!(cp->pchglist = jpc_pchglist_create())) {
        jas_free(cp->ccps);
        return 0;
    }
    for (compno = 0, ccp = cp->ccps; compno < cp->numcomps;
      ++compno, ++ccp) {
        ccp->flags = 0;
        ccp->numrlvls = 0;
        ccp->cblkwidthexpn = 0;
        ccp->cblkheightexpn = 0;
        ccp->qmfbid = 0;
        ccp->numstepsizes = 0;
        ccp->numguardbits = 0;
        ccp->roishift = 0;
        ccp->cblkctx = 0;
    }
    return cp;
}

static jpc_dec_cp_t *jpc_dec_cp_copy(jpc_dec_cp_t *cp)
{
    jpc_dec_cp_t *newcp;
    jpc_dec_ccp_t *newccp;
    jpc_dec_ccp_t *ccp;
    int compno;

    if (!(newcp = jpc_dec_cp_create(cp->numcomps))) {
        return 0;
    }
    newcp->flags = cp->flags;
    newcp->prgord = cp->prgord;
    newcp->numlyrs = cp->numlyrs;
    newcp->mctid = cp->mctid;
    newcp->csty = cp->csty;
    jpc_pchglist_destroy(newcp->pchglist);
    newcp->pchglist = 0;
    if (!(newcp->pchglist = jpc_pchglist_copy(cp->pchglist))) {
        jas_free(newcp);
        return 0;
    }
    for (compno = 0, newccp = newcp->ccps, ccp = cp->ccps;
      compno < cp->numcomps;
      ++compno, ++newccp, ++ccp) {
        *newccp = *ccp;
    }
    return newcp;
}

static void jpc_dec_cp_resetflags(jpc_dec_cp_t *cp)
{
    int compno;
    jpc_dec_ccp_t *ccp;
    cp->flags &= (JPC_CSET | JPC_QSET);
    for (compno = 0, ccp = cp->ccps; compno < cp->numcomps;
      ++compno, ++ccp) {
        ccp->flags = 0;
    }
}

static void jpc_dec_cp_destroy(jpc_dec_cp_t *cp)
{
    if (cp->ccps) {
        jas_free(cp->ccps);
    }
    if (cp->pchglist) {
        jpc_pchglist_destroy(cp->pchglist);
    }
    jas_free(cp);
}

static int jpc_dec_cp_isvalid(jpc_dec_cp_t *cp)
{
    uint_fast16_t compcnt;
    jpc_dec_ccp_t *ccp;

    if (!(cp->flags & JPC_CSET) || !(cp->flags & JPC_QSET)) {
        return 0;
    }
    for (compcnt = cp->numcomps, ccp = cp->ccps; compcnt > 0; --compcnt,
      ++ccp) {
        /* Is there enough step sizes for the number of bands? */
        if ((ccp->qsty != JPC_QCX_SIQNT && JAS_CAST(int, ccp->numstepsizes) < 3 *
          ccp->numrlvls - 2) || (ccp->qsty == JPC_QCX_SIQNT &&
          ccp->numstepsizes != 1)) {
            return 0;
        }
    }
    return 1;
}

static void calcstepsizes(uint_fast16_t refstepsize, int numrlvls,
  uint_fast16_t *stepsizes)
{
    int bandno;
    int numbands;
    uint_fast16_t expn;
    uint_fast16_t mant;
    expn = JPC_QCX_GETEXPN(refstepsize);
    mant = JPC_QCX_GETMANT(refstepsize);
    numbands = 3 * numrlvls - 2;
    for (bandno = 0; bandno < numbands; ++bandno) {
        stepsizes[bandno] = JPC_QCX_MANT(mant) | JPC_QCX_EXPN(expn +
          (numrlvls - 1) - (numrlvls - 1 - ((bandno > 0) ? ((bandno + 2) / 3) : (0))));
    }
}

static int jpc_dec_cp_prepare(jpc_dec_cp_t *cp)
{
    jpc_dec_ccp_t *ccp;
    int compno;
    int i;
    for (compno = 0, ccp = cp->ccps; compno < cp->numcomps;
      ++compno, ++ccp) {
        if (!(ccp->csty & JPC_COX_PRT)) {
            for (i = 0; i < JPC_MAXRLVLS; ++i) {
                ccp->prcwidthexpns[i] = 15;
                ccp->prcheightexpns[i] = 15;
            }
        }
        if (ccp->qsty == JPC_QCX_SIQNT) {
            calcstepsizes(ccp->stepsizes[0], ccp->numrlvls, ccp->stepsizes);
        }
    }
    return 0;
}

static int jpc_dec_cp_setfromcod(jpc_dec_cp_t *cp, jpc_cod_t *cod)
{
    jpc_dec_ccp_t *ccp;
    int compno;
    cp->flags |= JPC_CSET;
    cp->prgord = cod->prg;
    if (cod->mctrans) {
        cp->mctid = (cod->compparms.qmfbid == JPC_COX_INS) ? (JPC_MCT_ICT) : (JPC_MCT_RCT);
    } else {
        cp->mctid = JPC_MCT_NONE;
    }
    cp->numlyrs = cod->numlyrs;
    cp->csty = cod->csty & (JPC_COD_SOP | JPC_COD_EPH);
    for (compno = 0, ccp = cp->ccps; compno < cp->numcomps;
      ++compno, ++ccp) {
        jpc_dec_cp_setfromcox(cp, ccp, &cod->compparms, 0);
    }
    cp->flags |= JPC_CSET;
    return 0;
}

static int jpc_dec_cp_setfromcoc(jpc_dec_cp_t *cp, jpc_coc_t *coc)
{
    jpc_dec_cp_setfromcox(cp, &cp->ccps[coc->compno], &coc->compparms, JPC_COC);
    return 0;
}

static int jpc_dec_cp_setfromcox(jpc_dec_cp_t *cp, jpc_dec_ccp_t *ccp,
  jpc_coxcp_t *compparms, int flags)
{
    int rlvlno;

    /* Eliminate compiler warnings about unused variables. */
    cp = 0;

    if ((flags & JPC_COC) || !(ccp->flags & JPC_COC)) {
        ccp->numrlvls = compparms->numdlvls + 1;
        ccp->cblkwidthexpn = JPC_COX_GETCBLKSIZEEXPN(
          compparms->cblkwidthval);
        ccp->cblkheightexpn = JPC_COX_GETCBLKSIZEEXPN(
          compparms->cblkheightval);
        ccp->qmfbid = compparms->qmfbid;
        ccp->cblkctx = compparms->cblksty;
        ccp->csty = compparms->csty & JPC_COX_PRT;
        for (rlvlno = 0; rlvlno < compparms->numrlvls; ++rlvlno) {
            ccp->prcwidthexpns[rlvlno] =
              compparms->rlvls[rlvlno].parwidthval;
            ccp->prcheightexpns[rlvlno] =
              compparms->rlvls[rlvlno].parheightval;
        }
        ccp->flags |= flags | JPC_CSET;
    }
    return 0;
}

static int jpc_dec_cp_setfromqcd(jpc_dec_cp_t *cp, jpc_qcd_t *qcd)
{
    int compno;
    jpc_dec_ccp_t *ccp;
    for (compno = 0, ccp = cp->ccps; compno < cp->numcomps;
      ++compno, ++ccp) {
        jpc_dec_cp_setfromqcx(cp, ccp, &qcd->compparms, 0);
    }
    cp->flags |= JPC_QSET;
    return 0;
}

static int jpc_dec_cp_setfromqcc(jpc_dec_cp_t *cp, jpc_qcc_t *qcc)
{
    return jpc_dec_cp_setfromqcx(cp, &cp->ccps[qcc->compno], &qcc->compparms, JPC_QCC);
}

static int jpc_dec_cp_setfromqcx(jpc_dec_cp_t *cp, jpc_dec_ccp_t *ccp,
  jpc_qcxcp_t *compparms, int flags)
{
    int bandno;

    /* Eliminate compiler warnings about unused variables. */
    cp = 0;

    if ((flags & JPC_QCC) || !(ccp->flags & JPC_QCC)) {
        ccp->flags |= flags | JPC_QSET;
        for (bandno = 0; bandno < compparms->numstepsizes; ++bandno) {
            ccp->stepsizes[bandno] = compparms->stepsizes[bandno];
        }
        ccp->numstepsizes = compparms->numstepsizes;
        ccp->numguardbits = compparms->numguard;
        ccp->qsty = compparms->qntsty;
    }
    return 0;
}

static int jpc_dec_cp_setfromrgn(jpc_dec_cp_t *cp, jpc_rgn_t *rgn)
{
    jpc_dec_ccp_t *ccp;
    ccp = &cp->ccps[rgn->compno];
    ccp->roishift = rgn->roishift;
    return 0;
}

static int jpc_pi_addpchgfrompoc(jpc_pi_t *pi, jpc_poc_t *poc)
{
    int pchgno;
    jpc_pchg_t *pchg;
    for (pchgno = 0; pchgno < poc->numpchgs; ++pchgno) {
        if (!(pchg = jpc_pchg_copy(&poc->pchgs[pchgno]))) {
            return -1;
        }
        if (jpc_pchglist_insert(pi->pchglist, -1, pchg)) {
            return -1;
        }
    }
    return 0;
}

static int jpc_dec_cp_setfrompoc(jpc_dec_cp_t *cp, jpc_poc_t *poc, int reset)
{
    int pchgno;
    jpc_pchg_t *pchg;
    if (reset) {
        while (jpc_pchglist_numpchgs(cp->pchglist) > 0) {
            pchg = jpc_pchglist_remove(cp->pchglist, 0);
            jpc_pchg_destroy(pchg);
        }
    }
    for (pchgno = 0; pchgno < poc->numpchgs; ++pchgno) {
        if (!(pchg = jpc_pchg_copy(&poc->pchgs[pchgno]))) {
            return -1;
        }
        if (jpc_pchglist_insert(cp->pchglist, -1, pchg)) {
            return -1;
        }
    }
    return 0;
}

static jpc_fix_t jpc_calcabsstepsize(int stepsize, int numbits)
{
    jpc_fix_t absstepsize;
    int n;

    absstepsize = jpc_inttofix(1);
    n = JPC_FIX_FRACBITS - 11;
    absstepsize |= (n >= 0) ? (JPC_QCX_GETMANT(stepsize) << n) :
      (JPC_QCX_GETMANT(stepsize) >> (-n));
    n = numbits - JPC_QCX_GETEXPN(stepsize);
    absstepsize = (n >= 0) ? (absstepsize << n) : (absstepsize >> (-n));
    return absstepsize;
}

static void jpc_dequantize(jas_matrix_t *x, jpc_fix_t absstepsize)
{
    int i;
    int j;
    int t;

    assert(absstepsize >= 0);
    if (absstepsize == jpc_inttofix(1)) {
        return;
    }

    for (i = 0; i < jas_matrix_numrows(x); ++i) {
        for (j = 0; j < jas_matrix_numcols(x); ++j) {
            t = jas_matrix_get(x, i, j);
            if (t) {
                t = jpc_fix_mul(t, absstepsize);
            } else {
                t = 0;
            }
            jas_matrix_set(x, i, j, t);
        }
    }

}

static void jpc_undo_roi(jas_matrix_t *x, int roishift, int bgshift, int numbps)
{
    int i;
    int j;
    int thresh;
    jpc_fix_t val;
    jpc_fix_t mag;
    bool warn;
    uint_fast32_t mask;

    if (roishift == 0 && bgshift == 0) {
        return;
    }
    thresh = 1 << roishift;

    warn = false;
    for (i = 0; i < jas_matrix_numrows(x); ++i) {
        for (j = 0; j < jas_matrix_numcols(x); ++j) {
            val = jas_matrix_get(x, i, j);
            mag = JAS_ABS(val);
            if (mag >= thresh) {
                /* We are dealing with ROI data. */
                mag >>= roishift;
                val = (val < 0) ? (-mag) : mag;
                jas_matrix_set(x, i, j, val);
            } else {
                /* We are dealing with non-ROI (i.e., background) data. */
                mag <<= bgshift;
                mask = (1 << numbps) - 1;
                /* Perform a basic sanity check on the sample value. */
                /* Some implementations write garbage in the unused
                  most-significant bit planes introduced by ROI shifting.
                  Here we ensure that any such bits are masked off. */
                if (mag & (~mask)) {
                    if (!warn) {
                        jas_eprintf("warning: possibly corrupt code stream\n");
                        warn = true;
                    }
                    mag &= mask;
                }
                val = (val < 0) ? (-mag) : mag;
                jas_matrix_set(x, i, j, val);
            }
        }
    }
}

static jpc_dec_t *jpc_dec_create(jpc_dec_importopts_t *impopts, jas_stream_t *in)
{
    jpc_dec_t *dec;

    if (!(dec = jas_malloc(sizeof(jpc_dec_t)))) {
        return 0;
    }

    dec->image = 0;
    dec->xstart = 0;
    dec->ystart = 0;
    dec->xend = 0;
    dec->yend = 0;
    dec->tilewidth = 0;
    dec->tileheight = 0;
    dec->tilexoff = 0;
    dec->tileyoff = 0;
    dec->numhtiles = 0;
    dec->numvtiles = 0;
    dec->numtiles = 0;
    dec->tiles = 0;
    dec->curtile = 0;
    dec->numcomps = 0;
    dec->in = in;
    dec->cp = 0;
    dec->maxlyrs = impopts->maxlyrs;
    dec->maxpkts = impopts->maxpkts;
dec->numpkts = 0;
    dec->ppmseqno = 0;
    dec->state = 0;
    dec->cmpts = 0;
    dec->pkthdrstreams = 0;
    dec->ppmstab = 0;
    dec->curtileendoff = 0;

    return dec;
}

static void jpc_dec_destroy(jpc_dec_t *dec)
{
    if (dec->cstate) {
        jpc_cstate_destroy(dec->cstate);
    }
    if (dec->pkthdrstreams) {
        jpc_streamlist_destroy(dec->pkthdrstreams);
    }
    if (dec->image) {
        jas_image_destroy(dec->image);
    }

    if (dec->cp) {
        jpc_dec_cp_destroy(dec->cp);
    }

    if (dec->cmpts) {
        jas_free(dec->cmpts);
    }

    if (dec->tiles) {
        jas_free(dec->tiles);
    }

    jas_free(dec);
}

/******************************************************************************\
*
\******************************************************************************/

void jpc_seglist_insert(jpc_dec_seglist_t *list, jpc_dec_seg_t *ins, jpc_dec_seg_t *node)
{
    jpc_dec_seg_t *prev;
    jpc_dec_seg_t *next;

    prev = ins;
    node->prev = prev;
    next = prev ? (prev->next) : 0;
    node->prev = prev;
    node->next = next;
    if (prev) {
        prev->next = node;
    } else {
        list->head = node;
    }
    if (next) {
        next->prev = node;
    } else {
        list->tail = node;
    }
}

void jpc_seglist_remove(jpc_dec_seglist_t *list, jpc_dec_seg_t *seg)
{
    jpc_dec_seg_t *prev;
    jpc_dec_seg_t *next;

    prev = seg->prev;
    next = seg->next;
    if (prev) {
        prev->next = next;
    } else {
        list->head = next;
    }
    if (next) {
        next->prev = prev;
    } else {
        list->tail = prev;
    }
    seg->prev = 0;
    seg->next = 0;
}

jpc_dec_seg_t *jpc_seg_alloc()
{
    jpc_dec_seg_t *seg;

    if (!(seg = jas_malloc(sizeof(jpc_dec_seg_t)))) {
        return 0;
    }
    seg->prev = 0;
    seg->next = 0;
    seg->passno = -1;
    seg->numpasses = 0;
    seg->maxpasses = 0;
    seg->type = JPC_SEG_INVALID;
    seg->stream = 0;
    seg->cnt = 0;
    seg->complete = 0;
    seg->lyrno = -1;
    return seg;
}

void jpc_seg_destroy(jpc_dec_seg_t *seg)
{
    if (seg->stream) {
        jas_stream_close(seg->stream);
    }
    jas_free(seg);
}

static int jpc_dec_dump(jpc_dec_t *dec, FILE *out)
{
    jpc_dec_tile_t *tile;
    int tileno;
    jpc_dec_tcomp_t *tcomp;
    int compno;
    jpc_dec_rlvl_t *rlvl;
    int rlvlno;
    jpc_dec_band_t *band;
    int bandno;
    jpc_dec_prc_t *prc;
    int prcno;
    jpc_dec_cblk_t *cblk;
    int cblkno;

    for (tileno = 0, tile = dec->tiles; tileno < dec->numtiles;
      ++tileno, ++tile) {
        for (compno = 0, tcomp = tile->tcomps; compno < dec->numcomps;
          ++compno, ++tcomp) {
            for (rlvlno = 0, rlvl = tcomp->rlvls; rlvlno <
              tcomp->numrlvls; ++rlvlno, ++rlvl) {
fprintf(out, "RESOLUTION LEVEL %d\n", rlvlno);
fprintf(out, "xs =%d, ys = %d, xe = %d, ye = %d, w = %d, h = %d\n",
  (int)rlvl->xstart, (int)rlvl->ystart, (int)rlvl->xend, (int)rlvl->yend, (int)(rlvl->xend -
  rlvl->xstart), (int)(rlvl->yend - rlvl->ystart));
                for (bandno = 0, band = rlvl->bands;
                  bandno < rlvl->numbands; ++bandno, ++band) {
fprintf(out, "BAND %d\n", bandno);
fprintf(out, "xs =%d, ys = %d, xe = %d, ye = %d, w = %d, h = %d\n",
  (int)jas_seq2d_xstart(band->data), (int)jas_seq2d_ystart(band->data), (int)jas_seq2d_xend(band->data),
  (int)jas_seq2d_yend(band->data), (int)(jas_seq2d_xend(band->data) - jas_seq2d_xstart(band->data)),
  (int)(jas_seq2d_yend(band->data) - jas_seq2d_ystart(band->data)));
                    for (prcno = 0, prc = band->prcs;
                      prcno < rlvl->numprcs; ++prcno,
                      ++prc) {
fprintf(out, "CODE BLOCK GROUP %d\n", prcno);
fprintf(out, "xs =%d, ys = %d, xe = %d, ye = %d, w = %d, h = %d\n",
  (int)prc->xstart, (int)prc->ystart, (int)prc->xend, (int)prc->yend, (int)(prc->xend -
  prc->xstart), (int)(prc->yend - prc->ystart));
                        for (cblkno = 0, cblk =
                          prc->cblks; cblkno <
                          prc->numcblks; ++cblkno,
                          ++cblk) {
fprintf(out, "CODE BLOCK %d\n", cblkno);
fprintf(out, "xs =%d, ys = %d, xe = %d, ye = %d, w = %d, h = %d\n",
  (int)jas_seq2d_xstart(cblk->data), (int)jas_seq2d_ystart(cblk->data), (int)jas_seq2d_xend(cblk->data),
  (int)jas_seq2d_yend(cblk->data), (int)(jas_seq2d_xend(cblk->data) - jas_seq2d_xstart(cblk->data)),
  (int)(jas_seq2d_yend(cblk->data) - jas_seq2d_ystart(cblk->data)));
                        }
                    }
                }
            }
        }
    }

    return 0;
}

jpc_streamlist_t *jpc_streamlist_create()
{
    jpc_streamlist_t *streamlist;
    int i;

    if (!(streamlist = jas_malloc(sizeof(jpc_streamlist_t)))) {
        return 0;
    }
    streamlist->numstreams = 0;
    streamlist->maxstreams = 100;
    if (!(streamlist->streams = jas_alloc2(streamlist->maxstreams,
      sizeof(jas_stream_t *)))) {
        jas_free(streamlist);
        return 0;
    }
    for (i = 0; i < streamlist->maxstreams; ++i) {
        streamlist->streams[i] = 0;
    }
    return streamlist;
}

int jpc_streamlist_insert(jpc_streamlist_t *streamlist, int streamno,
  jas_stream_t *stream)
{
    jas_stream_t **newstreams;
    int newmaxstreams;
    int i;
    /* Grow the array of streams if necessary. */
    if (streamlist->numstreams >= streamlist->maxstreams) {
        newmaxstreams = streamlist->maxstreams + 1024;
        if (!(newstreams = jas_realloc2(streamlist->streams,
          (newmaxstreams + 1024), sizeof(jas_stream_t *)))) {
            return -1;
        }
        for (i = streamlist->numstreams; i < streamlist->maxstreams; ++i) {
            streamlist->streams[i] = 0;
        }
        streamlist->maxstreams = newmaxstreams;
        streamlist->streams = newstreams;
    }
    if (streamno != streamlist->numstreams) {
        /* Can only handle insertion at start of list. */
        return -1;
    }
    streamlist->streams[streamno] = stream;
    ++streamlist->numstreams;
    return 0;
}

jas_stream_t *jpc_streamlist_remove(jpc_streamlist_t *streamlist, int streamno)
{
    jas_stream_t *stream;
    int i;
    if (streamno >= streamlist->numstreams) {
        abort();
    }
    stream = streamlist->streams[streamno];
    for (i = streamno + 1; i < streamlist->numstreams; ++i) {
        streamlist->streams[i - 1] = streamlist->streams[i];
    }
    --streamlist->numstreams;
    return stream;
}

void jpc_streamlist_destroy(jpc_streamlist_t *streamlist)
{
    int streamno;
    if (streamlist->streams) {
        for (streamno = 0; streamno < streamlist->numstreams;
          ++streamno) {
            jas_stream_close(streamlist->streams[streamno]);
        }
        jas_free(streamlist->streams);
    }
    jas_free(streamlist);
}

jas_stream_t *jpc_streamlist_get(jpc_streamlist_t *streamlist, int streamno)
{
    assert(streamno < streamlist->numstreams);
    return streamlist->streams[streamno];
}

int jpc_streamlist_numstreams(jpc_streamlist_t *streamlist)
{
    return streamlist->numstreams;
}

jpc_ppxstab_t *jpc_ppxstab_create()
{
    jpc_ppxstab_t *tab;

    if (!(tab = jas_malloc(sizeof(jpc_ppxstab_t)))) {
        return 0;
    }
    tab->numents = 0;
    tab->maxents = 0;
    tab->ents = 0;
    return tab;
}

void jpc_ppxstab_destroy(jpc_ppxstab_t *tab)
{
    int i;
    for (i = 0; i < tab->numents; ++i) {
        jpc_ppxstabent_destroy(tab->ents[i]);
    }
    if (tab->ents) {
        jas_free(tab->ents);
    }
    jas_free(tab);
}

int jpc_ppxstab_grow(jpc_ppxstab_t *tab, int maxents)
{
    jpc_ppxstabent_t **newents;
    if (tab->maxents < maxents) {
        newents = jas_realloc2(tab->ents, maxents, sizeof(jpc_ppxstabent_t *));
        if (!newents) {
            return -1;
        }
        tab->ents = newents;
        tab->maxents = maxents;
    }
    return 0;
}

int jpc_ppxstab_insert(jpc_ppxstab_t *tab, jpc_ppxstabent_t *ent)
{
    int inspt;
    int i;

    for (i = 0; i < tab->numents; ++i) {
        if (tab->ents[i]->ind > ent->ind) {
            break;
        }
    }
    inspt = i;

    if (tab->numents >= tab->maxents) {
        if (jpc_ppxstab_grow(tab, tab->maxents + 128)) {
            return -1;
        }
    }

    for (i = tab->numents; i > inspt; --i) {
        tab->ents[i] = tab->ents[i - 1];
    }
    tab->ents[i] = ent;
    ++tab->numents;

    return 0;
}

jpc_streamlist_t *jpc_ppmstabtostreams(jpc_ppxstab_t *tab)
{
    jpc_streamlist_t *streams;
    uchar *dataptr;
    uint_fast32_t datacnt;
    uint_fast32_t tpcnt;
    jpc_ppxstabent_t *ent;
    int entno;
    jas_stream_t *stream;
    int n;

    if (!(streams = jpc_streamlist_create())) {
        goto error;
    }

    if (!tab->numents) {
        return streams;
    }

    entno = 0;
    ent = tab->ents[entno];
    dataptr = ent->data;
    datacnt = ent->len;
    for (;;) {

        /* Get the length of the packet header data for the current
          tile-part. */
        if (datacnt < 4) {
            goto error;
        }
        if (!(stream = jas_stream_memopen(0, 0))) {
            goto error;
        }
        if (jpc_streamlist_insert(streams, jpc_streamlist_numstreams(streams),
          stream)) {
            goto error;
        }
        tpcnt = (dataptr[0] << 24) | (dataptr[1] << 16) | (dataptr[2] << 8)
          | dataptr[3];
        datacnt -= 4;
        dataptr += 4;

        /* Get the packet header data for the current tile-part. */
        while (tpcnt) {
            if (!datacnt) {
                if (++entno >= tab->numents) {
                    goto error;
                }
                ent = tab->ents[entno];
                dataptr = ent->data;
                datacnt = ent->len;
            }
            n = JAS_MIN(tpcnt, datacnt);
            if (jas_stream_write(stream, dataptr, n) != n) {
                goto error;
            }
            tpcnt -= n;
            dataptr += n;
            datacnt -= n;
        }
        jas_stream_rewind(stream);
        if (!datacnt) {
            if (++entno >= tab->numents) {
                break;
            }
            ent = tab->ents[entno];
            dataptr = ent->data;
            datacnt = ent->len;
        }
    }

    return streams;

error:
    jpc_streamlist_destroy(streams);
    return 0;
}

int jpc_pptstabwrite(jas_stream_t *out, jpc_ppxstab_t *tab)
{
    int i;
    jpc_ppxstabent_t *ent;
    for (i = 0; i < tab->numents; ++i) {
        ent = tab->ents[i];
        if (jas_stream_write(out, ent->data, ent->len) != JAS_CAST(int, ent->len)) {
            return -1;
        }
    }
    return 0;
}

jpc_ppxstabent_t *jpc_ppxstabent_create()
{
    jpc_ppxstabent_t *ent;
    if (!(ent = jas_malloc(sizeof(jpc_ppxstabent_t)))) {
        return 0;
    }
    ent->data = 0;
    ent->len = 0;
    ent->ind = 0;
    return ent;
}

void jpc_ppxstabent_destroy(jpc_ppxstabent_t *ent)
{
    if (ent->data) {
        jas_free(ent->data);
    }
    jas_free(ent);
}
