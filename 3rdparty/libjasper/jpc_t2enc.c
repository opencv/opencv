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
 * Tier 2 Encoder
 *
 * $Id: jpc_t2enc.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "jasper/jas_fix.h"
#include "jasper/jas_malloc.h"
#include "jasper/jas_math.h"
#include "jasper/jas_debug.h"

#include "jpc_flt.h"
#include "jpc_t2enc.h"
#include "jpc_t2cod.h"
#include "jpc_tagtree.h"
#include "jpc_enc.h"
#include "jpc_math.h"

/******************************************************************************\
* Code.
\******************************************************************************/

static int jpc_putcommacode(jpc_bitstream_t *out, int n)
{
    assert(n >= 0);

    while (--n >= 0) {
        if (jpc_bitstream_putbit(out, 1) == EOF) {
            return -1;
        }
    }
    if (jpc_bitstream_putbit(out, 0) == EOF) {
        return -1;
    }
    return 0;
}

static int jpc_putnumnewpasses(jpc_bitstream_t *out, int n)
{
    int ret;

    if (n <= 0) {
        return -1;
    } else if (n == 1) {
        ret = jpc_bitstream_putbit(out, 0);
    } else if (n == 2) {
        ret = jpc_bitstream_putbits(out, 2, 2);
    } else if (n <= 5) {
        ret = jpc_bitstream_putbits(out, 4, 0xc | (n - 3));
    } else if (n <= 36) {
        ret = jpc_bitstream_putbits(out, 9, 0x1e0 | (n - 6));
    } else if (n <= 164) {
        ret = jpc_bitstream_putbits(out, 16, 0xff80 | (n - 37));
    } else {
        /* The standard has no provision for encoding a larger value.
        In practice, however, it is highly unlikely that this
        limitation will ever be encountered. */
        return -1;
    }

    return (ret != EOF) ? 0 : (-1);
}

int jpc_enc_encpkts(jpc_enc_t *enc, jas_stream_t *out)
{
    jpc_enc_tile_t *tile;
    jpc_pi_t *pi;

    tile = enc->curtile;

    jpc_init_t2state(enc, 0);
    pi = tile->pi;
    jpc_pi_init(pi);

    if (!jpc_pi_next(pi)) {
        for (;;) {
            if (jpc_enc_encpkt(enc, out, jpc_pi_cmptno(pi), jpc_pi_rlvlno(pi),
              jpc_pi_prcno(pi), jpc_pi_lyrno(pi))) {
                return -1;
            }
            if (jpc_pi_next(pi)) {
                break;
            }
        }
    }

    return 0;
}

int jpc_enc_encpkt(jpc_enc_t *enc, jas_stream_t *out, int compno, int lvlno, int prcno, int lyrno)
{
    jpc_enc_tcmpt_t *comp;
    jpc_enc_rlvl_t *lvl;
    jpc_enc_band_t *band;
    jpc_enc_band_t *endbands;
    jpc_enc_cblk_t *cblk;
    jpc_enc_cblk_t *endcblks;
    jpc_bitstream_t *outb;
    jpc_enc_pass_t *pass;
    jpc_enc_pass_t *startpass;
    jpc_enc_pass_t *lastpass;
    jpc_enc_pass_t *endpass;
    jpc_enc_pass_t *endpasses;
    int i;
    int included;
    int ret;
    jpc_tagtreenode_t *leaf;
    int n;
    int t1;
    int t2;
    int adjust;
    int maxadjust;
    int datalen;
    int numnewpasses;
    int passcount;
    jpc_enc_tile_t *tile;
    jpc_enc_prc_t *prc;
    jpc_enc_cp_t *cp;
    jpc_ms_t *ms;

    tile = enc->curtile;
    cp = enc->cp;

    if (cp->tcp.csty & JPC_COD_SOP) {
        if (!(ms = jpc_ms_create(JPC_MS_SOP))) {
            return -1;
        }
        ms->parms.sop.seqno = jpc_pi_getind(tile->pi);
        if (jpc_putms(out, enc->cstate, ms)) {
            return -1;
        }
        jpc_ms_destroy(ms);
    }

    outb = jpc_bitstream_sopen(out, "w+");
    assert(outb);

    if (jpc_bitstream_putbit(outb, 1) == EOF) {
        return -1;
    }
    JAS_DBGLOG(10, ("\n"));
    JAS_DBGLOG(10, ("present. "));

    comp = &tile->tcmpts[compno];
    lvl = &comp->rlvls[lvlno];
    endbands = &lvl->bands[lvl->numbands];
    for (band = lvl->bands; band != endbands; ++band) {
        if (!band->data) {
            continue;
        }
        prc = &band->prcs[prcno];
        if (!prc->cblks) {
            continue;
        }

        endcblks = &prc->cblks[prc->numcblks];
        for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
            if (!lyrno) {
                leaf = jpc_tagtree_getleaf(prc->nlibtree, cblk - prc->cblks);
                jpc_tagtree_setvalue(prc->nlibtree, leaf, cblk->numimsbs);
            }
            pass = cblk->curpass;
            included = (pass && pass->lyrno == lyrno);
            if (included && (!cblk->numencpasses)) {
                assert(pass->lyrno == lyrno);
                leaf = jpc_tagtree_getleaf(prc->incltree,
                  cblk - prc->cblks);
                jpc_tagtree_setvalue(prc->incltree, leaf, pass->lyrno);
            }
        }

        endcblks = &prc->cblks[prc->numcblks];
        for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
            pass = cblk->curpass;
            included = (pass && pass->lyrno == lyrno);
            if (!cblk->numencpasses) {
                leaf = jpc_tagtree_getleaf(prc->incltree,
                  cblk - prc->cblks);
                if (jpc_tagtree_encode(prc->incltree, leaf, lyrno
                  + 1, outb) < 0) {
                    return -1;
                }
            } else {
                if (jpc_bitstream_putbit(outb, included) == EOF) {
                    return -1;
                }
            }
            JAS_DBGLOG(10, ("included=%d ", included));
            if (!included) {
                continue;
            }
            if (!cblk->numencpasses) {
                i = 1;
                leaf = jpc_tagtree_getleaf(prc->nlibtree, cblk - prc->cblks);
                for (;;) {
                    if ((ret = jpc_tagtree_encode(prc->nlibtree, leaf, i, outb)) < 0) {
                        return -1;
                    }
                    if (ret) {
                        break;
                    }
                    ++i;
                }
                assert(leaf->known_ && i == leaf->value_ + 1);
            }

            endpasses = &cblk->passes[cblk->numpasses];
            startpass = pass;
            endpass = startpass;
            while (endpass != endpasses && endpass->lyrno == lyrno){
                ++endpass;
            }
            numnewpasses = endpass - startpass;
            if (jpc_putnumnewpasses(outb, numnewpasses)) {
                return -1;
            }
            JAS_DBGLOG(10, ("numnewpasses=%d ", numnewpasses));

            lastpass = endpass - 1;
            n = startpass->start;
            passcount = 1;
            maxadjust = 0;
            for (pass = startpass; pass != endpass; ++pass) {
                if (pass->term || pass == lastpass) {
                    datalen = pass->end - n;
                    t1 = jpc_firstone(datalen) + 1;
                    t2 = cblk->numlenbits + jpc_floorlog2(passcount);
                    adjust = JAS_MAX(t1 - t2, 0);
                    maxadjust = JAS_MAX(adjust, maxadjust);
                    n += datalen;
                    passcount = 1;
                } else {
                    ++passcount;
                }
            }
            if (jpc_putcommacode(outb, maxadjust)) {
                return -1;
            }
            cblk->numlenbits += maxadjust;

            lastpass = endpass - 1;
            n = startpass->start;
            passcount = 1;
            for (pass = startpass; pass != endpass; ++pass) {
                if (pass->term || pass == lastpass) {
                    datalen = pass->end - n;
assert(jpc_firstone(datalen) < cblk->numlenbits + jpc_floorlog2(passcount));
                    if (jpc_bitstream_putbits(outb, cblk->numlenbits + jpc_floorlog2(passcount), datalen) == EOF) {
                        return -1;
                    }
                    n += datalen;
                    passcount = 1;
                } else {
                    ++passcount;
                }
            }
        }
    }

    jpc_bitstream_outalign(outb, 0);
    jpc_bitstream_close(outb);

    if (cp->tcp.csty & JPC_COD_EPH) {
        if (!(ms = jpc_ms_create(JPC_MS_EPH))) {
            return -1;
        }
        jpc_putms(out, enc->cstate, ms);
        jpc_ms_destroy(ms);
    }

    comp = &tile->tcmpts[compno];
    lvl = &comp->rlvls[lvlno];
    endbands = &lvl->bands[lvl->numbands];
    for (band = lvl->bands; band != endbands; ++band) {
        if (!band->data) {
            continue;
        }
        prc = &band->prcs[prcno];
        if (!prc->cblks) {
            continue;
        }
        endcblks = &prc->cblks[prc->numcblks];
        for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
            pass = cblk->curpass;

            if (!pass) {
                continue;
            }
            if (pass->lyrno != lyrno) {
                assert(pass->lyrno < 0 || pass->lyrno > lyrno);
                continue;
            }

            endpasses = &cblk->passes[cblk->numpasses];
            startpass = pass;
            endpass = startpass;
            while (endpass != endpasses && endpass->lyrno == lyrno){
                ++endpass;
            }
            lastpass = endpass - 1;
            numnewpasses = endpass - startpass;

            jas_stream_seek(cblk->stream, startpass->start, SEEK_SET);
            assert(jas_stream_tell(cblk->stream) == startpass->start);
            if (jas_stream_copy(out, cblk->stream, lastpass->end - startpass->start)) {
                return -1;
            }
            cblk->curpass = (endpass != endpasses) ? endpass : 0;
            cblk->numencpasses += numnewpasses;

        }
    }

    return 0;
}

void jpc_save_t2state(jpc_enc_t *enc)
{
/* stream pos in embedded T1 stream may be wrong since not saved/restored! */

    jpc_enc_tcmpt_t *comp;
    jpc_enc_tcmpt_t *endcomps;
    jpc_enc_rlvl_t *lvl;
    jpc_enc_rlvl_t *endlvls;
    jpc_enc_band_t *band;
    jpc_enc_band_t *endbands;
    jpc_enc_cblk_t *cblk;
    jpc_enc_cblk_t *endcblks;
    jpc_enc_tile_t *tile;
    int prcno;
    jpc_enc_prc_t *prc;

    tile = enc->curtile;

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
                    jpc_tagtree_copy(prc->savincltree, prc->incltree);
                    jpc_tagtree_copy(prc->savnlibtree, prc->nlibtree);
                    endcblks = &prc->cblks[prc->numcblks];
                    for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                        cblk->savedcurpass = cblk->curpass;
                        cblk->savednumencpasses = cblk->numencpasses;
                        cblk->savednumlenbits = cblk->numlenbits;
                    }
                }
            }
        }
    }

}

void jpc_restore_t2state(jpc_enc_t *enc)
{

    jpc_enc_tcmpt_t *comp;
    jpc_enc_tcmpt_t *endcomps;
    jpc_enc_rlvl_t *lvl;
    jpc_enc_rlvl_t *endlvls;
    jpc_enc_band_t *band;
    jpc_enc_band_t *endbands;
    jpc_enc_cblk_t *cblk;
    jpc_enc_cblk_t *endcblks;
    jpc_enc_tile_t *tile;
    int prcno;
    jpc_enc_prc_t *prc;

    tile = enc->curtile;

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
                    jpc_tagtree_copy(prc->incltree, prc->savincltree);
                    jpc_tagtree_copy(prc->nlibtree, prc->savnlibtree);
                    endcblks = &prc->cblks[prc->numcblks];
                    for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                        cblk->curpass = cblk->savedcurpass;
                        cblk->numencpasses = cblk->savednumencpasses;
                        cblk->numlenbits = cblk->savednumlenbits;
                    }
                }
            }
        }
    }
}

void jpc_init_t2state(jpc_enc_t *enc, int raflag)
{
/* It is assumed that band->numbps and cblk->numbps precomputed */

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
    jpc_tagtreenode_t *leaf;
    jpc_enc_tile_t *tile;
    int prcno;
    jpc_enc_prc_t *prc;

    tile = enc->curtile;

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
                    jpc_tagtree_reset(prc->incltree);
                    jpc_tagtree_reset(prc->nlibtree);
                    endcblks = &prc->cblks[prc->numcblks];
                    for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                        if (jas_stream_rewind(cblk->stream)) {
                            assert(0);
                        }
                        cblk->curpass = (cblk->numpasses > 0) ? cblk->passes : 0;
                        cblk->numencpasses = 0;
                        cblk->numlenbits = 3;
                        cblk->numimsbs = band->numbps - cblk->numbps;
                        assert(cblk->numimsbs >= 0);
                        leaf = jpc_tagtree_getleaf(prc->nlibtree, cblk - prc->cblks);
                        jpc_tagtree_setvalue(prc->nlibtree, leaf, cblk->numimsbs);

                        if (raflag) {
                            endpasses = &cblk->passes[cblk->numpasses];
                            for (pass = cblk->passes; pass != endpasses; ++pass) {
                                pass->lyrno = -1;
                                pass->lyrno = 0;
                            }
                        }
                    }
                }
            }
        }
    }

}

jpc_pi_t *jpc_enc_pi_create(jpc_enc_cp_t *cp, jpc_enc_tile_t *tile)
{
    jpc_pi_t *pi;
    int compno;
    jpc_picomp_t *picomp;
    jpc_pirlvl_t *pirlvl;
    jpc_enc_tcmpt_t *tcomp;
    int rlvlno;
    jpc_enc_rlvl_t *rlvl;
    int prcno;
    int *prclyrno;

    if (!(pi = jpc_pi_create0())) {
        return 0;
    }
    pi->pktno = -1;
    pi->numcomps = cp->numcmpts;
    if (!(pi->picomps = jas_alloc2(pi->numcomps, sizeof(jpc_picomp_t)))) {
        jpc_pi_destroy(pi);
        return 0;
    }
    for (compno = 0, picomp = pi->picomps; compno < pi->numcomps; ++compno,
      ++picomp) {
        picomp->pirlvls = 0;
    }

    for (compno = 0, tcomp = tile->tcmpts, picomp = pi->picomps;
      compno < pi->numcomps; ++compno, ++tcomp, ++picomp) {
        picomp->numrlvls = tcomp->numrlvls;
        if (!(picomp->pirlvls = jas_alloc2(picomp->numrlvls,
          sizeof(jpc_pirlvl_t)))) {
            jpc_pi_destroy(pi);
            return 0;
        }
        for (rlvlno = 0, pirlvl = picomp->pirlvls; rlvlno <
          picomp->numrlvls; ++rlvlno, ++pirlvl) {
            pirlvl->prclyrnos = 0;
        }
        for (rlvlno = 0, pirlvl = picomp->pirlvls, rlvl = tcomp->rlvls;
          rlvlno < picomp->numrlvls; ++rlvlno, ++pirlvl, ++rlvl) {
/* XXX sizeof(long) should be sizeof different type */
            pirlvl->numprcs = rlvl->numprcs;
            if (rlvl->numprcs) {
                if (!(pirlvl->prclyrnos = jas_alloc2(pirlvl->numprcs,
                  sizeof(long)))) {
                    jpc_pi_destroy(pi);
                    return 0;
                }
            } else {
                pirlvl->prclyrnos = 0;
            }
        }
    }

    pi->maxrlvls = 0;
    for (compno = 0, tcomp = tile->tcmpts, picomp = pi->picomps;
      compno < pi->numcomps; ++compno, ++tcomp, ++picomp) {
        picomp->hsamp = cp->ccps[compno].sampgrdstepx;
        picomp->vsamp = cp->ccps[compno].sampgrdstepy;
        for (rlvlno = 0, pirlvl = picomp->pirlvls, rlvl = tcomp->rlvls;
          rlvlno < picomp->numrlvls; ++rlvlno, ++pirlvl, ++rlvl) {
            pirlvl->prcwidthexpn = rlvl->prcwidthexpn;
            pirlvl->prcheightexpn = rlvl->prcheightexpn;
            for (prcno = 0, prclyrno = pirlvl->prclyrnos;
              prcno < pirlvl->numprcs; ++prcno, ++prclyrno) {
                *prclyrno = 0;
            }
            pirlvl->numhprcs = rlvl->numhprcs;
        }
        if (pi->maxrlvls < tcomp->numrlvls) {
            pi->maxrlvls = tcomp->numrlvls;
        }
    }

    pi->numlyrs = tile->numlyrs;
    pi->xstart = tile->tlx;
    pi->ystart = tile->tly;
    pi->xend = tile->brx;
    pi->yend = tile->bry;

    pi->picomp = 0;
    pi->pirlvl = 0;
    pi->x = 0;
    pi->y = 0;
    pi->compno = 0;
    pi->rlvlno = 0;
    pi->prcno = 0;
    pi->lyrno = 0;
    pi->xstep = 0;
    pi->ystep = 0;

    pi->pchgno = -1;

    pi->defaultpchg.prgord = tile->prg;
    pi->defaultpchg.compnostart = 0;
    pi->defaultpchg.compnoend = pi->numcomps;
    pi->defaultpchg.rlvlnostart = 0;
    pi->defaultpchg.rlvlnoend = pi->maxrlvls;
    pi->defaultpchg.lyrnoend = pi->numlyrs;
    pi->pchg = 0;

    pi->valid = 0;

    return pi;
}
