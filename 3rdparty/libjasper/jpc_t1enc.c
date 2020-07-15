/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2002 Michael David Adams.
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
 * Tier 1 Encoder
 *
 * $Id: jpc_t1enc.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
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

#include "jpc_t1enc.h"
#include "jpc_t1cod.h"
#include "jpc_enc.h"
#include "jpc_cod.h"
#include "jpc_math.h"

static int jpc_encsigpass(jpc_mqenc_t *mqenc, int bitpos, int orient, int,
  jas_matrix_t *flags, jas_matrix_t *data, int term, long *nmsedec);

static int jpc_encrefpass(jpc_mqenc_t *mqenc, int bitpos, int, jas_matrix_t *flags,
  jas_matrix_t *data, int term, long *nmsedec);

static int jpc_encclnpass(jpc_mqenc_t *mqenc, int bitpos, int orient, int,
  int, jas_matrix_t *flags, jas_matrix_t *data, int term, long *nmsedec);

static int jpc_encrawsigpass(jpc_bitstream_t *out, int bitpos, int,
  jas_matrix_t *flags, jas_matrix_t *data, int term, long *nmsedec);

static int jpc_encrawrefpass(jpc_bitstream_t *out, int bitpos, int,
  jas_matrix_t *flags, jas_matrix_t *data, int term, long *nmsedec);

/******************************************************************************\
* Code for encoding code blocks.
\******************************************************************************/

/* Encode all of the code blocks associated with the current tile. */
int jpc_enc_enccblks(jpc_enc_t *enc)
{
    jpc_enc_tcmpt_t *tcmpt;
    jpc_enc_tcmpt_t *endcomps;
    jpc_enc_rlvl_t *lvl;
    jpc_enc_rlvl_t *endlvls;
    jpc_enc_band_t *band;
    jpc_enc_band_t *endbands;
    jpc_enc_cblk_t *cblk;
    jpc_enc_cblk_t *endcblks;
    int i;
    int j;
    int mx;
    int bmx;
    int v;
    jpc_enc_tile_t *tile;
    uint_fast32_t prcno;
    jpc_enc_prc_t *prc;

    tile = enc->curtile;

    endcomps = &tile->tcmpts[tile->numtcmpts];
    for (tcmpt = tile->tcmpts; tcmpt != endcomps; ++tcmpt) {
        endlvls = &tcmpt->rlvls[tcmpt->numrlvls];
        for (lvl = tcmpt->rlvls; lvl != endlvls; ++lvl) {
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
                    bmx = 0;
                    endcblks = &prc->cblks[prc->numcblks];
                    for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                        mx = 0;
                        for (i = 0; i < jas_matrix_numrows(cblk->data); ++i) {
                            for (j = 0; j < jas_matrix_numcols(cblk->data); ++j) {
                                v = abs(jas_matrix_get(cblk->data, i, j));
                                if (v > mx) {
                                    mx = v;
                                }
                            }
                        }
                        if (mx > bmx) {
                            bmx = mx;
                        }
                        cblk->numbps = JAS_MAX(jpc_firstone(mx) + 1 - JPC_NUMEXTRABITS, 0);
                    }

                    for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                        cblk->numimsbs = band->numbps - cblk->numbps;
                        assert(cblk->numimsbs >= 0);
                    }

                    for (cblk = prc->cblks; cblk != endcblks; ++cblk) {
                        if (jpc_enc_enccblk(enc, cblk->stream, tcmpt, band, cblk)) {
                            return -1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int getthebyte(jas_stream_t *in, long off)
{
    int c;
    long oldpos;
    oldpos = jas_stream_tell(in);
    assert(oldpos >= 0);
    jas_stream_seek(in, off, SEEK_SET);
    c = jas_stream_peekc(in);
    jas_stream_seek(in, oldpos, SEEK_SET);
    return c;
}

/* Encode a single code block. */
int jpc_enc_enccblk(jpc_enc_t *enc, jas_stream_t *out, jpc_enc_tcmpt_t *tcmpt, jpc_enc_band_t *band, jpc_enc_cblk_t *cblk)
{
    jpc_enc_pass_t *pass;
    jpc_enc_pass_t *endpasses;
    int bitpos;
    int n;
    int adjust;
    int ret;
    int passtype;
    int t;
    jpc_bitstream_t *bout;
    jpc_enc_pass_t *termpass;
    jpc_enc_rlvl_t *rlvl;
    int vcausal;
    int segsym;
    int termmode;
    int c;

    bout = 0;
    rlvl = band->rlvl;

    cblk->stream = jas_stream_memopen(0, 0);
    assert(cblk->stream);
    cblk->mqenc = jpc_mqenc_create(JPC_NUMCTXS, cblk->stream);
    assert(cblk->mqenc);
    jpc_mqenc_setctxs(cblk->mqenc, JPC_NUMCTXS, jpc_mqctxs);

    cblk->numpasses = (cblk->numbps > 0) ? (3 * cblk->numbps - 2) : 0;
    if (cblk->numpasses > 0) {
        cblk->passes = jas_alloc2(cblk->numpasses, sizeof(jpc_enc_pass_t));
        assert(cblk->passes);
    } else {
        cblk->passes = 0;
    }
    endpasses = &cblk->passes[cblk->numpasses];
    for (pass = cblk->passes; pass != endpasses; ++pass) {
        pass->start = 0;
        pass->end = 0;
        pass->term = JPC_ISTERMINATED(pass - cblk->passes, 0, cblk->numpasses, (tcmpt->cblksty & JPC_COX_TERMALL) != 0, (tcmpt->cblksty & JPC_COX_LAZY) != 0);
        pass->type = JPC_SEGTYPE(pass - cblk->passes, 0, (tcmpt->cblksty & JPC_COX_LAZY) != 0);
        pass->lyrno = -1;
if (pass == endpasses - 1) {
assert(pass->term == 1);
    pass->term = 1;
}
    }

    cblk->flags = jas_matrix_create(jas_matrix_numrows(cblk->data) + 2,
      jas_matrix_numcols(cblk->data) + 2);
    assert(cblk->flags);


    bitpos = cblk->numbps - 1;
    pass = cblk->passes;
    n = cblk->numpasses;
    while (--n >= 0) {

        if (pass->type == JPC_SEG_MQ) {
            /* NOP */
        } else {
            assert(pass->type == JPC_SEG_RAW);
            if (!bout) {
                bout = jpc_bitstream_sopen(cblk->stream, "w");
                assert(bout);
            }
        }

#if 1
        passtype = (pass - cblk->passes + 2) % 3;
#else
        passtype = JPC_PASSTYPE(pass - cblk->passes + 2);
#endif
        pass->start = jas_stream_tell(cblk->stream);
#if 0
assert(jas_stream_tell(cblk->stream) == jas_stream_getrwcount(cblk->stream));
#endif
        assert(bitpos >= 0);
        vcausal = (tcmpt->cblksty & JPC_COX_VSC) != 0;
        segsym = (tcmpt->cblksty & JPC_COX_SEGSYM) != 0;
        if (pass->term) {
            termmode = ((tcmpt->cblksty & JPC_COX_PTERM) ?
              JPC_MQENC_PTERM : JPC_MQENC_DEFTERM) + 1;
        } else {
            termmode = 0;
        }
        switch (passtype) {
        case JPC_SIGPASS:
            ret = (pass->type == JPC_SEG_MQ) ? jpc_encsigpass(cblk->mqenc,
              bitpos, band->orient, vcausal, cblk->flags,
              cblk->data, termmode, &pass->nmsedec) :
              jpc_encrawsigpass(bout, bitpos, vcausal, cblk->flags,
              cblk->data, termmode, &pass->nmsedec);
            break;
        case JPC_REFPASS:
            ret = (pass->type == JPC_SEG_MQ) ? jpc_encrefpass(cblk->mqenc,
              bitpos, vcausal, cblk->flags, cblk->data, termmode,
              &pass->nmsedec) : jpc_encrawrefpass(bout, bitpos,
              vcausal, cblk->flags, cblk->data, termmode,
              &pass->nmsedec);
            break;
        case JPC_CLNPASS:
            assert(pass->type == JPC_SEG_MQ);
            ret = jpc_encclnpass(cblk->mqenc, bitpos, band->orient,
              vcausal, segsym, cblk->flags, cblk->data, termmode,
              &pass->nmsedec);
            break;
        default:
            assert(0);
            break;
        }

        if (pass->type == JPC_SEG_MQ) {
            if (pass->term) {
                jpc_mqenc_init(cblk->mqenc);
            }
            jpc_mqenc_getstate(cblk->mqenc, &pass->mqencstate);
            pass->end = jas_stream_tell(cblk->stream);
            if (tcmpt->cblksty & JPC_COX_RESET) {
                jpc_mqenc_setctxs(cblk->mqenc, JPC_NUMCTXS, jpc_mqctxs);
            }
        } else {
            if (pass->term) {
                if (jpc_bitstream_pending(bout)) {
                    jpc_bitstream_outalign(bout, 0x2a);
                }
                jpc_bitstream_close(bout);
                bout = 0;
                pass->end = jas_stream_tell(cblk->stream);
            } else {
                pass->end = jas_stream_tell(cblk->stream) +
                  jpc_bitstream_pending(bout);
/* NOTE - This will not work.  need to adjust by # of pending output bytes */
            }
        }
#if 0
/* XXX - This assertion fails sometimes when various coding modes are used.
This seems to be harmless, but why does it happen at all? */
assert(jas_stream_tell(cblk->stream) == jas_stream_getrwcount(cblk->stream));
#endif

        pass->wmsedec = jpc_fixtodbl(band->rlvl->tcmpt->synweight) *
          jpc_fixtodbl(band->rlvl->tcmpt->synweight) *
          jpc_fixtodbl(band->synweight) *
          jpc_fixtodbl(band->synweight) *
          jpc_fixtodbl(band->absstepsize) * jpc_fixtodbl(band->absstepsize) *
          ((double) (1 << bitpos)) * ((double)(1 << bitpos)) *
          jpc_fixtodbl(pass->nmsedec);
        pass->cumwmsedec = pass->wmsedec;
        if (pass != cblk->passes) {
            pass->cumwmsedec += pass[-1].cumwmsedec;
        }
        if (passtype == JPC_CLNPASS) {
            --bitpos;
        }
        ++pass;
    }

#if 0
dump_passes(cblk->passes, cblk->numpasses, cblk);
#endif

    n = 0;
    endpasses = &cblk->passes[cblk->numpasses];
    for (pass = cblk->passes; pass != endpasses; ++pass) {
        if (pass->start < n) {
            pass->start = n;
        }
        if (pass->end < n) {
            pass->end = n;
        }
        if (!pass->term) {
            termpass = pass;
            while (termpass - pass < cblk->numpasses &&
              !termpass->term) {
                ++termpass;
            }
            if (pass->type == JPC_SEG_MQ) {
                t = (pass->mqencstate.lastbyte == 0xff) ? 1 : 0;
                if (pass->mqencstate.ctreg >= 5) {
                    adjust = 4 + t;
                } else {
                    adjust = 5 + t;
                }
                pass->end += adjust;
            }
            if (pass->end > termpass->end) {
                pass->end = termpass->end;
            }
            if ((c = getthebyte(cblk->stream, pass->end - 1)) == EOF) {
                abort();
            }
            if (c == 0xff) {
                ++pass->end;
            }
            n = JAS_MAX(n, pass->end);
        } else {
            n = JAS_MAX(n, pass->end);
        }
    }

#if 0
dump_passes(cblk->passes, cblk->numpasses, cblk);
#endif

    if (bout) {
        jpc_bitstream_close(bout);
    }

    return 0;
}

/******************************************************************************\
* Code for significance pass.
\******************************************************************************/

#define	sigpass_step(fp, frowstep, dp, bitpos, one, nmsedec, orient, mqenc, vcausalflag) \
{ \
    int f; \
    int v; \
    f = *(fp); \
    if ((f & JPC_OTHSIGMSK) && !(f & (JPC_SIG | JPC_VISIT))) { \
        v = (abs(*(dp)) & (one)) ? 1 : 0; \
        jpc_mqenc_setcurctx(mqenc, JPC_GETZCCTXNO(f, (orient))); \
        jpc_mqenc_putbit(mqenc, v); \
        if (v) { \
            *(nmsedec) += JPC_GETSIGNMSEDEC(abs(*(dp)), (bitpos) + JPC_NUMEXTRABITS); \
            v = ((*(dp) < 0) ? 1 : 0); \
            jpc_mqenc_setcurctx(mqenc, JPC_GETSCCTXNO(f)); \
            jpc_mqenc_putbit(mqenc, v ^ JPC_GETSPB(f)); \
            JPC_UPDATEFLAGS4(fp, frowstep, v, vcausalflag); \
            *(fp) |= JPC_SIG; \
        } \
        *(fp) |= JPC_VISIT; \
    } \
}

static int jpc_encsigpass(jpc_mqenc_t *mqenc, int bitpos, int orient, int vcausalflag,
  jas_matrix_t *flags, jas_matrix_t *data, int term, long *nmsedec)
{
    int i;
    int j;
    int one;
    int vscanlen;
    int width;
    int height;
    int frowstep;
    int drowstep;
    int fstripestep;
    int dstripestep;
    jpc_fix_t *fstripestart;
    jpc_fix_t *dstripestart;
    jpc_fix_t *fp;
    jpc_fix_t *dp;
    jpc_fix_t *fvscanstart;
    jpc_fix_t *dvscanstart;
    int k;

    *nmsedec = 0;
    width = jas_matrix_numcols(data);
    height = jas_matrix_numrows(data);
    frowstep = jas_matrix_rowstep(flags);
    drowstep = jas_matrix_rowstep(data);
    fstripestep = frowstep << 2;
    dstripestep = drowstep << 2;

    one = 1 << (bitpos + JPC_NUMEXTRABITS);

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

            sigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, orient, mqenc, vcausalflag);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            sigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, orient, mqenc, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            sigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, orient, mqenc, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            sigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, orient, mqenc, 0);

        }
    }

    if (term) {
        jpc_mqenc_flush(mqenc, term - 1);
    }

    return jpc_mqenc_error(mqenc) ? (-1) : 0;
}

#define	rawsigpass_step(fp, frowstep, dp, bitpos, one, nmsedec, out, vcausalflag) \
{ \
    jpc_fix_t f = *(fp); \
    jpc_fix_t v; \
    if ((f & JPC_OTHSIGMSK) && !(f & (JPC_SIG | JPC_VISIT))) { \
        v = (abs(*(dp)) & (one)) ? 1 : 0; \
        if ((jpc_bitstream_putbit((out), v)) == EOF) { \
            return -1; \
        } \
        if (v) { \
            *(nmsedec) += JPC_GETSIGNMSEDEC(abs(*(dp)), (bitpos) + JPC_NUMEXTRABITS); \
            v = ((*(dp) < 0) ? 1 : 0); \
            if (jpc_bitstream_putbit(out, v) == EOF) { \
                return -1; \
            } \
            JPC_UPDATEFLAGS4(fp, frowstep, v, vcausalflag); \
            *(fp) |= JPC_SIG; \
        } \
        *(fp) |= JPC_VISIT; \
    } \
}

static int jpc_encrawsigpass(jpc_bitstream_t *out, int bitpos, int vcausalflag, jas_matrix_t *flags,
  jas_matrix_t *data, int term, long *nmsedec)
{
    int i;
    int j;
    int k;
    int one;
    int vscanlen;
    int width;
    int height;
    int frowstep;
    int drowstep;
    int fstripestep;
    int dstripestep;
    jpc_fix_t *fstripestart;
    jpc_fix_t *dstripestart;
    jpc_fix_t *fp;
    jpc_fix_t *dp;
    jpc_fix_t *fvscanstart;
    jpc_fix_t *dvscanstart;

    *nmsedec = 0;
    width = jas_matrix_numcols(data);
    height = jas_matrix_numrows(data);
    frowstep = jas_matrix_rowstep(flags);
    drowstep = jas_matrix_rowstep(data);
    fstripestep = frowstep << 2;
    dstripestep = drowstep << 2;

    one = 1 << (bitpos + JPC_NUMEXTRABITS);

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

            rawsigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, out, vcausalflag);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;

            rawsigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, out, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;

            rawsigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, out, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;

            rawsigpass_step(fp, frowstep, dp, bitpos, one,
              nmsedec, out, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;

        }
    }

    if (term) {
        jpc_bitstream_outalign(out, 0x2a);
    }

    return 0;
}

/******************************************************************************\
* Code for refinement pass.
\******************************************************************************/

#define	refpass_step(fp, dp, bitpos, one, nmsedec, mqenc, vcausalflag) \
{ \
    int v; \
    if (((*(fp)) & (JPC_SIG | JPC_VISIT)) == JPC_SIG) { \
        (d) = *(dp); \
        *(nmsedec) += JPC_GETREFNMSEDEC(abs(d), (bitpos) + JPC_NUMEXTRABITS); \
        jpc_mqenc_setcurctx((mqenc), JPC_GETMAGCTXNO(*(fp))); \
        v = (abs(d) & (one)) ? 1 : 0; \
        jpc_mqenc_putbit((mqenc), v); \
        *(fp) |= JPC_REFINE; \
    } \
}

static int jpc_encrefpass(jpc_mqenc_t *mqenc, int bitpos, int vcausalflag, jas_matrix_t *flags, jas_matrix_t *data,
  int term, long *nmsedec)
{
    int i;
    int j;
    int one;
    int vscanlen;
    int d;
    int width;
    int height;
    int frowstep;
    int drowstep;
    int fstripestep;
    int dstripestep;
    jpc_fix_t *fstripestart;
    jpc_fix_t *dstripestart;
    jpc_fix_t *fvscanstart;
    jpc_fix_t *dvscanstart;
    jpc_fix_t *dp;
    jpc_fix_t *fp;
int k;

    *nmsedec = 0;
    width = jas_matrix_numcols(data);
    height = jas_matrix_numrows(data);
    frowstep = jas_matrix_rowstep(flags);
    drowstep = jas_matrix_rowstep(data);
    fstripestep = frowstep << 2;
    dstripestep = drowstep << 2;

    one = 1 << (bitpos + JPC_NUMEXTRABITS);

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

            refpass_step(fp, dp, bitpos, one, nmsedec,
              mqenc, vcausalflag);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            refpass_step(fp, dp, bitpos, one, nmsedec,
              mqenc, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            refpass_step(fp, dp, bitpos, one, nmsedec,
              mqenc, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            refpass_step(fp, dp, bitpos, one, nmsedec,
              mqenc, 0);

        }
    }

    if (term) {
        jpc_mqenc_flush(mqenc, term - 1);
    }

    return jpc_mqenc_error(mqenc) ? (-1) : 0;
}

#define	rawrefpass_step(fp, dp, bitpos, one, nmsedec, out, vcausalflag) \
{ \
    jpc_fix_t d; \
    jpc_fix_t v; \
    if (((*(fp)) & (JPC_SIG | JPC_VISIT)) == JPC_SIG) { \
        d = *(dp); \
        *(nmsedec) += JPC_GETREFNMSEDEC(abs(d), (bitpos) + JPC_NUMEXTRABITS); \
        v = (abs(d) & (one)) ? 1 : 0; \
        if (jpc_bitstream_putbit((out), v) == EOF) { \
            return -1; \
        } \
        *(fp) |= JPC_REFINE; \
    } \
}

static int jpc_encrawrefpass(jpc_bitstream_t *out, int bitpos, int vcausalflag, jas_matrix_t *flags,
  jas_matrix_t *data, int term, long *nmsedec)
{
    int i;
    int j;
    int k;
    int one;
    int vscanlen;
    int width;
    int height;
    int frowstep;
    int drowstep;
    int fstripestep;
    int dstripestep;
    jpc_fix_t *fstripestart;
    jpc_fix_t *dstripestart;
    jpc_fix_t *fvscanstart;
    jpc_fix_t *dvscanstart;
    jpc_fix_t *dp;
    jpc_fix_t *fp;

    *nmsedec = 0;
    width = jas_matrix_numcols(data);
    height = jas_matrix_numrows(data);
    frowstep = jas_matrix_rowstep(flags);
    drowstep = jas_matrix_rowstep(data);
    fstripestep = frowstep << 2;
    dstripestep = drowstep << 2;

    one = 1 << (bitpos + JPC_NUMEXTRABITS);

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

            rawrefpass_step(fp, dp, bitpos, one, nmsedec,
              out, vcausalflag);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            rawrefpass_step(fp, dp, bitpos, one, nmsedec,
              out, vcausalflag);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            rawrefpass_step(fp, dp, bitpos, one, nmsedec,
              out, vcausalflag);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            rawrefpass_step(fp, dp, bitpos, one, nmsedec,
              out, vcausalflag);

        }
    }

    if (term) {
        jpc_bitstream_outalign(out, 0x2a);
    }

    return 0;
}

/******************************************************************************\
* Code for cleanup pass.
\******************************************************************************/

#define	clnpass_step(fp, frowstep, dp, bitpos, one, orient, nmsedec, mqenc, label1, label2, vcausalflag) \
{ \
    int f; \
    int v; \
label1 \
    f = *(fp); \
    if (!(f & (JPC_SIG | JPC_VISIT))) { \
        jpc_mqenc_setcurctx(mqenc, JPC_GETZCCTXNO(f, (orient))); \
        v = (abs(*(dp)) & (one)) ? 1 : 0; \
        jpc_mqenc_putbit((mqenc), v); \
        if (v) { \
label2 \
            f = *(fp); \
            /* Coefficient is significant. */ \
            *(nmsedec) += JPC_GETSIGNMSEDEC(abs(*(dp)), (bitpos) + JPC_NUMEXTRABITS); \
            jpc_mqenc_setcurctx((mqenc), JPC_GETSCCTXNO(f)); \
            v = ((*(dp) < 0) ? 1 : 0); \
            jpc_mqenc_putbit((mqenc), v ^ JPC_GETSPB(f)); \
            JPC_UPDATEFLAGS4((fp), (frowstep), v, vcausalflag); \
            *(fp) |= JPC_SIG; \
        } \
    } \
    *(fp) &= ~JPC_VISIT; \
}

static int jpc_encclnpass(jpc_mqenc_t *mqenc, int bitpos, int orient, int vcausalflag, int segsymflag, jas_matrix_t *flags,
  jas_matrix_t *data, int term, long *nmsedec)
{
    int i;
    int j;
    int k;
    int vscanlen;
    int v;
    int runlen;
    jpc_fix_t *fp;
    int width;
    int height;
    jpc_fix_t *dp;
    int one;
    int frowstep;
    int drowstep;
    int fstripestep;
    int dstripestep;
    jpc_fix_t *fstripestart;
    jpc_fix_t *dstripestart;
    jpc_fix_t *fvscanstart;
    jpc_fix_t *dvscanstart;

    *nmsedec = 0;
    width = jas_matrix_numcols(data);
    height = jas_matrix_numrows(data);
    frowstep = jas_matrix_rowstep(flags);
    drowstep = jas_matrix_rowstep(data);
    fstripestep = frowstep << 2;
    dstripestep = drowstep << 2;

    one = 1 << (bitpos + JPC_NUMEXTRABITS);

    fstripestart = jas_matrix_getref(flags, 1, 1);
    dstripestart = jas_matrix_getref(data, 0, 0);
    for (i = height; i > 0; i -= 4, fstripestart += fstripestep,
      dstripestart += dstripestep) {
        fvscanstart = fstripestart;
        dvscanstart = dstripestart;
        vscanlen = JAS_MIN(i, 4);
        for (j = width; j > 0; --j, ++fvscanstart, ++dvscanstart) {

            fp = fvscanstart;
            if (vscanlen >= 4 && !((*fp) & (JPC_SIG | JPC_VISIT |
              JPC_OTHSIGMSK)) && (fp += frowstep, !((*fp) & (JPC_SIG |
              JPC_VISIT | JPC_OTHSIGMSK))) && (fp += frowstep, !((*fp) &
              (JPC_SIG | JPC_VISIT | JPC_OTHSIGMSK))) && (fp += frowstep,
              !((*fp) & (JPC_SIG | JPC_VISIT | JPC_OTHSIGMSK)))) {
                dp = dvscanstart;
                for (k = 0; k < vscanlen; ++k) {
                    v = (abs(*dp) & one) ? 1 : 0;
                    if (v) {
                        break;
                    }
                    dp += drowstep;
                }
                runlen = k;
                if (runlen >= 4) {
                    jpc_mqenc_setcurctx(mqenc, JPC_AGGCTXNO);
                    jpc_mqenc_putbit(mqenc, 0);
                    continue;
                }
                jpc_mqenc_setcurctx(mqenc, JPC_AGGCTXNO);
                jpc_mqenc_putbit(mqenc, 1);
                jpc_mqenc_setcurctx(mqenc, JPC_UCTXNO);
                jpc_mqenc_putbit(mqenc, runlen >> 1);
                jpc_mqenc_putbit(mqenc, runlen & 1);
                fp = fvscanstart + frowstep * runlen;
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
                runlen = 0;
                fp = fvscanstart;
                dp = dvscanstart;
                k = vscanlen;
                goto clnpass_full0;
            }
            clnpass_step(fp, frowstep, dp, bitpos, one,
              orient, nmsedec, mqenc, clnpass_full0:, clnpass_partial0:, vcausalflag);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            clnpass_step(fp, frowstep, dp, bitpos, one,
                orient, nmsedec, mqenc, ;, clnpass_partial1:, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            clnpass_step(fp, frowstep, dp, bitpos, one,
                orient, nmsedec, mqenc, ;, clnpass_partial2:, 0);
            if (--k <= 0) {
                continue;
            }
            fp += frowstep;
            dp += drowstep;
            clnpass_step(fp, frowstep, dp, bitpos, one,
                orient, nmsedec, mqenc, ;, clnpass_partial3:, 0);
        }
    }

    if (segsymflag) {
        jpc_mqenc_setcurctx(mqenc, JPC_UCTXNO);
        jpc_mqenc_putbit(mqenc, 1);
        jpc_mqenc_putbit(mqenc, 0);
        jpc_mqenc_putbit(mqenc, 1);
        jpc_mqenc_putbit(mqenc, 0);
    }

    if (term) {
        jpc_mqenc_flush(mqenc, term - 1);
    }

    return jpc_mqenc_error(mqenc) ? (-1) : 0;
}
