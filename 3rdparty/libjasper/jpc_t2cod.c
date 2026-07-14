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
 * Tier-2 Coding Library
 *
 * $Id: jpc_t2cod.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

#include "jasper/jas_math.h"
#include "jasper/jas_malloc.h"
#include "jasper/jas_math.h"

#include "jpc_cs.h"
#include "jpc_t2cod.h"
#include "jpc_math.h"

static int jpc_pi_nextlrcp(jpc_pi_t *pi);
static int jpc_pi_nextrlcp(jpc_pi_t *pi);
static int jpc_pi_nextrpcl(jpc_pi_t *pi);
static int jpc_pi_nextpcrl(jpc_pi_t *pi);
static int jpc_pi_nextcprl(jpc_pi_t *pi);

int jpc_pi_next(jpc_pi_t *pi)
{
    jpc_pchg_t *pchg;
    int ret;


    for (;;) {

        pi->valid = false;

        if (!pi->pchg) {
            ++pi->pchgno;
            pi->compno = 0;
            pi->rlvlno = 0;
            pi->prcno = 0;
            pi->lyrno = 0;
            pi->prgvolfirst = true;
            if (pi->pchgno < jpc_pchglist_numpchgs(pi->pchglist)) {
                pi->pchg = jpc_pchglist_get(pi->pchglist, pi->pchgno);
            } else if (pi->pchgno == jpc_pchglist_numpchgs(pi->pchglist)) {
                pi->pchg = &pi->defaultpchg;
            } else {
                return 1;
            }
        }

        pchg = pi->pchg;
        switch (pchg->prgord) {
        case JPC_COD_LRCPPRG:
            ret = jpc_pi_nextlrcp(pi);
            break;
        case JPC_COD_RLCPPRG:
            ret = jpc_pi_nextrlcp(pi);
            break;
        case JPC_COD_RPCLPRG:
            ret = jpc_pi_nextrpcl(pi);
            break;
        case JPC_COD_PCRLPRG:
            ret = jpc_pi_nextpcrl(pi);
            break;
        case JPC_COD_CPRLPRG:
            ret = jpc_pi_nextcprl(pi);
            break;
        default:
            ret = -1;
            break;
        }
        if (!ret) {
            pi->valid = true;
            ++pi->pktno;
            return 0;
        }
        pi->pchg = 0;
    }
}

static int jpc_pi_nextlrcp(register jpc_pi_t *pi)
{
    jpc_pchg_t *pchg;
    int *prclyrno;

    pchg = pi->pchg;
    if (!pi->prgvolfirst) {
        prclyrno = &pi->pirlvl->prclyrnos[pi->prcno];
        goto skip;
    } else {
        pi->prgvolfirst = false;
    }

    for (pi->lyrno = 0; pi->lyrno < pi->numlyrs && pi->lyrno <
      JAS_CAST(int, pchg->lyrnoend); ++pi->lyrno) {
        for (pi->rlvlno = pchg->rlvlnostart; pi->rlvlno < pi->maxrlvls &&
          pi->rlvlno < pchg->rlvlnoend; ++pi->rlvlno) {
            for (pi->compno = pchg->compnostart, pi->picomp =
              &pi->picomps[pi->compno]; pi->compno < pi->numcomps
              && pi->compno < JAS_CAST(int, pchg->compnoend); ++pi->compno,
              ++pi->picomp) {
                if (pi->rlvlno >= pi->picomp->numrlvls) {
                    continue;
                }
                pi->pirlvl = &pi->picomp->pirlvls[pi->rlvlno];
                for (pi->prcno = 0, prclyrno =
                  pi->pirlvl->prclyrnos; pi->prcno <
                  pi->pirlvl->numprcs; ++pi->prcno,
                  ++prclyrno) {
                    if (pi->lyrno >= *prclyrno) {
                        *prclyrno = pi->lyrno;
                        ++(*prclyrno);
                        return 0;
                    }
skip:
                    ;
                }
            }
        }
    }
    return 1;
}

static int jpc_pi_nextrlcp(register jpc_pi_t *pi)
{
    jpc_pchg_t *pchg;
    int *prclyrno;

    pchg = pi->pchg;
    if (!pi->prgvolfirst) {
        assert(pi->prcno < pi->pirlvl->numprcs);
        prclyrno = &pi->pirlvl->prclyrnos[pi->prcno];
        goto skip;
    } else {
        pi->prgvolfirst = 0;
    }

    for (pi->rlvlno = pchg->rlvlnostart; pi->rlvlno < pi->maxrlvls &&
      pi->rlvlno < pchg->rlvlnoend; ++pi->rlvlno) {
        for (pi->lyrno = 0; pi->lyrno < pi->numlyrs && pi->lyrno <
          JAS_CAST(int, pchg->lyrnoend); ++pi->lyrno) {
            for (pi->compno = pchg->compnostart, pi->picomp =
              &pi->picomps[pi->compno]; pi->compno < pi->numcomps &&
              pi->compno < JAS_CAST(int, pchg->compnoend); ++pi->compno, ++pi->picomp) {
                if (pi->rlvlno >= pi->picomp->numrlvls) {
                    continue;
                }
                pi->pirlvl = &pi->picomp->pirlvls[pi->rlvlno];
                for (pi->prcno = 0, prclyrno = pi->pirlvl->prclyrnos;
                  pi->prcno < pi->pirlvl->numprcs; ++pi->prcno, ++prclyrno) {
                    if (pi->lyrno >= *prclyrno) {
                        *prclyrno = pi->lyrno;
                        ++(*prclyrno);
                        return 0;
                    }
skip:
                    ;
                }
            }
        }
    }
    return 1;
}

static int jpc_pi_nextrpcl(register jpc_pi_t *pi)
{
    int rlvlno;
    jpc_pirlvl_t *pirlvl;
    jpc_pchg_t *pchg;
    int prchind;
    int prcvind;
    int *prclyrno;
    int compno;
    jpc_picomp_t *picomp;
    int xstep;
    int ystep;
    uint_fast32_t r;
    uint_fast32_t rpx;
    uint_fast32_t rpy;
    uint_fast32_t trx0;
    uint_fast32_t try0;

    pchg = pi->pchg;
    if (!pi->prgvolfirst) {
        goto skip;
    } else {
        pi->xstep = 0;
        pi->ystep = 0;
        for (compno = 0, picomp = pi->picomps; compno < pi->numcomps;
          ++compno, ++picomp) {
            for (rlvlno = 0, pirlvl = picomp->pirlvls; rlvlno <
              picomp->numrlvls; ++rlvlno, ++pirlvl) {
                xstep = picomp->hsamp * (1 << (pirlvl->prcwidthexpn +
                  picomp->numrlvls - rlvlno - 1));
                ystep = picomp->vsamp * (1 << (pirlvl->prcheightexpn +
                  picomp->numrlvls - rlvlno - 1));
                pi->xstep = (!pi->xstep) ? xstep : JAS_MIN(pi->xstep, xstep);
                pi->ystep = (!pi->ystep) ? ystep : JAS_MIN(pi->ystep, ystep);
            }
        }
        pi->prgvolfirst = 0;
    }

    for (pi->rlvlno = pchg->rlvlnostart; pi->rlvlno < pchg->rlvlnoend &&
      pi->rlvlno < pi->maxrlvls; ++pi->rlvlno) {
        for (pi->y = pi->ystart; pi->y < pi->yend; pi->y +=
          pi->ystep - (pi->y % pi->ystep)) {
            for (pi->x = pi->xstart; pi->x < pi->xend; pi->x +=
              pi->xstep - (pi->x % pi->xstep)) {
                for (pi->compno = pchg->compnostart,
                  pi->picomp = &pi->picomps[pi->compno];
                  pi->compno < JAS_CAST(int, pchg->compnoend) && pi->compno <
                  pi->numcomps; ++pi->compno, ++pi->picomp) {
                    if (pi->rlvlno >= pi->picomp->numrlvls) {
                        continue;
                    }
                    pi->pirlvl = &pi->picomp->pirlvls[pi->rlvlno];
                    if (pi->pirlvl->numprcs == 0) {
                        continue;
                    }
                    r = pi->picomp->numrlvls - 1 - pi->rlvlno;
                    rpx = r + pi->pirlvl->prcwidthexpn;
                    rpy = r + pi->pirlvl->prcheightexpn;
                    trx0 = JPC_CEILDIV(pi->xstart, pi->picomp->hsamp << r);
                    try0 = JPC_CEILDIV(pi->ystart, pi->picomp->vsamp << r);
                    if (((pi->x == pi->xstart && ((trx0 << r) % (1 << rpx)))
                      || !(pi->x % (1 << rpx))) &&
                      ((pi->y == pi->ystart && ((try0 << r) % (1 << rpy)))
                      || !(pi->y % (1 << rpy)))) {
                        prchind = JPC_FLOORDIVPOW2(JPC_CEILDIV(pi->x, pi->picomp->hsamp
                          << r), pi->pirlvl->prcwidthexpn) - JPC_FLOORDIVPOW2(trx0,
                          pi->pirlvl->prcwidthexpn);
                        prcvind = JPC_FLOORDIVPOW2(JPC_CEILDIV(pi->y, pi->picomp->vsamp
                          << r), pi->pirlvl->prcheightexpn) - JPC_FLOORDIVPOW2(try0,
                          pi->pirlvl->prcheightexpn);
                        pi->prcno = prcvind * pi->pirlvl->numhprcs + prchind;

                        assert(pi->prcno < pi->pirlvl->numprcs);
                        for (pi->lyrno = 0; pi->lyrno <
                          pi->numlyrs && pi->lyrno < JAS_CAST(int, pchg->lyrnoend); ++pi->lyrno) {
                            prclyrno = &pi->pirlvl->prclyrnos[pi->prcno];
                            if (pi->lyrno >= *prclyrno) {
                                ++(*prclyrno);
                                return 0;
                            }
skip:
                            ;
                        }
                    }
                }
            }
        }
    }
    return 1;
}

static int jpc_pi_nextpcrl(register jpc_pi_t *pi)
{
    int rlvlno;
    jpc_pirlvl_t *pirlvl;
    jpc_pchg_t *pchg;
    int prchind;
    int prcvind;
    int *prclyrno;
    int compno;
    jpc_picomp_t *picomp;
    int xstep;
    int ystep;
    uint_fast32_t trx0;
    uint_fast32_t try0;
    uint_fast32_t r;
    uint_fast32_t rpx;
    uint_fast32_t rpy;

    pchg = pi->pchg;
    if (!pi->prgvolfirst) {
        goto skip;
    } else {
        pi->xstep = 0;
        pi->ystep = 0;
        for (compno = 0, picomp = pi->picomps; compno < pi->numcomps;
          ++compno, ++picomp) {
            for (rlvlno = 0, pirlvl = picomp->pirlvls; rlvlno <
              picomp->numrlvls; ++rlvlno, ++pirlvl) {
                xstep = picomp->hsamp * (1 <<
                  (pirlvl->prcwidthexpn + picomp->numrlvls -
                  rlvlno - 1));
                ystep = picomp->vsamp * (1 <<
                  (pirlvl->prcheightexpn + picomp->numrlvls -
                  rlvlno - 1));
                pi->xstep = (!pi->xstep) ? xstep :
                  JAS_MIN(pi->xstep, xstep);
                pi->ystep = (!pi->ystep) ? ystep :
                  JAS_MIN(pi->ystep, ystep);
            }
        }
        pi->prgvolfirst = 0;
    }

    for (pi->y = pi->ystart; pi->y < pi->yend; pi->y += pi->ystep -
      (pi->y % pi->ystep)) {
        for (pi->x = pi->xstart; pi->x < pi->xend; pi->x += pi->xstep -
          (pi->x % pi->xstep)) {
            for (pi->compno = pchg->compnostart, pi->picomp =
              &pi->picomps[pi->compno]; pi->compno < pi->numcomps
              && pi->compno < JAS_CAST(int, pchg->compnoend); ++pi->compno,
              ++pi->picomp) {
                for (pi->rlvlno = pchg->rlvlnostart,
                  pi->pirlvl = &pi->picomp->pirlvls[pi->rlvlno];
                  pi->rlvlno < pi->picomp->numrlvls &&
                  pi->rlvlno < pchg->rlvlnoend; ++pi->rlvlno,
                  ++pi->pirlvl) {
                    if (pi->pirlvl->numprcs == 0) {
                        continue;
                    }
                    r = pi->picomp->numrlvls - 1 - pi->rlvlno;
                    trx0 = JPC_CEILDIV(pi->xstart, pi->picomp->hsamp << r);
                    try0 = JPC_CEILDIV(pi->ystart, pi->picomp->vsamp << r);
                    rpx = r + pi->pirlvl->prcwidthexpn;
                    rpy = r + pi->pirlvl->prcheightexpn;
                    if (((pi->x == pi->xstart && ((trx0 << r) % (1 << rpx))) ||
                      !(pi->x % (pi->picomp->hsamp << rpx))) &&
                      ((pi->y == pi->ystart && ((try0 << r) % (1 << rpy))) ||
                      !(pi->y % (pi->picomp->vsamp << rpy)))) {
                        prchind = JPC_FLOORDIVPOW2(JPC_CEILDIV(pi->x, pi->picomp->hsamp
                          << r), pi->pirlvl->prcwidthexpn) - JPC_FLOORDIVPOW2(trx0,
                          pi->pirlvl->prcwidthexpn);
                        prcvind = JPC_FLOORDIVPOW2(JPC_CEILDIV(pi->y, pi->picomp->vsamp
                          << r), pi->pirlvl->prcheightexpn) - JPC_FLOORDIVPOW2(try0,
                          pi->pirlvl->prcheightexpn);
                        pi->prcno = prcvind * pi->pirlvl->numhprcs + prchind;
                        assert(pi->prcno < pi->pirlvl->numprcs);
                        for (pi->lyrno = 0; pi->lyrno < pi->numlyrs &&
                          pi->lyrno < JAS_CAST(int, pchg->lyrnoend); ++pi->lyrno) {
                            prclyrno = &pi->pirlvl->prclyrnos[pi->prcno];
                            if (pi->lyrno >= *prclyrno) {
                                ++(*prclyrno);
                                return 0;
                            }
skip:
                            ;
                        }
                    }
                }
            }
        }
    }
    return 1;
}

static int jpc_pi_nextcprl(register jpc_pi_t *pi)
{
    int rlvlno;
    jpc_pirlvl_t *pirlvl;
    jpc_pchg_t *pchg;
    int prchind;
    int prcvind;
    int *prclyrno;
    uint_fast32_t trx0;
    uint_fast32_t try0;
    uint_fast32_t r;
    uint_fast32_t rpx;
    uint_fast32_t rpy;

    pchg = pi->pchg;
    if (!pi->prgvolfirst) {
        goto skip;
    } else {
        pi->prgvolfirst = 0;
    }

    for (pi->compno = pchg->compnostart, pi->picomp =
      &pi->picomps[pi->compno]; pi->compno < JAS_CAST(int, pchg->compnoend); ++pi->compno,
      ++pi->picomp) {
        pirlvl = pi->picomp->pirlvls;
        pi->xstep = pi->picomp->hsamp * (1 << (pirlvl->prcwidthexpn +
          pi->picomp->numrlvls - 1));
        pi->ystep = pi->picomp->vsamp * (1 << (pirlvl->prcheightexpn +
          pi->picomp->numrlvls - 1));
        for (rlvlno = 1, pirlvl = &pi->picomp->pirlvls[1];
          rlvlno < pi->picomp->numrlvls; ++rlvlno, ++pirlvl) {
            pi->xstep = JAS_MIN(pi->xstep, pi->picomp->hsamp * (1 <<
              (pirlvl->prcwidthexpn + pi->picomp->numrlvls -
              rlvlno - 1)));
            pi->ystep = JAS_MIN(pi->ystep, pi->picomp->vsamp * (1 <<
              (pirlvl->prcheightexpn + pi->picomp->numrlvls -
              rlvlno - 1)));
        }
        for (pi->y = pi->ystart; pi->y < pi->yend;
          pi->y += pi->ystep - (pi->y % pi->ystep)) {
            for (pi->x = pi->xstart; pi->x < pi->xend;
              pi->x += pi->xstep - (pi->x % pi->xstep)) {
                for (pi->rlvlno = pchg->rlvlnostart,
                  pi->pirlvl = &pi->picomp->pirlvls[pi->rlvlno];
                  pi->rlvlno < pi->picomp->numrlvls && pi->rlvlno <
                  pchg->rlvlnoend; ++pi->rlvlno, ++pi->pirlvl) {
                    if (pi->pirlvl->numprcs == 0) {
                        continue;
                    }
                    r = pi->picomp->numrlvls - 1 - pi->rlvlno;
                    trx0 = JPC_CEILDIV(pi->xstart, pi->picomp->hsamp << r);
                    try0 = JPC_CEILDIV(pi->ystart, pi->picomp->vsamp << r);
                    rpx = r + pi->pirlvl->prcwidthexpn;
                    rpy = r + pi->pirlvl->prcheightexpn;
                    if (((pi->x == pi->xstart && ((trx0 << r) % (1 << rpx))) ||
                      !(pi->x % (pi->picomp->hsamp << rpx))) &&
                      ((pi->y == pi->ystart && ((try0 << r) % (1 << rpy))) ||
                      !(pi->y % (pi->picomp->vsamp << rpy)))) {
                        prchind = JPC_FLOORDIVPOW2(JPC_CEILDIV(pi->x, pi->picomp->hsamp
                          << r), pi->pirlvl->prcwidthexpn) - JPC_FLOORDIVPOW2(trx0,
                          pi->pirlvl->prcwidthexpn);
                        prcvind = JPC_FLOORDIVPOW2(JPC_CEILDIV(pi->y, pi->picomp->vsamp
                          << r), pi->pirlvl->prcheightexpn) - JPC_FLOORDIVPOW2(try0,
                          pi->pirlvl->prcheightexpn);
                        pi->prcno = prcvind *
                          pi->pirlvl->numhprcs +
                          prchind;
                        assert(pi->prcno <
                          pi->pirlvl->numprcs);
                        for (pi->lyrno = 0; pi->lyrno <
                          pi->numlyrs && pi->lyrno < JAS_CAST(int, pchg->lyrnoend); ++pi->lyrno) {
                            prclyrno = &pi->pirlvl->prclyrnos[pi->prcno];
                            if (pi->lyrno >= *prclyrno) {
                                ++(*prclyrno);
                                return 0;
                            }
skip:
                            ;
                        }
                    }
                }
            }
        }
    }
    return 1;
}

static void pirlvl_destroy(jpc_pirlvl_t *rlvl)
{
    if (rlvl->prclyrnos) {
        jas_free(rlvl->prclyrnos);
    }
}

static void jpc_picomp_destroy(jpc_picomp_t *picomp)
{
    int rlvlno;
    jpc_pirlvl_t *pirlvl;
    if (picomp->pirlvls) {
        for (rlvlno = 0, pirlvl = picomp->pirlvls; rlvlno <
          picomp->numrlvls; ++rlvlno, ++pirlvl) {
            pirlvl_destroy(pirlvl);
        }
        jas_free(picomp->pirlvls);
    }
}

void jpc_pi_destroy(jpc_pi_t *pi)
{
    jpc_picomp_t *picomp;
    int compno;
    if (pi->picomps) {
        for (compno = 0, picomp = pi->picomps; compno < pi->numcomps;
          ++compno, ++picomp) {
            jpc_picomp_destroy(picomp);
        }
        jas_free(pi->picomps);
    }
    if (pi->pchglist) {
        jpc_pchglist_destroy(pi->pchglist);
    }
    jas_free(pi);
}

jpc_pi_t *jpc_pi_create0()
{
    jpc_pi_t *pi;
    if (!(pi = jas_malloc(sizeof(jpc_pi_t)))) {
        return 0;
    }
    pi->picomps = 0;
    pi->pchgno = 0;
    if (!(pi->pchglist = jpc_pchglist_create())) {
        jas_free(pi);
        return 0;
    }
    return pi;
}

int jpc_pi_addpchg(jpc_pi_t *pi, jpc_pocpchg_t *pchg)
{
    return jpc_pchglist_insert(pi->pchglist, -1, pchg);
}

jpc_pchglist_t *jpc_pchglist_create()
{
    jpc_pchglist_t *pchglist;
    if (!(pchglist = jas_malloc(sizeof(jpc_pchglist_t)))) {
        return 0;
    }
    pchglist->numpchgs = 0;
    pchglist->maxpchgs = 0;
    pchglist->pchgs = 0;
    return pchglist;
}

int jpc_pchglist_insert(jpc_pchglist_t *pchglist, int pchgno, jpc_pchg_t *pchg)
{
    int i;
    int newmaxpchgs;
    jpc_pchg_t **newpchgs;
    if (pchgno < 0) {
        pchgno = pchglist->numpchgs;
    }
    if (pchglist->numpchgs >= pchglist->maxpchgs) {
        newmaxpchgs = pchglist->maxpchgs + 128;
        if (!(newpchgs = jas_realloc2(pchglist->pchgs, newmaxpchgs, sizeof(jpc_pchg_t *)))) {
            return -1;
        }
        pchglist->maxpchgs = newmaxpchgs;
        pchglist->pchgs = newpchgs;
    }
    for (i = pchglist->numpchgs; i > pchgno; --i) {
        pchglist->pchgs[i] = pchglist->pchgs[i - 1];
    }
    pchglist->pchgs[pchgno] = pchg;
    ++pchglist->numpchgs;
    return 0;
}

jpc_pchg_t *jpc_pchglist_remove(jpc_pchglist_t *pchglist, int pchgno)
{
    int i;
    jpc_pchg_t *pchg;
    assert(pchgno < pchglist->numpchgs);
    pchg = pchglist->pchgs[pchgno];
    for (i = pchgno + 1; i < pchglist->numpchgs; ++i) {
        pchglist->pchgs[i - 1] = pchglist->pchgs[i];
    }
    --pchglist->numpchgs;
    return pchg;
}

jpc_pchg_t *jpc_pchg_copy(jpc_pchg_t *pchg)
{
    jpc_pchg_t *newpchg;
    if (!(newpchg = jas_malloc(sizeof(jpc_pchg_t)))) {
        return 0;
    }
    *newpchg = *pchg;
    return newpchg;
}

jpc_pchglist_t *jpc_pchglist_copy(jpc_pchglist_t *pchglist)
{
    jpc_pchglist_t *newpchglist;
    jpc_pchg_t *newpchg;
    int pchgno;
    if (!(newpchglist = jpc_pchglist_create())) {
        return 0;
    }
    for (pchgno = 0; pchgno < pchglist->numpchgs; ++pchgno) {
        if (!(newpchg = jpc_pchg_copy(pchglist->pchgs[pchgno])) ||
          jpc_pchglist_insert(newpchglist, -1, newpchg)) {
            jpc_pchglist_destroy(newpchglist);
            return 0;
        }
    }
    return newpchglist;
}

void jpc_pchglist_destroy(jpc_pchglist_t *pchglist)
{
    int pchgno;
    if (pchglist->pchgs) {
        for (pchgno = 0; pchgno < pchglist->numpchgs; ++pchgno) {
            jpc_pchg_destroy(pchglist->pchgs[pchgno]);
        }
        jas_free(pchglist->pchgs);
    }
    jas_free(pchglist);
}

void jpc_pchg_destroy(jpc_pchg_t *pchg)
{
    jas_free(pchg);
}

jpc_pchg_t *jpc_pchglist_get(jpc_pchglist_t *pchglist, int pchgno)
{
    return pchglist->pchgs[pchgno];
}

int jpc_pchglist_numpchgs(jpc_pchglist_t *pchglist)
{
    return pchglist->numpchgs;
}

int jpc_pi_init(jpc_pi_t *pi)
{
    int compno;
    int rlvlno;
    int prcno;
    jpc_picomp_t *picomp;
    jpc_pirlvl_t *pirlvl;
    int *prclyrno;

    pi->prgvolfirst = 0;
    pi->valid = 0;
    pi->pktno = -1;
    pi->pchgno = -1;
    pi->pchg = 0;

    for (compno = 0, picomp = pi->picomps; compno < pi->numcomps;
      ++compno, ++picomp) {
        for (rlvlno = 0, pirlvl = picomp->pirlvls; rlvlno <
          picomp->numrlvls; ++rlvlno, ++pirlvl) {
            for (prcno = 0, prclyrno = pirlvl->prclyrnos;
              prcno < pirlvl->numprcs; ++prcno, ++prclyrno) {
                *prclyrno = 0;
            }
        }
    }
    return 0;
}
