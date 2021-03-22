/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2006-2007, Parvatha Elangovan
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

#define OPJ_UINT32_SEMANTICALLY_BUT_INT32 OPJ_UINT32

#include "opj_includes.h"

/** @defgroup PI PI - Implementation of a packet iterator */
/*@{*/

/** @name Local static functions */
/*@{*/

/**
Get next packet in layer-resolution-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static OPJ_BOOL opj_pi_next_lrcp(opj_pi_iterator_t * pi);
/**
Get next packet in resolution-layer-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static OPJ_BOOL opj_pi_next_rlcp(opj_pi_iterator_t * pi);
/**
Get next packet in resolution-precinct-component-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static OPJ_BOOL opj_pi_next_rpcl(opj_pi_iterator_t * pi);
/**
Get next packet in precinct-component-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static OPJ_BOOL opj_pi_next_pcrl(opj_pi_iterator_t * pi);
/**
Get next packet in component-precinct-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static OPJ_BOOL opj_pi_next_cprl(opj_pi_iterator_t * pi);

/**
 * Updates the coding parameters if the encoding is used with Progression order changes and final (or cinema parameters are used).
 *
 * @param   p_cp        the coding parameters to modify
 * @param   p_tileno    the tile index being concerned.
 * @param   p_tx0       X0 parameter for the tile
 * @param   p_tx1       X1 parameter for the tile
 * @param   p_ty0       Y0 parameter for the tile
 * @param   p_ty1       Y1 parameter for the tile
 * @param   p_max_prec  the maximum precision for all the bands of the tile
 * @param   p_max_res   the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min        the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min        the minimum dy of all the components of all the resolutions for the tile.
 */
static void opj_pi_update_encode_poc_and_final(opj_cp_t *p_cp,
        OPJ_UINT32 p_tileno,
        OPJ_UINT32 p_tx0,
        OPJ_UINT32 p_tx1,
        OPJ_UINT32 p_ty0,
        OPJ_UINT32 p_ty1,
        OPJ_UINT32 p_max_prec,
        OPJ_UINT32 p_max_res,
        OPJ_UINT32 p_dx_min,
        OPJ_UINT32 p_dy_min);

/**
 * Updates the coding parameters if the encoding is not used with Progression order changes and final (and cinema parameters are used).
 *
 * @param   p_cp        the coding parameters to modify
 * @param   p_num_comps     the number of components
 * @param   p_tileno    the tile index being concerned.
 * @param   p_tx0       X0 parameter for the tile
 * @param   p_tx1       X1 parameter for the tile
 * @param   p_ty0       Y0 parameter for the tile
 * @param   p_ty1       Y1 parameter for the tile
 * @param   p_max_prec  the maximum precision for all the bands of the tile
 * @param   p_max_res   the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min        the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min        the minimum dy of all the components of all the resolutions for the tile.
 */
static void opj_pi_update_encode_not_poc(opj_cp_t *p_cp,
        OPJ_UINT32 p_num_comps,
        OPJ_UINT32 p_tileno,
        OPJ_UINT32 p_tx0,
        OPJ_UINT32 p_tx1,
        OPJ_UINT32 p_ty0,
        OPJ_UINT32 p_ty1,
        OPJ_UINT32 p_max_prec,
        OPJ_UINT32 p_max_res,
        OPJ_UINT32 p_dx_min,
        OPJ_UINT32 p_dy_min);
/**
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 *
 * @param   p_image         the image being encoded.
 * @param   p_cp            the coding parameters.
 * @param   tileno          the tile index of the tile being encoded.
 * @param   p_tx0           pointer that will hold the X0 parameter for the tile
 * @param   p_tx1           pointer that will hold the X1 parameter for the tile
 * @param   p_ty0           pointer that will hold the Y0 parameter for the tile
 * @param   p_ty1           pointer that will hold the Y1 parameter for the tile
 * @param   p_max_prec      pointer that will hold the maximum precision for all the bands of the tile
 * @param   p_max_res       pointer that will hold the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min            pointer that will hold the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min            pointer that will hold the minimum dy of all the components of all the resolutions for the tile.
 */
static void opj_get_encoding_parameters(const opj_image_t *p_image,
                                        const opj_cp_t *p_cp,
                                        OPJ_UINT32  tileno,
                                        OPJ_UINT32 * p_tx0,
                                        OPJ_UINT32 * p_tx1,
                                        OPJ_UINT32 * p_ty0,
                                        OPJ_UINT32 * p_ty1,
                                        OPJ_UINT32 * p_dx_min,
                                        OPJ_UINT32 * p_dy_min,
                                        OPJ_UINT32 * p_max_prec,
                                        OPJ_UINT32 * p_max_res);

/**
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 * The precinct widths, heights, dx and dy for each component at each resolution will be stored as well.
 * the last parameter of the function should be an array of pointers of size nb components, each pointer leading
 * to an area of size 4 * max_res. The data is stored inside this area with the following pattern :
 * dx_compi_res0 , dy_compi_res0 , w_compi_res0, h_compi_res0 , dx_compi_res1 , dy_compi_res1 , w_compi_res1, h_compi_res1 , ...
 *
 * @param   p_image         the image being encoded.
 * @param   p_cp            the coding parameters.
 * @param   tileno          the tile index of the tile being encoded.
 * @param   p_tx0           pointer that will hold the X0 parameter for the tile
 * @param   p_tx1           pointer that will hold the X1 parameter for the tile
 * @param   p_ty0           pointer that will hold the Y0 parameter for the tile
 * @param   p_ty1           pointer that will hold the Y1 parameter for the tile
 * @param   p_max_prec      pointer that will hold the maximum precision for all the bands of the tile
 * @param   p_max_res       pointer that will hold the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min        pointer that will hold the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min        pointer that will hold the minimum dy of all the components of all the resolutions for the tile.
 * @param   p_resolutions   pointer to an area corresponding to the one described above.
 */
static void opj_get_all_encoding_parameters(const opj_image_t *p_image,
        const opj_cp_t *p_cp,
        OPJ_UINT32 tileno,
        OPJ_UINT32 * p_tx0,
        OPJ_UINT32 * p_tx1,
        OPJ_UINT32 * p_ty0,
        OPJ_UINT32 * p_ty1,
        OPJ_UINT32 * p_dx_min,
        OPJ_UINT32 * p_dy_min,
        OPJ_UINT32 * p_max_prec,
        OPJ_UINT32 * p_max_res,
        OPJ_UINT32 ** p_resolutions);
/**
 * Allocates memory for a packet iterator. Data and data sizes are set by this operation.
 * No other data is set. The include section of the packet  iterator is not allocated.
 *
 * @param   p_image     the image used to initialize the packet iterator (in fact only the number of components is relevant.
 * @param   p_cp        the coding parameters.
 * @param   tileno  the index of the tile from which creating the packet iterator.
 * @param   manager Event manager
 */
static opj_pi_iterator_t * opj_pi_create(const opj_image_t *p_image,
        const opj_cp_t *p_cp,
        OPJ_UINT32 tileno,
        opj_event_mgr_t* manager);
/**
 * FIXME DOC
 */
static void opj_pi_update_decode_not_poc(opj_pi_iterator_t * p_pi,
        opj_tcp_t * p_tcp,
        OPJ_UINT32 p_max_precision,
        OPJ_UINT32 p_max_res);
/**
 * FIXME DOC
 */
static void opj_pi_update_decode_poc(opj_pi_iterator_t * p_pi,
                                     opj_tcp_t * p_tcp,
                                     OPJ_UINT32 p_max_precision,
                                     OPJ_UINT32 p_max_res);

/**
 * FIXME DOC
 */
static OPJ_BOOL opj_pi_check_next_level(OPJ_INT32 pos,
                                        opj_cp_t *cp,
                                        OPJ_UINT32 tileno,
                                        OPJ_UINT32 pino,
                                        const OPJ_CHAR *prog);

/*@}*/

/*@}*/

/*
==========================================================
   local functions
==========================================================
*/

static OPJ_BOOL opj_pi_next_lrcp(opj_pi_iterator_t * pi)
{
    opj_pi_comp_t *comp = NULL;
    opj_pi_resolution_t *res = NULL;
    OPJ_UINT32 index = 0;

    if (pi->poc.compno0 >= pi->numcomps ||
            pi->poc.compno1 >= pi->numcomps + 1) {
        opj_event_msg(pi->manager, EVT_ERROR,
                      "opj_pi_next_lrcp(): invalid compno0/compno1\n");
        return OPJ_FALSE;
    }

    if (!pi->first) {
        comp = &pi->comps[pi->compno];
        res = &comp->resolutions[pi->resno];
        goto LABEL_SKIP;
    } else {
        pi->first = 0;
    }

    for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
        for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1;
                pi->resno++) {
            for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
                comp = &pi->comps[pi->compno];
                if (pi->resno >= comp->numresolutions) {
                    continue;
                }
                res = &comp->resolutions[pi->resno];
                if (!pi->tp_on) {
                    pi->poc.precno1 = res->pw * res->ph;
                }
                for (pi->precno = pi->poc.precno0; pi->precno < pi->poc.precno1; pi->precno++) {
                    index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *
                            pi->step_c + pi->precno * pi->step_p;
                    /* Avoids index out of bounds access with */
                    /* id_000098,sig_11,src_005411,op_havoc,rep_2 of */
                    /* https://github.com/uclouvain/openjpeg/issues/938 */
                    /* Not sure if this is the most clever fix. Perhaps */
                    /* include should be resized when a POC arises, or */
                    /* the POC should be rejected */
                    if (index >= pi->include_size) {
                        opj_event_msg(pi->manager, EVT_ERROR, "Invalid access to pi->include");
                        return OPJ_FALSE;
                    }
                    if (!pi->include[index]) {
                        pi->include[index] = 1;
                        return OPJ_TRUE;
                    }
LABEL_SKIP:
                    ;
                }
            }
        }
    }

    return OPJ_FALSE;
}

static OPJ_BOOL opj_pi_next_rlcp(opj_pi_iterator_t * pi)
{
    opj_pi_comp_t *comp = NULL;
    opj_pi_resolution_t *res = NULL;
    OPJ_UINT32 index = 0;

    if (pi->poc.compno0 >= pi->numcomps ||
            pi->poc.compno1 >= pi->numcomps + 1) {
        opj_event_msg(pi->manager, EVT_ERROR,
                      "opj_pi_next_rlcp(): invalid compno0/compno1\n");
        return OPJ_FALSE;
    }

    if (!pi->first) {
        comp = &pi->comps[pi->compno];
        res = &comp->resolutions[pi->resno];
        goto LABEL_SKIP;
    } else {
        pi->first = 0;
    }

    for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
        for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
            for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
                comp = &pi->comps[pi->compno];
                if (pi->resno >= comp->numresolutions) {
                    continue;
                }
                res = &comp->resolutions[pi->resno];
                if (!pi->tp_on) {
                    pi->poc.precno1 = res->pw * res->ph;
                }
                for (pi->precno = pi->poc.precno0; pi->precno < pi->poc.precno1; pi->precno++) {
                    index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *
                            pi->step_c + pi->precno * pi->step_p;
                    if (index >= pi->include_size) {
                        opj_event_msg(pi->manager, EVT_ERROR, "Invalid access to pi->include");
                        return OPJ_FALSE;
                    }
                    if (!pi->include[index]) {
                        pi->include[index] = 1;
                        return OPJ_TRUE;
                    }
LABEL_SKIP:
                    ;
                }
            }
        }
    }

    return OPJ_FALSE;
}

static OPJ_BOOL opj_pi_next_rpcl(opj_pi_iterator_t * pi)
{
    opj_pi_comp_t *comp = NULL;
    opj_pi_resolution_t *res = NULL;
    OPJ_UINT32 index = 0;

    if (pi->poc.compno0 >= pi->numcomps ||
            pi->poc.compno1 >= pi->numcomps + 1) {
        opj_event_msg(pi->manager, EVT_ERROR,
                      "opj_pi_next_rpcl(): invalid compno0/compno1\n");
        return OPJ_FALSE;
    }

    if (!pi->first) {
        goto LABEL_SKIP;
    } else {
        OPJ_UINT32 compno, resno;
        pi->first = 0;
        pi->dx = 0;
        pi->dy = 0;
        for (compno = 0; compno < pi->numcomps; compno++) {
            comp = &pi->comps[compno];
            for (resno = 0; resno < comp->numresolutions; resno++) {
                OPJ_UINT32 dx, dy;
                res = &comp->resolutions[resno];
                if (res->pdx + comp->numresolutions - 1 - resno < 32 &&
                        comp->dx <= UINT_MAX / (1u << (res->pdx + comp->numresolutions - 1 - resno))) {
                    dx = comp->dx * (1u << (res->pdx + comp->numresolutions - 1 - resno));
                    pi->dx = !pi->dx ? dx : opj_uint_min(pi->dx, dx);
                }
                if (res->pdy + comp->numresolutions - 1 - resno < 32 &&
                        comp->dy <= UINT_MAX / (1u << (res->pdy + comp->numresolutions - 1 - resno))) {
                    dy = comp->dy * (1u << (res->pdy + comp->numresolutions - 1 - resno));
                    pi->dy = !pi->dy ? dy : opj_uint_min(pi->dy, dy);
                }
            }
        }
        if (pi->dx == 0 || pi->dy == 0) {
            return OPJ_FALSE;
        }
    }
    if (!pi->tp_on) {
        pi->poc.ty0 = pi->ty0;
        pi->poc.tx0 = pi->tx0;
        pi->poc.ty1 = pi->ty1;
        pi->poc.tx1 = pi->tx1;
    }
    for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
        for (pi->y = (OPJ_UINT32)pi->poc.ty0; pi->y < (OPJ_UINT32)pi->poc.ty1;
                pi->y += (pi->dy - (pi->y % pi->dy))) {
            for (pi->x = (OPJ_UINT32)pi->poc.tx0; pi->x < (OPJ_UINT32)pi->poc.tx1;
                    pi->x += (pi->dx - (pi->x % pi->dx))) {
                for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
                    OPJ_UINT32 levelno;
                    OPJ_UINT32 trx0, try0;
                    OPJ_UINT32  trx1, try1;
                    OPJ_UINT32  rpx, rpy;
                    OPJ_UINT32  prci, prcj;
                    comp = &pi->comps[pi->compno];
                    if (pi->resno >= comp->numresolutions) {
                        continue;
                    }
                    res = &comp->resolutions[pi->resno];
                    levelno = comp->numresolutions - 1 - pi->resno;
                    /* Avoids division by zero */
                    /* Relates to id_000004,sig_06,src_000679,op_arith8,pos_49,val_-17 */
                    /* of  https://github.com/uclouvain/openjpeg/issues/938 */
                    if (levelno >= 32 ||
                            ((comp->dx << levelno) >> levelno) != comp->dx ||
                            ((comp->dy << levelno) >> levelno) != comp->dy) {
                        continue;
                    }
                    if ((comp->dx << levelno) > INT_MAX ||
                            (comp->dy << levelno) > INT_MAX) {
                        continue;
                    }
                    trx0 = opj_uint_ceildiv(pi->tx0, (comp->dx << levelno));
                    try0 = opj_uint_ceildiv(pi->ty0, (comp->dy << levelno));
                    trx1 = opj_uint_ceildiv(pi->tx1, (comp->dx << levelno));
                    try1 = opj_uint_ceildiv(pi->ty1, (comp->dy << levelno));
                    rpx = res->pdx + levelno;
                    rpy = res->pdy + levelno;

                    /* To avoid divisions by zero / undefined behaviour on shift */
                    /* in below tests */
                    /* Fixes reading id:000026,sig:08,src:002419,op:int32,pos:60,val:+32 */
                    /* of https://github.com/uclouvain/openjpeg/issues/938 */
                    if (rpx >= 31 || ((comp->dx << rpx) >> rpx) != comp->dx ||
                            rpy >= 31 || ((comp->dy << rpy) >> rpy) != comp->dy) {
                        continue;
                    }

                    /* See ISO-15441. B.12.1.3 Resolution level-position-component-layer progression */
                    if (!((pi->y % (comp->dy << rpy) == 0) || ((pi->y == pi->ty0) &&
                            ((try0 << levelno) % (1U << rpy))))) {
                        continue;
                    }
                    if (!((pi->x % (comp->dx << rpx) == 0) || ((pi->x == pi->tx0) &&
                            ((trx0 << levelno) % (1U << rpx))))) {
                        continue;
                    }

                    if ((res->pw == 0) || (res->ph == 0)) {
                        continue;
                    }

                    if ((trx0 == trx1) || (try0 == try1)) {
                        continue;
                    }

                    prci = opj_uint_floordivpow2(opj_uint_ceildiv(pi->x,
                                                 (comp->dx << levelno)), res->pdx)
                           - opj_uint_floordivpow2(trx0, res->pdx);
                    prcj = opj_uint_floordivpow2(opj_uint_ceildiv(pi->y,
                                                 (comp->dy << levelno)), res->pdy)
                           - opj_uint_floordivpow2(try0, res->pdy);
                    pi->precno = prci + prcj * res->pw;
                    for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
                        index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *
                                pi->step_c + pi->precno * pi->step_p;
                        if (index >= pi->include_size) {
                            opj_event_msg(pi->manager, EVT_ERROR, "Invalid access to pi->include");
                            return OPJ_FALSE;
                        }
                        if (!pi->include[index]) {
                            pi->include[index] = 1;
                            return OPJ_TRUE;
                        }
LABEL_SKIP:
                        ;
                    }
                }
            }
        }
    }

    return OPJ_FALSE;
}

static OPJ_BOOL opj_pi_next_pcrl(opj_pi_iterator_t * pi)
{
    opj_pi_comp_t *comp = NULL;
    opj_pi_resolution_t *res = NULL;
    OPJ_UINT32 index = 0;

    if (pi->poc.compno0 >= pi->numcomps ||
            pi->poc.compno1 >= pi->numcomps + 1) {
        opj_event_msg(pi->manager, EVT_ERROR,
                      "opj_pi_next_pcrl(): invalid compno0/compno1\n");
        return OPJ_FALSE;
    }

    if (!pi->first) {
        comp = &pi->comps[pi->compno];
        goto LABEL_SKIP;
    } else {
        OPJ_UINT32 compno, resno;
        pi->first = 0;
        pi->dx = 0;
        pi->dy = 0;
        for (compno = 0; compno < pi->numcomps; compno++) {
            comp = &pi->comps[compno];
            for (resno = 0; resno < comp->numresolutions; resno++) {
                OPJ_UINT32 dx, dy;
                res = &comp->resolutions[resno];
                if (res->pdx + comp->numresolutions - 1 - resno < 32 &&
                        comp->dx <= UINT_MAX / (1u << (res->pdx + comp->numresolutions - 1 - resno))) {
                    dx = comp->dx * (1u << (res->pdx + comp->numresolutions - 1 - resno));
                    pi->dx = !pi->dx ? dx : opj_uint_min(pi->dx, dx);
                }
                if (res->pdy + comp->numresolutions - 1 - resno < 32 &&
                        comp->dy <= UINT_MAX / (1u << (res->pdy + comp->numresolutions - 1 - resno))) {
                    dy = comp->dy * (1u << (res->pdy + comp->numresolutions - 1 - resno));
                    pi->dy = !pi->dy ? dy : opj_uint_min(pi->dy, dy);
                }
            }
        }
        if (pi->dx == 0 || pi->dy == 0) {
            return OPJ_FALSE;
        }
    }
    if (!pi->tp_on) {
        pi->poc.ty0 = pi->ty0;
        pi->poc.tx0 = pi->tx0;
        pi->poc.ty1 = pi->ty1;
        pi->poc.tx1 = pi->tx1;
    }
    for (pi->y = (OPJ_UINT32)pi->poc.ty0; pi->y < (OPJ_UINT32)pi->poc.ty1;
            pi->y += (pi->dy - (pi->y % pi->dy))) {
        for (pi->x = (OPJ_UINT32)pi->poc.tx0; pi->x < (OPJ_UINT32)pi->poc.tx1;
                pi->x += (pi->dx - (pi->x % pi->dx))) {
            for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
                comp = &pi->comps[pi->compno];
                for (pi->resno = pi->poc.resno0;
                        pi->resno < opj_uint_min(pi->poc.resno1, comp->numresolutions); pi->resno++) {
                    OPJ_UINT32 levelno;
                    OPJ_UINT32 trx0, try0;
                    OPJ_UINT32 trx1, try1;
                    OPJ_UINT32 rpx, rpy;
                    OPJ_UINT32 prci, prcj;
                    res = &comp->resolutions[pi->resno];
                    levelno = comp->numresolutions - 1 - pi->resno;
                    /* Avoids division by zero */
                    /* Relates to id_000004,sig_06,src_000679,op_arith8,pos_49,val_-17 */
                    /* of  https://github.com/uclouvain/openjpeg/issues/938 */
                    if (levelno >= 32 ||
                            ((comp->dx << levelno) >> levelno) != comp->dx ||
                            ((comp->dy << levelno) >> levelno) != comp->dy) {
                        continue;
                    }
                    if ((comp->dx << levelno) > INT_MAX ||
                            (comp->dy << levelno) > INT_MAX) {
                        continue;
                    }
                    trx0 = opj_uint_ceildiv(pi->tx0, (comp->dx << levelno));
                    try0 = opj_uint_ceildiv(pi->ty0, (comp->dy << levelno));
                    trx1 = opj_uint_ceildiv(pi->tx1, (comp->dx << levelno));
                    try1 = opj_uint_ceildiv(pi->ty1, (comp->dy << levelno));
                    rpx = res->pdx + levelno;
                    rpy = res->pdy + levelno;

                    /* To avoid divisions by zero / undefined behaviour on shift */
                    /* in below tests */
                    /* Relates to id:000019,sig:08,src:001098,op:flip1,pos:49 */
                    /* of https://github.com/uclouvain/openjpeg/issues/938 */
                    if (rpx >= 31 || ((comp->dx << rpx) >> rpx) != comp->dx ||
                            rpy >= 31 || ((comp->dy << rpy) >> rpy) != comp->dy) {
                        continue;
                    }

                    /* See ISO-15441. B.12.1.4 Position-component-resolution level-layer progression */
                    if (!((pi->y % (comp->dy << rpy) == 0) || ((pi->y == pi->ty0) &&
                            ((try0 << levelno) % (1U << rpy))))) {
                        continue;
                    }
                    if (!((pi->x % (comp->dx << rpx) == 0) || ((pi->x == pi->tx0) &&
                            ((trx0 << levelno) % (1U << rpx))))) {
                        continue;
                    }

                    if ((res->pw == 0) || (res->ph == 0)) {
                        continue;
                    }

                    if ((trx0 == trx1) || (try0 == try1)) {
                        continue;
                    }

                    prci = opj_uint_floordivpow2(opj_uint_ceildiv(pi->x,
                                                 (comp->dx << levelno)), res->pdx)
                           - opj_uint_floordivpow2(trx0, res->pdx);
                    prcj = opj_uint_floordivpow2(opj_uint_ceildiv(pi->y,
                                                 (comp->dy << levelno)), res->pdy)
                           - opj_uint_floordivpow2(try0, res->pdy);
                    pi->precno = prci + prcj * res->pw;
                    for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
                        index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *
                                pi->step_c + pi->precno * pi->step_p;
                        if (index >= pi->include_size) {
                            opj_event_msg(pi->manager, EVT_ERROR, "Invalid access to pi->include");
                            return OPJ_FALSE;
                        }
                        if (!pi->include[index]) {
                            pi->include[index] = 1;
                            return OPJ_TRUE;
                        }
LABEL_SKIP:
                        ;
                    }
                }
            }
        }
    }

    return OPJ_FALSE;
}

static OPJ_BOOL opj_pi_next_cprl(opj_pi_iterator_t * pi)
{
    opj_pi_comp_t *comp = NULL;
    opj_pi_resolution_t *res = NULL;
    OPJ_UINT32 index = 0;

    if (pi->poc.compno0 >= pi->numcomps ||
            pi->poc.compno1 >= pi->numcomps + 1) {
        opj_event_msg(pi->manager, EVT_ERROR,
                      "opj_pi_next_cprl(): invalid compno0/compno1\n");
        return OPJ_FALSE;
    }

    if (!pi->first) {
        comp = &pi->comps[pi->compno];
        goto LABEL_SKIP;
    } else {
        pi->first = 0;
    }

    for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
        OPJ_UINT32 resno;
        comp = &pi->comps[pi->compno];
        pi->dx = 0;
        pi->dy = 0;
        for (resno = 0; resno < comp->numresolutions; resno++) {
            OPJ_UINT32 dx, dy;
            res = &comp->resolutions[resno];
            if (res->pdx + comp->numresolutions - 1 - resno < 32 &&
                    comp->dx <= UINT_MAX / (1u << (res->pdx + comp->numresolutions - 1 - resno))) {
                dx = comp->dx * (1u << (res->pdx + comp->numresolutions - 1 - resno));
                pi->dx = !pi->dx ? dx : opj_uint_min(pi->dx, dx);
            }
            if (res->pdy + comp->numresolutions - 1 - resno < 32 &&
                    comp->dy <= UINT_MAX / (1u << (res->pdy + comp->numresolutions - 1 - resno))) {
                dy = comp->dy * (1u << (res->pdy + comp->numresolutions - 1 - resno));
                pi->dy = !pi->dy ? dy : opj_uint_min(pi->dy, dy);
            }
        }
        if (pi->dx == 0 || pi->dy == 0) {
            return OPJ_FALSE;
        }
        if (!pi->tp_on) {
            pi->poc.ty0 = pi->ty0;
            pi->poc.tx0 = pi->tx0;
            pi->poc.ty1 = pi->ty1;
            pi->poc.tx1 = pi->tx1;
        }
        for (pi->y = (OPJ_UINT32)pi->poc.ty0; pi->y < (OPJ_UINT32)pi->poc.ty1;
                pi->y += (pi->dy - (pi->y % pi->dy))) {
            for (pi->x = (OPJ_UINT32)pi->poc.tx0; pi->x < (OPJ_UINT32)pi->poc.tx1;
                    pi->x += (pi->dx - (pi->x % pi->dx))) {
                for (pi->resno = pi->poc.resno0;
                        pi->resno < opj_uint_min(pi->poc.resno1, comp->numresolutions); pi->resno++) {
                    OPJ_UINT32 levelno;
                    OPJ_UINT32 trx0, try0;
                    OPJ_UINT32 trx1, try1;
                    OPJ_UINT32 rpx, rpy;
                    OPJ_UINT32 prci, prcj;
                    res = &comp->resolutions[pi->resno];
                    levelno = comp->numresolutions - 1 - pi->resno;
                    /* Avoids division by zero on id_000004,sig_06,src_000679,op_arith8,pos_49,val_-17 */
                    /* of  https://github.com/uclouvain/openjpeg/issues/938 */
                    if (levelno >= 32 ||
                            ((comp->dx << levelno) >> levelno) != comp->dx ||
                            ((comp->dy << levelno) >> levelno) != comp->dy) {
                        continue;
                    }
                    if ((comp->dx << levelno) > INT_MAX ||
                            (comp->dy << levelno) > INT_MAX) {
                        continue;
                    }
                    trx0 = opj_uint_ceildiv(pi->tx0, (comp->dx << levelno));
                    try0 = opj_uint_ceildiv(pi->ty0, (comp->dy << levelno));
                    trx1 = opj_uint_ceildiv(pi->tx1, (comp->dx << levelno));
                    try1 = opj_uint_ceildiv(pi->ty1, (comp->dy << levelno));
                    rpx = res->pdx + levelno;
                    rpy = res->pdy + levelno;

                    /* To avoid divisions by zero / undefined behaviour on shift */
                    /* in below tests */
                    /* Fixes reading id:000019,sig:08,src:001098,op:flip1,pos:49 */
                    /* of https://github.com/uclouvain/openjpeg/issues/938 */
                    if (rpx >= 31 || ((comp->dx << rpx) >> rpx) != comp->dx ||
                            rpy >= 31 || ((comp->dy << rpy) >> rpy) != comp->dy) {
                        continue;
                    }

                    /* See ISO-15441. B.12.1.5 Component-position-resolution level-layer progression */
                    if (!((pi->y % (comp->dy << rpy) == 0) || ((pi->y == pi->ty0) &&
                            ((try0 << levelno) % (1U << rpy))))) {
                        continue;
                    }
                    if (!((pi->x % (comp->dx << rpx) == 0) || ((pi->x == pi->tx0) &&
                            ((trx0 << levelno) % (1U << rpx))))) {
                        continue;
                    }

                    if ((res->pw == 0) || (res->ph == 0)) {
                        continue;
                    }

                    if ((trx0 == trx1) || (try0 == try1)) {
                        continue;
                    }

                    prci = opj_uint_floordivpow2(opj_uint_ceildiv(pi->x,
                                                 (comp->dx << levelno)), res->pdx)
                           - opj_uint_floordivpow2(trx0, res->pdx);
                    prcj = opj_uint_floordivpow2(opj_uint_ceildiv(pi->y,
                                                 (comp->dy << levelno)), res->pdy)
                           - opj_uint_floordivpow2(try0, res->pdy);
                    pi->precno = (OPJ_UINT32)(prci + prcj * res->pw);
                    for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
                        index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *
                                pi->step_c + pi->precno * pi->step_p;
                        if (index >= pi->include_size) {
                            opj_event_msg(pi->manager, EVT_ERROR, "Invalid access to pi->include");
                            return OPJ_FALSE;
                        }
                        if (!pi->include[index]) {
                            pi->include[index] = 1;
                            return OPJ_TRUE;
                        }
LABEL_SKIP:
                        ;
                    }
                }
            }
        }
    }

    return OPJ_FALSE;
}

static void opj_get_encoding_parameters(const opj_image_t *p_image,
                                        const opj_cp_t *p_cp,
                                        OPJ_UINT32 p_tileno,
                                        OPJ_UINT32 * p_tx0,
                                        OPJ_UINT32  * p_tx1,
                                        OPJ_UINT32  * p_ty0,
                                        OPJ_UINT32  * p_ty1,
                                        OPJ_UINT32 * p_dx_min,
                                        OPJ_UINT32 * p_dy_min,
                                        OPJ_UINT32 * p_max_prec,
                                        OPJ_UINT32 * p_max_res)
{
    /* loop */
    OPJ_UINT32  compno, resno;
    /* pointers */
    const opj_tcp_t *l_tcp = 00;
    const opj_tccp_t * l_tccp = 00;
    const opj_image_comp_t * l_img_comp = 00;

    /* position in x and y of tile */
    OPJ_UINT32 p, q;

    /* non-corrected (in regard to image offset) tile offset */
    OPJ_UINT32 l_tx0, l_ty0;

    /* preconditions */
    assert(p_cp != 00);
    assert(p_image != 00);
    assert(p_tileno < p_cp->tw * p_cp->th);

    /* initializations */
    l_tcp = &p_cp->tcps [p_tileno];
    l_img_comp = p_image->comps;
    l_tccp = l_tcp->tccps;

    /* here calculation of tx0, tx1, ty0, ty1, maxprec, dx and dy */
    p = p_tileno % p_cp->tw;
    q = p_tileno / p_cp->tw;

    /* find extent of tile */
    l_tx0 = p_cp->tx0 + p *
            p_cp->tdx; /* can't be greater than p_image->x1 so won't overflow */
    *p_tx0 = opj_uint_max(l_tx0, p_image->x0);
    *p_tx1 = opj_uint_min(opj_uint_adds(l_tx0, p_cp->tdx), p_image->x1);
    l_ty0 = p_cp->ty0 + q *
            p_cp->tdy; /* can't be greater than p_image->y1 so won't overflow */
    *p_ty0 = opj_uint_max(l_ty0, p_image->y0);
    *p_ty1 = opj_uint_min(opj_uint_adds(l_ty0, p_cp->tdy), p_image->y1);

    /* max precision is 0 (can only grow) */
    *p_max_prec = 0;
    *p_max_res = 0;

    /* take the largest value for dx_min and dy_min */
    *p_dx_min = 0x7fffffff;
    *p_dy_min  = 0x7fffffff;

    for (compno = 0; compno < p_image->numcomps; ++compno) {
        /* arithmetic variables to calculate */
        OPJ_UINT32 l_level_no;
        OPJ_UINT32 l_rx0, l_ry0, l_rx1, l_ry1;
        OPJ_UINT32 l_px0, l_py0, l_px1, py1;
        OPJ_UINT32 l_pdx, l_pdy;
        OPJ_UINT32 l_pw, l_ph;
        OPJ_UINT32 l_product;
        OPJ_UINT32 l_tcx0, l_tcy0, l_tcx1, l_tcy1;

        l_tcx0 = opj_uint_ceildiv(*p_tx0, l_img_comp->dx);
        l_tcy0 = opj_uint_ceildiv(*p_ty0, l_img_comp->dy);
        l_tcx1 = opj_uint_ceildiv(*p_tx1, l_img_comp->dx);
        l_tcy1 = opj_uint_ceildiv(*p_ty1, l_img_comp->dy);

        if (l_tccp->numresolutions > *p_max_res) {
            *p_max_res = l_tccp->numresolutions;
        }

        /* use custom size for precincts */
        for (resno = 0; resno < l_tccp->numresolutions; ++resno) {
            OPJ_UINT32 l_dx, l_dy;

            /* precinct width and height */
            l_pdx = l_tccp->prcw[resno];
            l_pdy = l_tccp->prch[resno];

            l_dx = l_img_comp->dx * (1u << (l_pdx + l_tccp->numresolutions - 1 - resno));
            l_dy = l_img_comp->dy * (1u << (l_pdy + l_tccp->numresolutions - 1 - resno));

            /* take the minimum size for dx for each comp and resolution */
            *p_dx_min = opj_uint_min(*p_dx_min, l_dx);
            *p_dy_min = opj_uint_min(*p_dy_min, l_dy);

            /* various calculations of extents */
            l_level_no = l_tccp->numresolutions - 1 - resno;

            l_rx0 = opj_uint_ceildivpow2(l_tcx0, l_level_no);
            l_ry0 = opj_uint_ceildivpow2(l_tcy0, l_level_no);
            l_rx1 = opj_uint_ceildivpow2(l_tcx1, l_level_no);
            l_ry1 = opj_uint_ceildivpow2(l_tcy1, l_level_no);

            l_px0 = opj_uint_floordivpow2(l_rx0, l_pdx) << l_pdx;
            l_py0 = opj_uint_floordivpow2(l_ry0, l_pdy) << l_pdy;
            l_px1 = opj_uint_ceildivpow2(l_rx1, l_pdx) << l_pdx;

            py1 = opj_uint_ceildivpow2(l_ry1, l_pdy) << l_pdy;

            l_pw = (l_rx0 == l_rx1) ? 0 : ((l_px1 - l_px0) >> l_pdx);
            l_ph = (l_ry0 == l_ry1) ? 0 : ((py1 - l_py0) >> l_pdy);

            l_product = l_pw * l_ph;

            /* update precision */
            if (l_product > *p_max_prec) {
                *p_max_prec = l_product;
            }
        }
        ++l_img_comp;
        ++l_tccp;
    }
}


static void opj_get_all_encoding_parameters(const opj_image_t *p_image,
        const opj_cp_t *p_cp,
        OPJ_UINT32 tileno,
        OPJ_UINT32 * p_tx0,
        OPJ_UINT32 * p_tx1,
        OPJ_UINT32 * p_ty0,
        OPJ_UINT32 * p_ty1,
        OPJ_UINT32 * p_dx_min,
        OPJ_UINT32 * p_dy_min,
        OPJ_UINT32 * p_max_prec,
        OPJ_UINT32 * p_max_res,
        OPJ_UINT32 ** p_resolutions)
{
    /* loop*/
    OPJ_UINT32 compno, resno;

    /* pointers*/
    const opj_tcp_t *tcp = 00;
    const opj_tccp_t * l_tccp = 00;
    const opj_image_comp_t * l_img_comp = 00;

    /* to store l_dx, l_dy, w and h for each resolution and component.*/
    OPJ_UINT32 * lResolutionPtr;

    /* position in x and y of tile*/
    OPJ_UINT32 p, q;

    /* non-corrected (in regard to image offset) tile offset */
    OPJ_UINT32 l_tx0, l_ty0;

    /* preconditions in debug*/
    assert(p_cp != 00);
    assert(p_image != 00);
    assert(tileno < p_cp->tw * p_cp->th);

    /* initializations*/
    tcp = &p_cp->tcps [tileno];
    l_tccp = tcp->tccps;
    l_img_comp = p_image->comps;

    /* position in x and y of tile*/
    p = tileno % p_cp->tw;
    q = tileno / p_cp->tw;

    /* here calculation of tx0, tx1, ty0, ty1, maxprec, l_dx and l_dy */
    l_tx0 = p_cp->tx0 + p *
            p_cp->tdx; /* can't be greater than p_image->x1 so won't overflow */
    *p_tx0 = opj_uint_max(l_tx0, p_image->x0);
    *p_tx1 = opj_uint_min(opj_uint_adds(l_tx0, p_cp->tdx), p_image->x1);
    l_ty0 = p_cp->ty0 + q *
            p_cp->tdy; /* can't be greater than p_image->y1 so won't overflow */
    *p_ty0 = opj_uint_max(l_ty0, p_image->y0);
    *p_ty1 = opj_uint_min(opj_uint_adds(l_ty0, p_cp->tdy), p_image->y1);

    /* max precision and resolution is 0 (can only grow)*/
    *p_max_prec = 0;
    *p_max_res = 0;

    /* take the largest value for dx_min and dy_min*/
    *p_dx_min = 0x7fffffff;
    *p_dy_min = 0x7fffffff;

    for (compno = 0; compno < p_image->numcomps; ++compno) {
        /* aritmetic variables to calculate*/
        OPJ_UINT32 l_level_no;
        OPJ_UINT32 l_rx0, l_ry0, l_rx1, l_ry1;
        OPJ_UINT32 l_px0, l_py0, l_px1, py1;
        OPJ_UINT32 l_product;
        OPJ_UINT32 l_tcx0, l_tcy0, l_tcx1, l_tcy1;
        OPJ_UINT32 l_pdx, l_pdy, l_pw, l_ph;

        lResolutionPtr = p_resolutions ? p_resolutions[compno] : NULL;

        l_tcx0 = opj_uint_ceildiv(*p_tx0, l_img_comp->dx);
        l_tcy0 = opj_uint_ceildiv(*p_ty0, l_img_comp->dy);
        l_tcx1 = opj_uint_ceildiv(*p_tx1, l_img_comp->dx);
        l_tcy1 = opj_uint_ceildiv(*p_ty1, l_img_comp->dy);

        if (l_tccp->numresolutions > *p_max_res) {
            *p_max_res = l_tccp->numresolutions;
        }

        /* use custom size for precincts*/
        l_level_no = l_tccp->numresolutions;
        for (resno = 0; resno < l_tccp->numresolutions; ++resno) {
            OPJ_UINT32 l_dx, l_dy;

            --l_level_no;

            /* precinct width and height*/
            l_pdx = l_tccp->prcw[resno];
            l_pdy = l_tccp->prch[resno];
            if (lResolutionPtr) {
                *lResolutionPtr++ = l_pdx;
                *lResolutionPtr++ = l_pdy;
            }
            if (l_pdx + l_level_no < 32 &&
                    l_img_comp->dx <= UINT_MAX / (1u << (l_pdx + l_level_no))) {
                l_dx = l_img_comp->dx * (1u << (l_pdx + l_level_no));
                /* take the minimum size for l_dx for each comp and resolution*/
                *p_dx_min = opj_uint_min(*p_dx_min, l_dx);
            }
            if (l_pdy + l_level_no < 32 &&
                    l_img_comp->dy <= UINT_MAX / (1u << (l_pdy + l_level_no))) {
                l_dy = l_img_comp->dy * (1u << (l_pdy + l_level_no));
                *p_dy_min = opj_uint_min(*p_dy_min, l_dy);
            }

            /* various calculations of extents*/
            l_rx0 = opj_uint_ceildivpow2(l_tcx0, l_level_no);
            l_ry0 = opj_uint_ceildivpow2(l_tcy0, l_level_no);
            l_rx1 = opj_uint_ceildivpow2(l_tcx1, l_level_no);
            l_ry1 = opj_uint_ceildivpow2(l_tcy1, l_level_no);
            l_px0 = opj_uint_floordivpow2(l_rx0, l_pdx) << l_pdx;
            l_py0 = opj_uint_floordivpow2(l_ry0, l_pdy) << l_pdy;
            l_px1 = opj_uint_ceildivpow2(l_rx1, l_pdx) << l_pdx;
            py1 = opj_uint_ceildivpow2(l_ry1, l_pdy) << l_pdy;
            l_pw = (l_rx0 == l_rx1) ? 0 : ((l_px1 - l_px0) >> l_pdx);
            l_ph = (l_ry0 == l_ry1) ? 0 : ((py1 - l_py0) >> l_pdy);
            if (lResolutionPtr) {
                *lResolutionPtr++ = l_pw;
                *lResolutionPtr++ = l_ph;
            }
            l_product = l_pw * l_ph;

            /* update precision*/
            if (l_product > *p_max_prec) {
                *p_max_prec = l_product;
            }

        }
        ++l_tccp;
        ++l_img_comp;
    }
}

static opj_pi_iterator_t * opj_pi_create(const opj_image_t *image,
        const opj_cp_t *cp,
        OPJ_UINT32 tileno,
        opj_event_mgr_t* manager)
{
    /* loop*/
    OPJ_UINT32 pino, compno;
    /* number of poc in the p_pi*/
    OPJ_UINT32 l_poc_bound;

    /* pointers to tile coding parameters and components.*/
    opj_pi_iterator_t *l_pi = 00;
    opj_tcp_t *tcp = 00;
    const opj_tccp_t *tccp = 00;

    /* current packet iterator being allocated*/
    opj_pi_iterator_t *l_current_pi = 00;

    /* preconditions in debug*/
    assert(cp != 00);
    assert(image != 00);
    assert(tileno < cp->tw * cp->th);

    /* initializations*/
    tcp = &cp->tcps[tileno];
    l_poc_bound = tcp->numpocs + 1;

    /* memory allocations*/
    l_pi = (opj_pi_iterator_t*) opj_calloc((l_poc_bound),
                                           sizeof(opj_pi_iterator_t));
    if (!l_pi) {
        return NULL;
    }

    l_current_pi = l_pi;
    for (pino = 0; pino < l_poc_bound ; ++pino) {

        l_current_pi->manager = manager;

        l_current_pi->comps = (opj_pi_comp_t*) opj_calloc(image->numcomps,
                              sizeof(opj_pi_comp_t));
        if (! l_current_pi->comps) {
            opj_pi_destroy(l_pi, l_poc_bound);
            return NULL;
        }

        l_current_pi->numcomps = image->numcomps;

        for (compno = 0; compno < image->numcomps; ++compno) {
            opj_pi_comp_t *comp = &l_current_pi->comps[compno];

            tccp = &tcp->tccps[compno];

            comp->resolutions = (opj_pi_resolution_t*) opj_calloc(tccp->numresolutions,
                                sizeof(opj_pi_resolution_t));
            if (!comp->resolutions) {
                opj_pi_destroy(l_pi, l_poc_bound);
                return 00;
            }

            comp->numresolutions = tccp->numresolutions;
        }
        ++l_current_pi;
    }
    return l_pi;
}

static void opj_pi_update_encode_poc_and_final(opj_cp_t *p_cp,
        OPJ_UINT32 p_tileno,
        OPJ_UINT32 p_tx0,
        OPJ_UINT32 p_tx1,
        OPJ_UINT32 p_ty0,
        OPJ_UINT32 p_ty1,
        OPJ_UINT32 p_max_prec,
        OPJ_UINT32 p_max_res,
        OPJ_UINT32 p_dx_min,
        OPJ_UINT32 p_dy_min)
{
    /* loop*/
    OPJ_UINT32 pino;
    /* tile coding parameter*/
    opj_tcp_t *l_tcp = 00;
    /* current poc being updated*/
    opj_poc_t * l_current_poc = 00;

    /* number of pocs*/
    OPJ_UINT32 l_poc_bound;

    OPJ_ARG_NOT_USED(p_max_res);

    /* preconditions in debug*/
    assert(p_cp != 00);
    assert(p_tileno < p_cp->tw * p_cp->th);

    /* initializations*/
    l_tcp = &p_cp->tcps [p_tileno];
    /* number of iterations in the loop */
    l_poc_bound = l_tcp->numpocs + 1;

    /* start at first element, and to make sure the compiler will not make a calculation each time in the loop
       store a pointer to the current element to modify rather than l_tcp->pocs[i]*/
    l_current_poc = l_tcp->pocs;

    l_current_poc->compS = l_current_poc->compno0;
    l_current_poc->compE = l_current_poc->compno1;
    l_current_poc->resS = l_current_poc->resno0;
    l_current_poc->resE = l_current_poc->resno1;
    l_current_poc->layE = l_current_poc->layno1;

    /* special treatment for the first element*/
    l_current_poc->layS = 0;
    l_current_poc->prg  = l_current_poc->prg1;
    l_current_poc->prcS = 0;

    l_current_poc->prcE = p_max_prec;
    l_current_poc->txS = (OPJ_UINT32)p_tx0;
    l_current_poc->txE = (OPJ_UINT32)p_tx1;
    l_current_poc->tyS = (OPJ_UINT32)p_ty0;
    l_current_poc->tyE = (OPJ_UINT32)p_ty1;
    l_current_poc->dx = p_dx_min;
    l_current_poc->dy = p_dy_min;

    ++ l_current_poc;
    for (pino = 1; pino < l_poc_bound ; ++pino) {
        l_current_poc->compS = l_current_poc->compno0;
        l_current_poc->compE = l_current_poc->compno1;
        l_current_poc->resS = l_current_poc->resno0;
        l_current_poc->resE = l_current_poc->resno1;
        l_current_poc->layE = l_current_poc->layno1;
        l_current_poc->prg  = l_current_poc->prg1;
        l_current_poc->prcS = 0;
        /* special treatment here different from the first element*/
        l_current_poc->layS = (l_current_poc->layE > (l_current_poc - 1)->layE) ?
                              l_current_poc->layE : 0;

        l_current_poc->prcE = p_max_prec;
        l_current_poc->txS = (OPJ_UINT32)p_tx0;
        l_current_poc->txE = (OPJ_UINT32)p_tx1;
        l_current_poc->tyS = (OPJ_UINT32)p_ty0;
        l_current_poc->tyE = (OPJ_UINT32)p_ty1;
        l_current_poc->dx = p_dx_min;
        l_current_poc->dy = p_dy_min;
        ++ l_current_poc;
    }
}

static void opj_pi_update_encode_not_poc(opj_cp_t *p_cp,
        OPJ_UINT32 p_num_comps,
        OPJ_UINT32 p_tileno,
        OPJ_UINT32 p_tx0,
        OPJ_UINT32 p_tx1,
        OPJ_UINT32 p_ty0,
        OPJ_UINT32 p_ty1,
        OPJ_UINT32 p_max_prec,
        OPJ_UINT32 p_max_res,
        OPJ_UINT32 p_dx_min,
        OPJ_UINT32 p_dy_min)
{
    /* loop*/
    OPJ_UINT32 pino;
    /* tile coding parameter*/
    opj_tcp_t *l_tcp = 00;
    /* current poc being updated*/
    opj_poc_t * l_current_poc = 00;
    /* number of pocs*/
    OPJ_UINT32 l_poc_bound;

    /* preconditions in debug*/
    assert(p_cp != 00);
    assert(p_tileno < p_cp->tw * p_cp->th);

    /* initializations*/
    l_tcp = &p_cp->tcps [p_tileno];

    /* number of iterations in the loop */
    l_poc_bound = l_tcp->numpocs + 1;

    /* start at first element, and to make sure the compiler will not make a calculation each time in the loop
       store a pointer to the current element to modify rather than l_tcp->pocs[i]*/
    l_current_poc = l_tcp->pocs;

    for (pino = 0; pino < l_poc_bound ; ++pino) {
        l_current_poc->compS = 0;
        l_current_poc->compE = p_num_comps;/*p_image->numcomps;*/
        l_current_poc->resS = 0;
        l_current_poc->resE = p_max_res;
        l_current_poc->layS = 0;
        l_current_poc->layE = l_tcp->numlayers;
        l_current_poc->prg  = l_tcp->prg;
        l_current_poc->prcS = 0;
        l_current_poc->prcE = p_max_prec;
        l_current_poc->txS = p_tx0;
        l_current_poc->txE = p_tx1;
        l_current_poc->tyS = p_ty0;
        l_current_poc->tyE = p_ty1;
        l_current_poc->dx = p_dx_min;
        l_current_poc->dy = p_dy_min;
        ++ l_current_poc;
    }
}

static void opj_pi_update_decode_poc(opj_pi_iterator_t * p_pi,
                                     opj_tcp_t * p_tcp,
                                     OPJ_UINT32 p_max_precision,
                                     OPJ_UINT32 p_max_res)
{
    /* loop*/
    OPJ_UINT32 pino;

    /* encoding prameters to set*/
    OPJ_UINT32 l_bound;

    opj_pi_iterator_t * l_current_pi = 00;
    opj_poc_t* l_current_poc = 0;

    OPJ_ARG_NOT_USED(p_max_res);

    /* preconditions in debug*/
    assert(p_pi != 00);
    assert(p_tcp != 00);

    /* initializations*/
    l_bound = p_tcp->numpocs + 1;
    l_current_pi = p_pi;
    l_current_poc = p_tcp->pocs;

    for (pino = 0; pino < l_bound; ++pino) {
        l_current_pi->poc.prg = l_current_poc->prg; /* Progression Order #0 */
        l_current_pi->first = 1;

        l_current_pi->poc.resno0 =
            l_current_poc->resno0; /* Resolution Level Index #0 (Start) */
        l_current_pi->poc.compno0 =
            l_current_poc->compno0; /* Component Index #0 (Start) */
        l_current_pi->poc.layno0 = 0;
        l_current_pi->poc.precno0 = 0;
        l_current_pi->poc.resno1 =
            l_current_poc->resno1; /* Resolution Level Index #0 (End) */
        l_current_pi->poc.compno1 =
            l_current_poc->compno1; /* Component Index #0 (End) */
        l_current_pi->poc.layno1 = opj_uint_min(l_current_poc->layno1,
                                                p_tcp->numlayers); /* Layer Index #0 (End) */
        l_current_pi->poc.precno1 = p_max_precision;
        ++l_current_pi;
        ++l_current_poc;
    }
}

static void opj_pi_update_decode_not_poc(opj_pi_iterator_t * p_pi,
        opj_tcp_t * p_tcp,
        OPJ_UINT32 p_max_precision,
        OPJ_UINT32 p_max_res)
{
    /* loop*/
    OPJ_UINT32 pino;

    /* encoding prameters to set*/
    OPJ_UINT32 l_bound;

    opj_pi_iterator_t * l_current_pi = 00;
    /* preconditions in debug*/
    assert(p_tcp != 00);
    assert(p_pi != 00);

    /* initializations*/
    l_bound = p_tcp->numpocs + 1;
    l_current_pi = p_pi;

    for (pino = 0; pino < l_bound; ++pino) {
        l_current_pi->poc.prg = p_tcp->prg;
        l_current_pi->first = 1;
        l_current_pi->poc.resno0 = 0;
        l_current_pi->poc.compno0 = 0;
        l_current_pi->poc.layno0 = 0;
        l_current_pi->poc.precno0 = 0;
        l_current_pi->poc.resno1 = p_max_res;
        l_current_pi->poc.compno1 = l_current_pi->numcomps;
        l_current_pi->poc.layno1 = p_tcp->numlayers;
        l_current_pi->poc.precno1 = p_max_precision;
        ++l_current_pi;
    }
}



static OPJ_BOOL opj_pi_check_next_level(OPJ_INT32 pos,
                                        opj_cp_t *cp,
                                        OPJ_UINT32 tileno,
                                        OPJ_UINT32 pino,
                                        const OPJ_CHAR *prog)
{
    OPJ_INT32 i;
    opj_tcp_t *tcps = &cp->tcps[tileno];
    opj_poc_t *tcp = &tcps->pocs[pino];

    if (pos >= 0) {
        for (i = pos; pos >= 0; i--) {
            switch (prog[i]) {
            case 'R':
                if (tcp->res_t == tcp->resE) {
                    if (opj_pi_check_next_level(pos - 1, cp, tileno, pino, prog)) {
                        return OPJ_TRUE;
                    } else {
                        return OPJ_FALSE;
                    }
                } else {
                    return OPJ_TRUE;
                }
                break;
            case 'C':
                if (tcp->comp_t == tcp->compE) {
                    if (opj_pi_check_next_level(pos - 1, cp, tileno, pino, prog)) {
                        return OPJ_TRUE;
                    } else {
                        return OPJ_FALSE;
                    }
                } else {
                    return OPJ_TRUE;
                }
                break;
            case 'L':
                if (tcp->lay_t == tcp->layE) {
                    if (opj_pi_check_next_level(pos - 1, cp, tileno, pino, prog)) {
                        return OPJ_TRUE;
                    } else {
                        return OPJ_FALSE;
                    }
                } else {
                    return OPJ_TRUE;
                }
                break;
            case 'P':
                switch (tcp->prg) {
                case OPJ_LRCP: /* fall through */
                case OPJ_RLCP:
                    if (tcp->prc_t == tcp->prcE) {
                        if (opj_pi_check_next_level(i - 1, cp, tileno, pino, prog)) {
                            return OPJ_TRUE;
                        } else {
                            return OPJ_FALSE;
                        }
                    } else {
                        return OPJ_TRUE;
                    }
                    break;
                default:
                    if (tcp->tx0_t == tcp->txE) {
                        /*TY*/
                        if (tcp->ty0_t == tcp->tyE) {
                            if (opj_pi_check_next_level(i - 1, cp, tileno, pino, prog)) {
                                return OPJ_TRUE;
                            } else {
                                return OPJ_FALSE;
                            }
                        } else {
                            return OPJ_TRUE;
                        }/*TY*/
                    } else {
                        return OPJ_TRUE;
                    }
                    break;
                }/*end case P*/
            }/*end switch*/
        }/*end for*/
    }/*end if*/
    return OPJ_FALSE;
}


/*
==========================================================
   Packet iterator interface
==========================================================
*/
opj_pi_iterator_t *opj_pi_create_decode(opj_image_t *p_image,
                                        opj_cp_t *p_cp,
                                        OPJ_UINT32 p_tile_no,
                                        opj_event_mgr_t* manager)
{
    OPJ_UINT32 numcomps = p_image->numcomps;

    /* loop */
    OPJ_UINT32 pino;
    OPJ_UINT32 compno, resno;

    /* to store w, h, dx and dy fro all components and resolutions */
    OPJ_UINT32 * l_tmp_data;
    OPJ_UINT32 ** l_tmp_ptr;

    /* encoding prameters to set */
    OPJ_UINT32 l_max_res;
    OPJ_UINT32 l_max_prec;
    OPJ_UINT32 l_tx0, l_tx1, l_ty0, l_ty1;
    OPJ_UINT32 l_dx_min, l_dy_min;
    OPJ_UINT32 l_bound;
    OPJ_UINT32 l_step_p, l_step_c, l_step_r, l_step_l ;
    OPJ_UINT32 l_data_stride;

    /* pointers */
    opj_pi_iterator_t *l_pi = 00;
    opj_tcp_t *l_tcp = 00;
    const opj_tccp_t *l_tccp = 00;
    opj_pi_comp_t *l_current_comp = 00;
    opj_image_comp_t * l_img_comp = 00;
    opj_pi_iterator_t * l_current_pi = 00;
    OPJ_UINT32 * l_encoding_value_ptr = 00;

    /* preconditions in debug */
    assert(p_cp != 00);
    assert(p_image != 00);
    assert(p_tile_no < p_cp->tw * p_cp->th);

    /* initializations */
    l_tcp = &p_cp->tcps[p_tile_no];
    l_bound = l_tcp->numpocs + 1;

    l_data_stride = 4 * OPJ_J2K_MAXRLVLS;
    l_tmp_data = (OPJ_UINT32*)opj_malloc(
                     l_data_stride * numcomps * sizeof(OPJ_UINT32));
    if
    (! l_tmp_data) {
        return 00;
    }
    l_tmp_ptr = (OPJ_UINT32**)opj_malloc(
                    numcomps * sizeof(OPJ_UINT32 *));
    if
    (! l_tmp_ptr) {
        opj_free(l_tmp_data);
        return 00;
    }

    /* memory allocation for pi */
    l_pi = opj_pi_create(p_image, p_cp, p_tile_no, manager);
    if (!l_pi) {
        opj_free(l_tmp_data);
        opj_free(l_tmp_ptr);
        return 00;
    }

    l_encoding_value_ptr = l_tmp_data;
    /* update pointer array */
    for
    (compno = 0; compno < numcomps; ++compno) {
        l_tmp_ptr[compno] = l_encoding_value_ptr;
        l_encoding_value_ptr += l_data_stride;
    }
    /* get encoding parameters */
    opj_get_all_encoding_parameters(p_image, p_cp, p_tile_no, &l_tx0, &l_tx1,
                                    &l_ty0, &l_ty1, &l_dx_min, &l_dy_min, &l_max_prec, &l_max_res, l_tmp_ptr);

    /* step calculations */
    l_step_p = 1;
    l_step_c = l_max_prec * l_step_p;
    l_step_r = numcomps * l_step_c;
    l_step_l = l_max_res * l_step_r;

    /* set values for first packet iterator */
    l_current_pi = l_pi;

    /* memory allocation for include */
    /* prevent an integer overflow issue */
    /* 0 < l_tcp->numlayers < 65536 c.f. opj_j2k_read_cod in j2k.c */
    l_current_pi->include = 00;
    if (l_step_l <= (UINT_MAX / (l_tcp->numlayers + 1U))) {
        l_current_pi->include_size = (l_tcp->numlayers + 1U) * l_step_l;
        l_current_pi->include = (OPJ_INT16*) opj_calloc(
                                    l_current_pi->include_size, sizeof(OPJ_INT16));
    }

    if (!l_current_pi->include) {
        opj_free(l_tmp_data);
        opj_free(l_tmp_ptr);
        opj_pi_destroy(l_pi, l_bound);
        return 00;
    }

    /* special treatment for the first packet iterator */
    l_current_comp = l_current_pi->comps;
    l_img_comp = p_image->comps;
    l_tccp = l_tcp->tccps;

    l_current_pi->tx0 = l_tx0;
    l_current_pi->ty0 = l_ty0;
    l_current_pi->tx1 = l_tx1;
    l_current_pi->ty1 = l_ty1;

    /*l_current_pi->dx = l_img_comp->dx;*/
    /*l_current_pi->dy = l_img_comp->dy;*/

    l_current_pi->step_p = l_step_p;
    l_current_pi->step_c = l_step_c;
    l_current_pi->step_r = l_step_r;
    l_current_pi->step_l = l_step_l;

    /* allocation for components and number of components has already been calculated by opj_pi_create */
    for
    (compno = 0; compno < numcomps; ++compno) {
        opj_pi_resolution_t *l_res = l_current_comp->resolutions;
        l_encoding_value_ptr = l_tmp_ptr[compno];

        l_current_comp->dx = l_img_comp->dx;
        l_current_comp->dy = l_img_comp->dy;
        /* resolutions have already been initialized */
        for
        (resno = 0; resno < l_current_comp->numresolutions; resno++) {
            l_res->pdx = *(l_encoding_value_ptr++);
            l_res->pdy = *(l_encoding_value_ptr++);
            l_res->pw =  *(l_encoding_value_ptr++);
            l_res->ph =  *(l_encoding_value_ptr++);
            ++l_res;
        }
        ++l_current_comp;
        ++l_img_comp;
        ++l_tccp;
    }
    ++l_current_pi;

    for (pino = 1 ; pino < l_bound ; ++pino) {
        l_current_comp = l_current_pi->comps;
        l_img_comp = p_image->comps;
        l_tccp = l_tcp->tccps;

        l_current_pi->tx0 = l_tx0;
        l_current_pi->ty0 = l_ty0;
        l_current_pi->tx1 = l_tx1;
        l_current_pi->ty1 = l_ty1;
        /*l_current_pi->dx = l_dx_min;*/
        /*l_current_pi->dy = l_dy_min;*/
        l_current_pi->step_p = l_step_p;
        l_current_pi->step_c = l_step_c;
        l_current_pi->step_r = l_step_r;
        l_current_pi->step_l = l_step_l;

        /* allocation for components and number of components has already been calculated by opj_pi_create */
        for
        (compno = 0; compno < numcomps; ++compno) {
            opj_pi_resolution_t *l_res = l_current_comp->resolutions;
            l_encoding_value_ptr = l_tmp_ptr[compno];

            l_current_comp->dx = l_img_comp->dx;
            l_current_comp->dy = l_img_comp->dy;
            /* resolutions have already been initialized */
            for
            (resno = 0; resno < l_current_comp->numresolutions; resno++) {
                l_res->pdx = *(l_encoding_value_ptr++);
                l_res->pdy = *(l_encoding_value_ptr++);
                l_res->pw =  *(l_encoding_value_ptr++);
                l_res->ph =  *(l_encoding_value_ptr++);
                ++l_res;
            }
            ++l_current_comp;
            ++l_img_comp;
            ++l_tccp;
        }
        /* special treatment*/
        l_current_pi->include = (l_current_pi - 1)->include;
        l_current_pi->include_size = (l_current_pi - 1)->include_size;
        ++l_current_pi;
    }
    opj_free(l_tmp_data);
    l_tmp_data = 00;
    opj_free(l_tmp_ptr);
    l_tmp_ptr = 00;
    if
    (l_tcp->POC) {
        opj_pi_update_decode_poc(l_pi, l_tcp, l_max_prec, l_max_res);
    } else {
        opj_pi_update_decode_not_poc(l_pi, l_tcp, l_max_prec, l_max_res);
    }
    return l_pi;
}


OPJ_UINT32 opj_get_encoding_packet_count(const opj_image_t *p_image,
        const opj_cp_t *p_cp,
        OPJ_UINT32 p_tile_no)
{
    OPJ_UINT32 l_max_res;
    OPJ_UINT32 l_max_prec;
    OPJ_UINT32 l_tx0, l_tx1, l_ty0, l_ty1;
    OPJ_UINT32 l_dx_min, l_dy_min;

    /* preconditions in debug*/
    assert(p_cp != 00);
    assert(p_image != 00);
    assert(p_tile_no < p_cp->tw * p_cp->th);

    /* get encoding parameters*/
    opj_get_all_encoding_parameters(p_image, p_cp, p_tile_no, &l_tx0, &l_tx1,
                                    &l_ty0, &l_ty1, &l_dx_min, &l_dy_min, &l_max_prec, &l_max_res, NULL);

    return p_cp->tcps[p_tile_no].numlayers * l_max_prec * p_image->numcomps *
           l_max_res;
}


opj_pi_iterator_t *opj_pi_initialise_encode(const opj_image_t *p_image,
        opj_cp_t *p_cp,
        OPJ_UINT32 p_tile_no,
        J2K_T2_MODE p_t2_mode,
        opj_event_mgr_t* manager)
{
    OPJ_UINT32 numcomps = p_image->numcomps;

    /* loop*/
    OPJ_UINT32 pino;
    OPJ_UINT32 compno, resno;

    /* to store w, h, dx and dy fro all components and resolutions*/
    OPJ_UINT32 * l_tmp_data;
    OPJ_UINT32 ** l_tmp_ptr;

    /* encoding prameters to set*/
    OPJ_UINT32 l_max_res;
    OPJ_UINT32 l_max_prec;
    OPJ_UINT32 l_tx0, l_tx1, l_ty0, l_ty1;
    OPJ_UINT32 l_dx_min, l_dy_min;
    OPJ_UINT32 l_bound;
    OPJ_UINT32 l_step_p, l_step_c, l_step_r, l_step_l ;
    OPJ_UINT32 l_data_stride;

    /* pointers*/
    opj_pi_iterator_t *l_pi = 00;
    opj_tcp_t *l_tcp = 00;
    const opj_tccp_t *l_tccp = 00;
    opj_pi_comp_t *l_current_comp = 00;
    opj_image_comp_t * l_img_comp = 00;
    opj_pi_iterator_t * l_current_pi = 00;
    OPJ_UINT32 * l_encoding_value_ptr = 00;

    /* preconditions in debug*/
    assert(p_cp != 00);
    assert(p_image != 00);
    assert(p_tile_no < p_cp->tw * p_cp->th);

    /* initializations*/
    l_tcp = &p_cp->tcps[p_tile_no];
    l_bound = l_tcp->numpocs + 1;

    l_data_stride = 4 * OPJ_J2K_MAXRLVLS;
    l_tmp_data = (OPJ_UINT32*)opj_malloc(
                     l_data_stride * numcomps * sizeof(OPJ_UINT32));
    if (! l_tmp_data) {
        return 00;
    }

    l_tmp_ptr = (OPJ_UINT32**)opj_malloc(
                    numcomps * sizeof(OPJ_UINT32 *));
    if (! l_tmp_ptr) {
        opj_free(l_tmp_data);
        return 00;
    }

    /* memory allocation for pi*/
    l_pi = opj_pi_create(p_image, p_cp, p_tile_no, manager);
    if (!l_pi) {
        opj_free(l_tmp_data);
        opj_free(l_tmp_ptr);
        return 00;
    }

    l_encoding_value_ptr = l_tmp_data;
    /* update pointer array*/
    for (compno = 0; compno < numcomps; ++compno) {
        l_tmp_ptr[compno] = l_encoding_value_ptr;
        l_encoding_value_ptr += l_data_stride;
    }

    /* get encoding parameters*/
    opj_get_all_encoding_parameters(p_image, p_cp, p_tile_no, &l_tx0, &l_tx1,
                                    &l_ty0, &l_ty1, &l_dx_min, &l_dy_min, &l_max_prec, &l_max_res, l_tmp_ptr);

    /* step calculations*/
    l_step_p = 1;
    l_step_c = l_max_prec * l_step_p;
    l_step_r = numcomps * l_step_c;
    l_step_l = l_max_res * l_step_r;

    /* set values for first packet iterator*/
    l_pi->tp_on = (OPJ_BYTE)p_cp->m_specific_param.m_enc.m_tp_on;
    l_current_pi = l_pi;

    /* memory allocation for include*/
    l_current_pi->include_size = l_tcp->numlayers * l_step_l;
    l_current_pi->include = (OPJ_INT16*) opj_calloc(l_current_pi->include_size,
                            sizeof(OPJ_INT16));
    if (!l_current_pi->include) {
        opj_free(l_tmp_data);
        opj_free(l_tmp_ptr);
        opj_pi_destroy(l_pi, l_bound);
        return 00;
    }

    /* special treatment for the first packet iterator*/
    l_current_comp = l_current_pi->comps;
    l_img_comp = p_image->comps;
    l_tccp = l_tcp->tccps;
    l_current_pi->tx0 = l_tx0;
    l_current_pi->ty0 = l_ty0;
    l_current_pi->tx1 = l_tx1;
    l_current_pi->ty1 = l_ty1;
    l_current_pi->dx = l_dx_min;
    l_current_pi->dy = l_dy_min;
    l_current_pi->step_p = l_step_p;
    l_current_pi->step_c = l_step_c;
    l_current_pi->step_r = l_step_r;
    l_current_pi->step_l = l_step_l;

    /* allocation for components and number of components has already been calculated by opj_pi_create */
    for (compno = 0; compno < numcomps; ++compno) {
        opj_pi_resolution_t *l_res = l_current_comp->resolutions;
        l_encoding_value_ptr = l_tmp_ptr[compno];

        l_current_comp->dx = l_img_comp->dx;
        l_current_comp->dy = l_img_comp->dy;

        /* resolutions have already been initialized */
        for (resno = 0; resno < l_current_comp->numresolutions; resno++) {
            l_res->pdx = *(l_encoding_value_ptr++);
            l_res->pdy = *(l_encoding_value_ptr++);
            l_res->pw =  *(l_encoding_value_ptr++);
            l_res->ph =  *(l_encoding_value_ptr++);
            ++l_res;
        }

        ++l_current_comp;
        ++l_img_comp;
        ++l_tccp;
    }
    ++l_current_pi;

    for (pino = 1 ; pino < l_bound ; ++pino) {
        l_current_comp = l_current_pi->comps;
        l_img_comp = p_image->comps;
        l_tccp = l_tcp->tccps;

        l_current_pi->tx0 = l_tx0;
        l_current_pi->ty0 = l_ty0;
        l_current_pi->tx1 = l_tx1;
        l_current_pi->ty1 = l_ty1;
        l_current_pi->dx = l_dx_min;
        l_current_pi->dy = l_dy_min;
        l_current_pi->step_p = l_step_p;
        l_current_pi->step_c = l_step_c;
        l_current_pi->step_r = l_step_r;
        l_current_pi->step_l = l_step_l;

        /* allocation for components and number of components has already been calculated by opj_pi_create */
        for (compno = 0; compno < numcomps; ++compno) {
            opj_pi_resolution_t *l_res = l_current_comp->resolutions;
            l_encoding_value_ptr = l_tmp_ptr[compno];

            l_current_comp->dx = l_img_comp->dx;
            l_current_comp->dy = l_img_comp->dy;
            /* resolutions have already been initialized */
            for (resno = 0; resno < l_current_comp->numresolutions; resno++) {
                l_res->pdx = *(l_encoding_value_ptr++);
                l_res->pdy = *(l_encoding_value_ptr++);
                l_res->pw =  *(l_encoding_value_ptr++);
                l_res->ph =  *(l_encoding_value_ptr++);
                ++l_res;
            }
            ++l_current_comp;
            ++l_img_comp;
            ++l_tccp;
        }

        /* special treatment*/
        l_current_pi->include = (l_current_pi - 1)->include;
        l_current_pi->include_size = (l_current_pi - 1)->include_size;
        ++l_current_pi;
    }

    opj_free(l_tmp_data);
    l_tmp_data = 00;
    opj_free(l_tmp_ptr);
    l_tmp_ptr = 00;

    if (l_tcp->POC && (OPJ_IS_CINEMA(p_cp->rsiz) || p_t2_mode == FINAL_PASS)) {
        opj_pi_update_encode_poc_and_final(p_cp, p_tile_no, l_tx0, l_tx1, l_ty0, l_ty1,
                                           l_max_prec, l_max_res, l_dx_min, l_dy_min);
    } else {
        opj_pi_update_encode_not_poc(p_cp, numcomps, p_tile_no, l_tx0, l_tx1,
                                     l_ty0, l_ty1, l_max_prec, l_max_res, l_dx_min, l_dy_min);
    }

    return l_pi;
}

void opj_pi_create_encode(opj_pi_iterator_t *pi,
                          opj_cp_t *cp,
                          OPJ_UINT32 tileno,
                          OPJ_UINT32 pino,
                          OPJ_UINT32 tpnum,
                          OPJ_INT32 tppos,
                          J2K_T2_MODE t2_mode)
{
    const OPJ_CHAR *prog;
    OPJ_INT32 i;
    OPJ_UINT32 incr_top = 1, resetX = 0;
    opj_tcp_t *tcps = &cp->tcps[tileno];
    opj_poc_t *tcp = &tcps->pocs[pino];

    prog = opj_j2k_convert_progression_order(tcp->prg);

    pi[pino].first = 1;
    pi[pino].poc.prg = tcp->prg;

    if (!(cp->m_specific_param.m_enc.m_tp_on && ((!OPJ_IS_CINEMA(cp->rsiz) &&
            !OPJ_IS_IMF(cp->rsiz) &&
            (t2_mode == FINAL_PASS)) || OPJ_IS_CINEMA(cp->rsiz) || OPJ_IS_IMF(cp->rsiz)))) {
        pi[pino].poc.resno0 = tcp->resS;
        pi[pino].poc.resno1 = tcp->resE;
        pi[pino].poc.compno0 = tcp->compS;
        pi[pino].poc.compno1 = tcp->compE;
        pi[pino].poc.layno0 = tcp->layS;
        pi[pino].poc.layno1 = tcp->layE;
        pi[pino].poc.precno0 = tcp->prcS;
        pi[pino].poc.precno1 = tcp->prcE;
        pi[pino].poc.tx0 = tcp->txS;
        pi[pino].poc.ty0 = tcp->tyS;
        pi[pino].poc.tx1 = tcp->txE;
        pi[pino].poc.ty1 = tcp->tyE;
    } else {
        for (i = tppos + 1; i < 4; i++) {
            switch (prog[i]) {
            case 'R':
                pi[pino].poc.resno0 = tcp->resS;
                pi[pino].poc.resno1 = tcp->resE;
                break;
            case 'C':
                pi[pino].poc.compno0 = tcp->compS;
                pi[pino].poc.compno1 = tcp->compE;
                break;
            case 'L':
                pi[pino].poc.layno0 = tcp->layS;
                pi[pino].poc.layno1 = tcp->layE;
                break;
            case 'P':
                switch (tcp->prg) {
                case OPJ_LRCP:
                case OPJ_RLCP:
                    pi[pino].poc.precno0 = tcp->prcS;
                    pi[pino].poc.precno1 = tcp->prcE;
                    break;
                default:
                    pi[pino].poc.tx0 = tcp->txS;
                    pi[pino].poc.ty0 = tcp->tyS;
                    pi[pino].poc.tx1 = tcp->txE;
                    pi[pino].poc.ty1 = tcp->tyE;
                    break;
                }
                break;
            }
        }

        if (tpnum == 0) {
            for (i = tppos; i >= 0; i--) {
                switch (prog[i]) {
                case 'C':
                    tcp->comp_t = tcp->compS;
                    pi[pino].poc.compno0 = tcp->comp_t;
                    pi[pino].poc.compno1 = tcp->comp_t + 1;
                    tcp->comp_t += 1;
                    break;
                case 'R':
                    tcp->res_t = tcp->resS;
                    pi[pino].poc.resno0 = tcp->res_t;
                    pi[pino].poc.resno1 = tcp->res_t + 1;
                    tcp->res_t += 1;
                    break;
                case 'L':
                    tcp->lay_t = tcp->layS;
                    pi[pino].poc.layno0 = tcp->lay_t;
                    pi[pino].poc.layno1 = tcp->lay_t + 1;
                    tcp->lay_t += 1;
                    break;
                case 'P':
                    switch (tcp->prg) {
                    case OPJ_LRCP:
                    case OPJ_RLCP:
                        tcp->prc_t = tcp->prcS;
                        pi[pino].poc.precno0 = tcp->prc_t;
                        pi[pino].poc.precno1 = tcp->prc_t + 1;
                        tcp->prc_t += 1;
                        break;
                    default:
                        tcp->tx0_t = tcp->txS;
                        tcp->ty0_t = tcp->tyS;
                        pi[pino].poc.tx0 = tcp->tx0_t;
                        pi[pino].poc.tx1 = tcp->tx0_t + tcp->dx - (tcp->tx0_t % tcp->dx);
                        pi[pino].poc.ty0 = tcp->ty0_t;
                        pi[pino].poc.ty1 = tcp->ty0_t + tcp->dy - (tcp->ty0_t % tcp->dy);
                        tcp->tx0_t = (OPJ_UINT32)pi[pino].poc.tx1;
                        tcp->ty0_t = (OPJ_UINT32)pi[pino].poc.ty1;
                        break;
                    }
                    break;
                }
            }
            incr_top = 1;
        } else {
            for (i = tppos; i >= 0; i--) {
                switch (prog[i]) {
                case 'C':
                    pi[pino].poc.compno0 = tcp->comp_t - 1;
                    pi[pino].poc.compno1 = tcp->comp_t;
                    break;
                case 'R':
                    pi[pino].poc.resno0 = tcp->res_t - 1;
                    pi[pino].poc.resno1 = tcp->res_t;
                    break;
                case 'L':
                    pi[pino].poc.layno0 = tcp->lay_t - 1;
                    pi[pino].poc.layno1 = tcp->lay_t;
                    break;
                case 'P':
                    switch (tcp->prg) {
                    case OPJ_LRCP:
                    case OPJ_RLCP:
                        pi[pino].poc.precno0 = tcp->prc_t - 1;
                        pi[pino].poc.precno1 = tcp->prc_t;
                        break;
                    default:
                        pi[pino].poc.tx0 = tcp->tx0_t - tcp->dx - (tcp->tx0_t % tcp->dx);
                        pi[pino].poc.tx1 = tcp->tx0_t ;
                        pi[pino].poc.ty0 = tcp->ty0_t - tcp->dy - (tcp->ty0_t % tcp->dy);
                        pi[pino].poc.ty1 = tcp->ty0_t ;
                        break;
                    }
                    break;
                }
                if (incr_top == 1) {
                    switch (prog[i]) {
                    case 'R':
                        if (tcp->res_t == tcp->resE) {
                            if (opj_pi_check_next_level(i - 1, cp, tileno, pino, prog)) {
                                tcp->res_t = tcp->resS;
                                pi[pino].poc.resno0 = tcp->res_t;
                                pi[pino].poc.resno1 = tcp->res_t + 1;
                                tcp->res_t += 1;
                                incr_top = 1;
                            } else {
                                incr_top = 0;
                            }
                        } else {
                            pi[pino].poc.resno0 = tcp->res_t;
                            pi[pino].poc.resno1 = tcp->res_t + 1;
                            tcp->res_t += 1;
                            incr_top = 0;
                        }
                        break;
                    case 'C':
                        if (tcp->comp_t == tcp->compE) {
                            if (opj_pi_check_next_level(i - 1, cp, tileno, pino, prog)) {
                                tcp->comp_t = tcp->compS;
                                pi[pino].poc.compno0 = tcp->comp_t;
                                pi[pino].poc.compno1 = tcp->comp_t + 1;
                                tcp->comp_t += 1;
                                incr_top = 1;
                            } else {
                                incr_top = 0;
                            }
                        } else {
                            pi[pino].poc.compno0 = tcp->comp_t;
                            pi[pino].poc.compno1 = tcp->comp_t + 1;
                            tcp->comp_t += 1;
                            incr_top = 0;
                        }
                        break;
                    case 'L':
                        if (tcp->lay_t == tcp->layE) {
                            if (opj_pi_check_next_level(i - 1, cp, tileno, pino, prog)) {
                                tcp->lay_t = tcp->layS;
                                pi[pino].poc.layno0 = tcp->lay_t;
                                pi[pino].poc.layno1 = tcp->lay_t + 1;
                                tcp->lay_t += 1;
                                incr_top = 1;
                            } else {
                                incr_top = 0;
                            }
                        } else {
                            pi[pino].poc.layno0 = tcp->lay_t;
                            pi[pino].poc.layno1 = tcp->lay_t + 1;
                            tcp->lay_t += 1;
                            incr_top = 0;
                        }
                        break;
                    case 'P':
                        switch (tcp->prg) {
                        case OPJ_LRCP:
                        case OPJ_RLCP:
                            if (tcp->prc_t == tcp->prcE) {
                                if (opj_pi_check_next_level(i - 1, cp, tileno, pino, prog)) {
                                    tcp->prc_t = tcp->prcS;
                                    pi[pino].poc.precno0 = tcp->prc_t;
                                    pi[pino].poc.precno1 = tcp->prc_t + 1;
                                    tcp->prc_t += 1;
                                    incr_top = 1;
                                } else {
                                    incr_top = 0;
                                }
                            } else {
                                pi[pino].poc.precno0 = tcp->prc_t;
                                pi[pino].poc.precno1 = tcp->prc_t + 1;
                                tcp->prc_t += 1;
                                incr_top = 0;
                            }
                            break;
                        default:
                            if (tcp->tx0_t >= tcp->txE) {
                                if (tcp->ty0_t >= tcp->tyE) {
                                    if (opj_pi_check_next_level(i - 1, cp, tileno, pino, prog)) {
                                        tcp->ty0_t = tcp->tyS;
                                        pi[pino].poc.ty0 = tcp->ty0_t;
                                        pi[pino].poc.ty1 = tcp->ty0_t + tcp->dy - (tcp->ty0_t % tcp->dy);
                                        tcp->ty0_t = (OPJ_UINT32)pi[pino].poc.ty1;
                                        incr_top = 1;
                                        resetX = 1;
                                    } else {
                                        incr_top = 0;
                                        resetX = 0;
                                    }
                                } else {
                                    pi[pino].poc.ty0 = tcp->ty0_t;
                                    pi[pino].poc.ty1 = tcp->ty0_t + tcp->dy - (tcp->ty0_t % tcp->dy);
                                    tcp->ty0_t = (OPJ_UINT32)pi[pino].poc.ty1;
                                    incr_top = 0;
                                    resetX = 1;
                                }
                                if (resetX == 1) {
                                    tcp->tx0_t = tcp->txS;
                                    pi[pino].poc.tx0 = tcp->tx0_t;
                                    pi[pino].poc.tx1 = tcp->tx0_t + tcp->dx - (tcp->tx0_t % tcp->dx);
                                    tcp->tx0_t = (OPJ_UINT32)pi[pino].poc.tx1;
                                }
                            } else {
                                pi[pino].poc.tx0 = tcp->tx0_t;
                                pi[pino].poc.tx1 = tcp->tx0_t + tcp->dx - (tcp->tx0_t % tcp->dx);
                                tcp->tx0_t = (OPJ_UINT32)pi[pino].poc.tx1;
                                incr_top = 0;
                            }
                            break;
                        }
                        break;
                    }
                }
            }
        }
    }
}

void opj_pi_destroy(opj_pi_iterator_t *p_pi,
                    OPJ_UINT32 p_nb_elements)
{
    OPJ_UINT32 compno, pino;
    opj_pi_iterator_t *l_current_pi = p_pi;
    if (p_pi) {
        if (p_pi->include) {
            opj_free(p_pi->include);
            p_pi->include = 00;
        }
        for (pino = 0; pino < p_nb_elements; ++pino) {
            if (l_current_pi->comps) {
                opj_pi_comp_t *l_current_component = l_current_pi->comps;
                for (compno = 0; compno < l_current_pi->numcomps; compno++) {
                    if (l_current_component->resolutions) {
                        opj_free(l_current_component->resolutions);
                        l_current_component->resolutions = 00;
                    }

                    ++l_current_component;
                }
                opj_free(l_current_pi->comps);
                l_current_pi->comps = 0;
            }
            ++l_current_pi;
        }
        opj_free(p_pi);
    }
}



void opj_pi_update_encoding_parameters(const opj_image_t *p_image,
                                       opj_cp_t *p_cp,
                                       OPJ_UINT32 p_tile_no)
{
    /* encoding parameters to set */
    OPJ_UINT32 l_max_res;
    OPJ_UINT32 l_max_prec;
    OPJ_UINT32 l_tx0, l_tx1, l_ty0, l_ty1;
    OPJ_UINT32 l_dx_min, l_dy_min;

    /* pointers */
    opj_tcp_t *l_tcp = 00;

    /* preconditions */
    assert(p_cp != 00);
    assert(p_image != 00);
    assert(p_tile_no < p_cp->tw * p_cp->th);

    l_tcp = &(p_cp->tcps[p_tile_no]);

    /* get encoding parameters */
    opj_get_encoding_parameters(p_image, p_cp, p_tile_no, &l_tx0, &l_tx1, &l_ty0,
                                &l_ty1, &l_dx_min, &l_dy_min, &l_max_prec, &l_max_res);

    if (l_tcp->POC) {
        opj_pi_update_encode_poc_and_final(p_cp, p_tile_no, l_tx0, l_tx1, l_ty0, l_ty1,
                                           l_max_prec, l_max_res, l_dx_min, l_dy_min);
    } else {
        opj_pi_update_encode_not_poc(p_cp, p_image->numcomps, p_tile_no, l_tx0, l_tx1,
                                     l_ty0, l_ty1, l_max_prec, l_max_res, l_dx_min, l_dy_min);
    }
}

OPJ_BOOL opj_pi_next(opj_pi_iterator_t * pi)
{
    switch (pi->poc.prg) {
    case OPJ_LRCP:
        return opj_pi_next_lrcp(pi);
    case OPJ_RLCP:
        return opj_pi_next_rlcp(pi);
    case OPJ_RPCL:
        return opj_pi_next_rpcl(pi);
    case OPJ_PCRL:
        return opj_pi_next_pcrl(pi);
    case OPJ_CPRL:
        return opj_pi_next_cprl(pi);
    case OPJ_PROG_UNKNOWN:
        return OPJ_FALSE;
    }

    return OPJ_FALSE;
}
