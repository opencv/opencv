/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2006-2007, Parvatha Elangovan
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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

#include "pi.h"
#include "int.h"
#include "opj_malloc.h"
#include "j2k.h"
/** @defgroup PI PI - Implementation of a packet iterator */
/*@{*/

/** @name Local static functions */
/*@{*/

/**
Get next packet in layer-resolution-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static bool pi_next_lrcp(opj_pi_iterator_t * pi);
/**
Get next packet in resolution-layer-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static bool pi_next_rlcp(opj_pi_iterator_t * pi);
/**
Get next packet in resolution-precinct-component-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static bool pi_next_rpcl(opj_pi_iterator_t * pi);
/**
Get next packet in precinct-component-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static bool pi_next_pcrl(opj_pi_iterator_t * pi);
/**
Get next packet in component-precinct-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
static bool pi_next_cprl(opj_pi_iterator_t * pi);

/**
 * Updates the coding parameters if the encoding is used with Progression order changes and final (or cinema parameters are used).
 *
 * @param  p_cp    the coding parameters to modify
 * @param  p_tileno  the tile index being concerned.
 * @param  p_tx0    X0 parameter for the tile
 * @param  p_tx1    X1 parameter for the tile
 * @param  p_ty0    Y0 parameter for the tile
 * @param  p_ty1    Y1 parameter for the tile
 * @param  p_max_prec  the maximum precision for all the bands of the tile
 * @param  p_max_res  the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min    the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min    the minimum dy of all the components of all the resolutions for the tile.
 */
void pi_update_encode_poc_and_final (
                   opj_cp_t *p_cp,
                   OPJ_UINT32 p_tileno,
                   OPJ_INT32 p_tx0,
                   OPJ_INT32 p_tx1,
                   OPJ_INT32 p_ty0,
                   OPJ_INT32 p_ty1,
                   OPJ_UINT32 p_max_prec,
                   OPJ_UINT32 p_max_res,
                                     OPJ_UINT32 p_dx_min,
                   OPJ_UINT32 p_dy_min);

/**
 * Updates the coding parameters if the encoding is not used with Progression order changes and final (and cinema parameters are used).
 *
 * @param  p_cp    the coding parameters to modify
 * @param  p_tileno  the tile index being concerned.
 * @param  p_tx0    X0 parameter for the tile
 * @param  p_tx1    X1 parameter for the tile
 * @param  p_ty0    Y0 parameter for the tile
 * @param  p_ty1    Y1 parameter for the tile
 * @param  p_max_prec  the maximum precision for all the bands of the tile
 * @param  p_max_res  the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min    the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min    the minimum dy of all the components of all the resolutions for the tile.
 */
void pi_update_encode_not_poc (
                opj_cp_t *p_cp,
                OPJ_UINT32 p_num_comps,
                OPJ_UINT32 p_tileno,
                OPJ_INT32 p_tx0,
                OPJ_INT32 p_tx1,
                OPJ_INT32 p_ty0,
                OPJ_INT32 p_ty1,
                OPJ_UINT32 p_max_prec,
                OPJ_UINT32 p_max_res,
                                OPJ_UINT32 p_dx_min,
                OPJ_UINT32 p_dy_min);

/**
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 *
 * @param  p_image      the image being encoded.
 * @param  p_cp      the coding parameters.
 * @param  tileno      the tile index of the tile being encoded.
 * @param  p_tx0      pointer that will hold the X0 parameter for the tile
 * @param  p_tx1      pointer that will hold the X1 parameter for the tile
 * @param  p_ty0      pointer that will hold the Y0 parameter for the tile
 * @param  p_ty1      pointer that will hold the Y1 parameter for the tile
 * @param  p_max_prec    pointer that will hold the the maximum precision for all the bands of the tile
 * @param  p_max_res    pointer that will hold the the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min      pointer that will hold the the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min      pointer that will hold the the minimum dy of all the components of all the resolutions for the tile.
 */
void get_encoding_parameters(
                const opj_image_t *p_image,
                const opj_cp_t *p_cp,
                OPJ_UINT32  tileno,
                OPJ_INT32  * p_tx0,
                OPJ_INT32 * p_tx1,
                OPJ_INT32 * p_ty0,
                OPJ_INT32 * p_ty1,
                OPJ_UINT32 * p_dx_min,
                OPJ_UINT32 * p_dy_min,
                OPJ_UINT32 * p_max_prec,
                OPJ_UINT32 * p_max_res
              );

/**
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 * The precinct widths, heights, dx and dy for each component at each resolution will be stored as well.
 * the last parameter of the function should be an array of pointers of size nb components, each pointer leading
 * to an area of size 4 * max_res. The data is stored inside this area with the following pattern :
 * dx_compi_res0 , dy_compi_res0 , w_compi_res0, h_compi_res0 , dx_compi_res1 , dy_compi_res1 , w_compi_res1, h_compi_res1 , ...
 *
 * @param  p_image      the image being encoded.
 * @param  p_cp      the coding parameters.
 * @param  tileno      the tile index of the tile being encoded.
 * @param  p_tx0      pointer that will hold the X0 parameter for the tile
 * @param  p_tx1      pointer that will hold the X1 parameter for the tile
 * @param  p_ty0      pointer that will hold the Y0 parameter for the tile
 * @param  p_ty1      pointer that will hold the Y1 parameter for the tile
 * @param  p_max_prec    pointer that will hold the the maximum precision for all the bands of the tile
 * @param  p_max_res    pointer that will hold the the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min      pointer that will hold the the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min      pointer that will hold the the minimum dy of all the components of all the resolutions for the tile.
 * @param  p_resolutions  pointer to an area corresponding to the one described above.
 */
void get_all_encoding_parameters(
                const opj_image_t *p_image,
                const opj_cp_t *p_cp,
                OPJ_UINT32 tileno,
                OPJ_INT32 * p_tx0,
                OPJ_INT32 * p_tx1,
                OPJ_INT32 * p_ty0,
                OPJ_INT32 * p_ty1,
                OPJ_UINT32 * p_dx_min,
                OPJ_UINT32 * p_dy_min,
                OPJ_UINT32 * p_max_prec,
                OPJ_UINT32 * p_max_res,
                OPJ_UINT32 ** p_resolutions
              );
/**
 * Allocates memory for a packet iterator. Data and data sizes are set by this operation.
 * No other data is set. The include section of the packet  iterator is not allocated.
 *
 * @param  p_image    the image used to initialize the packet iterator (in fact only the number of components is relevant.
 * @param  p_cp    the coding parameters.
 * @param  p_tile_no  the index of the tile from which creating the packet iterator.
 */
opj_pi_iterator_t * pi_create(
                const opj_image_t *image,
                const opj_cp_t *cp,
                OPJ_UINT32 tileno
              );
void pi_update_decode_not_poc (opj_pi_iterator_t * p_pi,opj_tcp_t * p_tcp,OPJ_UINT32 p_max_precision,OPJ_UINT32 p_max_res);
void pi_update_decode_poc (opj_pi_iterator_t * p_pi,opj_tcp_t * p_tcp,OPJ_UINT32 p_max_precision,OPJ_UINT32 p_max_res);


/*@}*/

/*@}*/

/*
==========================================================
   local functions
==========================================================
*/

static bool pi_next_lrcp(opj_pi_iterator_t * pi) {
  opj_pi_comp_t *comp = 00;
  opj_pi_resolution_t *res = 00;
  OPJ_UINT32 index = 0;

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
        if (!pi->tp_on){
          pi->poc.precno1 = res->pw * res->ph;
        }
        for (pi->precno = pi->poc.precno0; pi->precno < pi->poc.precno1; pi->precno++) {
          index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
          if (!pi->include[index]) {
            pi->include[index] = 1;
            return true;
          }
LABEL_SKIP:;
        }
      }
    }
  }

  return false;
}

static bool pi_next_rlcp(opj_pi_iterator_t * pi) {
  opj_pi_comp_t *comp = 00;
  opj_pi_resolution_t *res = 00;
  OPJ_UINT32 index = 0;

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
        if(!pi->tp_on){
          pi->poc.precno1 = res->pw * res->ph;
        }
        for (pi->precno = pi->poc.precno0; pi->precno < pi->poc.precno1; pi->precno++) {
          index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
          if (!pi->include[index]) {
            pi->include[index] = 1;
            return true;
          }
LABEL_SKIP:;
        }
      }
    }
  }

  return false;
}

static bool pi_next_rpcl(opj_pi_iterator_t * pi) {
  opj_pi_comp_t *comp = 00;
  opj_pi_resolution_t *res = 00;
  OPJ_UINT32 index = 0;

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
        dx = comp->dx * (1 << (res->pdx + comp->numresolutions - 1 - resno));
        dy = comp->dy * (1 << (res->pdy + comp->numresolutions - 1 - resno));
        pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
        pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
      }
    }
  }
if (!pi->tp_on){
      pi->poc.ty0 = pi->ty0;
      pi->poc.tx0 = pi->tx0;
      pi->poc.ty1 = pi->ty1;
      pi->poc.tx1 = pi->tx1;
    }
  for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1; pi->resno++) {
    for (pi->y = pi->poc.ty0; pi->y < pi->poc.ty1; pi->y += pi->dy - (pi->y % pi->dy)) {
      for (pi->x = pi->poc.tx0; pi->x < pi->poc.tx1; pi->x += pi->dx - (pi->x % pi->dx)) {
        for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
          OPJ_UINT32 levelno;
          OPJ_INT32 trx0, try0;
          OPJ_INT32  trx1, try1;
          OPJ_UINT32  rpx, rpy;
          OPJ_INT32  prci, prcj;
          comp = &pi->comps[pi->compno];
          if (pi->resno >= comp->numresolutions) {
            continue;
          }
          res = &comp->resolutions[pi->resno];
          levelno = comp->numresolutions - 1 - pi->resno;
          trx0 = int_ceildiv(pi->tx0, comp->dx << levelno);
          try0 = int_ceildiv(pi->ty0, comp->dy << levelno);
          trx1 = int_ceildiv(pi->tx1, comp->dx << levelno);
          try1 = int_ceildiv(pi->ty1, comp->dy << levelno);
          rpx = res->pdx + levelno;
          rpy = res->pdy + levelno;
          if (!((pi->y % (comp->dy << rpy) == 0) || ((pi->y == pi->ty0) && ((try0 << levelno) % (1 << rpy))))){
            continue;
          }
          if (!((pi->x % (comp->dx << rpx) == 0) || ((pi->x == pi->tx0) && ((trx0 << levelno) % (1 << rpx))))){
            continue;
          }

          if ((res->pw==0)||(res->ph==0)) continue;

          if ((trx0==trx1)||(try0==try1)) continue;

          prci = int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelno), res->pdx)
             - int_floordivpow2(trx0, res->pdx);
          prcj = int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelno), res->pdy)
             - int_floordivpow2(try0, res->pdy);
          pi->precno = prci + prcj * res->pw;
          for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
            index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
            if (!pi->include[index]) {
              pi->include[index] = 1;
              return true;
            }
LABEL_SKIP:;
          }
        }
      }
    }
  }

  return false;
}

static bool pi_next_pcrl(opj_pi_iterator_t * pi) {
  opj_pi_comp_t *comp = 00;
  opj_pi_resolution_t *res = 00;
  OPJ_UINT32 index = 0;

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
        dx = comp->dx * (1 << (res->pdx + comp->numresolutions - 1 - resno));
        dy = comp->dy * (1 << (res->pdy + comp->numresolutions - 1 - resno));
        pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
        pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
      }
    }
  }
  if (!pi->tp_on){
      pi->poc.ty0 = pi->ty0;
      pi->poc.tx0 = pi->tx0;
      pi->poc.ty1 = pi->ty1;
      pi->poc.tx1 = pi->tx1;
    }
  for (pi->y = pi->poc.ty0; pi->y < pi->poc.ty1; pi->y += pi->dy - (pi->y % pi->dy)) {
    for (pi->x = pi->poc.tx0; pi->x < pi->poc.tx1; pi->x += pi->dx - (pi->x % pi->dx)) {
      for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
        comp = &pi->comps[pi->compno];
        // TODO
        for (pi->resno = pi->poc.resno0; pi->resno < uint_min(pi->poc.resno1, comp->numresolutions); pi->resno++) {
          OPJ_UINT32 levelno;
          OPJ_INT32 trx0, try0;
          OPJ_INT32 trx1, try1;
          OPJ_UINT32 rpx, rpy;
          OPJ_INT32 prci, prcj;
          res = &comp->resolutions[pi->resno];
          levelno = comp->numresolutions - 1 - pi->resno;
          trx0 = int_ceildiv(pi->tx0, comp->dx << levelno);
          try0 = int_ceildiv(pi->ty0, comp->dy << levelno);
          trx1 = int_ceildiv(pi->tx1, comp->dx << levelno);
          try1 = int_ceildiv(pi->ty1, comp->dy << levelno);
          rpx = res->pdx + levelno;
          rpy = res->pdy + levelno;
          if (!((pi->y % (comp->dy << rpy) == 0) || ((pi->y == pi->ty0) && ((try0 << levelno) % (1 << rpy))))){
            continue;
          }
          if (!((pi->x % (comp->dx << rpx) == 0) || ((pi->x == pi->tx0) && ((trx0 << levelno) % (1 << rpx))))){
            continue;
          }

          if ((res->pw==0)||(res->ph==0)) continue;

          if ((trx0==trx1)||(try0==try1)) continue;

          prci = int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelno), res->pdx)
             - int_floordivpow2(trx0, res->pdx);
          prcj = int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelno), res->pdy)
             - int_floordivpow2(try0, res->pdy);
          pi->precno = prci + prcj * res->pw;
          for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
            index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
            if (!pi->include[index]) {
              pi->include[index] = 1;
              return true;
            }
LABEL_SKIP:;
          }
        }
      }
    }
  }

  return false;
}

static bool pi_next_cprl(opj_pi_iterator_t * pi) {
  opj_pi_comp_t *comp = 00;
  opj_pi_resolution_t *res = 00;
  OPJ_UINT32 index = 0;

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
      dx = comp->dx * (1 << (res->pdx + comp->numresolutions - 1 - resno));
      dy = comp->dy * (1 << (res->pdy + comp->numresolutions - 1 - resno));
      pi->dx = !pi->dx ? dx : int_min(pi->dx, dx);
      pi->dy = !pi->dy ? dy : int_min(pi->dy, dy);
    }
    if (!pi->tp_on){
      pi->poc.ty0 = pi->ty0;
      pi->poc.tx0 = pi->tx0;
      pi->poc.ty1 = pi->ty1;
      pi->poc.tx1 = pi->tx1;
    }
    for (pi->y = pi->poc.ty0; pi->y < pi->poc.ty1; pi->y += pi->dy - (pi->y % pi->dy)) {
      for (pi->x = pi->poc.tx0; pi->x < pi->poc.tx1; pi->x += pi->dx - (pi->x % pi->dx)) {
        // TODO
        for (pi->resno = pi->poc.resno0; pi->resno < uint_min(pi->poc.resno1, comp->numresolutions); pi->resno++) {
          OPJ_UINT32 levelno;
          OPJ_INT32 trx0, try0;
          OPJ_INT32 trx1, try1;
          OPJ_UINT32 rpx, rpy;
          OPJ_INT32 prci, prcj;
          res = &comp->resolutions[pi->resno];
          levelno = comp->numresolutions - 1 - pi->resno;
          trx0 = int_ceildiv(pi->tx0, comp->dx << levelno);
          try0 = int_ceildiv(pi->ty0, comp->dy << levelno);
          trx1 = int_ceildiv(pi->tx1, comp->dx << levelno);
          try1 = int_ceildiv(pi->ty1, comp->dy << levelno);
          rpx = res->pdx + levelno;
          rpy = res->pdy + levelno;
          if (!((pi->y % (comp->dy << rpy) == 0) || ((pi->y == pi->ty0) && ((try0 << levelno) % (1 << rpy))))){
            continue;
          }
          if (!((pi->x % (comp->dx << rpx) == 0) || ((pi->x == pi->tx0) && ((trx0 << levelno) % (1 << rpx))))){
            continue;
          }

          if ((res->pw==0)||(res->ph==0)) continue;

          if ((trx0==trx1)||(try0==try1)) continue;

          prci = int_floordivpow2(int_ceildiv(pi->x, comp->dx << levelno), res->pdx)
             - int_floordivpow2(trx0, res->pdx);
          prcj = int_floordivpow2(int_ceildiv(pi->y, comp->dy << levelno), res->pdy)
             - int_floordivpow2(try0, res->pdy);
          pi->precno = prci + prcj * res->pw;
          for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
            index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno * pi->step_c + pi->precno * pi->step_p;
            if (!pi->include[index]) {
              pi->include[index] = 1;
              return true;
            }
LABEL_SKIP:;
          }
        }
      }
    }
  }

  return false;
}

/*
==========================================================
   Packet iterator interface
==========================================================
*/
opj_pi_iterator_t *pi_create_decode(
                    opj_image_t *p_image,
                    opj_cp_t *p_cp,
                    OPJ_UINT32 p_tile_no
                    )
{
  // loop
  OPJ_UINT32 pino;
  OPJ_UINT32 compno, resno;

  // to store w, h, dx and dy fro all components and resolutions
  OPJ_UINT32 * l_tmp_data;
  OPJ_UINT32 ** l_tmp_ptr;

  // encoding prameters to set
  OPJ_UINT32 l_max_res;
  OPJ_UINT32 l_max_prec;
  OPJ_INT32 l_tx0,l_tx1,l_ty0,l_ty1;
  OPJ_UINT32 l_dx_min,l_dy_min;
  OPJ_UINT32 l_bound;
  OPJ_UINT32 l_step_p , l_step_c , l_step_r , l_step_l ;
  OPJ_UINT32 l_data_stride;

  // pointers
  opj_pi_iterator_t *l_pi = 00;
  opj_tcp_t *l_tcp = 00;
  const opj_tccp_t *l_tccp = 00;
  opj_pi_comp_t *l_current_comp = 00;
  opj_image_comp_t * l_img_comp = 00;
  opj_pi_iterator_t * l_current_pi = 00;
  OPJ_UINT32 * l_encoding_value_ptr = 00;

  // preconditions in debug
  assert(p_cp != 00);
  assert(p_image != 00);
  assert(p_tile_no < p_cp->tw * p_cp->th);

  // initializations
  l_tcp = &p_cp->tcps[p_tile_no];
  l_bound = l_tcp->numpocs+1;

  l_data_stride = 4 * J2K_MAXRLVLS;
  l_tmp_data = (OPJ_UINT32*)opj_malloc(
    l_data_stride * p_image->numcomps * sizeof(OPJ_UINT32));
  if
    (! l_tmp_data)
  {
    return 00;
  }
  l_tmp_ptr = (OPJ_UINT32**)opj_malloc(
    p_image->numcomps * sizeof(OPJ_UINT32 *));
  if
    (! l_tmp_ptr)
  {
    opj_free(l_tmp_data);
    return 00;
  }

  // memory allocation for pi
  l_pi = pi_create(p_image,p_cp,p_tile_no);
  if
    (!l_pi)
  {
    opj_free(l_tmp_data);
    opj_free(l_tmp_ptr);
    return 00;
  }

  l_encoding_value_ptr = l_tmp_data;
  // update pointer array
  for
    (compno = 0; compno < p_image->numcomps; ++compno)
  {
    l_tmp_ptr[compno] = l_encoding_value_ptr;
    l_encoding_value_ptr += l_data_stride;
  }
  // get encoding parameters
  get_all_encoding_parameters(p_image,p_cp,p_tile_no,&l_tx0,&l_tx1,&l_ty0,&l_ty1,&l_dx_min,&l_dy_min,&l_max_prec,&l_max_res,l_tmp_ptr);

  // step calculations
  l_step_p = 1;
  l_step_c = l_max_prec * l_step_p;
  l_step_r = p_image->numcomps * l_step_c;
  l_step_l = l_max_res * l_step_r;

  // set values for first packet iterator
  l_current_pi = l_pi;

  // memory allocation for include
  l_current_pi->include = (OPJ_INT16*) opj_calloc(l_tcp->numlayers * l_step_l, sizeof(OPJ_INT16));
  if
    (!l_current_pi->include)
  {
    opj_free(l_tmp_data);
    opj_free(l_tmp_ptr);
    pi_destroy(l_pi, l_bound);
    return 00;
  }
  memset(l_current_pi->include,0,l_tcp->numlayers * l_step_l* sizeof(OPJ_INT16));

  // special treatment for the first packet iterator
  l_current_comp = l_current_pi->comps;
  l_img_comp = p_image->comps;
  l_tccp = l_tcp->tccps;

  l_current_pi->tx0 = l_tx0;
  l_current_pi->ty0 = l_ty0;
  l_current_pi->tx1 = l_tx1;
  l_current_pi->ty1 = l_ty1;

  //l_current_pi->dx = l_img_comp->dx;
  //l_current_pi->dy = l_img_comp->dy;

  l_current_pi->step_p = l_step_p;
  l_current_pi->step_c = l_step_c;
  l_current_pi->step_r = l_step_r;
  l_current_pi->step_l = l_step_l;

  /* allocation for components and number of components has already been calculated by pi_create */
  for
    (compno = 0; compno < l_current_pi->numcomps; ++compno)
  {
    opj_pi_resolution_t *l_res = l_current_comp->resolutions;
    l_encoding_value_ptr = l_tmp_ptr[compno];

    l_current_comp->dx = l_img_comp->dx;
    l_current_comp->dy = l_img_comp->dy;
    /* resolutions have already been initialized */
    for
      (resno = 0; resno < l_current_comp->numresolutions; resno++)
    {
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

  for
    (pino = 1 ; pino<l_bound ; ++pino )
  {
    opj_pi_comp_t *l_current_comp = l_current_pi->comps;
    opj_image_comp_t * l_img_comp = p_image->comps;
    l_tccp = l_tcp->tccps;

    l_current_pi->tx0 = l_tx0;
    l_current_pi->ty0 = l_ty0;
    l_current_pi->tx1 = l_tx1;
    l_current_pi->ty1 = l_ty1;
    //l_current_pi->dx = l_dx_min;
    //l_current_pi->dy = l_dy_min;
    l_current_pi->step_p = l_step_p;
    l_current_pi->step_c = l_step_c;
    l_current_pi->step_r = l_step_r;
    l_current_pi->step_l = l_step_l;

    /* allocation for components and number of components has already been calculated by pi_create */
    for
      (compno = 0; compno < l_current_pi->numcomps; ++compno)
    {
      opj_pi_resolution_t *l_res = l_current_comp->resolutions;
      l_encoding_value_ptr = l_tmp_ptr[compno];

      l_current_comp->dx = l_img_comp->dx;
      l_current_comp->dy = l_img_comp->dy;
      /* resolutions have already been initialized */
      for
        (resno = 0; resno < l_current_comp->numresolutions; resno++)
      {
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
    // special treatment
    l_current_pi->include = (l_current_pi-1)->include;
    ++l_current_pi;
  }
  opj_free(l_tmp_data);
  l_tmp_data = 00;
  opj_free(l_tmp_ptr);
  l_tmp_ptr = 00;
  if
    (l_tcp->POC)
  {
    pi_update_decode_poc (l_pi,l_tcp,l_max_prec,l_max_res);
  }
  else
  {
    pi_update_decode_not_poc(l_pi,l_tcp,l_max_prec,l_max_res);
  }
  return l_pi;
}

void pi_update_decode_poc (opj_pi_iterator_t * p_pi,opj_tcp_t * p_tcp,OPJ_UINT32 p_max_precision,OPJ_UINT32 p_max_res)
{
  // loop
  OPJ_UINT32 pino;

  // encoding prameters to set
  OPJ_UINT32 l_bound;

  opj_pi_iterator_t * l_current_pi = 00;
  opj_poc_t* l_current_poc = 0;

  // preconditions in debug
  assert(p_pi != 00);
  assert(p_tcp != 00);

  // initializations
  l_bound = p_tcp->numpocs+1;
  l_current_pi = p_pi;
  l_current_poc = p_tcp->pocs;

  for
    (pino = 0;pino<l_bound;++pino)
  {
    l_current_pi->poc.prg = l_current_poc->prg;
    l_current_pi->first = 1;

    l_current_pi->poc.resno0 = l_current_poc->resno0;
    l_current_pi->poc.compno0 = l_current_poc->compno0;
    l_current_pi->poc.layno0 = 0;
    l_current_pi->poc.precno0 = 0;
    l_current_pi->poc.resno1 = l_current_poc->resno1;
    l_current_pi->poc.compno1 = l_current_poc->compno1;
    l_current_pi->poc.layno1 = l_current_poc->layno1;
    l_current_pi->poc.precno1 = p_max_precision;
    ++l_current_pi;
    ++l_current_poc;
  }
}

void pi_update_decode_not_poc (opj_pi_iterator_t * p_pi,opj_tcp_t * p_tcp,OPJ_UINT32 p_max_precision,OPJ_UINT32 p_max_res)
{
  // loop
  OPJ_UINT32 pino;

  // encoding prameters to set
  OPJ_UINT32 l_bound;

  opj_pi_iterator_t * l_current_pi = 00;
  // preconditions in debug
  assert(p_tcp != 00);
  assert(p_pi != 00);

  // initializations
  l_bound = p_tcp->numpocs+1;
  l_current_pi = p_pi;

  for
    (pino = 0;pino<l_bound;++pino)
  {
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

/**
 * Creates a packet iterator for encoding.
 *
 * @param  p_image    the image being encoded.
 * @param  p_cp    the coding parameters.
 * @param  p_tile_no  index of the tile being encoded.
 * @param  p_t2_mode  the type of pass for generating the packet iterator
 * @return  a list of packet iterator that points to the first packet of the tile (not true).
*/
opj_pi_iterator_t *pi_initialise_encode(
                    const opj_image_t *p_image,
                    opj_cp_t *p_cp,
                    OPJ_UINT32 p_tile_no,
                    J2K_T2_MODE p_t2_mode
                    )
{
  // loop
  OPJ_UINT32 pino;
  OPJ_UINT32 compno, resno;

  // to store w, h, dx and dy fro all components and resolutions
  OPJ_UINT32 * l_tmp_data;
  OPJ_UINT32 ** l_tmp_ptr;

  // encoding prameters to set
  OPJ_UINT32 l_max_res;
  OPJ_UINT32 l_max_prec;
  OPJ_INT32 l_tx0,l_tx1,l_ty0,l_ty1;
  OPJ_UINT32 l_dx_min,l_dy_min;
  OPJ_UINT32 l_bound;
  OPJ_UINT32 l_step_p , l_step_c , l_step_r , l_step_l ;
  OPJ_UINT32 l_data_stride;

  // pointers
  opj_pi_iterator_t *l_pi = 00;
  opj_tcp_t *l_tcp = 00;
  const opj_tccp_t *l_tccp = 00;
  opj_pi_comp_t *l_current_comp = 00;
  opj_image_comp_t * l_img_comp = 00;
  opj_pi_iterator_t * l_current_pi = 00;
  OPJ_UINT32 * l_encoding_value_ptr = 00;

  // preconditions in debug
  assert(p_cp != 00);
  assert(p_image != 00);
  assert(p_tile_no < p_cp->tw * p_cp->th);

  // initializations
  l_tcp = &p_cp->tcps[p_tile_no];
  l_bound = l_tcp->numpocs+1;

  l_data_stride = 4 * J2K_MAXRLVLS;
  l_tmp_data = (OPJ_UINT32*)opj_malloc(
    l_data_stride * p_image->numcomps * sizeof(OPJ_UINT32));
  if
    (! l_tmp_data)
  {
    return 00;
  }
  l_tmp_ptr = (OPJ_UINT32**)opj_malloc(
    p_image->numcomps * sizeof(OPJ_UINT32 *));
  if
    (! l_tmp_ptr)
  {
    opj_free(l_tmp_data);
    return 00;
  }

  // memory allocation for pi
  l_pi = pi_create(p_image,p_cp,p_tile_no);
  if
    (!l_pi)
  {
    opj_free(l_tmp_data);
    opj_free(l_tmp_ptr);
    return 00;
  }

  l_encoding_value_ptr = l_tmp_data;
  // update pointer array
  for
    (compno = 0; compno < p_image->numcomps; ++compno)
  {
    l_tmp_ptr[compno] = l_encoding_value_ptr;
    l_encoding_value_ptr += l_data_stride;
  }
  // get encoding parameters
  get_all_encoding_parameters(p_image,p_cp,p_tile_no,&l_tx0,&l_tx1,&l_ty0,&l_ty1,&l_dx_min,&l_dy_min,&l_max_prec,&l_max_res,l_tmp_ptr);

  // step calculations
  l_step_p = 1;
  l_step_c = l_max_prec * l_step_p;
  l_step_r = p_image->numcomps * l_step_c;
  l_step_l = l_max_res * l_step_r;

  // set values for first packet iterator
  l_pi->tp_on = p_cp->m_specific_param.m_enc.m_tp_on;
  l_current_pi = l_pi;

  // memory allocation for include
  l_current_pi->include = (OPJ_INT16*) opj_calloc(l_tcp->numlayers * l_step_l, sizeof(OPJ_INT16));
  if
    (!l_current_pi->include)
  {
    opj_free(l_tmp_data);
    opj_free(l_tmp_ptr);
    pi_destroy(l_pi, l_bound);
    return 00;
  }
  memset(l_current_pi->include,0,l_tcp->numlayers * l_step_l* sizeof(OPJ_INT16));

  // special treatment for the first packet iterator
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

  /* allocation for components and number of components has already been calculated by pi_create */
  for
    (compno = 0; compno < l_current_pi->numcomps; ++compno)
  {
    opj_pi_resolution_t *l_res = l_current_comp->resolutions;
    l_encoding_value_ptr = l_tmp_ptr[compno];

    l_current_comp->dx = l_img_comp->dx;
    l_current_comp->dy = l_img_comp->dy;
    /* resolutions have already been initialized */
    for
      (resno = 0; resno < l_current_comp->numresolutions; resno++)
    {
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

  for
    (pino = 1 ; pino<l_bound ; ++pino )
  {
    opj_pi_comp_t *l_current_comp = l_current_pi->comps;
    opj_image_comp_t * l_img_comp = p_image->comps;
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

    /* allocation for components and number of components has already been calculated by pi_create */
    for
      (compno = 0; compno < l_current_pi->numcomps; ++compno)
    {
      opj_pi_resolution_t *l_res = l_current_comp->resolutions;
      l_encoding_value_ptr = l_tmp_ptr[compno];

      l_current_comp->dx = l_img_comp->dx;
      l_current_comp->dy = l_img_comp->dy;
      /* resolutions have already been initialized */
      for
        (resno = 0; resno < l_current_comp->numresolutions; resno++)
      {
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
    // special treatment
    l_current_pi->include = (l_current_pi-1)->include;
    ++l_current_pi;
  }
  opj_free(l_tmp_data);
  l_tmp_data = 00;
  opj_free(l_tmp_ptr);
  l_tmp_ptr = 00;
  if
    (l_tcp->POC && ( p_cp->m_specific_param.m_enc.m_cinema || p_t2_mode == FINAL_PASS))
  {
    pi_update_encode_poc_and_final(p_cp,p_tile_no,l_tx0,l_tx1,l_ty0,l_ty1,l_max_prec,l_max_res,l_dx_min,l_dy_min);
  }
  else
  {
    pi_update_encode_not_poc(p_cp,p_image->numcomps,p_tile_no,l_tx0,l_tx1,l_ty0,l_ty1,l_max_prec,l_max_res,l_dx_min,l_dy_min);
  }
  return l_pi;
}

/**
 * Updates the encoding parameters of the codec.
 *
 * @param  p_image    the image being encoded.
 * @param  p_cp    the coding parameters.
 * @param  p_tile_no  index of the tile being encoded.
*/
void pi_update_encoding_parameters(
                    const opj_image_t *p_image,
                    opj_cp_t *p_cp,
                    OPJ_UINT32 p_tile_no
                    )
{
  // encoding prameters to set
  OPJ_UINT32 l_max_res;
  OPJ_UINT32 l_max_prec;
  OPJ_INT32 l_tx0,l_tx1,l_ty0,l_ty1;
  OPJ_UINT32 l_dx_min,l_dy_min;

  // pointers
  opj_tcp_t *l_tcp = 00;

  // preconditions in debug
  assert(p_cp != 00);
  assert(p_image != 00);
  assert(p_tile_no < p_cp->tw * p_cp->th);

  l_tcp = &(p_cp->tcps[p_tile_no]);
  // get encoding parameters
  get_encoding_parameters(p_image,p_cp,p_tile_no,&l_tx0,&l_tx1,&l_ty0,&l_ty1,&l_dx_min,&l_dy_min,&l_max_prec,&l_max_res);
  if
    (l_tcp->POC)
  {
    pi_update_encode_poc_and_final(p_cp,p_tile_no,l_tx0,l_tx1,l_ty0,l_ty1,l_max_prec,l_max_res,l_dx_min,l_dy_min);
  }
  else
  {
    pi_update_encode_not_poc(p_cp,p_image->numcomps,p_tile_no,l_tx0,l_tx1,l_ty0,l_ty1,l_max_prec,l_max_res,l_dx_min,l_dy_min);
  }
}


/**
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 *
 * @param  p_image      the image being encoded.
 * @param  p_cp      the coding parameters.
 * @param  p_tileno      the tile index of the tile being encoded.
 * @param  p_tx0      pointer that will hold the X0 parameter for the tile
 * @param  p_tx1      pointer that will hold the X1 parameter for the tile
 * @param  p_ty0      pointer that will hold the Y0 parameter for the tile
 * @param  p_ty1      pointer that will hold the Y1 parameter for the tile
 * @param  p_max_prec    pointer that will hold the the maximum precision for all the bands of the tile
 * @param  p_max_res    pointer that will hold the the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min      pointer that will hold the the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min      pointer that will hold the the minimum dy of all the components of all the resolutions for the tile.
 */
void get_encoding_parameters(
                const opj_image_t *p_image,
                const opj_cp_t *p_cp,
                OPJ_UINT32 p_tileno,
                OPJ_INT32 * p_tx0,
                OPJ_INT32  * p_tx1,
                OPJ_INT32  * p_ty0,
                OPJ_INT32  * p_ty1,
                OPJ_UINT32 * p_dx_min,
                OPJ_UINT32 * p_dy_min,
                OPJ_UINT32 * p_max_prec,
                OPJ_UINT32 * p_max_res
              )
{
  // loop
  OPJ_UINT32  compno, resno;
  // pointers
  const opj_tcp_t *l_tcp = 00;
  const opj_tccp_t * l_tccp = 00;
  const opj_image_comp_t * l_img_comp = 00;

  // position in x and y of tile
  OPJ_UINT32 p, q;

  // preconditions in debug
  assert(p_cp != 00);
  assert(p_image != 00);
  assert(p_tileno < p_cp->tw * p_cp->th);

  // initializations
  l_tcp = &p_cp->tcps [p_tileno];
  l_img_comp = p_image->comps;
  l_tccp = l_tcp->tccps;

  /* here calculation of tx0, tx1, ty0, ty1, maxprec, dx and dy */
  p = p_tileno % p_cp->tw;
  q = p_tileno / p_cp->tw;

  // find extent of tile
  *p_tx0 = int_max(p_cp->tx0 + p * p_cp->tdx, p_image->x0);
  *p_tx1 = int_min(p_cp->tx0 + (p + 1) * p_cp->tdx, p_image->x1);
  *p_ty0 = int_max(p_cp->ty0 + q * p_cp->tdy, p_image->y0);
  *p_ty1 = int_min(p_cp->ty0 + (q + 1) * p_cp->tdy, p_image->y1);

  // max precision is 0 (can only grow)
  *p_max_prec = 0;
  *p_max_res = 0;

  // take the largest value for dx_min and dy_min
  *p_dx_min = 0x7fffffff;
  *p_dy_min  = 0x7fffffff;

  for
    (compno = 0; compno < p_image->numcomps; ++compno)
  {
    // aritmetic variables to calculate
    OPJ_UINT32 l_level_no;
    OPJ_INT32 l_rx0, l_ry0, l_rx1, l_ry1;
    OPJ_INT32 l_px0, l_py0, l_px1, py1;
    OPJ_UINT32 l_pdx, l_pdy;
    OPJ_UINT32 l_pw, l_ph;
    OPJ_UINT32 l_product;
    OPJ_INT32 l_tcx0, l_tcy0, l_tcx1, l_tcy1;

    l_tcx0 = int_ceildiv(*p_tx0, l_img_comp->dx);
    l_tcy0 = int_ceildiv(*p_ty0, l_img_comp->dy);
    l_tcx1 = int_ceildiv(*p_tx1, l_img_comp->dx);
    l_tcy1 = int_ceildiv(*p_ty1, l_img_comp->dy);
    if
      (l_tccp->numresolutions > *p_max_res)
    {
      *p_max_res = l_tccp->numresolutions;
    }
    // use custom size for precincts
    for
      (resno = 0; resno < l_tccp->numresolutions; ++resno)
    {
      OPJ_UINT32 l_dx, l_dy;
      // precinct width and height
      l_pdx = l_tccp->prcw[resno];
      l_pdy = l_tccp->prch[resno];

      l_dx = l_img_comp->dx * (1 << (l_pdx + l_tccp->numresolutions - 1 - resno));
      l_dy = l_img_comp->dy * (1 << (l_pdy + l_tccp->numresolutions - 1 - resno));
      // take the minimum size for dx for each comp and resolution
      *p_dx_min = uint_min(*p_dx_min, l_dx);
      *p_dy_min = uint_min(*p_dy_min, l_dy);
      // various calculations of extents
      l_level_no = l_tccp->numresolutions - 1 - resno;
      l_rx0 = int_ceildivpow2(l_tcx0, l_level_no);
      l_ry0 = int_ceildivpow2(l_tcy0, l_level_no);
      l_rx1 = int_ceildivpow2(l_tcx1, l_level_no);
      l_ry1 = int_ceildivpow2(l_tcy1, l_level_no);
      l_px0 = int_floordivpow2(l_rx0, l_pdx) << l_pdx;
      l_py0 = int_floordivpow2(l_ry0, l_pdy) << l_pdy;
      l_px1 = int_ceildivpow2(l_rx1, l_pdx) << l_pdx;
      py1 = int_ceildivpow2(l_ry1, l_pdy) << l_pdy;
      l_pw = (l_rx0==l_rx1)?0:((l_px1 - l_px0) >> l_pdx);
      l_ph = (l_ry0==l_ry1)?0:((py1 - l_py0) >> l_pdy);
      l_product = l_pw * l_ph;
      // update precision
      if
        (l_product > *p_max_prec)
      {
        *p_max_prec = l_product;
      }
    }
    ++l_img_comp;
    ++l_tccp;
  }
}

/**
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 * The precinct widths, heights, dx and dy for each component at each resolution will be stored as well.
 * the last parameter of the function should be an array of pointers of size nb components, each pointer leading
 * to an area of size 4 * max_res. The data is stored inside this area with the following pattern :
 * dx_compi_res0 , dy_compi_res0 , w_compi_res0, h_compi_res0 , dx_compi_res1 , dy_compi_res1 , w_compi_res1, h_compi_res1 , ...
 *
 * @param  p_image      the image being encoded.
 * @param  p_cp      the coding parameters.
 * @param  tileno      the tile index of the tile being encoded.
 * @param  p_tx0      pointer that will hold the X0 parameter for the tile
 * @param  p_tx1      pointer that will hold the X1 parameter for the tile
 * @param  p_ty0      pointer that will hold the Y0 parameter for the tile
 * @param  p_ty1      pointer that will hold the Y1 parameter for the tile
 * @param  p_max_prec    pointer that will hold the the maximum precision for all the bands of the tile
 * @param  p_max_res    pointer that will hold the the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min      pointer that will hold the the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min      pointer that will hold the the minimum dy of all the components of all the resolutions for the tile.
 * @param  p_resolutions  pointer to an area corresponding to the one described above.
 */
void get_all_encoding_parameters(
                const opj_image_t *p_image,
                const opj_cp_t *p_cp,
                OPJ_UINT32 tileno,
                OPJ_INT32 * p_tx0,
                OPJ_INT32 * p_tx1,
                OPJ_INT32 * p_ty0,
                OPJ_INT32 * p_ty1,
                OPJ_UINT32 * p_dx_min,
                OPJ_UINT32 * p_dy_min,
                OPJ_UINT32 * p_max_prec,
                OPJ_UINT32 * p_max_res,
                OPJ_UINT32 ** p_resolutions
              )
{
  // loop
  OPJ_UINT32 compno, resno;

  // pointers
  const opj_tcp_t *tcp = 00;
  const opj_tccp_t * l_tccp = 00;
  const opj_image_comp_t * l_img_comp = 00;

  // to store l_dx, l_dy, w and h for each resolution and component.
  OPJ_UINT32 * lResolutionPtr;

  // position in x and y of tile
  OPJ_UINT32 p, q;

  // preconditions in debug
  assert(p_cp != 00);
  assert(p_image != 00);
  assert(tileno < p_cp->tw * p_cp->th);

  // initializations
  tcp = &p_cp->tcps [tileno];
  l_tccp = tcp->tccps;
  l_img_comp = p_image->comps;

  // position in x and y of tile

  p = tileno % p_cp->tw;
  q = tileno / p_cp->tw;

  /* here calculation of tx0, tx1, ty0, ty1, maxprec, l_dx and l_dy */
  *p_tx0 = int_max(p_cp->tx0 + p * p_cp->tdx, p_image->x0);
  *p_tx1 = int_min(p_cp->tx0 + (p + 1) * p_cp->tdx, p_image->x1);
  *p_ty0 = int_max(p_cp->ty0 + q * p_cp->tdy, p_image->y0);
  *p_ty1 = int_min(p_cp->ty0 + (q + 1) * p_cp->tdy, p_image->y1);

  // max precision and resolution is 0 (can only grow)
  *p_max_prec = 0;
  *p_max_res = 0;

  // take the largest value for dx_min and dy_min
  *p_dx_min = 0x7fffffff;
  *p_dy_min  = 0x7fffffff;

  for
    (compno = 0; compno < p_image->numcomps; ++compno)
  {
    // aritmetic variables to calculate
    OPJ_UINT32 l_level_no;
    OPJ_INT32 l_rx0, l_ry0, l_rx1, l_ry1;
    OPJ_INT32 l_px0, l_py0, l_px1, py1;
    OPJ_UINT32 l_product;
    OPJ_INT32 l_tcx0, l_tcy0, l_tcx1, l_tcy1;
    OPJ_UINT32 l_pdx, l_pdy , l_pw , l_ph;

    lResolutionPtr = p_resolutions[compno];

    l_tcx0 = int_ceildiv(*p_tx0, l_img_comp->dx);
    l_tcy0 = int_ceildiv(*p_ty0, l_img_comp->dy);
    l_tcx1 = int_ceildiv(*p_tx1, l_img_comp->dx);
    l_tcy1 = int_ceildiv(*p_ty1, l_img_comp->dy);
    if
      (l_tccp->numresolutions > *p_max_res)
    {
      *p_max_res = l_tccp->numresolutions;
    }

    // use custom size for precincts
    l_level_no = l_tccp->numresolutions - 1;
    for
      (resno = 0; resno < l_tccp->numresolutions; ++resno)
    {
      OPJ_UINT32 l_dx, l_dy;
      // precinct width and height
      l_pdx = l_tccp->prcw[resno];
      l_pdy = l_tccp->prch[resno];
      *lResolutionPtr++ = l_pdx;
      *lResolutionPtr++ = l_pdy;
      l_dx = l_img_comp->dx * (1 << (l_pdx + l_level_no));
      l_dy = l_img_comp->dy * (1 << (l_pdy + l_level_no));
      // take the minimum size for l_dx for each comp and resolution
      *p_dx_min = int_min(*p_dx_min, l_dx);
      *p_dy_min = int_min(*p_dy_min, l_dy);
      // various calculations of extents

      l_rx0 = int_ceildivpow2(l_tcx0, l_level_no);
      l_ry0 = int_ceildivpow2(l_tcy0, l_level_no);
      l_rx1 = int_ceildivpow2(l_tcx1, l_level_no);
      l_ry1 = int_ceildivpow2(l_tcy1, l_level_no);
      l_px0 = int_floordivpow2(l_rx0, l_pdx) << l_pdx;
      l_py0 = int_floordivpow2(l_ry0, l_pdy) << l_pdy;
      l_px1 = int_ceildivpow2(l_rx1, l_pdx) << l_pdx;
      py1 = int_ceildivpow2(l_ry1, l_pdy) << l_pdy;
      l_pw = (l_rx0==l_rx1)?0:((l_px1 - l_px0) >> l_pdx);
      l_ph = (l_ry0==l_ry1)?0:((py1 - l_py0) >> l_pdy);
      *lResolutionPtr++ = l_pw;
      *lResolutionPtr++ = l_ph;
      l_product = l_pw * l_ph;
      // update precision
      if
        (l_product > *p_max_prec)
      {
        *p_max_prec = l_product;
      }
      --l_level_no;
    }
    ++l_tccp;
    ++l_img_comp;
  }
}

/**
 * Allocates memory for a packet iterator. Data and data sizes are set by this operation.
 * No other data is set. The include section of the packet  iterator is not allocated.
 *
 * @param  p_image    the image used to initialize the packet iterator (in fact only the number of components is relevant.
 * @param  p_cp    the coding parameters.
 * @param  p_tile_no  the index of the tile from which creating the packet iterator.
 */
opj_pi_iterator_t * pi_create(
                const opj_image_t *image,
                const opj_cp_t *cp,
                OPJ_UINT32 tileno
              )
{
  // loop
  OPJ_UINT32 pino, compno;
  // number of poc in the p_pi
  OPJ_UINT32 l_poc_bound;

  // pointers to tile coding parameters and components.
  opj_pi_iterator_t *l_pi = 00;
  opj_tcp_t *tcp = 00;
  const opj_tccp_t *tccp = 00;

  // current packet iterator being allocated
  opj_pi_iterator_t *l_current_pi = 00;

  // preconditions in debug
  assert(cp != 00);
  assert(image != 00);
  assert(tileno < cp->tw * cp->th);

  // initializations
  tcp = &cp->tcps[tileno];
  l_poc_bound = tcp->numpocs+1;


  // memory allocations
  l_pi = (opj_pi_iterator_t*) opj_calloc((l_poc_bound), sizeof(opj_pi_iterator_t));

  if
    (!l_pi)
  {
    return 00;
  }
  memset(l_pi,0,l_poc_bound * sizeof(opj_pi_iterator_t));
  l_current_pi = l_pi;
  for
    (pino = 0; pino < l_poc_bound ; ++pino)
  {
    l_current_pi->comps = (opj_pi_comp_t*) opj_calloc(image->numcomps, sizeof(opj_pi_comp_t));
    if
      (! l_current_pi->comps)
    {
      pi_destroy(l_pi, l_poc_bound);
      return 00;
    }
    l_current_pi->numcomps = image->numcomps;
    memset(l_current_pi->comps,0,image->numcomps * sizeof(opj_pi_comp_t));
    for
      (compno = 0; compno < image->numcomps; ++compno)
    {
      opj_pi_comp_t *comp = &l_current_pi->comps[compno];
      tccp = &tcp->tccps[compno];
      comp->resolutions = (opj_pi_resolution_t*) opj_malloc(tccp->numresolutions * sizeof(opj_pi_resolution_t));
      if
        (!comp->resolutions)
      {
        pi_destroy(l_pi, l_poc_bound);
        return 00;
      }
      comp->numresolutions = tccp->numresolutions;
      memset(comp->resolutions,0,tccp->numresolutions * sizeof(opj_pi_resolution_t));
    }
    ++l_current_pi;
  }
  return l_pi;
}

/**
 * Updates the coding parameters if the encoding is used with Progression order changes and final (or cinema parameters are used).
 *
 * @param  p_cp    the coding parameters to modify
 * @param  p_tileno  the tile index being concerned.
 * @param  p_tx0    X0 parameter for the tile
 * @param  p_tx1    X1 parameter for the tile
 * @param  p_ty0    Y0 parameter for the tile
 * @param  p_ty1    Y1 parameter for the tile
 * @param  p_max_prec  the maximum precision for all the bands of the tile
 * @param  p_max_res  the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min    the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min    the minimum dy of all the components of all the resolutions for the tile.
 */
void pi_update_encode_poc_and_final (
                   opj_cp_t *p_cp,
                   OPJ_UINT32 p_tileno,
                   OPJ_INT32 p_tx0,
                   OPJ_INT32 p_tx1,
                   OPJ_INT32 p_ty0,
                   OPJ_INT32 p_ty1,
                   OPJ_UINT32 p_max_prec,
                   OPJ_UINT32 p_max_res,
                                     OPJ_UINT32 p_dx_min,
                   OPJ_UINT32 p_dy_min)
{
  // loop
  OPJ_UINT32 pino;
  // tile coding parameter
  opj_tcp_t *l_tcp = 00;
  // current poc being updated
  opj_poc_t * l_current_poc = 00;

  // number of pocs
  OPJ_UINT32 l_poc_bound;

  // preconditions in debug
  assert(p_cp != 00);
  assert(p_tileno < p_cp->tw * p_cp->th);

  // initializations
  l_tcp = &p_cp->tcps [p_tileno];
  /* number of iterations in the loop */
  l_poc_bound = l_tcp->numpocs+1;

  // start at first element, and to make sure the compiler will not make a calculation each time in the loop
  // store a pointer to the current element to modify rather than l_tcp->pocs[i]
  l_current_poc = l_tcp->pocs;

  l_current_poc->compS = l_current_poc->compno0;
  l_current_poc->compE = l_current_poc->compno1;
  l_current_poc->resS = l_current_poc->resno0;
  l_current_poc->resE = l_current_poc->resno1;
  l_current_poc->layE = l_current_poc->layno1;

  // special treatment for the first element
  l_current_poc->layS = 0;
  l_current_poc->prg  = l_current_poc->prg1;
  l_current_poc->prcS = 0;

  l_current_poc->prcE = p_max_prec;
  l_current_poc->txS = p_tx0;
  l_current_poc->txE = p_tx1;
  l_current_poc->tyS = p_ty0;
  l_current_poc->tyE = p_ty1;
  l_current_poc->dx = p_dx_min;
  l_current_poc->dy = p_dy_min;

  ++ l_current_poc;
  for
    (pino = 1;pino < l_poc_bound ; ++pino)
  {
    l_current_poc->compS = l_current_poc->compno0;
    l_current_poc->compE= l_current_poc->compno1;
    l_current_poc->resS = l_current_poc->resno0;
    l_current_poc->resE = l_current_poc->resno1;
    l_current_poc->layE = l_current_poc->layno1;
    l_current_poc->prg  = l_current_poc->prg1;
    l_current_poc->prcS = 0;
    // special treatment here different from the first element
    l_current_poc->layS = (l_current_poc->layE > (l_current_poc-1)->layE) ? l_current_poc->layE : 0;

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

/**
 * Updates the coding parameters if the encoding is not used with Progression order changes and final (and cinema parameters are used).
 *
 * @param  p_cp    the coding parameters to modify
 * @param  p_tileno  the tile index being concerned.
 * @param  p_tx0    X0 parameter for the tile
 * @param  p_tx1    X1 parameter for the tile
 * @param  p_ty0    Y0 parameter for the tile
 * @param  p_ty1    Y1 parameter for the tile
 * @param  p_max_prec  the maximum precision for all the bands of the tile
 * @param  p_max_res  the maximum number of resolutions for all the poc inside the tile.
 * @param  dx_min    the minimum dx of all the components of all the resolutions for the tile.
 * @param  dy_min    the minimum dy of all the components of all the resolutions for the tile.
 */
void pi_update_encode_not_poc (
                opj_cp_t *p_cp,
                OPJ_UINT32 p_num_comps,
                OPJ_UINT32 p_tileno,
                OPJ_INT32 p_tx0,
                OPJ_INT32 p_tx1,
                OPJ_INT32 p_ty0,
                OPJ_INT32 p_ty1,
                OPJ_UINT32 p_max_prec,
                OPJ_UINT32 p_max_res,
                                OPJ_UINT32 p_dx_min,
                OPJ_UINT32 p_dy_min)
{
  // loop
  OPJ_UINT32 pino;
  // tile coding parameter
  opj_tcp_t *l_tcp = 00;
  // current poc being updated
  opj_poc_t * l_current_poc = 00;
  // number of pocs
  OPJ_UINT32 l_poc_bound;

  // preconditions in debug
  assert(p_cp != 00);
  assert(p_tileno < p_cp->tw * p_cp->th);

  // initializations
  l_tcp = &p_cp->tcps [p_tileno];

  /* number of iterations in the loop */
  l_poc_bound = l_tcp->numpocs+1;

  // start at first element, and to make sure the compiler will not make a calculation each time in the loop
  // store a pointer to the current element to modify rather than l_tcp->pocs[i]
  l_current_poc = l_tcp->pocs;

  for
    (pino = 0; pino < l_poc_bound ; ++pino)
  {
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

/**
 * Destroys a packet iterator array.
 *
 * @param  p_pi      the packet iterator array to destroy.
 * @param  p_nb_elements  the number of elements in the array.
 */
void pi_destroy(
        opj_pi_iterator_t *p_pi,
        OPJ_UINT32 p_nb_elements)
{
  OPJ_UINT32 compno, pino;
  opj_pi_iterator_t *l_current_pi = p_pi;
  if
    (p_pi)
  {
    if
      (p_pi->include)
    {
      opj_free(p_pi->include);
      p_pi->include = 00;
    }
    // TODO
    for
      (pino = 0; pino < p_nb_elements; ++pino)
    {
      if
        (l_current_pi->comps)
      {
        opj_pi_comp_t *l_current_component = l_current_pi->comps;
        for
          (compno = 0; compno < l_current_pi->numcomps; compno++)
        {
          if
            (l_current_component->resolutions)
          {
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

bool pi_next(opj_pi_iterator_t * pi) {
  switch (pi->poc.prg) {
    case LRCP:
      return pi_next_lrcp(pi);
    case RLCP:
      return pi_next_rlcp(pi);
    case RPCL:
      return pi_next_rpcl(pi);
    case PCRL:
      return pi_next_pcrl(pi);
    case CPRL:
      return pi_next_cprl(pi);
    case PROG_UNKNOWN:
      return false;
  }

  return false;
}

OPJ_INT32 pi_check_next_level(OPJ_INT32 pos,opj_cp_t *cp,OPJ_UINT32 tileno, OPJ_UINT32 pino, const OPJ_CHAR *prog)
{
  OPJ_INT32 i,l;
  opj_tcp_t *tcps =&cp->tcps[tileno];
  opj_poc_t *tcp = &tcps->pocs[pino];
  if(pos>=0){
    for(i=pos;pos>=0;i--){
      switch(prog[i]){
    case 'R':
      if(tcp->res_t==tcp->resE){
        l=pi_check_next_level(pos-1,cp,tileno,pino,prog);
        if(l==1){
          return 1;
        }else{
          return 0;
        }
      }else{
        return 1;
      }
      break;
    case 'C':
      if(tcp->comp_t==tcp->compE){
        l=pi_check_next_level(pos-1,cp,tileno,pino,prog);
        if(l==1){
          return 1;
        }else{
          return 0;
        }
      }else{
        return 1;
      }
      break;
    case 'L':
      if(tcp->lay_t==tcp->layE){
        l=pi_check_next_level(pos-1,cp,tileno,pino,prog);
        if(l==1){
          return 1;
        }else{
          return 0;
        }
      }else{
        return 1;
      }
      break;
    case 'P':
      switch(tcp->prg){
        case LRCP||RLCP:
          if(tcp->prc_t == tcp->prcE){
            l=pi_check_next_level(i-1,cp,tileno,pino,prog);
            if(l==1){
              return 1;
            }else{
              return 0;
            }
          }else{
            return 1;
          }
          break;
      default:
        if(tcp->tx0_t == tcp->txE){
          //TY
          if(tcp->ty0_t == tcp->tyE){
            l=pi_check_next_level(i-1,cp,tileno,pino,prog);
            if(l==1){
              return 1;
            }else{
              return 0;
            }
          }else{
            return 1;
          }//TY
        }else{
          return 1;
        }
        break;
      }//end case P
    }//end switch
    }//end for
  }//end if
  return 0;
}


void pi_create_encode( opj_pi_iterator_t *pi, opj_cp_t *cp,OPJ_UINT32 tileno, OPJ_UINT32 pino,OPJ_UINT32 tpnum, OPJ_INT32 tppos, J2K_T2_MODE t2_mode)
{
  const OPJ_CHAR *prog;
  OPJ_INT32 i,l;
  OPJ_UINT32 incr_top=1,resetX=0;
  opj_tcp_t *tcps =&cp->tcps[tileno];
  opj_poc_t *tcp= &tcps->pocs[pino];

  prog = j2k_convert_progression_order(tcp->prg);

  pi[pino].first = 1;
  pi[pino].poc.prg = tcp->prg;

  if(!(cp->m_specific_param.m_enc.m_tp_on&& ((!cp->m_specific_param.m_enc.m_cinema && (t2_mode == FINAL_PASS)) || cp->m_specific_param.m_enc.m_cinema))){
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
  }else {
    for(i=tppos+1;i<4;i++){
      switch(prog[i]){
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
        switch(tcp->prg){
          case LRCP:
          case RLCP:
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

    if(tpnum==0){
      for(i=tppos;i>=0;i--){
        switch(prog[i]){
            case 'C':
              tcp->comp_t = tcp->compS;
              pi[pino].poc.compno0 = tcp->comp_t;
              pi[pino].poc.compno1 = tcp->comp_t+1;
              tcp->comp_t+=1;
              break;
            case 'R':
              tcp->res_t = tcp->resS;
              pi[pino].poc.resno0 = tcp->res_t;
              pi[pino].poc.resno1 = tcp->res_t+1;
              tcp->res_t+=1;
              break;
            case 'L':
              tcp->lay_t = tcp->layS;
              pi[pino].poc.layno0 = tcp->lay_t;
              pi[pino].poc.layno1 = tcp->lay_t+1;
              tcp->lay_t+=1;
              break;
            case 'P':
              switch(tcp->prg){
                case LRCP:
                case RLCP:
                  tcp->prc_t = tcp->prcS;
                  pi[pino].poc.precno0 = tcp->prc_t;
                  pi[pino].poc.precno1 = tcp->prc_t+1;
                  tcp->prc_t+=1;
                  break;
                default:
                  tcp->tx0_t = tcp->txS;
                  tcp->ty0_t = tcp->tyS;
                  pi[pino].poc.tx0 = tcp->tx0_t;
                  pi[pino].poc.tx1 = tcp->tx0_t + tcp->dx - (tcp->tx0_t % tcp->dx);
                  pi[pino].poc.ty0 = tcp->ty0_t;
                  pi[pino].poc.ty1 = tcp->ty0_t + tcp->dy - (tcp->ty0_t % tcp->dy);
                  tcp->tx0_t = pi[pino].poc.tx1;
                  tcp->ty0_t = pi[pino].poc.ty1;
                  break;
              }
              break;
        }
      }
      incr_top=1;
    }else{
      for(i=tppos;i>=0;i--){
        switch(prog[i]){
            case 'C':
              pi[pino].poc.compno0 = tcp->comp_t-1;
              pi[pino].poc.compno1 = tcp->comp_t;
              break;
            case 'R':
              pi[pino].poc.resno0 = tcp->res_t-1;
              pi[pino].poc.resno1 = tcp->res_t;
              break;
            case 'L':
              pi[pino].poc.layno0 = tcp->lay_t-1;
              pi[pino].poc.layno1 = tcp->lay_t;
              break;
            case 'P':
              switch(tcp->prg){
                case LRCP:
                case RLCP:
                  pi[pino].poc.precno0 = tcp->prc_t-1;
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
        if(incr_top==1){
          switch(prog[i]){
              case 'R':
                if(tcp->res_t==tcp->resE){
                  l=pi_check_next_level(i-1,cp,tileno,pino,prog);
                  if(l==1){
                    tcp->res_t = tcp->resS;
                    pi[pino].poc.resno0 = tcp->res_t;
                    pi[pino].poc.resno1 = tcp->res_t+1;
                    tcp->res_t+=1;
                    incr_top=1;
                  }else{
                    incr_top=0;
                  }
                }else{
                  pi[pino].poc.resno0 = tcp->res_t;
                  pi[pino].poc.resno1 = tcp->res_t+1;
                  tcp->res_t+=1;
                  incr_top=0;
                }
                break;
              case 'C':
                if(tcp->comp_t ==tcp->compE){
                  l=pi_check_next_level(i-1,cp,tileno,pino,prog);
                  if(l==1){
                    tcp->comp_t = tcp->compS;
                    pi[pino].poc.compno0 = tcp->comp_t;
                    pi[pino].poc.compno1 = tcp->comp_t+1;
                    tcp->comp_t+=1;
                    incr_top=1;
                  }else{
                    incr_top=0;
                  }
                }else{
                  pi[pino].poc.compno0 = tcp->comp_t;
                  pi[pino].poc.compno1 = tcp->comp_t+1;
                  tcp->comp_t+=1;
                  incr_top=0;
                }
                break;
              case 'L':
                if(tcp->lay_t == tcp->layE){
                  l=pi_check_next_level(i-1,cp,tileno,pino,prog);
                  if(l==1){
                    tcp->lay_t = tcp->layS;
                    pi[pino].poc.layno0 = tcp->lay_t;
                    pi[pino].poc.layno1 = tcp->lay_t+1;
                    tcp->lay_t+=1;
                    incr_top=1;
                  }else{
                    incr_top=0;
                  }
                }else{
                  pi[pino].poc.layno0 = tcp->lay_t;
                  pi[pino].poc.layno1 = tcp->lay_t+1;
                  tcp->lay_t+=1;
                  incr_top=0;
                }
                break;
              case 'P':
                switch(tcp->prg){
                  case LRCP:
                  case RLCP:
                    if(tcp->prc_t == tcp->prcE){
                      l=pi_check_next_level(i-1,cp,tileno,pino,prog);
                      if(l==1){
                        tcp->prc_t = tcp->prcS;
                        pi[pino].poc.precno0 = tcp->prc_t;
                        pi[pino].poc.precno1 = tcp->prc_t+1;
                        tcp->prc_t+=1;
                        incr_top=1;
                      }else{
                        incr_top=0;
                      }
                    }else{
                      pi[pino].poc.precno0 = tcp->prc_t;
                      pi[pino].poc.precno1 = tcp->prc_t+1;
                      tcp->prc_t+=1;
                      incr_top=0;
                    }
                    break;
                  default:
                    if(tcp->tx0_t >= tcp->txE){
                      if(tcp->ty0_t >= tcp->tyE){
                        l=pi_check_next_level(i-1,cp,tileno,pino,prog);
                        if(l==1){
                          tcp->ty0_t = tcp->tyS;
                          pi[pino].poc.ty0 = tcp->ty0_t;
                          pi[pino].poc.ty1 = tcp->ty0_t + tcp->dy - (tcp->ty0_t % tcp->dy);
                          tcp->ty0_t = pi[pino].poc.ty1;
                          incr_top=1;resetX=1;
                        }else{
                          incr_top=0;resetX=0;
                        }
                      }else{
                        pi[pino].poc.ty0 = tcp->ty0_t;
                        pi[pino].poc.ty1 = tcp->ty0_t + tcp->dy - (tcp->ty0_t % tcp->dy);
                        tcp->ty0_t = pi[pino].poc.ty1;
                        incr_top=0;resetX=1;
                      }
                      if(resetX==1){
                        tcp->tx0_t = tcp->txS;
                        pi[pino].poc.tx0 = tcp->tx0_t;
                        pi[pino].poc.tx1 = tcp->tx0_t + tcp->dx- (tcp->tx0_t % tcp->dx);
                        tcp->tx0_t = pi[pino].poc.tx1;
                      }
                    }else{
                      pi[pino].poc.tx0 = tcp->tx0_t;
                      pi[pino].poc.tx1 = tcp->tx0_t + tcp->dx- (tcp->tx0_t % tcp->dx);
                      tcp->tx0_t = pi[pino].poc.tx1;
                      incr_top=0;
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
