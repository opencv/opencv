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

#include "tcd.h"
#include "openjpeg.h"
#include "j2k.h"
#include "opj_includes.h"
#include "event.h"
#include "t2.h"
#include "t1.h"
#include "opj_malloc.h"
#include "int.h"
#include "tgt.h"
#include "dwt.h"
#include "mct.h"
#include "j2k_lib.h"
#include "profile.h"

/**
 * Deallocates the encoding data of the given precinct.
 */
static void tcd_code_block_enc_deallocate (opj_tcd_precinct_t * p_precinct);
/**
 * Allocates memory for an encoding code block.
 */
static bool tcd_code_block_enc_allocate (opj_tcd_cblk_enc_t * p_code_block);
/**
 * Allocates memory for a decoding code block.
 */
static bool tcd_code_block_dec_allocate (opj_tcd_cblk_dec_t * p_code_block);
/**
Free the memory allocated for encoding
@param tcd TCD handle
*/
static void tcd_free_tile(opj_tcd_t *tcd);

/* ----------------------------------------------------------------------- */

/**
Create a new TCD handle
*/
opj_tcd_t* tcd_create(bool p_is_decoder)
{
  opj_tcd_t *l_tcd = 00;

  /* create the tcd structure */
  l_tcd = (opj_tcd_t*)  opj_malloc(sizeof(opj_tcd_t));
  if
    (!l_tcd)
  {
    return 00;
  }
  memset(l_tcd,0,sizeof(opj_tcd_t));
  l_tcd->m_is_decoder = p_is_decoder ? 1 : 0;
  l_tcd->tcd_image = (opj_tcd_image_t*)opj_malloc(sizeof(opj_tcd_image_t));
  if
    (!l_tcd->tcd_image)
  {
    opj_free(l_tcd);
    return 00;
  }
  memset(l_tcd->tcd_image,0,sizeof(opj_tcd_image_t));
  return l_tcd;
}

/**
Destroy a previously created TCD handle
*/
void tcd_destroy(opj_tcd_t *tcd) {
  if
    (tcd)
  {
    tcd_free_tile(tcd);
    if
      (tcd->tcd_image)
    {
      opj_free(tcd->tcd_image);
      tcd->tcd_image = 00;
    }
    opj_free(tcd);
  }
}

/* ----------------------------------------------------------------------- */
/**
 * Initialize the tile coder and may reuse some meory.
 * @param  p_tcd    TCD handle.
 * @param  p_image    raw image.
 * @param  p_cp    coding parameters.
 * @param  p_tile_no  current tile index to encode.
 *
 * @return true if the encoding values could be set (false otherwise).
*/
#define MACRO_TCD_ALLOCATE(FUNCTION,TYPE,FRACTION,ELEMENT,FUNCTION_ELEMENT)  \
bool FUNCTION                                \
      (                                \
        opj_tcd_t *p_tcd,                      \
        OPJ_UINT32 p_tile_no                    \
      )                                \
{                                      \
  OPJ_UINT32 (*l_gain_ptr)(OPJ_UINT32) = 00;                \
  OPJ_UINT32 compno, resno, bandno, precno, cblkno;            \
  opj_tcp_t * l_tcp = 00;                          \
  opj_cp_t * l_cp = 00;                          \
  opj_tcd_tile_t * l_tile = 00;                      \
  opj_tccp_t *l_tccp = 00;                        \
  opj_tcd_tilecomp_t *l_tilec = 00;                    \
  opj_image_comp_t * l_image_comp = 00;                  \
  opj_tcd_resolution_t *l_res = 00;                    \
  opj_tcd_band_t *l_band = 00;                      \
  opj_stepsize_t * l_step_size = 00;                    \
  opj_tcd_precinct_t *l_current_precinct = 00;              \
  TYPE* l_code_block = 00;                        \
  opj_image_t *  l_image = 00;                      \
  OPJ_UINT32 p,q;                              \
  OPJ_UINT32 l_level_no;                          \
  OPJ_UINT32 l_pdx, l_pdy;                        \
  OPJ_UINT32 l_gain;                            \
  OPJ_INT32 l_x0b, l_y0b;                          \
  /* extent of precincts , top left, bottom right**/            \
  OPJ_INT32 l_tl_prc_x_start, l_tl_prc_y_start, l_br_prc_x_end, l_br_prc_y_end;  \
  /* number of precinct for a resolution */                \
  OPJ_UINT32 l_nb_precincts;                        \
  /* room needed to store l_nb_precinct precinct for a resolution */    \
  OPJ_UINT32 l_nb_precinct_size;                      \
  /* number of code blocks for a precinct*/                \
  OPJ_UINT32 l_nb_code_blocks;                      \
  /* room needed to store l_nb_code_blocks code blocks for a precinct*/  \
  OPJ_UINT32 l_nb_code_blocks_size;                    \
  /* size of data for a tile */                      \
  OPJ_UINT32 l_data_size;                          \
  l_cp = p_tcd->cp;                            \
  l_tcp = &(l_cp->tcps[p_tile_no]);                    \
  l_tile = p_tcd->tcd_image->tiles;                    \
  l_tccp = l_tcp->tccps;                          \
  l_tilec = l_tile->comps;                        \
  l_image = p_tcd->image;                          \
  l_image_comp = p_tcd->image->comps;                    \
                                      \
  p = p_tile_no % l_cp->tw;  /* tile coordinates */            \
  q = p_tile_no / l_cp->tw;                        \
                                      \
  /* 4 borders of the tile rescale on the image if necessary */      \
  l_tile->x0 = int_max(l_cp->tx0 + p * l_cp->tdx, l_image->x0);      \
  l_tile->y0 = int_max(l_cp->ty0 + q * l_cp->tdy, l_image->y0);      \
  l_tile->x1 = int_min(l_cp->tx0 + (p + 1) * l_cp->tdx, l_image->x1);    \
  l_tile->y1 = int_min(l_cp->ty0 + (q + 1) * l_cp->tdy, l_image->y1);    \
  /*tile->numcomps = image->numcomps; */                  \
  for                                    \
    (compno = 0; compno < l_tile->numcomps; ++compno)          \
  {                                    \
    /* border of each l_tile component (global) */            \
    l_tilec->x0 = int_ceildiv(l_tile->x0, l_image_comp->dx);      \
    l_tilec->y0 = int_ceildiv(l_tile->y0, l_image_comp->dy);      \
    l_tilec->x1 = int_ceildiv(l_tile->x1, l_image_comp->dx);      \
    l_tilec->y1 = int_ceildiv(l_tile->y1, l_image_comp->dy);      \
                                      \
    l_data_size = (l_tilec->x1 - l_tilec->x0)              \
          * (l_tilec->y1 - l_tilec->y0) * sizeof(OPJ_UINT32 );  \
    l_tilec->numresolutions = l_tccp->numresolutions;          \
    if                                  \
      (l_tccp->numresolutions < l_cp->m_specific_param.m_dec.m_reduce)\
    {                                  \
      l_tilec->minimum_num_resolutions = 1;              \
    }                                  \
    else                                \
    {                                  \
      l_tilec->minimum_num_resolutions = l_tccp->numresolutions - l_cp->m_specific_param.m_dec.m_reduce;\
    }                                  \
    if                                  \
      (l_tilec->data == 00)                      \
    {                                  \
            l_tilec->data = (OPJ_INT32 *) opj_aligned_malloc(l_data_size);  \
      if                                \
        (! l_tilec->data )                      \
      {                                \
        return false;                        \
      }                                \
      l_tilec->data_size = l_data_size;                \
    }                                  \
    else if                                \
      (l_data_size > l_tilec->data_size)                \
    {                                  \
      l_tilec->data = (OPJ_INT32 *) opj_realloc(l_tilec->data, l_data_size);\
      if                                \
        (! l_tilec->data)                      \
      {                                \
        return false;                        \
      }                                \
      l_tilec->data_size = l_data_size;                \
    }                                  \
    l_data_size = l_tilec->numresolutions * sizeof(opj_tcd_resolution_t);\
    if                                  \
      (l_tilec->resolutions == 00)                  \
    {                                  \
            l_tilec->resolutions = (opj_tcd_resolution_t *) opj_malloc(l_data_size);\
      if                                \
        (! l_tilec->resolutions )                  \
      {                                \
        return false;                        \
      }                                \
      l_tilec->resolutions_size = l_data_size;            \
      memset(l_tilec->resolutions,0,l_data_size);            \
    }                                  \
    else if                                \
      (l_data_size > l_tilec->resolutions_size)            \
    {                                  \
      l_tilec->resolutions = (opj_tcd_resolution_t *) opj_realloc(l_tilec->resolutions, l_data_size);\
      if                                \
        (! l_tilec->resolutions)                  \
      {                                \
        return false;                        \
      }                                \
      memset(((OPJ_BYTE*) l_tilec->resolutions)+l_tilec->resolutions_size,0,l_data_size - l_tilec->resolutions_size);\
      l_tilec->resolutions_size = l_data_size;            \
    }                                  \
    l_level_no = l_tilec->numresolutions - 1;              \
    l_res = l_tilec->resolutions;                    \
    l_step_size = l_tccp->stepsizes;                  \
    if                                  \
      (l_tccp->qmfbid == 0)                      \
    {                                  \
      l_gain_ptr = &dwt_getgain_real;                  \
    }                                  \
    else                                \
    {                                  \
      l_gain_ptr  = &dwt_getgain;                    \
    }                                  \
    for                                  \
      (resno = 0; resno < l_tilec->numresolutions; ++resno)      \
    {                                  \
      OPJ_INT32 tlcbgxstart, tlcbgystart, brcbgxend, brcbgyend;    \
      OPJ_UINT32 cbgwidthexpn, cbgheightexpn;              \
      OPJ_UINT32 cblkwidthexpn, cblkheightexpn;            \
      /* border for each resolution level (global) */          \
      l_res->x0 = int_ceildivpow2(l_tilec->x0, l_level_no);      \
      l_res->y0 = int_ceildivpow2(l_tilec->y0, l_level_no);      \
      l_res->x1 = int_ceildivpow2(l_tilec->x1, l_level_no);      \
      l_res->y1 = int_ceildivpow2(l_tilec->y1, l_level_no);      \
      /* p. 35, table A-23, ISO/IEC FDIS154444-1 : 2000 (18 august 2000) */\
      l_pdx = l_tccp->prcw[resno];                  \
      l_pdy = l_tccp->prch[resno];                  \
      /* p. 64, B.6, ISO/IEC FDIS15444-1 : 2000 (18 august 2000)  */  \
      l_tl_prc_x_start = int_floordivpow2(l_res->x0, l_pdx) << l_pdx;  \
      l_tl_prc_y_start = int_floordivpow2(l_res->y0, l_pdy) << l_pdy;  \
      l_br_prc_x_end = int_ceildivpow2(l_res->x1, l_pdx) << l_pdx;  \
      l_br_prc_y_end = int_ceildivpow2(l_res->y1, l_pdy) << l_pdy;  \
                                      \
      l_res->pw = (l_res->x0 == l_res->x1) ? 0 : ((l_br_prc_x_end - l_tl_prc_x_start) >> l_pdx);\
      l_res->ph = (l_res->y0 == l_res->y1) ? 0 : ((l_br_prc_y_end - l_tl_prc_y_start) >> l_pdy);\
      l_nb_precincts = l_res->pw * l_res->ph;              \
      l_nb_precinct_size = l_nb_precincts * sizeof(opj_tcd_precinct_t);\
      if                                \
        (resno == 0)                        \
      {                                \
        tlcbgxstart = l_tl_prc_x_start;                \
        tlcbgystart = l_tl_prc_y_start;                \
        brcbgxend = l_br_prc_x_end;                  \
        brcbgyend = l_br_prc_y_end;                  \
        cbgwidthexpn = l_pdx;                    \
        cbgheightexpn = l_pdy;                    \
        l_res->numbands = 1;                    \
      }                                \
      else                              \
      {                                \
        tlcbgxstart = int_ceildivpow2(l_tl_prc_x_start, 1);      \
        tlcbgystart = int_ceildivpow2(l_tl_prc_y_start, 1);      \
        brcbgxend = int_ceildivpow2(l_br_prc_x_end, 1);        \
        brcbgyend = int_ceildivpow2(l_br_prc_y_end, 1);        \
        cbgwidthexpn = l_pdx - 1;                  \
        cbgheightexpn = l_pdy - 1;                  \
        l_res->numbands = 3;                    \
      }                                \
                                      \
      cblkwidthexpn = uint_min(l_tccp->cblkw, cbgwidthexpn);      \
      cblkheightexpn = uint_min(l_tccp->cblkh, cbgheightexpn);    \
      l_band = l_res->bands;                      \
      for                                \
        (bandno = 0; bandno < l_res->numbands; ++bandno)      \
      {                                \
        OPJ_INT32 numbps;                      \
        if                              \
          (resno == 0)                      \
        {                              \
          l_band->bandno = 0 ;                  \
          l_band->x0 = int_ceildivpow2(l_tilec->x0, l_level_no);  \
          l_band->y0 = int_ceildivpow2(l_tilec->y0, l_level_no);  \
          l_band->x1 = int_ceildivpow2(l_tilec->x1, l_level_no);  \
          l_band->y1 = int_ceildivpow2(l_tilec->y1, l_level_no);  \
        }                              \
        else                            \
        {                              \
          l_band->bandno = bandno + 1;              \
          /* x0b = 1 if bandno = 1 or 3 */            \
          l_x0b = l_band->bandno&1;                \
          /* y0b = 1 if bandno = 2 or 3 */            \
          l_y0b = (l_band->bandno)>>1;              \
          /* l_band border (global) */              \
          l_band->x0 = int_ceildivpow2(l_tilec->x0 - (1 << l_level_no) * l_x0b, l_level_no + 1);\
          l_band->y0 = int_ceildivpow2(l_tilec->y0 - (1 << l_level_no) * l_y0b, l_level_no + 1);\
          l_band->x1 = int_ceildivpow2(l_tilec->x1 - (1 << l_level_no) * l_x0b, l_level_no + 1);\
          l_band->y1 = int_ceildivpow2(l_tilec->y1 - (1 << l_level_no) * l_y0b, l_level_no + 1);\
        }                              \
        /** avoid an if with storing function pointer */      \
        l_gain = (*l_gain_ptr) (l_band->bandno);          \
        numbps = l_image_comp->prec + l_gain;            \
        l_band->stepsize = (OPJ_FLOAT32)(((1.0 + l_step_size->mant / 2048.0) * pow(2.0, (OPJ_INT32) (numbps - l_step_size->expn)))) * FRACTION;\
        l_band->numbps = l_step_size->expn + l_tccp->numgbits - 1;  /* WHY -1 ? */\
        if                              \
          (! l_band->precincts)                  \
        {                              \
          l_band->precincts = (opj_tcd_precinct_t *) opj_malloc(/*3 * */ l_nb_precinct_size);\
          if                            \
            (! l_band->precincts)                \
          {                            \
            return false;                    \
          }                            \
          memset(l_band->precincts,0,l_nb_precinct_size);      \
          l_band->precincts_data_size = l_nb_precinct_size;    \
        }                              \
        else if                            \
          (l_band->precincts_data_size < l_nb_precinct_size)    \
        {                              \
          l_band->precincts = (opj_tcd_precinct_t *) opj_realloc(l_band->precincts,/*3 * */ l_nb_precinct_size);\
          if                            \
            (! l_band->precincts)                \
          {                            \
            return false;                    \
          }                            \
          memset(((OPJ_BYTE *) l_band->precincts) + l_band->precincts_data_size,0,l_nb_precinct_size - l_band->precincts_data_size);\
          l_band->precincts_data_size = l_nb_precinct_size;    \
        }                              \
        l_current_precinct = l_band->precincts;            \
        for                              \
          (precno = 0; precno < l_nb_precincts; ++precno)      \
        {                              \
          OPJ_INT32 tlcblkxstart, tlcblkystart, brcblkxend, brcblkyend;  \
          OPJ_INT32 cbgxstart = tlcbgxstart + (precno % l_res->pw) * (1 << cbgwidthexpn);\
          OPJ_INT32 cbgystart = tlcbgystart + (precno / l_res->pw) * (1 << cbgheightexpn);\
          OPJ_INT32 cbgxend = cbgxstart + (1 << cbgwidthexpn);      \
          OPJ_INT32 cbgyend = cbgystart + (1 << cbgheightexpn);      \
          /* precinct size (global) */              \
          l_current_precinct->x0 = int_max(cbgxstart, l_band->x0);\
          l_current_precinct->y0 = int_max(cbgystart, l_band->y0);\
          l_current_precinct->x1 = int_min(cbgxend, l_band->x1);  \
          l_current_precinct->y1 = int_min(cbgyend, l_band->y1);  \
          tlcblkxstart = int_floordivpow2(l_current_precinct->x0, cblkwidthexpn) << cblkwidthexpn;\
          tlcblkystart = int_floordivpow2(l_current_precinct->y0, cblkheightexpn) << cblkheightexpn;\
          brcblkxend = int_ceildivpow2(l_current_precinct->x1, cblkwidthexpn) << cblkwidthexpn;\
          brcblkyend = int_ceildivpow2(l_current_precinct->y1, cblkheightexpn) << cblkheightexpn;\
          l_current_precinct->cw = (brcblkxend - tlcblkxstart) >> cblkwidthexpn;\
          l_current_precinct->ch = (brcblkyend - tlcblkystart) >> cblkheightexpn;\
          l_nb_code_blocks = l_current_precinct->cw * l_current_precinct->ch;\
          l_nb_code_blocks_size = l_nb_code_blocks * sizeof(TYPE);\
          if                            \
            (! l_current_precinct->cblks.ELEMENT)        \
          {                            \
            l_current_precinct->cblks.ELEMENT = (TYPE*) opj_malloc(l_nb_code_blocks_size);\
            if                          \
              (! l_current_precinct->cblks.ELEMENT )      \
            {                          \
              return false;                  \
            }                          \
            memset(l_current_precinct->cblks.ELEMENT,0,l_nb_code_blocks_size);\
            l_current_precinct->block_size = l_nb_code_blocks_size;\
          }                            \
          else if                          \
            (l_nb_code_blocks_size > l_current_precinct->block_size)\
          {                            \
            l_current_precinct->cblks.ELEMENT = (TYPE*)      \
              opj_realloc(l_current_precinct->cblks.ELEMENT, l_nb_code_blocks_size);\
            if                          \
              (! l_current_precinct->cblks.ELEMENT )      \
            {                          \
              return false;                  \
            }                          \
            memset(((OPJ_BYTE *) l_current_precinct->cblks.ELEMENT) + l_current_precinct->block_size\
                    ,0                  \
                    ,l_nb_code_blocks_size - l_current_precinct->block_size);\
            l_current_precinct->block_size = l_nb_code_blocks_size;\
          }                            \
          if                            \
            (! l_current_precinct->incltree)          \
          {                            \
                        l_current_precinct->incltree = tgt_create(l_current_precinct->cw,\
                                  l_current_precinct->ch);\
          }                            \
          else                          \
          {                            \
            l_current_precinct->incltree = tgt_init(l_current_precinct->incltree,\
                                l_current_precinct->cw, \
                                l_current_precinct->ch);\
          }                            \
          if                            \
            (! l_current_precinct->incltree)          \
          {                            \
            return false;                    \
          }                            \
          if                            \
            (! l_current_precinct->imsbtree)          \
          {                            \
                        l_current_precinct->imsbtree = tgt_create(      \
                            l_current_precinct->cw,\
                            l_current_precinct->ch);\
          }                            \
          else                          \
          {                            \
            l_current_precinct->imsbtree = tgt_init(      \
                              l_current_precinct->imsbtree,\
                              l_current_precinct->cw,\
                              l_current_precinct->ch);\
          }                            \
          if                            \
            (! l_current_precinct->imsbtree)          \
          {                            \
            return false;                    \
          }                            \
          l_code_block = l_current_precinct->cblks.ELEMENT;    \
          for                            \
            (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno)  \
          {                            \
            OPJ_INT32 cblkxstart = tlcblkxstart + (cblkno % l_current_precinct->cw) * (1 << cblkwidthexpn);\
            OPJ_INT32 cblkystart = tlcblkystart + (cblkno / l_current_precinct->cw) * (1 << cblkheightexpn);\
            OPJ_INT32 cblkxend = cblkxstart + (1 << cblkwidthexpn);  \
            OPJ_INT32 cblkyend = cblkystart + (1 << cblkheightexpn);  \
            /* code-block size (global) */            \
            l_code_block->x0 = int_max(cblkxstart, l_current_precinct->x0);\
            l_code_block->y0 = int_max(cblkystart, l_current_precinct->y0);\
            l_code_block->x1 = int_min(cblkxend, l_current_precinct->x1);\
            l_code_block->y1 = int_min(cblkyend, l_current_precinct->y1);\
            if                          \
              (! FUNCTION_ELEMENT(l_code_block))        \
            {                          \
              return false;                  \
            }                          \
            ++l_code_block;                    \
          }                            \
          ++l_current_precinct;                  \
        } /* precno */                        \
        ++l_band;                          \
        ++l_step_size;                        \
      } /* bandno */                          \
      ++l_res;                            \
      --l_level_no;                          \
    } /* resno */                            \
    ++l_tccp;                              \
    ++l_tilec;                              \
    ++l_image_comp;                            \
  } /* compno */                              \
  return true;                              \
}                                      \

MACRO_TCD_ALLOCATE(tcd_init_encode_tile,opj_tcd_cblk_enc_t,1.f,enc,tcd_code_block_enc_allocate)
MACRO_TCD_ALLOCATE(tcd_init_decode_tile,opj_tcd_cblk_dec_t,0.5f,dec,tcd_code_block_dec_allocate)

#undef MACRO_TCD_ALLOCATE

/**
 * Allocates memory for an encoding code block.
 */
bool tcd_code_block_enc_allocate (opj_tcd_cblk_enc_t * p_code_block)
{
  if
    (! p_code_block->data)
  {
    p_code_block->data = (OPJ_BYTE*) opj_malloc(8192+1);
    if
      (! p_code_block->data)
    {
      return false;
    }
    p_code_block->data+=1;
    /* no memset since data */
    p_code_block->layers = (opj_tcd_layer_t*) opj_malloc(100 * sizeof(opj_tcd_layer_t));
    if
      (! p_code_block->layers)
    {
      return false;
    }
    p_code_block->passes = (opj_tcd_pass_t*) opj_malloc(100 * sizeof(opj_tcd_pass_t));
    if
      (! p_code_block->passes)
    {
      return false;
    }
  }
  memset(p_code_block->layers,0,100 * sizeof(opj_tcd_layer_t));
  memset(p_code_block->passes,0,100 * sizeof(opj_tcd_pass_t));
  return true;
}

/**
 * Allocates memory for a decoding code block.
 */
bool tcd_code_block_dec_allocate (opj_tcd_cblk_dec_t * p_code_block)
{
  OPJ_UINT32 l_seg_size;

  if
    (! p_code_block->data)
  {
    p_code_block->data = (OPJ_BYTE*) opj_malloc(8192);
    if
      (! p_code_block->data)
    {
      return false;
    }
    l_seg_size = J2K_DEFAULT_NB_SEGS * sizeof(opj_tcd_seg_t);
    p_code_block->segs = (opj_tcd_seg_t *) opj_malloc(l_seg_size);
    if
      (! p_code_block->segs)
    {
      return false;
    }
    memset(p_code_block->segs,0,l_seg_size);
    p_code_block->m_current_max_segs = J2K_DEFAULT_NB_SEGS;
  }
  // TODO
  //p_code_block->numsegs = 0;
  return true;
}

/**
 * Deallocates the encoding data of the given precinct.
 */
void tcd_code_block_enc_deallocate (opj_tcd_precinct_t * p_precinct)
{
  OPJ_UINT32 cblkno , l_nb_code_blocks;

  opj_tcd_cblk_enc_t * l_code_block = p_precinct->cblks.enc;
  if
    (l_code_block)
  {
    l_nb_code_blocks = p_precinct->block_size / sizeof(opj_tcd_cblk_enc_t);
    for
      (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno)
    {
      if
        (l_code_block->data)
      {
        opj_free(l_code_block->data-1);
        l_code_block->data = 00;
      }
      if
        (l_code_block->layers)
      {
        opj_free(l_code_block->layers );
        l_code_block->layers = 00;
      }
      if
        (l_code_block->passes)
      {
        opj_free(l_code_block->passes );
        l_code_block->passes = 00;
      }
      ++l_code_block;
    }
    opj_free(p_precinct->cblks.enc);
    p_precinct->cblks.enc = 00;
  }
}

/**
 * Deallocates the encoding data of the given precinct.
 */
void tcd_code_block_dec_deallocate (opj_tcd_precinct_t * p_precinct)
{
  OPJ_UINT32 cblkno , l_nb_code_blocks;

  opj_tcd_cblk_dec_t * l_code_block = p_precinct->cblks.dec;
  if
    (l_code_block)
  {
    l_nb_code_blocks = p_precinct->block_size / sizeof(opj_tcd_cblk_dec_t);
    for
      (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno)
    {
      if
        (l_code_block->data)
      {
        opj_free(l_code_block->data);
        l_code_block->data = 00;
      }
      if
        (l_code_block->segs)
      {
        opj_free(l_code_block->segs );
        l_code_block->segs = 00;
      }
      ++l_code_block;
    }
    opj_free(p_precinct->cblks.dec);
    p_precinct->cblks.dec = 00;
  }
}

void tcd_free_tile(opj_tcd_t *p_tcd)
{
  OPJ_UINT32 compno, resno, bandno, precno;
  opj_tcd_tile_t *l_tile = 00;
  opj_tcd_tilecomp_t *l_tile_comp = 00;
  opj_tcd_resolution_t *l_res = 00;
  opj_tcd_band_t *l_band = 00;
  opj_tcd_precinct_t *l_precinct = 00;
  OPJ_UINT32 l_nb_resolutions, l_nb_precincts;
  void (* l_tcd_code_block_deallocate) (opj_tcd_precinct_t *) = 00;

  if
    (! p_tcd)
  {
    return;
  }
  if
    (! p_tcd->tcd_image)
  {
    return;
  }
  if
    (p_tcd->m_is_decoder)
  {
    l_tcd_code_block_deallocate = tcd_code_block_dec_deallocate;
  }
  else
  {
    l_tcd_code_block_deallocate = tcd_code_block_enc_deallocate;
  }


  l_tile = p_tcd->tcd_image->tiles;
  if
    (! l_tile)
  {
    return;
  }
  l_tile_comp = l_tile->comps;

  for
    (compno = 0; compno < l_tile->numcomps; ++compno)
  {
    l_res = l_tile_comp->resolutions;
    if
      (l_res)
    {
      l_nb_resolutions = l_tile_comp->resolutions_size / sizeof(opj_tcd_resolution_t);
      for
        (resno = 0; resno < l_nb_resolutions; ++resno)
      {
        l_band = l_res->bands;
        for
          (bandno = 0; bandno < 3; ++bandno)
        {
          l_precinct = l_band->precincts;
          if
            (l_precinct)
          {
            l_nb_precincts = l_band->precincts_data_size / sizeof(opj_tcd_precinct_t);
            for
              (precno = 0; precno < l_nb_precincts; ++precno)
            {
              tgt_destroy(l_precinct->incltree);
              l_precinct->incltree = 00;
              tgt_destroy(l_precinct->imsbtree);
              l_precinct->imsbtree = 00;
              (*l_tcd_code_block_deallocate) (l_precinct);
              ++l_precinct;
            }
            opj_free(l_band->precincts);
            l_band->precincts = 00;
          }
          ++l_band;
        } /* for (resno */
        ++l_res;
      }
      opj_free(l_tile_comp->resolutions);
      l_tile_comp->resolutions = 00;
    }
    if
      (l_tile_comp->data)
    {
      opj_aligned_free(l_tile_comp->data);
      l_tile_comp->data = 00;
    }
    ++l_tile_comp;
  }
  opj_free(l_tile->comps);
  l_tile->comps = 00;
  opj_free(p_tcd->tcd_image->tiles);
  p_tcd->tcd_image->tiles = 00;
}

bool tcd_init(
             opj_tcd_t *p_tcd,
             opj_image_t * p_image,
             opj_cp_t * p_cp
             )
{
  OPJ_UINT32 l_tile_comp_size;

  p_tcd->image = p_image;
  p_tcd->cp = p_cp;
  p_tcd->tcd_image->tiles = (opj_tcd_tile_t *) opj_malloc(sizeof(opj_tcd_tile_t));

  if
    (! p_tcd->tcd_image->tiles)
  {
    return false;
  }
  memset(p_tcd->tcd_image->tiles,0, sizeof(opj_tcd_tile_t));

  l_tile_comp_size = p_image->numcomps * sizeof(opj_tcd_tilecomp_t);
  p_tcd->tcd_image->tiles->comps = (opj_tcd_tilecomp_t *) opj_malloc(l_tile_comp_size);
  if
    (! p_tcd->tcd_image->tiles->comps )
  {
    return false;
  }
  memset( p_tcd->tcd_image->tiles->comps , 0 , l_tile_comp_size);
  p_tcd->tcd_image->tiles->numcomps = p_image->numcomps;
  p_tcd->tp_pos = p_cp->m_specific_param.m_enc.m_tp_pos;
  return true;
}

void tcd_makelayer_fixed(opj_tcd_t *tcd, OPJ_UINT32 layno, OPJ_UINT32 final) {
  OPJ_UINT32 compno, resno, bandno, precno, cblkno;
  OPJ_INT32 value;      /*, matrice[tcd_tcp->numlayers][tcd_tile->comps[0].numresolutions][3]; */
  OPJ_INT32 matrice[10][10][3];
  OPJ_UINT32 i, j, k;

  opj_cp_t *cp = tcd->cp;
  opj_tcd_tile_t *tcd_tile = tcd->tcd_image->tiles;
  opj_tcp_t *tcd_tcp = tcd->tcp;

  for (compno = 0; compno < tcd_tile->numcomps; compno++) {
    opj_tcd_tilecomp_t *tilec = &tcd_tile->comps[compno];
    for (i = 0; i < tcd_tcp->numlayers; i++) {
      for (j = 0; j < tilec->numresolutions; j++) {
        for (k = 0; k < 3; k++) {
          matrice[i][j][k] =
            (OPJ_INT32) (cp->m_specific_param.m_enc.m_matrice[i * tilec->numresolutions * 3 + j * 3 + k]
            * (OPJ_FLOAT32) (tcd->image->comps[compno].prec / 16.0));
        }
      }
    }

    for (resno = 0; resno < tilec->numresolutions; resno++) {
      opj_tcd_resolution_t *res = &tilec->resolutions[resno];
      for (bandno = 0; bandno < res->numbands; bandno++) {
        opj_tcd_band_t *band = &res->bands[bandno];
        for (precno = 0; precno < res->pw * res->ph; precno++) {
          opj_tcd_precinct_t *prc = &band->precincts[precno];
          for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
            opj_tcd_cblk_enc_t *cblk = &prc->cblks.enc[cblkno];
            opj_tcd_layer_t *layer = &cblk->layers[layno];
            OPJ_UINT32 n;
            OPJ_INT32 imsb = tcd->image->comps[compno].prec - cblk->numbps;  /* number of bit-plan equal to zero */
            /* Correction of the matrix of coefficient to include the IMSB information */
            if (layno == 0) {
              value = matrice[layno][resno][bandno];
              if (imsb >= value) {
                value = 0;
              } else {
                value -= imsb;
              }
            } else {
              value =  matrice[layno][resno][bandno] -  matrice[layno - 1][resno][bandno];
              if (imsb >= matrice[layno - 1][resno][bandno]) {
                value -= (imsb - matrice[layno - 1][resno][bandno]);
                if (value < 0) {
                  value = 0;
                }
              }
            }

            if (layno == 0) {
              cblk->numpassesinlayers = 0;
            }

            n = cblk->numpassesinlayers;
            if (cblk->numpassesinlayers == 0) {
              if (value != 0) {
                n = 3 * value - 2 + cblk->numpassesinlayers;
              } else {
                n = cblk->numpassesinlayers;
              }
            } else {
              n = 3 * value + cblk->numpassesinlayers;
            }

            layer->numpasses = n - cblk->numpassesinlayers;

            if (!layer->numpasses)
              continue;

            if (cblk->numpassesinlayers == 0) {
              layer->len = cblk->passes[n - 1].rate;
              layer->data = cblk->data;
            } else {
              layer->len = cblk->passes[n - 1].rate - cblk->passes[cblk->numpassesinlayers - 1].rate;
              layer->data = cblk->data + cblk->passes[cblk->numpassesinlayers - 1].rate;
            }
            if (final)
              cblk->numpassesinlayers = n;
          }
        }
      }
    }
  }
}

void tcd_rateallocate_fixed(opj_tcd_t *tcd) {
  OPJ_UINT32 layno;
  for (layno = 0; layno < tcd->tcp->numlayers; layno++) {
    tcd_makelayer_fixed(tcd, layno, 1);
  }
}

void tcd_makelayer(opj_tcd_t *tcd, OPJ_UINT32 layno, OPJ_FLOAT64 thresh, OPJ_UINT32 final) {
  OPJ_UINT32 compno, resno, bandno, precno, cblkno;
  OPJ_UINT32 passno;

  opj_tcd_tile_t *tcd_tile = tcd->tcd_image->tiles;

  tcd_tile->distolayer[layno] = 0;  /* fixed_quality */

  for (compno = 0; compno < tcd_tile->numcomps; compno++) {
    opj_tcd_tilecomp_t *tilec = &tcd_tile->comps[compno];
    for (resno = 0; resno < tilec->numresolutions; resno++) {
      opj_tcd_resolution_t *res = &tilec->resolutions[resno];
      for (bandno = 0; bandno < res->numbands; bandno++) {
        opj_tcd_band_t *band = &res->bands[bandno];
        for (precno = 0; precno < res->pw * res->ph; precno++) {
          opj_tcd_precinct_t *prc = &band->precincts[precno];
          for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
            opj_tcd_cblk_enc_t *cblk = &prc->cblks.enc[cblkno];
            opj_tcd_layer_t *layer = &cblk->layers[layno];

            OPJ_UINT32 n;
            if (layno == 0) {
              cblk->numpassesinlayers = 0;
            }
            n = cblk->numpassesinlayers;
            for (passno = cblk->numpassesinlayers; passno < cblk->totalpasses; passno++) {
              OPJ_INT32 dr;
              OPJ_FLOAT64 dd;
              opj_tcd_pass_t *pass = &cblk->passes[passno];
              if (n == 0) {
                dr = pass->rate;
                dd = pass->distortiondec;
              } else {
                dr = pass->rate - cblk->passes[n - 1].rate;
                dd = pass->distortiondec - cblk->passes[n - 1].distortiondec;
              }
              if (!dr) {
                if (dd != 0)
                  n = passno + 1;
                continue;
              }
              if (dd / dr >= thresh)
                n = passno + 1;
            }
            layer->numpasses = n - cblk->numpassesinlayers;

            if (!layer->numpasses) {
              layer->disto = 0;
              continue;
            }
            if (cblk->numpassesinlayers == 0) {
              layer->len = cblk->passes[n - 1].rate;
              layer->data = cblk->data;
              layer->disto = cblk->passes[n - 1].distortiondec;
            } else {
              layer->len = cblk->passes[n - 1].rate -  cblk->passes[cblk->numpassesinlayers - 1].rate;
              layer->data = cblk->data + cblk->passes[cblk->numpassesinlayers - 1].rate;
              layer->disto = cblk->passes[n - 1].distortiondec - cblk->passes[cblk->numpassesinlayers - 1].distortiondec;
            }

            tcd_tile->distolayer[layno] += layer->disto;  /* fixed_quality */

            if (final)
              cblk->numpassesinlayers = n;
          }
        }
      }
    }
  }
}

bool tcd_rateallocate(opj_tcd_t *tcd, OPJ_BYTE *dest, OPJ_UINT32 * p_data_written, OPJ_UINT32 len, opj_codestream_info_t *cstr_info) {
  OPJ_UINT32 compno, resno, bandno, precno, cblkno, layno;
  OPJ_UINT32 passno;
  OPJ_FLOAT64 min, max;
  OPJ_FLOAT64 cumdisto[100];  /* fixed_quality */
  const OPJ_FLOAT64 K = 1;    /* 1.1; fixed_quality */
  OPJ_FLOAT64 maxSE = 0;

  opj_cp_t *cp = tcd->cp;
  opj_tcd_tile_t *tcd_tile = tcd->tcd_image->tiles;
  opj_tcp_t *tcd_tcp = tcd->tcp;

  min = DBL_MAX;
  max = 0;

  tcd_tile->numpix = 0;    /* fixed_quality */

  for (compno = 0; compno < tcd_tile->numcomps; compno++) {
    opj_tcd_tilecomp_t *tilec = &tcd_tile->comps[compno];
    tilec->numpix = 0;

    for (resno = 0; resno < tilec->numresolutions; resno++) {
      opj_tcd_resolution_t *res = &tilec->resolutions[resno];

      for (bandno = 0; bandno < res->numbands; bandno++) {
        opj_tcd_band_t *band = &res->bands[bandno];

        for (precno = 0; precno < res->pw * res->ph; precno++) {
          opj_tcd_precinct_t *prc = &band->precincts[precno];

          for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
            opj_tcd_cblk_enc_t *cblk = &prc->cblks.enc[cblkno];

            for (passno = 0; passno < cblk->totalpasses; passno++) {
              opj_tcd_pass_t *pass = &cblk->passes[passno];
              OPJ_INT32 dr;
              OPJ_FLOAT64 dd, rdslope;
              if (passno == 0) {
                dr = pass->rate;
                dd = pass->distortiondec;
              } else {
                dr = pass->rate - cblk->passes[passno - 1].rate;
                dd = pass->distortiondec - cblk->passes[passno - 1].distortiondec;
              }
              if (dr == 0) {
                continue;
              }
              rdslope = dd / dr;
              if (rdslope < min) {
                min = rdslope;
              }
              if (rdslope > max) {
                max = rdslope;
              }
            } /* passno */

            /* fixed_quality */
            tcd_tile->numpix += ((cblk->x1 - cblk->x0) * (cblk->y1 - cblk->y0));
            tilec->numpix += ((cblk->x1 - cblk->x0) * (cblk->y1 - cblk->y0));
          } /* cbklno */
        } /* precno */
      } /* bandno */
    } /* resno */

    maxSE += (((OPJ_FLOAT64)(1 << tcd->image->comps[compno].prec) - 1.0)
      * ((OPJ_FLOAT64)(1 << tcd->image->comps[compno].prec) -1.0))
      * ((OPJ_FLOAT64)(tilec->numpix));
  } /* compno */

  /* index file */
  if(cstr_info) {
    opj_tile_info_t *tile_info = &cstr_info->tile[tcd->tcd_tileno];
    tile_info->numpix = tcd_tile->numpix;
    tile_info->distotile = tcd_tile->distotile;
    tile_info->thresh = (OPJ_FLOAT64 *) opj_malloc(tcd_tcp->numlayers * sizeof(OPJ_FLOAT64));
  }

  for (layno = 0; layno < tcd_tcp->numlayers; layno++) {
    OPJ_FLOAT64 lo = min;
    OPJ_FLOAT64 hi = max;
    bool success = false;
    OPJ_UINT32 maxlen = tcd_tcp->rates[layno] ? uint_min(((OPJ_UINT32) ceil(tcd_tcp->rates[layno])), len) : len;
    OPJ_FLOAT64 goodthresh = 0;
    OPJ_FLOAT64 stable_thresh = 0;
    OPJ_UINT32 i;
    OPJ_FLOAT64 distotarget;    /* fixed_quality */

    /* fixed_quality */
    distotarget = tcd_tile->distotile - ((K * maxSE) / pow((OPJ_FLOAT32)10, tcd_tcp->distoratio[layno] / 10));

    /* Don't try to find an optimal threshold but rather take everything not included yet, if
      -r xx,yy,zz,0   (disto_alloc == 1 and rates == 0)
      -q xx,yy,zz,0    (fixed_quality == 1 and distoratio == 0)
      ==> possible to have some lossy layers and the last layer for sure lossless */
    if ( ((cp->m_specific_param.m_enc.m_disto_alloc==1) && (tcd_tcp->rates[layno]>0)) || ((cp->m_specific_param.m_enc.m_fixed_quality==1) && (tcd_tcp->distoratio[layno]>0))) {
      opj_t2_t *t2 = t2_create(tcd->image, cp);
      OPJ_FLOAT64 thresh = 0;
      if
        (t2 == 00)
      {
        return false;
      }

      for
        (i = 0; i < 128; ++i)
      {
        OPJ_FLOAT64 distoachieved = 0;  /* fixed_quality */
        thresh = (lo + hi) / 2;

        tcd_makelayer(tcd, layno, thresh, 0);

        if (cp->m_specific_param.m_enc.m_fixed_quality) {  /* fixed_quality */
          if(cp->m_specific_param.m_enc.m_cinema){
            if
              (! t2_encode_packets(t2,tcd->tcd_tileno, tcd_tile, layno + 1, dest, p_data_written, maxlen, cstr_info,tcd->cur_tp_num,tcd->tp_pos,tcd->cur_pino,THRESH_CALC))
            {
              lo = thresh;
              continue;
            }
            else
            {
               distoachieved =  layno == 0 ?
              tcd_tile->distolayer[0]  : cumdisto[layno - 1] + tcd_tile->distolayer[layno];
              if (distoachieved < distotarget) {
                hi=thresh;
                stable_thresh = thresh;
                continue;
              }else{
                lo=thresh;
              }
            }
          }else{
            distoachieved =  (layno == 0) ?
              tcd_tile->distolayer[0]  : (cumdisto[layno - 1] + tcd_tile->distolayer[layno]);
            if (distoachieved < distotarget) {
              hi = thresh;
              stable_thresh = thresh;
              continue;
            }
            lo = thresh;
          }
        } else {
          if
            (! t2_encode_packets(t2, tcd->tcd_tileno, tcd_tile, layno + 1, dest,p_data_written, maxlen, cstr_info,tcd->cur_tp_num,tcd->tp_pos,tcd->cur_pino,THRESH_CALC))
          {
            /* TODO: what to do with l ??? seek / tell ??? */
            /* opj_event_msg(tcd->cinfo, EVT_INFO, "rate alloc: len=%d, max=%d\n", l, maxlen); */
            lo = thresh;
            continue;
          }
          hi = thresh;
          stable_thresh = thresh;
        }
      }
      success = true;
      goodthresh = stable_thresh == 0? thresh : stable_thresh;
      t2_destroy(t2);
    } else {
      success = true;
      goodthresh = min;
    }

    if (!success) {
      return false;
    }

    if(cstr_info) {  /* Threshold for Marcela Index */
      cstr_info->tile[tcd->tcd_tileno].thresh[layno] = goodthresh;
    }
    tcd_makelayer(tcd, layno, goodthresh, 1);

    /* fixed_quality */
    cumdisto[layno] = (layno == 0) ? tcd_tile->distolayer[0] : (cumdisto[layno - 1] + tcd_tile->distolayer[layno]);
  }

  return true;
}

OPJ_UINT32 tcd_get_encoded_tile_size (
             opj_tcd_t *p_tcd
             )
{
  OPJ_UINT32 i,l_data_size = 0;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcd_tilecomp_t * l_tilec = 00;
  OPJ_UINT32 l_size_comp, l_remaining;

  l_tilec = p_tcd->tcd_image->tiles->comps;
  l_img_comp = p_tcd->image->comps;
  for
    (i=0;i<p_tcd->image->numcomps;++i)
  {
    l_size_comp = l_img_comp->prec >> 3; /*(/ 8)*/
    l_remaining = l_img_comp->prec & 7;  /* (%8) */
    if
      (l_remaining)
    {
      ++l_size_comp;
    }
    if
      (l_size_comp == 3)
    {
      l_size_comp = 4;
    }
    l_data_size += l_size_comp * (l_tilec->x1 - l_tilec->x0) * (l_tilec->y1 - l_tilec->y0);
    ++l_img_comp;
    ++l_tilec;
  }
  return l_data_size;
}

bool tcd_copy_tile_data (
             opj_tcd_t *p_tcd,
             OPJ_BYTE * p_src,
             OPJ_UINT32 p_src_length
             )
{
  OPJ_UINT32 i,j,l_data_size = 0;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcd_tilecomp_t * l_tilec = 00;
  OPJ_UINT32 l_size_comp, l_remaining;
  OPJ_UINT32 l_nb_elem;

  l_data_size = tcd_get_encoded_tile_size(p_tcd);
  if
    (l_data_size != p_src_length)
  {
    return false;
  }
  l_tilec = p_tcd->tcd_image->tiles->comps;
  l_img_comp = p_tcd->image->comps;
  for
    (i=0;i<p_tcd->image->numcomps;++i)
  {
    l_size_comp = l_img_comp->prec >> 3; /*(/ 8)*/
    l_remaining = l_img_comp->prec & 7;  /* (%8) */
    l_nb_elem = (l_tilec->x1 - l_tilec->x0) * (l_tilec->y1 - l_tilec->y0);
    if
      (l_remaining)
    {
      ++l_size_comp;
    }
    if
      (l_size_comp == 3)
    {
      l_size_comp = 4;
    }
    switch
      (l_size_comp)
    {
      case 1:
        {
          OPJ_CHAR * l_src_ptr = (OPJ_CHAR *) p_src;
          OPJ_INT32 * l_dest_ptr = l_tilec->data;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_nb_elem;++j)
            {
              *(l_dest_ptr++) = (OPJ_INT32) (*(l_src_ptr++));
            }
          }
          else
          {
            for
              (j=0;j<l_nb_elem;++j)
            {
              *(l_dest_ptr++) = (*(l_src_ptr++))&0xff;
            }
          }
          p_src = (OPJ_BYTE*) l_src_ptr;
        }
        break;
      case 2:
        {
          OPJ_INT32 * l_dest_ptr = l_tilec->data;
          OPJ_INT16 * l_src_ptr = (OPJ_INT16 *) p_src;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_nb_elem;++j)
            {
              *(l_dest_ptr++) = (OPJ_INT32) (*(l_src_ptr++));
            }
          }
          else
          {
            for
              (j=0;j<l_nb_elem;++j)
            {
              *(l_dest_ptr++) = (*(l_src_ptr++))&0xffff;
            }

          }
          p_src = (OPJ_BYTE*) l_src_ptr;
        }
        break;
      case 4:
        {
          OPJ_INT32 * l_src_ptr = (OPJ_INT32 *) p_src;
          OPJ_INT32 * l_dest_ptr = l_tilec->data;
          for
            (j=0;j<l_nb_elem;++j)
          {
            *(l_dest_ptr++) = (OPJ_INT32) (*(l_src_ptr++));
          }
          p_src = (OPJ_BYTE*) l_src_ptr;
        }
        break;
    }
    ++l_img_comp;
    ++l_tilec;
  }
  return true;
}

bool tcd_update_tile_data (
             opj_tcd_t *p_tcd,
             OPJ_BYTE * p_dest,
             OPJ_UINT32 p_dest_length
             )
{
  OPJ_UINT32 i,j,k,l_data_size = 0;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcd_tilecomp_t * l_tilec = 00;
  opj_tcd_resolution_t * l_res;
  OPJ_UINT32 l_size_comp, l_remaining;
  OPJ_UINT32 l_stride, l_width,l_height;

  l_data_size = tcd_get_decoded_tile_size(p_tcd);
  if
    (l_data_size > p_dest_length)
  {
    return false;
  }

  l_tilec = p_tcd->tcd_image->tiles->comps;
  l_img_comp = p_tcd->image->comps;
  for
    (i=0;i<p_tcd->image->numcomps;++i)
  {
    l_size_comp = l_img_comp->prec >> 3; /*(/ 8)*/
    l_remaining = l_img_comp->prec & 7;  /* (%8) */
    l_res = l_tilec->resolutions + l_img_comp->resno_decoded;
    l_width = (l_res->x1 - l_res->x0);
    l_height = (l_res->y1 - l_res->y0);
    l_stride = (l_tilec->x1 - l_tilec->x0) - l_width;
    if
      (l_remaining)
    {
      ++l_size_comp;
    }
    if
      (l_size_comp == 3)
    {
      l_size_comp = 4;
    }
    switch
      (l_size_comp)
    {
      case 1:
        {
          OPJ_CHAR * l_dest_ptr = (OPJ_CHAR *) p_dest;
          const OPJ_INT32 * l_src_ptr = l_tilec->data;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (OPJ_CHAR) (*(l_src_ptr++));
              }
              l_src_ptr += l_stride;
            }
          }
          else
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (OPJ_BYTE) ((*(l_src_ptr++))&0xff);
              }
              l_src_ptr += l_stride;
            }
          }
          p_dest = (OPJ_BYTE *)l_dest_ptr;

        }
        break;
      case 2:
        {
          const OPJ_INT32 * l_src_ptr = l_tilec->data;
          OPJ_INT16 * l_dest_ptr = (OPJ_INT16 *) p_dest;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (OPJ_INT16) (*(l_src_ptr++));
              }
              l_src_ptr += l_stride;
            }
          }
          else
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (OPJ_UINT16) ((*(l_src_ptr++))&0xffff);
              }
              l_src_ptr += l_stride;
            }
          }
          p_dest = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
      case 4:
        {
          OPJ_INT32 * l_dest_ptr = (OPJ_INT32 *) p_dest;
          OPJ_INT32 * l_src_ptr = l_tilec->data;
          for
            (j=0;j<l_height;++j)
          {
            for
              (k=0;k<l_width;++k)
            {
              *(l_dest_ptr++) = (*(l_src_ptr++));
            }
            l_src_ptr += l_stride;
          }
          p_dest = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
    }
    ++l_img_comp;
    ++l_tilec;
  }
  return true;
}

OPJ_UINT32 tcd_get_decoded_tile_size (
             opj_tcd_t *p_tcd
             )
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_data_size = 0;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcd_tilecomp_t * l_tile_comp = 00;
  opj_tcd_resolution_t * l_res = 00;
  OPJ_UINT32 l_size_comp, l_remaining;

  l_tile_comp = p_tcd->tcd_image->tiles->comps;
  l_img_comp = p_tcd->image->comps;
  for
    (i=0;i<p_tcd->image->numcomps;++i)
  {
    l_size_comp = l_img_comp->prec >> 3; /*(/ 8)*/
    l_remaining = l_img_comp->prec & 7;  /* (%8) */
    if
      (l_remaining)
    {
      ++l_size_comp;
    }
    if
      (l_size_comp == 3)
    {
      l_size_comp = 4;
    }
    l_res = l_tile_comp->resolutions + l_tile_comp->minimum_num_resolutions - 1;
    l_data_size += l_size_comp * (l_res->x1 - l_res->x0) * (l_res->y1 - l_res->y0);
    ++l_img_comp;
    ++l_tile_comp;
  }
  return l_data_size;
}

bool tcd_dc_level_shift_encode (
             opj_tcd_t *p_tcd
             )
{
  OPJ_UINT32 compno;
  opj_tcd_tilecomp_t * l_tile_comp = 00;
  opj_tccp_t * l_tccp = 00;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcp_t * l_tcp = 00;
  opj_tcd_tile_t * l_tile;
  OPJ_UINT32 l_nb_elem,i;
  OPJ_INT32 * l_current_ptr;

  l_tile = p_tcd->tcd_image->tiles;
  l_tile_comp = l_tile->comps;
  l_tcp = p_tcd->tcp;
  l_tccp = p_tcd->tcp->tccps;
  l_img_comp = p_tcd->image->comps;
  for
    (compno = 0; compno < l_tile->numcomps; compno++)
  {
    l_current_ptr = l_tile_comp->data;
    l_nb_elem = (l_tile_comp->x1 - l_tile_comp->x0) * (l_tile_comp->y1 - l_tile_comp->y0);
    if
      (l_tccp->qmfbid == 1)
    {
      for
        (i = 0; i < l_nb_elem; ++i)
      {
        *l_current_ptr -= l_tccp->m_dc_level_shift ;
        ++l_current_ptr;
      }
    }
    else
    {
      for
        (i = 0; i < l_nb_elem; ++i)
      {
        *l_current_ptr = (*l_current_ptr - l_tccp->m_dc_level_shift) << 11 ;
        ++l_current_ptr;
      }
    }
    ++l_img_comp;
    ++l_tccp;
    ++l_tile_comp;
  }
  return true;
}

bool tcd_mct_encode (
           opj_tcd_t *p_tcd
           )
{
  opj_tcd_tile_t * l_tile = p_tcd->tcd_image->tiles;
  opj_tcd_tilecomp_t * l_tile_comp = p_tcd->tcd_image->tiles->comps;
  OPJ_UINT32 samples = (l_tile_comp->x1 - l_tile_comp->x0) * (l_tile_comp->y1 - l_tile_comp->y0);
  OPJ_UINT32 i;
  OPJ_BYTE ** l_data = 00;
  opj_tcp_t * l_tcp = p_tcd->tcp;
  if
    (!p_tcd->tcp->mct)
  {
    return true;
  }

  if
    (p_tcd->tcp->mct == 2)
  {
    if
      (! p_tcd->tcp->m_mct_coding_matrix)
    {
      return true;
    }
        l_data = (OPJ_BYTE **) opj_malloc(l_tile->numcomps*sizeof(OPJ_BYTE*));
    if
      (! l_data)
    {
      return false;
    }
    for
      (i=0;i<l_tile->numcomps;++i)
    {
      l_data[i] = (OPJ_BYTE*) l_tile_comp->data;
      ++l_tile_comp;
    }
    if
      (! mct_encode_custom(// MCT data
          (OPJ_BYTE*) p_tcd->tcp->m_mct_coding_matrix,
          // size of components
          samples,
          // components
          l_data,
          // nb of components (i.e. size of pData)
          l_tile->numcomps,
          // tells if the data is signed
          p_tcd->image->comps->sgnd)
      )
    {
            opj_free(l_data);
      return false;
    }
    opj_free(l_data);
  }
  else if (l_tcp->tccps->qmfbid == 0)
  {
    mct_encode_real(l_tile->comps[0].data, l_tile->comps[1].data, l_tile->comps[2].data, samples);
  }
  else
  {
    mct_encode(l_tile->comps[0].data, l_tile->comps[1].data, l_tile->comps[2].data, samples);
  }
  return true;
}

bool tcd_dwt_encode (
            opj_tcd_t *p_tcd
           )
{
  opj_tcd_tile_t * l_tile = p_tcd->tcd_image->tiles;
  opj_tcd_tilecomp_t * l_tile_comp = p_tcd->tcd_image->tiles->comps;
  opj_tccp_t * l_tccp = p_tcd->tcp->tccps;
  OPJ_UINT32 compno;


   for
     (compno = 0; compno < l_tile->numcomps; ++compno)
  {
    if
      (l_tccp->qmfbid == 1)
    {
      if
        (! dwt_encode(l_tile_comp))
      {
        return false;
      }
    }
    else if
      (l_tccp->qmfbid == 0)
    {
      if
        (! dwt_encode_real(l_tile_comp))
      {
        return false;
      }
    }
    ++l_tile_comp;
    ++l_tccp;
  }
  return true;
}

bool tcd_t1_encode (
            opj_tcd_t *p_tcd
           )
{
  opj_t1_t * l_t1;
  const OPJ_FLOAT64 * l_mct_norms;
  opj_tcp_t * l_tcp = p_tcd->tcp;

  l_t1 = t1_create();
  if
    (l_t1 == 00)
  {
    return false;
  }
  if
    (l_tcp->mct == 1)
  {
    // irreversible encoding
    if
      (l_tcp->tccps->qmfbid == 0)
    {
      l_mct_norms = get_mct_norms_real();
    }
    else
    {
      l_mct_norms = get_mct_norms();
    }
  }
  else
  {
    l_mct_norms = (const OPJ_FLOAT64 *) (l_tcp->mct_norms);
  }

  if
    (! t1_encode_cblks(l_t1, p_tcd->tcd_image->tiles , l_tcp, l_mct_norms))
  {
        t1_destroy(l_t1);
    return false;
  }
  t1_destroy(l_t1);
  return true;
}

bool tcd_t2_encode (
          opj_tcd_t *p_tcd,
          OPJ_BYTE * p_dest_data,
          OPJ_UINT32 * p_data_written,
          OPJ_UINT32 p_max_dest_size,
          opj_codestream_info_t *p_cstr_info
          )
{
  opj_t2_t * l_t2;

  l_t2 = t2_create(p_tcd->image, p_tcd->cp);
  if
    (l_t2 == 00)
  {
    return false;
  }

  if
    (! t2_encode_packets(
          l_t2,
          p_tcd->tcd_tileno,
          p_tcd->tcd_image->tiles,
          p_tcd->tcp->numlayers,
          p_dest_data,
          p_data_written,
          p_max_dest_size,
          p_cstr_info,
          p_tcd->tp_num,
          p_tcd->tp_pos,
          p_tcd->cur_pino,
          FINAL_PASS))
  {
    t2_destroy(l_t2);
    return false;
  }
  t2_destroy(l_t2);

  /*---------------CLEAN-------------------*/
  return true;
}

bool tcd_rate_allocate_encode(
            opj_tcd_t *p_tcd,
            OPJ_BYTE * p_dest_data,
            OPJ_UINT32 p_max_dest_size,
            opj_codestream_info_t *p_cstr_info
            )
{
  opj_cp_t * l_cp = p_tcd->cp;
  OPJ_UINT32 l_nb_written = 0;

  if
    (p_cstr_info)
  {
    p_cstr_info->index_write = 0;
  }
  if
    (l_cp->m_specific_param.m_enc.m_disto_alloc|| l_cp->m_specific_param.m_enc.m_fixed_quality)
  {  /* fixed_quality */
    /* Normal Rate/distortion allocation */
    if
      (! tcd_rateallocate(p_tcd, p_dest_data,&l_nb_written, p_max_dest_size, p_cstr_info))
    {
      return false;
    }
  }
  else
  {
    /* Fixed layer allocation */
    tcd_rateallocate_fixed(p_tcd);
  }
  return true;
}

bool tcd_t2_decode (
          opj_tcd_t *p_tcd,
          OPJ_BYTE * p_src_data,
          OPJ_UINT32 * p_data_read,
          OPJ_UINT32 p_max_src_size,
          opj_codestream_info_t *p_cstr_info
          )
{
  opj_t2_t * l_t2;

  l_t2 = t2_create(p_tcd->image, p_tcd->cp);
  if
    (l_t2 == 00)
  {
    return false;
  }

  if
    (! t2_decode_packets(
          l_t2,
          p_tcd->tcd_tileno,
          p_tcd->tcd_image->tiles,
          p_src_data,
          p_data_read,
          p_max_src_size,
          p_cstr_info))
  {
    t2_destroy(l_t2);
    return false;
  }
  t2_destroy(l_t2);

  /*---------------CLEAN-------------------*/
  return true;
}

bool tcd_t1_decode (
            opj_tcd_t *p_tcd
           )
{
  OPJ_UINT32 compno;
  opj_t1_t * l_t1;
  opj_tcd_tile_t * l_tile = p_tcd->tcd_image->tiles;
  opj_tcd_tilecomp_t* l_tile_comp = l_tile->comps;
  opj_tccp_t * l_tccp = p_tcd->tcp->tccps;


  l_t1 = t1_create();
  if
    (l_t1 == 00)
  {
    return false;
  }
  for
    (compno = 0; compno < l_tile->numcomps; ++compno)
  {
    /* The +3 is headroom required by the vectorized DWT */
    t1_decode_cblks(l_t1, l_tile_comp, l_tccp);
    ++l_tile_comp;
    ++l_tccp;
  }
  t1_destroy(l_t1);
  return true;
}

bool tcd_dwt_decode (
            opj_tcd_t *p_tcd
           )
{
  OPJ_UINT32 compno;
  opj_tcd_tile_t * l_tile = p_tcd->tcd_image->tiles;
  opj_tcd_tilecomp_t * l_tile_comp = l_tile->comps;
  opj_tccp_t * l_tccp = p_tcd->tcp->tccps;
  opj_image_comp_t * l_img_comp = p_tcd->image->comps;

  for
    (compno = 0; compno < l_tile->numcomps; compno++)
  {
    /*
    if (tcd->cp->reduce != 0) {
      tcd->image->comps[compno].resno_decoded =
        tile->comps[compno].numresolutions - tcd->cp->reduce - 1;
      if (tcd->image->comps[compno].resno_decoded < 0)
      {
        return false;
      }
    }
    numres2decode = tcd->image->comps[compno].resno_decoded + 1;
    if(numres2decode > 0){
    */
    if
      (l_tccp->qmfbid == 1)
    {
      if
        (! dwt_decode(l_tile_comp, l_img_comp->resno_decoded+1))
      {
        return false;
      }
    }
    else
    {
      if
        (! dwt_decode_real(l_tile_comp, l_img_comp->resno_decoded+1))
      {
        return false;
      }
    }
    ++l_tile_comp;
    ++l_img_comp;
    ++l_tccp;
  }
  return true;
}
bool tcd_mct_decode (
            opj_tcd_t *p_tcd
           )
{
  opj_tcd_tile_t * l_tile = p_tcd->tcd_image->tiles;
  opj_tcp_t * l_tcp = p_tcd->tcp;
  opj_tcd_tilecomp_t * l_tile_comp = l_tile->comps;
  OPJ_UINT32 l_samples,i;

  if
    (! l_tcp->mct)
  {
    return true;
  }
  l_samples = (l_tile_comp->x1 - l_tile_comp->x0) * (l_tile_comp->y1 - l_tile_comp->y0);
  if
    (l_tcp->mct == 2)
  {
    OPJ_BYTE ** l_data;
    if
      (! l_tcp->m_mct_decoding_matrix)
    {
      return true;
    }
    l_data = (OPJ_BYTE **) opj_malloc(l_tile->numcomps*sizeof(OPJ_BYTE*));
    if
      (! l_data)
    {
      return false;
    }
    for
      (i=0;i<l_tile->numcomps;++i)
    {
      l_data[i] = (OPJ_BYTE*) l_tile_comp->data;
      ++l_tile_comp;
    }
    if
      (! mct_decode_custom(  // MCT data
                (OPJ_BYTE*) l_tcp->m_mct_decoding_matrix,
                // size of components
                l_samples,
                // components
                l_data,
                // nb of components (i.e. size of pData)
                l_tile->numcomps,
                // tells if the data is signed
                p_tcd->image->comps->sgnd))
    {
      opj_free(l_data);
      return false;
    }
    opj_free(l_data);
  }
  else
  {
    if
      (l_tcp->tccps->qmfbid == 1)
    {
      mct_decode(
          l_tile->comps[0].data,
          l_tile->comps[1].data,
          l_tile->comps[2].data,
          l_samples);
    }
    else
    {
      mct_decode_real(
          (float*)l_tile->comps[0].data,
          (float*)l_tile->comps[1].data,
          (float*)l_tile->comps[2].data,
          l_samples);
    }
  }
  return true;
}

bool tcd_dc_level_shift_decode (
             opj_tcd_t *p_tcd
             )
{
  OPJ_UINT32 compno;
  opj_tcd_tilecomp_t * l_tile_comp = 00;
  opj_tccp_t * l_tccp = 00;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcd_resolution_t* l_res = 00;
  opj_tcp_t * l_tcp = 00;
  opj_tcd_tile_t * l_tile;
  OPJ_UINT32 l_width,l_height,i,j;
  OPJ_INT32 * l_current_ptr;
  OPJ_INT32 l_min, l_max;
  OPJ_UINT32 l_stride;

  l_tile = p_tcd->tcd_image->tiles;
  l_tile_comp = l_tile->comps;
  l_tcp = p_tcd->tcp;
  l_tccp = p_tcd->tcp->tccps;
  l_img_comp = p_tcd->image->comps;

  for
    (compno = 0; compno < l_tile->numcomps; compno++)
  {
    l_res = l_tile_comp->resolutions + l_img_comp->resno_decoded;
    l_width = (l_res->x1 - l_res->x0);
    l_height = (l_res->y1 - l_res->y0);
    l_stride = (l_tile_comp->x1 - l_tile_comp->x0) - l_width;
    if
      (l_img_comp->sgnd)
    {
      l_min = -(1 << (l_img_comp->prec - 1));
      l_max = (1 << (l_img_comp->prec - 1)) - 1;
    }
    else
    {
            l_min = 0;
      l_max = (1 << l_img_comp->prec) - 1;
    }
    l_current_ptr = l_tile_comp->data;
    if
      (l_tccp->qmfbid == 1)
    {
      for
        (j=0;j<l_height;++j)
      {
        for
          (i = 0; i < l_width; ++i)
        {
          *l_current_ptr = int_clamp(*l_current_ptr + l_tccp->m_dc_level_shift, l_min, l_max);
          ++l_current_ptr;
        }
        l_current_ptr += l_stride;
      }
    }
    else
    {
      for
        (j=0;j<l_height;++j)
      {
        for
          (i = 0; i < l_width; ++i)
        {
          OPJ_FLOAT32 l_value = *((OPJ_FLOAT32 *) l_current_ptr);
          *l_current_ptr = int_clamp(lrintf(l_value) + l_tccp->m_dc_level_shift, l_min, l_max); ;
          ++l_current_ptr;
        }
        l_current_ptr += l_stride;
      }
    }
    ++l_img_comp;
    ++l_tccp;
    ++l_tile_comp;
  }
  return true;
}

bool tcd_encode_tile(
           opj_tcd_t *p_tcd,
           OPJ_UINT32 p_tile_no,
           OPJ_BYTE *p_dest,
           OPJ_UINT32 * p_data_written,
           OPJ_UINT32 p_max_length,
           opj_codestream_info_t *p_cstr_info)
{

  if
    (p_tcd->cur_tp_num == 0)
  {
    p_tcd->tcd_tileno = p_tile_no;
    p_tcd->tcp = &p_tcd->cp->tcps[p_tile_no];
    /* INDEX >> "Precinct_nb_X et Precinct_nb_Y" */
    if(p_cstr_info)
    {
      OPJ_UINT32 l_num_packs = 0;
      OPJ_UINT32 i;
      opj_tcd_tilecomp_t *l_tilec_idx = &p_tcd->tcd_image->tiles->comps[0];  /* based on component 0 */
      opj_tccp_t *l_tccp = p_tcd->tcp->tccps;  /* based on component 0 */
      for (i = 0; i < l_tilec_idx->numresolutions; i++) {
        opj_tcd_resolution_t *l_res_idx = &l_tilec_idx->resolutions[i];

        p_cstr_info->tile[p_tile_no].pw[i] = l_res_idx->pw;
        p_cstr_info->tile[p_tile_no].ph[i] = l_res_idx->ph;

        l_num_packs += l_res_idx->pw * l_res_idx->ph;
        p_cstr_info->tile[p_tile_no].pdx[i] = l_tccp->prcw[i];
        p_cstr_info->tile[p_tile_no].pdy[i] = l_tccp->prch[i];
      }
      p_cstr_info->tile[p_tile_no].packet = (opj_packet_info_t*) opj_calloc(p_cstr_info->numcomps * p_cstr_info->numlayers * l_num_packs, sizeof(opj_packet_info_t));
    }
    /* << INDEX */
    _ProfStart(PGROUP_DC_SHIFT);
    /*---------------TILE-------------------*/
    if
      (! tcd_dc_level_shift_encode(p_tcd))
    {
      return false;
    }
    _ProfStop(PGROUP_DC_SHIFT);

    _ProfStart(PGROUP_MCT);
    if
      (! tcd_mct_encode(p_tcd))
    {
      return false;
    }
    _ProfStop(PGROUP_MCT);

    _ProfStart(PGROUP_DWT);
    if
      (! tcd_dwt_encode(p_tcd))
    {
      return false;
    }
    _ProfStop(PGROUP_DWT);

    _ProfStart(PGROUP_T1);
    if
      (! tcd_t1_encode(p_tcd))
    {
      return false;
    }
    _ProfStop(PGROUP_T1);

    _ProfStart(PGROUP_RATE);
    if
      (! tcd_rate_allocate_encode(p_tcd,p_dest,p_max_length,p_cstr_info))
    {
      return false;
    }
    _ProfStop(PGROUP_RATE);

  }
  /*--------------TIER2------------------*/

  /* INDEX */
  if
    (p_cstr_info)
  {
    p_cstr_info->index_write = 1;
  }
  _ProfStart(PGROUP_T2);
  if
    (! tcd_t2_encode(p_tcd,p_dest,p_data_written,p_max_length,p_cstr_info))
  {
    return false;
  }
  _ProfStop(PGROUP_T2);
  /*---------------CLEAN-------------------*/
  return true;
}

bool tcd_decode_tile(
           opj_tcd_t *p_tcd,
           OPJ_BYTE *p_src,
           OPJ_UINT32 p_max_length,
           OPJ_UINT32 p_tile_no,
           opj_codestream_info_t *p_cstr_info)
{
  OPJ_UINT32 l_data_read;
  p_tcd->tcd_tileno = p_tile_no;
  p_tcd->tcp = &(p_tcd->cp->tcps[p_tile_no]);

  /* INDEX >>  */
  if(p_cstr_info) {
    OPJ_UINT32 resno, compno, numprec = 0;
    for (compno = 0; compno < (OPJ_UINT32) p_cstr_info->numcomps; compno++) {
      opj_tcp_t *tcp = &p_tcd->cp->tcps[0];
      opj_tccp_t *tccp = &tcp->tccps[compno];
      opj_tcd_tilecomp_t *tilec_idx = &p_tcd->tcd_image->tiles->comps[compno];
      for (resno = 0; resno < tilec_idx->numresolutions; resno++) {
        opj_tcd_resolution_t *res_idx = &tilec_idx->resolutions[resno];
        p_cstr_info->tile[p_tile_no].pw[resno] = res_idx->pw;
        p_cstr_info->tile[p_tile_no].ph[resno] = res_idx->ph;
        numprec += res_idx->pw * res_idx->ph;
        p_cstr_info->tile[p_tile_no].pdx[resno] = tccp->prcw[resno];
        p_cstr_info->tile[p_tile_no].pdy[resno] = tccp->prch[resno];
      }
    }
    p_cstr_info->tile[p_tile_no].packet = (opj_packet_info_t *) opj_malloc(p_cstr_info->numlayers * numprec * sizeof(opj_packet_info_t));
    p_cstr_info->packno = 0;
  }
  /* << INDEX */

  /*--------------TIER2------------------*/
  _ProfStart(PGROUP_T2);
  l_data_read = 0;
  if
    (! tcd_t2_decode(p_tcd,p_src,&l_data_read,p_max_length,p_cstr_info))
  {
    return false;
  }
  _ProfStop(PGROUP_T2);

  /*------------------TIER1-----------------*/

  _ProfStart(PGROUP_T1);
  if
    (! tcd_t1_decode(p_tcd))
  {
    return false;
  }
  _ProfStop(PGROUP_T1);

  /*----------------DWT---------------------*/

  _ProfStart(PGROUP_DWT);
  if
    (! tcd_dwt_decode(p_tcd))
  {
    return false;
  }
  _ProfStop(PGROUP_DWT);

  /*----------------MCT-------------------*/
  _ProfStart(PGROUP_MCT);
  if
    (! tcd_mct_decode(p_tcd))
  {
    return false;
  }
  _ProfStop(PGROUP_MCT);

  _ProfStart(PGROUP_DC_SHIFT);
  if
    (! tcd_dc_level_shift_decode(p_tcd))
  {
    return false;
  }
  _ProfStop(PGROUP_DC_SHIFT);


  /*---------------TILE-------------------*/
  return true;
}
