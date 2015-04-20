/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
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
#include "openjpeg.h"
#include "opj_includes.h"
#include "t2.h"
#include "bio.h"
#include "tcd.h"
#include "pi.h"
#include "event.h"
#include "j2k.h"
#include "tgt.h"
#include "int.h"
#include "opj_malloc.h"
#include "pi.h"


/** @defgroup T2 T2 - Implementation of a tier-2 coding */
/*@{*/

/** @name Local static functions */
/*@{*/

static void t2_putcommacode(opj_bio_t *bio, OPJ_UINT32 n);
static OPJ_UINT32 t2_getcommacode(opj_bio_t *bio);
/**
Variable length code for signalling delta Zil (truncation point)
@param bio Bit Input/Output component
@param n delta Zil
*/
static void t2_putnumpasses(opj_bio_t *bio, OPJ_UINT32 n);
static OPJ_UINT32 t2_getnumpasses(opj_bio_t *bio);
/**
Encode a packet of a tile to a destination buffer
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@param dest Destination buffer
@param len Length of the destination buffer
@param cstr_info Codestream information structure
@param tileno Number of the tile encoded
@return
*/
static bool t2_encode_packet(
               OPJ_UINT32 tileno,
               opj_tcd_tile_t *tile,
               opj_tcp_t *tcp,
               opj_pi_iterator_t *pi,
               OPJ_BYTE *dest,
               OPJ_UINT32 * p_data_written,
               OPJ_UINT32 len,
               opj_codestream_info_t *cstr_info);
/**
@param seg
@param cblksty
@param first
*/
static bool t2_init_seg(opj_tcd_cblk_dec_t* cblk, OPJ_UINT32 index, OPJ_UINT32 cblksty, OPJ_UINT32 first);
/**
Decode a packet of a tile from a source buffer
@param t2 T2 handle
@param src Source buffer
@param len Length of the source buffer
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@return
*/
static bool t2_decode_packet(
               opj_t2_t* p_t2,
               opj_tcd_tile_t *p_tile,
                             opj_tcp_t *p_tcp,
               opj_pi_iterator_t *p_pi,
               OPJ_BYTE *p_src,
               OPJ_UINT32 * p_data_read,
               OPJ_UINT32 p_max_length,
               opj_packet_info_t *p_pack_info);

/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */

/* #define RESTART 0x04 */

static void t2_putcommacode(opj_bio_t *bio, OPJ_UINT32 n) {
  while
    (--n != -1)
  {
    bio_write(bio, 1, 1);
  }
  bio_write(bio, 0, 1);
}

static OPJ_UINT32 t2_getcommacode(opj_bio_t *bio) {
  OPJ_UINT32 n = 0;
  while
    (bio_read(bio, 1))
  {
    ++n;
  }
  return n;
}

static void t2_putnumpasses(opj_bio_t *bio, OPJ_UINT32 n) {
  if (n == 1) {
    bio_write(bio, 0, 1);
  } else if (n == 2) {
    bio_write(bio, 2, 2);
  } else if (n <= 5) {
    bio_write(bio, 0xc | (n - 3), 4);
  } else if (n <= 36) {
    bio_write(bio, 0x1e0 | (n - 6), 9);
  } else if (n <= 164) {
    bio_write(bio, 0xff80 | (n - 37), 16);
  }
}

static OPJ_UINT32 t2_getnumpasses(opj_bio_t *bio) {
  OPJ_UINT32 n;
  if (!bio_read(bio, 1))
    return 1;
  if (!bio_read(bio, 1))
    return 2;
  if ((n = bio_read(bio, 2)) != 3)
    return (3 + n);
  if ((n = bio_read(bio, 5)) != 31)
    return (6 + n);
  return (37 + bio_read(bio, 7));
}

static bool t2_encode_packet(
               OPJ_UINT32 tileno,
               opj_tcd_tile_t * tile,
               opj_tcp_t * tcp,
               opj_pi_iterator_t *pi,
               OPJ_BYTE *dest,
               OPJ_UINT32 * p_data_written,
               OPJ_UINT32 length,
               opj_codestream_info_t *cstr_info)
{
  OPJ_UINT32 bandno, cblkno;
  OPJ_BYTE *c = dest;
  OPJ_UINT32 l_nb_bytes;
  OPJ_UINT32 compno = pi->compno;  /* component value */
  OPJ_UINT32 resno  = pi->resno;    /* resolution level value */
  OPJ_UINT32 precno = pi->precno;  /* precinct value */
  OPJ_UINT32 layno  = pi->layno;    /* quality layer value */
  OPJ_UINT32 l_nb_blocks;
  opj_tcd_band_t *band = 00;
  opj_tcd_cblk_enc_t* cblk = 00;
  opj_tcd_pass_t *pass = 00;

  opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
  opj_tcd_resolution_t *res = &tilec->resolutions[resno];

  opj_bio_t *bio = 00;  /* BIO component */

  /* <SOP 0xff91> */
  if (tcp->csty & J2K_CP_CSTY_SOP) {
    c[0] = 255;
    c[1] = 145;
    c[2] = 0;
    c[3] = 4;
    c[4] = (tile->packno % 65536) / 256;
    c[5] = (tile->packno % 65536) % 256;
    c += 6;
    length -= 6;
  }
  /* </SOP> */

  if (!layno) {
    band = res->bands;
    for
      (bandno = 0; bandno < res->numbands; ++bandno)
    {
      opj_tcd_precinct_t *prc = &band->precincts[precno];
      tgt_reset(prc->incltree);
      tgt_reset(prc->imsbtree);
      l_nb_blocks = prc->cw * prc->ch;
      for
        (cblkno = 0; cblkno < l_nb_blocks; ++cblkno)
      {
        opj_tcd_cblk_enc_t* cblk = &prc->cblks.enc[cblkno];
        cblk->numpasses = 0;
        tgt_setvalue(prc->imsbtree, cblkno, band->numbps - cblk->numbps);
      }
      ++band;
    }
  }

  bio = bio_create();
  bio_init_enc(bio, c, length);
  bio_write(bio, 1, 1);    /* Empty header bit */

  /* Writing Packet header */
  band = res->bands;
  for
    (bandno = 0; bandno < res->numbands; ++bandno)
  {
    opj_tcd_precinct_t *prc = &band->precincts[precno];
    l_nb_blocks = prc->cw * prc->ch;
    cblk = prc->cblks.enc;
    for (cblkno = 0; cblkno < l_nb_blocks; ++cblkno)
    {
      opj_tcd_layer_t *layer = &cblk->layers[layno];
      if
        (!cblk->numpasses && layer->numpasses)
      {
        tgt_setvalue(prc->incltree, cblkno, layno);
      }
      ++cblk;
    }
    cblk = prc->cblks.enc;
    for
      (cblkno = 0; cblkno < l_nb_blocks; cblkno++)
    {
      opj_tcd_layer_t *layer = &cblk->layers[layno];
      OPJ_UINT32 increment = 0;
      OPJ_UINT32 nump = 0;
      OPJ_UINT32 len = 0, passno;
      OPJ_UINT32 l_nb_passes;
      /* cblk inclusion bits */
      if (!cblk->numpasses) {
        tgt_encode(bio, prc->incltree, cblkno, layno + 1);
      } else {
        bio_write(bio, layer->numpasses != 0, 1);
      }
      /* if cblk not included, go to the next cblk  */
      if
        (!layer->numpasses)
      {
        ++cblk;
        continue;
      }
      /* if first instance of cblk --> zero bit-planes information */
      if
        (!cblk->numpasses)
      {
        cblk->numlenbits = 3;
        tgt_encode(bio, prc->imsbtree, cblkno, 999);
      }
      /* number of coding passes included */
      t2_putnumpasses(bio, layer->numpasses);
      l_nb_passes = cblk->numpasses + layer->numpasses;
      pass = cblk->passes +  cblk->numpasses;
      /* computation of the increase of the length indicator and insertion in the header     */
      for
        (passno = cblk->numpasses; passno < l_nb_passes; ++passno)
      {
        ++nump;
        len += pass->len;
        if
          (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1)
        {
          increment = int_max(increment, int_floorlog2(len) + 1 - (cblk->numlenbits + int_floorlog2(nump)));
          len = 0;
          nump = 0;
        }
        ++pass;
      }
      t2_putcommacode(bio, increment);

      /* computation of the new Length indicator */
      cblk->numlenbits += increment;

      pass = cblk->passes +  cblk->numpasses;
      /* insertion of the codeword segment length */
      for
        (passno = cblk->numpasses; passno < l_nb_passes; ++passno)
      {
        nump++;
        len += pass->len;
        if
          (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1)
        {
          bio_write(bio, len, cblk->numlenbits + int_floorlog2(nump));
          len = 0;
          nump = 0;
        }
        ++pass;
      }
      ++cblk;
    }
    ++band;
  }

  if
    (bio_flush(bio))
  {
    bio_destroy(bio);
    return false;    /* modified to eliminate longjmp !! */
  }
  l_nb_bytes = bio_numbytes(bio);
  c += l_nb_bytes;
  length -= l_nb_bytes;
  bio_destroy(bio);

  /* <EPH 0xff92> */
  if (tcp->csty & J2K_CP_CSTY_EPH) {
    c[0] = 255;
    c[1] = 146;
    c += 2;
    length -= 2;
  }
  /* </EPH> */

  /* << INDEX */
  // End of packet header position. Currently only represents the distance to start of packet
  // Will be updated later by incrementing with packet start value
  if(cstr_info && cstr_info->index_write) {
    opj_packet_info_t *info_PK = &cstr_info->tile[tileno].packet[cstr_info->packno];
    info_PK->end_ph_pos = (OPJ_INT32)(c - dest);
  }
  /* INDEX >> */

  /* Writing the packet body */
  band = res->bands;
  for
    (bandno = 0; bandno < res->numbands; bandno++)
  {
    opj_tcd_precinct_t *prc = &band->precincts[precno];
    l_nb_blocks = prc->cw * prc->ch;
    cblk = prc->cblks.enc;
    for
      (cblkno = 0; cblkno < l_nb_blocks; ++cblkno)
    {
      opj_tcd_layer_t *layer = &cblk->layers[layno];
      if
        (!layer->numpasses)
      {
        ++cblk;
        continue;
      }
      if
        (layer->len > length)
      {
        return false;
      }
      memcpy(c, layer->data, layer->len);
      cblk->numpasses += layer->numpasses;
      c += layer->len;
      length -= layer->len;
      /* << INDEX */
      if(cstr_info && cstr_info->index_write) {
        opj_packet_info_t *info_PK = &cstr_info->tile[tileno].packet[cstr_info->packno];
        info_PK->disto += layer->disto;
        if (cstr_info->D_max < info_PK->disto) {
          cstr_info->D_max = info_PK->disto;
        }
      }
      ++cblk;
      /* INDEX >> */
    }
    ++band;
  }
  * p_data_written += (c - dest);
  return true;
}

static bool t2_init_seg(opj_tcd_cblk_dec_t* cblk, OPJ_UINT32 index, OPJ_UINT32 cblksty, OPJ_UINT32 first)
{
  opj_tcd_seg_t* seg = 00;
  OPJ_UINT32 l_nb_segs = index + 1;

  if
    (l_nb_segs > cblk->m_current_max_segs)
  {
    cblk->m_current_max_segs += J2K_DEFAULT_NB_SEGS;
        cblk->segs = (opj_tcd_seg_t*) opj_realloc(cblk->segs, cblk->m_current_max_segs * sizeof(opj_tcd_seg_t));
    if
      (! cblk->segs)
    {
      return false;
    }
  }
  seg = &cblk->segs[index];
  memset(seg,0,sizeof(opj_tcd_seg_t));

  if (cblksty & J2K_CCP_CBLKSTY_TERMALL) {
    seg->maxpasses = 1;
  }
  else if (cblksty & J2K_CCP_CBLKSTY_LAZY) {
    if (first) {
      seg->maxpasses = 10;
    } else {
      seg->maxpasses = (((seg - 1)->maxpasses == 1) || ((seg - 1)->maxpasses == 10)) ? 2 : 1;
    }
  } else {
    seg->maxpasses = 109;
  }
  return true;
}

static bool t2_read_packet_header(
               opj_t2_t* p_t2,
               opj_tcd_tile_t *p_tile,
                             opj_tcp_t *p_tcp,
               opj_pi_iterator_t *p_pi,
               bool * p_is_data_present,
               OPJ_BYTE *p_src_data,
               OPJ_UINT32 * p_data_read,
               OPJ_UINT32 p_max_length,
               opj_packet_info_t *p_pack_info)
{
  /* loop */
  OPJ_UINT32 bandno, cblkno;
  OPJ_UINT32 l_nb_code_blocks;
  OPJ_UINT32 l_remaining_length;
  OPJ_UINT32 l_header_length;
  OPJ_UINT32 * l_modified_length_ptr = 00;
  OPJ_BYTE *l_current_data = p_src_data;
  opj_cp_t *l_cp = p_t2->cp;
  opj_bio_t *l_bio = 00;  /* BIO component */
  opj_tcd_band_t *l_band = 00;
  opj_tcd_cblk_dec_t* l_cblk = 00;
  opj_tcd_resolution_t* l_res = &p_tile->comps[p_pi->compno].resolutions[p_pi->resno];

  OPJ_BYTE *l_header_data = 00;
  OPJ_BYTE **l_header_data_start = 00;

  OPJ_UINT32 l_present;

  if
    (p_pi->layno == 0)
  {
    l_band = l_res->bands;
    /* reset tagtrees */
    for
      (bandno = 0; bandno < l_res->numbands; ++bandno)
    {
      opj_tcd_precinct_t *l_prc = &l_band->precincts[p_pi->precno];

      if (
        ! ((l_band->x1-l_band->x0 == 0)||(l_band->y1-l_band->y0 == 0)))
      {
        tgt_reset(l_prc->incltree);
        tgt_reset(l_prc->imsbtree);
        l_cblk = l_prc->cblks.dec;
        l_nb_code_blocks = l_prc->cw * l_prc->ch;
        for
          (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno)
        {
          l_cblk->numsegs = 0;
          l_cblk->real_num_segs = 0;
          ++l_cblk;
        }
      }
      ++l_band;
    }
  }

  /* SOP markers */

  if (p_tcp->csty & J2K_CP_CSTY_SOP) {
    if ((*l_current_data) != 0xff || (*(l_current_data + 1) != 0x91)) {
      // TODO opj_event_msg(t2->cinfo->event_mgr, EVT_WARNING, "Expected SOP marker\n");
    } else {
      l_current_data += 6;
    }

    /** TODO : check the Nsop value */
  }

  /*
  When the marker PPT/PPM is used the packet header are store in PPT/PPM marker
  This part deal with this caracteristic
  step 1: Read packet header in the saved structure
  step 2: Return to codestream for decoding
  */

  l_bio = bio_create();
  if
    (! l_bio)
  {
    return false;
  }

  if
    (l_cp->ppm == 1)
  {    /* PPM */
    l_header_data_start = &l_cp->ppm_data;
    l_header_data = *l_header_data_start;
    l_modified_length_ptr = &(l_cp->ppm_len);

  }
  else if
    (p_tcp->ppt == 1)
  {  /* PPT */
    l_header_data_start = &(p_tcp->ppt_data);
    l_header_data = *l_header_data_start;
    l_modified_length_ptr = &(p_tcp->ppt_len);
  }
  else
  {  /* Normal Case */
    l_header_data_start = &(l_current_data);
    l_header_data = *l_header_data_start;
    l_remaining_length = p_src_data+p_max_length-l_header_data;
    l_modified_length_ptr = &(l_remaining_length);
  }
  bio_init_dec(l_bio, l_header_data,*l_modified_length_ptr);
  l_present = bio_read(l_bio, 1);
  if
    (!l_present)
  {
    bio_inalign(l_bio);
    l_header_data += bio_numbytes(l_bio);
    bio_destroy(l_bio);
    /* EPH markers */
    if (p_tcp->csty & J2K_CP_CSTY_EPH) {
      if ((*l_header_data) != 0xff || (*(l_header_data + 1) != 0x92)) {
        printf("Error : expected EPH marker\n");
      } else {
        l_header_data += 2;
      }
    }
    l_header_length = (l_header_data - *l_header_data_start);
    *l_modified_length_ptr -= l_header_length;
    *l_header_data_start += l_header_length;
    /* << INDEX */
    // End of packet header position. Currently only represents the distance to start of packet
    // Will be updated later by incrementing with packet start value
    if
      (p_pack_info)
    {
      p_pack_info->end_ph_pos = (OPJ_INT32)(l_current_data - p_src_data);
    }
    /* INDEX >> */
    * p_is_data_present = false;
    *p_data_read = l_current_data - p_src_data;
    return true;
  }

  l_band = l_res->bands;
  for
    (bandno = 0; bandno < l_res->numbands; ++bandno)
  {
    opj_tcd_precinct_t *l_prc = &(l_band->precincts[p_pi->precno]);

    if ((l_band->x1-l_band->x0 == 0)||(l_band->y1-l_band->y0 == 0))
    {
      ++l_band;
      continue;
    }
    l_nb_code_blocks = l_prc->cw * l_prc->ch;
    l_cblk = l_prc->cblks.dec;
    for
      (cblkno = 0; cblkno < l_nb_code_blocks; cblkno++)
    {
      OPJ_UINT32 l_included,l_increment, l_segno;
      OPJ_INT32 n;
      /* if cblk not yet included before --> inclusion tagtree */
      if
        (!l_cblk->numsegs)
      {
        l_included = tgt_decode(l_bio, l_prc->incltree, cblkno, p_pi->layno + 1);
        /* else one bit */
      }
      else
      {
        l_included = bio_read(l_bio, 1);
      }
      /* if cblk not included */
      if
        (!l_included)
      {
        l_cblk->numnewpasses = 0;
        ++l_cblk;
        continue;
      }
      /* if cblk not yet included --> zero-bitplane tagtree */
      if
        (!l_cblk->numsegs)
      {
        OPJ_UINT32 i = 0;
        while
          (!tgt_decode(l_bio, l_prc->imsbtree, cblkno, i))
        {
          ++i;
        }
        l_cblk->numbps = l_band->numbps + 1 - i;
        l_cblk->numlenbits = 3;
      }
      /* number of coding passes */
      l_cblk->numnewpasses = t2_getnumpasses(l_bio);
      l_increment = t2_getcommacode(l_bio);
      /* length indicator increment */
      l_cblk->numlenbits += l_increment;
      l_segno = 0;
      if
        (!l_cblk->numsegs)
      {
        if
          (! t2_init_seg(l_cblk, l_segno, p_tcp->tccps[p_pi->compno].cblksty, 1))
        {
          bio_destroy(l_bio);
          return false;
        }

      }
      else
      {
        l_segno = l_cblk->numsegs - 1;
        if
          (l_cblk->segs[l_segno].numpasses == l_cblk->segs[l_segno].maxpasses)
        {
          ++l_segno;
          if
            (! t2_init_seg(l_cblk, l_segno, p_tcp->tccps[p_pi->compno].cblksty, 0))
          {
            bio_destroy(l_bio);
            return false;
          }
        }
      }
      n = l_cblk->numnewpasses;

      do {
        l_cblk->segs[l_segno].numnewpasses = int_min(l_cblk->segs[l_segno].maxpasses - l_cblk->segs[l_segno].numpasses, n);
        l_cblk->segs[l_segno].newlen = bio_read(l_bio, l_cblk->numlenbits + uint_floorlog2(l_cblk->segs[l_segno].numnewpasses));
        n -= l_cblk->segs[l_segno].numnewpasses;
        if
          (n > 0)
        {
          ++l_segno;
          if
            (! t2_init_seg(l_cblk, l_segno, p_tcp->tccps[p_pi->compno].cblksty, 0))
          {
            bio_destroy(l_bio);
            return false;
          }
        }
      }
      while (n > 0);
      ++l_cblk;
    }
    ++l_band;
  }

  if
    (bio_inalign(l_bio))
  {
    bio_destroy(l_bio);
    return false;
  }

  l_header_data += bio_numbytes(l_bio);
  bio_destroy(l_bio);

  /* EPH markers */
  if (p_tcp->csty & J2K_CP_CSTY_EPH) {
    if ((*l_header_data) != 0xff || (*(l_header_data + 1) != 0x92)) {
      // TODO opj_event_msg(t2->cinfo->event_mgr, EVT_ERROR, "Expected EPH marker\n");
    } else {
      l_header_data += 2;
    }
  }


  l_header_length = (l_header_data - *l_header_data_start);
  *l_modified_length_ptr -= l_header_length;
  *l_header_data_start += l_header_length;
  /* << INDEX */
  // End of packet header position. Currently only represents the distance to start of packet
  // Will be updated later by incrementing with packet start value
  if
    (p_pack_info)
  {
    p_pack_info->end_ph_pos = (OPJ_INT32)(l_current_data - p_src_data);
  }
  /* INDEX >> */
  * p_is_data_present = true;
  *p_data_read = l_current_data - p_src_data;
  return true;
}

static bool t2_read_packet_data(
               opj_t2_t* p_t2,
               opj_tcd_tile_t *p_tile,
               opj_pi_iterator_t *p_pi,
               OPJ_BYTE *p_src_data,
               OPJ_UINT32 * p_data_read,
               OPJ_UINT32 p_max_length,
               opj_packet_info_t *pack_info)
{
  OPJ_UINT32 bandno, cblkno;
  OPJ_UINT32 l_nb_code_blocks;
  OPJ_BYTE *l_current_data = p_src_data;
  opj_tcd_band_t *l_band = 00;
  opj_tcd_cblk_dec_t* l_cblk = 00;
  opj_tcd_resolution_t* l_res = &p_tile->comps[p_pi->compno].resolutions[p_pi->resno];

  l_band = l_res->bands;
  for
    (bandno = 0; bandno < l_res->numbands; ++bandno)
  {
    opj_tcd_precinct_t *l_prc = &l_band->precincts[p_pi->precno];

    if
      ((l_band->x1-l_band->x0 == 0)||(l_band->y1-l_band->y0 == 0))
    {
      ++l_band;
      continue;
    }
    l_nb_code_blocks = l_prc->cw * l_prc->ch;
    l_cblk = l_prc->cblks.dec;
    for
      (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno)
    {
      opj_tcd_seg_t *l_seg = 00;
      if
        (!l_cblk->numnewpasses)
      {
        /* nothing to do */
        ++l_cblk;
        continue;
      }
      if
        (!l_cblk->numsegs)
      {
        l_seg = l_cblk->segs;
        ++l_cblk->numsegs;
        l_cblk->len = 0;
      }
      else
      {
        l_seg = &l_cblk->segs[l_cblk->numsegs - 1];
        if
          (l_seg->numpasses == l_seg->maxpasses)
        {
          ++l_seg;
          ++l_cblk->numsegs;
        }
      }

      do
      {
        if
          (l_current_data + l_seg->newlen > p_src_data + p_max_length)
        {
          return false;
        }

#ifdef USE_JPWL
      /* we need here a j2k handle to verify if making a check to
      the validity of cblocks parameters is selected from user (-W) */

        /* let's check that we are not exceeding */
        if ((cblk->len + seg->newlen) > 8192) {
          opj_event_msg(t2->cinfo, EVT_WARNING,
            "JPWL: segment too long (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
            seg->newlen, cblkno, precno, bandno, resno, compno);
          if (!JPWL_ASSUME) {
            opj_event_msg(t2->cinfo, EVT_ERROR, "JPWL: giving up\n");
            return -999;
          }
          seg->newlen = 8192 - cblk->len;
          opj_event_msg(t2->cinfo, EVT_WARNING, "      - truncating segment to %d\n", seg->newlen);
          break;
        };

#endif /* USE_JPWL */

        memcpy(l_cblk->data + l_cblk->len, l_current_data, l_seg->newlen);
        if
          (l_seg->numpasses == 0)
        {
          l_seg->data = &l_cblk->data;
          l_seg->dataindex = l_cblk->len;
        }
        l_current_data += l_seg->newlen;
        l_seg->numpasses += l_seg->numnewpasses;
        l_cblk->numnewpasses -= l_seg->numnewpasses;

        l_seg->real_num_passes = l_seg->numpasses;
        l_cblk->len += l_seg->newlen;
        l_seg->len += l_seg->newlen;
        if
          (l_cblk->numnewpasses > 0)
        {
          ++l_seg;
          ++l_cblk->numsegs;
        }
      }
      while (l_cblk->numnewpasses > 0);
      l_cblk->real_num_segs = l_cblk->numsegs;
      ++l_cblk;
    }
    ++l_band;
  }
  *(p_data_read) = l_current_data - p_src_data;
  return true;
}


static bool t2_skip_packet_data(
               opj_t2_t* p_t2,
               opj_tcd_tile_t *p_tile,
               opj_pi_iterator_t *p_pi,
               OPJ_UINT32 * p_data_read,
               OPJ_UINT32 p_max_length,
               opj_packet_info_t *pack_info)
{
  OPJ_UINT32 bandno, cblkno;
  OPJ_UINT32 l_nb_code_blocks;
  opj_tcd_band_t *l_band = 00;
  opj_tcd_cblk_dec_t* l_cblk = 00;

  opj_tcd_resolution_t* l_res = &p_tile->comps[p_pi->compno].resolutions[p_pi->resno];

  *p_data_read = 0;
  l_band = l_res->bands;
  for
    (bandno = 0; bandno < l_res->numbands; ++bandno)
  {
    opj_tcd_precinct_t *l_prc = &l_band->precincts[p_pi->precno];

    if
      ((l_band->x1-l_band->x0 == 0)||(l_band->y1-l_band->y0 == 0))
    {
      ++l_band;
      continue;
    }
    l_nb_code_blocks = l_prc->cw * l_prc->ch;
    l_cblk = l_prc->cblks.dec;
    for
      (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno)
    {
      opj_tcd_seg_t *l_seg = 00;
      if
        (!l_cblk->numnewpasses)
      {
        /* nothing to do */
        ++l_cblk;
        continue;
      }
      if
        (!l_cblk->numsegs)
      {
        l_seg = l_cblk->segs;
        ++l_cblk->numsegs;
        l_cblk->len = 0;
      }
      else
      {
        l_seg = &l_cblk->segs[l_cblk->numsegs - 1];
        if
          (l_seg->numpasses == l_seg->maxpasses)
        {
          ++l_seg;
          ++l_cblk->numsegs;
        }
      }

      do
      {
        if
          (* p_data_read + l_seg->newlen > p_max_length)
        {
          return false;
        }

#ifdef USE_JPWL
      /* we need here a j2k handle to verify if making a check to
      the validity of cblocks parameters is selected from user (-W) */

        /* let's check that we are not exceeding */
        if ((cblk->len + seg->newlen) > 8192) {
          opj_event_msg(t2->cinfo, EVT_WARNING,
            "JPWL: segment too long (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
            seg->newlen, cblkno, precno, bandno, resno, compno);
          if (!JPWL_ASSUME) {
            opj_event_msg(t2->cinfo, EVT_ERROR, "JPWL: giving up\n");
            return -999;
          }
          seg->newlen = 8192 - cblk->len;
          opj_event_msg(t2->cinfo, EVT_WARNING, "      - truncating segment to %d\n", seg->newlen);
          break;
        };

#endif /* USE_JPWL */
        *(p_data_read) += l_seg->newlen;
        l_seg->numpasses += l_seg->numnewpasses;
        l_cblk->numnewpasses -= l_seg->numnewpasses;
        if
          (l_cblk->numnewpasses > 0)
        {
          ++l_seg;
          ++l_cblk->numsegs;
        }
      }
      while (l_cblk->numnewpasses > 0);
      ++l_cblk;
    }
    ++l_band;
  }
  return true;
}

static bool t2_decode_packet(
               opj_t2_t* p_t2,
               opj_tcd_tile_t *p_tile,
                             opj_tcp_t *p_tcp,
               opj_pi_iterator_t *p_pi,
               OPJ_BYTE *p_src,
               OPJ_UINT32 * p_data_read,
               OPJ_UINT32 p_max_length,
               opj_packet_info_t *p_pack_info)
{
  bool l_read_data;
  OPJ_UINT32 l_nb_bytes_read = 0;
  OPJ_UINT32 l_nb_total_bytes_read = 0;

  *p_data_read = 0;

  if
    (! t2_read_packet_header(p_t2,p_tile,p_tcp,p_pi,&l_read_data,p_src,&l_nb_bytes_read,p_max_length,p_pack_info))
  {
    return false;
  }
  p_src += l_nb_bytes_read;
  l_nb_total_bytes_read += l_nb_bytes_read;
  p_max_length -= l_nb_bytes_read;
  /* we should read data for the packet */
  if
    (l_read_data)
  {
    l_nb_bytes_read = 0;
    if
      (! t2_read_packet_data(p_t2,p_tile,p_pi,p_src,&l_nb_bytes_read,p_max_length,p_pack_info))
    {
      return false;
    }
    l_nb_total_bytes_read += l_nb_bytes_read;
  }
  *p_data_read = l_nb_total_bytes_read;
  return true;
}

static bool t2_skip_packet(
               opj_t2_t* p_t2,
               opj_tcd_tile_t *p_tile,
                             opj_tcp_t *p_tcp,
               opj_pi_iterator_t *p_pi,
               OPJ_BYTE *p_src,
               OPJ_UINT32 * p_data_read,
               OPJ_UINT32 p_max_length,
               opj_packet_info_t *p_pack_info)
{
  bool l_read_data;
  OPJ_UINT32 l_nb_bytes_read = 0;
  OPJ_UINT32 l_nb_total_bytes_read = 0;

  *p_data_read = 0;

  if
    (! t2_read_packet_header(p_t2,p_tile,p_tcp,p_pi,&l_read_data,p_src,&l_nb_bytes_read,p_max_length,p_pack_info))
  {
    return false;
  }
  p_src += l_nb_bytes_read;
  l_nb_total_bytes_read += l_nb_bytes_read;
  p_max_length -= l_nb_bytes_read;
  /* we should read data for the packet */
  if
    (l_read_data)
  {
    l_nb_bytes_read = 0;
    if
      (! t2_skip_packet_data(p_t2,p_tile,p_pi,&l_nb_bytes_read,p_max_length,p_pack_info))
    {
      return false;
    }
    l_nb_total_bytes_read += l_nb_bytes_read;
  }
  *p_data_read = l_nb_total_bytes_read;
  return true;
}

/* ----------------------------------------------------------------------- */

bool t2_encode_packets(
             opj_t2_t* p_t2,
             OPJ_UINT32 p_tile_no,
             opj_tcd_tile_t *p_tile,
             OPJ_UINT32 p_maxlayers,
             OPJ_BYTE *p_dest,
             OPJ_UINT32 * p_data_written,
             OPJ_UINT32 p_max_len,
             opj_codestream_info_t *cstr_info,
             OPJ_UINT32 p_tp_num,
             OPJ_INT32 p_tp_pos,
             OPJ_UINT32 p_pino,
             J2K_T2_MODE p_t2_mode)
{
  OPJ_BYTE *l_current_data = p_dest;
  OPJ_UINT32 l_nb_bytes = 0;
  OPJ_UINT32 compno;
  OPJ_UINT32 poc;
  opj_pi_iterator_t *l_pi = 00;
  opj_pi_iterator_t *l_current_pi = 00;
  opj_image_t *l_image = p_t2->image;
  opj_cp_t *l_cp = p_t2->cp;
  opj_tcp_t *l_tcp = &l_cp->tcps[p_tile_no];
  OPJ_UINT32 pocno = l_cp->m_specific_param.m_enc.m_cinema == CINEMA4K_24? 2: 1;
  OPJ_UINT32 l_max_comp = l_cp->m_specific_param.m_enc.m_max_comp_size > 0 ? l_image->numcomps : 1;
  OPJ_UINT32 l_nb_pocs = l_tcp->numpocs + 1;

  l_pi = pi_initialise_encode(l_image, l_cp, p_tile_no, p_t2_mode);
  if
    (!l_pi)
  {
    return false;
  }
  * p_data_written = 0;
  if
    (p_t2_mode == THRESH_CALC )
  { /* Calculating threshold */
    l_current_pi = l_pi;
    for
      (compno = 0; compno < l_max_comp; ++compno)
    {
      OPJ_UINT32 l_comp_len = 0;
      l_current_pi = l_pi;

      for
        (poc = 0; poc < pocno ; ++poc)
      {
        OPJ_UINT32 l_tp_num = compno;
        pi_create_encode(l_pi, l_cp,p_tile_no,poc,l_tp_num,p_tp_pos,p_t2_mode);
        while
          (pi_next(l_current_pi))
        {
          if
            (l_current_pi->layno < p_maxlayers)
          {
            l_nb_bytes = 0;
            if
              (! t2_encode_packet(p_tile_no,p_tile, l_tcp, l_current_pi, l_current_data, &l_nb_bytes, p_max_len, cstr_info))
            {
              pi_destroy(l_pi, l_nb_pocs);
              return false;
            }
            l_comp_len += l_nb_bytes;
            l_current_data += l_nb_bytes;
            p_max_len -= l_nb_bytes;
            * p_data_written += l_nb_bytes;
          }
        }
        if
          (l_cp->m_specific_param.m_enc.m_max_comp_size)
        {
          if
            (l_comp_len > l_cp->m_specific_param.m_enc.m_max_comp_size)
          {
            pi_destroy(l_pi, l_nb_pocs);
            return false;
          }
        }
        ++l_current_pi;
      }
    }
  }
  else
  {  /* t2_mode == FINAL_PASS  */
    pi_create_encode(l_pi, l_cp,p_tile_no,p_pino,p_tp_num,p_tp_pos,p_t2_mode);
    l_current_pi = &l_pi[p_pino];
    while
      (pi_next(l_current_pi))
    {
      if
        (l_current_pi->layno < p_maxlayers)
      {
        l_nb_bytes=0;
        if
          (! t2_encode_packet(p_tile_no,p_tile, l_tcp, l_current_pi, l_current_data, &l_nb_bytes, p_max_len, cstr_info))
        {
          pi_destroy(l_pi, l_nb_pocs);
          return false;
        }
        l_current_data += l_nb_bytes;
        p_max_len -= l_nb_bytes;
        * p_data_written += l_nb_bytes;

        /* INDEX >> */
        if(cstr_info) {
          if(cstr_info->index_write) {
            opj_tile_info_t *info_TL = &cstr_info->tile[p_tile_no];
            opj_packet_info_t *info_PK = &info_TL->packet[cstr_info->packno];
            if (!cstr_info->packno) {
              info_PK->start_pos = info_TL->end_header + 1;
            } else {
              info_PK->start_pos = ((l_cp->m_specific_param.m_enc.m_tp_on | l_tcp->POC)&& info_PK->start_pos) ? info_PK->start_pos : info_TL->packet[cstr_info->packno - 1].end_pos + 1;
            }
            info_PK->end_pos = info_PK->start_pos + l_nb_bytes - 1;
            info_PK->end_ph_pos += info_PK->start_pos - 1;  // End of packet header which now only represents the distance
                                                            // to start of packet is incremented by value of start of packet
          }

          cstr_info->packno++;
        }
        /* << INDEX */
        ++p_tile->packno;
      }
    }
  }
  pi_destroy(l_pi, l_nb_pocs);
  return true;
}

bool t2_decode_packets(
            opj_t2_t *p_t2,
            OPJ_UINT32 p_tile_no,
            struct opj_tcd_tile *p_tile,
            OPJ_BYTE *p_src,
            OPJ_UINT32 * p_data_read,
            OPJ_UINT32 p_max_len,
            struct opj_codestream_info *p_cstr_info)
{
  OPJ_BYTE *l_current_data = p_src;
  opj_pi_iterator_t *l_pi = 00;
  OPJ_UINT32 pino;
  opj_image_t *l_image = p_t2->image;
  opj_cp_t *l_cp = p_t2->cp;
  opj_cp_t *cp = p_t2->cp;
  opj_tcp_t *l_tcp = &(p_t2->cp->tcps[p_tile_no]);
  OPJ_UINT32 l_nb_bytes_read;
  OPJ_UINT32 l_nb_pocs = l_tcp->numpocs + 1;
  opj_pi_iterator_t *l_current_pi = 00;
  OPJ_UINT32 curtp = 0;
  OPJ_UINT32 tp_start_packno;
  opj_packet_info_t *l_pack_info = 00;
  opj_image_comp_t* l_img_comp = 00;


  if
    (p_cstr_info)
  {
    l_pack_info = p_cstr_info->tile[p_tile_no].packet;
  }

  /* create a packet iterator */
  l_pi = pi_create_decode(l_image, l_cp, p_tile_no);
  if
    (!l_pi)
  {
    return false;
  }

  tp_start_packno = 0;
  l_current_pi = l_pi;

  for
    (pino = 0; pino <= l_tcp->numpocs; ++pino)
  {
    while
      (pi_next(l_current_pi))
    {

      if
        (l_tcp->num_layers_to_decode > l_current_pi->layno && l_current_pi->resno < p_tile->comps[l_current_pi->compno].minimum_num_resolutions)
      {
        l_nb_bytes_read = 0;
        if
          (! t2_decode_packet(p_t2,p_tile,l_tcp,l_current_pi,l_current_data,&l_nb_bytes_read,p_max_len,l_pack_info))
        {
          pi_destroy(l_pi,l_nb_pocs);
          return false;
        }
        l_img_comp = &(l_image->comps[l_current_pi->compno]);
        l_img_comp->resno_decoded = uint_max(l_current_pi->resno, l_img_comp->resno_decoded);
      }
      else
      {
        l_nb_bytes_read = 0;
        if
          (! t2_skip_packet(p_t2,p_tile,l_tcp,l_current_pi,l_current_data,&l_nb_bytes_read,p_max_len,l_pack_info))
        {
          pi_destroy(l_pi,l_nb_pocs);
          return false;
        }
      }
      l_current_data += l_nb_bytes_read;
      p_max_len -= l_nb_bytes_read;

      /* INDEX >> */
      if(p_cstr_info) {
        opj_tile_info_t *info_TL = &p_cstr_info->tile[p_tile_no];
        opj_packet_info_t *info_PK = &info_TL->packet[p_cstr_info->packno];
        if (!p_cstr_info->packno) {
          info_PK->start_pos = info_TL->end_header + 1;
        } else if (info_TL->packet[p_cstr_info->packno-1].end_pos >= (OPJ_INT32)p_cstr_info->tile[p_tile_no].tp[curtp].tp_end_pos){ // New tile part
          info_TL->tp[curtp].tp_numpacks = p_cstr_info->packno - tp_start_packno; // Number of packets in previous tile-part
          tp_start_packno = p_cstr_info->packno;
          curtp++;
          info_PK->start_pos = p_cstr_info->tile[p_tile_no].tp[curtp].tp_end_header+1;
        } else {
          info_PK->start_pos = (cp->m_specific_param.m_enc.m_tp_on && info_PK->start_pos) ? info_PK->start_pos : info_TL->packet[p_cstr_info->packno - 1].end_pos + 1;
        }
        info_PK->end_pos = info_PK->start_pos + l_nb_bytes_read - 1;
        info_PK->end_ph_pos += info_PK->start_pos - 1;  // End of packet header which now only represents the distance
        ++p_cstr_info->packno;
      }
      /* << INDEX */
    }
    ++l_current_pi;
  }
  /* INDEX >> */
  if
    (p_cstr_info) {
    p_cstr_info->tile[p_tile_no].tp[curtp].tp_numpacks = p_cstr_info->packno - tp_start_packno; // Number of packets in last tile-part
  }
  /* << INDEX */

  /* don't forget to release pi */
  pi_destroy(l_pi,l_nb_pocs);
  *p_data_read = l_current_data - p_src;
  return true;
}

/* ----------------------------------------------------------------------- */
/**
 * Creates a Tier 2 handle
 *
 * @param  p_image    Source or destination image
 * @param  p_cp    Image coding parameters.
 * @return    a new T2 handle if successful, NULL otherwise.
*/
opj_t2_t* t2_create(
          opj_image_t *p_image,
          opj_cp_t *p_cp)
{
  /* create the tcd structure */
  opj_t2_t *l_t2 = (opj_t2_t*)opj_malloc(sizeof(opj_t2_t));
  if
    (!l_t2)
  {
    return 00;
  }
  memset(l_t2,0,sizeof(opj_t2_t));
  l_t2->image = p_image;
  l_t2->cp = p_cp;
  return l_t2;
}

/**
 * Destroys a Tier 2 handle.
 *
 * @param  p_t2  the Tier 2 handle to destroy
*/
void t2_destroy(opj_t2_t *p_t2)
{
  if
    (p_t2)
  {
    opj_free(p_t2);
  }
}
