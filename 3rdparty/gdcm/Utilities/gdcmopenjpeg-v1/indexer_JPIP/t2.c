/*
 * Copyright (c) 2001-2002, David Janssens
 * Copyright (c) 2003, Yannick Verschueren
 * Copyright (c) 2003,  Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#include "t2.h"
#include "tcd.h"
#include "bio.h"
#include "j2k.h"
#include "pi.h"
#include "tgt.h"
#include "int.h"
#include "cio.h"
#include <stdio.h>
#include <setjmp.h>
#include <string.h>
#include <stdlib.h> 

#define RESTART 0x04

extern jmp_buf j2k_error;

int t2_getcommacode() {
    int n;
    for (n=0; bio_read(1); n++) {}
    return n;
}

int t2_getnumpasses()
{
    int n;
    if (!bio_read(1)) return 1;
    if (!bio_read(1)) return 2;
    if ((n=bio_read(2))!=3) return 3+n;
    if ((n=bio_read(5))!=31) return 6+n;
    return 37+bio_read(7);
}

void t2_init_seg(tcd_seg_t *seg, int cblksty) {
    seg->numpasses=0;
    seg->len=0;
    seg->maxpasses=cblksty&J2K_CCP_CBLKSTY_TERMALL?1:100;
}

int t2_decode_packet(unsigned char *src, int len, tcd_tile_t *tile, j2k_cp_t * cp, j2k_tcp_t *tcp, int compno, int resno, int precno, int layno, info_layer_t *layer_Idx) {
    int bandno, cblkno;
    tcd_tilecomp_t *tilec = &tile->comps[compno];
    tcd_resolution_t *res = &tilec->resolutions[resno];
    unsigned char *c = src;
    unsigned char *d = c;
    int e;
    int present;

    if (layno == 0) {
        for (bandno = 0; bandno < res->numbands; bandno++) {
            tcd_band_t *band = &res->bands[bandno];
            tcd_precinct_t *prc = &band->precincts[precno];
            tgt_reset(prc->incltree);
            tgt_reset(prc->imsbtree);
            for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
                tcd_cblk_t *cblk = &prc->cblks[cblkno];
                cblk->numsegs = 0;
            }
        }  
    }
    /* INDEX */
    layer_Idx->len_header = 0;

    /* When the marker PPT/PPM is used the packet header are store in PPT/PPM marker
       This part deal with this caracteristic
       step 1: Read packet header in the saved structure
       step 2: (futher) return to codestream for decoding */
    if (cp->ppm == 1) /* PPM */
      {	    
	c = cp->ppm_data;
	d = c;
	bio_init_dec(c, 1000);
      } else 
	{
	  if (tcp->ppt == 1) /* PPT */
	    {
	      c = tcp->ppt_data;
	      d = c;
	      bio_init_dec(c, 1000);
	    } else /* Normal Case */
	      {
		if (tcp->csty & J2K_CP_CSTY_SOP) 
		  {
		    if ((*c) != 255 || (*(c+1) != 145)) {printf("Error : expected SOP marker [1]!!!\n");}
		    c += 6;
		  }
		bio_init_dec(c, src + len - c);
		layer_Idx->len_header = -6;
	      }
	}
    
    present = bio_read(1);
    
    if (!present) 
      {
	bio_inalign();
	/* Normal case */
	c += bio_numbytes();
	if (tcp->csty & J2K_CP_CSTY_EPH) 
	  {
	    if ((*c) != 255 || (*(c+1) != 146)) {printf("Error : expected EPH marker [1]!!!\n");}
	    c += 2;
	  }
	/* INDEX */
	layer_Idx->len_header += (c-d);

	/* PPT and PPM dealing */
	if (cp->ppm == 1) /* PPM */
	  {	
	    cp->ppm_data = c;	    
	    return 0;
	  }
	if (tcp->ppt == 1) /* PPT */
	  {
	    tcp->ppt_data = c;
	    return 0;
	  }
	return c - src;
      }
    
    for (bandno=0; bandno<res->numbands; bandno++) {
        tcd_band_t *band = &res->bands[bandno];
        tcd_precinct_t *prc = &band->precincts[precno];
        for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
            int included, increment, n;
            tcd_cblk_t *cblk = &prc->cblks[cblkno];
            tcd_seg_t *seg;
            if (!cblk->numsegs) {
                included = tgt_decode(prc->incltree, cblkno, layno+1);
            } else {
                included = bio_read(1);
            }
            if (!included) {
                cblk->numnewpasses = 0;
                continue;
            }
            if (!cblk->numsegs) {
                int i, numimsbs;
                for (i = 0; !tgt_decode(prc->imsbtree, cblkno, i); i++) {}
                numimsbs = i-1;
                cblk->numbps = band->numbps - numimsbs;
                cblk->numlenbits = 3;
            }
            cblk->numnewpasses = t2_getnumpasses();
            increment = t2_getcommacode();
            cblk->numlenbits += increment;
            if (!cblk->numsegs) {
                seg = &cblk->segs[0];
                t2_init_seg(seg, tcp->tccps[compno].cblksty);
            } else {
                seg = &cblk->segs[cblk->numsegs - 1];
                if (seg->numpasses == seg->maxpasses) {
                    t2_init_seg(++seg, tcp->tccps[compno].cblksty);
                }
            }
            n = cblk->numnewpasses;
            do {
                seg->numnewpasses = int_min(seg->maxpasses-seg->numpasses, n);
                seg->newlen = bio_read(cblk->numlenbits + int_floorlog2(seg->numnewpasses));
                n -= seg->numnewpasses;
                if (n > 0) {
                    t2_init_seg(++seg, tcp->tccps[compno].cblksty);
                }
            } while (n > 0);
        }
    }
    if(bio_inalign()) return -999;
    c += bio_numbytes();

    if (tcp->csty & J2K_CP_CSTY_EPH) { /* EPH marker */
      if ((*c) != 255 || (*(c+1) != 146)) {printf("Error : expected EPH marker [2]!!!\n"); }
      c += 2;
    }
    
    /* INDEX */
    layer_Idx->len_header += (c-d);

    /* PPT Step 2 : see above for details */
    if (cp->ppm == 1)
      {
	cp->ppm_data = c; /* Update pointer */

	/* INDEX */
	layer_Idx->len_header = c-d;

	c = src;
	d = c;
	if (tcp->csty & J2K_CP_CSTY_SOP) 
	  {
	    if ((*c) != 255 || (*(c+1) != 145)) {printf("Error : expected SOP marker [2] !!!\n"); }
	    c += 6;
	  }
	bio_init_dec(c, src + len - c);
      } else 
	{
	  if (tcp->ppt == 1)
	    { 
	      tcp->ppt_data = c; /* Update pointer */
	      /* INDEX */
	      layer_Idx->len_header = c-d;

	      c = src;
	      d = c;
	      if (tcp->csty & J2K_CP_CSTY_SOP) /* SOP marker */
		{ 
		  if ((*c) != 255 || (*(c+1) != 145)) {printf("Error : expected SOP marker [2] !!!\n"); }
		  c += 6;
		}
	      bio_init_dec(c, src + len - c);
	      
	    }
	}

    for (bandno = 0; bandno < res->numbands; bandno++) {
        tcd_band_t *band = &res->bands[bandno];
        tcd_precinct_t *prc = &band->precincts[precno];
        for (cblkno = 0; cblkno < prc->cw*prc->ch; cblkno++) {
            tcd_cblk_t *cblk = &prc->cblks[cblkno];
            tcd_seg_t *seg;
            if (!cblk->numnewpasses) continue;
            if (!cblk->numsegs) {
                seg = &cblk->segs[cblk->numsegs++];
                cblk->len = 0;
            } else {
                seg = &cblk->segs[cblk->numsegs-1];
                if (seg->numpasses == seg->maxpasses) {
                    seg++;
                    cblk->numsegs++;
                }
            }
            do {
	      if (c + seg->newlen > src + len) return -999;
                memcpy(cblk->data + cblk->len, c, seg->newlen);
                if (seg->numpasses == 0) {
                    seg->data = cblk->data + cblk->len;
                }
                c += seg->newlen;
                cblk->len += seg->newlen;
                seg->len += seg->newlen;
                seg->numpasses += seg->numnewpasses;
                cblk->numnewpasses -= seg->numnewpasses;
                if (cblk->numnewpasses > 0) {
                    seg++;
                    cblk->numsegs++;
                }
            } while (cblk->numnewpasses > 0);
        }
    }
    /* <INDEX> */
    e = c-d;
    layer_Idx->len = e;
    /* </INDEX> */

    return c-src;
}

void t2_init_info_packets(info_image_t *img, j2k_cp_t *cp)
{
  int compno, tileno, resno, precno, layno;

  for(compno = 0; compno < img->Comp; compno++)
    {
      for(tileno = 0; tileno < img->tw*img->th; tileno++)
	{
	  info_tile_t *tile_Idx = &img->tile[tileno];
	  info_compo_t *compo_Idx = &tile_Idx->compo[compno];
	  for(resno = 0; resno < img->Decomposition + 1 ; resno++)
	    {
	      info_reso_t *reso_Idx = &compo_Idx->reso[resno];
	      for (precno = 0; precno < img->tile[tileno].pw * img->tile[tileno].ph; precno++)
		{
		  info_prec_t *prec_Idx = &reso_Idx->prec[precno];
		  for(layno = 0; layno < img->Layer ; layno++)
		    {
		      info_layer_t *layer_Idx = &prec_Idx->layer[layno];
		      layer_Idx->offset = 0;        /* start position */
		      layer_Idx->len_header = 0;    /* length         */
		    }
		}
	    }
	}
    }
}

int t2_decode_packets(unsigned char *src, int len, j2k_image_t *img, j2k_cp_t *cp, int tileno, tcd_tile_t *tile, info_image_t *imgg) {
    unsigned char *c = src;
    pi_iterator_t *pi;
    int pino, compno,e;
    int partno;
    info_tile_part_t *tile_part;
    int position;
    int length_read;
    info_tile_t *tile_Idx;
    info_compo_t *compo_Idx;
    info_reso_t *reso_Idx;
    info_prec_t *prec_Idx;
    info_layer_t *layer_Idx;

    t2_init_info_packets(imgg, cp); /* Initialize the packets information : LEN and OFFSET to 0 */

    tile_Idx = &imgg->tile[tileno];
    tile_Idx->num_packet = 0;
    pi = pi_create(img, cp, tileno);
    partno = 0;
    tile_part = &tile_Idx->tile_parts[partno];
    position = tile_part->end_header + 1;
    length_read = 0;

    for (pino = 0; pino <= cp->tcps[tileno].numpocs; pino++)
      {
	while (pi_next(&pi[pino])) 
	  {   
	    compo_Idx = &tile_Idx->compo[pi[pino].compno];
	    reso_Idx = &compo_Idx->reso[pi[pino].resno];
	    prec_Idx = &reso_Idx->prec[pi[pino].precno];
	    layer_Idx = &prec_Idx->layer[pi[pino].layno];
	    
	    layer_Idx->offset = position;
	    layer_Idx->offset_header = position;
	    
	    e = t2_decode_packet(c, src+len-c, tile, cp, &cp->tcps[tileno], pi[pino].compno, pi[pino].resno, pi[pino].precno, pi[pino].layno,layer_Idx);
	    if (e == -999)
	      {
		break;
	      } else
		c += e;
	    position += e;
	    
	    /* Update position in case of multiple tile-parts for a tile >> */
	    length_read += e;
	    if (length_read >= (tile_part->end_pos - tile_part->end_header))
	      {
		partno++;
		tile_part = &tile_Idx->tile_parts[partno];
		position = tile_part->end_header + 1;
		length_read = 0;
	      }
	    /* << end_update */
	    
	    tile_Idx->num_packet++;
	  }
	
	// FREE space memory taken by pi
	for (compno = 0; compno < pi[pino].numcomps; compno++) 
	  { 
	    free(pi[pino].comps[compno].resolutions);
	  } 
	free(pi[pino].comps);
      }
    
    free(pi[0].include);
    free(pi);
 
    if (e==-999)
      return e;
    else
      {
	imgg->num_packet_max=int_max(imgg->num_packet_max,tile_Idx->num_packet);
	return c-src;
      }
}
