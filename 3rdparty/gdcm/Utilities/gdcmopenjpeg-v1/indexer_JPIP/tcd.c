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

#include "tcd.h"
#include "int.h"
#include "t2.h"
#include <setjmp.h>
#include <float.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static tcd_image_t tcd_image;

static j2k_image_t *tcd_img;
static j2k_cp_t *tcd_cp;

extern jmp_buf j2k_error;

void tcd_init(j2k_image_t *img, j2k_cp_t *cp, info_image_t *imgg) {
    int tileno, compno, resno, bandno, precno, cblkno;
    tcd_img=img;
    tcd_cp=cp;
    tcd_image.tw=cp->tw;
    tcd_image.th=cp->th;
    tcd_image.tiles=(tcd_tile_t*)malloc(cp->tw*cp->th*sizeof(tcd_tile_t));
    for (tileno=0; tileno<cp->tw*cp->th; tileno++) {
        j2k_tcp_t *tcp=&cp->tcps[tileno];
        tcd_tile_t *tile=&tcd_image.tiles[tileno];
        // cfr p59 ISO/IEC FDIS15444-1 : 2000 (18 august 2000)
        int p=tileno%cp->tw;  // si numerotation matricielle ..
        int q=tileno/cp->tw;  // .. coordonnees de la tile (q,p) q pour ligne et p pour colonne
	info_tile_t *tile_Idx=&imgg->tile[tileno]; // INDEX

	// 4 borders of the tile rescale on the image if necessary
        tile->x0=int_max(cp->tx0+p*cp->tdx, img->x0);
        tile->y0=int_max(cp->ty0+q*cp->tdy, img->y0);
        tile->x1=int_min(cp->tx0+(p+1)*cp->tdx, img->x1);
        tile->y1=int_min(cp->ty0+(q+1)*cp->tdy, img->y1);
	
        tile->numcomps=img->numcomps;
        tile->comps=(tcd_tilecomp_t*)malloc(img->numcomps*sizeof(tcd_tilecomp_t));
        tile_Idx->compo=(info_compo_t*)malloc(img->numcomps*sizeof(info_compo_t)); // INDEX
	for (compno=0; compno<tile->numcomps; compno++) {
            j2k_tccp_t *tccp=&tcp->tccps[compno];
            tcd_tilecomp_t *tilec=&tile->comps[compno];
	    info_compo_t *compo_Idx=&tile_Idx->compo[compno]; // INDEX

	    // border of each tile component (global)
            tilec->x0=int_ceildiv(tile->x0, img->comps[compno].dx);
            tilec->y0=int_ceildiv(tile->y0, img->comps[compno].dy);
	    tilec->x1=int_ceildiv(tile->x1, img->comps[compno].dx);
            tilec->y1=int_ceildiv(tile->y1, img->comps[compno].dy);
	    
            tilec->data=(int*)malloc(sizeof(int)*(tilec->x1-tilec->x0)*(tilec->y1-tilec->y0));
            tilec->numresolutions=tccp->numresolutions;
            tilec->resolutions=(tcd_resolution_t*)malloc(tilec->numresolutions*sizeof(tcd_resolution_t));
	    compo_Idx->reso=(info_reso_t*)malloc(tilec->numresolutions*sizeof(info_reso_t)); // INDEX
            for (resno=0; resno<tilec->numresolutions; resno++) {
                int pdx, pdy;
                int levelno=tilec->numresolutions-1-resno;
                int tlprcxstart, tlprcystart, brprcxend, brprcyend;
                int tlcbgxstart, tlcbgystart, brcbgxend, brcbgyend;
                int cbgwidthexpn, cbgheightexpn;
                int cblkwidthexpn, cblkheightexpn;
                tcd_resolution_t *res=&tilec->resolutions[resno];
		info_reso_t *res_Idx=&compo_Idx->reso[resno]; // INDEX
		int precno_Idx; // INDEX

		// border for each resolution level (global)
                res->x0=int_ceildivpow2(tilec->x0, levelno);
                res->y0=int_ceildivpow2(tilec->y0, levelno);
                res->x1=int_ceildivpow2(tilec->x1, levelno);
                res->y1=int_ceildivpow2(tilec->y1, levelno);
		
		res->numbands=resno==0?1:3;
		// p. 35, table A-23, ISO/IEC FDIS154444-1 : 2000 (18 august 2000)
                if (tccp->csty&J2K_CCP_CSTY_PRT) {
                    pdx=tccp->prcw[resno];
                    pdy=tccp->prch[resno];
		} else {
                    pdx=15;
                    pdy=15;
                }
		// p. 64, B.6, ISO/IEC FDIS15444-1 : 2000 (18 august 2000) 
                tlprcxstart=int_floordivpow2(res->x0, pdx)<<pdx;
                tlprcystart=int_floordivpow2(res->y0, pdy)<<pdy;
                brprcxend=int_ceildivpow2(res->x1, pdx)<<pdx;
                brprcyend=int_ceildivpow2(res->y1, pdy)<<pdy;
                res->pw=(brprcxend-tlprcxstart)>>pdx;
                res->ph=(brprcyend-tlprcystart)>>pdy;

		// <INDEX>
		imgg->tile[tileno].pw=res->pw;
		imgg->tile[tileno].ph=res->ph;
		
		res_Idx->prec=(info_prec_t*)malloc(res->pw*res->ph*sizeof(info_prec_t));
		for (precno_Idx=0;precno_Idx<res->pw*res->ph;precno_Idx++)
		  {
		    info_prec_t *prec_Idx = &res_Idx->prec[precno_Idx];
		    prec_Idx->layer=(info_layer_t*)malloc(imgg->Layer*sizeof(info_layer_t));
		  }
		
		imgg->pw=res->pw;  // old parser version
		imgg->ph=res->ph;  // old parser version
		imgg->pdx=1<<pdx;
		imgg->pdy=1<<pdy;
		// </INDEX>

                if (resno==0) {
                    tlcbgxstart=tlprcxstart;
                    tlcbgystart=tlprcystart;
                    brcbgxend=brprcxend;
                    brcbgyend=brprcyend;
                    cbgwidthexpn=pdx;
                    cbgheightexpn=pdy;
                } else {
                    tlcbgxstart=int_ceildivpow2(tlprcxstart, 1);
                    tlcbgystart=int_ceildivpow2(tlprcystart, 1);
                    brcbgxend=int_ceildivpow2(brprcxend, 1);
                    brcbgyend=int_ceildivpow2(brprcyend, 1);
                    cbgwidthexpn=pdx-1;
                    cbgheightexpn=pdy-1;
                }

                cblkwidthexpn=int_min(tccp->cblkw, cbgwidthexpn);
                cblkheightexpn=int_min(tccp->cblkh, cbgheightexpn);

                for (bandno=0; bandno<res->numbands; bandno++) {
                    int x0b, y0b;
                    tcd_band_t *band=&res->bands[bandno];
                    band->bandno=resno==0?0:bandno+1;
                    x0b=(band->bandno==1)||(band->bandno==3)?1:0;
                    y0b=(band->bandno==2)||(band->bandno==3)?1:0;

                    if (band->bandno==0) {
		      // band border (global)
		      band->x0=int_ceildivpow2(tilec->x0, levelno);
		      band->y0=int_ceildivpow2(tilec->y0, levelno);
		      band->x1=int_ceildivpow2(tilec->x1, levelno);
		      band->y1=int_ceildivpow2(tilec->y1, levelno);
                    } else {
		      // band border (global)
		      band->x0=int_ceildivpow2(tilec->x0-(1<<levelno)*x0b, levelno+1);
		      band->y0=int_ceildivpow2(tilec->y0-(1<<levelno)*y0b, levelno+1);
		      band->x1=int_ceildivpow2(tilec->x1-(1<<levelno)*x0b, levelno+1);
		      band->y1=int_ceildivpow2(tilec->y1-(1<<levelno)*y0b, levelno+1);
                    }

                    band->precincts=(tcd_precinct_t*)malloc(res->pw*res->ph*sizeof(tcd_precinct_t));

                    for (precno=0; precno<res->pw*res->ph; precno++) {
                        int tlcblkxstart, tlcblkystart, brcblkxend, brcblkyend;
                        int cbgxstart=tlcbgxstart+(precno%res->pw)*(1<<cbgwidthexpn);
                        int cbgystart=tlcbgystart+(precno/res->pw)*(1<<cbgheightexpn);
                        int cbgxend=cbgxstart+(1<<cbgwidthexpn);
                        int cbgyend=cbgystart+(1<<cbgheightexpn);
                        tcd_precinct_t *prc=&band->precincts[precno];
			// precinct size (global)
                        prc->x0=int_max(cbgxstart, band->x0);
                        prc->y0=int_max(cbgystart, band->y0);
                        prc->x1=int_min(cbgxend, band->x1);
                        prc->y1=int_min(cbgyend, band->y1);

                        tlcblkxstart=int_floordivpow2(prc->x0, cblkwidthexpn)<<cblkwidthexpn;
                        tlcblkystart=int_floordivpow2(prc->y0, cblkheightexpn)<<cblkheightexpn;
                        brcblkxend=int_ceildivpow2(prc->x1, cblkwidthexpn)<<cblkwidthexpn;
                        brcblkyend=int_ceildivpow2(prc->y1, cblkheightexpn)<<cblkheightexpn;
                        prc->cw=(brcblkxend-tlcblkxstart)>>cblkwidthexpn;
                        prc->ch=(brcblkyend-tlcblkystart)>>cblkheightexpn;

                        prc->cblks=(tcd_cblk_t*)malloc(prc->cw*prc->ch*sizeof(tcd_cblk_t));

                        prc->incltree=tgt_create(prc->cw, prc->ch);
                        prc->imsbtree=tgt_create(prc->cw, prc->ch);

                        for (cblkno=0; cblkno<prc->cw*prc->ch; cblkno++) {
                            int cblkxstart=tlcblkxstart+(cblkno%prc->cw)*(1<<cblkwidthexpn);
                            int cblkystart=tlcblkystart+(cblkno/prc->cw)*(1<<cblkheightexpn);
                            int cblkxend=cblkxstart+(1<<cblkwidthexpn);
                            int cblkyend=cblkystart+(1<<cblkheightexpn);
                            tcd_cblk_t *cblk=&prc->cblks[cblkno];
			    // code-block size (global)
                            cblk->x0=int_max(cblkxstart, prc->x0);
                            cblk->y0=int_max(cblkystart, prc->y0);
                            cblk->x1=int_min(cblkxend, prc->x1);
                            cblk->y1=int_min(cblkyend, prc->y1);
                        }
                    }
                }
            }
        }
    }
}


void tcd_free(j2k_image_t *img, j2k_cp_t *cp) {
  int tileno, compno, resno, bandno, precno;
  tcd_img=img;
  tcd_cp=cp;
  tcd_image.tw=cp->tw;
  tcd_image.th=cp->th;
  for (tileno=0; tileno<tcd_image.tw*tcd_image.th; tileno++) 
    {
      //  j2k_tcp_t *tcp=&cp->tcps[curtileno];
      tcd_tile_t *tile=&tcd_image.tiles[tileno];
      for (compno=0; compno<tile->numcomps; compno++) 
	{
	 tcd_tilecomp_t *tilec=&tile->comps[compno];
	  for (resno=0; resno<tilec->numresolutions; resno++) 
	    {
	      tcd_resolution_t *res=&tilec->resolutions[resno];
	      for (bandno=0; bandno<res->numbands; bandno++) 
		{
		 tcd_band_t *band=&res->bands[bandno];
		  for (precno=0; precno<res->pw*res->ph; precno++) 
		    {
		     tcd_precinct_t *prc=&band->precincts[precno];
		      
		      if (prc->incltree!=NULL)
			tgt_destroy(prc->incltree);
		      if (prc->imsbtree!=NULL)
			tgt_destroy(prc->imsbtree);
		      free(prc->cblks);
		    } // for (precno
		  free(band->precincts);
		} // for (bandno
	    }	// for (resno
	  free(tilec->resolutions);
	}	// for (compno
      free(tile->comps);
    }	// for (tileno
  free(tcd_image.tiles);
}


int tcd_decode_tile(unsigned char *src, int len, int tileno, info_image_t *imgg) {
    int l;
    int eof=0;
    tcd_tile_t *tile;

    tile = &tcd_image.tiles[tileno];

    l = t2_decode_packets(src, len, tcd_img, tcd_cp, tileno, tile, imgg);

    if (l==-999)
      {
	eof=1;
	fprintf(stderr, "tcd_decode: incomplete bistream\n");
      }
    
     if (eof) {
       longjmp(j2k_error, 1);
     }

    l=1;
    return l;
}
