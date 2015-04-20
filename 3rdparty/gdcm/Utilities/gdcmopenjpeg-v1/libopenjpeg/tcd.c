/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
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

#include "opj_includes.h"

void tcd_dump(FILE *fd, opj_tcd_t *tcd, opj_tcd_image_t * img) {
	int tileno, compno, resno, bandno, precno;//, cblkno;

	fprintf(fd, "image {\n");
	fprintf(fd, "  tw=%d, th=%d x0=%d x1=%d y0=%d y1=%d\n", 
		img->tw, img->th, tcd->image->x0, tcd->image->x1, tcd->image->y0, tcd->image->y1);

	for (tileno = 0; tileno < img->th * img->tw; tileno++) {
		opj_tcd_tile_t *tile = &tcd->tcd_image->tiles[tileno];
		fprintf(fd, "  tile {\n");
		fprintf(fd, "    x0=%d, y0=%d, x1=%d, y1=%d, numcomps=%d\n",
			tile->x0, tile->y0, tile->x1, tile->y1, tile->numcomps);
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
			fprintf(fd, "    tilec {\n");
			fprintf(fd,
				"      x0=%d, y0=%d, x1=%d, y1=%d, numresolutions=%d\n",
				tilec->x0, tilec->y0, tilec->x1, tilec->y1, tilec->numresolutions);
			for (resno = 0; resno < tilec->numresolutions; resno++) {
				opj_tcd_resolution_t *res = &tilec->resolutions[resno];
				fprintf(fd, "\n   res {\n");
				fprintf(fd,
					"          x0=%d, y0=%d, x1=%d, y1=%d, pw=%d, ph=%d, numbands=%d\n",
					res->x0, res->y0, res->x1, res->y1, res->pw, res->ph, res->numbands);
				for (bandno = 0; bandno < res->numbands; bandno++) {
					opj_tcd_band_t *band = &res->bands[bandno];
					fprintf(fd, "        band {\n");
					fprintf(fd,
						"          x0=%d, y0=%d, x1=%d, y1=%d, stepsize=%f, numbps=%d\n",
						band->x0, band->y0, band->x1, band->y1, band->stepsize, band->numbps);
					for (precno = 0; precno < res->pw * res->ph; precno++) {
						opj_tcd_precinct_t *prec = &band->precincts[precno];
						fprintf(fd, "          prec {\n");
						fprintf(fd,
							"            x0=%d, y0=%d, x1=%d, y1=%d, cw=%d, ch=%d\n",
							prec->x0, prec->y0, prec->x1, prec->y1, prec->cw, prec->ch);
						/*
						for (cblkno = 0; cblkno < prec->cw * prec->ch; cblkno++) {
							opj_tcd_cblk_t *cblk = &prec->cblks[cblkno];
							fprintf(fd, "            cblk {\n");
							fprintf(fd,
								"              x0=%d, y0=%d, x1=%d, y1=%d\n",
								cblk->x0, cblk->y0, cblk->x1, cblk->y1);
							fprintf(fd, "            }\n");
						}
						*/
						fprintf(fd, "          }\n");
					}
					fprintf(fd, "        }\n");
				}
				fprintf(fd, "      }\n");
			}
			fprintf(fd, "    }\n");
		}
		fprintf(fd, "  }\n");
	}
	fprintf(fd, "}\n");
}

/* ----------------------------------------------------------------------- */

/**
Create a new TCD handle
*/
opj_tcd_t* tcd_create(opj_common_ptr cinfo) {
	/* create the tcd structure */
	opj_tcd_t *tcd = (opj_tcd_t*)opj_malloc(sizeof(opj_tcd_t));
	if(!tcd) return NULL;
	tcd->cinfo = cinfo;
	tcd->tcd_image = (opj_tcd_image_t*)opj_malloc(sizeof(opj_tcd_image_t));
	if(!tcd->tcd_image) {
		opj_free(tcd);
		return NULL;
	}

	return tcd;
}

/**
Destroy a previously created TCD handle
*/
void tcd_destroy(opj_tcd_t *tcd) {
	if(tcd) {
		opj_free(tcd->tcd_image);
		opj_free(tcd);
	}
}

/* ----------------------------------------------------------------------- */

void tcd_malloc_encode(opj_tcd_t *tcd, opj_image_t * image, opj_cp_t * cp, int curtileno) {
	int tileno, compno, resno, bandno, precno, cblkno;

	tcd->image = image;
	tcd->cp = cp;
	tcd->tcd_image->tw = cp->tw;
	tcd->tcd_image->th = cp->th;
	tcd->tcd_image->tiles = (opj_tcd_tile_t *) opj_malloc(sizeof(opj_tcd_tile_t));
	
	for (tileno = 0; tileno < 1; tileno++) {
		opj_tcp_t *tcp = &cp->tcps[curtileno];
		int j;

		/* cfr p59 ISO/IEC FDIS15444-1 : 2000 (18 august 2000) */
		int p = curtileno % cp->tw;	/* si numerotation matricielle .. */
		int q = curtileno / cp->tw;	/* .. coordonnees de la tile (q,p) q pour ligne et p pour colonne */

		/* opj_tcd_tile_t *tile=&tcd->tcd_image->tiles[tileno]; */
		opj_tcd_tile_t *tile = tcd->tcd_image->tiles;

		/* 4 borders of the tile rescale on the image if necessary */
		tile->x0 = int_max(cp->tx0 + p * cp->tdx, image->x0);
		tile->y0 = int_max(cp->ty0 + q * cp->tdy, image->y0);
		tile->x1 = int_min(cp->tx0 + (p + 1) * cp->tdx, image->x1);
		tile->y1 = int_min(cp->ty0 + (q + 1) * cp->tdy, image->y1);
		tile->numcomps = image->numcomps;
		/* tile->PPT=image->PPT;  */

		/* Modification of the RATE >> */
		for (j = 0; j < tcp->numlayers; j++) {
			tcp->rates[j] = tcp->rates[j] ? 
				cp->tp_on ? 
					(((float) (tile->numcomps 
					* (tile->x1 - tile->x0) 
					* (tile->y1 - tile->y0)
					* image->comps[0].prec))
					/(tcp->rates[j] * 8 * image->comps[0].dx * image->comps[0].dy)) - (((tcd->cur_totnum_tp - 1) * 14 )/ tcp->numlayers)
					:
				((float) (tile->numcomps 
					* (tile->x1 - tile->x0) 
					* (tile->y1 - tile->y0) 
					* image->comps[0].prec))/ 
					(tcp->rates[j] * 8 * image->comps[0].dx * image->comps[0].dy)
					: 0;

			if (tcp->rates[j]) {
				if (j && tcp->rates[j] < tcp->rates[j - 1] + 10) {
					tcp->rates[j] = tcp->rates[j - 1] + 20;
				} else {
					if (!j && tcp->rates[j] < 30)
						tcp->rates[j] = 30;
				}
				
				if(j == (tcp->numlayers-1)){
					tcp->rates[j] = tcp->rates[j]- 2;
				}
			}
		}
		/* << Modification of the RATE */
		
		tile->comps = (opj_tcd_tilecomp_t *) opj_malloc(image->numcomps * sizeof(opj_tcd_tilecomp_t));
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tccp_t *tccp = &tcp->tccps[compno];

			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];

			/* border of each tile component (global) */
			tilec->x0 = int_ceildiv(tile->x0, image->comps[compno].dx);
			tilec->y0 = int_ceildiv(tile->y0, image->comps[compno].dy);
			tilec->x1 = int_ceildiv(tile->x1, image->comps[compno].dx);
			tilec->y1 = int_ceildiv(tile->y1, image->comps[compno].dy);
			
			tilec->data = (int *) opj_aligned_malloc((tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0) * sizeof(int));
			tilec->numresolutions = tccp->numresolutions;

			tilec->resolutions = (opj_tcd_resolution_t *) opj_malloc(tilec->numresolutions * sizeof(opj_tcd_resolution_t));
			
			for (resno = 0; resno < tilec->numresolutions; resno++) {
				int pdx, pdy;
				int levelno = tilec->numresolutions - 1 - resno;
				int tlprcxstart, tlprcystart, brprcxend, brprcyend;
				int tlcbgxstart, tlcbgystart, brcbgxend, brcbgyend;
				int cbgwidthexpn, cbgheightexpn;
				int cblkwidthexpn, cblkheightexpn;

				opj_tcd_resolution_t *res = &tilec->resolutions[resno];
				
				/* border for each resolution level (global) */
				res->x0 = int_ceildivpow2(tilec->x0, levelno);
				res->y0 = int_ceildivpow2(tilec->y0, levelno);
				res->x1 = int_ceildivpow2(tilec->x1, levelno);
				res->y1 = int_ceildivpow2(tilec->y1, levelno);
				
				res->numbands = resno == 0 ? 1 : 3;
				/* p. 35, table A-23, ISO/IEC FDIS154444-1 : 2000 (18 august 2000) */
				if (tccp->csty & J2K_CCP_CSTY_PRT) {
					pdx = tccp->prcw[resno];
					pdy = tccp->prch[resno];
				} else {
					pdx = 15;
					pdy = 15;
				}
				/* p. 64, B.6, ISO/IEC FDIS15444-1 : 2000 (18 august 2000)  */
				tlprcxstart = int_floordivpow2(res->x0, pdx) << pdx;
				tlprcystart = int_floordivpow2(res->y0, pdy) << pdy;
				
				brprcxend = int_ceildivpow2(res->x1, pdx) << pdx;
				brprcyend = int_ceildivpow2(res->y1, pdy) << pdy;
				
				res->pw = (brprcxend - tlprcxstart) >> pdx;
				res->ph = (brprcyend - tlprcystart) >> pdy;
				
				if (resno == 0) {
					tlcbgxstart = tlprcxstart;
					tlcbgystart = tlprcystart;
					brcbgxend = brprcxend;
					brcbgyend = brprcyend;
					cbgwidthexpn = pdx;
					cbgheightexpn = pdy;
				} else {
					tlcbgxstart = int_ceildivpow2(tlprcxstart, 1);
					tlcbgystart = int_ceildivpow2(tlprcystart, 1);
					brcbgxend = int_ceildivpow2(brprcxend, 1);
					brcbgyend = int_ceildivpow2(brprcyend, 1);
					cbgwidthexpn = pdx - 1;
					cbgheightexpn = pdy - 1;
				}
				
				cblkwidthexpn = int_min(tccp->cblkw, cbgwidthexpn);
				cblkheightexpn = int_min(tccp->cblkh, cbgheightexpn);
				
				for (bandno = 0; bandno < res->numbands; bandno++) {
					int x0b, y0b, i;
					int gain, numbps;
					opj_stepsize_t *ss = NULL;

					opj_tcd_band_t *band = &res->bands[bandno];

					band->bandno = resno == 0 ? 0 : bandno + 1;
					x0b = (band->bandno == 1) || (band->bandno == 3) ? 1 : 0;
					y0b = (band->bandno == 2) || (band->bandno == 3) ? 1 : 0;
					
					if (band->bandno == 0) {
						/* band border (global) */
						band->x0 = int_ceildivpow2(tilec->x0, levelno);
						band->y0 = int_ceildivpow2(tilec->y0, levelno);
						band->x1 = int_ceildivpow2(tilec->x1, levelno);
						band->y1 = int_ceildivpow2(tilec->y1, levelno);
					} else {
						/* band border (global) */
						band->x0 = int_ceildivpow2(tilec->x0 - (1 << levelno) * x0b, levelno + 1);
						band->y0 = int_ceildivpow2(tilec->y0 - (1 << levelno) * y0b, levelno + 1);
						band->x1 = int_ceildivpow2(tilec->x1 - (1 << levelno) * x0b, levelno + 1);
						band->y1 = int_ceildivpow2(tilec->y1 - (1 << levelno) * y0b, levelno + 1);
					}
					
					ss = &tccp->stepsizes[resno == 0 ? 0 : 3 * (resno - 1) + bandno + 1];
					gain = tccp->qmfbid == 0 ? dwt_getgain_real(band->bandno) : dwt_getgain(band->bandno);					
					numbps = image->comps[compno].prec + gain;
					
					band->stepsize = (float)((1.0 + ss->mant / 2048.0) * pow(2.0, numbps - ss->expn));
					band->numbps = ss->expn + tccp->numgbits - 1;	/* WHY -1 ? */
					
					band->precincts = (opj_tcd_precinct_t *) opj_malloc(3 * res->pw * res->ph * sizeof(opj_tcd_precinct_t));
					
					for (i = 0; i < res->pw * res->ph * 3; i++) {
						band->precincts[i].imsbtree = NULL;
						band->precincts[i].incltree = NULL;
					}
					
					for (precno = 0; precno < res->pw * res->ph; precno++) {
						int tlcblkxstart, tlcblkystart, brcblkxend, brcblkyend;

						int cbgxstart = tlcbgxstart + (precno % res->pw) * (1 << cbgwidthexpn);
						int cbgystart = tlcbgystart + (precno / res->pw) * (1 << cbgheightexpn);
						int cbgxend = cbgxstart + (1 << cbgwidthexpn);
						int cbgyend = cbgystart + (1 << cbgheightexpn);

						opj_tcd_precinct_t *prc = &band->precincts[precno];

						/* precinct size (global) */
						prc->x0 = int_max(cbgxstart, band->x0);
						prc->y0 = int_max(cbgystart, band->y0);
						prc->x1 = int_min(cbgxend, band->x1);
						prc->y1 = int_min(cbgyend, band->y1);

						tlcblkxstart = int_floordivpow2(prc->x0, cblkwidthexpn) << cblkwidthexpn;
						tlcblkystart = int_floordivpow2(prc->y0, cblkheightexpn) << cblkheightexpn;
						brcblkxend = int_ceildivpow2(prc->x1, cblkwidthexpn) << cblkwidthexpn;
						brcblkyend = int_ceildivpow2(prc->y1, cblkheightexpn) << cblkheightexpn;
						prc->cw = (brcblkxend - tlcblkxstart) >> cblkwidthexpn;
						prc->ch = (brcblkyend - tlcblkystart) >> cblkheightexpn;

						prc->cblks.enc = (opj_tcd_cblk_enc_t*) opj_calloc((prc->cw * prc->ch), sizeof(opj_tcd_cblk_enc_t));
						prc->incltree = tgt_create(prc->cw, prc->ch);
						prc->imsbtree = tgt_create(prc->cw, prc->ch);
						
						for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
							int cblkxstart = tlcblkxstart + (cblkno % prc->cw) * (1 << cblkwidthexpn);
							int cblkystart = tlcblkystart + (cblkno / prc->cw) * (1 << cblkheightexpn);
							int cblkxend = cblkxstart + (1 << cblkwidthexpn);
							int cblkyend = cblkystart + (1 << cblkheightexpn);
							
							opj_tcd_cblk_enc_t* cblk = &prc->cblks.enc[cblkno];

							/* code-block size (global) */
							cblk->x0 = int_max(cblkxstart, prc->x0);
							cblk->y0 = int_max(cblkystart, prc->y0);
							cblk->x1 = int_min(cblkxend, prc->x1);
							cblk->y1 = int_min(cblkyend, prc->y1);
							cblk->data = (unsigned char*) opj_calloc(8192+2, sizeof(unsigned char));
							/* FIXME: mqc_init_enc and mqc_byteout underrun the buffer if we don't do this. Why? */
							cblk->data += 2;
							cblk->layers = (opj_tcd_layer_t*) opj_calloc(100, sizeof(opj_tcd_layer_t));
							cblk->passes = (opj_tcd_pass_t*) opj_calloc(100, sizeof(opj_tcd_pass_t));
						}
					}
				}
			}
		}
	}
	
	/* tcd_dump(stdout, tcd, &tcd->tcd_image); */
}

void tcd_free_encode(opj_tcd_t *tcd) {
	int tileno, compno, resno, bandno, precno, cblkno;

	for (tileno = 0; tileno < 1; tileno++) {
		opj_tcd_tile_t *tile = tcd->tcd_image->tiles;

		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];

			for (resno = 0; resno < tilec->numresolutions; resno++) {
				opj_tcd_resolution_t *res = &tilec->resolutions[resno];

				for (bandno = 0; bandno < res->numbands; bandno++) {
					opj_tcd_band_t *band = &res->bands[bandno];

					for (precno = 0; precno < res->pw * res->ph; precno++) {
						opj_tcd_precinct_t *prc = &band->precincts[precno];

						if (prc->incltree != NULL) {
							tgt_destroy(prc->incltree);
							prc->incltree = NULL;
						}
						if (prc->imsbtree != NULL) {
							tgt_destroy(prc->imsbtree);	
							prc->imsbtree = NULL;
						}
						for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
							opj_free(prc->cblks.enc[cblkno].data - 2);
							opj_free(prc->cblks.enc[cblkno].layers);
							opj_free(prc->cblks.enc[cblkno].passes);
						}
						opj_free(prc->cblks.enc);
					} /* for (precno */
					opj_free(band->precincts);
					band->precincts = NULL;
				} /* for (bandno */
			} /* for (resno */
			opj_free(tilec->resolutions);
			tilec->resolutions = NULL;
		} /* for (compno */
		opj_free(tile->comps);
		tile->comps = NULL;
	} /* for (tileno */
	opj_free(tcd->tcd_image->tiles);
	tcd->tcd_image->tiles = NULL;
}

void tcd_init_encode(opj_tcd_t *tcd, opj_image_t * image, opj_cp_t * cp, int curtileno) {
	int tileno, compno, resno, bandno, precno, cblkno;

	for (tileno = 0; tileno < 1; tileno++) {
		opj_tcp_t *tcp = &cp->tcps[curtileno];
		int j;
		/* cfr p59 ISO/IEC FDIS15444-1 : 2000 (18 august 2000) */
		int p = curtileno % cp->tw;
		int q = curtileno / cp->tw;

		opj_tcd_tile_t *tile = tcd->tcd_image->tiles;
		
		/* 4 borders of the tile rescale on the image if necessary */
		tile->x0 = int_max(cp->tx0 + p * cp->tdx, image->x0);
		tile->y0 = int_max(cp->ty0 + q * cp->tdy, image->y0);
		tile->x1 = int_min(cp->tx0 + (p + 1) * cp->tdx, image->x1);
		tile->y1 = int_min(cp->ty0 + (q + 1) * cp->tdy, image->y1);
		
		tile->numcomps = image->numcomps;
		/* tile->PPT=image->PPT; */

		/* Modification of the RATE >> */
		for (j = 0; j < tcp->numlayers; j++) {
			tcp->rates[j] = tcp->rates[j] ? 
				cp->tp_on ? 
					(((float) (tile->numcomps 
					* (tile->x1 - tile->x0) 
					* (tile->y1 - tile->y0)
					* image->comps[0].prec))
					/(tcp->rates[j] * 8 * image->comps[0].dx * image->comps[0].dy)) - (((tcd->cur_totnum_tp - 1) * 14 )/ tcp->numlayers)
					:
				((float) (tile->numcomps 
					* (tile->x1 - tile->x0) 
					* (tile->y1 - tile->y0) 
					* image->comps[0].prec))/ 
					(tcp->rates[j] * 8 * image->comps[0].dx * image->comps[0].dy)
					: 0;

			if (tcp->rates[j]) {
				if (j && tcp->rates[j] < tcp->rates[j - 1] + 10) {
					tcp->rates[j] = tcp->rates[j - 1] + 20;
				} else {
					if (!j && tcp->rates[j] < 30)
						tcp->rates[j] = 30;
				}
			}
		}
		/* << Modification of the RATE */

		/* tile->comps=(opj_tcd_tilecomp_t*)opj_realloc(tile->comps,image->numcomps*sizeof(opj_tcd_tilecomp_t)); */
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tccp_t *tccp = &tcp->tccps[compno];
			
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];

			/* border of each tile component (global) */
			tilec->x0 = int_ceildiv(tile->x0, image->comps[compno].dx);
			tilec->y0 = int_ceildiv(tile->y0, image->comps[compno].dy);
			tilec->x1 = int_ceildiv(tile->x1, image->comps[compno].dx);
			tilec->y1 = int_ceildiv(tile->y1, image->comps[compno].dy);
			
			tilec->data = (int *) opj_aligned_malloc((tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0) * sizeof(int));
			tilec->numresolutions = tccp->numresolutions;
			/* tilec->resolutions=(opj_tcd_resolution_t*)opj_realloc(tilec->resolutions,tilec->numresolutions*sizeof(opj_tcd_resolution_t)); */
			for (resno = 0; resno < tilec->numresolutions; resno++) {
				int pdx, pdy;

				int levelno = tilec->numresolutions - 1 - resno;
				int tlprcxstart, tlprcystart, brprcxend, brprcyend;
				int tlcbgxstart, tlcbgystart, brcbgxend, brcbgyend;
				int cbgwidthexpn, cbgheightexpn;
				int cblkwidthexpn, cblkheightexpn;
				
				opj_tcd_resolution_t *res = &tilec->resolutions[resno];

				/* border for each resolution level (global) */
				res->x0 = int_ceildivpow2(tilec->x0, levelno);
				res->y0 = int_ceildivpow2(tilec->y0, levelno);
				res->x1 = int_ceildivpow2(tilec->x1, levelno);
				res->y1 = int_ceildivpow2(tilec->y1, levelno);	
				res->numbands = resno == 0 ? 1 : 3;

				/* p. 35, table A-23, ISO/IEC FDIS154444-1 : 2000 (18 august 2000) */
				if (tccp->csty & J2K_CCP_CSTY_PRT) {
					pdx = tccp->prcw[resno];
					pdy = tccp->prch[resno];
				} else {
					pdx = 15;
					pdy = 15;
				}
				/* p. 64, B.6, ISO/IEC FDIS15444-1 : 2000 (18 august 2000)  */
				tlprcxstart = int_floordivpow2(res->x0, pdx) << pdx;
				tlprcystart = int_floordivpow2(res->y0, pdy) << pdy;
				brprcxend = int_ceildivpow2(res->x1, pdx) << pdx;
				brprcyend = int_ceildivpow2(res->y1, pdy) << pdy;
				
				res->pw = (brprcxend - tlprcxstart) >> pdx;
				res->ph = (brprcyend - tlprcystart) >> pdy;
				
				if (resno == 0) {
					tlcbgxstart = tlprcxstart;
					tlcbgystart = tlprcystart;
					brcbgxend = brprcxend;
					brcbgyend = brprcyend;
					cbgwidthexpn = pdx;
					cbgheightexpn = pdy;
				} else {
					tlcbgxstart = int_ceildivpow2(tlprcxstart, 1);
					tlcbgystart = int_ceildivpow2(tlprcystart, 1);
					brcbgxend = int_ceildivpow2(brprcxend, 1);
					brcbgyend = int_ceildivpow2(brprcyend, 1);
					cbgwidthexpn = pdx - 1;
					cbgheightexpn = pdy - 1;
				}
				
				cblkwidthexpn = int_min(tccp->cblkw, cbgwidthexpn);
				cblkheightexpn = int_min(tccp->cblkh, cbgheightexpn);
				
				for (bandno = 0; bandno < res->numbands; bandno++) {
					int x0b, y0b;
					int gain, numbps;
					opj_stepsize_t *ss = NULL;

					opj_tcd_band_t *band = &res->bands[bandno];

					band->bandno = resno == 0 ? 0 : bandno + 1;
					x0b = (band->bandno == 1) || (band->bandno == 3) ? 1 : 0;
					y0b = (band->bandno == 2) || (band->bandno == 3) ? 1 : 0;
					
					if (band->bandno == 0) {
						/* band border */
						band->x0 = int_ceildivpow2(tilec->x0, levelno);
						band->y0 = int_ceildivpow2(tilec->y0, levelno);
						band->x1 = int_ceildivpow2(tilec->x1, levelno);
						band->y1 = int_ceildivpow2(tilec->y1, levelno);
					} else {
						band->x0 = int_ceildivpow2(tilec->x0 - (1 << levelno) * x0b, levelno + 1);
						band->y0 = int_ceildivpow2(tilec->y0 - (1 << levelno) * y0b, levelno + 1);
						band->x1 = int_ceildivpow2(tilec->x1 - (1 << levelno) * x0b, levelno + 1);
						band->y1 = int_ceildivpow2(tilec->y1 - (1 << levelno) * y0b, levelno + 1);
					}
					
					ss = &tccp->stepsizes[resno == 0 ? 0 : 3 * (resno - 1) + bandno + 1];
					gain = tccp->qmfbid == 0 ? dwt_getgain_real(band->bandno) : dwt_getgain(band->bandno);
					numbps = image->comps[compno].prec + gain;
					band->stepsize = (float)((1.0 + ss->mant / 2048.0) * pow(2.0, numbps - ss->expn));
					band->numbps = ss->expn + tccp->numgbits - 1;	/* WHY -1 ? */
					
					for (precno = 0; precno < res->pw * res->ph; precno++) {
						int tlcblkxstart, tlcblkystart, brcblkxend, brcblkyend;

						int cbgxstart = tlcbgxstart + (precno % res->pw) * (1 << cbgwidthexpn);
						int cbgystart = tlcbgystart + (precno / res->pw) * (1 << cbgheightexpn);
						int cbgxend = cbgxstart + (1 << cbgwidthexpn);
						int cbgyend = cbgystart + (1 << cbgheightexpn);
						
						opj_tcd_precinct_t *prc = &band->precincts[precno];

						/* precinct size (global) */
						prc->x0 = int_max(cbgxstart, band->x0);
						prc->y0 = int_max(cbgystart, band->y0);
						prc->x1 = int_min(cbgxend, band->x1);
						prc->y1 = int_min(cbgyend, band->y1);

						tlcblkxstart = int_floordivpow2(prc->x0, cblkwidthexpn) << cblkwidthexpn;
						tlcblkystart = int_floordivpow2(prc->y0, cblkheightexpn) << cblkheightexpn;
						brcblkxend = int_ceildivpow2(prc->x1, cblkwidthexpn) << cblkwidthexpn;
						brcblkyend = int_ceildivpow2(prc->y1, cblkheightexpn) << cblkheightexpn;
						prc->cw = (brcblkxend - tlcblkxstart) >> cblkwidthexpn;
						prc->ch = (brcblkyend - tlcblkystart) >> cblkheightexpn;

						opj_free(prc->cblks.enc);
						prc->cblks.enc = (opj_tcd_cblk_enc_t*) opj_calloc(prc->cw * prc->ch, sizeof(opj_tcd_cblk_enc_t));

						if (prc->incltree != NULL) {
							tgt_destroy(prc->incltree);
						}
						if (prc->imsbtree != NULL) {
							tgt_destroy(prc->imsbtree);
						}
						
						prc->incltree = tgt_create(prc->cw, prc->ch);
						prc->imsbtree = tgt_create(prc->cw, prc->ch);

						for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
							int cblkxstart = tlcblkxstart + (cblkno % prc->cw) * (1 << cblkwidthexpn);
							int cblkystart = tlcblkystart + (cblkno / prc->cw) * (1 << cblkheightexpn);
							int cblkxend = cblkxstart + (1 << cblkwidthexpn);
							int cblkyend = cblkystart + (1 << cblkheightexpn);

							opj_tcd_cblk_enc_t* cblk = &prc->cblks.enc[cblkno];

							/* code-block size (global) */
							cblk->x0 = int_max(cblkxstart, prc->x0);
							cblk->y0 = int_max(cblkystart, prc->y0);
							cblk->x1 = int_min(cblkxend, prc->x1);
							cblk->y1 = int_min(cblkyend, prc->y1);
							cblk->data = (unsigned char*) opj_calloc(8192+2, sizeof(unsigned char));
							/* FIXME: mqc_init_enc and mqc_byteout underrun the buffer if we don't do this. Why? */
							cblk->data += 2;
							cblk->layers = (opj_tcd_layer_t*) opj_calloc(100, sizeof(opj_tcd_layer_t));
							cblk->passes = (opj_tcd_pass_t*) opj_calloc(100, sizeof(opj_tcd_pass_t));
						}
					} /* precno */
				} /* bandno */
			} /* resno */
		} /* compno */
	} /* tileno */

	/* tcd_dump(stdout, tcd, &tcd->tcd_image); */
}

void tcd_malloc_decode(opj_tcd_t *tcd, opj_image_t * image, opj_cp_t * cp) {
	int i, j, tileno, p, q;
	unsigned int x0 = 0, y0 = 0, x1 = 0, y1 = 0, w, h;

	tcd->image = image;
	tcd->tcd_image->tw = cp->tw;
	tcd->tcd_image->th = cp->th;
	tcd->tcd_image->tiles = (opj_tcd_tile_t *) opj_malloc(cp->tw * cp->th * sizeof(opj_tcd_tile_t));

	/* 
	Allocate place to store the decoded data = final image
	Place limited by the tile really present in the codestream 
	*/

	for (j = 0; j < cp->tileno_size; j++) {
		opj_tcd_tile_t *tile;
		
		tileno = cp->tileno[j];		
		tile = &(tcd->tcd_image->tiles[cp->tileno[tileno]]);		
		tile->numcomps = image->numcomps;
		tile->comps = (opj_tcd_tilecomp_t*) opj_calloc(image->numcomps, sizeof(opj_tcd_tilecomp_t));
	}

	for (i = 0; i < image->numcomps; i++) {
		for (j = 0; j < cp->tileno_size; j++) {
			opj_tcd_tile_t *tile;
			opj_tcd_tilecomp_t *tilec;
			
			/* cfr p59 ISO/IEC FDIS15444-1 : 2000 (18 august 2000) */
			
			tileno = cp->tileno[j];
			
			tile = &(tcd->tcd_image->tiles[cp->tileno[tileno]]);
			tilec = &tile->comps[i];
			
			p = tileno % cp->tw;	/* si numerotation matricielle .. */
			q = tileno / cp->tw;	/* .. coordonnees de la tile (q,p) q pour ligne et p pour colonne */
			
			/* 4 borders of the tile rescale on the image if necessary */
			tile->x0 = int_max(cp->tx0 + p * cp->tdx, image->x0);
			tile->y0 = int_max(cp->ty0 + q * cp->tdy, image->y0);
			tile->x1 = int_min(cp->tx0 + (p + 1) * cp->tdx, image->x1);
			tile->y1 = int_min(cp->ty0 + (q + 1) * cp->tdy, image->y1);

			tilec->x0 = int_ceildiv(tile->x0, image->comps[i].dx);
			tilec->y0 = int_ceildiv(tile->y0, image->comps[i].dy);
			tilec->x1 = int_ceildiv(tile->x1, image->comps[i].dx);
			tilec->y1 = int_ceildiv(tile->y1, image->comps[i].dy);

			x0 = j == 0 ? tilec->x0 : int_min(x0, (unsigned int) tilec->x0);
			y0 = j == 0 ? tilec->y0 : int_min(y0,	(unsigned int) tilec->x0);
			x1 = j == 0 ? tilec->x1 : int_max(x1,	(unsigned int) tilec->x1);
			y1 = j == 0 ? tilec->y1 : int_max(y1,	(unsigned int) tilec->y1);
		}

		w = int_ceildivpow2(x1 - x0, image->comps[i].factor);
		h = int_ceildivpow2(y1 - y0, image->comps[i].factor);

		image->comps[i].w = w;
		image->comps[i].h = h;
		image->comps[i].x0 = x0;
		image->comps[i].y0 = y0;
	}
}

void tcd_malloc_decode_tile(opj_tcd_t *tcd, opj_image_t * image, opj_cp_t * cp, int tileno, opj_codestream_info_t *cstr_info) {
	int compno, resno, bandno, precno, cblkno;
	opj_tcp_t *tcp;
	opj_tcd_tile_t *tile;

	tcd->cp = cp;
	
	tcp = &(cp->tcps[cp->tileno[tileno]]);
	tile = &(tcd->tcd_image->tiles[cp->tileno[tileno]]);
	
	tileno = cp->tileno[tileno];
	
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tccp_t *tccp = &tcp->tccps[compno];
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
		
		/* border of each tile component (global) */
		tilec->x0 = int_ceildiv(tile->x0, image->comps[compno].dx);
		tilec->y0 = int_ceildiv(tile->y0, image->comps[compno].dy);
		tilec->x1 = int_ceildiv(tile->x1, image->comps[compno].dx);
		tilec->y1 = int_ceildiv(tile->y1, image->comps[compno].dy);

		tilec->numresolutions = tccp->numresolutions;
		tilec->resolutions = (opj_tcd_resolution_t *) opj_malloc(tilec->numresolutions * sizeof(opj_tcd_resolution_t));
		
		for (resno = 0; resno < tilec->numresolutions; resno++) {
			int pdx, pdy;
			int levelno = tilec->numresolutions - 1 - resno;
			int tlprcxstart, tlprcystart, brprcxend, brprcyend;
			int tlcbgxstart, tlcbgystart, brcbgxend, brcbgyend;
			int cbgwidthexpn, cbgheightexpn;
			int cblkwidthexpn, cblkheightexpn;
			
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];
			
			/* border for each resolution level (global) */
			res->x0 = int_ceildivpow2(tilec->x0, levelno);
			res->y0 = int_ceildivpow2(tilec->y0, levelno);
			res->x1 = int_ceildivpow2(tilec->x1, levelno);
			res->y1 = int_ceildivpow2(tilec->y1, levelno);
			res->numbands = resno == 0 ? 1 : 3;
			
			/* p. 35, table A-23, ISO/IEC FDIS154444-1 : 2000 (18 august 2000) */
			if (tccp->csty & J2K_CCP_CSTY_PRT) {
				pdx = tccp->prcw[resno];
				pdy = tccp->prch[resno];
			} else {
				pdx = 15;
				pdy = 15;
			}			
			
			/* p. 64, B.6, ISO/IEC FDIS15444-1 : 2000 (18 august 2000)  */
			tlprcxstart = int_floordivpow2(res->x0, pdx) << pdx;
			tlprcystart = int_floordivpow2(res->y0, pdy) << pdy;
			brprcxend = int_ceildivpow2(res->x1, pdx) << pdx;
			brprcyend = int_ceildivpow2(res->y1, pdy) << pdy;
			
			res->pw = (res->x0 == res->x1) ? 0 : ((brprcxend - tlprcxstart) >> pdx);
			res->ph = (res->y0 == res->y1) ? 0 : ((brprcyend - tlprcystart) >> pdy);
			
			if (resno == 0) {
				tlcbgxstart = tlprcxstart;
				tlcbgystart = tlprcystart;
				brcbgxend = brprcxend;
				brcbgyend = brprcyend;
				cbgwidthexpn = pdx;
				cbgheightexpn = pdy;
			} else {
				tlcbgxstart = int_ceildivpow2(tlprcxstart, 1);
				tlcbgystart = int_ceildivpow2(tlprcystart, 1);
				brcbgxend = int_ceildivpow2(brprcxend, 1);
				brcbgyend = int_ceildivpow2(brprcyend, 1);
				cbgwidthexpn = pdx - 1;
				cbgheightexpn = pdy - 1;
			}
			
			cblkwidthexpn = int_min(tccp->cblkw, cbgwidthexpn);
			cblkheightexpn = int_min(tccp->cblkh, cbgheightexpn);
			
			for (bandno = 0; bandno < res->numbands; bandno++) {
				int x0b, y0b;
				int gain, numbps;
				opj_stepsize_t *ss = NULL;
				
				opj_tcd_band_t *band = &res->bands[bandno];
				band->bandno = resno == 0 ? 0 : bandno + 1;
				x0b = (band->bandno == 1) || (band->bandno == 3) ? 1 : 0;
				y0b = (band->bandno == 2) || (band->bandno == 3) ? 1 : 0;
				
				if (band->bandno == 0) {
					/* band border (global) */
					band->x0 = int_ceildivpow2(tilec->x0, levelno);
					band->y0 = int_ceildivpow2(tilec->y0, levelno);
					band->x1 = int_ceildivpow2(tilec->x1, levelno);
					band->y1 = int_ceildivpow2(tilec->y1, levelno);
				} else {
					/* band border (global) */
					band->x0 = int_ceildivpow2(tilec->x0 - (1 << levelno) * x0b, levelno + 1);
					band->y0 = int_ceildivpow2(tilec->y0 - (1 << levelno) * y0b, levelno + 1);
					band->x1 = int_ceildivpow2(tilec->x1 - (1 << levelno) * x0b, levelno + 1);
					band->y1 = int_ceildivpow2(tilec->y1 - (1 << levelno) * y0b, levelno + 1);
				}
				
				ss = &tccp->stepsizes[resno == 0 ? 0 : 3 * (resno - 1) + bandno + 1];
				gain = tccp->qmfbid == 0 ? dwt_getgain_real(band->bandno) : dwt_getgain(band->bandno);
				numbps = image->comps[compno].prec + gain;
				band->stepsize = (float)(((1.0 + ss->mant / 2048.0) * pow(2.0, numbps - ss->expn)) * 0.5);
				band->numbps = ss->expn + tccp->numgbits - 1;	/* WHY -1 ? */
				
				band->precincts = (opj_tcd_precinct_t *) opj_malloc(res->pw * res->ph * sizeof(opj_tcd_precinct_t));
				
				for (precno = 0; precno < res->pw * res->ph; precno++) {
					int tlcblkxstart, tlcblkystart, brcblkxend, brcblkyend;
					int cbgxstart = tlcbgxstart + (precno % res->pw) * (1 << cbgwidthexpn);
					int cbgystart = tlcbgystart + (precno / res->pw) * (1 << cbgheightexpn);
					int cbgxend = cbgxstart + (1 << cbgwidthexpn);
					int cbgyend = cbgystart + (1 << cbgheightexpn);
					
					opj_tcd_precinct_t *prc = &band->precincts[precno];
					/* precinct size (global) */
					prc->x0 = int_max(cbgxstart, band->x0);
					prc->y0 = int_max(cbgystart, band->y0);
					prc->x1 = int_min(cbgxend, band->x1);
					prc->y1 = int_min(cbgyend, band->y1);
					
					tlcblkxstart = int_floordivpow2(prc->x0, cblkwidthexpn) << cblkwidthexpn;
					tlcblkystart = int_floordivpow2(prc->y0, cblkheightexpn) << cblkheightexpn;
					brcblkxend = int_ceildivpow2(prc->x1, cblkwidthexpn) << cblkwidthexpn;
					brcblkyend = int_ceildivpow2(prc->y1, cblkheightexpn) << cblkheightexpn;
					prc->cw = (brcblkxend - tlcblkxstart) >> cblkwidthexpn;
					prc->ch = (brcblkyend - tlcblkystart) >> cblkheightexpn;

					prc->cblks.dec = (opj_tcd_cblk_dec_t*) opj_malloc(prc->cw * prc->ch * sizeof(opj_tcd_cblk_dec_t));

					prc->incltree = tgt_create(prc->cw, prc->ch);
					prc->imsbtree = tgt_create(prc->cw, prc->ch);
					
					for (cblkno = 0; cblkno < prc->cw * prc->ch; cblkno++) {
						int cblkxstart = tlcblkxstart + (cblkno % prc->cw) * (1 << cblkwidthexpn);
						int cblkystart = tlcblkystart + (cblkno / prc->cw) * (1 << cblkheightexpn);
						int cblkxend = cblkxstart + (1 << cblkwidthexpn);
						int cblkyend = cblkystart + (1 << cblkheightexpn);					

						opj_tcd_cblk_dec_t* cblk = &prc->cblks.dec[cblkno];
						cblk->data = NULL;
						cblk->segs = NULL;
						/* code-block size (global) */
						cblk->x0 = int_max(cblkxstart, prc->x0);
						cblk->y0 = int_max(cblkystart, prc->y0);
						cblk->x1 = int_min(cblkxend, prc->x1);
						cblk->y1 = int_min(cblkyend, prc->y1);
						cblk->numsegs = 0;
					}
				} /* precno */
			} /* bandno */
		} /* resno */
	} /* compno */
	/* tcd_dump(stdout, tcd, &tcd->tcd_image); */
}

void tcd_makelayer_fixed(opj_tcd_t *tcd, int layno, int final) {
	int compno, resno, bandno, precno, cblkno;
	int value;			/*, matrice[tcd_tcp->numlayers][tcd_tile->comps[0].numresolutions][3]; */
	int matrice[10][10][3];
	int i, j, k;

	opj_cp_t *cp = tcd->cp;
	opj_tcd_tile_t *tcd_tile = tcd->tcd_tile;
	opj_tcp_t *tcd_tcp = tcd->tcp;

	/*matrice=(int*)opj_malloc(tcd_tcp->numlayers*tcd_tile->comps[0].numresolutions*3*sizeof(int)); */
	
	for (compno = 0; compno < tcd_tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tcd_tile->comps[compno];
		for (i = 0; i < tcd_tcp->numlayers; i++) {
			for (j = 0; j < tilec->numresolutions; j++) {
				for (k = 0; k < 3; k++) {
					matrice[i][j][k] =
						(int) (cp->matrice[i * tilec->numresolutions * 3 + j * 3 + k] 
						* (float) (tcd->image->comps[compno].prec / 16.0));
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
						int n;
						int imsb = tcd->image->comps[compno].prec - cblk->numbps;	/* number of bit-plan equal to zero */
						/* Correction of the matrix of coefficient to include the IMSB information */
						if (layno == 0) {
							value = matrice[layno][resno][bandno];
							if (imsb >= value) {
								value = 0;
							} else {
								value -= imsb;
							}
						} else {
							value =	matrice[layno][resno][bandno] -	matrice[layno - 1][resno][bandno];
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
	int layno;
	for (layno = 0; layno < tcd->tcp->numlayers; layno++) {
		tcd_makelayer_fixed(tcd, layno, 1);
	}
}

void tcd_makelayer(opj_tcd_t *tcd, int layno, double thresh, int final) {
	int compno, resno, bandno, precno, cblkno, passno;
	
	opj_tcd_tile_t *tcd_tile = tcd->tcd_tile;

	tcd_tile->distolayer[layno] = 0;	/* fixed_quality */
	
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
						
						int n;
						if (layno == 0) {
							cblk->numpassesinlayers = 0;
						}
						n = cblk->numpassesinlayers;
						for (passno = cblk->numpassesinlayers; passno < cblk->totalpasses; passno++) {
							int dr;
							double dd;
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
							layer->len = cblk->passes[n - 1].rate -	cblk->passes[cblk->numpassesinlayers - 1].rate;
							layer->data = cblk->data + cblk->passes[cblk->numpassesinlayers - 1].rate;
							layer->disto = cblk->passes[n - 1].distortiondec - cblk->passes[cblk->numpassesinlayers - 1].distortiondec;
						}
						
						tcd_tile->distolayer[layno] += layer->disto;	/* fixed_quality */
						
						if (final)
							cblk->numpassesinlayers = n;
					}
				}
			}
		}
	}
}

bool tcd_rateallocate(opj_tcd_t *tcd, unsigned char *dest, int len, opj_codestream_info_t *cstr_info) {
	int compno, resno, bandno, precno, cblkno, passno, layno;
	double min, max;
	double cumdisto[100];	/* fixed_quality */
	const double K = 1;		/* 1.1; fixed_quality */
	double maxSE = 0;

	opj_cp_t *cp = tcd->cp;
	opj_tcd_tile_t *tcd_tile = tcd->tcd_tile;
	opj_tcp_t *tcd_tcp = tcd->tcp;

	min = DBL_MAX;
	max = 0;
	
	tcd_tile->numpix = 0;		/* fixed_quality */
	
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
							int dr;
							double dd, rdslope;
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
		
		maxSE += (((double)(1 << tcd->image->comps[compno].prec) - 1.0) 
			* ((double)(1 << tcd->image->comps[compno].prec) -1.0)) 
			* ((double)(tilec->numpix));
	} /* compno */
	
	/* index file */
	if(cstr_info) {
		opj_tile_info_t *tile_info = &cstr_info->tile[tcd->tcd_tileno];
		tile_info->numpix = tcd_tile->numpix;
		tile_info->distotile = tcd_tile->distotile;
		tile_info->thresh = (double *) opj_malloc(tcd_tcp->numlayers * sizeof(double));
	}
	
	for (layno = 0; layno < tcd_tcp->numlayers; layno++) {
		double lo = min;
		double hi = max;
		int success = 0;
		int maxlen = tcd_tcp->rates[layno] ? int_min(((int) ceil(tcd_tcp->rates[layno])), len) : len;
		double goodthresh = 0;
		double stable_thresh = 0;
		int i;
		double distotarget;		/* fixed_quality */
		
		/* fixed_quality */
		distotarget = tcd_tile->distotile - ((K * maxSE) / pow((float)10, tcd_tcp->distoratio[layno] / 10));
        
		/* Don't try to find an optimal threshold but rather take everything not included yet, if
		  -r xx,yy,zz,0   (disto_alloc == 1 and rates == 0)
		  -q xx,yy,zz,0	  (fixed_quality == 1 and distoratio == 0)
		  ==> possible to have some lossy layers and the last layer for sure lossless */
		if ( ((cp->disto_alloc==1) && (tcd_tcp->rates[layno]>0)) || ((cp->fixed_quality==1) && (tcd_tcp->distoratio[layno]>0))) {
			opj_t2_t *t2 = t2_create(tcd->cinfo, tcd->image, cp);
			double thresh = 0;

			for (i = 0; i < 128; i++) {
				int l = 0;
				double distoachieved = 0;	/* fixed_quality */
				thresh = (lo + hi) / 2;
				
				tcd_makelayer(tcd, layno, thresh, 0);
				
				if (cp->fixed_quality) {	/* fixed_quality */
					if(cp->cinema){
						l = t2_encode_packets(t2,tcd->tcd_tileno, tcd_tile, layno + 1, dest, maxlen, cstr_info,tcd->cur_tp_num,tcd->tp_pos,tcd->cur_pino,THRESH_CALC, tcd->cur_totnum_tp);
						if (l == -999) {
							lo = thresh;
							continue;
						}else{
           		distoachieved =	layno == 0 ? 
							tcd_tile->distolayer[0]	: cumdisto[layno - 1] + tcd_tile->distolayer[layno];
							if (distoachieved < distotarget) {
								hi=thresh; 
								stable_thresh = thresh;
								continue;
							}else{
								lo=thresh;
							}
						}
					}else{
						distoachieved =	(layno == 0) ? 
							tcd_tile->distolayer[0]	: (cumdisto[layno - 1] + tcd_tile->distolayer[layno]);
						if (distoachieved < distotarget) {
							hi = thresh;
							stable_thresh = thresh;
							continue;
						}
						lo = thresh;
					}
				} else {
					l = t2_encode_packets(t2, tcd->tcd_tileno, tcd_tile, layno + 1, dest, maxlen, cstr_info,tcd->cur_tp_num,tcd->tp_pos,tcd->cur_pino,THRESH_CALC, tcd->cur_totnum_tp);
					/* TODO: what to do with l ??? seek / tell ??? */
					/* opj_event_msg(tcd->cinfo, EVT_INFO, "rate alloc: len=%d, max=%d\n", l, maxlen); */
					if (l == -999) {
						lo = thresh;
						continue;
					}
					hi = thresh;
					stable_thresh = thresh;
				}
			}
			success = 1;
			goodthresh = stable_thresh == 0? thresh : stable_thresh;
			t2_destroy(t2);
		} else {
			success = 1;
			goodthresh = min;
		}
		
		if (!success) {
			return false;
		}
		
		if(cstr_info) {	/* Threshold for Marcela Index */
			cstr_info->tile[tcd->tcd_tileno].thresh[layno] = goodthresh;
		}
		tcd_makelayer(tcd, layno, goodthresh, 1);
        
		/* fixed_quality */
		cumdisto[layno] = (layno == 0) ? tcd_tile->distolayer[0] : (cumdisto[layno - 1] + tcd_tile->distolayer[layno]);	
	}

	return true;
}

int tcd_encode_tile(opj_tcd_t *tcd, int tileno, unsigned char *dest, int len, opj_codestream_info_t *cstr_info) {
	int compno;
	int l, i, numpacks = 0;
	opj_tcd_tile_t *tile = NULL;
	opj_tcp_t *tcd_tcp = NULL;
	opj_cp_t *cp = NULL;

	opj_tcp_t *tcp = &tcd->cp->tcps[0];
	opj_tccp_t *tccp = &tcp->tccps[0];
	opj_image_t *image = tcd->image;
	
	opj_t1_t *t1 = NULL;		/* T1 component */
	opj_t2_t *t2 = NULL;		/* T2 component */

	tcd->tcd_tileno = tileno;
	tcd->tcd_tile = tcd->tcd_image->tiles;
	tcd->tcp = &tcd->cp->tcps[tileno];

	tile = tcd->tcd_tile;
	tcd_tcp = tcd->tcp;
	cp = tcd->cp;

	if(tcd->cur_tp_num == 0){
		tcd->encoding_time = opj_clock();	/* time needed to encode a tile */
		/* INDEX >> "Precinct_nb_X et Precinct_nb_Y" */
		if(cstr_info) {
			opj_tcd_tilecomp_t *tilec_idx = &tile->comps[0];	/* based on component 0 */
			for (i = 0; i < tilec_idx->numresolutions; i++) {
				opj_tcd_resolution_t *res_idx = &tilec_idx->resolutions[i];
				
				cstr_info->tile[tileno].pw[i] = res_idx->pw;
				cstr_info->tile[tileno].ph[i] = res_idx->ph;
				
				numpacks += res_idx->pw * res_idx->ph;
				
				cstr_info->tile[tileno].pdx[i] = tccp->prcw[i];
				cstr_info->tile[tileno].pdy[i] = tccp->prch[i];
			}
			cstr_info->tile[tileno].packet = (opj_packet_info_t*) opj_calloc(cstr_info->numcomps * cstr_info->numlayers * numpacks, sizeof(opj_packet_info_t));
		}
		/* << INDEX */
		
		/*---------------TILE-------------------*/
		
		for (compno = 0; compno < tile->numcomps; compno++) {
			int x, y;
			
			int adjust = image->comps[compno].sgnd ? 0 : 1 << (image->comps[compno].prec - 1);
			int offset_x = int_ceildiv(image->x0, image->comps[compno].dx);
			int offset_y = int_ceildiv(image->y0, image->comps[compno].dy);
			
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
			int tw = tilec->x1 - tilec->x0;
			int w = int_ceildiv(image->x1 - image->x0, image->comps[compno].dx);
			
			/* extract tile data */
			
			if (tcd_tcp->tccps[compno].qmfbid == 1) {
				for (y = tilec->y0; y < tilec->y1; y++) {
					/* start of the src tile scanline */
					int *data = &image->comps[compno].data[(tilec->x0 - offset_x) + (y - offset_y) * w];
					/* start of the dst tile scanline */
					int *tile_data = &tilec->data[(y - tilec->y0) * tw];
					for (x = tilec->x0; x < tilec->x1; x++) {
						*tile_data++ = *data++ - adjust;
					}
				}
			} else if (tcd_tcp->tccps[compno].qmfbid == 0) {
				for (y = tilec->y0; y < tilec->y1; y++) {
					/* start of the src tile scanline */
					int *data = &image->comps[compno].data[(tilec->x0 - offset_x) + (y - offset_y) * w];
					/* start of the dst tile scanline */
					int *tile_data = &tilec->data[(y - tilec->y0) * tw];
					for (x = tilec->x0; x < tilec->x1; x++) {
						*tile_data++ = (*data++ - adjust) << 11;
					}
					
				}
			}
		}
		
		/*----------------MCT-------------------*/
		if (tcd_tcp->mct) {
			int samples = (tile->comps[0].x1 - tile->comps[0].x0) * (tile->comps[0].y1 - tile->comps[0].y0);
			if (tcd_tcp->tccps[0].qmfbid == 0) {
				mct_encode_real(tile->comps[0].data, tile->comps[1].data, tile->comps[2].data, samples);
			} else {
				mct_encode(tile->comps[0].data, tile->comps[1].data, tile->comps[2].data, samples);
			}
		}
		
		/*----------------DWT---------------------*/
		
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
			if (tcd_tcp->tccps[compno].qmfbid == 1) {
				dwt_encode(tilec);
			} else if (tcd_tcp->tccps[compno].qmfbid == 0) {
				dwt_encode_real(tilec);
			}
		}
		
		/*------------------TIER1-----------------*/
		t1 = t1_create(tcd->cinfo);
		t1_encode_cblks(t1, tile, tcd_tcp);
		t1_destroy(t1);
		
		/*-----------RATE-ALLOCATE------------------*/
		
		/* INDEX */
		if(cstr_info) {
			cstr_info->index_write = 0;
		}
		if (cp->disto_alloc || cp->fixed_quality) {	/* fixed_quality */
			/* Normal Rate/distortion allocation */
			tcd_rateallocate(tcd, dest, len, cstr_info);
		} else {
			/* Fixed layer allocation */
			tcd_rateallocate_fixed(tcd);
		}
	}
	/*--------------TIER2------------------*/

	/* INDEX */
	if(cstr_info) {
		cstr_info->index_write = 1;
	}

	t2 = t2_create(tcd->cinfo, image, cp);
	l = t2_encode_packets(t2,tileno, tile, tcd_tcp->numlayers, dest, len, cstr_info,tcd->tp_num,tcd->tp_pos,tcd->cur_pino,FINAL_PASS,tcd->cur_totnum_tp);
	t2_destroy(t2);
	
	/*---------------CLEAN-------------------*/

	
	if(tcd->cur_tp_num == tcd->cur_totnum_tp - 1){
		tcd->encoding_time = opj_clock() - tcd->encoding_time;
		opj_event_msg(tcd->cinfo, EVT_INFO, "- tile encoded in %f s\n", tcd->encoding_time);

		/* cleaning memory */
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
			opj_aligned_free(tilec->data);
		}
	}

	return l;
}

bool tcd_decode_tile(opj_tcd_t *tcd, unsigned char *src, int len, int tileno, opj_codestream_info_t *cstr_info) {
	int l;
	int compno;
	int eof = 0;
	double tile_time, t1_time, dwt_time;
	opj_tcd_tile_t *tile = NULL;

	opj_t1_t *t1 = NULL;		/* T1 component */
	opj_t2_t *t2 = NULL;		/* T2 component */
	
	tcd->tcd_tileno = tileno;
	tcd->tcd_tile = &(tcd->tcd_image->tiles[tileno]);
	tcd->tcp = &(tcd->cp->tcps[tileno]);
	tile = tcd->tcd_tile;
	
	tile_time = opj_clock();	/* time needed to decode a tile */
	opj_event_msg(tcd->cinfo, EVT_INFO, "tile %d of %d\n", tileno + 1, tcd->cp->tw * tcd->cp->th);

	/* INDEX >>  */
	if(cstr_info) {
		int resno, compno, numprec = 0;
		for (compno = 0; compno < cstr_info->numcomps; compno++) {
			opj_tcp_t *tcp = &tcd->cp->tcps[0];
			opj_tccp_t *tccp = &tcp->tccps[compno];
			opj_tcd_tilecomp_t *tilec_idx = &tile->comps[compno];	
			for (resno = 0; resno < tilec_idx->numresolutions; resno++) {
				opj_tcd_resolution_t *res_idx = &tilec_idx->resolutions[resno];
				cstr_info->tile[tileno].pw[resno] = res_idx->pw;
				cstr_info->tile[tileno].ph[resno] = res_idx->ph;
				numprec += res_idx->pw * res_idx->ph;
				if (tccp->csty & J2K_CP_CSTY_PRT) {
					cstr_info->tile[tileno].pdx[resno] = tccp->prcw[resno];
					cstr_info->tile[tileno].pdy[resno] = tccp->prch[resno];
				}
				else {
					cstr_info->tile[tileno].pdx[resno] = 15;
					cstr_info->tile[tileno].pdx[resno] = 15;
				}
			}
		}
		cstr_info->tile[tileno].packet = (opj_packet_info_t *) opj_malloc(cstr_info->numlayers * numprec * sizeof(opj_packet_info_t));
		cstr_info->packno = 0;
	}
	/* << INDEX */
	
	/*--------------TIER2------------------*/
	
	t2 = t2_create(tcd->cinfo, tcd->image, tcd->cp);
	l = t2_decode_packets(t2, src, len, tileno, tile, cstr_info);
	t2_destroy(t2);

	if (l == -999) {
		eof = 1;
		opj_event_msg(tcd->cinfo, EVT_ERROR, "tcd_decode: incomplete bistream\n");
	}
	
	/*------------------TIER1-----------------*/
	
	t1_time = opj_clock();	/* time needed to decode a tile */
	t1 = t1_create(tcd->cinfo);
	for (compno = 0; compno < tile->numcomps; ++compno) {
		opj_tcd_tilecomp_t* tilec = &tile->comps[compno];
		/* The +3 is headroom required by the vectorized DWT */
		tilec->data = (int*) opj_aligned_malloc((((tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0))+3) * sizeof(int));
		t1_decode_cblks(t1, tilec, &tcd->tcp->tccps[compno]);
	}
	t1_destroy(t1);
	t1_time = opj_clock() - t1_time;
	opj_event_msg(tcd->cinfo, EVT_INFO, "- tiers-1 took %f s\n", t1_time);
	
	/*----------------DWT---------------------*/

	dwt_time = opj_clock();	/* time needed to decode a tile */
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
		int numres2decode;

		if (tcd->cp->reduce != 0) {
			tcd->image->comps[compno].resno_decoded =
				tile->comps[compno].numresolutions - tcd->cp->reduce - 1;
			if (tcd->image->comps[compno].resno_decoded < 0) {				
				opj_event_msg(tcd->cinfo, EVT_ERROR, "Error decoding tile. The number of resolutions to remove [%d+1] is higher than the number "
					" of resolutions in the original codestream [%d]\nModify the cp_reduce parameter.\n", tcd->cp->reduce, tile->comps[compno].numresolutions);
				return false;
			}
		}

		numres2decode = tcd->image->comps[compno].resno_decoded + 1;
		if(numres2decode > 0){
			if (tcd->tcp->tccps[compno].qmfbid == 1) {
				dwt_decode(tilec, numres2decode);
			} else {
				dwt_decode_real(tilec, numres2decode);
			}
		}
	}
	dwt_time = opj_clock() - dwt_time;
	opj_event_msg(tcd->cinfo, EVT_INFO, "- dwt took %f s\n", dwt_time);

	/*----------------MCT-------------------*/

	if (tcd->tcp->mct) {
		int n = (tile->comps[0].x1 - tile->comps[0].x0) * (tile->comps[0].y1 - tile->comps[0].y0);
		if (tcd->tcp->tccps[0].qmfbid == 1) {
			mct_decode(
					tile->comps[0].data,
					tile->comps[1].data,
					tile->comps[2].data, 
					n);
		} else {
			mct_decode_real(
					(float*)tile->comps[0].data,
					(float*)tile->comps[1].data,
					(float*)tile->comps[2].data, 
					n);
		}
	}

	/*---------------TILE-------------------*/

	for (compno = 0; compno < tile->numcomps; ++compno) {
		opj_tcd_tilecomp_t* tilec = &tile->comps[compno];
		opj_image_comp_t* imagec = &tcd->image->comps[compno];
		opj_tcd_resolution_t* res = &tilec->resolutions[imagec->resno_decoded];
		int adjust = imagec->sgnd ? 0 : 1 << (imagec->prec - 1);
		int min = imagec->sgnd ? -(1 << (imagec->prec - 1)) : 0;
		int max = imagec->sgnd ?  (1 << (imagec->prec - 1)) - 1 : (1 << imagec->prec) - 1;

		int tw = tilec->x1 - tilec->x0;
		int w = imagec->w;

		int offset_x = int_ceildivpow2(imagec->x0, imagec->factor);
		int offset_y = int_ceildivpow2(imagec->y0, imagec->factor);

		int i, j;
		if(!imagec->data){
			imagec->data = (int*) opj_malloc(imagec->w * imagec->h * sizeof(int));
		}
		if(tcd->tcp->tccps[compno].qmfbid == 1) {
			for(j = res->y0; j < res->y1; ++j) {
				for(i = res->x0; i < res->x1; ++i) {
					int v = tilec->data[i - res->x0 + (j - res->y0) * tw];
					v += adjust;
					imagec->data[(i - offset_x) + (j - offset_y) * w] = int_clamp(v, min, max);
				}
			}
		}else{
			for(j = res->y0; j < res->y1; ++j) {
				for(i = res->x0; i < res->x1; ++i) {
					float tmp = ((float*)tilec->data)[i - res->x0 + (j - res->y0) * tw];
					int v = lrintf(tmp);
					v += adjust;
					imagec->data[(i - offset_x) + (j - offset_y) * w] = int_clamp(v, min, max);
				}
			}
		}
		opj_aligned_free(tilec->data);
	}

	tile_time = opj_clock() - tile_time;	/* time needed to decode a tile */
	opj_event_msg(tcd->cinfo, EVT_INFO, "- tile decoded in %f s\n", tile_time);

	if (eof) {
		return false;
	}
	
	return true;
}

void tcd_free_decode(opj_tcd_t *tcd) {
	opj_tcd_image_t *tcd_image = tcd->tcd_image;	
	opj_free(tcd_image->tiles);
}

void tcd_free_decode_tile(opj_tcd_t *tcd, int tileno) {
	int compno,resno,bandno,precno;

	opj_tcd_image_t *tcd_image = tcd->tcd_image;

	opj_tcd_tile_t *tile = &tcd_image->tiles[tileno];
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
		for (resno = 0; resno < tilec->numresolutions; resno++) {
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];
			for (bandno = 0; bandno < res->numbands; bandno++) {
				opj_tcd_band_t *band = &res->bands[bandno];
				for (precno = 0; precno < res->ph * res->pw; precno++) {
					opj_tcd_precinct_t *prec = &band->precincts[precno];
					if (prec->imsbtree != NULL) tgt_destroy(prec->imsbtree);
					if (prec->incltree != NULL) tgt_destroy(prec->incltree);
				}
				opj_free(band->precincts);
			}
		}
		opj_free(tilec->resolutions);
	}
	opj_free(tile->comps);
}



