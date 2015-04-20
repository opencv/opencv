/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2006, Mónica Díez, LPI-UVA, Spain
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

void tcd_dump(FILE *fd, opj_tcd_t *tcd, opj_tcd_volume_t * vol) {
	int tileno, compno, resno, bandno, precno, cblkno;

	fprintf(fd, "volume {\n");
	fprintf(fd, "  tw=%d, th=%d, tl=%d, x0=%d x1=%d y0=%d y1=%d z0=%d z1=%d\n", 
		vol->tw, vol->th, vol->tl, tcd->volume->x0, tcd->volume->x1, tcd->volume->y0, tcd->volume->y1, tcd->volume->z0, tcd->volume->z1);

	for (tileno = 0; tileno < vol->th * vol->tw * vol->tl; tileno++) {
		opj_tcd_tile_t *tile = &tcd->tcd_volume->tiles[tileno];
		fprintf(fd, "  tile {\n");
		fprintf(fd, "    x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d, numcomps=%d\n",
			tile->x0, tile->y0, tile->z0, tile->x1, tile->y1, tile->z1, tile->numcomps);
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
			fprintf(fd, "    tilecomp %d {\n",compno);
			fprintf(fd,	"     x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d, numresx=%d, numresy=%d, numresz=%d\n",
				tilec->x0, tilec->y0, tilec->z0, tilec->x1, tilec->y1, tilec->z1, tilec->numresolution[0], tilec->numresolution[1], tilec->numresolution[2]);
			for (resno = 0; resno < tilec->numresolution[0]; resno++) {
				opj_tcd_resolution_t *res = &tilec->resolutions[resno];
				fprintf(fd, "     res %d{\n",resno);
				fprintf(fd,"      x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d, pw=%d, ph=%d, pl=%d, numbands=%d\n",
					res->x0, res->y0, res->z0, res->x1, res->y1, res->z1, res->prctno[0], res->prctno[1], res->prctno[2], res->numbands);
				for (bandno = 0; bandno < res->numbands; bandno++) {
					opj_tcd_band_t *band = &res->bands[bandno];
					fprintf(fd, "       band %d{\n", bandno);
					fprintf(fd,	"		 x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d, stepsize=%f, numbps=%d\n",
						band->x0, band->y0, band->z0, band->x1, band->y1, band->z1, band->stepsize, band->numbps);
					for (precno = 0; precno < (res->prctno[0] * res->prctno[1] * res->prctno[2]); precno++) {
						opj_tcd_precinct_t *prec = &band->precincts[precno];
						fprintf(fd, "		  prec %d{\n",precno);
						fprintf(fd,	"		   x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d, cw=%d, ch=%d, cl=%d,\n",
							prec->x0, prec->y0, prec->z0, prec->x1, prec->y1, prec->z1, prec->cblkno[0], prec->cblkno[1], prec->cblkno[2]);
						for (cblkno = 0; cblkno < (prec->cblkno[0] * prec->cblkno[1] * prec->cblkno[2]); cblkno++) {
							opj_tcd_cblk_t *cblk = &prec->cblks[cblkno];
							fprintf(fd, "		    cblk %d{\n",cblkno);
							fprintf(fd,	"		     x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d\n", cblk->x0, cblk->y0, cblk->z0, cblk->x1, cblk->y1, cblk->z1);
							fprintf(fd, "            }\n");
						}
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

void tilec_dump(FILE *fd, opj_tcd_tilecomp_t *tilec) {

	int i=0,k;
	int datalen;
	int *a;

	fprintf(fd, "    tilecomp{\n");
	fprintf(fd,	"     x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d, numresx=%d, numresy=%d, numresz=%d\n",
		tilec->x0, tilec->y0, tilec->z0, tilec->x1, tilec->y1, tilec->z1, tilec->numresolution[0], tilec->numresolution[1], tilec->numresolution[2]);
	fprintf(fd, "     data {\n");
	datalen = (tilec->z1 - tilec->z0) * (tilec->y1 - tilec->y0) * (tilec->x1 - tilec->x0);
	a = tilec->data;
	for (k = 0; k < datalen; k++) {
		if (!(k % tilec->x1)){
			fprintf(fd, "\n");
		}
		if (!(k % (tilec->y1 * tilec->x1))){
			fprintf(fd, "Slice %d\n",i++);
		}
		fprintf(fd," %d",a[k]);
		
		
	}			
	fprintf(fd, "     }\n");
	/*i=0;
	fprintf(fd, "Slice %d\n");
	if (tilec->prediction->prederr) {
		fprintf(fd, "     prederror {\n");
		a = tilec->prediction->prederr;
		for (k = 0; k < datalen; k++) {
			fprintf(fd," %d",*(a++));
			if (!(k % (tilec->y1 - tilec->y0) * (tilec->x1 - tilec->x0))){
				fprintf(fd, "\n");fprintf(fd, "Slice %d\n",i++);
			}
			if (!(k % (tilec->x1 - tilec->x0))){
				fprintf(fd, "\n");
			}
		}
	}
	fprintf(fd, "     }\n");*/
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
	tcd->tcd_volume = (opj_tcd_volume_t*)opj_malloc(sizeof(opj_tcd_volume_t));
	if(!tcd->tcd_volume) {
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
		opj_free(tcd->tcd_volume);
		opj_free(tcd);
	}
}

/* ----------------------------------------------------------------------- */
void tcd_malloc_encode(opj_tcd_t *tcd, opj_volume_t * volume, opj_cp_t * cp, int curtileno) {
	int compno, resno, bandno, precno, cblkno, i, j;//, k;

	opj_tcd_tile_t		*tile = NULL;		/* pointer to tcd->tile */
	opj_tcd_tilecomp_t	*tilec = NULL;		/* pointer to tcd->tilec */
	opj_tcd_resolution_t	*res = NULL;		/* pointer to tcd->res */
	opj_tcd_band_t		*band = NULL;		/* pointer to tcd->band */
	opj_tcd_precinct_t	*prc = NULL;		/* pointer to tcd->prc */
	opj_tcd_cblk_t		*cblk = NULL;		/* pointer to tcd->cblk */
	opj_tcp_t		*tcp = &cp->tcps[curtileno];
	int p,q,r;

	tcd->volume = volume;
	tcd->cp = cp;
	tcd->tcd_volume->tw = cp->tw;
	tcd->tcd_volume->th = cp->th;
	tcd->tcd_volume->tl = cp->tl;
	tcd->tcd_volume->tiles = (opj_tcd_tile_t *) opj_malloc(sizeof(opj_tcd_tile_t));
	tcd->tile = tcd->tcd_volume->tiles;
	tile = tcd->tile;
	

	/* p61 ISO/IEC IS15444-1 : 2002 */
	/* curtileno --> raster scanned index of tiles */
	/* p,q,r --> matricial index of tiles */
	p = curtileno % cp->tw;	
	q = curtileno / cp->tw;	
	r = curtileno / (cp->tw * cp->th); /* extension to 3-D */

	/* 4 borders of the tile rescale on the volume if necessary (B.3)*/
	tile->x0 = int_max(cp->tx0 + p * cp->tdx, volume->x0);
	tile->y0 = int_max(cp->ty0 + q * cp->tdy, volume->y0);
	tile->z0 = int_max(cp->tz0 + r * cp->tdz, volume->z0);
	tile->x1 = int_min(cp->tx0 + (p + 1) * cp->tdx, volume->x1);
	tile->y1 = int_min(cp->ty0 + (q + 1) * cp->tdy, volume->y1);
	tile->z1 = int_min(cp->tz0 + (r + 1) * cp->tdz, volume->z1);
	tile->numcomps = volume->numcomps;

	/* Modification of the RATE >> */
	for (j = 0; j < tcp->numlayers; j++) {
		if (tcp->rates[j] <= 1) {
			tcp->rates[j] = 0;
		} else {
			float num = (float) (tile->numcomps * (tile->x1 - tile->x0) * (tile->y1 - tile->y0) * (tile->z1 - tile->z0) * volume->comps[0].prec);
			float den = (float) (8 * volume->comps[0].dx * volume->comps[0].dy * volume->comps[0].dz);
			den = tcp->rates[j] * den;
			tcp->rates[j] = (num + den - 1) / den;
		}
		/*tcp->rates[j] = tcp->rates[j] ? int_ceildiv(
			tile->numcomps * (tile->x1 - tile->x0) * (tile->y1 - tile->y0) * (tile->z1 - tile->z0) * volume->comps[0].prec,
            (tcp->rates[j] * 8 * volume->comps[0].dx * volume->comps[0].dy * volume->comps[0].dz)) : 0;*/
		if (tcp->rates[j]) {
			if (j && tcp->rates[j] < tcp->rates[j - 1] + 10) {
				tcp->rates[j] = tcp->rates[j - 1] + 20;
			} else if (!j && tcp->rates[j] < 30){
				tcp->rates[j] = 30;
			}
		}
	}
	/* << Modification of the RATE */

	tile->comps = (opj_tcd_tilecomp_t *) opj_malloc(volume->numcomps * sizeof(opj_tcd_tilecomp_t));
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tccp_t *tccp = &tcp->tccps[compno];
		int res_max;
		int prevnumbands = 0;

		/* opj_tcd_tilecomp_t *tilec=&tile->comps[compno]; */
		tcd->tilec = &tile->comps[compno];
		tilec = tcd->tilec;

		/* border of each tile component (global) (B.3) */
		tilec->x0 = int_ceildiv(tile->x0, volume->comps[compno].dx);
		tilec->y0 = int_ceildiv(tile->y0, volume->comps[compno].dy);
		tilec->z0 = int_ceildiv(tile->z0, volume->comps[compno].dz);
		tilec->x1 = int_ceildiv(tile->x1, volume->comps[compno].dx);
		tilec->y1 = int_ceildiv(tile->y1, volume->comps[compno].dy);
		tilec->z1 = int_ceildiv(tile->z1, volume->comps[compno].dz);

		tilec->data = (int *) opj_malloc((tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0) * (tilec->z1 - tilec->z0) * sizeof(int));
		
		res_max = 0;
		for (i = 0;i < 3; i++){
			tilec->numresolution[i] = tccp->numresolution[i];
			//Greater of 3 resolutions contains all information
			res_max = (tilec->numresolution[i] > res_max) ? tilec->numresolution[i] : res_max;
		}
		

		tilec->resolutions = (opj_tcd_resolution_t *) opj_malloc(res_max * sizeof(opj_tcd_resolution_t));
		for (resno = 0; resno < res_max; resno++) {
			
			int pdx, pdy, pdz;
			int tlprcxstart, tlprcystart, tlprczstart;
			int brprcxend, brprcyend, brprczend;
			int tlcbgxstart, tlcbgystart, tlcbgzstart;
			int brcbgxend, brcbgyend, brcbgzend;
			int cbgwidthexpn, cbgheightexpn, cbglengthexpn;
			int cblkwidthexpn, cblkheightexpn, cblklengthexpn;

			int diff = tccp->numresolution[0] - tccp->numresolution[2]; 
			int levelnox = tilec->numresolution[0] - 1 - resno; 
			int levelnoy = tilec->numresolution[1] - 1 - resno;
			int levelnoz = tilec->numresolution[2] - 1 - ((resno <= diff) ? 0 : (resno - diff));
				if (levelnoz < 0) levelnoz = 0;

			/* opj_tcd_resolution_t *res=&tilec->resolutions[resno]; */
			tcd->res = &tilec->resolutions[resno];
			res = tcd->res;
			
			/* border for each resolution level (global) (B.14)*/
			res->x0 = int_ceildivpow2(tilec->x0, levelnox);
			res->y0 = int_ceildivpow2(tilec->y0, levelnoy);
			res->z0 = int_ceildivpow2(tilec->z0, levelnoz);
			res->x1 = int_ceildivpow2(tilec->x1, levelnox);
			res->y1 = int_ceildivpow2(tilec->y1, levelnoy);
			res->z1 = int_ceildivpow2(tilec->z1, levelnoz);
			//if (res->z1 < 0)fprintf(stdout,"Res: %d       %d/%d --> %d\n",resno,tilec->z1, levelnoz, int_ceildivpow2(tilec->z1, levelnoz));

			res->numbands = (resno == 0) ? 1 : (resno <= diff) ? 3 : 7; /* --> 3D */

			/* p. 30, table A-13, ISO/IEC IS154444-1 : 2002 */
			if (tccp->csty & J3D_CCP_CSTY_PRT) {
				pdx = tccp->prctsiz[0][resno];
				pdy = tccp->prctsiz[1][resno];
				pdz = tccp->prctsiz[2][resno];
			} else {
				pdx = 15;
				pdy = 15;
				pdz = 15;
			}
			
			/* p. 66, B.16, ISO/IEC IS15444-1 : 2002  */
			tlprcxstart = int_floordivpow2(res->x0, pdx) << pdx;
			tlprcystart = int_floordivpow2(res->y0, pdy) << pdy;
			tlprczstart = int_floordivpow2(res->z0, pdz) << pdz;
			brprcxend = int_ceildivpow2(res->x1, pdx) << pdx;
			brprcyend = int_ceildivpow2(res->y1, pdy) << pdy;
			brprczend = int_ceildivpow2(res->z1, pdz) << pdz;
			
			res->prctno[0] = (brprcxend - tlprcxstart) >> pdx;
			res->prctno[1] = (brprcyend - tlprcystart) >> pdy;
			res->prctno[2] = (brprczend - tlprczstart) >> pdz;
				if (res->prctno[2] == 0) res->prctno[2] = 1;
				
			/* p. 67, B.17 & B.18, ISO/IEC IS15444-1 : 2002  */
			if (resno == 0) {
				tlcbgxstart = tlprcxstart;
				tlcbgystart = tlprcystart;
				tlcbgzstart = tlprczstart;
				brcbgxend = brprcxend;
				brcbgyend = brprcyend;
				brcbgzend = brprczend;
				cbgwidthexpn = pdx;
				cbgheightexpn = pdy;
				cbglengthexpn = pdz;
			} else {
				tlcbgxstart = int_ceildivpow2(tlprcxstart, 1);
				tlcbgystart = int_ceildivpow2(tlprcystart, 1);
				tlcbgzstart = int_ceildivpow2(tlprczstart, 1);
				brcbgxend = int_ceildivpow2(brprcxend, 1);
				brcbgyend = int_ceildivpow2(brprcyend, 1);
				brcbgzend = int_ceildivpow2(brprczend, 1);
				cbgwidthexpn = pdx - 1;
				cbgheightexpn = pdy - 1;
				cbglengthexpn = pdz - 1;
			}
			
			cblkwidthexpn = int_min(tccp->cblk[0], cbgwidthexpn); //6
			cblkheightexpn = int_min(tccp->cblk[1], cbgheightexpn); //6
			cblklengthexpn = int_min(tccp->cblk[2], cbglengthexpn); //6
			
			res->bands = (opj_tcd_band_t *) opj_malloc(res->numbands * sizeof(opj_tcd_band_t));
			for (bandno = 0; bandno < res->numbands; bandno++) {
				int x0b, y0b, z0b, i;
				int gain, numbps;
				opj_stepsize_t *ss = NULL;

				tcd->band = &res->bands[bandno];
				band = tcd->band;

				band->bandno = (resno == 0) ? 0 : bandno + 1;
				/* Bandno:	0 - LLL 	2 - LHL 
							1 - HLL		3 - HHL
							4 - LLH		6 - LHH
							5 - HLH		7 - HHH		*/
				x0b = (band->bandno == 1) || (band->bandno == 3) || (band->bandno == 5 ) || (band->bandno == 7 ) ? 1 : 0; 
				y0b = (band->bandno == 2) || (band->bandno == 3) || (band->bandno == 6 ) || (band->bandno == 7 ) ? 1 : 0;
				z0b = (band->bandno == 4) || (band->bandno == 5) || (band->bandno == 6 ) || (band->bandno == 7 ) ? 1 : 0; 
				
				/* p. 65, B.15, ISO/IEC IS15444-1 : 2002  */
				if (band->bandno == 0) {
					/* band border (global) */
					band->x0 = int_ceildivpow2(tilec->x0, levelnox);
					band->y0 = int_ceildivpow2(tilec->y0, levelnoy);
					band->z0 = int_ceildivpow2(tilec->z0, levelnoz);
					band->x1 = int_ceildivpow2(tilec->x1, levelnox);
					band->y1 = int_ceildivpow2(tilec->y1, levelnoy);
					band->z1 = int_ceildivpow2(tilec->z1, levelnoz);
				} else {
					band->x0 = int_ceildivpow2(tilec->x0 - (1 << levelnox) * x0b, levelnox + 1);
					band->y0 = int_ceildivpow2(tilec->y0 - (1 << levelnoy) * y0b, levelnoy + 1);
					band->z0 = int_ceildivpow2(tilec->z0 - (1 << levelnoz) * z0b, (resno <= diff) ? levelnoz : levelnoz + 1);
					band->x1 = int_ceildivpow2(tilec->x1 - (1 << levelnox) * x0b, levelnox + 1);
					band->y1 = int_ceildivpow2(tilec->y1 - (1 << levelnoy) * y0b, levelnoy + 1);
					band->z1 = int_ceildivpow2(tilec->z1 - (1 << levelnoz) * z0b, (resno <= diff) ? levelnoz : levelnoz + 1);
				}
				
				ss = &tccp->stepsizes[(resno == 0) ? 0 : (prevnumbands + bandno + 1)];
				if (bandno == (res->numbands - 1)) 
					prevnumbands += (resno == 0) ? 0 : res->numbands;
				gain = dwt_getgain(band->bandno,tccp->reversible);					
				numbps = volume->comps[compno].prec + gain;
 				band->stepsize = (float)((1.0 + ss->mant / 2048.0) * pow(2.0, numbps - ss->expn));
				band->numbps = ss->expn + tccp->numgbits - 1;	/* WHY -1 ? */
				
				band->precincts = (opj_tcd_precinct_t *) opj_malloc((res->prctno[0] * res->prctno[1] * res->prctno[2]) * sizeof(opj_tcd_precinct_t));
				
				for (i = 0; i < (res->prctno[0] * res->prctno[1] * res->prctno[2]); i++) {
					band->precincts[i].imsbtree = NULL;
					band->precincts[i].incltree = NULL;
				}

				for (precno = 0; precno < (res->prctno[0] * res->prctno[1] * res->prctno[2]); precno++) {
					int tlcblkxstart, tlcblkystart, tlcblkzstart, brcblkxend, brcblkyend, brcblkzend;
					int cbgxstart, cbgystart, cbgzstart, cbgxend, cbgyend, cbgzend;

					cbgxstart = tlcbgxstart + (precno % res->prctno[0]) * (1 << cbgwidthexpn);
					cbgystart = tlcbgystart + ((precno % (res->prctno[0] * res->prctno[1])) / res->prctno[0]) * (1 << cbgheightexpn);
					cbgzstart = tlcbgzstart + (precno / (res->prctno[0] * res->prctno[1])) * (1 << cbglengthexpn);
					cbgxend = cbgxstart + (1 << cbgwidthexpn);
					cbgyend = cbgystart + (1 << cbgheightexpn);
					cbgzend = cbgzstart + (1 << cbglengthexpn);
					
					tcd->prc = &band->precincts[precno];
					prc = tcd->prc;

					/* precinct size (global) */
					prc->x0 = int_max(cbgxstart, band->x0);
					prc->y0 = int_max(cbgystart, band->y0);
					prc->z0 = int_max(cbgzstart, band->z0);
					prc->x1 = int_min(cbgxend, band->x1);
					prc->y1 = int_min(cbgyend, band->y1);
					prc->z1 = int_min(cbgzend, band->z1);
					
					tlcblkxstart = int_floordivpow2(prc->x0, cblkwidthexpn) << cblkwidthexpn;
					tlcblkystart = int_floordivpow2(prc->y0, cblkheightexpn) << cblkheightexpn;
					tlcblkzstart = int_floordivpow2(prc->z0, cblklengthexpn) << cblklengthexpn;
					brcblkxend = int_ceildivpow2(prc->x1, cblkwidthexpn) << cblkwidthexpn;
					brcblkyend = int_ceildivpow2(prc->y1, cblkheightexpn) << cblkheightexpn;
					brcblkzend = int_ceildivpow2(prc->z1, cblklengthexpn) << cblklengthexpn;
					prc->cblkno[0] = (brcblkxend - tlcblkxstart) >> cblkwidthexpn;
					prc->cblkno[1] = (brcblkyend - tlcblkystart) >> cblkheightexpn;
					prc->cblkno[2] = (brcblkzend - tlcblkzstart) >> cblklengthexpn;
					prc->cblkno[2] = (prc->cblkno[2] == 0) ? 1 : prc->cblkno[2];

					prc->cblks = (opj_tcd_cblk_t *) opj_malloc((prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]) * sizeof(opj_tcd_cblk_t));
					prc->incltree = tgt_create(prc->cblkno[0], prc->cblkno[1], prc->cblkno[2]);
					prc->imsbtree = tgt_create(prc->cblkno[0], prc->cblkno[1], prc->cblkno[2]);
					//tgt_tree_dump(stdout,prc->incltree);
					for (cblkno = 0; cblkno < (prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]); cblkno++) {
						int cblkxstart = tlcblkxstart + (cblkno % prc->cblkno[0]) * (1 << cblkwidthexpn);
						int cblkystart = tlcblkystart + ((cblkno % (prc->cblkno[0] * prc->cblkno[1])) / prc->cblkno[0]) * (1 << cblkheightexpn);
						int cblkzstart = tlcblkzstart + (cblkno / (prc->cblkno[0] * prc->cblkno[1])) * (1 << cblklengthexpn);
						int cblkxend = cblkxstart + (1 << cblkwidthexpn);
						int cblkyend = cblkystart + (1 << cblkheightexpn);
						int cblkzend = cblkzstart + (1 << cblklengthexpn);
						int prec = ((tilec->bpp > 16) ? 3 : ((tilec->bpp > 8) ? 2 : 1));
						
						tcd->cblk = &prc->cblks[cblkno];
						cblk = tcd->cblk;

						/* code-block size (global) */
						cblk->x0 = int_max(cblkxstart, prc->x0);
						cblk->y0 = int_max(cblkystart, prc->y0);
						cblk->z0 = int_max(cblkzstart, prc->z0);
						cblk->x1 = int_min(cblkxend, prc->x1);
						cblk->y1 = int_min(cblkyend, prc->y1);
						cblk->z1 = int_min(cblkzend, prc->z1);
					}					
				}
			}
		}
	}
	//tcd_dump(stdout, tcd, tcd->tcd_volume);

}
void tcd_init_encode(opj_tcd_t *tcd, opj_volume_t * volume, opj_cp_t * cp, int curtileno) {
	int compno, resno, bandno, precno, cblkno;
	int j, p, q, r;

	opj_tcd_tile_t		*tile = NULL;		/* pointer to tcd->tile */
	opj_tcd_tilecomp_t	*tilec = NULL;		/* pointer to tcd->tilec */
	opj_tcd_resolution_t	*res = NULL;	/* pointer to tcd->res */
	opj_tcd_band_t		*band = NULL;		/* pointer to tcd->band */
	opj_tcd_precinct_t	*prc = NULL;		/* pointer to tcd->prc */
	opj_tcd_cblk_t		*cblk = NULL;		/* pointer to tcd->cblk */
	opj_tcp_t *tcp = &cp->tcps[curtileno];

	tcd->tile = tcd->tcd_volume->tiles;
	tile = tcd->tile;

	/* p61 ISO/IEC IS15444-1 : 2002 */
	/* curtileno --> raster scanned index of tiles */
	/* p,q,r --> matricial index of tiles */
	p = curtileno % cp->tw;	
	q = curtileno / cp->tw;	
	r = curtileno / (cp->tw * cp->th); /* extension to 3-D */
	
	/* 4 borders of the tile rescale on the volume if necessary (B.3)*/
	tile->x0 = int_max(cp->tx0 + p * cp->tdx, volume->x0);
	tile->y0 = int_max(cp->ty0 + q * cp->tdy, volume->y0);
	tile->z0 = int_max(cp->tz0 + r * cp->tdz, volume->z0);
	tile->x1 = int_min(cp->tx0 + (p + 1) * cp->tdx, volume->x1);
	tile->y1 = int_min(cp->ty0 + (q + 1) * cp->tdy, volume->y1);
	tile->z1 = int_min(cp->tz0 + (r + 1) * cp->tdz, volume->z1);
	tile->numcomps = volume->numcomps;

	/* Modification of the RATE >> */
	for (j = 0; j < tcp->numlayers; j++) {
		if (tcp->rates[j] <= 1) {
			tcp->rates[j] = 0;
		} else {
			float num = (float) (tile->numcomps * (tile->x1 - tile->x0) * (tile->y1 - tile->y0) * (tile->z1 - tile->z0) * volume->comps[0].prec);
			float den = (float) (8 * volume->comps[0].dx * volume->comps[0].dy * volume->comps[0].dz);
			den = tcp->rates[j] * den;
			tcp->rates[j] = (num + den - 1) / den;
		}
		/*tcp->rates[j] = tcp->rates[j] ? int_ceildiv(
			tile->numcomps * (tile->x1 - tile->x0) * (tile->y1 - tile->y0) * (tile->z1 - tile->z0) * volume->comps[0].prec,
            (tcp->rates[j] * 8 * volume->comps[0].dx * volume->comps[0].dy * volume->comps[0].dz)) : 0;*/
		if (tcp->rates[j]) {
			if (j && tcp->rates[j] < tcp->rates[j - 1] + 10) {
				tcp->rates[j] = tcp->rates[j - 1] + 20;
			} else if (!j && tcp->rates[j] < 30){
				tcp->rates[j] = 30;
			}
		}
	}
	/* << Modification of the RATE */

	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tccp_t *tccp = &tcp->tccps[compno];
		int res_max, i;
		int prevnumbands = 0;

		/* opj_tcd_tilecomp_t *tilec=&tile->comps[compno]; */
		tcd->tilec = &tile->comps[compno];
		tilec = tcd->tilec;

		/* border of each tile component (global) (B.3) */
		tilec->x0 = int_ceildiv(tile->x0, volume->comps[compno].dx);
		tilec->y0 = int_ceildiv(tile->y0, volume->comps[compno].dy);
		tilec->z0 = int_ceildiv(tile->z0, volume->comps[compno].dz);
		tilec->x1 = int_ceildiv(tile->x1, volume->comps[compno].dx);
		tilec->y1 = int_ceildiv(tile->y1, volume->comps[compno].dy);
		tilec->z1 = int_ceildiv(tile->z1, volume->comps[compno].dz);

		tilec->data = (int *) opj_malloc((tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0) * (tilec->z1 - tilec->z0) * sizeof(int));
		
		res_max = 0;
		for (i = 0;i < 3; i++){
			tilec->numresolution[i] = tccp->numresolution[i];
			//Greater of 3 resolutions contains all information
			res_max = (tilec->numresolution[i] > res_max) ? tilec->numresolution[i] : res_max;
		}

		tilec->resolutions = (opj_tcd_resolution_t *) opj_malloc(res_max * sizeof(opj_tcd_resolution_t));
		for (resno = 0; resno < res_max; resno++) {
			int pdx, pdy, pdz;
			int tlprcxstart, tlprcystart, tlprczstart, brprcxend, brprcyend, brprczend;
			int tlcbgxstart, tlcbgystart, tlcbgzstart, brcbgxend, brcbgyend, brcbgzend;
			int cbgwidthexpn, cbgheightexpn, cbglengthexpn;
			int cblkwidthexpn, cblkheightexpn, cblklengthexpn;
			
			int levelnox = tilec->numresolution[0] - 1 - resno; 
			int levelnoy = tilec->numresolution[1] - 1 - resno;
			int diff = tccp->numresolution[0] - tccp->numresolution[2]; 
			int levelnoz = tilec->numresolution[2] - 1 - ((resno <= diff) ? 0 : (resno - diff));
				if (levelnoz < 0) levelnoz = 0;

			tcd->res = &tilec->resolutions[resno];
			res = tcd->res;
			
			/* border for each resolution level (global) (B.14)*/
			res->x0 = int_ceildivpow2(tilec->x0, levelnox);
			res->y0 = int_ceildivpow2(tilec->y0, levelnoy);
			res->z0 = int_ceildivpow2(tilec->z0, levelnoz);
			res->x1 = int_ceildivpow2(tilec->x1, levelnox);
			res->y1 = int_ceildivpow2(tilec->y1, levelnoy);
			res->z1 = int_ceildivpow2(tilec->z1, levelnoz);

			// res->numbands = resno == 0 ? 1 : 3; /* --> 2D */
			res->numbands = (resno == 0) ? 1 : (resno <= diff) ? 3 : 7; /* --> 3D */

			/* p. 30, table A-13, ISO/IEC IS154444-1 : 2002 */			
			if (tccp->csty & J3D_CCP_CSTY_PRT) {
				pdx = tccp->prctsiz[0][resno];
				pdy = tccp->prctsiz[1][resno];
				pdz = tccp->prctsiz[2][resno];
			} else {
				pdx = 15;
				pdy = 15;
				pdz = 15;
			}
			/* p. 66, B.16, ISO/IEC IS15444-1 : 2002  */
			tlprcxstart = int_floordivpow2(res->x0, pdx) << pdx;
			tlprcystart = int_floordivpow2(res->y0, pdy) << pdy;
			tlprczstart = int_floordivpow2(res->z0, pdz) << pdz;
			brprcxend = int_ceildivpow2(res->x1, pdx) << pdx;
			brprcyend = int_ceildivpow2(res->y1, pdy) << pdy;
			brprczend = int_ceildivpow2(res->z1, pdz) << pdz;
			
			res->prctno[0] = (brprcxend - tlprcxstart) >> pdx;
			res->prctno[1] = (brprcyend - tlprcystart) >> pdy;
			res->prctno[2] = (brprczend - tlprczstart) >> pdz;
			if (res->prctno[2] == 0) res->prctno[2] = 1;

			/* p. 67, B.17 & B.18, ISO/IEC IS15444-1 : 2002  */
			if (resno == 0) {
				tlcbgxstart = tlprcxstart;
				tlcbgystart = tlprcystart;
				tlcbgzstart = tlprczstart;
				brcbgxend = brprcxend;
				brcbgyend = brprcyend;
				brcbgzend = brprczend;
				cbgwidthexpn = pdx;
				cbgheightexpn = pdy;
				cbglengthexpn = pdz;
			} else {
				tlcbgxstart = int_ceildivpow2(tlprcxstart, 1);
				tlcbgystart = int_ceildivpow2(tlprcystart, 1);
				tlcbgzstart = int_ceildivpow2(tlprczstart, 1);
				brcbgxend = int_ceildivpow2(brprcxend, 1);
				brcbgyend = int_ceildivpow2(brprcyend, 1);
				brcbgzend = int_ceildivpow2(brprczend, 1);
				cbgwidthexpn = pdx - 1;
				cbgheightexpn = pdy - 1;
				cbglengthexpn = pdz - 1;
			}
			
			cblkwidthexpn = int_min(tccp->cblk[0], cbgwidthexpn);
			cblkheightexpn = int_min(tccp->cblk[1], cbgheightexpn);
			cblklengthexpn = int_min(tccp->cblk[2], cbglengthexpn);
			
			res->bands = (opj_tcd_band_t *) opj_malloc(res->numbands * sizeof(opj_tcd_band_t));
			for (bandno = 0; bandno < res->numbands; bandno++) {
				int x0b, y0b, z0b;
				int gain, numbps;
				opj_stepsize_t *ss = NULL;

				tcd->band = &res->bands[bandno];
				band = tcd->band;

				band->bandno = resno == 0 ? 0 : bandno + 1;
				/* Bandno:	0 - LLL 	2 - LHL 
							1 - HLL		3 - HHL
							4 - LLH		6 - LHH
							5 - HLH		7 - HHH		*/
				x0b = (band->bandno == 1) || (band->bandno == 3) || (band->bandno == 5 ) || (band->bandno == 7 ) ? 1 : 0; 
				y0b = (band->bandno == 2) || (band->bandno == 3) || (band->bandno == 6 ) || (band->bandno == 7 ) ? 1 : 0;
				z0b = (band->bandno == 4) || (band->bandno == 5) || (band->bandno == 6 ) || (band->bandno == 7 ) ? 1 : 0; 
				
				/* p. 65, B.15, ISO/IEC IS15444-1 : 2002  */
				if (band->bandno == 0) {
					/* band border (global) */
					band->x0 = int_ceildivpow2(tilec->x0, levelnox);
					band->y0 = int_ceildivpow2(tilec->y0, levelnoy);
					band->z0 = int_ceildivpow2(tilec->z0, levelnoz);
					band->x1 = int_ceildivpow2(tilec->x1, levelnox);
					band->y1 = int_ceildivpow2(tilec->y1, levelnoy);
					band->z1 = int_ceildivpow2(tilec->z1, levelnoz);
				} else {
					band->x0 = int_ceildivpow2(tilec->x0 - (1 << levelnox) * x0b, levelnox + 1);
					band->y0 = int_ceildivpow2(tilec->y0 - (1 << levelnoy) * y0b, levelnoy + 1);
					band->z0 = int_ceildivpow2(tilec->z0 - (1 << levelnoz) * z0b, (resno <= diff) ? levelnoz : levelnoz + 1);
					band->x1 = int_ceildivpow2(tilec->x1 - (1 << levelnox) * x0b, levelnox + 1);
					band->y1 = int_ceildivpow2(tilec->y1 - (1 << levelnoy) * y0b, levelnoy + 1);
					band->z1 = int_ceildivpow2(tilec->z1 - (1 << levelnoz) * z0b, (resno <= diff) ? levelnoz : levelnoz + 1);
				}
								
				ss = &tccp->stepsizes[(resno == 0) ? 0 : (prevnumbands + bandno + 1)];
				if (bandno == (res->numbands - 1)) 
					prevnumbands += (resno == 0) ? 0 : res->numbands;
				gain = dwt_getgain(band->bandno,tccp->reversible);					
				numbps = volume->comps[compno].prec + gain;
				
				band->stepsize = (float)((1.0 + ss->mant / 2048.0) * pow(2.0, numbps - ss->expn));
				band->numbps = ss->expn + tccp->numgbits - 1;	/* WHY -1 ? */
				
				for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
					int tlcblkxstart, tlcblkystart, tlcblkzstart, brcblkxend, brcblkyend, brcblkzend;

					int cbgxstart = tlcbgxstart + (precno % res->prctno[0]) * (1 << cbgwidthexpn);
					int cbgystart = tlcbgystart + ((precno / (res->prctno[0] * res->prctno[1])) / res->prctno[0]) * (1 << cbgheightexpn);
					int cbgzstart = tlcbgzstart + (precno / (res->prctno[0] * res->prctno[1])) * (1 << cbglengthexpn);
					int cbgxend = cbgxstart + (1 << cbgwidthexpn);
					int cbgyend = cbgystart + (1 << cbgheightexpn);
					int cbgzend = cbgzstart + (1 << cbglengthexpn);

					/* opj_tcd_precinct_t *prc=&band->precincts[precno]; */
					tcd->prc = &band->precincts[precno];
					prc = tcd->prc;

					/* precinct size (global) */
					prc->x0 = int_max(cbgxstart, band->x0);
					prc->y0 = int_max(cbgystart, band->y0);
					prc->z0 = int_max(cbgzstart, band->z0);
					prc->x1 = int_min(cbgxend, band->x1);
					prc->y1 = int_min(cbgyend, band->y1);
					prc->z1 = int_min(cbgzend, band->z1);

					tlcblkxstart = int_floordivpow2(prc->x0, cblkwidthexpn) << cblkwidthexpn;
					tlcblkystart = int_floordivpow2(prc->y0, cblkheightexpn) << cblkheightexpn;
					tlcblkzstart = int_floordivpow2(prc->z0, cblklengthexpn) << cblklengthexpn;
					brcblkxend = int_ceildivpow2(prc->x1, cblkwidthexpn) << cblkwidthexpn;
					brcblkyend = int_ceildivpow2(prc->y1, cblkheightexpn) << cblkheightexpn;
					brcblkzend = int_ceildivpow2(prc->z1, cblklengthexpn) << cblklengthexpn;
					prc->cblkno[0] = (brcblkxend - tlcblkxstart) >> cblkwidthexpn;
					prc->cblkno[1] = (brcblkyend - tlcblkystart) >> cblkheightexpn;
					prc->cblkno[2] = (brcblkzend - tlcblkzstart) >> cblklengthexpn;
					prc->cblkno[2] = (prc->cblkno[2] == 0) ? 1 : prc->cblkno[2];

					opj_free(prc->cblks);
					prc->cblks = (opj_tcd_cblk_t *) opj_malloc((prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]) * sizeof(opj_tcd_cblk_t));
					prc->incltree = tgt_create(prc->cblkno[0], prc->cblkno[1], prc->cblkno[2]);
					prc->imsbtree = tgt_create(prc->cblkno[0], prc->cblkno[1], prc->cblkno[2]);

					for (cblkno = 0; cblkno < (prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]); cblkno++) {
							int cblkxstart = tlcblkxstart + (cblkno % prc->cblkno[0]) * (1 << cblkwidthexpn);
							int cblkystart = tlcblkystart + ((cblkno % (prc->cblkno[0] * prc->cblkno[1])) / prc->cblkno[0]) * (1 << cblkheightexpn);
							int cblkzstart = tlcblkzstart + (cblkno / (prc->cblkno[0] * prc->cblkno[1])) * (1 << cblklengthexpn);
							int cblkxend = cblkxstart + (1 << cblkwidthexpn);
							int cblkyend = cblkystart + (1 << cblkheightexpn);
							int cblkzend = cblkzstart + (1 << cblklengthexpn);
							int prec = ((tilec->bpp > 16) ? 3 : ((tilec->bpp > 8) ? 2 : 1));

							tcd->cblk = &prc->cblks[cblkno];
							cblk = tcd->cblk;

							/* code-block size (global) */
							cblk->x0 = int_max(cblkxstart, prc->x0);
							cblk->y0 = int_max(cblkystart, prc->y0);
							cblk->z0 = int_max(cblkzstart, prc->z0);
							cblk->x1 = int_min(cblkxend, prc->x1);
							cblk->y1 = int_min(cblkyend, prc->y1);
							cblk->z1 = int_min(cblkzend, prc->z1);
					}
				} /* precno */
			} /* bandno */
		} /* resno */
	} /* compno */
	//tcd_dump(stdout, tcd, tcd->tcd_volume);
}


void tcd_free_encode(opj_tcd_t *tcd) {
	int tileno, compno, resno, bandno, precno;

	opj_tcd_tile_t *tile = NULL;		/* pointer to tcd->tile		*/
//	opj_tcd_slice_t *slice = NULL;		/* pointer to tcd->slice */
	opj_tcd_tilecomp_t *tilec = NULL;	/* pointer to tcd->tilec	*/
	opj_tcd_resolution_t *res = NULL;	/* pointer to tcd->res		*/
	opj_tcd_band_t *band = NULL;		/* pointer to tcd->band		*/
	opj_tcd_precinct_t *prc = NULL;		/* pointer to tcd->prc		*/

	for (tileno = 0; tileno < 1; tileno++) {
		tcd->tile = tcd->tcd_volume->tiles;
		tile = tcd->tile;

		for (compno = 0; compno < tile->numcomps; compno++) {
			tcd->tilec = &tile->comps[compno];
			tilec = tcd->tilec;

			for (resno = 0; resno < tilec->numresolution[0]; resno++) {
				tcd->res = &tilec->resolutions[resno];
				res = tcd->res;

				for (bandno = 0; bandno < res->numbands; bandno++) {
					tcd->band = &res->bands[bandno];
					band = tcd->band;

					for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
						tcd->prc = &band->precincts[precno];
						prc = tcd->prc;

						if (prc->incltree != NULL) {
                            tgt_destroy(prc->incltree);
                            prc->incltree = NULL;
						}
						if (prc->imsbtree != NULL) {
                            tgt_destroy(prc->imsbtree);
                            prc->imsbtree = NULL;
						}
						opj_free(prc->cblks);
						prc->cblks = NULL;
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
	opj_free(tcd->tcd_volume->tiles);
	tcd->tcd_volume->tiles = NULL;
}

/* ----------------------------------------------------------------------- */
void tcd_malloc_decode(opj_tcd_t *tcd, opj_volume_t * volume, opj_cp_t * cp) {
	int tileno, compno, resno, bandno, precno, cblkno, res_max,
		i, j, p, q, r;
	unsigned int x0 = 0, y0 = 0, z0 = 0, 
		x1 = 0, y1 = 0, z1 = 0, 
		w, h, l;

	tcd->volume = volume;
	tcd->cp = cp;
	tcd->tcd_volume->tw = cp->tw;
	tcd->tcd_volume->th = cp->th;
	tcd->tcd_volume->tl = cp->tl;
	tcd->tcd_volume->tiles = (opj_tcd_tile_t *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(opj_tcd_tile_t));
	
	for (i = 0; i < cp->tileno_size; i++) {
		opj_tcp_t *tcp = &(cp->tcps[cp->tileno[i]]);
		opj_tcd_tile_t *tile = &(tcd->tcd_volume->tiles[cp->tileno[i]]);
	
		/* p61 ISO/IEC IS15444-1 : 2002 */
		/* curtileno --> raster scanned index of tiles */
		/* p,q,r --> matricial index of tiles */
		tileno = cp->tileno[i];
		p = tileno % cp->tw;	
		q = tileno / cp->tw;	
		r = tileno / (cp->tw * cp->th); /* extension to 3-D */

		/* 4 borders of the tile rescale on the volume if necessary (B.3)*/
		tile->x0 = int_max(cp->tx0 + p * cp->tdx, volume->x0);
		tile->y0 = int_max(cp->ty0 + q * cp->tdy, volume->y0);
		tile->z0 = int_max(cp->tz0 + r * cp->tdz, volume->z0);
		tile->x1 = int_min(cp->tx0 + (p + 1) * cp->tdx, volume->x1);
		tile->y1 = int_min(cp->ty0 + (q + 1) * cp->tdy, volume->y1);
		tile->z1 = int_min(cp->tz0 + (r + 1) * cp->tdz, volume->z1);
		tile->numcomps = volume->numcomps;		
		
		tile->comps = (opj_tcd_tilecomp_t *) opj_malloc(volume->numcomps * sizeof(opj_tcd_tilecomp_t));
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tccp_t *tccp = &tcp->tccps[compno];
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
			int prevnumbands = 0;

			/* border of each tile component (global) */
			tilec->x0 = int_ceildiv(tile->x0, volume->comps[compno].dx);
			tilec->y0 = int_ceildiv(tile->y0, volume->comps[compno].dy);
			tilec->z0 = int_ceildiv(tile->z0, volume->comps[compno].dz);
			tilec->x1 = int_ceildiv(tile->x1, volume->comps[compno].dx);
			tilec->y1 = int_ceildiv(tile->y1, volume->comps[compno].dy);
			tilec->z1 = int_ceildiv(tile->z1, volume->comps[compno].dz);
			
			tilec->data = (int *) opj_malloc((tilec->x1 - tilec->x0) * (tilec->y1 - tilec->y0) * (tilec->z1 - tilec->z0) * sizeof(int));

			res_max = 0;
			for (i = 0;i < 3; i++){
				tilec->numresolution[i] = tccp->numresolution[i];
				//Greater of 3 resolutions contains all information
				res_max = (tilec->numresolution[i] > res_max) ? tilec->numresolution[i] : res_max;
			}

			tilec->resolutions = (opj_tcd_resolution_t *) opj_malloc(res_max * sizeof(opj_tcd_resolution_t));

			for (resno = 0; resno < res_max; resno++) {
				opj_tcd_resolution_t *res = &tilec->resolutions[resno];
				int pdx, pdy, pdz;
				int tlprcxstart, tlprcystart, tlprczstart, brprcxend, brprcyend, brprczend;
				int tlcbgxstart, tlcbgystart, tlcbgzstart, brcbgxend, brcbgyend, brcbgzend;
				int cbgwidthexpn, cbgheightexpn, cbglengthexpn;
				int cblkwidthexpn, cblkheightexpn, cblklengthexpn;
				int levelnox = tilec->numresolution[0] - 1 - resno; 
				int levelnoy = tilec->numresolution[1] - 1 - resno;
				int diff = tccp->numresolution[0] - tccp->numresolution[2]; 
				int levelnoz = tilec->numresolution[2] - 1 - ((resno <= diff) ? 0 : (resno - diff));
					if (levelnoz < 0) levelnoz = 0;

				/* border for each resolution level (global) */
				res->x0 = int_ceildivpow2(tilec->x0, levelnox);
				res->y0 = int_ceildivpow2(tilec->y0, levelnoy);
				res->z0 = int_ceildivpow2(tilec->z0, levelnoz);
				res->x1 = int_ceildivpow2(tilec->x1, levelnox);
				res->y1 = int_ceildivpow2(tilec->y1, levelnoy);
				res->z1 = int_ceildivpow2(tilec->z1, levelnoz);
				res->numbands = (resno == 0) ? 1 : (resno <= diff) ? 3 : 7; /* --> 3D */
				
				/* p. 30, table A-13, ISO/IEC IS154444-1 : 2002 */
				if (tccp->csty & J3D_CCP_CSTY_PRT) {
					pdx = tccp->prctsiz[0][resno];
					pdy = tccp->prctsiz[1][resno];
					pdz = tccp->prctsiz[2][resno];
				} else {
					pdx = 15;
					pdy = 15;
					pdz = 15;
				}
				
				/* p. 66, B.16, ISO/IEC IS15444-1 : 2002  */
				tlprcxstart = int_floordivpow2(res->x0, pdx) << pdx;
				tlprcystart = int_floordivpow2(res->y0, pdy) << pdy;
				tlprczstart = int_floordivpow2(res->z0, pdz) << pdz;
				brprcxend = int_ceildivpow2(res->x1, pdx) << pdx;
				brprcyend = int_ceildivpow2(res->y1, pdy) << pdy;
				brprczend = int_ceildivpow2(res->z1, pdz) << pdz;
				
				res->prctno[0] = (brprcxend - tlprcxstart) >> pdx;
				res->prctno[1] = (brprcyend - tlprcystart) >> pdy;
				res->prctno[2] = (brprczend - tlprczstart) >> pdz;
				
				/* p. 67, B.17 & B.18, ISO/IEC IS15444-1 : 2002  */
				if (resno == 0) {
					tlcbgxstart = tlprcxstart;//0
					tlcbgystart = tlprcystart;
					tlcbgzstart = tlprczstart;
					brcbgxend = brprcxend;//1
					brcbgyend = brprcyend;
					brcbgzend = brprczend;
					cbgwidthexpn = pdx; //15
					cbgheightexpn = pdy;
					cbglengthexpn = pdz;
				} else {
					tlcbgxstart = int_ceildivpow2(tlprcxstart, 1);
					tlcbgystart = int_ceildivpow2(tlprcystart, 1);
					tlcbgzstart = int_ceildivpow2(tlprczstart, 1);
					brcbgxend = int_ceildivpow2(brprcxend, 1);
					brcbgyend = int_ceildivpow2(brprcyend, 1);
					brcbgzend = int_ceildivpow2(brprczend, 1);
					cbgwidthexpn = pdx - 1;
					cbgheightexpn = pdy - 1;
					cbglengthexpn = pdz - 1;
				}
				
				cblkwidthexpn = int_min(tccp->cblk[0], cbgwidthexpn); //6
				cblkheightexpn = int_min(tccp->cblk[1], cbgheightexpn); //6
				cblklengthexpn = int_min(tccp->cblk[2], cbglengthexpn); //6

				res->bands = (opj_tcd_band_t *) opj_malloc(res->numbands * sizeof(opj_tcd_band_t));
				for (bandno = 0; bandno < res->numbands; bandno++) {
					int x0b, y0b, z0b;
					int gain, numbps;
					opj_stepsize_t *ss = NULL;

					opj_tcd_band_t *band = &res->bands[bandno];
					band->bandno = resno == 0 ? 0 : bandno + 1;
					/* Bandno:	0 - LLL 	2 - LHL 
								1 - HLL		3 - HHL
								4 - LLH		6 - LHH
								5 - HLH		7 - HHH		*/
					x0b = (band->bandno == 1) || (band->bandno == 3) || (band->bandno == 5 ) || (band->bandno == 7 ) ? 1 : 0; 
					y0b = (band->bandno == 2) || (band->bandno == 3) || (band->bandno == 6 ) || (band->bandno == 7 ) ? 1 : 0;
					z0b = (band->bandno == 4) || (band->bandno == 5) || (band->bandno == 6 ) || (band->bandno == 7 ) ? 1 : 0; 
					
					/* p. 65, B.15, ISO/IEC IS15444-1 : 2002  */
					if (band->bandno == 0) {
						/* band border (global) */
						band->x0 = int_ceildivpow2(tilec->x0, levelnox);
						band->y0 = int_ceildivpow2(tilec->y0, levelnoy);
						band->z0 = int_ceildivpow2(tilec->z0, levelnoz);
						band->x1 = int_ceildivpow2(tilec->x1, levelnox);
						band->y1 = int_ceildivpow2(tilec->y1, levelnoy);
						band->z1 = int_ceildivpow2(tilec->z1, levelnoz);
					} else {
						band->x0 = int_ceildivpow2(tilec->x0 - (1 << levelnox) * x0b, levelnox + 1);
						band->y0 = int_ceildivpow2(tilec->y0 - (1 << levelnoy) * y0b, levelnoy + 1);
						band->z0 = int_ceildivpow2(tilec->z0 - (1 << levelnoz) * z0b, (resno <= diff) ? levelnoz : levelnoz + 1);
						band->x1 = int_ceildivpow2(tilec->x1 - (1 << levelnox) * x0b, levelnox + 1);
						band->y1 = int_ceildivpow2(tilec->y1 - (1 << levelnoy) * y0b, levelnoy + 1);
						band->z1 = int_ceildivpow2(tilec->z1 - (1 << levelnoz) * z0b, (resno <= diff) ? levelnoz : levelnoz + 1);
					}	

					ss = &tccp->stepsizes[(resno == 0) ? 0 : (prevnumbands + bandno + 1)];
					if (bandno == (res->numbands - 1)) 
						prevnumbands += (resno == 0) ? 0 : res->numbands;
					gain = dwt_getgain(band->bandno,tccp->reversible);					
					numbps = volume->comps[compno].prec + gain;

					band->stepsize = (float)((1.0 + ss->mant / 2048.0) * pow(2.0, numbps - ss->expn));
					band->numbps = ss->expn + tccp->numgbits - 1;	/* WHY -1 ? */
					
					band->precincts = (opj_tcd_precinct_t *) opj_malloc(res->prctno[0] * res->prctno[1] * res->prctno[2] * sizeof(opj_tcd_precinct_t));
					
					for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
						int tlcblkxstart, tlcblkystart, tlcblkzstart, brcblkxend, brcblkyend, brcblkzend;

						int cbgxstart = tlcbgxstart + (precno % res->prctno[0]) * (1 << cbgwidthexpn);
						int cbgystart = tlcbgystart + (precno / res->prctno[0]) * (1 << cbgheightexpn);
						int cbgzstart = tlcbgzstart + (precno / (res->prctno[0] * res->prctno[1])) * (1 << cbglengthexpn);
						int cbgxend = cbgxstart + (1 << cbgwidthexpn);
						int cbgyend = cbgystart + (1 << cbgheightexpn);
						int cbgzend = cbgzstart + (1 << cbglengthexpn);

						opj_tcd_precinct_t *prc = &band->precincts[precno];
						/* precinct size (global) */
						prc->x0 = int_max(cbgxstart, band->x0);
						prc->y0 = int_max(cbgystart, band->y0);
						prc->z0 = int_max(cbgzstart, band->z0);
						prc->x1 = int_min(cbgxend, band->x1);
						prc->y1 = int_min(cbgyend, band->y1);
						prc->z1 = int_min(cbgzend, band->z1);

						tlcblkxstart = int_floordivpow2(prc->x0, cblkwidthexpn) << cblkwidthexpn;
						tlcblkystart = int_floordivpow2(prc->y0, cblkheightexpn) << cblkheightexpn;
						tlcblkzstart = int_floordivpow2(prc->z0, cblklengthexpn) << cblklengthexpn;
						brcblkxend = int_ceildivpow2(prc->x1, cblkwidthexpn) << cblkwidthexpn;
						brcblkyend = int_ceildivpow2(prc->y1, cblkheightexpn) << cblkheightexpn;
						brcblkzend = int_ceildivpow2(prc->z1, cblklengthexpn) << cblklengthexpn;
						prc->cblkno[0] = (brcblkxend - tlcblkxstart) >> cblkwidthexpn;
						prc->cblkno[1] = (brcblkyend - tlcblkystart) >> cblkheightexpn;
						prc->cblkno[2] = (brcblkzend - tlcblkzstart) >> cblklengthexpn;
						prc->cblkno[2] = (prc->cblkno[2] == 0) ? 1 : prc->cblkno[2];

						prc->cblks = (opj_tcd_cblk_t *) opj_malloc((prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]) * sizeof(opj_tcd_cblk_t));
						prc->incltree = tgt_create(prc->cblkno[0], prc->cblkno[1], prc->cblkno[2]);
						prc->imsbtree = tgt_create(prc->cblkno[0], prc->cblkno[1], prc->cblkno[2]);
						
						for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
							int cblkxstart = tlcblkxstart + (cblkno % prc->cblkno[0]) * (1 << cblkwidthexpn);
							int cblkystart = tlcblkystart + ((cblkno % (prc->cblkno[0] * prc->cblkno[1])) / prc->cblkno[0]) * (1 << cblkheightexpn);
							int cblkzstart = tlcblkzstart + (cblkno / (prc->cblkno[0] * prc->cblkno[1])) * (1 << cblklengthexpn);
							int cblkxend = cblkxstart + (1 << cblkwidthexpn);
							int cblkyend = cblkystart + (1 << cblkheightexpn);
							int cblkzend = cblkzstart + (1 << cblklengthexpn);
							int prec = ((tilec->bpp > 16) ? 3 : ((tilec->bpp > 8) ? 2 : 1));
							/* code-block size (global) */
							opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
							
							/* code-block size (global) */
							cblk->x0 = int_max(cblkxstart, prc->x0);
							cblk->y0 = int_max(cblkystart, prc->y0);
							cblk->z0 = int_max(cblkzstart, prc->z0);
							cblk->x1 = int_min(cblkxend, prc->x1);
							cblk->y1 = int_min(cblkyend, prc->y1);
							cblk->z1 = int_min(cblkzend, prc->z1);
						}
					} /* precno */
				} /* bandno */
			} /* resno */
		} /* compno */
	} /* i = 0..cp->tileno_size */

	//tcd_dump(stdout, tcd, tcd->tcd_volume);

	/* 
	Allocate place to store the decoded data = final volume
	Place limited by the tile really present in the codestream 
	*/
	
	for (i = 0; i < volume->numcomps; i++) {
		for (j = 0; j < cp->tileno_size; j++) {
			tileno = cp->tileno[j];
			x0 = (j == 0) ? tcd->tcd_volume->tiles[tileno].comps[i].x0 : int_min(x0,(unsigned int) tcd->tcd_volume->tiles[tileno].comps[i].x0);
			y0 = (j == 0) ? tcd->tcd_volume->tiles[tileno].comps[i].y0 : int_min(y0,(unsigned int) tcd->tcd_volume->tiles[tileno].comps[i].y0);
			z0 = (j == 0) ? tcd->tcd_volume->tiles[tileno].comps[i].z0 : int_min(z0,(unsigned int) tcd->tcd_volume->tiles[tileno].comps[i].z0);
			x1 = (j == 0) ? tcd->tcd_volume->tiles[tileno].comps[i].x1 : int_max(x1,(unsigned int) tcd->tcd_volume->tiles[tileno].comps[i].x1);
			y1 = (j == 0) ? tcd->tcd_volume->tiles[tileno].comps[i].y1 : int_max(y1,(unsigned int) tcd->tcd_volume->tiles[tileno].comps[i].y1);
			z1 = (j == 0) ? tcd->tcd_volume->tiles[tileno].comps[i].z1 : int_max(z1,(unsigned int) tcd->tcd_volume->tiles[tileno].comps[i].z1);
		}
		
		w = x1 - x0;
		h = y1 - y0;
		l = z1 - z0;
		
		volume->comps[i].data = (int *) opj_malloc(w * h * l * sizeof(int));
		volume->comps[i].w = w;
		volume->comps[i].h = h;
		volume->comps[i].l = l;
		volume->comps[i].x0 = x0;
		volume->comps[i].y0 = y0;
		volume->comps[i].z0 = z0;
		volume->comps[i].bigendian = cp->bigendian;
	}
}

void tcd_free_decode(opj_tcd_t *tcd) {
	int tileno,compno,resno,bandno,precno;

	opj_tcd_volume_t *tcd_volume = tcd->tcd_volume;
	
	for (tileno = 0; tileno < tcd_volume->tw * tcd_volume->th * tcd_volume->tl; tileno++) {
		opj_tcd_tile_t *tile = &tcd_volume->tiles[tileno];
		for (compno = 0; compno < tile->numcomps; compno++) {
			opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
			for (resno = 0; resno < tilec->numresolution[0]; resno++) {
				opj_tcd_resolution_t *res = &tilec->resolutions[resno];
				for (bandno = 0; bandno < res->numbands; bandno++) {
					opj_tcd_band_t *band = &res->bands[bandno];
					for (precno = 0; precno < res->prctno[1] * res->prctno[0] * res->prctno[2]; precno++) {
						opj_tcd_precinct_t *prec = &band->precincts[precno];
						if (prec->cblks != NULL) opj_free(prec->cblks);
						if (prec->imsbtree != NULL) tgt_destroy(prec->imsbtree);
                        if (prec->incltree != NULL) tgt_destroy(prec->incltree);
						/*for (treeno = 0; treeno < prec->numtrees; treeno++){
                            if (prec->imsbtree[treeno] != NULL) tgt_destroy(prec->imsbtree[treeno]);
                            if (prec->incltree[treeno] != NULL) tgt_destroy(prec->incltree[treeno]);
						}*/
					}
					if (band->precincts != NULL) opj_free(band->precincts);
				}
			}
			if (tilec->resolutions != NULL) opj_free(tilec->resolutions);
		}
		if (tile->comps != NULL) opj_free(tile->comps);
	}

	if (tcd_volume->tiles != NULL) opj_free(tcd_volume->tiles);
}



/* ----------------------------------------------------------------------- */
void tcd_makelayer_fixed(opj_tcd_t *tcd, int layno, int final) {
	int compno, resno, bandno, precno, cblkno;
	int value;			/*, matrice[tcd_tcp->numlayers][tcd_tile->comps[0].numresolution[0]][3]; */
	int matrice[10][10][3];
	int i, j, k;

	opj_cp_t *cp = tcd->cp;
	opj_tcd_tile_t *tcd_tile = tcd->tcd_tile;
	opj_tcp_t *tcd_tcp = tcd->tcp;

	/*matrice=(int*)opj_malloc(tcd_tcp->numlayers*tcd_tile->comps[0].numresolution[0]*3*sizeof(int)); */
	
	for (compno = 0; compno < tcd_tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tcd_tile->comps[compno];
		for (i = 0; i < tcd_tcp->numlayers; i++) {
			for (j = 0; j < tilec->numresolution[0]; j++) {
				for (k = 0; k < 3; k++) {
					matrice[i][j][k] =
						(int) (cp->matrice[i * tilec->numresolution[0] * 3 + j * 3 + k] 
						* (float) (tcd->volume->comps[compno].prec / 16.0));
				}
			}
		}
        
		for (resno = 0; resno < tilec->numresolution[0]; resno++) {
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];
			for (bandno = 0; bandno < res->numbands; bandno++) {
				opj_tcd_band_t *band = &res->bands[bandno];
				for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
					opj_tcd_precinct_t *prc = &band->precincts[precno];
					for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
						opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
						opj_tcd_layer_t *layer = &cblk->layers[layno];
						int n;
						int imsb = tcd->volume->comps[compno].prec - cblk->numbps;	/* number of bit-plan equal to zero */
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
		for (resno = 0; resno < tilec->numresolution[0]; resno++) {
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];
			for (bandno = 0; bandno < res->numbands; bandno++) {
				opj_tcd_band_t *band = &res->bands[bandno];
				for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
					opj_tcd_precinct_t *prc = &band->precincts[precno];
					for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
						opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
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
								if (dd)
									n = passno + 1;
								continue;
							}
							if (dd / dr >= thresh){
								n = passno + 1;
							}
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

					//	fprintf(stdout,"MakeLayer : %d %f %d %d \n",layer->len, layer->disto, layer->numpasses, n);
					}
				}
			}
		}
	}
}

bool tcd_rateallocate(opj_tcd_t *tcd, unsigned char *dest, int len, opj_volume_info_t * volume_info) {
	int compno, resno, bandno, precno, cblkno, passno, layno;
	double min, max;
	double cumdisto[100];	/* fixed_quality */
	const double K = 1;		/* 1.1; // fixed_quality */
	double maxSE = 0;

	opj_cp_t *cp = tcd->cp;
	opj_tcd_tile_t *tcd_tile = tcd->tcd_tile;
	opj_tcp_t *tcd_tcp = tcd->tcp;

	min = DBL_MAX;
	max = 0;
	
	tcd_tile->nbpix = 0;		/* fixed_quality */
	
	for (compno = 0; compno < tcd_tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tcd_tile->comps[compno];
		tilec->nbpix = 0;
		for (resno = 0; resno < tilec->numresolution[0]; resno++) {
			opj_tcd_resolution_t *res = &tilec->resolutions[resno];
			for (bandno = 0; bandno < res->numbands; bandno++) {
				opj_tcd_band_t *band = &res->bands[bandno];
				for (precno = 0; precno < res->prctno[0] * res->prctno[1] * res->prctno[2]; precno++) {
					opj_tcd_precinct_t *prc = &band->precincts[precno];
					for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
						opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
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
						tcd_tile->nbpix += ((cblk->x1 - cblk->x0) * (cblk->y1 - cblk->y0) * (cblk->z1 - cblk->z0));
                        tilec->nbpix += ((cblk->x1 - cblk->x0) * (cblk->y1 - cblk->y0) * (cblk->z1 - cblk->z0));
					} /* cbklno */ 
				} /* precno */
			} /* bandno */
		} /* resno */
		
		maxSE += (((double)(1 << tcd->volume->comps[compno].prec) - 1.0) 
			* ((double)(1 << tcd->volume->comps[compno].prec) -1.0)) 
			* ((double)(tilec->nbpix));
	} /* compno */
	
	/* add antonin index */
	if(volume_info && volume_info->index_on) {
		opj_tile_info_t *info_TL = &volume_info->tile[tcd->tcd_tileno];
		info_TL->nbpix = tcd_tile->nbpix;
		info_TL->distotile = tcd_tile->distotile;
		info_TL->thresh = (double *) opj_malloc(tcd_tcp->numlayers * sizeof(double));
	}
	/* dda */
	
	for (layno = 0; layno < tcd_tcp->numlayers; layno++) {
		double lo = min;
		double hi = max;
		int success = 0;
		int maxlen = tcd_tcp->rates[layno] ? int_min(((int) tcd_tcp->rates[layno]), len) : len;
		double goodthresh;
		double distotarget;		/* fixed_quality */
		int i = 0;
		
        /* fixed_quality */
		distotarget = tcd_tile->distotile - ((K * maxSE) / pow((float)10, tcd_tcp->distoratio[layno] / 10));
        
		if ((tcd_tcp->rates[layno]) || (cp->disto_alloc==0)) {
			opj_t2_t *t2 = t2_create(tcd->cinfo, tcd->volume, cp);
			int oldl = 0, oldoldl = 0;
			for (i = 0; i < 128; i++) {
				double thresh = (lo + hi) / 2;
				int l = 0;
				double distoachieved = 0;	/* fixed_quality -q */
			
				tcd_makelayer(tcd, layno, thresh, 0);
		
				if (cp->fixed_quality) {	/* fixed_quality -q */
					distoachieved =	(layno == 0) ? tcd_tile->distolayer[0] : cumdisto[layno - 1] + tcd_tile->distolayer[layno];
					if (distoachieved < distotarget) {
						hi = thresh; 
						continue;
					}
					lo = thresh;
				} else {		/* disto_alloc -r, fixed_alloc -f */
					l = t2_encode_packets(t2, tcd->tcd_tileno, tcd_tile, layno + 1, dest, maxlen, volume_info);
					//fprintf(stdout, "layno %d i %d len=%d max=%d \n",layno,i,l,maxlen);
					if (l == -999) {
						lo = thresh; 
						continue;
					} else if (l == oldl && oldl == oldoldl && tcd_tile->distolayer[layno] > 0.0 && i>32)
						break;
					hi = thresh;
					oldoldl = oldl;
					oldl = l;
				}
				success = 1;
				goodthresh = thresh;
			} 
			t2_destroy(t2);
		} else {
			success = 1;
			goodthresh = min;
		}
		if (!success) {
			return false;
		}
		
		if(volume_info && volume_info->index_on) {	/* Threshold for Marcela Index */
			volume_info->tile[tcd->tcd_tileno].thresh[layno] = goodthresh;
		}
		tcd_makelayer(tcd, layno, goodthresh, 1);
	        
		/* fixed_quality */
		cumdisto[layno] = (layno == 0) ? tcd_tile->distolayer[0] : cumdisto[layno - 1] + tcd_tile->distolayer[layno];	
	}

	return true;
}

/* ----------------------------------------------------------------------- */
int tcd_encode_tile(opj_tcd_t *tcd, int tileno, unsigned char *dest, int len, opj_volume_info_t * volume_info) {
	int compno;
	int l, i, npck = 0;
	double encoding_time;
	
	opj_tcd_tile_t	*tile = NULL;
	opj_tcp_t		*tcd_tcp = NULL;
	opj_cp_t		*cp = NULL;

	opj_tcp_t		*tcp = &tcd->cp->tcps[0];
	opj_tccp_t		*tccp = &tcp->tccps[0];
	opj_volume_t	*volume = tcd->volume;
	opj_t2_t		*t2 = NULL;		/* T2 component */

	tcd->tcd_tileno = tileno;			/* current encoded/decoded tile */
	
	tcd->tcd_tile = tcd->tcd_volume->tiles; /* tile information */
	tile = tcd->tcd_tile;
	
	tcd->tcp = &tcd->cp->tcps[tileno];	/* coding/decoding params of tileno */	
	tcd_tcp = tcd->tcp;
	
	cp = tcd->cp;		/* coding parameters */

	/* INDEX >> */
	if(volume_info && volume_info->index_on) {
		opj_tcd_tilecomp_t *tilec_idx = &tile->comps[0];	/* based on component 0 */
		for (i = 0; i < tilec_idx->numresolution[0]; i++) {
			opj_tcd_resolution_t *res_idx = &tilec_idx->resolutions[i];

			volume_info->tile[tileno].prctno[0][i] = res_idx->prctno[0];
			volume_info->tile[tileno].prctno[1][i] = res_idx->prctno[1];
			volume_info->tile[tileno].prctno[2][i] = res_idx->prctno[2];

			npck += res_idx->prctno[0] * res_idx->prctno[1] * res_idx->prctno[2];

			volume_info->tile[tileno].prctsiz[0][i] = tccp->prctsiz[0][i];
			volume_info->tile[tileno].prctsiz[1][i] = tccp->prctsiz[1][i];
			volume_info->tile[tileno].prctsiz[2][i] = tccp->prctsiz[2][i];
		}
		volume_info->tile[tileno].packet = (opj_packet_info_t *) opj_malloc(volume_info->comp * volume_info->layer * npck * sizeof(opj_packet_info_t));
	}
	/* << INDEX */
	
	/*---------------TILE-------------------*/
	encoding_time = opj_clock();	/* time needed to encode a tile */
	
	for (compno = 0; compno < tile->numcomps; compno++) {
		int x, y, z;
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
		
		int adjust;
		int offset_x = int_ceildiv(volume->x0, volume->comps[compno].dx); //ceil(x0 / subsampling_dx)
		int offset_y = int_ceildiv(volume->y0, volume->comps[compno].dy);
		int offset_z = int_ceildiv(volume->z0, volume->comps[compno].dz);
		
		int tw = tilec->x1 - tilec->x0;
		int w = int_ceildiv(volume->x1 - volume->x0, volume->comps[compno].dx);
		int th = tilec->y1 - tilec->y0;
		int h = int_ceildiv(volume->y1 - volume->y0, volume->comps[compno].dy);
		int tl = tilec->z1 - tilec->z0;
		int l = int_ceildiv(volume->z1 - volume->z0, volume->comps[compno].dz);

		
		
		/* extract tile data from volume.comps[0].data to tile.comps[0].data */
		//fprintf(stdout,"[INFO] Extract tile data\n");
		if (tcd->cp->transform_format == TRF_3D_RLS || tcd->cp->transform_format == TRF_3D_LSE) {
			adjust = 0;
		} else {
            adjust = volume->comps[compno].sgnd ? 0 : 1 << (volume->comps[compno].prec - 1); //sign=='+' --> 2^(prec-1)
			if (volume->comps[compno].dcoffset != 0){
				adjust += volume->comps[compno].dcoffset;
				fprintf(stdout,"[INFO] DC Offset applied: DCO = %d -> adjust = %d\n",volume->comps[compno].dcoffset,adjust);
			}
		}		

		if (tcd_tcp->tccps[compno].reversible == 1) { //IF perfect reconstruction (DWT.5-3)
			for (z = tilec->z0; z < tilec->z1; z++) {
				for (y = tilec->y0; y < tilec->y1; y++) {
					/* start of the src tile scanline */
					int *data = &volume->comps[compno].data[(tilec->x0 - offset_x) + (y - offset_y) * w + (z - offset_z) * w * h];
					/* start of the dst tile scanline */
					int *tile_data = &tilec->data[(y - tilec->y0) * tw + (z - tilec->z0) * tw * th];
					for (x = tilec->x0; x < tilec->x1; x++) {
						*tile_data++ = *data++ - adjust;
					}
				}
			}
		} else if (tcd_tcp->tccps[compno].reversible == 0) { //IF not (DWT.9-7)
			for (z = tilec->z0; z < tilec->z1; z++) {
				for (y = tilec->y0; y < tilec->y1; y++) {
					/* start of the src tile scanline */
					int *data = &volume->comps[compno].data[(tilec->x0 - offset_x) + (y - offset_y) * w + (z - offset_z) * w * h];
					/* start of the dst tile scanline */
					int *tile_data = &tilec->data[(y - tilec->y0) * tw + (z - tilec->z0) * tw * th];
					for (x = tilec->x0; x < tilec->x1; x++) {
						*tile_data++ = (*data++ - adjust) << 13;
					}
				}
			}
		}
	
	}

	/*----------------MCT-------------------*/
	if (tcd_tcp->mct) {
		int samples = (tile->comps[0].x1 - tile->comps[0].x0) * (tile->comps[0].y1 - tile->comps[0].y0) * (tile->comps[0].z1 - tile->comps[0].z0);
		fprintf(stdout,"[INFO] Tcd_encode_tile: mct\n");
		if (tcd_tcp->tccps[0].reversible == 0) {
			mct_encode_real(tile->comps[0].data, tile->comps[1].data, tile->comps[2].data, samples);
		} else {
			mct_encode(tile->comps[0].data, tile->comps[1].data, tile->comps[2].data, samples);
		}
	}
	/*----------------TRANSFORM---------------------------------*/
	fprintf(stdout,"[INFO] Tcd_encode_tile: Transform\n");
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
		dwt_encode(tilec, tcd_tcp->tccps[compno].dwtid);
	} 

	/*-------------------ENTROPY CODING-----------------------------*/
	fprintf(stdout,"[INFO] Tcd_encode_tile: Entropy coding\n");
	if ((cp->encoding_format == ENCOD_2EB)||(cp->encoding_format == ENCOD_3EB))
	{
		if (cp->encoding_format == ENCOD_2EB) {
			opj_t1_t *t1 = NULL;
			t1 = t1_create(tcd->cinfo);
			t1_encode_cblks(t1, tile, tcd_tcp);
			t1_destroy(t1);	
		} else if (cp->encoding_format == ENCOD_3EB) {
			opj_t1_3d_t *t1 = NULL;		
			t1 = t1_3d_create(tcd->cinfo);
			t1_3d_encode_cblks(t1, tile, tcd_tcp);
			t1_3d_destroy(t1);	
		}
		/*-----------RATE-ALLOCATE------------------*/
		/* INDEX */
		if(volume_info) {
			volume_info->index_write = 0;
		}
		if (cp->disto_alloc || cp->fixed_quality) {	
   			fprintf(stdout,"[INFO] Tcd_encode_tile: Rate-allocate\n");
			tcd_rateallocate(tcd, dest, len, volume_info);			/* Normal Rate/distortion allocation */
		} else {/* fixed_alloc */
    	    fprintf(stdout,"[INFO] Tcd_encode_tile: Rate-allocate fixed\n");
            tcd_rateallocate_fixed(tcd);							/* Fixed layer allocation */
		}

		/*--------------TIER2------------------*/
		/* INDEX */
		if(volume_info) {
			volume_info->index_write = 1;
		}
		fprintf(stdout,"[INFO] Tcd_encode_tile: Tier - 2\n");
        t2 = t2_create(tcd->cinfo, volume, cp);
		l = t2_encode_packets(t2, tileno, tile, tcd_tcp->numlayers, dest, len, volume_info);
        t2_destroy(t2);
	} else if ((cp->encoding_format == ENCOD_2GR)||(cp->encoding_format == ENCOD_3GR)) {
		/*if(volume_info) {
			volume_info->index_write = 1;
		}
		gr = golomb_create(tcd->cinfo, volume, cp);
		l = golomb_encode(gr, tileno, tile, dest, len, volume_info);
		golomb_destroy(gr);*/
	}

	
	/*---------------CLEAN-------------------*/
	fprintf(stdout,"[INFO] Tcd_encode_tile: %d bytes coded\n",l);
	encoding_time = opj_clock() - encoding_time;
	opj_event_msg(tcd->cinfo, EVT_INFO, "- tile encoded in %f s\n", encoding_time);
	
	/* cleaning memory */
	for (compno = 0; compno < tile->numcomps; compno++) {
		tcd->tilec = &tile->comps[compno];
		opj_free(tcd->tilec->data);
	}
	
	if (l == -999){
		fprintf(stdout,"[ERROR] Unable to perform T2 tier. Return -999.\n");
		return 0;
	}

	return l;
}


bool tcd_decode_tile(opj_tcd_t *tcd, unsigned char *src, int len, int tileno) {
	int l, i;
	int compno, eof = 0;
	double tile_time, t1_time, dwt_time;

	opj_tcd_tile_t *tile = NULL;
	opj_t2_t *t2 = NULL;		/* T2 component */
	
	tcd->tcd_tileno = tileno;
	tcd->tcd_tile = &(tcd->tcd_volume->tiles[tileno]);
	tcd->tcp = &(tcd->cp->tcps[tileno]);
	tile = tcd->tcd_tile;
	
	tile_time = opj_clock();	/* time needed to decode a tile */
	opj_event_msg(tcd->cinfo, EVT_INFO, "tile %d / %d\n", tileno + 1, tcd->cp->tw * tcd->cp->th * tcd->cp->tl);

	if ((tcd->cp->encoding_format == ENCOD_2EB) || (tcd->cp->encoding_format == ENCOD_3EB)) {
		/*--------------TIER2------------------*/
		t2 = t2_create(tcd->cinfo, tcd->volume, tcd->cp);
		l = t2_decode_packets(t2, src, len, tileno, tile);
		t2_destroy(t2);
		opj_event_msg(tcd->cinfo, EVT_INFO, "Tcd_decode_tile: %d bytes decoded\n",l);
		
		if (l == -999) {
			eof = 1;
			opj_event_msg(tcd->cinfo, EVT_ERROR, "Tcd_decode_tile: incomplete bistream\n");
		}
	
		/*------------------TIER1-----------------*/
		opj_event_msg(tcd->cinfo, EVT_INFO, "Tcd_decode_tile: Entropy decoding %d \n",tcd->cp->encoding_format);
		t1_time = opj_clock();	/* time needed to decode a tile */
		if (tcd->cp->encoding_format == ENCOD_2EB) {
			opj_t1_t *t1 = NULL;		/* T1 component */
			t1 = t1_create(tcd->cinfo);
			t1_decode_cblks(t1, tile, tcd->tcp);
			t1_destroy(t1);
		}else if (tcd->cp->encoding_format == ENCOD_3EB) {
			opj_t1_3d_t *t1 = NULL;		/* T1 component */
			t1 = t1_3d_create(tcd->cinfo);
			t1_3d_decode_cblks(t1, tile, tcd->tcp);
			t1_3d_destroy(t1);
		}

		t1_time = opj_clock() - t1_time;
		#ifdef VERBOSE
				opj_event_msg(tcd->cinfo, EVT_INFO, "- tier-1 took %f s\n", t1_time);
		#endif
	} else if ((tcd->cp->encoding_format == ENCOD_2GR)||(tcd->cp->encoding_format == ENCOD_3GR)) {
		opj_event_msg(tcd->cinfo, EVT_INFO, "Tcd_decode_tile: Entropy decoding -- Does nothing :-D\n");
		/*
		gr = golomb_create(tcd->cinfo, tcd->volume, tcd->cp);
		l = golomb_decode(gr, tileno, tile, src, len);
		golomb_destroy(gr);
		if (l == -999) {
			eof = 1;
			opj_event_msg(tcd->cinfo, EVT_ERROR, "Tcd_decode_tile: incomplete bistream\n");
		}
		*/
	} 

	/*----------------DWT---------------------*/
	fprintf(stdout,"[INFO] Tcd_decode_tile: Inverse DWT\n");
	dwt_time = opj_clock();	/* time needed to decode a tile */
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
		int stops[3], dwtid[3];
	
		for (i = 0; i < 3; i++) {
			if (tcd->cp->reduce[i] != 0) 
				tcd->volume->comps[compno].resno_decoded[i] = tile->comps[compno].numresolution[i] - tcd->cp->reduce[i] - 1;
			stops[i] = tilec->numresolution[i] - 1 - tcd->volume->comps[compno].resno_decoded[i];
			if (stops[i] < 0) stops[i]=0;
			dwtid[i] = tcd->cp->tcps->tccps[compno].dwtid[i];
		}
		
		dwt_decode(tilec, stops, dwtid);

		for (i = 0; i < 3; i++) {
			if (tile->comps[compno].numresolution[i] > 0) {
				tcd->volume->comps[compno].factor[i] = tile->comps[compno].numresolution[i] - (tcd->volume->comps[compno].resno_decoded[i] + 1);
				if ( (tcd->volume->comps[compno].factor[i]) < 0 )
					tcd->volume->comps[compno].factor[i] = 0;
			}
		}
	}
	dwt_time = opj_clock() - dwt_time;
	#ifdef VERBOSE
			opj_event_msg(tcd->cinfo, EVT_INFO, "- dwt took %f s\n", dwt_time);
	#endif

	/*----------------MCT-------------------*/
	
	if (tcd->tcp->mct) {
		if (tcd->tcp->tccps[0].reversible == 1) {
			mct_decode(tile->comps[0].data, tile->comps[1].data, tile->comps[2].data, 
				(tile->comps[0].x1 - tile->comps[0].x0) * (tile->comps[0].y1 - tile->comps[0].y0) * (tile->comps[0].z1 - tile->comps[0].z0));
		} else {
			mct_decode_real(tile->comps[0].data, tile->comps[1].data, tile->comps[2].data, 
				(tile->comps[0].x1 - tile->comps[0].x0) * (tile->comps[0].y1 - tile->comps[0].y0)* (tile->comps[0].z1 - tile->comps[0].z0));
		}
	}
	
	/*---------------TILE-------------------*/
	
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
		opj_tcd_resolution_t *res =	&tilec->resolutions[tcd->volume->comps[compno].resno_decoded[0]];
		int adjust;
		int minval = tcd->volume->comps[compno].sgnd ? -(1 << (tcd->volume->comps[compno].prec - 1)) : 0;
		int maxval = tcd->volume->comps[compno].sgnd ? (1 << (tcd->volume->comps[compno].prec - 1)) - 1 : (1 << tcd->volume->comps[compno].prec) - 1;
		
		int tw = tilec->x1 - tilec->x0;
		int w = tcd->volume->comps[compno].w;
		int th = tilec->y1 - tilec->y0;
		int h = tcd->volume->comps[compno].h;

		int i, j, k;
		int offset_x = int_ceildivpow2(tcd->volume->comps[compno].x0, tcd->volume->comps[compno].factor[0]);
		int offset_y = int_ceildivpow2(tcd->volume->comps[compno].y0, tcd->volume->comps[compno].factor[1]);
		int offset_z = int_ceildivpow2(tcd->volume->comps[compno].z0, tcd->volume->comps[compno].factor[2]);
		
		if (tcd->cp->transform_format == TRF_3D_RLS || tcd->cp->transform_format == TRF_3D_LSE) {
			adjust = 0;
		} else {
            adjust = tcd->volume->comps[compno].sgnd ? 0 : 1 << (tcd->volume->comps[compno].prec - 1); //sign=='+' --> 2^(prec-1)
			if (tcd->volume->comps[compno].dcoffset != 0){
				adjust += tcd->volume->comps[compno].dcoffset;
				fprintf(stdout,"[INFO] DC Offset applied: DCO = %d -> adjust = %d\n",tcd->volume->comps[compno].dcoffset,adjust);
			}
		}

		for (k = res->z0; k < res->z1; k++) {
			for (j = res->y0; j < res->y1; j++) {
				for (i = res->x0; i < res->x1; i++) {
					int v;
					float tmp = (float)((tilec->data[i - res->x0 + (j - res->y0) * tw + (k - res->z0) * tw * th]) / 8192.0);

					if (tcd->tcp->tccps[compno].reversible == 1) {
						v = tilec->data[i - res->x0 + (j - res->y0) * tw + (k - res->z0) * tw * th];
					} else {
						int tmp2 = ((int) (floor(fabs(tmp)))) + ((int) floor(fabs(tmp*2))%2);
						v = ((tmp < 0) ? -tmp2:tmp2);
					}
					v += adjust;
					
					tcd->volume->comps[compno].data[(i - offset_x) + (j - offset_y) * w + (k - offset_z) * w * h] = int_clamp(v, minval, maxval);
				}
			}
		}
	}
	
	tile_time = opj_clock() - tile_time;	/* time needed to decode a tile */
	opj_event_msg(tcd->cinfo, EVT_INFO, "- tile decoded in %f s\n", tile_time);
		
	for (compno = 0; compno < tile->numcomps; compno++) {
		opj_free(tcd->tcd_volume->tiles[tileno].comps[compno].data);
		tcd->tcd_volume->tiles[tileno].comps[compno].data = NULL;
	}
	
	if (eof) {
		return false;
	}
	
	return true;
}

