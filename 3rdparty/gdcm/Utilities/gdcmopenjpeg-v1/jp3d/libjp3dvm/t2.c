/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

/** @defgroup T2 T2 - Implementation of a tier-2 coding */
/*@{*/

/** @name Local static functions */
/*@{*/

static void t2_putcommacode(opj_bio_t *bio, int n);
static int t2_getcommacode(opj_bio_t *bio);
/**
Variable length code for signalling delta Zil (truncation point)
@param bio Bit Input/Output component
@param n delta Zil
*/
static void t2_putnumpasses(opj_bio_t *bio, int n);
static int t2_getnumpasses(opj_bio_t *bio);
/**
Encode a packet of a tile to a destination buffer
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@param dest Destination buffer
@param len Length of the destination buffer
@param volume_info Structure to create an index file
@param tileno Number of the tile encoded
@param cp Coding parameters
@return Number of bytes encoded from the packet
*/
static int t2_encode_packet(opj_tcd_tile_t *tile, opj_tcp_t *tcp, opj_pi_iterator_t *pi, unsigned char *dest, int len, opj_volume_info_t *volume_info, int tileno, opj_cp_t *cp);
/**
Initialize the segment decoder
@param seg Segment instance
@param cblksty Codeblock style
@param first Is first segment
*/
static void t2_init_seg(opj_tcd_seg_t *seg, int cblksty, int first);
/**
Decode a packet of a tile from a source buffer
@param t2 T2 handle
@param src Source buffer
@param len Length of the source buffer
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@return Number of bytes decoded from the packet
*/
int t2_decode_packet(opj_t2_t* t2, unsigned char *src, int len, opj_tcd_tile_t *tile, opj_tcp_t *tcp, opj_pi_iterator_t *pi);

/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */

/* #define RESTART 0x04 */
static void t2_putcommacode(opj_bio_t *bio, int n) {
	while (--n >= 0) {
		bio_write(bio, 1, 1);
	}
	bio_write(bio, 0, 1);
}

static int t2_getcommacode(opj_bio_t *bio) {
	int n;
	for (n = 0; bio_read(bio, 1); n++) {
		;
	}
	return n;
}

static void t2_putnumpasses(opj_bio_t *bio, int n) {
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

static int t2_getnumpasses(opj_bio_t *bio) {
	int n;
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

static int t2_encode_packet(opj_tcd_tile_t * tile, opj_tcp_t * tcp, opj_pi_iterator_t *pi, unsigned char *dest, int len, opj_volume_info_t * volume_info, int tileno, opj_cp_t *cp) {
	int bandno, cblkno;
	unsigned char *sop = 0, *eph = 0;
	unsigned char *c = dest;

	int compno = pi->compno;	/* component value */
	int resno  = pi->resno;		/* resolution level value */
	int precno = pi->precno;	/* precinct value */
	int layno  = pi->layno;		/* quality layer value */

	opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
	opj_tcd_resolution_t *res = &tilec->resolutions[resno];
	
	opj_bio_t *bio = NULL;	/* BIO component */

	/* <SOP 0xff91> */
	if ((tcp->csty & J3D_CP_CSTY_SOP)) {
		sop = (unsigned char *) opj_malloc(6 * sizeof(unsigned char));
		sop[0] = 255;
		sop[1] = 145;
		sop[2] = 0;
		sop[3] = 4;
		sop[4] = (volume_info) ? (volume_info->num % 65536) / 256 : (0 % 65536) / 256 ;
		sop[5] = (volume_info) ? (volume_info->num % 65536) % 256 : (0 % 65536) % 256 ;
		memcpy(c, sop, 6);
		opj_free(sop);
		c += 6;
	} 
	/* </SOP> */
	
	if (!layno) {
		for (bandno = 0; bandno < res->numbands; bandno++) {
			opj_tcd_band_t *band = &res->bands[bandno];
			opj_tcd_precinct_t *prc = &band->precincts[precno];
			tgt_reset(prc->incltree);
			tgt_reset(prc->imsbtree);
			for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
				opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
				cblk->numpasses = 0;
                tgt_setvalue(prc->imsbtree, cblkno, band->numbps - cblk->numbps);
			}
		}
	}
		
	bio = bio_create();
	bio_init_enc(bio, c, len);
	bio_write(bio, 1, 1);		/* Empty header bit */
	
	/* Writing Packet header */
	for (bandno = 0; bandno < res->numbands; bandno++) {
		opj_tcd_band_t *band = &res->bands[bandno];
		opj_tcd_precinct_t *prc = &band->precincts[precno];
		for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
			opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
			opj_tcd_layer_t *layer = &cblk->layers[layno];
			if (!cblk->numpasses && layer->numpasses) {
                tgt_setvalue(prc->incltree, cblkno, layno);
			}
		}

		for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
			opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
			opj_tcd_layer_t *layer = &cblk->layers[layno];
			int increment = 0;
			int nump = 0;
			int len = 0, passno;
			/* cblk inclusion bits */
			if (!cblk->numpasses) {
				tgt_encode(bio, prc->incltree, cblkno, layno + 1);
			} else {
				bio_write(bio, layer->numpasses != 0, 1);
			}
			/* if cblk not included, go to the next cblk  */
			if (!layer->numpasses) {
				continue;
			}
			/* if first instance of cblk --> zero bit-planes information */
			if (!cblk->numpasses) {
				cblk->numlenbits = 3;
				tgt_encode(bio, prc->imsbtree, cblkno, 999);
			}
			/* number of coding passes included */
			t2_putnumpasses(bio, layer->numpasses);
			
			/* computation of the increase of the length indicator and insertion in the header     */
			for (passno = cblk->numpasses; passno < cblk->numpasses + layer->numpasses; passno++) {
				opj_tcd_pass_t *pass = &cblk->passes[passno];
				nump++;
				len += pass->len;
				if (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1) {
					increment = int_max(increment, int_floorlog2(len) + 1 - (cblk->numlenbits + int_floorlog2(nump)));
					len = 0;
					nump = 0;
				}
			}
			t2_putcommacode(bio, increment);

			/* computation of the new Length indicator */
			cblk->numlenbits += increment;

			/* insertion of the codeword segment length */
			for (passno = cblk->numpasses; passno < cblk->numpasses + layer->numpasses; passno++) {
				opj_tcd_pass_t *pass = &cblk->passes[passno];
				nump++;
				len += pass->len;
				if (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1) {
					bio_write(bio, len, cblk->numlenbits + int_floorlog2(nump));
					len = 0;
					nump = 0;
				}
			}

		}
	}
	
	
	if (bio_flush(bio)) {
		return -999;		/* modified to eliminate longjmp !! */
	}
	
	c += bio_numbytes(bio);

	bio_destroy(bio);
	
	/* <EPH 0xff92> */
	if (tcp->csty & J3D_CP_CSTY_EPH) {
		eph = (unsigned char *) opj_malloc(2 * sizeof(unsigned char));
		eph[0] = 255;
		eph[1] = 146;
		memcpy(c, eph, 2);
		opj_free(eph);
		c += 2;
	}
	/* </EPH> */
	
	/* Writing the packet body */
	
	for (bandno = 0; bandno < res->numbands; bandno++) {
		opj_tcd_band_t *band = &res->bands[bandno];
		opj_tcd_precinct_t *prc = &band->precincts[precno];
		for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
			opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
			opj_tcd_layer_t *layer = &cblk->layers[layno];
			if (!layer->numpasses) {
				continue;
			}
			if (c + layer->len > dest + len) {
				return -999;
			}
			
			memcpy(c, layer->data, layer->len);
			cblk->numpasses += layer->numpasses;
			c += layer->len;
			/* ADD for index Cfr. Marcela --> delta disto by packet */
			if(volume_info && volume_info->index_write && volume_info->index_on) {
				opj_tile_info_t *info_TL = &volume_info->tile[tileno];
				opj_packet_info_t *info_PK = &info_TL->packet[volume_info->num];
				info_PK->disto += layer->disto;
				if (volume_info->D_max < info_PK->disto) {
					volume_info->D_max = info_PK->disto;
				}
			}
			/* </ADD> */
		}
	}
	
	return (c - dest);
}

static void t2_init_seg(opj_tcd_seg_t * seg, int cblksty, int first) {
	seg->numpasses = 0;
	seg->len = 0;
	if (cblksty & J3D_CCP_CBLKSTY_TERMALL) {
		seg->maxpasses = 1;
	}
	else if (cblksty & J3D_CCP_CBLKSTY_LAZY) {
		if (first) {
			seg->maxpasses = 10;
		} else {
			seg->maxpasses = (((seg - 1)->maxpasses == 1) || ((seg - 1)->maxpasses == 10)) ? 2 : 1;
		}
	} else {
		seg->maxpasses = 109;
	}
}

int t2_decode_packet(opj_t2_t* t2, unsigned char *src, int len, opj_tcd_tile_t *tile, opj_tcp_t *tcp, opj_pi_iterator_t *pi) {
	int bandno, cblkno;
	unsigned char *c = src;

	opj_cp_t *cp = t2->cp;

	int compno = pi->compno;	/* component value */
	int resno  = pi->resno;		/* resolution level value */
	int precno = pi->precno;	/* precinct value */
	int layno  = pi->layno;		/* quality layer value */

	opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
	opj_tcd_resolution_t *res = &tilec->resolutions[resno];
	
	unsigned char *hd = NULL;
	int present;
	
	opj_bio_t *bio = NULL;	/* BIO component */
	
	if (layno == 0) {
		for (bandno = 0; bandno < res->numbands; bandno++) {
			opj_tcd_band_t *band = &res->bands[bandno];
			opj_tcd_precinct_t *prc = &band->precincts[precno];
			
			if ((band->x1-band->x0 == 0)||(band->y1-band->y0 == 0)||(band->z1-band->z0 == 0)) continue;

			tgt_reset(prc->incltree);
			tgt_reset(prc->imsbtree);
			for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
				opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
				cblk->numsegs = 0;
			}
		}
	}
	
	/* SOP markers */
	
	if (tcp->csty & J3D_CP_CSTY_SOP) {
		if ((*c) != 0xff || (*(c + 1) != 0x91)) {
			opj_event_msg(t2->cinfo, EVT_WARNING, "Expected SOP marker\n");
		} else {
			c += 6;
		}
		
		/** TODO : check the Nsop value */
	}
	
	/* 
	When the marker PPT/PPM is used the packet header are store in PPT/PPM marker
	This part deal with this caracteristic
	step 1: Read packet header in the saved structure
	step 2: Return to codestream for decoding 
	*/

	bio = bio_create();
	
	if (cp->ppm == 1) {		/* PPM */
		hd = cp->ppm_data;
		bio_init_dec(bio, hd, cp->ppm_len);
	} else if (tcp->ppt == 1) {	/* PPT */
		hd = tcp->ppt_data;
		bio_init_dec(bio, hd, tcp->ppt_len);
	} else {			/* Normal Case */
		hd = c;
		bio_init_dec(bio, hd, src+len-hd);
	}
	
	present = bio_read(bio, 1);
	
	if (!present) {
		bio_inalign(bio);
		hd += bio_numbytes(bio);
		bio_destroy(bio);
		
		/* EPH markers */
		
		if (tcp->csty & J3D_CP_CSTY_EPH) {
			if ((*hd) != 0xff || (*(hd + 1) != 0x92)) {
				printf("Error : expected EPH marker\n");
			} else {
				hd += 2;
			}
		}
		
		if (cp->ppm == 1) {		/* PPM case */
			cp->ppm_len += cp->ppm_data-hd;
			cp->ppm_data = hd;
			return (c - src);
		}
		if (tcp->ppt == 1) {	/* PPT case */
			tcp->ppt_len+=tcp->ppt_data-hd;
			tcp->ppt_data = hd;
			return (c - src);
		}
		
		return (hd - src);
	}
	
	for (bandno = 0; bandno < res->numbands; bandno++) {
		opj_tcd_band_t *band = &res->bands[bandno];
		opj_tcd_precinct_t *prc = &band->precincts[precno];
		
		if ((band->x1-band->x0 == 0)||(band->y1-band->y0 == 0)||(band->z1-band->z0 == 0)) continue;
		
		for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
			int included, increment, n;
			opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
			opj_tcd_seg_t *seg = NULL;
			/* if cblk not yet included before --> inclusion tagtree */
			if (!cblk->numsegs) {
                included = tgt_decode(bio, prc->incltree, cblkno, layno + 1);
				/* else one bit */
			} else {
				included = bio_read(bio, 1);
			}
			/* if cblk not included */
			if (!included) {
				cblk->numnewpasses = 0;
				continue;
			}
			/* if cblk not yet included --> zero-bitplane tagtree */
			if (!cblk->numsegs) {
				int i, numimsbs;
				for (i = 0; !tgt_decode(bio, prc->imsbtree, cblkno, i); i++);
				numimsbs = i - 1;
				cblk->numbps = band->numbps - numimsbs;
				cblk->numlenbits = 3;
			}
			/* number of coding passes */
			cblk->numnewpasses = t2_getnumpasses(bio);
			increment = t2_getcommacode(bio);
			/* length indicator increment */
			cblk->numlenbits += increment;
			if (!cblk->numsegs) {
				seg = &cblk->segs[0];
				t2_init_seg(seg, tcp->tccps[compno].cblksty, 1);
			} else {
				seg = &cblk->segs[cblk->numsegs - 1];
				if (seg->numpasses == seg->maxpasses) {
					t2_init_seg(++seg, tcp->tccps[compno].cblksty, 0);
				}
			}
			n = cblk->numnewpasses;
			
			do {
				seg->numnewpasses = int_min(seg->maxpasses - seg->numpasses, n);
				seg->newlen = bio_read(bio, cblk->numlenbits + int_floorlog2(seg->numnewpasses));
				n -= seg->numnewpasses;
				if (n > 0) {
					t2_init_seg(++seg, tcp->tccps[compno].cblksty, 0);
				}
			} while (n > 0);
		}
	}
	
	if (bio_inalign(bio)) {
		bio_destroy(bio);
		return -999;
	}
	
	hd += bio_numbytes(bio);
	bio_destroy(bio);
	
	/* EPH markers */
	if (tcp->csty & J3D_CP_CSTY_EPH) {
		if ((*hd) != 0xff || (*(hd + 1) != 0x92)) {
			opj_event_msg(t2->cinfo, EVT_ERROR, "Expected EPH marker\n");
		} else {
			hd += 2;
		}
	}
	
	if (cp->ppm==1) {
		cp->ppm_len+=cp->ppm_data-hd;
		cp->ppm_data = hd;
	} else if (tcp->ppt == 1) {
		tcp->ppt_len+=tcp->ppt_data-hd;
		tcp->ppt_data = hd;
	} else {
		c=hd;
	}
	
	for (bandno = 0; bandno < res->numbands; bandno++) {
		opj_tcd_band_t *band = &res->bands[bandno];
		opj_tcd_precinct_t *prc = &band->precincts[precno];
		
		if ((band->x1-band->x0 == 0)||(band->y1-band->y0 == 0)||(band->z1-band->z0 == 0)) continue;		

		for (cblkno = 0; cblkno < prc->cblkno[0] * prc->cblkno[1] * prc->cblkno[2]; cblkno++) {
			opj_tcd_cblk_t *cblk = &prc->cblks[cblkno];
			opj_tcd_seg_t *seg = NULL;
			if (!cblk->numnewpasses)
				continue;
			if (!cblk->numsegs) {
				seg = &cblk->segs[0];
				cblk->numsegs++;
				cblk->len = 0;
			} else {
				seg = &cblk->segs[cblk->numsegs - 1];
				if (seg->numpasses == seg->maxpasses) {
					seg++;
					cblk->numsegs++;
				}
			}
			
			do {
				if (c + seg->newlen > src + len) {
					return -999;
				}
				
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
	
	return (c - src);
}

/* ----------------------------------------------------------------------- */

int t2_encode_packets(opj_t2_t* t2, int tileno, opj_tcd_tile_t *tile, int maxlayers, unsigned char *dest, int len, opj_volume_info_t *volume_info) {
	unsigned char *c = dest;
	int e = 0;
	opj_pi_iterator_t *pi = NULL;
	int pino;

	opj_volume_t *volume = t2->volume;
	opj_cp_t *cp = t2->cp;
	
	/* create a packet iterator */
	pi = pi_create(volume, cp, tileno);
	if(!pi) {
		fprintf(stdout,"[ERROR] Failed to create a pi structure\n");
		return -999;
	}
	
	if(volume_info) {
		volume_info->num = 0;
	}
	
	for (pino = 0; pino <= cp->tcps[tileno].numpocs; pino++) {
		while (pi_next(&pi[pino])) {
			if (pi[pino].layno < maxlayers) {
				e = t2_encode_packet(tile, &cp->tcps[tileno], &pi[pino], c, dest + len - c, volume_info, tileno, cp);
				//opj_event_msg(t2->cinfo, EVT_INFO, "  t2_encode_packet: %d bytes coded\n",e);
				if (e == -999) {
					break;
				} else {
					c += e;
				}
				
				/* INDEX >> */
				if(volume_info && volume_info->index_on) {
					if(volume_info->index_write) {
						opj_tile_info_t *info_TL = &volume_info->tile[tileno];
						opj_packet_info_t *info_PK = &info_TL->packet[volume_info->num];
						if (!volume_info->num) {
							info_PK->start_pos = info_TL->end_header + 1;
						} else {
							info_PK->start_pos = info_TL->packet[volume_info->num - 1].end_pos + 1;
						}
						info_PK->end_pos = info_PK->start_pos + e - 1;
					}

					volume_info->num++;
				}
				/* << INDEX */
			}
		}
	}

	/* don't forget to release pi */
	pi_destroy(pi, cp, tileno);
	
	if (e == -999) {
		return e;
	}

    return (c - dest);
}

int t2_decode_packets(opj_t2_t *t2, unsigned char *src, int len, int tileno, opj_tcd_tile_t *tile) {
	unsigned char *c = src;
	opj_pi_iterator_t *pi;
	int pino, e = 0;
	int n = 0,i;

	opj_volume_t *volume = t2->volume;
	opj_cp_t *cp = t2->cp;
	
	/* create a packet iterator */
	pi = pi_create(volume, cp, tileno);
	if(!pi) {
		/* TODO: throw an error */
		return -999;
	}
	
	for (pino = 0; pino <= cp->tcps[tileno].numpocs; pino++) {
		while (pi_next(&pi[pino])) {
			if ((cp->layer==0) || (cp->layer>=((pi[pino].layno)+1))) {
				e = t2_decode_packet(t2, c, src + len - c, tile, &cp->tcps[tileno], &pi[pino]);
			} else {
				e = 0;
			}
			
			/* progression in resolution */
			for (i = 0; i < 3; i++){
                volume->comps[pi[pino].compno].resno_decoded[i] = (e > 0) ? int_max(pi[pino].resno, volume->comps[pi[pino].compno].resno_decoded[i]) : volume->comps[pi[pino].compno].resno_decoded[i];
			}
			n++;
			
			if (e == -999) {		/* ADD */
				break;
			} else {
				opj_event_msg(t2->cinfo, EVT_INFO, "  t2_decode_packet: %d bytes decoded\n",e);
				c += e;
			}
		}
	}

	/* don't forget to release pi */
	pi_destroy(pi, cp, tileno);
	
	if (e == -999) {
		return e;
	}
	
    return (c - src);
}

/* ----------------------------------------------------------------------- */

opj_t2_t* t2_create(opj_common_ptr cinfo, opj_volume_t *volume, opj_cp_t *cp) {
	/* create the tcd structure */
	opj_t2_t *t2 = (opj_t2_t*)opj_malloc(sizeof(opj_t2_t));
	if(!t2) return NULL;
	t2->cinfo = cinfo;
	t2->volume = volume;
	t2->cp = cp;

	return t2;
}

void t2_destroy(opj_t2_t *t2) {
	if(t2) {
		opj_free(t2);
	}
}

