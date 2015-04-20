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

/** @defgroup J2K J2K - JPEG-2000 codestream reader/writer */
/*@{*/

/** @name Local static functions */
/*@{*/

/**
Write the SOC marker (Start Of Codestream)
@param j2k J2K handle
*/
static void j2k_write_soc(opj_j2k_t *j2k);
/**
Read the SOC marker (Start of Codestream)
@param j2k J2K handle
*/
static void j2k_read_soc(opj_j2k_t *j2k);
/**
Write the SIZ marker (image and tile size)
@param j2k J2K handle
*/
static void j2k_write_siz(opj_j2k_t *j2k);
/**
Read the SIZ marker (image and tile size)
@param j2k J2K handle
*/
static void j2k_read_siz(opj_j2k_t *j2k);
/**
Write the COM marker (comment)
@param j2k J2K handle
*/
static void j2k_write_com(opj_j2k_t *j2k);
/**
Read the COM marker (comment)
@param j2k J2K handle
*/
static void j2k_read_com(opj_j2k_t *j2k);
/**
Write the value concerning the specified component in the marker COD and COC
@param j2k J2K handle
@param compno Number of the component concerned by the information written
*/
static void j2k_write_cox(opj_j2k_t *j2k, int compno);
/**
Read the value concerning the specified component in the marker COD and COC
@param j2k J2K handle
@param compno Number of the component concerned by the information read
*/
static void j2k_read_cox(opj_j2k_t *j2k, int compno);
/**
Write the COD marker (coding style default)
@param j2k J2K handle
*/
static void j2k_write_cod(opj_j2k_t *j2k);
/**
Read the COD marker (coding style default)
@param j2k J2K handle
*/
static void j2k_read_cod(opj_j2k_t *j2k);
/**
Write the COC marker (coding style component)
@param j2k J2K handle
@param compno Number of the component concerned by the information written
*/
static void j2k_write_coc(opj_j2k_t *j2k, int compno);
/**
Read the COC marker (coding style component)
@param j2k J2K handle
*/
static void j2k_read_coc(opj_j2k_t *j2k);
/**
Write the value concerning the specified component in the marker QCD and QCC
@param j2k J2K handle
@param compno Number of the component concerned by the information written
*/
static void j2k_write_qcx(opj_j2k_t *j2k, int compno);
/**
Read the value concerning the specified component in the marker QCD and QCC
@param j2k J2K handle
@param compno Number of the component concern by the information read
@param len Length of the information in the QCX part of the marker QCD/QCC
*/
static void j2k_read_qcx(opj_j2k_t *j2k, int compno, int len);
/**
Write the QCD marker (quantization default)
@param j2k J2K handle
*/
static void j2k_write_qcd(opj_j2k_t *j2k);
/**
Read the QCD marker (quantization default)
@param j2k J2K handle
*/
static void j2k_read_qcd(opj_j2k_t *j2k);
/**
Write the QCC marker (quantization component)
@param j2k J2K handle
@param compno Number of the component concerned by the information written
*/
static void j2k_write_qcc(opj_j2k_t *j2k, int compno);
/**
Read the QCC marker (quantization component)
@param j2k J2K handle
*/
static void j2k_read_qcc(opj_j2k_t *j2k);
/**
Write the POC marker (progression order change)
@param j2k J2K handle
*/
static void j2k_write_poc(opj_j2k_t *j2k);
/**
Read the POC marker (progression order change)
@param j2k J2K handle
*/
static void j2k_read_poc(opj_j2k_t *j2k);
/**
Read the CRG marker (component registration)
@param j2k J2K handle
*/
static void j2k_read_crg(opj_j2k_t *j2k);
/**
Read the TLM marker (tile-part lengths)
@param j2k J2K handle
*/
static void j2k_read_tlm(opj_j2k_t *j2k);
/**
Read the PLM marker (packet length, main header)
@param j2k J2K handle
*/
static void j2k_read_plm(opj_j2k_t *j2k);
/**
Read the PLT marker (packet length, tile-part header)
@param j2k J2K handle
*/
static void j2k_read_plt(opj_j2k_t *j2k);
/**
Read the PPM marker (packet packet headers, main header)
@param j2k J2K handle
*/
static void j2k_read_ppm(opj_j2k_t *j2k);
/**
Read the PPT marker (packet packet headers, tile-part header)
@param j2k J2K handle
*/
static void j2k_read_ppt(opj_j2k_t *j2k);
/**
Write the TLM marker (Mainheader)
@param j2k J2K handle
*/
static void j2k_write_tlm(opj_j2k_t *j2k);
/**
Write the SOT marker (start of tile-part)
@param j2k J2K handle
*/
static void j2k_write_sot(opj_j2k_t *j2k);
/**
Read the SOT marker (start of tile-part)
@param j2k J2K handle
*/
static void j2k_read_sot(opj_j2k_t *j2k);
/**
Write the SOD marker (start of data)
@param j2k J2K handle
@param tile_coder Pointer to a TCD handle
*/
static void j2k_write_sod(opj_j2k_t *j2k, void *tile_coder);
/**
Read the SOD marker (start of data)
@param j2k J2K handle
*/
static void j2k_read_sod(opj_j2k_t *j2k);
/**
Write the RGN marker (region-of-interest)
@param j2k J2K handle
@param compno Number of the component concerned by the information written
@param tileno Number of the tile concerned by the information written
*/
static void j2k_write_rgn(opj_j2k_t *j2k, int compno, int tileno);
/**
Read the RGN marker (region-of-interest)
@param j2k J2K handle
*/
static void j2k_read_rgn(opj_j2k_t *j2k);
/**
Write the EOC marker (end of codestream)
@param j2k J2K handle
*/
static void j2k_write_eoc(opj_j2k_t *j2k);
/**
Read the EOC marker (end of codestream)
@param j2k J2K handle
*/
static void j2k_read_eoc(opj_j2k_t *j2k);
/**
Read an unknown marker
@param j2k J2K handle
*/
static void j2k_read_unk(opj_j2k_t *j2k);

/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */
typedef struct j2k_prog_order{
	OPJ_PROG_ORDER enum_prog;
	char str_prog[4];
}j2k_prog_order_t;

j2k_prog_order_t j2k_prog_order_list[] = {
	{CPRL, "CPRL"},
	{LRCP, "LRCP"},
	{PCRL, "PCRL"},
	{RLCP, "RLCP"},
	{RPCL, "RPCL"},
	{(OPJ_PROG_ORDER)-1, ""}
};

char *j2k_convert_progression_order(OPJ_PROG_ORDER prg_order){
	j2k_prog_order_t *po;
	for(po = j2k_prog_order_list; po->enum_prog != -1; po++ ){
		if(po->enum_prog == prg_order){
			break;
		}
	}
	return po->str_prog;
}

/* ----------------------------------------------------------------------- */
static int j2k_get_num_tp(opj_cp_t *cp,int pino,int tileno){
	char *prog;
	int i;
	int tpnum=1,tpend=0;
	opj_tcp_t *tcp = &cp->tcps[tileno];
	prog = j2k_convert_progression_order(tcp->prg);
	
	if(cp->tp_on == 1){
		for(i=0;i<4;i++){
			if(tpend!=1){
				if( cp->tp_flag == prog[i] ){
					tpend=1;cp->tp_pos=i;
				}
				switch(prog[i]){
				case 'C':
					tpnum= tpnum * tcp->pocs[pino].compE;
					break;
				case 'R':
					tpnum= tpnum * tcp->pocs[pino].resE;
					break;
				case 'P':
					tpnum= tpnum * tcp->pocs[pino].prcE;
					break;
				case 'L':
					tpnum= tpnum * tcp->pocs[pino].layE;
					break;
				}
			}
		}
	}else{
		tpnum=1;
	}
	return tpnum;
}

/**	mem allocation for TLM marker*/
int j2k_calculate_tp(opj_cp_t *cp,int img_numcomp,opj_image_t *image,opj_j2k_t *j2k ){
	int pino,tileno,totnum_tp=0;
	j2k->cur_totnum_tp = (int *) opj_malloc(cp->tw * cp->th * sizeof(int));
	for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
		int cur_totnum_tp = 0;
		opj_tcp_t *tcp = &cp->tcps[tileno];
		for(pino = 0; pino <= tcp->numpocs; pino++) {
			int tp_num=0;
			opj_pi_iterator_t *pi = pi_initialise_encode(image, cp, tileno,FINAL_PASS);
			if(!pi) { return -1;}
			tp_num = j2k_get_num_tp(cp,pino,tileno);
			totnum_tp = totnum_tp + tp_num;
			cur_totnum_tp = cur_totnum_tp + tp_num;
			pi_destroy(pi, cp, tileno);
		}
		j2k->cur_totnum_tp[tileno] = cur_totnum_tp;
		/* INDEX >> */
		if (j2k->cstr_info) {
			j2k->cstr_info->tile[tileno].num_tps = cur_totnum_tp;
			j2k->cstr_info->tile[tileno].tp = (opj_tp_info_t *) opj_malloc(cur_totnum_tp * sizeof(opj_tp_info_t));
		}
		/* << INDEX */
	}
	return totnum_tp;
}

static void j2k_write_soc(opj_j2k_t *j2k) {
	opj_cio_t *cio = j2k->cio;
	cio_write(cio, J2K_MS_SOC, 2);

/* UniPG>> */
#ifdef USE_JPWL

	/* update markers struct */
	j2k_add_marker(j2k->cstr_info, J2K_MS_SOC, cio_tell(cio) - 2, 2);

#endif /* USE_JPWL */
/* <<UniPG */
}

static void j2k_read_soc(opj_j2k_t *j2k) {	
	j2k->state = J2K_STATE_MHSIZ;
	/* Index */
	if (j2k->cstr_info) {
		j2k->cstr_info->main_head_start = cio_tell(j2k->cio) - 2;
		j2k->cstr_info->codestream_size = cio_numbytesleft(j2k->cio) + 2 - j2k->cstr_info->main_head_start;
	}
}

static void j2k_write_siz(opj_j2k_t *j2k) {
	int i;
	int lenp, len;

	opj_cio_t *cio = j2k->cio;
	opj_image_t *image = j2k->image;
	opj_cp_t *cp = j2k->cp;
	
	cio_write(cio, J2K_MS_SIZ, 2);	/* SIZ */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio, cp->rsiz, 2);			/* Rsiz (capabilities) */
	cio_write(cio, image->x1, 4);	/* Xsiz */
	cio_write(cio, image->y1, 4);	/* Ysiz */
	cio_write(cio, image->x0, 4);	/* X0siz */
	cio_write(cio, image->y0, 4);	/* Y0siz */
	cio_write(cio, cp->tdx, 4);		/* XTsiz */
	cio_write(cio, cp->tdy, 4);		/* YTsiz */
	cio_write(cio, cp->tx0, 4);		/* XT0siz */
	cio_write(cio, cp->ty0, 4);		/* YT0siz */
	cio_write(cio, image->numcomps, 2);	/* Csiz */
	for (i = 0; i < image->numcomps; i++) {
		cio_write(cio, image->comps[i].prec - 1 + (image->comps[i].sgnd << 7), 1);	/* Ssiz_i */
		cio_write(cio, image->comps[i].dx, 1);	/* XRsiz_i */
		cio_write(cio, image->comps[i].dy, 1);	/* YRsiz_i */
	}
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Lsiz */
	cio_seek(cio, lenp + len);
}

static void j2k_read_siz(opj_j2k_t *j2k) {
	int len, i;
	
	opj_cio_t *cio = j2k->cio;
	opj_image_t *image = j2k->image;
	opj_cp_t *cp = j2k->cp;
	
	len = cio_read(cio, 2);			/* Lsiz */
	cio_read(cio, 2);				/* Rsiz (capabilities) */
	image->x1 = cio_read(cio, 4);	/* Xsiz */
	image->y1 = cio_read(cio, 4);	/* Ysiz */
	image->x0 = cio_read(cio, 4);	/* X0siz */
	image->y0 = cio_read(cio, 4);	/* Y0siz */
	cp->tdx = cio_read(cio, 4);		/* XTsiz */
	cp->tdy = cio_read(cio, 4);		/* YTsiz */
	cp->tx0 = cio_read(cio, 4);		/* XT0siz */
	cp->ty0 = cio_read(cio, 4);		/* YT0siz */
	
	if ((image->x0<0)||(image->x1<0)||(image->y0<0)||(image->y1<0)) {
		opj_event_msg(j2k->cinfo, EVT_ERROR,
									"%s: invalid image size (x0:%d, x1:%d, y0:%d, y1:%d)\n",
									image->x0,image->x1,image->y0,image->y1);
		return;
	}
	
	image->numcomps = cio_read(cio, 2);	/* Csiz */

#ifdef USE_JPWL
	if (j2k->cp->correct) {
		/* if JPWL is on, we check whether TX errors have damaged
		  too much the SIZ parameters */
		if (!(image->x1 * image->y1)) {
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"JPWL: bad image size (%d x %d)\n",
				image->x1, image->y1);
			if (!JPWL_ASSUME || JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
		}
		if (image->numcomps != ((len - 38) / 3)) {
			opj_event_msg(j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
				"JPWL: Csiz is %d => space in SIZ only for %d comps.!!!\n",
				image->numcomps, ((len - 38) / 3));
			if (!JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
			/* we try to correct */
			opj_event_msg(j2k->cinfo, EVT_WARNING, "- trying to adjust this\n");
			if (image->numcomps < ((len - 38) / 3)) {
				len = 38 + 3 * image->numcomps;
				opj_event_msg(j2k->cinfo, EVT_WARNING, "- setting Lsiz to %d => HYPOTHESIS!!!\n",
					len);				
			} else {
				image->numcomps = ((len - 38) / 3);
				opj_event_msg(j2k->cinfo, EVT_WARNING, "- setting Csiz to %d => HYPOTHESIS!!!\n",
					image->numcomps);				
			}
		}

		/* update components number in the jpwl_exp_comps filed */
		cp->exp_comps = image->numcomps;
	}
#endif /* USE_JPWL */

	image->comps = (opj_image_comp_t*) opj_calloc(image->numcomps, sizeof(opj_image_comp_t));
	for (i = 0; i < image->numcomps; i++) {
		int tmp, w, h;
		tmp = cio_read(cio, 1);		/* Ssiz_i */
		image->comps[i].prec = (tmp & 0x7f) + 1;
		image->comps[i].sgnd = tmp >> 7;
		image->comps[i].dx = cio_read(cio, 1);	/* XRsiz_i */
		image->comps[i].dy = cio_read(cio, 1);	/* YRsiz_i */
		
#ifdef USE_JPWL
		if (j2k->cp->correct) {
		/* if JPWL is on, we check whether TX errors have damaged
			too much the SIZ parameters, again */
			if (!(image->comps[i].dx * image->comps[i].dy)) {
				opj_event_msg(j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
					"JPWL: bad XRsiz_%d/YRsiz_%d (%d x %d)\n",
					i, i, image->comps[i].dx, image->comps[i].dy);
				if (!JPWL_ASSUME) {
					opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
					return;
				}
				/* we try to correct */
				opj_event_msg(j2k->cinfo, EVT_WARNING, "- trying to adjust them\n");
				if (!image->comps[i].dx) {
					image->comps[i].dx = 1;
					opj_event_msg(j2k->cinfo, EVT_WARNING, "- setting XRsiz_%d to %d => HYPOTHESIS!!!\n",
						i, image->comps[i].dx);
				}
				if (!image->comps[i].dy) {
					image->comps[i].dy = 1;
					opj_event_msg(j2k->cinfo, EVT_WARNING, "- setting YRsiz_%d to %d => HYPOTHESIS!!!\n",
						i, image->comps[i].dy);
				}
			}
			
		}
#endif /* USE_JPWL */

		/* TODO: unused ? */
		w = int_ceildiv(image->x1 - image->x0, image->comps[i].dx);
		h = int_ceildiv(image->y1 - image->y0, image->comps[i].dy);

		image->comps[i].resno_decoded = 0;	/* number of resolution decoded */
		image->comps[i].factor = cp->reduce; /* reducing factor per component */
	}
	
	cp->tw = int_ceildiv(image->x1 - cp->tx0, cp->tdx);
	cp->th = int_ceildiv(image->y1 - cp->ty0, cp->tdy);

#ifdef USE_JPWL
	if (j2k->cp->correct) {
		/* if JPWL is on, we check whether TX errors have damaged
		  too much the SIZ parameters */
		if ((cp->tw < 1) || (cp->th < 1) || (cp->tw > cp->max_tiles) || (cp->th > cp->max_tiles)) {
			opj_event_msg(j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
				"JPWL: bad number of tiles (%d x %d)\n",
				cp->tw, cp->th);
			if (!JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
			/* we try to correct */
			opj_event_msg(j2k->cinfo, EVT_WARNING, "- trying to adjust them\n");
			if (cp->tw < 1) {
				cp->tw= 1;
				opj_event_msg(j2k->cinfo, EVT_WARNING, "- setting %d tiles in x => HYPOTHESIS!!!\n",
					cp->tw);
			}
			if (cp->tw > cp->max_tiles) {
				cp->tw= 1;
				opj_event_msg(j2k->cinfo, EVT_WARNING, "- too large x, increase expectance of %d\n"
					"- setting %d tiles in x => HYPOTHESIS!!!\n",
					cp->max_tiles, cp->tw);
			}
			if (cp->th < 1) {
				cp->th= 1;
				opj_event_msg(j2k->cinfo, EVT_WARNING, "- setting %d tiles in y => HYPOTHESIS!!!\n",
					cp->th);
			}
			if (cp->th > cp->max_tiles) {
				cp->th= 1;
				opj_event_msg(j2k->cinfo, EVT_WARNING, "- too large y, increase expectance of %d to continue\n",
					"- setting %d tiles in y => HYPOTHESIS!!!\n",
					cp->max_tiles, cp->th);
			}
		}
	}
#endif /* USE_JPWL */

	cp->tcps = (opj_tcp_t*) opj_calloc(cp->tw * cp->th, sizeof(opj_tcp_t));
	cp->tileno = (int*) opj_malloc(cp->tw * cp->th * sizeof(int));
	cp->tileno_size = 0;
	
#ifdef USE_JPWL
	if (j2k->cp->correct) {
		if (!cp->tcps) {
			opj_event_msg(j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
				"JPWL: could not alloc tcps field of cp\n");
			if (!JPWL_ASSUME || JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
		}
	}
#endif /* USE_JPWL */

	for (i = 0; i < cp->tw * cp->th; i++) {
		cp->tcps[i].POC = 0;
		cp->tcps[i].numpocs = 0;
		cp->tcps[i].first = 1;
	}
	
	/* Initialization for PPM marker */
	cp->ppm = 0;
	cp->ppm_data = NULL;
	cp->ppm_data_first = NULL;
	cp->ppm_previous = 0;
	cp->ppm_store = 0;

	j2k->default_tcp->tccps = (opj_tccp_t*) opj_calloc(image->numcomps, sizeof(opj_tccp_t));
	for (i = 0; i < cp->tw * cp->th; i++) {
		cp->tcps[i].tccps = (opj_tccp_t*) opj_malloc(image->numcomps * sizeof(opj_tccp_t));
	}	
	j2k->tile_data = (unsigned char**) opj_calloc(cp->tw * cp->th, sizeof(unsigned char*));
	j2k->tile_len = (int*) opj_calloc(cp->tw * cp->th, sizeof(int));
	j2k->state = J2K_STATE_MH;

	/* Index */
	if (j2k->cstr_info) {
		opj_codestream_info_t *cstr_info = j2k->cstr_info;
		cstr_info->image_w = image->x1 - image->x0;
		cstr_info->image_h = image->y1 - image->y0;
		cstr_info->numcomps = image->numcomps;
		cstr_info->tw = cp->tw;
		cstr_info->th = cp->th;
		cstr_info->tile_x = cp->tdx;	
		cstr_info->tile_y = cp->tdy;	
		cstr_info->tile_Ox = cp->tx0;	
		cstr_info->tile_Oy = cp->ty0;			
		cstr_info->tile = (opj_tile_info_t*) opj_calloc(cp->tw * cp->th, sizeof(opj_tile_info_t));		
	}
}

static void j2k_write_com(opj_j2k_t *j2k) {
	unsigned int i;
	int lenp, len;

	if(j2k->cp->comment) {
		opj_cio_t *cio = j2k->cio;
		char *comment = j2k->cp->comment;

		cio_write(cio, J2K_MS_COM, 2);
		lenp = cio_tell(cio);
		cio_skip(cio, 2);
		cio_write(cio, 1, 2);		/* General use (IS 8859-15:1999 (Latin) values) */
		for (i = 0; i < strlen(comment); i++) {
			cio_write(cio, comment[i], 1);
		}
		len = cio_tell(cio) - lenp;
		cio_seek(cio, lenp);
		cio_write(cio, len, 2);
		cio_seek(cio, lenp + len);
	}
}

static void j2k_read_com(opj_j2k_t *j2k) {
	int len;
	
	opj_cio_t *cio = j2k->cio;

	len = cio_read(cio, 2);
	cio_skip(cio, len - 2);  
}

static void j2k_write_cox(opj_j2k_t *j2k, int compno) {
	int i;

	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = &cp->tcps[j2k->curtileno];
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j2k->cio;
	
	cio_write(cio, tccp->numresolutions - 1, 1);	/* SPcox (D) */
	cio_write(cio, tccp->cblkw - 2, 1);				/* SPcox (E) */
	cio_write(cio, tccp->cblkh - 2, 1);				/* SPcox (F) */
	cio_write(cio, tccp->cblksty, 1);				/* SPcox (G) */
	cio_write(cio, tccp->qmfbid, 1);				/* SPcox (H) */
	
	if (tccp->csty & J2K_CCP_CSTY_PRT) {
		for (i = 0; i < tccp->numresolutions; i++) {
			cio_write(cio, tccp->prcw[i] + (tccp->prch[i] << 4), 1);	/* SPcox (I_i) */
		}
	}
}

static void j2k_read_cox(opj_j2k_t *j2k, int compno) {
	int i;

	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = j2k->state == J2K_STATE_TPH ? &cp->tcps[j2k->curtileno] : j2k->default_tcp;
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j2k->cio;

	tccp->numresolutions = cio_read(cio, 1) + 1;	/* SPcox (D) */

	// If user wants to remove more resolutions than the codestream contains, return error
	if (cp->reduce >= tccp->numresolutions) {
		opj_event_msg(j2k->cinfo, EVT_ERROR, "Error decoding component %d.\nThe number of resolutions to remove is higher than the number "
					"of resolutions of this component\nModify the cp_reduce parameter.\n\n", compno);
		j2k->state |= J2K_STATE_ERR;
	}

	tccp->cblkw = cio_read(cio, 1) + 2;	/* SPcox (E) */
	tccp->cblkh = cio_read(cio, 1) + 2;	/* SPcox (F) */
	tccp->cblksty = cio_read(cio, 1);	/* SPcox (G) */
	tccp->qmfbid = cio_read(cio, 1);	/* SPcox (H) */
	if (tccp->csty & J2K_CP_CSTY_PRT) {
		for (i = 0; i < tccp->numresolutions; i++) {
			int tmp = cio_read(cio, 1);	/* SPcox (I_i) */
			tccp->prcw[i] = tmp & 0xf;
			tccp->prch[i] = tmp >> 4;
		}
	}

	/* INDEX >> */
	if(j2k->cstr_info && compno == 0) {
		for (i = 0; i < tccp->numresolutions; i++) {
			if (tccp->csty & J2K_CP_CSTY_PRT) {
				j2k->cstr_info->tile[j2k->curtileno].pdx[i] = tccp->prcw[i];
				j2k->cstr_info->tile[j2k->curtileno].pdy[i] = tccp->prch[i];
			}
			else {
				j2k->cstr_info->tile[j2k->curtileno].pdx[i] = 15;
				j2k->cstr_info->tile[j2k->curtileno].pdx[i] = 15;
			}
		}
	}
	/* << INDEX */
}

static void j2k_write_cod(opj_j2k_t *j2k) {
	opj_cp_t *cp = NULL;
	opj_tcp_t *tcp = NULL;
	int lenp, len;

	opj_cio_t *cio = j2k->cio;
	
	cio_write(cio, J2K_MS_COD, 2);	/* COD */
	
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	
	cp = j2k->cp;
	tcp = &cp->tcps[j2k->curtileno];

	cio_write(cio, tcp->csty, 1);		/* Scod */
	cio_write(cio, tcp->prg, 1);		/* SGcod (A) */
	cio_write(cio, tcp->numlayers, 2);	/* SGcod (B) */
	cio_write(cio, tcp->mct, 1);		/* SGcod (C) */
	
	j2k_write_cox(j2k, 0);
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Lcod */
	cio_seek(cio, lenp + len);
}

static void j2k_read_cod(opj_j2k_t *j2k) {
	int len, i, pos;
	
	opj_cio_t *cio = j2k->cio;
	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = j2k->state == J2K_STATE_TPH ? &cp->tcps[j2k->curtileno] : j2k->default_tcp;
	opj_image_t *image = j2k->image;
	
	len = cio_read(cio, 2);				/* Lcod */
	tcp->csty = cio_read(cio, 1);		/* Scod */
	tcp->prg = (OPJ_PROG_ORDER)cio_read(cio, 1);		/* SGcod (A) */
	tcp->numlayers = cio_read(cio, 2);	/* SGcod (B) */
	tcp->mct = cio_read(cio, 1);		/* SGcod (C) */
	
	pos = cio_tell(cio);
	for (i = 0; i < image->numcomps; i++) {
		tcp->tccps[i].csty = tcp->csty & J2K_CP_CSTY_PRT;
		cio_seek(cio, pos);
		j2k_read_cox(j2k, i);
	}

	/* Index */
	if (j2k->cstr_info) {
		opj_codestream_info_t *cstr_info = j2k->cstr_info;
		cstr_info->prog = tcp->prg;
		cstr_info->numlayers = tcp->numlayers;
		cstr_info->numdecompos = (int*) opj_malloc(image->numcomps * sizeof(int));
		for (i = 0; i < image->numcomps; i++) {
			cstr_info->numdecompos[i] = tcp->tccps[i].numresolutions - 1;
		}
	}
}

static void j2k_write_coc(opj_j2k_t *j2k, int compno) {
	int lenp, len;

	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = &cp->tcps[j2k->curtileno];
	opj_image_t *image = j2k->image;
	opj_cio_t *cio = j2k->cio;
	
	cio_write(cio, J2K_MS_COC, 2);	/* COC */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio, compno, image->numcomps <= 256 ? 1 : 2);	/* Ccoc */
	cio_write(cio, tcp->tccps[compno].csty, 1);	/* Scoc */
	j2k_write_cox(j2k, compno);
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);			/* Lcoc */
	cio_seek(cio, lenp + len);
}

static void j2k_read_coc(opj_j2k_t *j2k) {
	int len, compno;

	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = j2k->state == J2K_STATE_TPH ? &cp->tcps[j2k->curtileno] : j2k->default_tcp;
	opj_image_t *image = j2k->image;
	opj_cio_t *cio = j2k->cio;
	
	len = cio_read(cio, 2);		/* Lcoc */
	compno = cio_read(cio, image->numcomps <= 256 ? 1 : 2);	/* Ccoc */
	tcp->tccps[compno].csty = cio_read(cio, 1);	/* Scoc */
	j2k_read_cox(j2k, compno);
}

static void j2k_write_qcx(opj_j2k_t *j2k, int compno) {
	int bandno, numbands;
	int expn, mant;
	
	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = &cp->tcps[j2k->curtileno];
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j2k->cio;
	
	cio_write(cio, tccp->qntsty + (tccp->numgbits << 5), 1);	/* Sqcx */
	numbands = tccp->qntsty == J2K_CCP_QNTSTY_SIQNT ? 1 : tccp->numresolutions * 3 - 2;
	
	for (bandno = 0; bandno < numbands; bandno++) {
		expn = tccp->stepsizes[bandno].expn;
		mant = tccp->stepsizes[bandno].mant;
		
		if (tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) {
			cio_write(cio, expn << 3, 1);	/* SPqcx_i */
		} else {
			cio_write(cio, (expn << 11) + mant, 2);	/* SPqcx_i */
		}
	}
}

static void j2k_read_qcx(opj_j2k_t *j2k, int compno, int len) {
	int tmp;
	int bandno, numbands;

	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = j2k->state == J2K_STATE_TPH ? &cp->tcps[j2k->curtileno] : j2k->default_tcp;
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j2k->cio;

	tmp = cio_read(cio, 1);		/* Sqcx */
	tccp->qntsty = tmp & 0x1f;
	tccp->numgbits = tmp >> 5;
	numbands = (tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) ? 
		1 : ((tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) ? len - 1 : (len - 1) / 2);

#ifdef USE_JPWL
	if (j2k->cp->correct) {

		/* if JPWL is on, we check whether there are too many subbands */
		if ((numbands < 0) || (numbands >= J2K_MAXBANDS)) {
			opj_event_msg(j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
				"JPWL: bad number of subbands in Sqcx (%d)\n",
				numbands);
			if (!JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
			/* we try to correct */
			numbands = 1;
			opj_event_msg(j2k->cinfo, EVT_WARNING, "- trying to adjust them\n"
				"- setting number of bands to %d => HYPOTHESIS!!!\n",
				numbands);
		};

	};
#endif /* USE_JPWL */

	for (bandno = 0; bandno < numbands; bandno++) {
		int expn, mant;
		if (tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) {
			expn = cio_read(cio, 1) >> 3;	/* SPqcx_i */
			mant = 0;
		} else {
			tmp = cio_read(cio, 2);	/* SPqcx_i */
			expn = tmp >> 11;
			mant = tmp & 0x7ff;
		}
		tccp->stepsizes[bandno].expn = expn;
		tccp->stepsizes[bandno].mant = mant;
	}
	
	/* Add Antonin : if scalar_derived -> compute other stepsizes */
	if (tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) {
		for (bandno = 1; bandno < J2K_MAXBANDS; bandno++) {
			tccp->stepsizes[bandno].expn = 
				((tccp->stepsizes[0].expn) - ((bandno - 1) / 3) > 0) ? 
					(tccp->stepsizes[0].expn) - ((bandno - 1) / 3) : 0;
			tccp->stepsizes[bandno].mant = tccp->stepsizes[0].mant;
		}
	}
	/* ddA */
}

static void j2k_write_qcd(opj_j2k_t *j2k) {
	int lenp, len;

	opj_cio_t *cio = j2k->cio;
	
	cio_write(cio, J2K_MS_QCD, 2);	/* QCD */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	j2k_write_qcx(j2k, 0);
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);			/* Lqcd */
	cio_seek(cio, lenp + len);
}

static void j2k_read_qcd(opj_j2k_t *j2k) {
	int len, i, pos;

	opj_cio_t *cio = j2k->cio;
	opj_image_t *image = j2k->image;
	
	len = cio_read(cio, 2);		/* Lqcd */
	pos = cio_tell(cio);
	for (i = 0; i < image->numcomps; i++) {
		cio_seek(cio, pos);
		j2k_read_qcx(j2k, i, len - 2);
	}
}

static void j2k_write_qcc(opj_j2k_t *j2k, int compno) {
	int lenp, len;

	opj_cio_t *cio = j2k->cio;
	
	cio_write(cio, J2K_MS_QCC, 2);	/* QCC */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio, compno, j2k->image->numcomps <= 256 ? 1 : 2);	/* Cqcc */
	j2k_write_qcx(j2k, compno);
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);			/* Lqcc */
	cio_seek(cio, lenp + len);
}

static void j2k_read_qcc(opj_j2k_t *j2k) {
	int len, compno;
	int numcomp = j2k->image->numcomps;
	opj_cio_t *cio = j2k->cio;
	
	len = cio_read(cio, 2);	/* Lqcc */
	compno = cio_read(cio, numcomp <= 256 ? 1 : 2);	/* Cqcc */

#ifdef USE_JPWL
	if (j2k->cp->correct) {

		static int backup_compno = 0;

		/* compno is negative or larger than the number of components!!! */
		if ((compno < 0) || (compno >= numcomp)) {
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"JPWL: bad component number in QCC (%d out of a maximum of %d)\n",
				compno, numcomp);
			if (!JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
			/* we try to correct */
			compno = backup_compno % numcomp;
			opj_event_msg(j2k->cinfo, EVT_WARNING, "- trying to adjust this\n"
				"- setting component number to %d\n",
				compno);
		}

		/* keep your private count of tiles */
		backup_compno++;
	};
#endif /* USE_JPWL */

	j2k_read_qcx(j2k, compno, len - 2 - (numcomp <= 256 ? 1 : 2));
}

static void j2k_write_poc(opj_j2k_t *j2k) {
	int len, numpchgs, i;

	int numcomps = j2k->image->numcomps;
	
	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = &cp->tcps[j2k->curtileno];
	opj_tccp_t *tccp = &tcp->tccps[0];
	opj_cio_t *cio = j2k->cio;

	numpchgs = 1 + tcp->numpocs;
	cio_write(cio, J2K_MS_POC, 2);	/* POC  */
	len = 2 + (5 + 2 * (numcomps <= 256 ? 1 : 2)) * numpchgs;
	cio_write(cio, len, 2);		/* Lpoc */
	for (i = 0; i < numpchgs; i++) {
		opj_poc_t *poc = &tcp->pocs[i];
		cio_write(cio, poc->resno0, 1);	/* RSpoc_i */
		cio_write(cio, poc->compno0, (numcomps <= 256 ? 1 : 2));	/* CSpoc_i */
		cio_write(cio, poc->layno1, 2);	/* LYEpoc_i */
		poc->layno1 = int_min(poc->layno1, tcp->numlayers);
		cio_write(cio, poc->resno1, 1);	/* REpoc_i */
		poc->resno1 = int_min(poc->resno1, tccp->numresolutions);
		cio_write(cio, poc->compno1, (numcomps <= 256 ? 1 : 2));	/* CEpoc_i */
		poc->compno1 = int_min(poc->compno1, numcomps);
		cio_write(cio, poc->prg, 1);	/* Ppoc_i */
	}
}

static void j2k_read_poc(opj_j2k_t *j2k) {
	int len, numpchgs, i, old_poc;

	int numcomps = j2k->image->numcomps;
	
	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = j2k->state == J2K_STATE_TPH ? &cp->tcps[j2k->curtileno] : j2k->default_tcp;
	opj_cio_t *cio = j2k->cio;
	
	old_poc = tcp->POC ? tcp->numpocs + 1 : 0;
	tcp->POC = 1;
	len = cio_read(cio, 2);		/* Lpoc */
	numpchgs = (len - 2) / (5 + 2 * (numcomps <= 256 ? 1 : 2));
	
	for (i = old_poc; i < numpchgs + old_poc; i++) {
		opj_poc_t *poc;
		poc = &tcp->pocs[i];
		poc->resno0 = cio_read(cio, 1);	/* RSpoc_i */
		poc->compno0 = cio_read(cio, numcomps <= 256 ? 1 : 2);	/* CSpoc_i */
		poc->layno1 = cio_read(cio, 2);    /* LYEpoc_i */
		poc->resno1 = cio_read(cio, 1);    /* REpoc_i */
		poc->compno1 = int_min(
			cio_read(cio, numcomps <= 256 ? 1 : 2), (unsigned int) numcomps);	/* CEpoc_i */
		poc->prg = (OPJ_PROG_ORDER)cio_read(cio, 1);	/* Ppoc_i */
	}
	
	tcp->numpocs = numpchgs + old_poc - 1;
}

static void j2k_read_crg(opj_j2k_t *j2k) {
	int len, i, Xcrg_i, Ycrg_i;
	
	opj_cio_t *cio = j2k->cio;
	int numcomps = j2k->image->numcomps;
	
	len = cio_read(cio, 2);			/* Lcrg */
	for (i = 0; i < numcomps; i++) {
		Xcrg_i = cio_read(cio, 2);	/* Xcrg_i */
		Ycrg_i = cio_read(cio, 2);	/* Ycrg_i */
	}
}

static void j2k_read_tlm(opj_j2k_t *j2k) {
	int len, Ztlm, Stlm, ST, SP, tile_tlm, i;
	long int Ttlm_i, Ptlm_i;

	opj_cio_t *cio = j2k->cio;
	
	len = cio_read(cio, 2);		/* Ltlm */
	Ztlm = cio_read(cio, 1);	/* Ztlm */
	Stlm = cio_read(cio, 1);	/* Stlm */
	ST = ((Stlm >> 4) & 0x01) + ((Stlm >> 4) & 0x02);
	SP = (Stlm >> 6) & 0x01;
	tile_tlm = (len - 4) / ((SP + 1) * 2 + ST);
	for (i = 0; i < tile_tlm; i++) {
		Ttlm_i = cio_read(cio, ST);	/* Ttlm_i */
		Ptlm_i = cio_read(cio, SP ? 4 : 2);	/* Ptlm_i */
	}
}

static void j2k_read_plm(opj_j2k_t *j2k) {
	int len, i, Zplm, Nplm, add, packet_len = 0;
	
	opj_cio_t *cio = j2k->cio;

	len = cio_read(cio, 2);		/* Lplm */
	Zplm = cio_read(cio, 1);	/* Zplm */
	len -= 3;
	while (len > 0) {
		Nplm = cio_read(cio, 4);		/* Nplm */
		len -= 4;
		for (i = Nplm; i > 0; i--) {
			add = cio_read(cio, 1);
			len--;
			packet_len = (packet_len << 7) + add;	/* Iplm_ij */
			if ((add & 0x80) == 0) {
				/* New packet */
				packet_len = 0;
			}
			if (len <= 0)
				break;
		}
	}
}

static void j2k_read_plt(opj_j2k_t *j2k) {
	int len, i, Zplt, packet_len = 0, add;
	
	opj_cio_t *cio = j2k->cio;
	
	len = cio_read(cio, 2);		/* Lplt */
	Zplt = cio_read(cio, 1);	/* Zplt */
	for (i = len - 3; i > 0; i--) {
		add = cio_read(cio, 1);
		packet_len = (packet_len << 7) + add;	/* Iplt_i */
		if ((add & 0x80) == 0) {
			/* New packet */
			packet_len = 0;
		}
	}
}

static void j2k_read_ppm(opj_j2k_t *j2k) {
	int len, Z_ppm, i, j;
	int N_ppm;

	opj_cp_t *cp = j2k->cp;
	opj_cio_t *cio = j2k->cio;
	
	len = cio_read(cio, 2);
	cp->ppm = 1;
	
	Z_ppm = cio_read(cio, 1);	/* Z_ppm */
	len -= 3;
	while (len > 0) {
		if (cp->ppm_previous == 0) {
			N_ppm = cio_read(cio, 4);	/* N_ppm */
			len -= 4;
		} else {
			N_ppm = cp->ppm_previous;
		}
		j = cp->ppm_store;
		if (Z_ppm == 0) {	/* First PPM marker */
			cp->ppm_data = (unsigned char *) opj_malloc(N_ppm * sizeof(unsigned char));
			cp->ppm_data_first = cp->ppm_data;
			cp->ppm_len = N_ppm;
		} else {			/* NON-first PPM marker */
			cp->ppm_data = (unsigned char *) opj_realloc(cp->ppm_data, (N_ppm +	cp->ppm_store) * sizeof(unsigned char));

#ifdef USE_JPWL
			/* this memory allocation check could be done even in non-JPWL cases */
			if (cp->correct) {
				if (!cp->ppm_data) {
					opj_event_msg(j2k->cinfo, EVT_ERROR,
						"JPWL: failed memory allocation during PPM marker parsing (pos. %x)\n",
						cio_tell(cio));
					if (!JPWL_ASSUME || JPWL_ASSUME) {
						opj_free(cp->ppm_data);
						opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
						return;
					}
				}
			}
#endif

			cp->ppm_data_first = cp->ppm_data;
			cp->ppm_len = N_ppm + cp->ppm_store;
		}
		for (i = N_ppm; i > 0; i--) {	/* Read packet header */
			cp->ppm_data[j] = cio_read(cio, 1);
			j++;
			len--;
			if (len == 0)
				break;			/* Case of non-finished packet header in present marker but finished in next one */
		}
		cp->ppm_previous = i - 1;
		cp->ppm_store = j;
	}
}

static void j2k_read_ppt(opj_j2k_t *j2k) {
	int len, Z_ppt, i, j = 0;

	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = cp->tcps + j2k->curtileno;
	opj_cio_t *cio = j2k->cio;

	len = cio_read(cio, 2);
	Z_ppt = cio_read(cio, 1);
	tcp->ppt = 1;
	if (Z_ppt == 0) {		/* First PPT marker */
		tcp->ppt_data = (unsigned char *) opj_malloc((len - 3) * sizeof(unsigned char));
		tcp->ppt_data_first = tcp->ppt_data;
		tcp->ppt_store = 0;
		tcp->ppt_len = len - 3;
	} else {			/* NON-first PPT marker */
		tcp->ppt_data =	(unsigned char *) opj_realloc(tcp->ppt_data, (len - 3 + tcp->ppt_store) * sizeof(unsigned char));
		tcp->ppt_data_first = tcp->ppt_data;
		tcp->ppt_len = len - 3 + tcp->ppt_store;
	}
	j = tcp->ppt_store;
	for (i = len - 3; i > 0; i--) {
		tcp->ppt_data[j] = cio_read(cio, 1);
		j++;
	}
	tcp->ppt_store = j;
}

static void j2k_write_tlm(opj_j2k_t *j2k){
	int lenp;
	opj_cio_t *cio = j2k->cio;
	j2k->tlm_start = cio_tell(cio);
	cio_write(cio, J2K_MS_TLM, 2);/* TLM */
	lenp = 4 + (5*j2k->totnum_tp);
	cio_write(cio,lenp,2);				/* Ltlm */
	cio_write(cio, 0,1);					/* Ztlm=0*/
	cio_write(cio,80,1);					/* Stlm ST=1(8bits-255 tiles max),SP=1(Ptlm=32bits) */
	cio_skip(cio,5*j2k->totnum_tp);
}

static void j2k_write_sot(opj_j2k_t *j2k) {
	int lenp, len;

	opj_cio_t *cio = j2k->cio;

	j2k->sot_start = cio_tell(cio);
	cio_write(cio, J2K_MS_SOT, 2);		/* SOT */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);					/* Lsot (further) */
	cio_write(cio, j2k->curtileno, 2);	/* Isot */
	cio_skip(cio, 4);					/* Psot (further in j2k_write_sod) */
	cio_write(cio, j2k->cur_tp_num , 1);	/* TPsot */
	cio_write(cio, j2k->cur_totnum_tp[j2k->curtileno], 1);		/* TNsot */
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);				/* Lsot */
	cio_seek(cio, lenp + len);

	/* UniPG>> */
#ifdef USE_JPWL
	/* update markers struct */
	j2k_add_marker(j2k->cstr_info, J2K_MS_SOT, j2k->sot_start, len + 2);
#endif /* USE_JPWL */
	/* <<UniPG */
}

static void j2k_read_sot(opj_j2k_t *j2k) {
	int len, tileno, totlen, partno, numparts, i;
	opj_tcp_t *tcp = NULL;
	char status = 0;

	opj_cp_t *cp = j2k->cp;
	opj_cio_t *cio = j2k->cio;

	len = cio_read(cio, 2);
	tileno = cio_read(cio, 2);

#ifdef USE_JPWL
	if (j2k->cp->correct) {

		static int backup_tileno = 0;

		/* tileno is negative or larger than the number of tiles!!! */
		if ((tileno < 0) || (tileno > (cp->tw * cp->th))) {
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"JPWL: bad tile number (%d out of a maximum of %d)\n",
				tileno, (cp->tw * cp->th));
			if (!JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
			/* we try to correct */
			tileno = backup_tileno;
			opj_event_msg(j2k->cinfo, EVT_WARNING, "- trying to adjust this\n"
				"- setting tile number to %d\n",
				tileno);
		}

		/* keep your private count of tiles */
		backup_tileno++;
	};
#endif /* USE_JPWL */
	
	if (cp->tileno_size == 0) {
		cp->tileno[cp->tileno_size] = tileno;
		cp->tileno_size++;
	} else {
		i = 0;
		while (i < cp->tileno_size && status == 0) {
			status = cp->tileno[i] == tileno ? 1 : 0;
			i++;
		}
		if (status == 0) {
			cp->tileno[cp->tileno_size] = tileno;
			cp->tileno_size++;
		}
	}
	
	totlen = cio_read(cio, 4);

#ifdef USE_JPWL
	if (j2k->cp->correct) {

		/* totlen is negative or larger than the bytes left!!! */
		if ((totlen < 0) || (totlen > (cio_numbytesleft(cio) + 8))) {
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"JPWL: bad tile byte size (%d bytes against %d bytes left)\n",
				totlen, cio_numbytesleft(cio) + 8);
			if (!JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
			/* we try to correct */
			totlen = 0;
			opj_event_msg(j2k->cinfo, EVT_WARNING, "- trying to adjust this\n"
				"- setting Psot to %d => assuming it is the last tile\n",
				totlen);
		}

	};
#endif /* USE_JPWL */

	if (!totlen)
		totlen = cio_numbytesleft(cio) + 8;
	
	partno = cio_read(cio, 1);
	numparts = cio_read(cio, 1);
	
	j2k->curtileno = tileno;
	j2k->cur_tp_num = partno;
	j2k->eot = cio_getbp(cio) - 12 + totlen;
	j2k->state = J2K_STATE_TPH;
	tcp = &cp->tcps[j2k->curtileno];

	/* Index */
	if (j2k->cstr_info) {
		if (tcp->first) {
			if (tileno == 0) 
				j2k->cstr_info->main_head_end = cio_tell(cio) - 13;
			j2k->cstr_info->tile[tileno].tileno = tileno;
			j2k->cstr_info->tile[tileno].start_pos = cio_tell(cio) - 12;
			j2k->cstr_info->tile[tileno].end_pos = j2k->cstr_info->tile[tileno].start_pos + totlen - 1;				
			j2k->cstr_info->tile[tileno].num_tps = numparts;
			if (numparts)
				j2k->cstr_info->tile[tileno].tp = (opj_tp_info_t *) opj_malloc(numparts * sizeof(opj_tp_info_t));
			else
				j2k->cstr_info->tile[tileno].tp = (opj_tp_info_t *) opj_malloc(10 * sizeof(opj_tp_info_t)); // Fixme (10)
		}
		else {
			j2k->cstr_info->tile[tileno].end_pos += totlen;
		}		
		j2k->cstr_info->tile[tileno].tp[partno].tp_start_pos = cio_tell(cio) - 12;
		j2k->cstr_info->tile[tileno].tp[partno].tp_end_pos = 
			j2k->cstr_info->tile[tileno].tp[partno].tp_start_pos + totlen - 1;
	}
	
	if (tcp->first == 1) {		
		/* Initialization PPT */
		opj_tccp_t *tmp = tcp->tccps;
		memcpy(tcp, j2k->default_tcp, sizeof(opj_tcp_t));
		tcp->ppt = 0;
		tcp->ppt_data = NULL;
		tcp->ppt_data_first = NULL;
		tcp->tccps = tmp;

		for (i = 0; i < j2k->image->numcomps; i++) {
			tcp->tccps[i] = j2k->default_tcp->tccps[i];
		}
		cp->tcps[j2k->curtileno].first = 0;
	}
}

static void j2k_write_sod(opj_j2k_t *j2k, void *tile_coder) {
	int l, layno;
	int totlen;
	opj_tcp_t *tcp = NULL;
	opj_codestream_info_t *cstr_info = NULL;
	
	opj_tcd_t *tcd = (opj_tcd_t*)tile_coder;	/* cast is needed because of conflicts in header inclusions */
	opj_cp_t *cp = j2k->cp;
	opj_cio_t *cio = j2k->cio;

	tcd->tp_num = j2k->tp_num ;
	tcd->cur_tp_num = j2k->cur_tp_num;
	
	cio_write(cio, J2K_MS_SOD, 2);
	if (j2k->curtileno == 0) {
		j2k->sod_start = cio_tell(cio) + j2k->pos_correction;
	}

	/* INDEX >> */
	cstr_info = j2k->cstr_info;
	if (cstr_info) {
		if (!j2k->cur_tp_num ) {
			cstr_info->tile[j2k->curtileno].end_header = cio_tell(cio) + j2k->pos_correction - 1;
			j2k->cstr_info->tile[j2k->curtileno].tileno = j2k->curtileno;
		}
		else{
			if(cstr_info->tile[j2k->curtileno].packet[cstr_info->packno - 1].end_pos < cio_tell(cio))
				cstr_info->tile[j2k->curtileno].packet[cstr_info->packno].start_pos = cio_tell(cio);
		}
		/* UniPG>> */
#ifdef USE_JPWL
		/* update markers struct */
		j2k_add_marker(j2k->cstr_info, J2K_MS_SOD, j2k->sod_start, 2);
#endif /* USE_JPWL */
		/* <<UniPG */
	}
	/* << INDEX */
	
	tcp = &cp->tcps[j2k->curtileno];
	for (layno = 0; layno < tcp->numlayers; layno++) {
		if (tcp->rates[layno]>(j2k->sod_start / (cp->th * cp->tw))) {
			tcp->rates[layno]-=(j2k->sod_start / (cp->th * cp->tw));
		} else if (tcp->rates[layno]) {
			tcp->rates[layno]=1;
		}
	}
	if(j2k->cur_tp_num == 0){
		tcd->tcd_image->tiles->packno = 0;
		if(cstr_info)
			cstr_info->packno = 0;
	}
	
	l = tcd_encode_tile(tcd, j2k->curtileno, cio_getbp(cio), cio_numbytesleft(cio) - 2, cstr_info);
	
	/* Writing Psot in SOT marker */
	totlen = cio_tell(cio) + l - j2k->sot_start;
	cio_seek(cio, j2k->sot_start + 6);
	cio_write(cio, totlen, 4);
	cio_seek(cio, j2k->sot_start + totlen);
	/* Writing Ttlm and Ptlm in TLM marker */
	if(cp->cinema){
		cio_seek(cio, j2k->tlm_start + 6 + (5*j2k->cur_tp_num));
		cio_write(cio, j2k->curtileno, 1);
		cio_write(cio, totlen, 4);
	}
	cio_seek(cio, j2k->sot_start + totlen);
}

static void j2k_read_sod(opj_j2k_t *j2k) {
	int len, truncate = 0, i;
	unsigned char *data = NULL, *data_ptr = NULL;

	opj_cio_t *cio = j2k->cio;
	int curtileno = j2k->curtileno;

	/* Index */
	if (j2k->cstr_info) {
		j2k->cstr_info->tile[j2k->curtileno].tp[j2k->cur_tp_num].tp_end_header =
			cio_tell(cio) + j2k->pos_correction - 1;
		if (j2k->cur_tp_num == 0)
			j2k->cstr_info->tile[j2k->curtileno].end_header = cio_tell(cio) + j2k->pos_correction - 1;
		j2k->cstr_info->packno = 0;
	}
	
	len = int_min(j2k->eot - cio_getbp(cio), cio_numbytesleft(cio) + 1);

	if (len == cio_numbytesleft(cio) + 1) {
		truncate = 1;		/* Case of a truncate codestream */
	}	

	data = j2k->tile_data[curtileno];
	data = (unsigned char*) opj_realloc(data, (j2k->tile_len[curtileno] + len) * sizeof(unsigned char));

	data_ptr = data + j2k->tile_len[curtileno];
	for (i = 0; i < len; i++) {
		data_ptr[i] = cio_read(cio, 1);
	}

	j2k->tile_len[curtileno] += len;
	j2k->tile_data[curtileno] = data;
	
	if (!truncate) {
		j2k->state = J2K_STATE_TPHSOT;
	} else {
		j2k->state = J2K_STATE_NEOC;	/* RAJOUTE !! */
	}
	j2k->cur_tp_num++;
}

static void j2k_write_rgn(opj_j2k_t *j2k, int compno, int tileno) {
	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = &cp->tcps[tileno];
	opj_cio_t *cio = j2k->cio;
	int numcomps = j2k->image->numcomps;
	
	cio_write(cio, J2K_MS_RGN, 2);						/* RGN  */
	cio_write(cio, numcomps <= 256 ? 5 : 6, 2);			/* Lrgn */
	cio_write(cio, compno, numcomps <= 256 ? 1 : 2);	/* Crgn */
	cio_write(cio, 0, 1);								/* Srgn */
	cio_write(cio, tcp->tccps[compno].roishift, 1);		/* SPrgn */
}

static void j2k_read_rgn(opj_j2k_t *j2k) {
	int len, compno, roisty;

	opj_cp_t *cp = j2k->cp;
	opj_tcp_t *tcp = j2k->state == J2K_STATE_TPH ? &cp->tcps[j2k->curtileno] : j2k->default_tcp;
	opj_cio_t *cio = j2k->cio;
	int numcomps = j2k->image->numcomps;

	len = cio_read(cio, 2);										/* Lrgn */
	compno = cio_read(cio, numcomps <= 256 ? 1 : 2);			/* Crgn */
	roisty = cio_read(cio, 1);									/* Srgn */

#ifdef USE_JPWL
	if (j2k->cp->correct) {
		/* totlen is negative or larger than the bytes left!!! */
		if (compno >= numcomps) {
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"JPWL: bad component number in RGN (%d when there are only %d)\n",
				compno, numcomps);
			if (!JPWL_ASSUME || JPWL_ASSUME) {
				opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
				return;
			}
		}
	};
#endif /* USE_JPWL */

	tcp->tccps[compno].roishift = cio_read(cio, 1);				/* SPrgn */
}

static void j2k_write_eoc(opj_j2k_t *j2k) {
	opj_cio_t *cio = j2k->cio;
	/* opj_event_msg(j2k->cinfo, "%.8x: EOC\n", cio_tell(cio) + j2k->pos_correction); */
	cio_write(cio, J2K_MS_EOC, 2);

/* UniPG>> */
#ifdef USE_JPWL
	/* update markers struct */
	j2k_add_marker(j2k->cstr_info, J2K_MS_EOC, cio_tell(cio) - 2, 2);
#endif /* USE_JPWL */
/* <<UniPG */
}

static void j2k_read_eoc(opj_j2k_t *j2k) {
	int i, tileno;
	bool success;

	/* if packets should be decoded */
	if (j2k->cp->limit_decoding != DECODE_ALL_BUT_PACKETS) {
		opj_tcd_t *tcd = tcd_create(j2k->cinfo);
		tcd_malloc_decode(tcd, j2k->image, j2k->cp);
		for (i = 0; i < j2k->cp->tileno_size; i++) {
			tcd_malloc_decode_tile(tcd, j2k->image, j2k->cp, i, j2k->cstr_info);
			tileno = j2k->cp->tileno[i];
			success = tcd_decode_tile(tcd, j2k->tile_data[tileno], j2k->tile_len[tileno], tileno, j2k->cstr_info);
			opj_free(j2k->tile_data[tileno]);
			j2k->tile_data[tileno] = NULL;
			tcd_free_decode_tile(tcd, i);
			if (success == false) {
				j2k->state |= J2K_STATE_ERR;
				break;
			}
		}
		tcd_free_decode(tcd);
		tcd_destroy(tcd);
	}
	/* if packets should not be decoded  */
	else {
		for (i = 0; i < j2k->cp->tileno_size; i++) {
			tileno = j2k->cp->tileno[i];
			opj_free(j2k->tile_data[tileno]);
			j2k->tile_data[tileno] = NULL;
		}
	}	
	if (j2k->state & J2K_STATE_ERR)
		j2k->state = J2K_STATE_MT + J2K_STATE_ERR;
	else
		j2k->state = J2K_STATE_MT; 
}

typedef struct opj_dec_mstabent {
	/** marker value */
	int id;
	/** value of the state when the marker can appear */
	int states;
	/** action linked to the marker */
	void (*handler) (opj_j2k_t *j2k);
} opj_dec_mstabent_t;

opj_dec_mstabent_t j2k_dec_mstab[] = {
  {J2K_MS_SOC, J2K_STATE_MHSOC, j2k_read_soc},
  {J2K_MS_SOT, J2K_STATE_MH | J2K_STATE_TPHSOT, j2k_read_sot},
  {J2K_MS_SOD, J2K_STATE_TPH, j2k_read_sod},
  {J2K_MS_EOC, J2K_STATE_TPHSOT, j2k_read_eoc},
  {J2K_MS_SIZ, J2K_STATE_MHSIZ, j2k_read_siz},
  {J2K_MS_COD, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_cod},
  {J2K_MS_COC, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_coc},
  {J2K_MS_RGN, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_rgn},
  {J2K_MS_QCD, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_qcd},
  {J2K_MS_QCC, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_qcc},
  {J2K_MS_POC, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_poc},
  {J2K_MS_TLM, J2K_STATE_MH, j2k_read_tlm},
  {J2K_MS_PLM, J2K_STATE_MH, j2k_read_plm},
  {J2K_MS_PLT, J2K_STATE_TPH, j2k_read_plt},
  {J2K_MS_PPM, J2K_STATE_MH, j2k_read_ppm},
  {J2K_MS_PPT, J2K_STATE_TPH, j2k_read_ppt},
  {J2K_MS_SOP, 0, 0},
  {J2K_MS_CRG, J2K_STATE_MH, j2k_read_crg},
  {J2K_MS_COM, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_com},

#ifdef USE_JPWL
  {J2K_MS_EPC, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_epc},
  {J2K_MS_EPB, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_epb},
  {J2K_MS_ESD, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_esd},
  {J2K_MS_RED, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_red},
#endif /* USE_JPWL */
#ifdef USE_JPSEC
  {J2K_MS_SEC, J2K_STATE_MH, j2k_read_sec},
  {J2K_MS_INSEC, 0, j2k_read_insec},
#endif /* USE_JPSEC */

  {0, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_unk}
};

static void j2k_read_unk(opj_j2k_t *j2k) {
	opj_event_msg(j2k->cinfo, EVT_WARNING, "Unknown marker\n");

#ifdef USE_JPWL
	if (j2k->cp->correct) {
		int m = 0, id, i;
		int min_id = 0, min_dist = 17, cur_dist = 0, tmp_id;
		cio_seek(j2k->cio, cio_tell(j2k->cio) - 2);
		id = cio_read(j2k->cio, 2);
		opj_event_msg(j2k->cinfo, EVT_ERROR,
			"JPWL: really don't know this marker %x\n",
			id);
		if (!JPWL_ASSUME) {
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"- possible synch loss due to uncorrectable codestream errors => giving up\n");
			return;
		}
		/* OK, activate this at your own risk!!! */
		/* we look for the marker at the minimum hamming distance from this */
		while (j2k_dec_mstab[m].id) {
			
			/* 1's where they differ */
			tmp_id = j2k_dec_mstab[m].id ^ id;

			/* compute the hamming distance between our id and the current */
			cur_dist = 0;
			for (i = 0; i < 16; i++) {
				if ((tmp_id >> i) & 0x0001) {
					cur_dist++;
				}
			}

			/* if current distance is smaller, set the minimum */
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				min_id = j2k_dec_mstab[m].id;
			}
			
			/* jump to the next marker */
			m++;
		}

		/* do we substitute the marker? */
		if (min_dist < JPWL_MAXIMUM_HAMMING) {
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"- marker %x is at distance %d from the read %x\n",
				min_id, min_dist, id);
			opj_event_msg(j2k->cinfo, EVT_ERROR,
				"- trying to substitute in place and crossing fingers!\n");
			cio_seek(j2k->cio, cio_tell(j2k->cio) - 2);
			cio_write(j2k->cio, min_id, 2);

			/* rewind */
			cio_seek(j2k->cio, cio_tell(j2k->cio) - 2);

		}

	};
#endif /* USE_JPWL */

}

/**
Read the lookup table containing all the marker, status and action
@param id Marker value
*/
static opj_dec_mstabent_t *j2k_dec_mstab_lookup(int id) {
	opj_dec_mstabent_t *e;
	for (e = j2k_dec_mstab; e->id != 0; e++) {
		if (e->id == id) {
			break;
		}
	}
	return e;
}

/* ----------------------------------------------------------------------- */
/* J2K / JPT decoder interface                                             */
/* ----------------------------------------------------------------------- */

opj_j2k_t* j2k_create_decompress(opj_common_ptr cinfo) {
	opj_j2k_t *j2k = (opj_j2k_t*) opj_calloc(1, sizeof(opj_j2k_t));
	if(!j2k)
		return NULL;

	j2k->default_tcp = (opj_tcp_t*) opj_calloc(1, sizeof(opj_tcp_t));
	if(!j2k->default_tcp) {
		opj_free(j2k);
		return NULL;
	}

	j2k->cinfo = cinfo;
	j2k->tile_data = NULL;

	return j2k;
}

void j2k_destroy_decompress(opj_j2k_t *j2k) {
	int i = 0;

	if(j2k->tile_len != NULL) {
		opj_free(j2k->tile_len);
	}
	if(j2k->tile_data != NULL) {
		opj_free(j2k->tile_data);
	}
	if(j2k->default_tcp != NULL) {
		opj_tcp_t *default_tcp = j2k->default_tcp;
		if(default_tcp->ppt_data_first != NULL) {
			opj_free(default_tcp->ppt_data_first);
		}
		if(j2k->default_tcp->tccps != NULL) {
			opj_free(j2k->default_tcp->tccps);
		}
		opj_free(j2k->default_tcp);
	}
	if(j2k->cp != NULL) {
		opj_cp_t *cp = j2k->cp;
		if(cp->tcps != NULL) {
			for(i = 0; i < cp->tw * cp->th; i++) {
				if(cp->tcps[i].ppt_data_first != NULL) {
					opj_free(cp->tcps[i].ppt_data_first);
				}
				if(cp->tcps[i].tccps != NULL) {
					opj_free(cp->tcps[i].tccps);
				}
			}
			opj_free(cp->tcps);
		}
		if(cp->ppm_data_first != NULL) {
			opj_free(cp->ppm_data_first);
		}
		if(cp->tileno != NULL) {
			opj_free(cp->tileno);  
		}
		if(cp->comment != NULL) {
			opj_free(cp->comment);
		}

		opj_free(cp);
	}
	opj_free(j2k);
}

void j2k_setup_decoder(opj_j2k_t *j2k, opj_dparameters_t *parameters) {
	if(j2k && parameters) {
		/* create and initialize the coding parameters structure */
		opj_cp_t *cp = (opj_cp_t*) opj_calloc(1, sizeof(opj_cp_t));
		cp->reduce = parameters->cp_reduce;	
		cp->layer = parameters->cp_layer;
		cp->limit_decoding = parameters->cp_limit_decoding;

#ifdef USE_JPWL
		cp->correct = parameters->jpwl_correct;
		cp->exp_comps = parameters->jpwl_exp_comps;
		cp->max_tiles = parameters->jpwl_max_tiles;
#endif /* USE_JPWL */


		/* keep a link to cp so that we can destroy it later in j2k_destroy_decompress */
		j2k->cp = cp;
	}
}

opj_image_t* j2k_decode(opj_j2k_t *j2k, opj_cio_t *cio, opj_codestream_info_t *cstr_info) {
	opj_image_t *image = NULL;

	opj_common_ptr cinfo = j2k->cinfo;	

	j2k->cio = cio;
	j2k->cstr_info = cstr_info;
	if (cstr_info)
		memset(cstr_info, 0, sizeof(opj_codestream_info_t));

	/* create an empty image */
	image = opj_image_create0();
	j2k->image = image;

	j2k->state = J2K_STATE_MHSOC;

	for (;;) {
		opj_dec_mstabent_t *e;
		int id = cio_read(cio, 2);

#ifdef USE_JPWL
		/* we try to honor JPWL correction power */
		if (j2k->cp->correct) {

			int orig_pos = cio_tell(cio);
			bool status;

			/* call the corrector */
			status = jpwl_correct(j2k);

			/* go back to where you were */
			cio_seek(cio, orig_pos - 2);

			/* re-read the marker */
			id = cio_read(cio, 2);

			/* check whether it begins with ff */
			if (id >> 8 != 0xff) {
				opj_event_msg(cinfo, EVT_ERROR,
					"JPWL: possible bad marker %x at %d\n",
					id, cio_tell(cio) - 2);
				if (!JPWL_ASSUME) {
					opj_image_destroy(image);
					opj_event_msg(cinfo, EVT_ERROR, "JPWL: giving up\n");
					return 0;
				}
				/* we try to correct */
				id = id | 0xff00;
				cio_seek(cio, cio_tell(cio) - 2);
				cio_write(cio, id, 2);
				opj_event_msg(cinfo, EVT_WARNING, "- trying to adjust this\n"
					"- setting marker to %x\n",
					id);
			}

		}
#endif /* USE_JPWL */

		if (id >> 8 != 0xff) {
			opj_image_destroy(image);
			opj_event_msg(cinfo, EVT_ERROR, "%.8x: expected a marker instead of %x\n", cio_tell(cio) - 2, id);
			return 0;
		}
		e = j2k_dec_mstab_lookup(id);
		// Check if the marker is known
		if (!(j2k->state & e->states)) {
			opj_image_destroy(image);
			opj_event_msg(cinfo, EVT_ERROR, "%.8x: unexpected marker %x\n", cio_tell(cio) - 2, id);
			return 0;
		}
		// Check if the decoding is limited to the main header
		if (e->id == J2K_MS_SOT && j2k->cp->limit_decoding == LIMIT_TO_MAIN_HEADER) {
			opj_event_msg(cinfo, EVT_INFO, "Main Header decoded.\n");
			return image;
		}		

		if (e->handler) {
			(*e->handler)(j2k);
		}
		if (j2k->state & J2K_STATE_ERR) 
			return NULL;	

		if (j2k->state == J2K_STATE_MT) {
			break;
		}
		if (j2k->state == J2K_STATE_NEOC) {
			break;
		}
	}
	if (j2k->state == J2K_STATE_NEOC) {
		j2k_read_eoc(j2k);
	}

	if (j2k->state != J2K_STATE_MT) {
		opj_event_msg(cinfo, EVT_WARNING, "Incomplete bitstream\n");
	}

	return image;
}

/*
* Read a JPT-stream and decode file
*
*/
opj_image_t* j2k_decode_jpt_stream(opj_j2k_t *j2k, opj_cio_t *cio,  opj_codestream_info_t *cstr_info) {
	opj_image_t *image = NULL;
	opj_jpt_msg_header_t header;
	int position;

	opj_common_ptr cinfo = j2k->cinfo;
	
	j2k->cio = cio;

	/* create an empty image */
	image = opj_image_create0();
	j2k->image = image;

	j2k->state = J2K_STATE_MHSOC;
	
	/* Initialize the header */
	jpt_init_msg_header(&header);
	/* Read the first header of the message */
	jpt_read_msg_header(cinfo, cio, &header);
	
	position = cio_tell(cio);
	if (header.Class_Id != 6) {	/* 6 : Main header data-bin message */
		opj_image_destroy(image);
		opj_event_msg(cinfo, EVT_ERROR, "[JPT-stream] : Expecting Main header first [class_Id %d] !\n", header.Class_Id);
		return 0;
	}
	
	for (;;) {
		opj_dec_mstabent_t *e = NULL;
		int id;
		
		if (!cio_numbytesleft(cio)) {
			j2k_read_eoc(j2k);
			return image;
		}
		/* data-bin read -> need to read a new header */
		if ((unsigned int) (cio_tell(cio) - position) == header.Msg_length) {
			jpt_read_msg_header(cinfo, cio, &header);
			position = cio_tell(cio);
			if (header.Class_Id != 4) {	/* 4 : Tile data-bin message */
				opj_image_destroy(image);
				opj_event_msg(cinfo, EVT_ERROR, "[JPT-stream] : Expecting Tile info !\n");
				return 0;
			}
		}
		
		id = cio_read(cio, 2);
		if (id >> 8 != 0xff) {
			opj_image_destroy(image);
			opj_event_msg(cinfo, EVT_ERROR, "%.8x: expected a marker instead of %x\n", cio_tell(cio) - 2, id);
			return 0;
		}
		e = j2k_dec_mstab_lookup(id);
		if (!(j2k->state & e->states)) {
			opj_image_destroy(image);
			opj_event_msg(cinfo, EVT_ERROR, "%.8x: unexpected marker %x\n", cio_tell(cio) - 2, id);
			return 0;
		}
		if (e->handler) {
			(*e->handler)(j2k);
		}
		if (j2k->state == J2K_STATE_MT) {
			break;
		}
		if (j2k->state == J2K_STATE_NEOC) {
			break;
		}
	}
	if (j2k->state == J2K_STATE_NEOC) {
		j2k_read_eoc(j2k);
	}
	
	if (j2k->state != J2K_STATE_MT) {
		opj_event_msg(cinfo, EVT_WARNING, "Incomplete bitstream\n");
	}

	return image;
}

/* ----------------------------------------------------------------------- */
/* J2K encoder interface                                                       */
/* ----------------------------------------------------------------------- */

opj_j2k_t* j2k_create_compress(opj_common_ptr cinfo) {
	opj_j2k_t *j2k = (opj_j2k_t*) opj_calloc(1, sizeof(opj_j2k_t));
	if(j2k) {
		j2k->cinfo = cinfo;
	}
	return j2k;
}

void j2k_destroy_compress(opj_j2k_t *j2k) {
	int tileno;

	if(!j2k) return;
	if(j2k->cp != NULL) {
		opj_cp_t *cp = j2k->cp;

		if(cp->comment) {
			opj_free(cp->comment);
		}
		if(cp->matrice) {
			opj_free(cp->matrice);
		}
		for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
			opj_free(cp->tcps[tileno].tccps);
		}
		opj_free(cp->tcps);
		opj_free(cp);
	}

	opj_free(j2k);
}

void j2k_setup_encoder(opj_j2k_t *j2k, opj_cparameters_t *parameters, opj_image_t *image) {
	int i, j, tileno, numpocs_tile;
	opj_cp_t *cp = NULL;

	if(!j2k || !parameters || ! image) {
		return;
	}

	/* create and initialize the coding parameters structure */
	cp = (opj_cp_t*) opj_calloc(1, sizeof(opj_cp_t));

	/* keep a link to cp so that we can destroy it later in j2k_destroy_compress */
	j2k->cp = cp;

	/* set default values for cp */
	cp->tw = 1;
	cp->th = 1;

	/* 
	copy user encoding parameters 
	*/
	cp->cinema = parameters->cp_cinema;
	cp->max_comp_size =	parameters->max_comp_size;
	cp->rsiz   = parameters->cp_rsiz;
	cp->disto_alloc = parameters->cp_disto_alloc;
	cp->fixed_alloc = parameters->cp_fixed_alloc;
	cp->fixed_quality = parameters->cp_fixed_quality;

	/* mod fixed_quality */
	if(parameters->cp_matrice) {
		size_t array_size = parameters->tcp_numlayers * parameters->numresolution * 3 * sizeof(int);
		cp->matrice = (int *) opj_malloc(array_size);
		memcpy(cp->matrice, parameters->cp_matrice, array_size);
	}

	/* tiles */
	cp->tdx = parameters->cp_tdx;
	cp->tdy = parameters->cp_tdy;

	/* tile offset */
	cp->tx0 = parameters->cp_tx0;
	cp->ty0 = parameters->cp_ty0;

	/* comment string */
	if(parameters->cp_comment) {
		cp->comment = (char*)opj_malloc(strlen(parameters->cp_comment) + 1);
		if(cp->comment) {
			strcpy(cp->comment, parameters->cp_comment);
		}
	}

	/*
	calculate other encoding parameters
	*/

	if (parameters->tile_size_on) {
		cp->tw = int_ceildiv(image->x1 - cp->tx0, cp->tdx);
		cp->th = int_ceildiv(image->y1 - cp->ty0, cp->tdy);
	} else {
		cp->tdx = image->x1 - cp->tx0;
		cp->tdy = image->y1 - cp->ty0;
	}

	if(parameters->tp_on){
		cp->tp_flag = parameters->tp_flag;
		cp->tp_on = 1;
	}
	
	cp->img_size = 0;
	for(i=0;i<image->numcomps ;i++){
	cp->img_size += (image->comps[i].w *image->comps[i].h * image->comps[i].prec);
	}


#ifdef USE_JPWL
	/*
	calculate JPWL encoding parameters
	*/

	if (parameters->jpwl_epc_on) {
		int i;

		/* set JPWL on */
		cp->epc_on = true;
		cp->info_on = false; /* no informative technique */

		/* set EPB on */
		if ((parameters->jpwl_hprot_MH > 0) || (parameters->jpwl_hprot_TPH[0] > 0)) {
			cp->epb_on = true;
			
			cp->hprot_MH = parameters->jpwl_hprot_MH;
			for (i = 0; i < JPWL_MAX_NO_TILESPECS; i++) {
				cp->hprot_TPH_tileno[i] = parameters->jpwl_hprot_TPH_tileno[i];
				cp->hprot_TPH[i] = parameters->jpwl_hprot_TPH[i];
			}
			/* if tile specs are not specified, copy MH specs */
			if (cp->hprot_TPH[0] == -1) {
				cp->hprot_TPH_tileno[0] = 0;
				cp->hprot_TPH[0] = parameters->jpwl_hprot_MH;
			}
			for (i = 0; i < JPWL_MAX_NO_PACKSPECS; i++) {
				cp->pprot_tileno[i] = parameters->jpwl_pprot_tileno[i];
				cp->pprot_packno[i] = parameters->jpwl_pprot_packno[i];
				cp->pprot[i] = parameters->jpwl_pprot[i];
			}
		}

		/* set ESD writing */
		if ((parameters->jpwl_sens_size == 1) || (parameters->jpwl_sens_size == 2)) {
			cp->esd_on = true;

			cp->sens_size = parameters->jpwl_sens_size;
			cp->sens_addr = parameters->jpwl_sens_addr;
			cp->sens_range = parameters->jpwl_sens_range;

			cp->sens_MH = parameters->jpwl_sens_MH;
			for (i = 0; i < JPWL_MAX_NO_TILESPECS; i++) {
				cp->sens_TPH_tileno[i] = parameters->jpwl_sens_TPH_tileno[i];
				cp->sens_TPH[i] = parameters->jpwl_sens_TPH[i];
			}
		}

		/* always set RED writing to false: we are at the encoder */
		cp->red_on = false;

	} else {
		cp->epc_on = false;
	}
#endif /* USE_JPWL */


	/* initialize the mutiple tiles */
	/* ---------------------------- */
	cp->tcps = (opj_tcp_t*) opj_calloc(cp->tw * cp->th, sizeof(opj_tcp_t));

	for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
		opj_tcp_t *tcp = &cp->tcps[tileno];
		tcp->numlayers = parameters->tcp_numlayers;
		for (j = 0; j < tcp->numlayers; j++) {
			if(cp->cinema){
				if (cp->fixed_quality) {
					tcp->distoratio[j] = parameters->tcp_distoratio[j];
				}
				tcp->rates[j] = parameters->tcp_rates[j];
			}else{
				if (cp->fixed_quality) {	/* add fixed_quality */
					tcp->distoratio[j] = parameters->tcp_distoratio[j];
				} else {
					tcp->rates[j] = parameters->tcp_rates[j];
				}
			}
		}
		tcp->csty = parameters->csty;
		tcp->prg = parameters->prog_order;
		tcp->mct = parameters->tcp_mct; 

		numpocs_tile = 0;
		tcp->POC = 0;
		if (parameters->numpocs) {
			/* initialisation of POC */
			tcp->POC = 1;
			for (i = 0; i < parameters->numpocs; i++) {
				if((tileno == parameters->POC[i].tile - 1) || (parameters->POC[i].tile == -1)) {
					opj_poc_t *tcp_poc = &tcp->pocs[numpocs_tile];
					tcp_poc->resno0		= parameters->POC[numpocs_tile].resno0;
					tcp_poc->compno0	= parameters->POC[numpocs_tile].compno0;
					tcp_poc->layno1		= parameters->POC[numpocs_tile].layno1;
					tcp_poc->resno1		= parameters->POC[numpocs_tile].resno1;
					tcp_poc->compno1	= parameters->POC[numpocs_tile].compno1;
					tcp_poc->prg1		= parameters->POC[numpocs_tile].prg1;
					tcp_poc->tile		= parameters->POC[numpocs_tile].tile;
					numpocs_tile++;
				}
			}
			tcp->numpocs = numpocs_tile -1 ;
		}else{ 
			tcp->numpocs = 0;
		}

		tcp->tccps = (opj_tccp_t*) opj_calloc(image->numcomps, sizeof(opj_tccp_t));

		for (i = 0; i < image->numcomps; i++) {
			opj_tccp_t *tccp = &tcp->tccps[i];
			tccp->csty = parameters->csty & 0x01;	/* 0 => one precinct || 1 => custom precinct  */
			tccp->numresolutions = parameters->numresolution;
			tccp->cblkw = int_floorlog2(parameters->cblockw_init);
			tccp->cblkh = int_floorlog2(parameters->cblockh_init);
			tccp->cblksty = parameters->mode;
			tccp->qmfbid = parameters->irreversible ? 0 : 1;
			tccp->qntsty = parameters->irreversible ? J2K_CCP_QNTSTY_SEQNT : J2K_CCP_QNTSTY_NOQNT;
			tccp->numgbits = 2;
			if (i == parameters->roi_compno) {
				tccp->roishift = parameters->roi_shift;
			} else {
				tccp->roishift = 0;
			}

			if(parameters->cp_cinema)
			{
				//Precinct size for lowest frequency subband=128
				tccp->prcw[0] = 7;
				tccp->prch[0] = 7;
				//Precinct size at all other resolutions = 256
				for (j = 1; j < tccp->numresolutions; j++) {
					tccp->prcw[j] = 8;
					tccp->prch[j] = 8;
				}
			}else{
				if (parameters->csty & J2K_CCP_CSTY_PRT) {
					int p = 0;
					for (j = tccp->numresolutions - 1; j >= 0; j--) {
						if (p < parameters->res_spec) {
							
							if (parameters->prcw_init[p] < 1) {
								tccp->prcw[j] = 1;
							} else {
								tccp->prcw[j] = int_floorlog2(parameters->prcw_init[p]);
							}
							
							if (parameters->prch_init[p] < 1) {
								tccp->prch[j] = 1;
							}else {
								tccp->prch[j] = int_floorlog2(parameters->prch_init[p]);
							}

						} else {
							int res_spec = parameters->res_spec;
							int size_prcw = parameters->prcw_init[res_spec - 1] >> (p - (res_spec - 1));
							int size_prch = parameters->prch_init[res_spec - 1] >> (p - (res_spec - 1));
							
							if (size_prcw < 1) {
								tccp->prcw[j] = 1;
							} else {
								tccp->prcw[j] = int_floorlog2(size_prcw);
							}
							
							if (size_prch < 1) {
								tccp->prch[j] = 1;
							} else {
								tccp->prch[j] = int_floorlog2(size_prch);
							}
						}
						p++;
						/*printf("\nsize precinct for level %d : %d,%d\n", j,tccp->prcw[j], tccp->prch[j]); */
					}	//end for
				} else {
					for (j = 0; j < tccp->numresolutions; j++) {
						tccp->prcw[j] = 15;
						tccp->prch[j] = 15;
					}
				}
			}

			dwt_calc_explicit_stepsizes(tccp, image->comps[i].prec);
		}
	}
}

bool j2k_encode(opj_j2k_t *j2k, opj_cio_t *cio, opj_image_t *image, opj_codestream_info_t *cstr_info) {
	int tileno, compno;
	opj_cp_t *cp = NULL;

	opj_tcd_t *tcd = NULL;	/* TCD component */

	j2k->cio = cio;	
	j2k->image = image;

	cp = j2k->cp;

	/* INDEX >> */
	j2k->cstr_info = cstr_info;
	if (cstr_info) {
		int compno;
		cstr_info->tile = (opj_tile_info_t *) opj_malloc(cp->tw * cp->th * sizeof(opj_tile_info_t));
		cstr_info->image_w = image->x1 - image->x0;
		cstr_info->image_h = image->y1 - image->y0;
		cstr_info->prog = (&cp->tcps[0])->prg;
		cstr_info->tw = cp->tw;
		cstr_info->th = cp->th;
		cstr_info->tile_x = cp->tdx;	/* new version parser */
		cstr_info->tile_y = cp->tdy;	/* new version parser */
		cstr_info->tile_Ox = cp->tx0;	/* new version parser */
		cstr_info->tile_Oy = cp->ty0;	/* new version parser */
		cstr_info->numcomps = image->numcomps;
		cstr_info->numlayers = (&cp->tcps[0])->numlayers;
		cstr_info->numdecompos = (int*) opj_malloc(image->numcomps * sizeof(int));
		for (compno=0; compno < image->numcomps; compno++) {
			cstr_info->numdecompos[compno] = (&cp->tcps[0])->tccps->numresolutions - 1;
		}
		cstr_info->D_max = 0.0;		/* ADD Marcela */
		cstr_info->main_head_start = cio_tell(cio); /* position of SOC */
		cstr_info->maxmarknum = 100;
		cstr_info->marker = (opj_marker_info_t *) opj_malloc(cstr_info->maxmarknum * sizeof(opj_marker_info_t));
		cstr_info->marknum = 0;
	}
	/* << INDEX */

	j2k_write_soc(j2k);
	j2k_write_siz(j2k);
	j2k_write_cod(j2k);
	j2k_write_qcd(j2k);

	if(cp->cinema){
		for (compno = 1; compno < image->numcomps; compno++) {
			j2k_write_coc(j2k, compno);
			j2k_write_qcc(j2k, compno);
		}
	}

	for (compno = 0; compno < image->numcomps; compno++) {
		opj_tcp_t *tcp = &cp->tcps[0];
		if (tcp->tccps[compno].roishift)
			j2k_write_rgn(j2k, compno, 0);
	}
	if (cp->comment != NULL) {
		j2k_write_com(j2k);
	}

	j2k->totnum_tp = j2k_calculate_tp(cp,image->numcomps,image,j2k);
	/* TLM Marker*/
	if(cp->cinema){
		j2k_write_tlm(j2k);
		if (cp->cinema == CINEMA4K_24) {
			j2k_write_poc(j2k);
		}
	}

	/* uncomment only for testing JPSEC marker writing */
	/* j2k_write_sec(j2k); */

	/* INDEX >> */
	if(cstr_info) {
		cstr_info->main_head_end = cio_tell(cio) - 1;
	}
	/* << INDEX */
	/**** Main Header ENDS here ***/

	/* create the tile encoder */
	tcd = tcd_create(j2k->cinfo);

	/* encode each tile */
	for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
		int pino;
		int tilepartno=0;
		/* UniPG>> */
		int acc_pack_num = 0;
		/* <<UniPG */


		opj_tcp_t *tcp = &cp->tcps[tileno];
		opj_event_msg(j2k->cinfo, EVT_INFO, "tile number %d / %d\n", tileno + 1, cp->tw * cp->th);

		j2k->curtileno = tileno;
		j2k->cur_tp_num = 0;
		tcd->cur_totnum_tp = j2k->cur_totnum_tp[j2k->curtileno];
		/* initialisation before tile encoding  */
		if (tileno == 0) {
			tcd_malloc_encode(tcd, image, cp, j2k->curtileno);
		} else {
			tcd_init_encode(tcd, image, cp, j2k->curtileno);
		}

		/* INDEX >> */
		if(cstr_info) {
			cstr_info->tile[j2k->curtileno].start_pos = cio_tell(cio) + j2k->pos_correction;
		}
		/* << INDEX */

		for(pino = 0; pino <= tcp->numpocs; pino++) {
			int tot_num_tp;
			tcd->cur_pino=pino;

			/*Get number of tile parts*/
			tot_num_tp = j2k_get_num_tp(cp,pino,tileno);
			tcd->tp_pos = cp->tp_pos;

			for(tilepartno = 0; tilepartno < tot_num_tp ; tilepartno++){
				j2k->tp_num = tilepartno;
				/* INDEX >> */
				if(cstr_info)
					cstr_info->tile[j2k->curtileno].tp[j2k->cur_tp_num].tp_start_pos =
					cio_tell(cio) + j2k->pos_correction;
				/* << INDEX */
				j2k_write_sot(j2k);

				if(j2k->cur_tp_num == 0 && cp->cinema == 0){
					for (compno = 1; compno < image->numcomps; compno++) {
						j2k_write_coc(j2k, compno);
						j2k_write_qcc(j2k, compno);
					}
					if (cp->tcps[tileno].numpocs) {
						j2k_write_poc(j2k);
					}
				}

				/* INDEX >> */
				if(cstr_info)
					cstr_info->tile[j2k->curtileno].tp[j2k->cur_tp_num].tp_end_header =
					cio_tell(cio) + j2k->pos_correction + 1;
				/* << INDEX */

				j2k_write_sod(j2k, tcd);

				/* INDEX >> */
				if(cstr_info) {
					cstr_info->tile[j2k->curtileno].tp[j2k->cur_tp_num].tp_end_pos =
						cio_tell(cio) + j2k->pos_correction - 1;
					cstr_info->tile[j2k->curtileno].tp[j2k->cur_tp_num].tp_start_pack =
						acc_pack_num;
					cstr_info->tile[j2k->curtileno].tp[j2k->cur_tp_num].tp_numpacks =
						cstr_info->packno - acc_pack_num;
					acc_pack_num = cstr_info->packno;
				}
				/* << INDEX */

				j2k->cur_tp_num++;
			}			
		}
		if(cstr_info) {
			cstr_info->tile[j2k->curtileno].end_pos = cio_tell(cio) + j2k->pos_correction - 1;
		}


		/*
		if (tile->PPT) { // BAD PPT !!! 
		FILE *PPT_file;
		int i;
		PPT_file=fopen("PPT","rb");
		fprintf(stderr,"%c%c%c%c",255,97,tile->len_ppt/256,tile->len_ppt%256);
		for (i=0;i<tile->len_ppt;i++) {
		unsigned char elmt;
		fread(&elmt, 1, 1, PPT_file);
		fwrite(&elmt,1,1,f);
		}
		fclose(PPT_file);
		unlink("PPT");
		}
		*/

	}

	/* destroy the tile encoder */
	tcd_free_encode(tcd);
	tcd_destroy(tcd);

	opj_free(j2k->cur_totnum_tp);

	j2k_write_eoc(j2k);

	if(cstr_info) {
		cstr_info->codestream_size = cio_tell(cio) + j2k->pos_correction;
		/* UniPG>> */
		/* The following adjustment is done to adjust the codestream size */
		/* if SOD is not at 0 in the buffer. Useful in case of JP2, where */
		/* the first bunch of bytes is not in the codestream              */
		cstr_info->codestream_size -= cstr_info->main_head_start;
		/* <<UniPG */
	}

#ifdef USE_JPWL
	/*
	preparation of JPWL marker segments
	*/
	if(cp->epc_on) {

		/* encode according to JPWL */
		jpwl_encode(j2k, cio, image);

	}
#endif /* USE_JPWL */

	return true;
}






