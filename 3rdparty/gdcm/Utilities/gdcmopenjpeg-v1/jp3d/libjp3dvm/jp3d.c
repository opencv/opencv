/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2006, Mónica Díez García, Image Processing Laboratory, University of Valladolid, Spain
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

/** @defgroup J3D J3D - JPEG-2000 PART 10 codestream reader/writer */
/*@{*/

/** @name Funciones locales */
/*@{*/

/**
Write the SOC marker (Start Of Codestream)
@param j3d J3D handle
*/
static void j3d_write_soc(opj_j3d_t *j3d);
/**
Read the SOC marker (Start of Codestream)
@param j3d J3D handle
*/
static void j3d_read_soc(opj_j3d_t *j3d);
/**
Write the SIZ marker (2D volume and tile size)
@param j3d J3D handle
*/
static void j3d_write_siz(opj_j3d_t *j3d);
/**
Read the SIZ marker (2D volume and tile size)
@param j3d J3D handle
*/
static void j3d_read_siz(opj_j3d_t *j3d);
/**
Write the ZSI marker (3rd volume and tile size)
@param j3d J3D handle
*/
static void j3d_write_zsi(opj_j3d_t *j3d);
/**
Read the ZSI marker (3rd volume and tile size)
@param j3d J3D handle
*/
static void j3d_read_zsi(opj_j3d_t *j3d);
/**
Write the COM marker (comment)
@param j3d J3D handle
*/
static void j3d_write_com(opj_j3d_t *j3d);
/**
Read the COM marker (comment)
@param j3d J3D handle
*/
static void j3d_read_com(opj_j3d_t *j3d);
/**
Write the value concerning the specified component in the marker COD and COC
@param j3d J3D handle
@param compno Number of the component concerned by the information written
*/
static void j3d_write_cox(opj_j3d_t *j3d, int compno);
/**
Read the value concerning the specified component in the marker COD and COC
@param j3d J3D handle
@param compno Number of the component concerned by the information read
*/
static void j3d_read_cox(opj_j3d_t *j3d, int compno);
/**
Write the COD marker (coding style default)
@param j3d J3D handle
*/
static void j3d_write_cod(opj_j3d_t *j3d);
/**
Read the COD marker (coding style default)
@param j3d J3D handle
*/
static void j3d_read_cod(opj_j3d_t *j3d);
/**
Write the COC marker (coding style component)
@param j3d J3D handle
@param compno Number of the component concerned by the information written
*/
static void j3d_write_coc(opj_j3d_t *j3d, int compno);
/**
Read the COC marker (coding style component)
@param j3d J3D handle
*/
static void j3d_read_coc(opj_j3d_t *j3d);
/**
Write the value concerning the specified component in the marker QCD and QCC
@param j3d J3D handle
@param compno Number of the component concerned by the information written
*/
static void j3d_write_qcx(opj_j3d_t *j3d, int compno);
/**
Read the value concerning the specified component in the marker QCD and QCC
@param j3d J3D handle
@param compno Number of the component concern by the information read
@param len Length of the information in the QCX part of the marker QCD/QCC
*/
static void j3d_read_qcx(opj_j3d_t *j3d, int compno, int len);
/**
Write the QCD marker (quantization default)
@param j3d J3D handle
*/
static void j3d_write_qcd(opj_j3d_t *j3d);
/**
Read the QCD marker (quantization default)
@param j3d J3D handle
*/
static void j3d_read_qcd(opj_j3d_t *j3d);
/**
Write the QCC marker (quantization component)
@param j3d J3D handle
@param compno Number of the component concerned by the information written
*/
static void j3d_write_qcc(opj_j3d_t *j3d, int compno);
/**
Read the QCC marker (quantization component)
@param j3d J3D handle
*/
static void j3d_read_qcc(opj_j3d_t *j3d);
/**
Write the POC marker (progression order change)
@param j3d J3D handle
*/
static void j3d_write_poc(opj_j3d_t *j3d);
/**
Read the POC marker (progression order change)
@param j3d J3D handle
*/
static void j3d_read_poc(opj_j3d_t *j3d);
/**
Read the CRG marker (component registration)
@param j3d J3D handle
*/
static void j3d_read_crg(opj_j3d_t *j3d);
/**
Read the TLM marker (tile-part lengths)
@param j3d J3D handle
*/
static void j3d_read_tlm(opj_j3d_t *j3d);
/**
Read the PLM marker (packet length, main header)
@param j3d J3D handle
*/
static void j3d_read_plm(opj_j3d_t *j3d);
/**
Read the PLT marker (packet length, tile-part header)
@param j3d J3D handle
*/
static void j3d_read_plt(opj_j3d_t *j3d);
/**
Read the PPM marker (packet packet headers, main header)
@param j3d J3D handle
*/
static void j3d_read_ppm(opj_j3d_t *j3d);
/**
Read the PPT marker (packet packet headers, tile-part header)
@param j3d J3D handle
*/
static void j3d_read_ppt(opj_j3d_t *j3d);
/**
Write the SOT marker (start of tile-part)
@param j3d J3D handle
*/
static void j3d_write_sot(opj_j3d_t *j3d);
/**
Read the SOT marker (start of tile-part)
@param j3d J3D handle
*/
static void j3d_read_sot(opj_j3d_t *j3d);
/**
Write the SOD marker (start of data)
@param j3d J3D handle
@param tile_coder Pointer to a TCD handle
*/
static void j3d_write_sod(opj_j3d_t *j3d, void *tile_coder);
/**
Read the SOD marker (start of data)
@param j3d J3D handle
*/
static void j3d_read_sod(opj_j3d_t *j3d);
/**
Write the RGN marker (region-of-interest)
@param j3d J3D handle
@param compno Number of the component concerned by the information written
@param tileno Number of the tile concerned by the information written
*/
static void j3d_write_rgn(opj_j3d_t *j3d, int compno, int tileno);
/**
Read the RGN marker (region-of-interest)
@param j3d J3D handle
*/
static void j3d_read_rgn(opj_j3d_t *j3d);
/**
Write the EOC marker (end of codestream)
@param j3d J3D handle
*/
static void j3d_write_eoc(opj_j3d_t *j3d);
/**
Read the EOC marker (end of codestream)
@param j3d J3D handle
*/
static void j3d_read_eoc(opj_j3d_t *j3d);
/**
Read an unknown marker
@param j3d J3D handle
*/
static void j3d_read_unk(opj_j3d_t *j3d);
/**
Write the CAP marker (extended capabilities)
@param j3d J3D handle
*/
static void j3d_write_cap(opj_j3d_t *j3d);
/**
Read the CAP marker (extended capabilities)
@param j3d J3D handle
*/
static void j3d_read_cap(opj_j3d_t *j3d);
/**
Write the DCO marker (Variable DC offset)
@param j3d J3D handle
*/
static void j3d_write_dco(opj_j3d_t *j3d);
/**
Read the DCO marker (Variable DC offset)
@param j3d J3D handle
*/
static void j3d_read_dco(opj_j3d_t *j3d);
/**
Write the ATK marker (arbitrary transformation kernel)
@param j3d J3D handle
*/
static void j3d_write_atk(opj_j3d_t *j3d);
/**
Read the ATK marker (arbitrary transformation kernel)
@param j3d J3D handle
*/
static void j3d_read_atk(opj_j3d_t *j3d);
/**
Write the CBD marker (component bit depth definition)
@param j3d J3D handle
*/
static void j3d_write_cbd(opj_j3d_t *j3d);
/**
Read the CBD marker (component bit depth definition)
@param j3d J3D handle
*/
static void j3d_read_cbd(opj_j3d_t *j3d);
/**
Write the MCT marker (multiple component transfomation definition)
@param j3d J3D handle
*/
static void j3d_write_mct(opj_j3d_t *j3d);
/**
Read the MCT marker (multiple component transfomation definition)
@param j3d J3D handle
*/
static void j3d_read_mct(opj_j3d_t *j3d);
/**
Write the MCC marker (multiple component transfomation collection)
@param j3d J3D handle
*/
static void j3d_write_mcc(opj_j3d_t *j3d);
/**
Read the MCC marker (multiple component transfomation collection)
@param j3d J3D handle
*/
static void j3d_read_mcc(opj_j3d_t *j3d);
/**
Write the MCO marker (multiple component transfomation ordering)
@param j3d J3D handle
*/
static void j3d_write_mco(opj_j3d_t *j3d);
/**
Read the MCO marker (multiple component transfomation ordering)
@param j3d J3D handle
*/
static void j3d_read_mco(opj_j3d_t *j3d);
/**
Write the NLT marker (non-linearity point transformation)
@param j3d J3D handle
*/
static void j3d_write_nlt(opj_j3d_t *j3d);
/**
Read the NLT marker (non-linearity point transformation)
@param j3d J3D handle
*/
static void j3d_read_nlt(opj_j3d_t *j3d);
/*@}*/

/* ----------------------------------------------------------------------- */

void j3d_dump_volume(FILE *fd, opj_volume_t * vol) {
	int compno;
	fprintf(fd, "volume {\n");
	fprintf(fd, "  x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d\n", vol->x0, vol->y0, vol->z0,vol->x1, vol->y1,  vol->z1);
	fprintf(fd, "  numcomps=%d\n", vol->numcomps);
	for (compno = 0; compno < vol->numcomps; compno++) {
		opj_volume_comp_t *comp = &vol->comps[compno];
		fprintf(fd, "  comp %d {\n", compno);
		fprintf(fd, "    dx=%d, dy=%d, dz=%d\n", comp->dx, comp->dy, comp->dz);
		fprintf(fd, "    prec=%d\n", comp->prec);
		fprintf(fd, "    sgnd=%d\n", comp->sgnd);
		fprintf(fd, "  }\n");
	}
	fprintf(fd, "}\n");
}

void j3d_dump_cp(FILE *fd, opj_volume_t * vol, opj_cp_t * cp) {
	int tileno, compno, layno, bandno, resno, numbands;
	fprintf(fd, "coding parameters {\n");
	fprintf(fd, "  tx0=%d, ty0=%d, tz0=%d\n", cp->tx0, cp->ty0, cp->tz0);
	fprintf(fd, "  tdx=%d, tdy=%d, tdz=%d\n", cp->tdx, cp->tdy, cp->tdz);
	fprintf(fd, "  tw=%d, th=%d, tl=%d\n", cp->tw, cp->th, cp->tl);
	fprintf(fd, "  transform format: %d\n", cp->transform_format);
	fprintf(fd, "  encoding format: %d\n", cp->encoding_format);
	for (tileno = 0; tileno < cp->tw * cp->th * cp->tl; tileno++) {
		opj_tcp_t *tcp = &cp->tcps[tileno];
		fprintf(fd, "  tile %d {\n", tileno);
		fprintf(fd, "    csty=%x\n", tcp->csty);
		fprintf(fd, "    prg=%d\n", tcp->prg);
		fprintf(fd, "    numlayers=%d\n", tcp->numlayers);
		fprintf(fd, "    mct=%d\n", tcp->mct);
		fprintf(fd, "    rates=");
		for (layno = 0; layno < tcp->numlayers; layno++) {
			fprintf(fd, "%f ", tcp->rates[layno]);
		}
		fprintf(fd, "\n");
		fprintf(fd, "    first=%d\n", tcp->first);
		for (compno = 0; compno < vol->numcomps; compno++) {
			opj_tccp_t *tccp = &tcp->tccps[compno];
			fprintf(fd, "    comp %d {\n", compno);
			fprintf(fd, "      csty=%x\n", tccp->csty);
			fprintf(fd, "      numresx=%d, numresy=%d, numresz=%d\n", tccp->numresolution[0], tccp->numresolution[1], tccp->numresolution[2]);
			fprintf(fd, "      cblkw=%d, cblkh=%d, cblkl=%d\n", tccp->cblk[0], tccp->cblk[1], tccp->cblk[2]);
			fprintf(fd, "      cblksty=%x\n", tccp->cblksty);
			fprintf(fd, "      qntsty=%d\n", tccp->qntsty);
			fprintf(fd, "      numgbits=%d\n", tccp->numgbits);
			fprintf(fd, "      roishift=%d\n", tccp->roishift);
			fprintf(fd, "      reversible=%d\n", tccp->reversible);
			fprintf(fd, "      dwtidx=%d dwtidy=%d dwtidz=%d\n", tccp->dwtid[0], tccp->dwtid[1], tccp->dwtid[2]);
			if (tccp->atk != NULL) {
                fprintf(fd, "      atk.index=%d\n", tccp->atk->index);
				fprintf(fd, "      atk.coeff_typ=%d\n", tccp->atk->coeff_typ);
				fprintf(fd, "      atk.filt_cat=%d\n", tccp->atk->filt_cat);
				fprintf(fd, "      atk.exten=%d\n", tccp->atk->exten);
				fprintf(fd, "      atk.minit=%d\n", tccp->atk->minit);
				fprintf(fd, "      atk.wt_typ=%d\n", tccp->atk->wt_typ);
			}
			fprintf(fd, "      stepsizes of bands=");
            numbands = (tccp->qntsty == J3D_CCP_QNTSTY_SIQNT) ? 1 :
			( (cp->transform_format == TRF_2D_DWT) ? (tccp->numresolution[0] * 3 - 2) :
				(tccp->numresolution[0] * 7 - 6) - 4 *(tccp->numresolution[0] - tccp->numresolution[2]) );
			for (bandno = 0; bandno < numbands; bandno++) {
				fprintf(fd, "(%d,%d) ", tccp->stepsizes[bandno].mant,tccp->stepsizes[bandno].expn);
			}
			fprintf(fd, "\n");
			
			if (tccp->csty & J3D_CCP_CSTY_PRT) {
				fprintf(fd, "      prcw=");
				for (resno = 0; resno < tccp->numresolution[0]; resno++) {
					fprintf(fd, "%d ", tccp->prctsiz[0][resno]);
				}
				fprintf(fd, "\n");
				fprintf(fd, "      prch=");
				for (resno = 0; resno < tccp->numresolution[0]; resno++) {
					fprintf(fd, "%d ", tccp->prctsiz[1][resno]);
				}
				fprintf(fd, "\n");
				fprintf(fd, "      prcl=");
				for (resno = 0; resno < tccp->numresolution[0]; resno++) {
					fprintf(fd, "%d ", tccp->prctsiz[2][resno]);
				}
				fprintf(fd, "\n");
			}
			fprintf(fd, "    }\n");
		}
		fprintf(fd, "  }\n");
	}
	fprintf(fd, "}\n");
}

/* ----------------------------------------------------------------------- 
Extended capabilities
------------------------------------------------------------------------*/

static void j3d_write_cap(opj_j3d_t *j3d){
	int len,lenp;

	opj_cio_t *cio = j3d->cio;
	cio_write(cio, J3D_MS_CAP, 2);	/* CAP */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio,J3D_CAP_10, 4); 
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Lsiz */
	cio_seek(cio, lenp + len);

}
static void j3d_read_cap(opj_j3d_t *j3d){
	int len, Cap;
	opj_cio_t *cio = j3d->cio;
	/*cio_read(cio, 2);	 CAP */
	len = cio_read(cio, 2);
	Cap = cio_read(cio, 4);
}
static void j3d_write_zsi(opj_j3d_t *j3d) {
	int i;
	int lenp, len;

	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_cp_t *cp = j3d->cp;
	
	cio_write(cio, J3D_MS_ZSI, 2);	/* ZSI */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio, volume->z1, 4);	/* Zsiz */
	cio_write(cio, volume->z0, 4);	/* Z0siz */
	cio_write(cio, cp->tdz, 4);		/* ZTsiz */
	cio_write(cio, cp->tz0, 4);		/* ZT0siz */
	for (i = 0; i < volume->numcomps; i++) {
		cio_write(cio, volume->comps[i].dz, 1);	/* ZRsiz_i */
	}
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Lsiz */
	cio_seek(cio, lenp + len);
}

static void j3d_read_zsi(opj_j3d_t *j3d) {
	int len, i;
	
	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_cp_t *cp = j3d->cp;
	
	len = cio_read(cio, 2);			/* Lsiz */
	volume->z1 = cio_read(cio, 4);	/* Zsiz */
	volume->z0 = cio_read(cio, 4);	/* Z0siz */
	cp->tdz = cio_read(cio, 4);		/* ZTsiz */
	cp->tz0 = cio_read(cio, 4);		/* ZT0siz */
	for (i = 0; i < volume->numcomps; i++) {
		volume->comps[i].dz = cio_read(cio, 1);	/* ZRsiz_i */
	}
	
	//Initialization of volume
	cp->tw = int_ceildiv(volume->x1 - cp->tx0, cp->tdx);
	cp->th = int_ceildiv(volume->y1 - cp->ty0, cp->tdy);
	cp->tl = int_ceildiv(volume->z1 - cp->tz0, cp->tdz);
	cp->tcps = (opj_tcp_t *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(opj_tcp_t));
	cp->tileno = (int *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(int));
	cp->tileno_size = 0;
	
	for (i = 0; i < cp->tw * cp->th * cp->tl ; i++) {
		cp->tcps[i].POC = 0;
		cp->tcps[i].numpocs = 0;
		cp->tcps[i].first = 1;
	}
	
	/* Initialization for PPM marker (Packets header)*/
	cp->ppm = 0;
	cp->ppm_data = NULL;
	cp->ppm_data_first = NULL;
	cp->ppm_previous = 0;
	cp->ppm_store = 0;
	
	j3d->default_tcp->tccps = (opj_tccp_t *) opj_malloc(sizeof(opj_tccp_t) * volume->numcomps);
	for (i = 0; i < cp->tw * cp->th * cp->tl ; i++) {
		cp->tcps[i].tccps = (opj_tccp_t *) opj_malloc(sizeof(opj_tccp_t) * volume->numcomps);
	}
	j3d->tile_data = (unsigned char **) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(unsigned char *));
	j3d->tile_len = (int *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(int));
	j3d->state = J3D_STATE_MH;
	
}
static void j3d_write_dco(opj_j3d_t *j3d){
	int lenp, len, i;
	int dcotype;	

	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_cp_t *cp = j3d->cp;
	
	dcotype = 1; /* Offsets are 16bit signed integers Table A21 15444-2 */
	cio_write(cio, J3D_MS_DCO, 2);	/* DCO */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio, dcotype, 1);	
	if (dcotype == 0) {
		for (i = 0; i < volume->numcomps; i++) 
			cio_write(cio, volume->comps[i].dcoffset, 1);	/* SPdco_i */
	} else if (dcotype == 1) {
		for (i = 0; i < volume->numcomps; i++){ 
			cio_write(cio, volume->comps[i].dcoffset, 1);	/* SPdco_i */
			opj_event_msg(j3d->cinfo, EVT_INFO, "dcotype %d DCO %d \n",dcotype,volume->comps[i].dcoffset);
		}
	}
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Ldco */
	cio_seek(cio, lenp + len);

}
static void j3d_read_dco(opj_j3d_t *j3d){
	int len, i;
	int dcotype;

	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_cp_t *cp = j3d->cp;
	
	len = cio_read(cio, 2);			/* Lsiz */
	dcotype = cio_read(cio, 1); //offset 8bit unsigned / 16bit signed integers
	if (dcotype == 0) {
		for (i = 0; i < volume->numcomps; i++) {
			volume->comps[i].dcoffset = cio_read(cio, 1);
			if (volume->comps[i].dcoffset > 128) 
				volume->comps[i].dcoffset = volume->comps[i].dcoffset - 256;
		}
	} else if (dcotype == 1) {
		for (i = 0; i < volume->numcomps; i++) {
			volume->comps[i].dcoffset = cio_read(cio, 1);
			if (volume->comps[i].dcoffset > 128) 
				volume->comps[i].dcoffset = volume->comps[i].dcoffset - 256;
		}
	}
	
}
static void j3d_write_atk(opj_j3d_t *j3d){
	int lenp, len, s, k;

	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_atk_t *atk = j3d->cp->tcps->tccps->atk;
	
	cio_write(cio, J3D_MS_ATK, 2);	/* ATK */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);				
	cio_write(cio, atk->index + (atk->coeff_typ << 8) + (atk->filt_cat << 11) 
		+ (atk->wt_typ << 12) + (atk->minit << 13) + (atk->exten << 14), 2);			/* Satk */
    if (atk->wt_typ == J3D_ATK_IRR) 
		cio_write(cio,(unsigned int) (atk->Katk * 8192.0), 1 << atk->coeff_typ);
	cio_write(cio, atk->Natk, 1);
	for (s = 0; s < atk->Natk; s++){
		if (atk->filt_cat == J3D_ATK_ARB) 
			cio_write(cio, atk->Oatk[s], 1);
		if (atk->wt_typ == J3D_ATK_REV){
			cio_write(cio, atk->Eatk[s], 1);
			cio_write(cio, atk->Batk[s], 1);
		}
		cio_write(cio, atk->LCatk[s], 1);
		for (k = 0; k < atk->LCatk[s]; k++)
			cio_write(cio,(unsigned int) (atk->Aatk[s][k] * 8192.0), 1 << atk->coeff_typ);
	}
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Latk */
	cio_seek(cio, lenp + len);
}
static void j3d_read_atk(opj_j3d_t *j3d){
	int len, i, Satk, k;
	
	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_cp_t *cp = j3d->cp;
	opj_atk_t *atk = cp->tcps->tccps->atk; 
	
	len = cio_read(cio, 2);			/* Latk */
	Satk = cio_read(cio, 2); 
	atk->index = Satk & 0x00ff;
	atk->coeff_typ = Satk >> 8 & 0x0007;
	atk->filt_cat = Satk >> 11 & 0x0001;
	atk->wt_typ = Satk >> 12  & 0x0001;
	atk->minit = Satk >> 13 & 0x0001;
	atk->exten = Satk >> 14 & 0x0001;
    if (atk->wt_typ == J3D_ATK_IRR) 
		atk->Katk = ((double) cio_read(cio, 1 << atk->coeff_typ) / 8192.0);
	atk->Natk = cio_read(cio, 1);
	for (i = 0; i < atk->Natk; i++) {
		if (atk->filt_cat == J3D_ATK_ARB) 
			atk->Oatk[i] = cio_read(cio, 1);
		if (atk->wt_typ == J3D_ATK_REV){
			atk->Eatk[i] = cio_read(cio, 1);
			atk->Batk[i] = cio_read(cio, 1);
		}
		atk->LCatk[i] = cio_read(cio, 1);
		for (k = 0; k < atk->LCatk[i]; k++)
			atk->Aatk[i][k] = ((double) cio_read(cio, 1 << atk->coeff_typ) / 8192.0);
	}
}
static void j3d_write_cbd(opj_j3d_t *j3d){
}
static void j3d_read_cbd(opj_j3d_t *j3d){
}
static void j3d_write_mct(opj_j3d_t *j3d){
}
static void j3d_read_mct(opj_j3d_t *j3d){
}
static void j3d_write_mcc(opj_j3d_t *j3d){
}
static void j3d_read_mcc(opj_j3d_t *j3d){
}
static void j3d_write_mco(opj_j3d_t *j3d){
}
static void j3d_read_mco(opj_j3d_t *j3d){
}
static void j3d_write_nlt(opj_j3d_t *j3d){
}
static void j3d_read_nlt(opj_j3d_t *j3d){
}
/* ----------------------------------------------------------------------- 
15444-1 codestream syntax
------------------------------------------------------------------------*/
static void j3d_write_soc(opj_j3d_t *j3d) {
	opj_cio_t *cio = j3d->cio;
	cio_write(cio, J3D_MS_SOC, 2);
}

static void j3d_read_soc(opj_j3d_t *j3d) {
	j3d->state = J3D_STATE_MHSIZ;
}

static void j3d_write_siz(opj_j3d_t *j3d) {
	int i;
	int lenp, len;
	int Rsiz;

	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_cp_t *cp = j3d->cp;
	
	cio_write(cio, J3D_MS_SIZ, 2);	/* SIZ */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	//cio_write(cio, 0, 2);			/* Rsiz (capabilities of 15444-1 only) */
	Rsiz = J3D_RSIZ_DCO | J3D_RSIZ_ATK; /** | J3D_RSIZ_MCT | J3D_RSIZ_NONLT (not implemented yet)*/
	cio_write(cio, Rsiz, 2); /* capabilities of WDv5.2*/
	cio_write(cio, volume->x1, 4);	/* Xsiz */
	cio_write(cio, volume->y1, 4);	/* Ysiz */
	cio_write(cio, volume->x0, 4);	/* X0siz */
	cio_write(cio, volume->y0, 4);	/* Y0siz */
	cio_write(cio, cp->tdx, 4);		/* XTsiz */
	cio_write(cio, cp->tdy, 4);		/* YTsiz */
	cio_write(cio, cp->tx0, 4);		/* XT0siz */
	cio_write(cio, cp->ty0, 4);		/* YT0siz */
	cio_write(cio, volume->numcomps, 2);	/* Csiz */
	for (i = 0; i < volume->numcomps; i++) {
		cio_write(cio, volume->comps[i].prec - 1 + (volume->comps[i].sgnd << 7), 1);	/* Ssiz_i */
		cio_write(cio, volume->comps[i].dx, 1);	/* XRsiz_i */
		cio_write(cio, volume->comps[i].dy, 1);	/* YRsiz_i */
	}
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Lsiz */
	cio_seek(cio, lenp + len);
}

static void j3d_read_siz(opj_j3d_t *j3d) {
	int len, i;
	
	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	opj_cp_t *cp = j3d->cp;
	
	len = cio_read(cio, 2);			/* Lsiz */
	cp->rsiz = cio_read(cio, 2);	/* Rsiz (capabilities) */
	volume->x1 = cio_read(cio, 4);	/* Xsiz */
	volume->y1 = cio_read(cio, 4);	/* Ysiz */
	volume->x0 = cio_read(cio, 4);	/* X0siz */
	volume->y0 = cio_read(cio, 4);	/* Y0siz */
	cp->tdx = cio_read(cio, 4);		/* XTsiz */
	cp->tdy = cio_read(cio, 4);		/* YTsiz */
	cp->tx0 = cio_read(cio, 4);		/* XT0siz */
	cp->ty0 = cio_read(cio, 4);		/* YT0siz */
	
	volume->numcomps = cio_read(cio, 2);	/* Csiz */
	volume->comps = (opj_volume_comp_t *) opj_malloc(volume->numcomps * sizeof(opj_volume_comp_t));
	for (i = 0; i < volume->numcomps; i++) {
		int tmp, j;
		tmp = cio_read(cio, 1);		/* Ssiz_i */
		volume->comps[i].prec = (tmp & 0x7f) + 1;
		volume->comps[i].sgnd = tmp >> 7;
		volume->comps[i].dx = cio_read(cio, 1);	/* XRsiz_i */
		volume->comps[i].dy = cio_read(cio, 1);	/* YRsiz_i */
		for (j = 0; j < 3; j++) {
			volume->comps[i].resno_decoded[j] = 0;		/* number of resolution decoded */
			volume->comps[i].factor[j] = 0;		/* reducing factor per component */
		}
	}

	if (j3d->cinfo->codec_format == CODEC_J2K){
		volume->z1 = 1;
		volume->z0 = 0;
		volume->numslices = 1;
		cp->tdz = 1;
		cp->tz0 = 0;
		for (i = 0; i < volume->numcomps; i++) 
			volume->comps[i].dz = 1;

		//Initialization of volume
		cp->tw = int_ceildiv(volume->x1 - cp->tx0, cp->tdx);
		cp->th = int_ceildiv(volume->y1 - cp->ty0, cp->tdy);
		cp->tl = int_ceildiv(volume->z1 - cp->tz0, cp->tdz);
		cp->tcps = (opj_tcp_t *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(opj_tcp_t));
		cp->tileno = (int *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(int));
		cp->tileno_size = 0;
		
		for (i = 0; i < cp->tw * cp->th * cp->tl ; i++) {
			cp->tcps[i].POC = 0;
			cp->tcps[i].numpocs = 0;
			cp->tcps[i].first = 1;
		}
		
		/* Initialization for PPM marker (Packets header)*/
		cp->ppm = 0;
		cp->ppm_data = NULL;
		cp->ppm_data_first = NULL;
		cp->ppm_previous = 0;
		cp->ppm_store = 0;
		
		j3d->default_tcp->tccps = (opj_tccp_t *) opj_malloc(sizeof(opj_tccp_t) * volume->numcomps);
		for (i = 0; i < cp->tw * cp->th * cp->tl ; i++) {
			cp->tcps[i].tccps = (opj_tccp_t *) opj_malloc(sizeof(opj_tccp_t) * volume->numcomps);
		}
		j3d->tile_data = (unsigned char **) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(unsigned char *));
		j3d->tile_len = (int *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(int));
		j3d->state = J3D_STATE_MH;
	}
}



static void j3d_write_com(opj_j3d_t *j3d) {
	unsigned int i;
	int lenp, len;

	opj_cio_t *cio = j3d->cio;

	cio_write(cio, J3D_MS_COM, 2);
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	//cio_write(cio, 0, 2);
	cio_write(cio, j3d->cp->transform_format,1);
	cio_write(cio, j3d->cp->encoding_format,1);
	//opj_event_msg(j3d->cinfo, EVT_INFO, "TRF %D ENCOD %d\n",j3d->cp->transform_format,j3d->cp->encoding_format);
	if (j3d->cp->comment != NULL) {
		char *comment = j3d->cp->comment;
		for (i = 0; i < strlen(comment); i++) {
            cio_write(cio, comment[i], 1);
		}
	}
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);
	cio_seek(cio, lenp + len);
}

static void j3d_read_com(opj_j3d_t *j3d) {
	int len;
	opj_cio_t *cio = j3d->cio;

	len = cio_read(cio, 2);
	
	j3d->cp->transform_format = (OPJ_TRANSFORM) cio_read(cio, 1);
	j3d->cp->encoding_format = (OPJ_ENTROPY_CODING) cio_read(cio, 1);
	//opj_event_msg(j3d->cinfo, EVT_INFO, "TRF %D ENCOD %d\n",j3d->cp->transform_format,j3d->cp->encoding_format);

	cio_skip(cio, len - 4);  //posible comments
}

static void j3d_write_cox(opj_j3d_t *j3d, int compno) {
	int i;

	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = &cp->tcps[j3d->curtileno];
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j3d->cio;
	
	cio_write(cio, tccp->numresolution[0] - 1, 1);	/* SPcox (D) No of decomposition levels in x-axis*/
	if (j3d->cinfo->codec_format == CODEC_J3D) {
		cio_write(cio, tccp->numresolution[1] - 1, 1);	/* SPcox (E) No of decomposition levels in y-axis*/
		cio_write(cio, tccp->numresolution[2] - 1, 1);	/* SPcox (F) No of decomposition levels in z-axis*/
	}
	/* (cblkw - 2) + (cblkh - 2) + (cblkl - 2) <= 18*/
	cio_write(cio, tccp->cblk[0] - 2, 1);				/* SPcox (G) Cblk width entre 10 y 2 (8 y 0)*/
	cio_write(cio, tccp->cblk[1] - 2, 1);				/* SPcox (H) Cblk height*/
	if (j3d->cinfo->codec_format == CODEC_J3D) {
		cio_write(cio, tccp->cblk[2] - 2, 1);			/* SPcox (I) Cblk depth*/
	}
	cio_write(cio, tccp->cblksty, 1);				/* SPcox (J) Cblk style*/
	cio_write(cio, tccp->dwtid[0], 1);				/* SPcox (K) WT in x-axis 15444-2 Table A10*/
	if (j3d->cinfo->codec_format == CODEC_J3D) {
		cio_write(cio, tccp->dwtid[1], 1);				/* SPcox (L) WT in y-axis 15444-2 Table A10*/
		cio_write(cio, tccp->dwtid[2], 1);				/* SPcox (M) WT in z-axis 15444-2 Table A10*/
	}
	
	if (tccp->csty & J3D_CCP_CSTY_PRT) {
		for (i = 0; i < tccp->numresolution[0]; i++) {
			if (i < tccp->numresolution[2])
                cio_write(cio, tccp->prctsiz[0][i] + (tccp->prctsiz[1][i] << 4) + (tccp->prctsiz[2][i] << 8), 2);	/* SPcox (N_i) Table A9*/
			else
				if (j3d->cinfo->codec_format == CODEC_J3D) 
                    cio_write(cio, tccp->prctsiz[0][i] + (tccp->prctsiz[1][i] << 4), 2);	/* SPcox (N_i) Table A9*/
				else
                    cio_write(cio, tccp->prctsiz[0][i] + (tccp->prctsiz[1][i] << 4), 1);	/* SPcox (N_i) Table A9*/		}
	}
}

static void j3d_read_cox(opj_j3d_t *j3d, int compno) {
	int i;

	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = j3d->state == J3D_STATE_TPH ? &cp->tcps[j3d->curtileno] : j3d->default_tcp;
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j3d->cio;

	tccp->numresolution[0] = cio_read(cio, 1) + 1;	/* SPcox (D) No of decomposition levels in x-axis*/
	if (j3d->cinfo->codec_format == CODEC_J3D) {
		tccp->numresolution[1] = cio_read(cio, 1) + 1;	/* SPcox (E) No of decomposition levels in y-axis*/
		tccp->numresolution[2] = cio_read(cio, 1) + 1;	/* SPcox (F) No of decomposition levels in z-axis*/
	}else if (j3d->cinfo->codec_format == CODEC_J2K) {
		tccp->numresolution[1] = tccp->numresolution[0];	
		tccp->numresolution[2] = 1;							
	}
	/* check the reduce value */
	cp->reduce[0] = int_min((tccp->numresolution[0])-1, cp->reduce[0]);
	cp->reduce[1] = int_min((tccp->numresolution[1])-1, cp->reduce[1]);
	cp->reduce[2] = int_min((tccp->numresolution[2])-1, cp->reduce[2]);
	
	tccp->cblk[0] = cio_read(cio, 1) + 2;	/* SPcox (G) */
	tccp->cblk[1] = cio_read(cio, 1) + 2;	/* SPcox (H) */
	if (j3d->cinfo->codec_format == CODEC_J3D)
		tccp->cblk[2] = cio_read(cio, 1) + 2;	/* SPcox (I) */
	else
		tccp->cblk[2] = tccp->cblk[0];

	tccp->cblksty = cio_read(cio, 1);	/* SPcox (J) */
	tccp->dwtid[0] = cio_read(cio, 1);	/* SPcox (K) */
	if (j3d->cinfo->codec_format == CODEC_J3D) {
		tccp->dwtid[1] = cio_read(cio, 1);	/* SPcox (L) */
		tccp->dwtid[2] = cio_read(cio, 1);	/* SPcox (M) */
	}else{
		tccp->dwtid[1] = tccp->dwtid[0];	/* SPcox (L) */
		tccp->dwtid[2] = tccp->dwtid[0];	/* SPcox (M) */
	}
	tccp->reversible = (tccp->dwtid[0]>=1 && tccp->dwtid[1]>=1 && tccp->dwtid[2]>=1); //TODO: only valid for irreversible 9x7 WTs
	if (tccp->csty & J3D_CP_CSTY_PRT) {
		for (i = 0; i < tccp->numresolution[0]; i++) {
			int tmp = cio_read(cio, 2);	/* SPcox (N_i) */
			tccp->prctsiz[0][i] = tmp & 0xf;
			tccp->prctsiz[1][i] = tmp >> 4;
			tccp->prctsiz[2][i] = tmp >> 8;
		}
	}
}

static void j3d_write_cod(opj_j3d_t *j3d) {
	opj_cp_t *cp = NULL;
	opj_tcp_t *tcp = NULL;
	int lenp, len;

	opj_cio_t *cio = j3d->cio;
	
	cio_write(cio, J3D_MS_COD, 2);	/* COD */
	
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	
	cp = j3d->cp;
	tcp = &cp->tcps[j3d->curtileno];

	/* Scod : Table A-4*/
	cio_write(cio, tcp->csty, 1);		/* Scod : Coding style parameters */
	/* SGcod : Table A-5*/
	cio_write(cio, tcp->prg, 1);		/* SGcod (A) : Progression order */
	cio_write(cio, tcp->numlayers, 2);	/* SGcod (B) : No of layers */
	cio_write(cio, tcp->mct, 1);		/* SGcod (C) : Multiple component transformation usage */
	/* SPcod : Table A-6*/
	j3d_write_cox(j3d, 0);				
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);		/* Lcod */
	cio_seek(cio, lenp + len);
}

static void j3d_read_cod(opj_j3d_t *j3d) {
	int len, i, pos;
	
	opj_cio_t *cio = j3d->cio;
	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = j3d->state == J3D_STATE_TPH ? &cp->tcps[j3d->curtileno] : j3d->default_tcp;
	opj_volume_t *volume = j3d->volume;

	/* Lcod */
	len = cio_read(cio, 2);				
	/* Scod : Table A-4*/
	tcp->csty = cio_read(cio, 1);		
	/* SGcod : Table A-5*/
	tcp->prg = (OPJ_PROG_ORDER)cio_read(cio, 1);
	tcp->numlayers = cio_read(cio, 2);	
	tcp->mct = cio_read(cio, 1);		
	
	pos = cio_tell(cio);
	for (i = 0; i < volume->numcomps; i++) {
		tcp->tccps[i].csty = tcp->csty & J3D_CP_CSTY_PRT;
		cio_seek(cio, pos);
		j3d_read_cox(j3d, i);
	}
}

static void j3d_write_coc(opj_j3d_t *j3d, int compno) {
	int lenp, len;

	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = &cp->tcps[j3d->curtileno];
	opj_volume_t *volume = j3d->volume;
	opj_cio_t *cio = j3d->cio;
	
	cio_write(cio, J3D_MS_COC, 2);	/* COC */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio, compno, volume->numcomps <= 256 ? 1 : 2);	/* Ccoc */
	cio_write(cio, tcp->tccps[compno].csty, 1);					/* Scoc */
	
	j3d_write_cox(j3d, compno);
	
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);			/* Lcoc */
	cio_seek(cio, lenp + len);
}

static void j3d_read_coc(opj_j3d_t *j3d) {
	int len, compno;

	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = j3d->state == J3D_STATE_TPH ? &cp->tcps[j3d->curtileno] : j3d->default_tcp;
	opj_volume_t *volume = j3d->volume;
	opj_cio_t *cio = j3d->cio;
	
	len = cio_read(cio, 2);		/* Lcoc */
	compno = cio_read(cio, volume->numcomps <= 256 ? 1 : 2);	/* Ccoc */
	tcp->tccps[compno].csty = cio_read(cio, 1);	/* Scoc */
	j3d_read_cox(j3d, compno);
}

static void j3d_write_qcx(opj_j3d_t *j3d, int compno) {
	int bandno, numbands;
	int expn, mant;
	
	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = &cp->tcps[j3d->curtileno];
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j3d->cio;
	
	cio_write(cio, tccp->qntsty + (tccp->numgbits << 5), 1);	/* Sqcx : Table A28 de 15444-1*/
	
	if (j3d->cinfo->codec_format == CODEC_J2K)
        numbands = tccp->qntsty == J3D_CCP_QNTSTY_SIQNT ? 1 : tccp->numresolution[0] * 3 - 2; 
	else if (j3d->cinfo->codec_format == CODEC_J3D) {
		int diff = tccp->numresolution[0] - tccp->numresolution[2];
        numbands = (tccp->qntsty == J3D_CCP_QNTSTY_SIQNT) ? 1 : (tccp->numresolution[0] * 7 - 6) - 4 *diff; /* SIQNT vs. SEQNT */
	}
	
	for (bandno = 0; bandno < numbands; bandno++) {
		expn = tccp->stepsizes[bandno].expn;
		mant = tccp->stepsizes[bandno].mant;
		
		if (tccp->qntsty == J3D_CCP_QNTSTY_NOQNT) {
			cio_write(cio, expn << 3, 1);	/* SPqcx_i */
		} else {
			cio_write(cio, (expn << 11) + mant, 2);	/* SPqcx_i */
		}
	}
}

static void j3d_read_qcx(opj_j3d_t *j3d, int compno, int len) {
	int tmp;
	int bandno, numbands;

	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = j3d->state == J3D_STATE_TPH ? &cp->tcps[j3d->curtileno] : j3d->default_tcp;
	opj_tccp_t *tccp = &tcp->tccps[compno];
	opj_cio_t *cio = j3d->cio;

	tmp = cio_read(cio, 1);		/* Sqcx */
	tccp->qntsty = tmp & 0x1f;
	tccp->numgbits = tmp >> 5;

	/*Numbands = 1				si SIQNT
			     len - 1		si NOQNT
				 (len - 1) / 2	si SEQNT */
	numbands = tccp->qntsty == J3D_CCP_QNTSTY_SIQNT ? 1 : ((tccp->qntsty == J3D_CCP_QNTSTY_NOQNT) ? len - 1 : (len - 1) / 2);

	for (bandno = 0; bandno < numbands; bandno++) {
		int expn, mant;
		if (tccp->qntsty == J3D_CCP_QNTSTY_NOQNT) {
			expn = cio_read(cio, 1) >> 3;	/* SPqcx_i */
			mant = 0;
		} else {
			tmp = cio_read(cio, 2);			/* SPqcx_i */
			expn = tmp >> 11;
			mant = tmp & 0x7ff;
		}
		tccp->stepsizes[bandno].expn = expn;
		tccp->stepsizes[bandno].mant = mant;
	}
	
	/* Add Antonin : if scalar_derived -> compute other stepsizes */
	if (tccp->qntsty == J3D_CCP_QNTSTY_SIQNT) {
		for (bandno = 1; bandno < J3D_MAXBANDS; bandno++) {
			int numbands = (cp->transform_format==TRF_2D_DWT) ? 3 : 7;
			tccp->stepsizes[bandno].expn = tccp->stepsizes[0].expn - ((bandno - 1) / numbands);
			tccp->stepsizes[bandno].mant = tccp->stepsizes[0].mant;
		}
	}
	/* ddA */
}

static void j3d_write_qcd(opj_j3d_t *j3d) {
	int lenp, len;

	opj_cio_t *cio = j3d->cio;
	
	cio_write(cio, J3D_MS_QCD, 2);	/* QCD */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	j3d_write_qcx(j3d, 0);			/* Sqcd*/
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);			/* Lqcd */
	cio_seek(cio, lenp + len);
}

static void j3d_read_qcd(opj_j3d_t *j3d) {
	int len, i, pos;

	opj_cio_t *cio = j3d->cio;
	opj_volume_t *volume = j3d->volume;
	
	len = cio_read(cio, 2);		/* Lqcd */
	pos = cio_tell(cio);
	for (i = 0; i < volume->numcomps; i++) {
		cio_seek(cio, pos);
		j3d_read_qcx(j3d, i, len - 2);
	}
}

static void j3d_write_qcc(opj_j3d_t *j3d, int compno) {
	int lenp, len;

	opj_cio_t *cio = j3d->cio;
	
	cio_write(cio, J3D_MS_QCC, 2);	/* QCC */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);
	cio_write(cio, compno, j3d->volume->numcomps <= 256 ? 1 : 2);	/* Cqcc */
	j3d_write_qcx(j3d, compno);
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);			/* Lqcc */
	cio_seek(cio, lenp + len);
}

static void j3d_read_qcc(opj_j3d_t *j3d) {
	int len, compno;
	int numcomp = j3d->volume->numcomps;
	opj_cio_t *cio = j3d->cio;
	
	len = cio_read(cio, 2);	/* Lqcc */
	compno = cio_read(cio, numcomp <= 256 ? 1 : 2);	/* Cqcc */
	j3d_read_qcx(j3d, compno, len - 2 - (numcomp <= 256 ? 1 : 2));
}

static void j3d_write_poc(opj_j3d_t *j3d) {
	int len, numpchgs, i;

	int numcomps = j3d->volume->numcomps;
	
	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = &cp->tcps[j3d->curtileno];
	opj_tccp_t *tccp = &tcp->tccps[0];
	opj_cio_t *cio = j3d->cio;

	numpchgs = tcp->numpocs;
	cio_write(cio, J3D_MS_POC, 2);	/* POC  */
	len = 2 + (5 + 2 * (numcomps <= 256 ? 1 : 2)) * numpchgs;
	cio_write(cio, len, 2);		/* Lpoc */
	for (i = 0; i < numpchgs; i++) {
		opj_poc_t *poc = &tcp->pocs[i];
		cio_write(cio, poc->resno0, 1);	/* RSpoc_i */
		cio_write(cio, poc->compno0, (numcomps <= 256 ? 1 : 2));	/* CSpoc_i */
		cio_write(cio, poc->layno1, 2);	/* LYEpoc_i */
		poc->layno1 = int_min(poc->layno1, tcp->numlayers);
		cio_write(cio, poc->resno1, 1);	/* REpoc_i */
		poc->resno1 = int_min(poc->resno1, tccp->numresolution[0]);
		cio_write(cio, poc->compno1, (numcomps <= 256 ? 1 : 2));	/* CEpoc_i */
		poc->compno1 = int_min(poc->compno1, numcomps);
		cio_write(cio, poc->prg, 1);	/* Ppoc_i */
	}
}

static void j3d_read_poc(opj_j3d_t *j3d) {
	int len, numpchgs, i, old_poc;

	int numcomps = j3d->volume->numcomps;
	
	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = j3d->state == J3D_STATE_TPH ? &cp->tcps[j3d->curtileno] : j3d->default_tcp;
	opj_tccp_t *tccp = &tcp->tccps[0];
	opj_cio_t *cio = j3d->cio;
	
	old_poc = tcp->POC ? tcp->numpocs + 1 : 0;
	tcp->POC = 1;
	len = cio_read(cio, 2);		/* Lpoc */
	numpchgs = (len - 2) / (5 + 2 * (numcomps <= 256 ? 1 : 2));
	
	for (i = old_poc; i < numpchgs + old_poc; i++) {
		opj_poc_t *poc;
		poc = &tcp->pocs[i];
		poc->resno0 = cio_read(cio, 1);	/* RSpoc_i */
		poc->compno0 = cio_read(cio, numcomps <= 256 ? 1 : 2);	/* CSpoc_i */
		poc->layno1 = int_min(cio_read(cio, 2), (unsigned int) tcp->numlayers);	/* LYEpoc_i */
		poc->resno1 = int_min(cio_read(cio, 1), (unsigned int) tccp->numresolution[0]);	/* REpoc_i */
		poc->compno1 = int_min(
			cio_read(cio, numcomps <= 256 ? 1 : 2), (unsigned int) numcomps);	/* CEpoc_i */
		poc->prg = (OPJ_PROG_ORDER)cio_read(cio, 1);	/* Ppoc_i */
	}
	
	tcp->numpocs = numpchgs + old_poc - 1;
}

static void j3d_read_crg(opj_j3d_t *j3d) {
	int len, i, Xcrg_i, Ycrg_i, Zcrg_i;
	
	opj_cio_t *cio = j3d->cio;
	int numcomps = j3d->volume->numcomps;
	
	len = cio_read(cio, 2);			/* Lcrg */
	for (i = 0; i < numcomps; i++) {
		Xcrg_i = cio_read(cio, 2);	/* Xcrg_i */
		Ycrg_i = cio_read(cio, 2);	/* Ycrg_i */
		Zcrg_i = cio_read(cio, 2);	/* Zcrg_i */
	}
}

static void j3d_read_tlm(opj_j3d_t *j3d) {
	int len, Ztlm, Stlm, ST, SP, tile_tlm, i;
	long int Ttlm_i, Ptlm_i;

	opj_cio_t *cio = j3d->cio;
	
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

static void j3d_read_plm(opj_j3d_t *j3d) {
	int len, i, Zplm, Nplm, add, packet_len = 0;
	
	opj_cio_t *cio = j3d->cio;

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

static void j3d_read_plt(opj_j3d_t *j3d) {
	int len, i, Zplt, packet_len = 0, add;
	
	opj_cio_t *cio = j3d->cio;
	
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

static void j3d_read_ppm(opj_j3d_t *j3d) {
	int len, Z_ppm, i, j;
	int N_ppm;

	opj_cp_t *cp = j3d->cp;
	opj_cio_t *cio = j3d->cio;
	
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

static void j3d_read_ppt(opj_j3d_t *j3d) {
	int len, Z_ppt, i, j = 0;

	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = cp->tcps + j3d->curtileno;
	opj_cio_t *cio = j3d->cio;

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

static void j3d_write_sot(opj_j3d_t *j3d) {
	int lenp, len;

	opj_cio_t *cio = j3d->cio;

	j3d->sot_start = cio_tell(cio);
	cio_write(cio, J3D_MS_SOT, 2);		/* SOT */
	lenp = cio_tell(cio);
	cio_skip(cio, 2);					/* Lsot (further) */
	cio_write(cio, j3d->curtileno, 2);	/* Isot */
	cio_skip(cio, 4);					/* Psot (further in j3d_write_sod) */
	cio_write(cio, 0, 1);				/* TPsot */
	cio_write(cio, 1, 1);				/* TNsot (no of tile-parts of this tile in this codestream)*/
	len = cio_tell(cio) - lenp;
	cio_seek(cio, lenp);
	cio_write(cio, len, 2);				/* Lsot */
	cio_seek(cio, lenp + len);
}

static void j3d_read_sot(opj_j3d_t *j3d) {
	int len, tileno, totlen, partno, numparts, i;
	opj_tcp_t *tcp = NULL;
	char status = 0;

	opj_cp_t *cp = j3d->cp;
	opj_cio_t *cio = j3d->cio;
	
	len = cio_read(cio, 2);
	tileno = cio_read(cio, 2);
	
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
	if (!totlen)
		totlen = cio_numbytesleft(cio) + 8;
	
	partno = cio_read(cio, 1);
	numparts = cio_read(cio, 1);
	
	j3d->curtileno = tileno;
	j3d->eot = cio_getbp(cio) - 12 + totlen;
	j3d->state = J3D_STATE_TPH;
	tcp = &cp->tcps[j3d->curtileno];
	
	if (tcp->first == 1) {
		
		/* Initialization PPT */
		opj_tccp_t *tmp = tcp->tccps;
		memcpy(tcp, j3d->default_tcp, sizeof(opj_tcp_t));
		tcp->ppt = 0;
		tcp->ppt_data = NULL;
		tcp->ppt_data_first = NULL;
		tcp->tccps = tmp;

		for (i = 0; i < j3d->volume->numcomps; i++) {
			tcp->tccps[i] = j3d->default_tcp->tccps[i];
		}
		cp->tcps[j3d->curtileno].first = 0;
	}
}

static void j3d_write_sod(opj_j3d_t *j3d, void *tile_coder) {
	int l, layno;
	int totlen;
	opj_tcp_t *tcp = NULL;
	opj_volume_info_t *volume_info = NULL;
	
	opj_tcd_t *tcd = (opj_tcd_t*)tile_coder;	/* cast is needed because of conflicts in header inclusions */
	opj_cp_t *cp = j3d->cp;
	opj_cio_t *cio = j3d->cio;
	
	cio_write(cio, J3D_MS_SOD, 2);
	if (j3d->curtileno == 0) {
		j3d->sod_start = cio_tell(cio) + j3d->pos_correction;
	}
	
	/* INDEX >> */
	volume_info = j3d->volume_info;
	if (volume_info && volume_info->index_on) {
		volume_info->tile[j3d->curtileno].end_header = cio_tell(cio) + j3d->pos_correction - 1;
	}
	/* << INDEX */
	
	tcp = &cp->tcps[j3d->curtileno];
	for (layno = 0; layno < tcp->numlayers; layno++) {
		tcp->rates[layno] -= tcp->rates[layno] ? (j3d->sod_start / (cp->th * cp->tw * cp->tl)) : 0;
	}
	
	if(volume_info) {
		volume_info->num = 0;
	}

	l = tcd_encode_tile(tcd, j3d->curtileno, cio_getbp(cio), cio_numbytesleft(cio) - 2, volume_info);
	
	/* Writing Psot in SOT marker */
	totlen = cio_tell(cio) + l - j3d->sot_start;
	cio_seek(cio, j3d->sot_start + 6);
	cio_write(cio, totlen, 4);
	cio_seek(cio, j3d->sot_start + totlen);
}

static void j3d_read_sod(opj_j3d_t *j3d) {
	int len, truncate = 0, i;
	unsigned char *data = NULL, *data_ptr = NULL;

	opj_cio_t *cio = j3d->cio;
	int curtileno = j3d->curtileno;
	
	len = int_min(j3d->eot - cio_getbp(cio), cio_numbytesleft(cio) + 1);
	
	if (len == cio_numbytesleft(cio) + 1) {
		truncate = 1;		/* Case of a truncate codestream */
	}
	
	data = (unsigned char *) opj_malloc((j3d->tile_len[curtileno] + len) * sizeof(unsigned char));

	for (i = 0; i < j3d->tile_len[curtileno]; i++) {
		data[i] = j3d->tile_data[curtileno][i];
	}

	data_ptr = data + j3d->tile_len[curtileno];
	for (i = 0; i < len; i++) {
		data_ptr[i] = cio_read(cio, 1);
	}
	
	j3d->tile_len[curtileno] += len;
	opj_free(j3d->tile_data[curtileno]);
	j3d->tile_data[curtileno] = data;
	
	if (!truncate) {
		j3d->state = J3D_STATE_TPHSOT;
	} else {
		j3d->state = J3D_STATE_NEOC;	/* RAJOUTE !! */
	}
}

static void j3d_write_rgn(opj_j3d_t *j3d, int compno, int tileno) {
	
	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = &cp->tcps[tileno];
	opj_cio_t *cio = j3d->cio;
	int numcomps = j3d->volume->numcomps;
	
	cio_write(cio, J3D_MS_RGN, 2);						/* RGN  */
	cio_write(cio, numcomps <= 256 ? 5 : 6, 2);			/* Lrgn */
	cio_write(cio, compno, numcomps <= 256 ? 1 : 2);	/* Crgn */
	cio_write(cio, 0, 1);								/* Srgn */
	cio_write(cio, tcp->tccps[compno].roishift, 1);		/* SPrgn */
}

static void j3d_read_rgn(opj_j3d_t *j3d) {
	int len, compno, roisty;

	opj_cp_t *cp = j3d->cp;
	opj_tcp_t *tcp = j3d->state == J3D_STATE_TPH ? &cp->tcps[j3d->curtileno] : j3d->default_tcp;
	opj_cio_t *cio = j3d->cio;
	int numcomps = j3d->volume->numcomps;

	len = cio_read(cio, 2);										/* Lrgn */
	compno = cio_read(cio, numcomps <= 256 ? 1 : 2);			/* Crgn */
	roisty = cio_read(cio, 1);									/* Srgn */
	tcp->tccps[compno].roishift = cio_read(cio, 1);				/* SPrgn */
}

static void j3d_write_eoc(opj_j3d_t *j3d) {
	opj_cio_t *cio = j3d->cio;
	/* opj_event_msg(j3d->cinfo, "%.8x: EOC\n", cio_tell(cio) + j3d->pos_correction); */
	cio_write(cio, J3D_MS_EOC, 2);
}

static void j3d_read_eoc(opj_j3d_t *j3d) {
	int i, tileno;

#ifndef NO_PACKETS_DECODING  
	opj_tcd_t *tcd = tcd_create(j3d->cinfo);
    tcd_malloc_decode(tcd, j3d->volume, j3d->cp);
	/*j3d_dump_volume(stdout, tcd->volume);
	j3d_dump_cp(stdout, tcd->volume, tcd->cp);*/
	for (i = 0; i < j3d->cp->tileno_size; i++) {
		tileno = j3d->cp->tileno[i];
		//opj_event_msg(j3d->cinfo, EVT_INFO, "tcd_decode_tile \n");
		tcd_decode_tile(tcd, j3d->tile_data[tileno], j3d->tile_len[tileno], tileno);
		opj_free(j3d->tile_data[tileno]);
		j3d->tile_data[tileno] = NULL;
	}
	tcd_free_decode(tcd);
	tcd_destroy(tcd);
#else 
	for (i = 0; i < j3d->cp->tileno_size; i++) {
		tileno = j3d->cp->tileno[i];
		opj_free(j3d->tile_data[tileno]);
		j3d->tile_data[tileno] = NULL;
	}
#endif
	
	j3d->state = J3D_STATE_MT;
}

static void j3d_read_unk(opj_j3d_t *j3d) {
	opj_event_msg(j3d->cinfo, EVT_WARNING, "Unknown marker\n");
}

static opj_atk_t atk_info_wt[] = {
	{0, 1, J3D_ATK_WS, J3D_ATK_IRR, 0, J3D_ATK_WS, 1.230174104, 4, {0}, {0}, {0}, {1,1,1,1}, {-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971}},/* WT 9-7 IRR*/
	{1, 0, J3D_ATK_WS, J3D_ATK_REV, 0, J3D_ATK_WS, 0, 2, {0}, {1,2}, {1,2}, {1,1}, {-1.0,1.0}},/* WT 5-3 REV*/
	{2, 0, J3D_ATK_ARB, J3D_ATK_REV, 0, J3D_ATK_CON, 0, 2, {0,0}, {0,1}, {0,1}, {1,1}, {{-1.0},{1.0}}}, /* WT 2-2 REV*/
	{3, 0, J3D_ATK_ARB, J3D_ATK_REV, 1, J3D_ATK_CON, 0, 3, {0,0,-1}, {0,1,2}, {0,1,2}, {1,1,3}, {{-1.0},{1.0},{1.0,0.0,-1.0}}}, /* WT 2-6 REV*/
	{4, 0, J3D_ATK_ARB, J3D_ATK_REV, 1, J3D_ATK_CON, 0, 3, {0,0,-2}, {0,1,6}, {0,1,32}, {1,1,5}, {{-1},{1},{-3.0,22.0,0.0,-22.0,3.0}}}, /* WT 2-10 REV*/
	{5, 1, J3D_ATK_ARB, J3D_ATK_IRR, 1, J3D_ATK_WS, 1, 7, {0}, {0}, {0}, {1,1,2,1,2,1,3},{{-1},{1.58613434206},{-0.460348209828, 0.460348209828},{0.25},{0.374213867768,-0.374213867768},{-1.33613434206},{0.29306717103,0,-0.29306717103}}}, /* WT 6-10 IRR*/
	{6, 1, J3D_ATK_ARB, J3D_ATK_IRR, 0, J3D_ATK_WS, 1, 11, {0}, {0}, {0}, {1,1,2,1,2,1,2,1,2,1,5},{{-1},{0,99715069105},{-1.00573127827, 1.00573127827},{-0.27040357631},{2.20509972343, -2.20509972343},{0.08059995736},
		{-1.62682532350, 1.62682532350},{0.52040357631},{0.60404664250, -0.60404664250},{-0.82775064841},{-0.06615812964, 0.29402137720, 0, -0.29402137720, 0.06615812964}}}, /* WT 10-18 IRR*/
	{7, 1, J3D_ATK_WS, J3D_ATK_IRR, 0, J3D_ATK_WS, 1, 2, {0}, {0}, {0}, {1,1}, {-0.5, 0.25}},	/* WT 5-3 IRR*/
	{8, 0, J3D_ATK_WS, J3D_ATK_REV, 0, J3D_ATK_WS, 0, 2, {0}, {4,4}, {8,8}, {2,2}, {{-9,1},{5,-1}}}		/* WT 13-7 REV*/
};

typedef struct opj_dec_mstabent {
	/** marker value */
	int id;
	/** value of the state when the marker can appear */
	int states;
	/** action linked to the marker */
	void (*handler) (opj_j3d_t *j3d);
} opj_dec_mstabent_t;

opj_dec_mstabent_t j3d_dec_mstab[] = {
  {J3D_MS_SOC, J3D_STATE_MHSOC, j3d_read_soc},
  {J3D_MS_SOT, J3D_STATE_MH | J3D_STATE_TPHSOT, j3d_read_sot},
  {J3D_MS_SOD, J3D_STATE_TPH, j3d_read_sod},
  {J3D_MS_EOC, J3D_STATE_TPHSOT, j3d_read_eoc},
  {J3D_MS_CAP, J3D_STATE_MHSIZ, j3d_read_cap},
  {J3D_MS_SIZ, J3D_STATE_MHSIZ, j3d_read_siz},
  {J3D_MS_ZSI, J3D_STATE_MHSIZ, j3d_read_zsi},
  {J3D_MS_COD, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_cod},
  {J3D_MS_COC, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_coc},
  {J3D_MS_RGN, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_rgn},
  {J3D_MS_QCD, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_qcd},
  {J3D_MS_QCC, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_qcc},
  {J3D_MS_POC, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_poc},
  {J3D_MS_TLM, J3D_STATE_MH, j3d_read_tlm},
  {J3D_MS_PLM, J3D_STATE_MH, j3d_read_plm},
  {J3D_MS_PLT, J3D_STATE_TPH, j3d_read_plt},
  {J3D_MS_PPM, J3D_STATE_MH, j3d_read_ppm},
  {J3D_MS_PPT, J3D_STATE_TPH, j3d_read_ppt},
  {J3D_MS_SOP, 0, 0},
  {J3D_MS_CRG, J3D_STATE_MH, j3d_read_crg},
  {J3D_MS_COM, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_com},
  {J3D_MS_DCO, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_dco},
  {J3D_MS_ATK, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_atk},
  {0, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_unk}
  /*, -->must define the j3d_read functions
  {J3D_MS_CBD, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_cbd},
  {J3D_MS_MCT, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_mct},
  {J3D_MS_MCC, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_mcc},
  {J3D_MS_MCO, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_mco},
  {J3D_MS_NLT, J3D_STATE_MH | J3D_STATE_TPH, j3d_read_nlt},
  {J3D_MS_VMS, J3D_STATE_MH, j3d_read_vms},
  {J3D_MS_DFS, J3D_STATE_MH, j3d_read_dfs},
  {J3D_MS_ADS, J3D_STATE_MH, j3d_read_ads},
  {J3D_MS_QPD, J3D_STATE_MH, j3d_read_qpd},
  {J3D_MS_QPC, J3D_STATE_TPH, j3d_read_qpc}*/
};

/**
Read the lookup table containing all the marker, status and action
@param id Marker value
*/
static opj_dec_mstabent_t *j3d_dec_mstab_lookup(int id) {
	opj_dec_mstabent_t *e;
	for (e = j3d_dec_mstab; e->id != 0; e++) {
		if (e->id == id) {
			break;
		}
	}
	return e;
}

/* ----------------------------------------------------------------------- */
/* J3D / JPT decoder interface                                             */
/* ----------------------------------------------------------------------- */

opj_j3d_t* j3d_create_decompress(opj_common_ptr cinfo) {
	opj_j3d_t *j3d = (opj_j3d_t*)opj_malloc(sizeof(opj_j3d_t));
	if(j3d) {
		j3d->cinfo = cinfo;
		j3d->default_tcp = (opj_tcp_t*)opj_malloc(sizeof(opj_tcp_t));
		if(!j3d->default_tcp) {
			opj_free(j3d);
			return NULL;
		}
	}
	return j3d;
}

void j3d_destroy_decompress(opj_j3d_t *j3d) {
	int i = 0;

	if(j3d->tile_len != NULL) {
		opj_free(j3d->tile_len);
	}
	if(j3d->tile_data != NULL) {
		opj_free(j3d->tile_data);
	}
	if(j3d->default_tcp != NULL) {
		opj_tcp_t *default_tcp = j3d->default_tcp;
		if(default_tcp->ppt_data_first != NULL) {
			opj_free(default_tcp->ppt_data_first);
		}
		if(j3d->default_tcp->tccps != NULL) {
			opj_free(j3d->default_tcp->tccps);
		}
		opj_free(j3d->default_tcp);
	}
	if(j3d->cp != NULL) {
		opj_cp_t *cp = j3d->cp;
		if(cp->tcps != NULL) {
			for(i = 0; i < cp->tw * cp->th * cp->tl; i++) {
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

	opj_free(j3d);
}

void j3d_setup_decoder(opj_j3d_t *j3d, opj_dparameters_t *parameters) {
	if(j3d && parameters) {
		/* create and initialize the coding parameters structure */
		opj_cp_t *cp = (opj_cp_t*)opj_malloc(sizeof(opj_cp_t));
		cp->reduce[0] = parameters->cp_reduce[0];
		cp->reduce[1] = parameters->cp_reduce[1];
		cp->reduce[2] = parameters->cp_reduce[2];
		cp->layer = parameters->cp_layer;
		cp->bigendian = parameters->bigendian;
		
		
		cp->encoding_format = ENCOD_2EB;
		cp->transform_format = TRF_2D_DWT;
		
		/* keep a link to cp so that we can destroy it later in j3d_destroy_decompress */
		j3d->cp = cp;
	}
}

opj_volume_t* j3d_decode(opj_j3d_t *j3d, opj_cio_t *cio) {
	opj_volume_t *volume = NULL;

	opj_common_ptr cinfo = j3d->cinfo;

	j3d->cio = cio;

	/* create an empty volume */
	volume = (opj_volume_t*)opj_malloc(sizeof(opj_volume_t));
	j3d->volume = volume;

	j3d->state = J3D_STATE_MHSOC;
	
	for (;;) {
		opj_dec_mstabent_t *e;
		int id = cio_read(cio, 2);
		if (id >> 8 != 0xff) {
			opj_volume_destroy(volume);
			opj_event_msg(cinfo, EVT_ERROR, "%.8x: expected a marker instead of %x\n", cio_tell(cio) - 2, id);
			return 0;
		}
		e = j3d_dec_mstab_lookup(id);
		//opj_event_msg(cinfo, EVT_INFO, "MARKER %x PREVSTATE %d E->STATE %d\n",e->id,j3d->state,e->states);
		if (!(j3d->state & e->states)) {
			opj_volume_destroy(volume);
			opj_event_msg(cinfo, EVT_ERROR, "%.8x: unexpected marker %x\n", cio_tell(cio) - 2, id);
			return 0;
		}
		if (e->handler) {
			(*e->handler)(j3d);
		}
		//opj_event_msg(cinfo, EVT_INFO, "POSTSTATE %d\n",j3d->state);
		if (j3d->state == J3D_STATE_MT) {
			break;
		}
		if (j3d->state == J3D_STATE_NEOC) {
			break;
		}
	}
	if (j3d->state == J3D_STATE_NEOC) {
		j3d_read_eoc(j3d);
	}

	if (j3d->state != J3D_STATE_MT) {
		opj_event_msg(cinfo, EVT_WARNING, "Incomplete bitstream\n");
	}
	
	return volume;
}

/* ----------------------------------------------------------------------- */
/* J3D encoder interface                                                       */
/* ----------------------------------------------------------------------- */

opj_j3d_t* j3d_create_compress(opj_common_ptr cinfo) {
	opj_j3d_t *j3d = (opj_j3d_t*)opj_malloc(sizeof(opj_j3d_t));
	if(j3d) {
		j3d->cinfo = cinfo;
	}
	return j3d;
}

void j3d_destroy_compress(opj_j3d_t *j3d) {
	int tileno;

	if(!j3d) return;

	if(j3d->volume_info != NULL) {
		opj_volume_info_t *volume_info = j3d->volume_info;
		if (volume_info->index_on && j3d->cp) {
			opj_cp_t *cp = j3d->cp;
			for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
				opj_tile_info_t *tile_info = &volume_info->tile[tileno];
				opj_free(tile_info->thresh);
				opj_free(tile_info->packet);
			}
			opj_free(volume_info->tile);
		}
		opj_free(volume_info);
	}
	if(j3d->cp != NULL) {
		opj_cp_t *cp = j3d->cp;

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

	opj_free(j3d);
}

void j3d_setup_encoder(opj_j3d_t *j3d, opj_cparameters_t *parameters, opj_volume_t *volume) {
	int i, j, tileno, numpocs_tile;
	opj_cp_t *cp = NULL;

	if(!j3d || !parameters || ! volume) {
		return;
	}

	/* create and initialize the coding parameters structure */
	cp = (opj_cp_t*)opj_malloc(sizeof(opj_cp_t));

	/* keep a link to cp so that we can destroy it later in j3d_destroy_compress */
	j3d->cp = cp;

	/* set default values for cp */
	cp->tw = 1;
	cp->th = 1;
	cp->tl = 1;

	/* copy user encoding parameters */
	cp->disto_alloc = parameters->cp_disto_alloc;
	cp->fixed_alloc = parameters->cp_fixed_alloc;
	cp->fixed_quality = parameters->cp_fixed_quality;

	/* transform and coding method */
	cp->transform_format = parameters->transform_format;
	cp->encoding_format = parameters->encoding_format;

	/* mod fixed_quality */
	if(parameters->cp_matrice) {
		size_t array_size = parameters->tcp_numlayers * 3 * parameters->numresolution[0] * sizeof(int);
		cp->matrice = (int *) opj_malloc(array_size);
		memcpy(cp->matrice, parameters->cp_matrice, array_size);
	} 

	/* creation of an index file ? */
	cp->index_on = parameters->index_on;
	if(cp->index_on) {
		j3d->volume_info = (opj_volume_info_t*)opj_malloc(sizeof(opj_volume_info_t));
	}
	
	/* tiles */
	cp->tdx = parameters->cp_tdx;
	cp->tdy = parameters->cp_tdy;
	cp->tdz = parameters->cp_tdz;
	/* tile offset */
	cp->tx0 = parameters->cp_tx0;
	cp->ty0 = parameters->cp_ty0;
	cp->tz0 = parameters->cp_tz0;
	/* comment string */
	if(parameters->cp_comment) {
		cp->comment = (char*)opj_malloc(strlen(parameters->cp_comment) + 1);
		if(cp->comment) {
			strcpy(cp->comment, parameters->cp_comment);
		}
	}

	/*calculate other encoding parameters*/
	if (parameters->tile_size_on) {
		cp->tw = int_ceildiv(volume->x1 - cp->tx0, cp->tdx);
		cp->th = int_ceildiv(volume->y1 - cp->ty0, cp->tdy);
		cp->tl = int_ceildiv(volume->z1 - cp->tz0, cp->tdz);
	} else {
		cp->tdx = volume->x1 - cp->tx0;
		cp->tdy = volume->y1 - cp->ty0;
		cp->tdz = volume->z1 - cp->tz0;
	}

	/* initialize the multiple tiles */
	/* ---------------------------- */
	cp->tcps = (opj_tcp_t *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(opj_tcp_t));

	for (tileno = 0; tileno < cp->tw * cp->th * cp->tl; tileno++) {
		opj_tcp_t *tcp = &cp->tcps[tileno];
		tcp->numlayers = parameters->tcp_numlayers;
		for (j = 0; j < tcp->numlayers; j++) {
			if (cp->fixed_quality) {	/* add fixed_quality */
				tcp->distoratio[j] = parameters->tcp_distoratio[j];
			} else {
				tcp->rates[j] = parameters->tcp_rates[j];
			}
		}
		tcp->csty = parameters->csty;
		tcp->prg = parameters->prog_order;
		tcp->mct = volume->numcomps == 3 ? 1 : 0;

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
					tcp_poc->prg		= parameters->POC[numpocs_tile].prg;
					tcp_poc->tile		= parameters->POC[numpocs_tile].tile;
					numpocs_tile++;
				}
			}
		}
		tcp->numpocs = numpocs_tile;

		tcp->tccps = (opj_tccp_t *) opj_malloc(volume->numcomps * sizeof(opj_tccp_t));
		
		for (i = 0; i < volume->numcomps; i++) {
			opj_tccp_t *tccp = &tcp->tccps[i];
			tccp->csty = parameters->csty & J3D_CCP_CSTY_PRT;	/* 0 => standard precint || 1 => custom-defined precinct  */
			tccp->numresolution[0] = parameters->numresolution[0];
			tccp->numresolution[1] = parameters->numresolution[1];
			tccp->numresolution[2] = parameters->numresolution[2];
						assert (parameters->cblock_init[0] <= T1_MAXCBLKW);
						assert (parameters->cblock_init[0] >= T1_MINCBLKW);
						assert (parameters->cblock_init[1] <= T1_MAXCBLKH);
						assert (parameters->cblock_init[1] >= T1_MINCBLKH);
						assert (parameters->cblock_init[2] <= T1_MAXCBLKD);
						assert (parameters->cblock_init[2] >= T1_MINCBLKD);
			tccp->cblk[0] = int_floorlog2(parameters->cblock_init[0]); 
			tccp->cblk[1] = int_floorlog2(parameters->cblock_init[1]); 
			tccp->cblk[2] = int_floorlog2(parameters->cblock_init[2]); 
						assert (tccp->cblk[0]+tccp->cblk[1]+tccp->cblk[1] <= T1_MAXWHD);
			tccp->cblksty = parameters->mode; //Codeblock style --> Table A.19 (default 0)

			/*ATK / transform */
			tccp->reversible = parameters->irreversible ? 0 : 1; /* 0 => DWT 9-7 || 1 => DWT 5-3  */
			for (j = 0; j < 3; j++) {
					tccp->dwtid[j] = parameters->irreversible ? 0 : 1; /* 0 => DWT 9-7 || 1 => DWT 5-3  */
			}
      						
			/* Quantification: SEQNT (Scalar Expounded, value for each subband) / NOQNT (no quant)*/
			tccp->qntsty = parameters->irreversible ? J3D_CCP_QNTSTY_SEQNT : J3D_CCP_QNTSTY_NOQNT;
			tccp->numgbits = 2;
			if (i == parameters->roi_compno) {
				tccp->roishift = parameters->roi_shift;
			} else {
				tccp->roishift = 0;
			}
			/* Custom defined precints */
			if (parameters->csty & J3D_CCP_CSTY_PRT) {
				int k;
				for (k = 0; k < 3; k++) {
					int p = 0;
					for (j = tccp->numresolution[k] - 1; j >= 0; j--) {
						if (p < parameters->res_spec) {/* p < number of precinct size specifications */
							if (parameters->prct_init[k][p] < 1) {
								tccp->prctsiz[k][j] = 1;
							} else {
								tccp->prctsiz[k][j] = int_floorlog2(parameters->prct_init[k][p]);
							}
						} else {
							int res_spec = parameters->res_spec;
							int size_prct = parameters->prct_init[k][res_spec - 1] >> (p - (res_spec - 1));
							if (size_prct < 1) {
								tccp->prctsiz[k][j] = 1;
							} else {
								tccp->prctsiz[k][j] = int_floorlog2(size_prct);
							}
						}
					}
					p++;
				}
			} else {
				int k;
				for (k = 0; k < 3; k++) {
                    for (j = 0; j < tccp->numresolution[k]; j++) {
                        tccp->prctsiz[k][j] = 15;
					}
				}
			}
			//Calcular stepsize for each subband (if NOQNT -->stepsize = 1.0)
			dwt_calc_explicit_stepsizes(tccp, volume->comps[i].prec);
		}
	}
}

/**
Create an index file
@param j3d
@param cio
@param volume_info
@param index Index filename
@return Returns 1 if successful, returns 0 otherwise
*/
static int j3d_create_index(opj_j3d_t *j3d, opj_cio_t *cio, opj_volume_info_t *volume_info, char *index) {
	
	int tileno, compno, layno, resno, precno, pack_nb, x, y, z;
	FILE *stream = NULL;
	double total_disto = 0;

	volume_info->codestream_size = cio_tell(cio) + j3d->pos_correction;	/* Correction 14/4/03 suite rmq de Patrick */

	stream = fopen(index, "w");
	if (!stream) {
		opj_event_msg(j3d->cinfo, EVT_ERROR, "failed to open %s for writing\n", index);
		return 0;
	}
	
	fprintf(stream, "w %d\t h %d\t l %d\n", volume_info->volume_w, volume_info->volume_h, volume_info->volume_l);
	fprintf(stream, "TRASNFORM\t%d\n", volume_info->transform_format);
	fprintf(stream, "ENTROPY CODING\t%d\n", volume_info->encoding_format);
	fprintf(stream, "PROG\t%d\n", volume_info->prog);
	fprintf(stream, "TILE\tx %d y %d z %d\n", volume_info->tile_x, volume_info->tile_y, volume_info->tile_z);
	fprintf(stream, "NOTILE\tx %d y %d z %d\n", volume_info->tw, volume_info->th, volume_info->tl);
	fprintf(stream, "COMPONENTS\t%d\n", volume_info->comp);
	fprintf(stream, "LAYER\t%d\n", volume_info->layer);
	fprintf(stream, "RESOLUTIONS\tx %d y %d z %d\n", volume_info->decomposition[0], volume_info->decomposition[1], volume_info->decomposition[2]);
	
	fprintf(stream, "Precint sizes for each resolution:\n");
	for (resno = volume_info->decomposition[0]; resno >= 0; resno--) {
		fprintf(stream, "Resno %d \t [%d,%d,%d] \n", resno,
			(1 << volume_info->tile[0].prctsiz[0][resno]), (1 << volume_info->tile[0].prctsiz[0][resno]), (1 << volume_info->tile[0].prctsiz[2][resno]));	/* based on tile 0 */
	}
	fprintf(stream, "HEADER_END\t%d\n", volume_info->main_head_end);
	fprintf(stream, "CODESTREAM\t%d\n", volume_info->codestream_size);
	fprintf(stream, "Num_tile Start_pos End_header End_pos Distotile Nbpix Ratio\n");
	for (tileno = 0; tileno < (volume_info->tw * volume_info->th * volume_info->tl); tileno++) {
		fprintf(stream, "%4d\t%9d\t%9d\t%9d\t%9e\t%9d\t%9e\n",
			volume_info->tile[tileno].num_tile,
			volume_info->tile[tileno].start_pos,
			volume_info->tile[tileno].end_header,
			volume_info->tile[tileno].end_pos,
			volume_info->tile[tileno].distotile, volume_info->tile[tileno].nbpix,
			volume_info->tile[tileno].distotile / volume_info->tile[tileno].nbpix);
	}
	
	for (tileno = 0; tileno < (volume_info->tw * volume_info->th * volume_info->tl); tileno++) {
		int start_pos, end_pos;
		double disto = 0;
		pack_nb = 0;
		if (volume_info->prog == LRCP) {	/* LRCP */
			fprintf(stream, "pack_nb tileno layno resno compno precno start_pos  end_pos   disto\n");
			for (layno = 0; layno < volume_info->layer; layno++) {
				for (resno = 0; resno < volume_info->decomposition[0] + 1; resno++) {
					for (compno = 0; compno < volume_info->comp; compno++) {
						int prec_max = volume_info->tile[tileno].prctno[0][resno] * volume_info->tile[tileno].prctno[1][resno] * volume_info->tile[tileno].prctno[2][resno];
						for (precno = 0; precno < prec_max; precno++) {
							start_pos = volume_info->tile[tileno].packet[pack_nb].start_pos;
							end_pos = volume_info->tile[tileno].packet[pack_nb].end_pos;
							disto = volume_info->tile[tileno].packet[pack_nb].disto;
							fprintf(stream, "%4d %6d %7d %5d %6d %6d %9d %9d %8e\n",pack_nb, tileno, layno, resno, compno, precno, start_pos, end_pos, disto);
							total_disto += disto;
							pack_nb++;
						}
					}
				}
			}
		} /* LRCP */
		else if (volume_info->prog == RLCP) {	/* RLCP */
			/*
			fprintf(stream, "pack_nb tileno resno layno compno precno start_pos  end_pos   disto");
			*/
			for (resno = 0; resno < volume_info->decomposition[0] + 1; resno++) {
				for (layno = 0; layno < volume_info->layer; layno++) {
					for (compno = 0; compno < volume_info->comp; compno++) {
						int prec_max = volume_info->tile[tileno].prctno[0][resno] * volume_info->tile[tileno].prctno[1][resno]* volume_info->tile[tileno].prctno[2][resno];
						for (precno = 0; precno < prec_max; precno++) {
							start_pos = volume_info->tile[tileno].packet[pack_nb].start_pos;
							end_pos = volume_info->tile[tileno].packet[pack_nb].end_pos;
							disto = volume_info->tile[tileno].packet[pack_nb].disto;
							fprintf(stream, "%4d %6d %5d %7d %6d %6d %9d %9d %8e\n",
								pack_nb, tileno, resno, layno, compno, precno, start_pos, end_pos, disto);
							total_disto += disto;
							pack_nb++;
						}
					}
				}
			}
		} /* RLCP */
		else if (volume_info->prog == RPCL) {	/* RPCL */
			/*
			fprintf(stream, "\npack_nb tileno resno precno compno layno start_pos  end_pos   disto\n"); 
			*/
			for (resno = 0; resno < volume_info->decomposition[0] + 1; resno++) {
				/* I suppose components have same XRsiz, YRsiz */
				//int x0 = volume_info->tile_Ox + tileno - (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tw * volume_info->tile_x;
				//int y0 = volume_info->tile_Ox + (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tile_y;
				int x0 = volume_info->tile_Ox + (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tile_x;
				int y0 = volume_info->tile_Oy + (int)floor( (float)tileno/(float)volume_info->th ) * volume_info->tile_y;
				int z0 = volume_info->tile_Ox + (int)floor( (float)tileno/(float)volume_info->tl ) * volume_info->tile_z;
				int x1 = x0 + volume_info->tile_x;
				int y1 = y0 + volume_info->tile_y;
				int z1 = z0 + volume_info->tile_z;
				for(z = z0; z < z1; z++) {
					for(y = y0; y < y1; y++) {
						for(x = x0; x < x1; x++) {
							for (compno = 0; compno < volume_info->comp; compno++) {
								int prec_max = volume_info->tile[tileno].prctno[0][resno] * volume_info->tile[tileno].prctno[1][resno] * volume_info->tile[tileno].prctno[2][resno];
								for (precno = 0; precno < prec_max; precno++) {
									int pcnx = volume_info->tile[tileno].prctno[0][resno];
									int pcx = (int) pow( 2, volume_info->tile[tileno].prctsiz[0][resno] + volume_info->decomposition[0] - resno );
									int pcy = (int) pow( 2, volume_info->tile[tileno].prctsiz[1][resno] + volume_info->decomposition[1] - resno );
									int pcz = (int) pow( 2, volume_info->tile[tileno].prctsiz[2][resno] + volume_info->decomposition[2] - resno );
									int precno_x = precno - (int) floor( (float)precno/(float)pcnx ) * pcnx;
									int precno_y = (int) floor( (float)precno/(float)pcnx );
									if (precno_y*pcy == y ) {
										if (precno_x*pcx == x ) {
											for (layno = 0; layno < volume_info->layer; layno++) {
												start_pos = volume_info->tile[tileno].packet[pack_nb].start_pos;
												end_pos = volume_info->tile[tileno].packet[pack_nb].end_pos;
												disto = volume_info->tile[tileno].packet[pack_nb].disto;
												fprintf(stream, "%4d %6d %5d %6d %6d %7d %9d %9d %8e\n",
													pack_nb, tileno, resno, precno, compno, layno, start_pos, end_pos, disto); 
												total_disto += disto;
												pack_nb++; 
											}
										}
									}
								} /* precno */
							} /* compno */
						} /* x = x0..x1 */
					} /* y = y0..y1 */
				} /* z = z0..z1 */
			} /* resno */
		} /* RPCL */
		else if (volume_info->prog == PCRL) {	/* PCRL */
			/* I suppose components have same XRsiz, YRsiz */
			int x0 = volume_info->tile_Ox + tileno - (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tw * volume_info->tile_x;
			int y0 = volume_info->tile_Ox + (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tile_y;
			int z0 = volume_info->tile_Oz + (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tile_z;
			int x1 = x0 + volume_info->tile_x;
			int y1 = y0 + volume_info->tile_y;
			int z1 = z0 + volume_info->tile_z;
			/*
			fprintf(stream, "\npack_nb tileno precno compno resno layno start_pos  end_pos   disto\n"); 
			*/
			for(z = z0; z < z1; z++) {
				for(y = y0; y < y1; y++) {
					for(x = x0; x < x1; x++) {
						for (compno = 0; compno < volume_info->comp; compno++) {
							for (resno = 0; resno < volume_info->decomposition[0] + 1; resno++) {
								int prec_max = volume_info->tile[tileno].prctno[0][resno] * volume_info->tile[tileno].prctno[1][resno];
								for (precno = 0; precno < prec_max; precno++) {
								int pcnx = volume_info->tile[tileno].prctno[0][resno];
								int pcx = (int) pow( 2, volume_info->tile[tileno].prctsiz[0][resno] + volume_info->decomposition[0] - resno );
								int pcy = (int) pow( 2, volume_info->tile[tileno].prctsiz[1][resno] + volume_info->decomposition[1] - resno );
								int pcz = (int) pow( 2, volume_info->tile[tileno].prctsiz[2][resno] + volume_info->decomposition[2] - resno );
								int precno_x = precno - (int) floor( (float)precno/(float)pcnx ) * pcnx;
								int precno_y = (int) floor( (float)precno/(float)pcnx );
								int precno_z = (int) floor( (float)precno/(float)pcnx );
								if (precno_z*pcz == z ) {
									if (precno_y*pcy == y ) {
										if (precno_x*pcx == x ) {
											for (layno = 0; layno < volume_info->layer; layno++) {
												start_pos = volume_info->tile[tileno].packet[pack_nb].start_pos;
												end_pos = volume_info->tile[tileno].packet[pack_nb].end_pos;
												disto = volume_info->tile[tileno].packet[pack_nb].disto;
												fprintf(stream, "%4d %6d %6d %6d %5d %7d %9d %9d %8e\n",
													pack_nb, tileno, precno, compno, resno, layno, start_pos, end_pos, disto); 
												total_disto += disto;
												pack_nb++; 
											}
										}
									}
								}
							} /* precno */
						} /* resno */
					} /* compno */
				} /* x = x0..x1 */
			} /* y = y0..y1 */
			}
		} /* PCRL */
		else {	/* CPRL */
			/*
			fprintf(stream, "\npack_nb tileno compno precno resno layno start_pos  end_pos   disto\n"); 
			*/
			for (compno = 0; compno < volume_info->comp; compno++) {
				/* I suppose components have same XRsiz, YRsiz */
				int x0 = volume_info->tile_Ox + tileno - (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tw * volume_info->tile_x;
				int y0 = volume_info->tile_Ox + (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tile_y;
				int z0 = volume_info->tile_Oz + (int)floor( (float)tileno/(float)volume_info->tw ) * volume_info->tile_z;
				int x1 = x0 + volume_info->tile_x;
				int y1 = y0 + volume_info->tile_y;
				int z1 = z0 + volume_info->tile_z;
				for(z = z0; z < z1; z++) {
					for(y = y0; y < y1; y++) {
						for(x = x0; x < x1; x++) {
							for (resno = 0; resno < volume_info->decomposition[0] + 1; resno++) {
								int prec_max = volume_info->tile[tileno].prctno[0][resno] * volume_info->tile[tileno].prctno[1][resno] * volume_info->tile[tileno].prctno[2][resno];
								for (precno = 0; precno < prec_max; precno++) {
									int pcnx = volume_info->tile[tileno].prctno[0][resno];
									int pcny = volume_info->tile[tileno].prctno[1][resno];
									int pcx = (int) pow( 2, volume_info->tile[tileno].prctsiz[0][resno] + volume_info->decomposition[0] - resno );
									int pcy = (int) pow( 2, volume_info->tile[tileno].prctsiz[1][resno] + volume_info->decomposition[1] - resno );
									int pcz = (int) pow( 2, volume_info->tile[tileno].prctsiz[2][resno] + volume_info->decomposition[2] - resno );
									int precno_x = precno - (int) floor( (float)precno/(float)pcnx ) * pcnx;
									int precno_y = (int) floor( (float)precno/(float)pcnx );
									int precno_z = 0; /*???*/
									if (precno_z*pcz == z ) {
										if (precno_y*pcy == y ) {
											if (precno_x*pcx == x ) {
												for (layno = 0; layno < volume_info->layer; layno++) {
													start_pos = volume_info->tile[tileno].packet[pack_nb].start_pos;
													end_pos = volume_info->tile[tileno].packet[pack_nb].end_pos;
													disto = volume_info->tile[tileno].packet[pack_nb].disto;
													fprintf(stream, "%4d %6d %6d %6d %5d %7d %9d %9d %8e\n",
														pack_nb, tileno, compno, precno, resno, layno, start_pos, end_pos, disto); 
													total_disto += disto;
													pack_nb++; 
												}
											}
										}
									}
								} /* precno */
							} /* resno */
						} /* x = x0..x1 */
					} /* y = y0..y1 */
				} /* z = z0..z1 */
			} /* comno */
		} /* CPRL */   
	} /* tileno */
	
	fprintf(stream, "SE_MAX\t%8e\n", volume_info->D_max);	/* SE max */
	fprintf(stream, "SE_TOTAL\t%.8e\n", total_disto);			/* SE totale */
	

	fclose(stream);

	return 1;
}

bool j3d_encode(opj_j3d_t *j3d, opj_cio_t *cio, opj_volume_t *volume, char *index) {
	int tileno, compno;
	opj_volume_info_t *volume_info = NULL;
	opj_cp_t *cp = NULL;
	opj_tcd_t *tcd = NULL;	/* TCD component */

	j3d->cio = cio;	
	j3d->volume = volume;
	cp = j3d->cp;

	/*j3d_dump_volume(stdout, volume);
	j3d_dump_cp(stdout, volume, cp);*/

	/* INDEX >> */
	volume_info = j3d->volume_info;
	if (volume_info && cp->index_on) {
		volume_info->index_on = cp->index_on;
		volume_info->tile = (opj_tile_info_t *) opj_malloc(cp->tw * cp->th * cp->tl * sizeof(opj_tile_info_t));
		volume_info->volume_w = volume->x1 - volume->x0;
		volume_info->volume_h = volume->y1 - volume->y0;
		volume_info->volume_l = volume->z1 - volume->z0;
		volume_info->prog = (&cp->tcps[0])->prg;
		volume_info->tw = cp->tw;
		volume_info->th = cp->th;
		volume_info->tl = cp->tl;
		volume_info->tile_x = cp->tdx;	/* new version parser */
		volume_info->tile_y = cp->tdy;	/* new version parser */
		volume_info->tile_z = cp->tdz;	/* new version parser */
		volume_info->tile_Ox = cp->tx0;	/* new version parser */
		volume_info->tile_Oy = cp->ty0;	/* new version parser */
		volume_info->tile_Oz = cp->tz0;	/* new version parser */
		volume_info->transform_format = cp->transform_format;
		volume_info->encoding_format = cp->encoding_format;
		volume_info->comp = volume->numcomps;
		volume_info->layer = (&cp->tcps[0])->numlayers;
		volume_info->decomposition[0] = (&cp->tcps[0])->tccps->numresolution[0] - 1;
		volume_info->decomposition[1] = (&cp->tcps[0])->tccps->numresolution[1] - 1;
		volume_info->decomposition[2] = (&cp->tcps[0])->tccps->numresolution[2] - 1;
		volume_info->D_max = 0;		/* ADD Marcela */
	}
	/* << INDEX */

	j3d_write_soc(j3d);
	j3d_write_siz(j3d);
	if (j3d->cinfo->codec_format == CODEC_J3D) {
		j3d_write_cap(j3d);
		j3d_write_zsi(j3d);
	}
	j3d_write_cod(j3d);
	j3d_write_qcd(j3d);
	for (compno = 0; compno < volume->numcomps; compno++) {
		opj_tcp_t *tcp = &cp->tcps[0];
		if (tcp->tccps[compno].roishift)
			j3d_write_rgn(j3d, compno, 0);			
	}
	/*Optional 15444-2 markers*/
	if (j3d->cp->tcps->tccps[0].atk != NULL)
        j3d_write_atk(j3d);
	if (j3d->volume->comps[0].dcoffset != 0)
        j3d_write_dco(j3d);

	if (j3d->cp->transform_format != TRF_2D_DWT || j3d->cp->encoding_format != ENCOD_2EB)
		j3d_write_com(j3d);
	
	/* INDEX >> */
	if(volume_info && volume_info->index_on) {
		volume_info->main_head_end = cio_tell(cio) - 1;
	}
	/* << INDEX */

	/* create the tile encoder */
	tcd = tcd_create(j3d->cinfo);

	/* encode each tile */
	for (tileno = 0; tileno < cp->tw * cp->th * cp->tl; tileno++) {
		opj_event_msg(j3d->cinfo, EVT_INFO, "tile number %d / %d\n", tileno + 1, cp->tw * cp->th * cp->tl);
		
		j3d->curtileno = tileno;

		/* initialisation before tile encoding  */
		if (tileno == 0) {	
			tcd_malloc_encode(tcd, volume, cp, j3d->curtileno);
		} else {
			tcd_init_encode(tcd, volume, cp, j3d->curtileno);
		}
		
		/* INDEX >> */
		if(volume_info && volume_info->index_on) {
			volume_info->tile[j3d->curtileno].num_tile = j3d->curtileno;
			volume_info->tile[j3d->curtileno].start_pos = cio_tell(cio) + j3d->pos_correction;
		}
		/* << INDEX */
		
		j3d_write_sot(j3d);
	
		for (compno = 1; compno < volume->numcomps; compno++) {
			j3d_write_coc(j3d, compno);
			j3d_write_qcc(j3d, compno);
		}

		if (cp->tcps[tileno].numpocs) {
			j3d_write_poc(j3d);
		}
		j3d_write_sod(j3d, tcd); //--> tcd_encode_tile

		/* INDEX >> */
		if(volume_info && volume_info->index_on) {
			volume_info->tile[j3d->curtileno].end_pos = cio_tell(cio) + j3d->pos_correction - 1;
		}
		/* << INDEX */		
	}
	
	/* destroy the tile encoder */
	tcd_free_encode(tcd);
	tcd_destroy(tcd);

	j3d_write_eoc(j3d);
	
	/* Creation of the index file */
	if(volume_info && volume_info->index_on) {
		if(!j3d_create_index(j3d, cio, volume_info, index)) {
			opj_event_msg(j3d->cinfo, EVT_ERROR, "failed to create index file %s\n", index);
			return false;
		}
	}
	  
	return true;
}

