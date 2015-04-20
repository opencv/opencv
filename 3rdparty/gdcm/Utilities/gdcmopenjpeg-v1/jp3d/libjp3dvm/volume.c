/*
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
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

opj_volume_t* OPJ_CALLCONV opj_volume_create(int numcmpts, opj_volume_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc) {
	int compno;
	opj_volume_t *volume = NULL;

	volume = (opj_volume_t*)opj_malloc(sizeof(opj_volume_t));
	if(volume) {
		volume->color_space = clrspc;
		volume->numcomps = numcmpts;
		/* allocate memory for the per-component information */
		volume->comps = (opj_volume_comp_t*)opj_malloc(volume->numcomps * sizeof(opj_volume_comp_t));
		if(!volume->comps) {
			opj_volume_destroy(volume);
			return NULL;
		}
		/* create the individual volume components */
		for(compno = 0; compno < numcmpts; compno++) {
			opj_volume_comp_t *comp = &volume->comps[compno];
			comp->dx = cmptparms[compno].dx;
			comp->dy = cmptparms[compno].dy;
			comp->dz = cmptparms[compno].dz;
			comp->w = cmptparms[compno].w;
			comp->h = cmptparms[compno].h;
			comp->l = cmptparms[compno].l;
			comp->x0 = cmptparms[compno].x0;
			comp->y0 = cmptparms[compno].y0;
			comp->z0 = cmptparms[compno].z0;
			comp->prec = cmptparms[compno].prec;
			comp->bpp = cmptparms[compno].bpp;
			comp->sgnd = cmptparms[compno].sgnd;
			comp->bigendian = cmptparms[compno].bigendian;
			comp->dcoffset = cmptparms[compno].dcoffset;
			comp->data = (int*)opj_malloc(comp->w * comp->h * comp->l * sizeof(int));
			if(!comp->data) {
				fprintf(stdout,"Unable to malloc comp->data (%d x %d x %d x bytes)",comp->w,comp->h,comp->l);
				opj_volume_destroy(volume);
				return NULL;
			}
			//fprintf(stdout,"%d %d %d %d %d %d %d %d %d", comp->w,comp->h, comp->l, comp->dx, comp->dy, comp->dz, comp->prec, comp->bpp, comp->sgnd);
		}
	}

	return volume;
}

void OPJ_CALLCONV opj_volume_destroy(opj_volume_t *volume) {
	int i;
	if(volume) {
		if(volume->comps) {
			/* volume components */
			for(i = 0; i < volume->numcomps; i++) {
				opj_volume_comp_t *volume_comp = &volume->comps[i];
				if(volume_comp->data) {
					opj_free(volume_comp->data);
				}
			}
			opj_free(volume->comps);
		}
		opj_free(volume);
	}
}

