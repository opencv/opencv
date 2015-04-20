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

opj_image_t* opj_image_create0(void) {
	opj_image_t *image = (opj_image_t*)opj_calloc(1, sizeof(opj_image_t));
	return image;
}

opj_image_t* OPJ_CALLCONV opj_image_create(int numcmpts, opj_image_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc) {
	int compno;
	opj_image_t *image = NULL;

	image = (opj_image_t*) opj_calloc(1, sizeof(opj_image_t));
	if(image) {
		image->color_space = clrspc;
		image->numcomps = numcmpts;
		/* allocate memory for the per-component information */
		image->comps = (opj_image_comp_t*)opj_malloc(image->numcomps * sizeof(opj_image_comp_t));
		if(!image->comps) {
			fprintf(stderr,"Unable to allocate memory for image.\n");
			opj_image_destroy(image);
			return NULL;
		}
		/* create the individual image components */
		for(compno = 0; compno < numcmpts; compno++) {
			opj_image_comp_t *comp = &image->comps[compno];
			comp->dx = cmptparms[compno].dx;
			comp->dy = cmptparms[compno].dy;
			comp->w = cmptparms[compno].w;
			comp->h = cmptparms[compno].h;
			comp->x0 = cmptparms[compno].x0;
			comp->y0 = cmptparms[compno].y0;
			comp->prec = cmptparms[compno].prec;
			comp->bpp = cmptparms[compno].bpp;
			comp->sgnd = cmptparms[compno].sgnd;
			comp->data = (int*) opj_calloc(comp->w * comp->h, sizeof(int));
			if(!comp->data) {
				fprintf(stderr,"Unable to allocate memory for image.\n");
				opj_image_destroy(image);
				return NULL;
			}
		}
	}

	return image;
}

void OPJ_CALLCONV opj_image_destroy(opj_image_t *image) {
	int i;
	if(image) {
		if(image->comps) {
			/* image components */
			for(i = 0; i < image->numcomps; i++) {
				opj_image_comp_t *image_comp = &image->comps[i];
				if(image_comp->data) {
					opj_free(image_comp->data);
				}
			}
			opj_free(image->comps);
		}
		opj_free(image);
	}
}

