/*
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
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

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "opj_includes.h"
#define JP3D_VERSION "1.3.0"
/* ---------------------------------------------------------------------- */
#ifdef _WIN32
#ifndef OPJ_STATIC
BOOL APIENTRY
DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
	switch (ul_reason_for_call) {
		case DLL_PROCESS_ATTACH :
			break;
		case DLL_PROCESS_DETACH :
			break;
		case DLL_THREAD_ATTACH :
		case DLL_THREAD_DETACH :
			break;
    }

    return TRUE;
}
#endif /* OPJ_STATIC */
#endif /* _WIN32 */

/* ---------------------------------------------------------------------- */

const char* OPJ_CALLCONV opj_version() {
    return JP3D_VERSION;
}
opj_dinfo_t* OPJ_CALLCONV opj_create_decompress(OPJ_CODEC_FORMAT format) {
	opj_dinfo_t *dinfo = (opj_dinfo_t*)opj_malloc(sizeof(opj_dinfo_t));
	if(!dinfo) return NULL;
	dinfo->is_decompressor = true;
	switch(format) {
		case CODEC_J3D:
		case CODEC_J2K:
			/* get a J3D decoder handle */
			dinfo->j3d_handle = (void*)j3d_create_decompress((opj_common_ptr)dinfo);
			if(!dinfo->j3d_handle) {
				opj_free(dinfo);
				return NULL;
			}
			break;
		default:
			opj_free(dinfo);
			return NULL;
	}

	dinfo->codec_format = format;

	return dinfo;
}

void OPJ_CALLCONV opj_destroy_decompress(opj_dinfo_t *dinfo) {
	if(dinfo) {
		/* destroy the codec */
		if(dinfo->codec_format != CODEC_UNKNOWN) {
			j3d_destroy_decompress((opj_j3d_t*)dinfo->j3d_handle);
		}
		/* destroy the decompressor */
		opj_free(dinfo);
	}
}

void OPJ_CALLCONV opj_set_default_decoder_parameters(opj_dparameters_t *parameters) {
	if(parameters) {
		memset(parameters, 0, sizeof(opj_dparameters_t));
		/* default decoding parameters */
		parameters->cp_layer = 0;
		parameters->cp_reduce[0] = 0;
		parameters->cp_reduce[1] = 0;
		parameters->cp_reduce[2] = 0;
		parameters->bigendian = 0;

		parameters->decod_format = -1;
		parameters->cod_format = -1;
	}
}

void OPJ_CALLCONV opj_setup_decoder(opj_dinfo_t *dinfo, opj_dparameters_t *parameters) {
	if(dinfo && parameters) {
		if (dinfo->codec_format != CODEC_UNKNOWN) {
			j3d_setup_decoder((opj_j3d_t*)dinfo->j3d_handle, parameters);
		}
	}
}

opj_volume_t* OPJ_CALLCONV opj_decode(opj_dinfo_t *dinfo, opj_cio_t *cio) {
	if(dinfo && cio) {
		if (dinfo->codec_format != CODEC_UNKNOWN) {
			return j3d_decode((opj_j3d_t*)dinfo->j3d_handle, cio);
		}
	}

	return NULL;
}

opj_cinfo_t* OPJ_CALLCONV opj_create_compress(OPJ_CODEC_FORMAT format) {
	opj_cinfo_t *cinfo = (opj_cinfo_t*)opj_malloc(sizeof(opj_cinfo_t));
	if(!cinfo) return NULL;
	cinfo->is_decompressor = false;
	switch(format) {
		case CODEC_J3D:
		case CODEC_J2K:
			/* get a J3D coder handle */
			cinfo->j3d_handle = (void*)j3d_create_compress((opj_common_ptr)cinfo);
			if(!cinfo->j3d_handle) {
				opj_free(cinfo);
				return NULL;
			}
			break;
		default:
			opj_free(cinfo);
			return NULL;
	}

	cinfo->codec_format = format;

	return cinfo;
}

void OPJ_CALLCONV opj_destroy_compress(opj_cinfo_t *cinfo) {
	if(cinfo) {
		/* destroy the codec */
		if (cinfo->codec_format != CODEC_UNKNOWN) {
				j3d_destroy_compress((opj_j3d_t*)cinfo->j3d_handle);
		}
		/* destroy the decompressor */
		opj_free(cinfo);
	}
}

void OPJ_CALLCONV opj_set_default_encoder_parameters(opj_cparameters_t *parameters) {
	if(parameters) {
		memset(parameters, 0, sizeof(opj_cparameters_t));
		/* default coding parameters */
		parameters->numresolution[0] = 3;
		parameters->numresolution[1] = 3;
		parameters->numresolution[2] = 1;
		parameters->cblock_init[0] = 64;
		parameters->cblock_init[1] = 64;
		parameters->cblock_init[2] = 64;
		parameters->prog_order = LRCP;
		parameters->roi_compno = -1;		/* no ROI */
		parameters->atk_wt[0] = 1;				/* 5-3 WT */
		parameters->atk_wt[1] = 1;				/* 5-3 WT */
		parameters->atk_wt[2] = 1;				/* 5-3 WT */
		parameters->irreversible = 0;
		parameters->subsampling_dx = 1;
		parameters->subsampling_dy = 1;
		parameters->subsampling_dz = 1;

		parameters->decod_format = -1;
		parameters->cod_format = -1;
		parameters->encoding_format = ENCOD_2EB;
		parameters->transform_format = TRF_2D_DWT;
	}
}

void OPJ_CALLCONV opj_setup_encoder(opj_cinfo_t *cinfo, opj_cparameters_t *parameters, opj_volume_t *volume) {
	if(cinfo && parameters && volume) {
		if (cinfo->codec_format != CODEC_UNKNOWN) {
			j3d_setup_encoder((opj_j3d_t*)cinfo->j3d_handle, parameters, volume);
		}
	}
}

bool OPJ_CALLCONV opj_encode(opj_cinfo_t *cinfo, opj_cio_t *cio, opj_volume_t *volume, char *index) {
	if(cinfo && cio && volume) {
		if (cinfo->codec_format != CODEC_UNKNOWN) {
			return j3d_encode((opj_j3d_t*)cinfo->j3d_handle, cio, volume, index);
		}
	}

	return false;
}


