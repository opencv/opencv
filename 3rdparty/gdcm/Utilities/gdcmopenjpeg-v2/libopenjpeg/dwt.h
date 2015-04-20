/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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

#ifndef __DWT_H
#define __DWT_H
/**
@file dwt.h
@brief Implementation of a discrete wavelet transform (DWT)

The functions in DWT.C have for goal to realize forward and inverse discret wavelet
transform with filter 5-3 (reversible) and filter 9-7 (irreversible). The functions in
DWT.C are used by some function in TCD.C.
*/
#include "openjpeg.h"

struct opj_tcd_tilecomp;
struct opj_tccp;

/** @defgroup DWT DWT - Implementation of a discrete wavelet transform */
/*@{*/


/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Forward 5-3 wavelet tranform in 2-D.
Apply a reversible DWT transform to a component of an image.
@param tilec Tile component information (current tile)
*/
bool dwt_encode(struct opj_tcd_tilecomp * tilec);
/**
Inverse 5-3 wavelet tranform in 2-D.
Apply a reversible inverse DWT transform to a component of an image.
@param tilec Tile component information (current tile)
@param numres Number of resolution levels to decode
*/
bool dwt_decode(struct opj_tcd_tilecomp* tilec, OPJ_UINT32 numres);
/**
Get the gain of a subband for the reversible 5-3 DWT.
@param orient Number that identifies the subband (0->LL, 1->HL, 2->LH, 3->HH)
@return Returns 0 if orient = 0, returns 1 if orient = 1 or 2, returns 2 otherwise
*/
OPJ_UINT32 dwt_getgain(OPJ_UINT32 orient);
/**
Get the norm of a wavelet function of a subband at a specified level for the reversible 5-3 DWT.
@param level Level of the wavelet function
@param orient Band of the wavelet function
@return Returns the norm of the wavelet function
*/
OPJ_FLOAT64 dwt_getnorm(OPJ_UINT32 level, OPJ_UINT32 orient);
/**
Forward 9-7 wavelet transform in 2-D.
Apply an irreversible DWT transform to a component of an image.
@param tilec Tile component information (current tile)
*/
bool dwt_encode_real(struct opj_tcd_tilecomp * tilec);
/**
Inverse 9-7 wavelet transform in 2-D.
Apply an irreversible inverse DWT transform to a component of an image.
@param tilec Tile component information (current tile)
@param numres Number of resolution levels to decode
*/
bool dwt_decode_real(struct opj_tcd_tilecomp* tilec, OPJ_UINT32 numres);
/**
Get the gain of a subband for the irreversible 9-7 DWT.
@param orient Number that identifies the subband (0->LL, 1->HL, 2->LH, 3->HH)
@return Returns the gain of the 9-7 wavelet transform
*/
OPJ_UINT32 dwt_getgain_real(OPJ_UINT32 orient);
/**
Get the norm of a wavelet function of a subband at a specified level for the irreversible 9-7 DWT
@param level Level of the wavelet function
@param orient Band of the wavelet function
@return Returns the norm of the 9-7 wavelet
*/
OPJ_FLOAT64 dwt_getnorm_real(OPJ_UINT32 level, OPJ_UINT32 orient);
/**
Explicit calculation of the Quantization Stepsizes
@param tccp Tile-component coding parameters
@param prec Precint analyzed
*/
void dwt_calc_explicit_stepsizes(struct opj_tccp * tccp, OPJ_UINT32 prec);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __DWT_H */
