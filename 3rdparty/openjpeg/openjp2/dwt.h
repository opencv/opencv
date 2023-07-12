/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
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

#ifndef OPJ_DWT_H
#define OPJ_DWT_H
/**
@file dwt.h
@brief Implementation of a discrete wavelet transform (DWT)

The functions in DWT.C have for goal to realize forward and inverse discret wavelet
transform with filter 5-3 (reversible) and filter 9-7 (irreversible). The functions in
DWT.C are used by some function in TCD.C.
*/

/** @defgroup DWT DWT - Implementation of a discrete wavelet transform */
/*@{*/


/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Forward 5-3 wavelet transform in 2-D.
Apply a reversible DWT transform to a component of an image.
@param p_tcd TCD handle
@param tilec Tile component information (current tile)
*/
OPJ_BOOL opj_dwt_encode(opj_tcd_t *p_tcd,
                        opj_tcd_tilecomp_t * tilec);

/**
Inverse 5-3 wavelet transform in 2-D.
Apply a reversible inverse DWT transform to a component of an image.
@param p_tcd TCD handle
@param tilec Tile component information (current tile)
@param numres Number of resolution levels to decode
*/
OPJ_BOOL opj_dwt_decode(opj_tcd_t *p_tcd,
                        opj_tcd_tilecomp_t* tilec,
                        OPJ_UINT32 numres);

/**
Get the norm of a wavelet function of a subband at a specified level for the reversible 5-3 DWT.
@param level Level of the wavelet function
@param orient Band of the wavelet function
@return Returns the norm of the wavelet function
*/
OPJ_FLOAT64 opj_dwt_getnorm(OPJ_UINT32 level, OPJ_UINT32 orient);
/**
Forward 9-7 wavelet transform in 2-D.
Apply an irreversible DWT transform to a component of an image.
@param p_tcd TCD handle
@param tilec Tile component information (current tile)
*/
OPJ_BOOL opj_dwt_encode_real(opj_tcd_t *p_tcd,
                             opj_tcd_tilecomp_t * tilec);
/**
Inverse 9-7 wavelet transform in 2-D.
Apply an irreversible inverse DWT transform to a component of an image.
@param p_tcd TCD handle
@param tilec Tile component information (current tile)
@param numres Number of resolution levels to decode
*/
OPJ_BOOL opj_dwt_decode_real(opj_tcd_t *p_tcd,
                             opj_tcd_tilecomp_t* OPJ_RESTRICT tilec,
                             OPJ_UINT32 numres);

/**
Get the norm of a wavelet function of a subband at a specified level for the irreversible 9-7 DWT
@param level Level of the wavelet function
@param orient Band of the wavelet function
@return Returns the norm of the 9-7 wavelet
*/
OPJ_FLOAT64 opj_dwt_getnorm_real(OPJ_UINT32 level, OPJ_UINT32 orient);
/**
Explicit calculation of the Quantization Stepsizes
@param tccp Tile-component coding parameters
@param prec Precint analyzed
*/
void opj_dwt_calc_explicit_stepsizes(opj_tccp_t * tccp, OPJ_UINT32 prec);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_DWT_H */
