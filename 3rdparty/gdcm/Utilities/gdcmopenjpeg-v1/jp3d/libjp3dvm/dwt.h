/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyrigth (c) 2006, Mónica Díez, LPI-UVA, Spain
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

/** @defgroup DWT DWT - Implementation of a discrete wavelet transform */
/*@{*/

/**
DCCS-LIWT properties
*/


typedef struct opj_wtfilt {
	double *LPS;
	int lenLPS;
	double *HPS;
	int lenHPS;
} opj_wtfilt_t;
/** @name Funciones generales */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Forward 5-3 wavelet tranform in 3-D. 
Apply a reversible DWT transform to a component of an volume.
@param tilec Tile component information (current tile)
@param dwtid Number of identification of wavelet kernel(s) used in DWT in each direction
*/
void dwt_encode(opj_tcd_tilecomp_t * tilec, int dwtid[3]);
/**
Inverse 5-3 wavelet tranform in 3-D.
Apply a reversible inverse DWT transform to a component of an volume.
@param tilec Tile component information (current tile)
@param stops Number of decoded resolution levels in each dimension
@param dwtid Number of identification of wavelet kernel(s) used in DWT in each dimension
*/
void dwt_decode(opj_tcd_tilecomp_t * tilec, int stops[3], int dwtid[3]);
/* ----------------------------------------------------------------------- */
/**
Get the gain of a subband for the reversible 3-D DWT.
@param orient Number that identifies the subband (0->LLL, 1->HLL, 2->LHL, 3->HHL, 4->LLH, 5->HLH, 6->LHH, 7->HHH)
@param reversible Wavelet transformation type
@return Returns 0 if orient = 0, returns 1 if orient = 1,2 or 4, returns 2 if orient = 3,5 or 6, returns 3 otherwise
*/
int dwt_getgain(int orient, int reversible);
/**
Get the norm of a wavelet function of a subband at a specified level for the reversible 5-3 DWT or irreversible 9-7 in 3-D.
@param orient Band of the wavelet function
@param level Levels of the wavelet function in X,Y,Z axis
@param dwtid Wavelet transformation identifier
@return Returns the norm of the wavelet function
*/
double dwt_getnorm(int orient, int level[3], int dwtid[3]);
/* ----------------------------------------------------------------------- */
/**
Calcula el valor del escalón de cuantificación correspondiente a cada subbanda.
@param tccp Tile component coding parameters
@param prec Precision of data
*/
void dwt_calc_explicit_stepsizes(opj_tccp_t * tccp, int prec);
/*@}*/

#endif /* __DWT_H */
