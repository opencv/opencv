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

#ifndef __MCT_H
#define __MCT_H
/**
@file mct.h
@brief Implementation of a multi-component transforms (MCT)

The functions in MCT.C have for goal to realize reversible and irreversible multicomponent
transform. The functions in MCT.C are used by some function in TCD.C.
*/
#include "openjpeg.h"
/** @defgroup MCT MCT - Implementation of a multi-component transform */
/*@{*/

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Apply a reversible multi-component transform to an image
@param c0 Samples for red component
@param c1 Samples for green component
@param c2 Samples blue component
@param n Number of samples for each component
*/
void mct_encode(OPJ_INT32 *c0, OPJ_INT32 *c1, OPJ_INT32 *c2, OPJ_UINT32 n);
/**
Apply a reversible multi-component inverse transform to an image
@param c0 Samples for luminance component
@param c1 Samples for red chrominance component
@param c2 Samples for blue chrominance component
@param n Number of samples for each component
*/
void mct_decode(OPJ_INT32 *c0, OPJ_INT32 *c1, OPJ_INT32 *c2, OPJ_UINT32 n);
/**
Get norm of the basis function used for the reversible multi-component transform
@param compno Number of the component (0->Y, 1->U, 2->V)
@return
*/
OPJ_FLOAT64 mct_getnorm(OPJ_UINT32 compno);

/**
Apply an irreversible multi-component transform to an image
@param c0 Samples for red component
@param c1 Samples for green component
@param c2 Samples blue component
@param n Number of samples for each component
*/
void mct_encode_real(OPJ_INT32 *c0, OPJ_INT32 *c1, OPJ_INT32 *c2, OPJ_UINT32 n);
/**
Apply an irreversible multi-component inverse transform to an image
@param c0 Samples for luminance component
@param c1 Samples for red chrominance component
@param c2 Samples for blue chrominance component
@param n Number of samples for each component
*/
void mct_decode_real(OPJ_FLOAT32* c0, OPJ_FLOAT32* c1, OPJ_FLOAT32* c2, OPJ_UINT32 n);
/**
Get norm of the basis function used for the irreversible multi-component transform
@param compno Number of the component (0->Y, 1->U, 2->V)
@return
*/
OPJ_FLOAT64 mct_getnorm_real(OPJ_UINT32 compno);

bool mct_encode_custom(
             // MCT data
             OPJ_BYTE * p_coding_data,
             // size of components
             OPJ_UINT32 n,
             // components
             OPJ_BYTE ** p_data,
             // nb of components (i.e. size of p_data)
             OPJ_UINT32 p_nb_comp,
             // tells if the data is signed
             OPJ_UINT32 is_signed);

bool mct_decode_custom(
             // MCT data
             OPJ_BYTE * pDecodingData,
             // size of components
             OPJ_UINT32 n,
             // components
             OPJ_BYTE ** pData,
             // nb of components (i.e. size of pData)
             OPJ_UINT32 pNbComp,
             // tells if the data is signed
             OPJ_UINT32 isSigned);

void opj_calculate_norms(OPJ_FLOAT64 * pNorms,OPJ_UINT32 p_nb_comps,OPJ_FLOAT32 * pMatrix);

const OPJ_FLOAT64 * get_mct_norms ();
const OPJ_FLOAT64 * get_mct_norms_real ();


/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __MCT_H */
