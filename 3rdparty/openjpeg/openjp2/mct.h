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
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
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

#ifndef OPJ_MCT_H
#define OPJ_MCT_H
/**
@file mct.h
@brief Implementation of a multi-component transforms (MCT)

The functions in MCT.C have for goal to realize reversible and irreversible multicomponent
transform. The functions in MCT.C are used by some function in TCD.C.
*/

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
void opj_mct_encode(OPJ_INT32* OPJ_RESTRICT c0, OPJ_INT32* OPJ_RESTRICT c1,
                    OPJ_INT32* OPJ_RESTRICT c2, OPJ_SIZE_T n);
/**
Apply a reversible multi-component inverse transform to an image
@param c0 Samples for luminance component
@param c1 Samples for red chrominance component
@param c2 Samples for blue chrominance component
@param n Number of samples for each component
*/
void opj_mct_decode(OPJ_INT32* OPJ_RESTRICT c0, OPJ_INT32* OPJ_RESTRICT c1,
                    OPJ_INT32* OPJ_RESTRICT c2, OPJ_SIZE_T n);
/**
Get norm of the basis function used for the reversible multi-component transform
@param compno Number of the component (0->Y, 1->U, 2->V)
@return
*/
OPJ_FLOAT64 opj_mct_getnorm(OPJ_UINT32 compno);

/**
Apply an irreversible multi-component transform to an image
@param c0 Samples for red component
@param c1 Samples for green component
@param c2 Samples blue component
@param n Number of samples for each component
*/
void opj_mct_encode_real(OPJ_INT32* OPJ_RESTRICT c0, OPJ_INT32* OPJ_RESTRICT c1,
                         OPJ_INT32* OPJ_RESTRICT c2, OPJ_SIZE_T n);
/**
Apply an irreversible multi-component inverse transform to an image
@param c0 Samples for luminance component
@param c1 Samples for red chrominance component
@param c2 Samples for blue chrominance component
@param n Number of samples for each component
*/
void opj_mct_decode_real(OPJ_FLOAT32* OPJ_RESTRICT c0,
                         OPJ_FLOAT32* OPJ_RESTRICT c1, OPJ_FLOAT32* OPJ_RESTRICT c2, OPJ_SIZE_T n);
/**
Get norm of the basis function used for the irreversible multi-component transform
@param compno Number of the component (0->Y, 1->U, 2->V)
@return
*/
OPJ_FLOAT64 opj_mct_getnorm_real(OPJ_UINT32 compno);

/**
FIXME DOC
@param p_coding_data    MCT data
@param n                size of components
@param p_data           components
@param p_nb_comp        nb of components (i.e. size of p_data)
@param is_signed        tells if the data is signed
@return OPJ_FALSE if function encounter a problem, OPJ_TRUE otherwise
*/
OPJ_BOOL opj_mct_encode_custom(
    OPJ_BYTE * p_coding_data,
    OPJ_SIZE_T n,
    OPJ_BYTE ** p_data,
    OPJ_UINT32 p_nb_comp,
    OPJ_UINT32 is_signed);
/**
FIXME DOC
@param pDecodingData    MCT data
@param n                size of components
@param pData            components
@param pNbComp          nb of components (i.e. size of p_data)
@param isSigned         tells if the data is signed
@return OPJ_FALSE if function encounter a problem, OPJ_TRUE otherwise
*/
OPJ_BOOL opj_mct_decode_custom(
    OPJ_BYTE * pDecodingData,
    OPJ_SIZE_T n,
    OPJ_BYTE ** pData,
    OPJ_UINT32 pNbComp,
    OPJ_UINT32 isSigned);
/**
FIXME DOC
@param pNorms           MCT data
@param p_nb_comps       size of components
@param pMatrix          components
@return
*/
void opj_calculate_norms(OPJ_FLOAT64 * pNorms,
                         OPJ_UINT32 p_nb_comps,
                         OPJ_FLOAT32 * pMatrix);
/**
FIXME DOC
*/
const OPJ_FLOAT64 * opj_mct_get_mct_norms(void);
/**
FIXME DOC
*/
const OPJ_FLOAT64 * opj_mct_get_mct_norms_real(void);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_MCT_H */
