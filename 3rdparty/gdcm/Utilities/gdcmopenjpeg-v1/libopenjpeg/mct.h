/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
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

#ifndef __MCT_H
#define __MCT_H
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
void mct_encode(int *c0, int *c1, int *c2, int n);
/**
Apply a reversible multi-component inverse transform to an image
@param c0 Samples for luminance component
@param c1 Samples for red chrominance component
@param c2 Samples for blue chrominance component
@param n Number of samples for each component
*/
void mct_decode(int *c0, int *c1, int *c2, int n);
/**
Get norm of the basis function used for the reversible multi-component transform
@param compno Number of the component (0->Y, 1->U, 2->V)
@return 
*/
double mct_getnorm(int compno);

/**
Apply an irreversible multi-component transform to an image
@param c0 Samples for red component
@param c1 Samples for green component
@param c2 Samples blue component
@param n Number of samples for each component
*/
void mct_encode_real(int *c0, int *c1, int *c2, int n);
/**
Apply an irreversible multi-component inverse transform to an image
@param c0 Samples for luminance component
@param c1 Samples for red chrominance component
@param c2 Samples for blue chrominance component
@param n Number of samples for each component
*/
void mct_decode_real(float* c0, float* c1, float* c2, int n);
/**
Get norm of the basis function used for the irreversible multi-component transform
@param compno Number of the component (0->Y, 1->U, 2->V)
@return 
*/
double mct_getnorm_real(int compno);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __MCT_H */
