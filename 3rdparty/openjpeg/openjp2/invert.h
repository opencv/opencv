/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
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

#ifndef OPJ_INVERT_H
#define OPJ_INVERT_H
/**
@file invert.h
@brief Implementation of the matrix inversion

The function in INVERT.H compute a matrix inversion with a LUP method
*/

/** @defgroup INVERT INVERT - Implementation of a matrix inversion */
/*@{*/
/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
 * Calculates a n x n double matrix inversion with a LUP method. Data is aligned, rows after rows (or columns after columns).
 * The function does not take ownership of any memory block, data must be fred by the user.
 *
 * @param pSrcMatrix    the matrix to invert.
 * @param pDestMatrix   data to store the inverted matrix.
 * @param nb_compo      size of the matrix
 * @return OPJ_TRUE if the inversion is successful, OPJ_FALSE if the matrix is singular.
 */
OPJ_BOOL opj_matrix_inversion_f(OPJ_FLOAT32 * pSrcMatrix,
                                OPJ_FLOAT32 * pDestMatrix,
                                OPJ_UINT32 nb_compo);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_INVERT_H */
