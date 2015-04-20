/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#ifndef __CIO_H
#define __CIO_H
/**
@file cio.h
@brief Implementation of a byte input-output process (CIO)

The functions in CIO.C have for goal to realize a byte input / output process.
*/

/** @defgroup CIO CIO - byte input-output stream */
/*@{*/

/** @name Funciones generales (see also openjpeg3d.h) */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Number of bytes left before the end of the stream
@param cio CIO handle
@return Returns the number of bytes before the end of the stream
*/
int cio_numbytesleft(opj_cio_t *cio);
/**
Get pointer to the current position in the stream
@param cio CIO handle
@return Returns a pointer to the current position
*/
unsigned char *cio_getbp(opj_cio_t *cio);
/**
Write some bytes
@param cio CIO handle
@param v Value to write
@param n Number of bytes to write
@return Returns the number of bytes written or 0 if an error occured
*/
unsigned int cio_write(opj_cio_t *cio, unsigned int v, int n);
/**
Read some bytes
@param cio CIO handle
@param n Number of bytes to read
@return Returns the value of the n bytes read
*/
unsigned int cio_read(opj_cio_t *cio, int n);
/**
Skip some bytes
@param cio CIO handle
@param n Number of bytes to skip
*/
void cio_skip(opj_cio_t *cio, int n);
/**
Write some bytes
@param cio CIO handle
@param v Signed integer value to write
@param n Number of bytes to write
@return Returns the number of bytes written or 0 if an error occured
*/
int cio_write_int(opj_cio_t *cio, int v, int n);
/**
Read some bytes
@param cio CIO handle
@param n Number of bytes to read
@return Returns the value of the n bytes read
*/
int cio_read_int(opj_cio_t *cio, int n);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __CIO_H */

