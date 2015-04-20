/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
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

#ifndef __RAW_H
#define __RAW_H
/**
@file raw.h
@brief Implementation of operations for raw encoding (RAW)

The functions in RAW.C have for goal to realize the operation of raw encoding linked
with the corresponding mode switch.
*/

/** @defgroup RAW RAW - Implementation of operations for raw encoding */
/*@{*/

/**
RAW encoding operations
*/
typedef struct opj_raw {
	/** temporary buffer where bits are coded or decoded */
	unsigned char c;
	/** number of bits already read or free to write */
	unsigned int ct;
	/** maximum length to decode */
	unsigned int lenmax;
	/** length decoded */
	unsigned int len;
	/** pointer to the current position in the buffer */
	unsigned char *bp;
	/** pointer to the start of the buffer */
	unsigned char *start;
	/** pointer to the end of the buffer */
	unsigned char *end;
} opj_raw_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Create a new RAW handle 
@return Returns a new RAW handle if successful, returns NULL otherwise
*/
opj_raw_t* raw_create(void);
/**
Destroy a previously created RAW handle
@param raw RAW handle to destroy
*/
void raw_destroy(opj_raw_t *raw);
/**
Return the number of bytes written/read since initialisation
@param raw RAW handle to destroy
@return Returns the number of bytes already encoded
*/
int raw_numbytes(opj_raw_t *raw);
/**
Initialize the decoder
@param raw RAW handle
@param bp Pointer to the start of the buffer from which the bytes will be read
@param len Length of the input buffer
*/
void raw_init_dec(opj_raw_t *raw, unsigned char *bp, int len);
/**
Decode a symbol using raw-decoder. Cfr p.506 TAUBMAN
@param raw RAW handle
@return Returns the decoded symbol (0 or 1)
*/
int raw_decode(opj_raw_t *raw);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __RAW_H */
