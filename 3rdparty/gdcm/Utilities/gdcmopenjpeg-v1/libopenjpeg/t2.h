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
#ifndef __T2_H
#define __T2_H
/**
@file t2.h
@brief Implementation of a tier-2 coding (packetization of code-block data) (T2)

*/

/** @defgroup T2 T2 - Implementation of a tier-2 coding */
/*@{*/

/**
Tier-2 coding
*/
typedef struct opj_t2 {
	/** codec context */
	opj_common_ptr cinfo;

	/** Encoding: pointer to the src image. Decoding: pointer to the dst image. */
	opj_image_t *image;
	/** pointer to the image coding parameters */
	opj_cp_t *cp;
} opj_t2_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
Encode the packets of a tile to a destination buffer
@param t2 T2 handle
@param tileno number of the tile encoded
@param tile the tile for which to write the packets
@param maxlayers maximum number of layers
@param dest the destination buffer
@param len the length of the destination buffer
@param cstr_info Codestream information structure 
@param tpnum Tile part number of the current tile
@param tppos The position of the tile part flag in the progression order
@param pino 
@param t2_mode If == 0 In Threshold calculation ,If == 1 Final pass
@param cur_totnum_tp The total number of tile parts in the current tile
*/
int t2_encode_packets(opj_t2_t* t2,int tileno, opj_tcd_tile_t *tile, int maxlayers, unsigned char *dest, int len, opj_codestream_info_t *cstr_info,int tpnum, int tppos,int pino,J2K_T2_MODE t2_mode,int cur_totnum_tp);
/**
Decode the packets of a tile from a source buffer
@param t2 T2 handle
@param src the source buffer
@param len length of the source buffer
@param tileno number that identifies the tile for which to decode the packets
@param tile tile for which to decode the packets
@param cstr_info Codestream information structure
 */
int t2_decode_packets(opj_t2_t *t2, unsigned char *src, int len, int tileno, opj_tcd_tile_t *tile, opj_codestream_info_t *cstr_info);

/**
Create a T2 handle
@param cinfo Codec context info
@param image Source or destination image
@param cp Image coding parameters
@return Returns a new T2 handle if successful, returns NULL otherwise
*/
opj_t2_t* t2_create(opj_common_ptr cinfo, opj_image_t *image, opj_cp_t *cp);
/**
Destroy a T2 handle
@param t2 T2 handle to destroy
*/
void t2_destroy(opj_t2_t *t2);

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __T2_H */
