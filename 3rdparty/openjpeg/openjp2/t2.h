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
 * Copyright (c) 2017, IntoPIX SA <support@intopix.com>
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
#ifndef OPJ_T2_H
#define OPJ_T2_H
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
@param t2               T2 handle
@param tileno           number of the tile encoded
@param tile             the tile for which to write the packets
@param maxlayers        maximum number of layers
@param dest             the destination buffer
@param p_data_written   FIXME DOC
@param len              the length of the destination buffer
@param cstr_info        Codestream information structure
@param p_marker_info    Marker information structure
@param tpnum            Tile part number of the current tile
@param tppos            The position of the tile part flag in the progression order
@param pino             FIXME DOC
@param t2_mode          If == THRESH_CALC In Threshold calculation ,If == FINAL_PASS Final pass
@param p_manager        the user event manager
*/
OPJ_BOOL opj_t2_encode_packets(opj_t2_t* t2,
                               OPJ_UINT32 tileno,
                               opj_tcd_tile_t *tile,
                               OPJ_UINT32 maxlayers,
                               OPJ_BYTE *dest,
                               OPJ_UINT32 * p_data_written,
                               OPJ_UINT32 len,
                               opj_codestream_info_t *cstr_info,
                               opj_tcd_marker_info_t* p_marker_info,
                               OPJ_UINT32 tpnum,
                               OPJ_INT32 tppos,
                               OPJ_UINT32 pino,
                               J2K_T2_MODE t2_mode,
                               opj_event_mgr_t *p_manager);

/**
Decode the packets of a tile from a source buffer
@param tcd TCD handle
@param t2 T2 handle
@param tileno number that identifies the tile for which to decode the packets
@param tile tile for which to decode the packets
@param src         FIXME DOC
@param p_data_read the source buffer
@param len length of the source buffer
@param cstr_info   FIXME DOC
@param p_manager the user event manager

@return FIXME DOC
 */
OPJ_BOOL opj_t2_decode_packets(opj_tcd_t* tcd,
                               opj_t2_t *t2,
                               OPJ_UINT32 tileno,
                               opj_tcd_tile_t *tile,
                               OPJ_BYTE *src,
                               OPJ_UINT32 * p_data_read,
                               OPJ_UINT32 len,
                               opj_codestream_index_t *cstr_info,
                               opj_event_mgr_t *p_manager);

/**
 * Creates a Tier 2 handle
 *
 * @param   p_image     Source or destination image
 * @param   p_cp        Image coding parameters.
 * @return      a new T2 handle if successful, NULL otherwise.
*/
opj_t2_t* opj_t2_create(opj_image_t *p_image, opj_cp_t *p_cp);

/**
Destroy a T2 handle
@param t2 T2 handle to destroy
*/
void opj_t2_destroy(opj_t2_t *t2);

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_T2_H */
