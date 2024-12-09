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

#ifndef OPJ_PI_H
#define OPJ_PI_H
/**
@file pi.h
@brief Implementation of a packet iterator (PI)

The functions in PI.C have for goal to realize a packet iterator that permits to get the next
packet following the progression order and change of it. The functions in PI.C are used
by some function in T2.C.
*/

/** @defgroup PI PI - Implementation of a packet iterator */
/*@{*/

/**
FIXME DOC
*/
typedef struct opj_pi_resolution {
    OPJ_UINT32 pdx, pdy;
    OPJ_UINT32 pw, ph;
} opj_pi_resolution_t;

/**
FIXME DOC
*/
typedef struct opj_pi_comp {
    OPJ_UINT32 dx, dy;
    /** number of resolution levels */
    OPJ_UINT32 numresolutions;
    opj_pi_resolution_t *resolutions;
} opj_pi_comp_t;

/**
Packet iterator
*/
typedef struct opj_pi_iterator {
    /** Enabling Tile part generation*/
    OPJ_BYTE tp_on;
    /** precise if the packet has been already used (useful for progression order change) */
    OPJ_INT16 *include;
    /** Number of elements in include array */
    OPJ_UINT32 include_size;
    /** layer step used to localize the packet in the include vector */
    OPJ_UINT32 step_l;
    /** resolution step used to localize the packet in the include vector */
    OPJ_UINT32 step_r;
    /** component step used to localize the packet in the include vector */
    OPJ_UINT32 step_c;
    /** precinct step used to localize the packet in the include vector */
    OPJ_UINT32 step_p;
    /** component that identify the packet */
    OPJ_UINT32 compno;
    /** resolution that identify the packet */
    OPJ_UINT32 resno;
    /** precinct that identify the packet */
    OPJ_UINT32 precno;
    /** layer that identify the packet */
    OPJ_UINT32 layno;
    /** 0 if the first packet */
    OPJ_BOOL first;
    /** progression order change information */
    opj_poc_t poc;
    /** number of components in the image */
    OPJ_UINT32 numcomps;
    /** Components*/
    opj_pi_comp_t *comps;
    /** FIXME DOC*/
    OPJ_UINT32 tx0, ty0, tx1, ty1;
    /** FIXME DOC*/
    OPJ_UINT32 x, y;
    /** FIXME DOC*/
    OPJ_UINT32 dx, dy;
    /** event manager */
    opj_event_mgr_t* manager;
} opj_pi_iterator_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
 * Creates a packet iterator for encoding.
 *
 * @param   image       the image being encoded.
 * @param   cp      the coding parameters.
 * @param   tileno  index of the tile being encoded.
 * @param   t2_mode the type of pass for generating the packet iterator
 * @param   manager Event manager
 *
 * @return  a list of packet iterator that points to the first packet of the tile (not true).
*/
opj_pi_iterator_t *opj_pi_initialise_encode(const opj_image_t *image,
        opj_cp_t *cp,
        OPJ_UINT32 tileno,
        J2K_T2_MODE t2_mode,
        opj_event_mgr_t* manager);

/**
 * Updates the encoding parameters of the codec.
 *
 * @param   p_image     the image being encoded.
 * @param   p_cp        the coding parameters.
 * @param   p_tile_no   index of the tile being encoded.
*/
void opj_pi_update_encoding_parameters(const opj_image_t *p_image,
                                       opj_cp_t *p_cp,
                                       OPJ_UINT32 p_tile_no);

/**
Modify the packet iterator for enabling tile part generation
@param pi Handle to the packet iterator generated in pi_initialise_encode
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@param pino   FIXME DOC
@param tpnum Tile part number of the current tile
@param tppos The position of the tile part flag in the progression order
@param t2_mode FIXME DOC
*/
void opj_pi_create_encode(opj_pi_iterator_t *pi,
                          opj_cp_t *cp,
                          OPJ_UINT32 tileno,
                          OPJ_UINT32 pino,
                          OPJ_UINT32 tpnum,
                          OPJ_INT32 tppos,
                          J2K_T2_MODE t2_mode);

/**
Create a packet iterator for Decoder
@param image Raw image for which the packets will be listed
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@param manager Event manager
@return Returns a packet iterator that points to the first packet of the tile
@see opj_pi_destroy
*/
opj_pi_iterator_t *opj_pi_create_decode(opj_image_t * image,
                                        opj_cp_t * cp,
                                        OPJ_UINT32 tileno,
                                        opj_event_mgr_t* manager);
/**
 * Destroys a packet iterator array.
 *
 * @param   p_pi            the packet iterator array to destroy.
 * @param   p_nb_elements   the number of elements in the array.
 */
void opj_pi_destroy(opj_pi_iterator_t *p_pi,
                    OPJ_UINT32 p_nb_elements);

/**
Modify the packet iterator to point to the next packet
@param pi Packet iterator to modify
@return Returns false if pi pointed to the last packet or else returns true
*/
OPJ_BOOL opj_pi_next(opj_pi_iterator_t * pi);

/**
 * Return the number of packets in the tile.
 * @param   image       the image being encoded.
 * @param cp Coding parameters
 * @param tileno Number that identifies the tile.
 */
OPJ_UINT32 opj_get_encoding_packet_count(const opj_image_t *p_image,
        const opj_cp_t *p_cp,
        OPJ_UINT32 p_tile_no);

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_PI_H */
