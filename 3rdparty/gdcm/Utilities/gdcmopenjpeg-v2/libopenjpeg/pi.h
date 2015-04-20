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

#ifndef __PI_H
#define __PI_H
/**
@file pi.h
@brief Implementation of a packet iterator (PI)

The functions in PI.C have for goal to realize a packet iterator that permits to get the next
packet following the progression order and change of it. The functions in PI.C are used
by some function in T2.C.
*/
#include "openjpeg.h"
#include "t2.h"
/** @defgroup PI PI - Implementation of a packet iterator */
/*@{*/
struct opj_poc;
struct opj_image;
struct opj_cp;

/**
FIXME: documentation
*/
typedef struct opj_pi_resolution {
  OPJ_UINT32 pdx, pdy;
  OPJ_UINT32 pw, ph;
} opj_pi_resolution_t;

/**
FIXME: documentation
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
  /** precise if the packet has been already used (usefull for progression order change) */
  OPJ_INT16 *include;
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
  /** progression order change information */
  struct opj_poc poc;
  /** number of components in the image */
  OPJ_UINT32 numcomps;
  /** Components*/
  opj_pi_comp_t *comps;
  OPJ_INT32 tx0, ty0, tx1, ty1;
  OPJ_INT32 x, y;
  OPJ_UINT32 dx, dy;
  /** 0 if the first packet */
  OPJ_UINT32 first : 1;
} opj_pi_iterator_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
 * Creates a packet iterator for encoding.
 *
 * @param  p_image    the image being encoded.
 * @param  p_cp    the coding parameters.
 * @param  p_tile_no  index of the tile being encoded.
 * @param  p_t2_mode  the type of pass for generating the packet iterator
 * @return  a list of packet iterator that points to the first packet of the tile (not true).
*/
opj_pi_iterator_t *pi_initialise_encode(const struct opj_image *image,struct opj_cp *cp, OPJ_UINT32 tileno,J2K_T2_MODE t2_mode);

/**
 * Updates the encoding parameters of the codec.
 *
 * @param  p_image    the image being encoded.
 * @param  p_cp    the coding parameters.
 * @param  p_tile_no  index of the tile being encoded.
*/
void pi_update_encoding_parameters(
                    const struct opj_image *p_image,
                    struct opj_cp *p_cp,
                    OPJ_UINT32 p_tile_no
                    );



/**
Modify the packet iterator for enabling tile part generation
@param pi Handle to the packet iterator generated in pi_initialise_encode
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@param tpnum Tile part number of the current tile
@param tppos The position of the tile part flag in the progression order
*/
void pi_create_encode( opj_pi_iterator_t *pi, struct opj_cp *cp,OPJ_UINT32 tileno, OPJ_UINT32 pino,OPJ_UINT32 tpnum, OPJ_INT32 tppos, J2K_T2_MODE t2_mode);


/**
Create a packet iterator for Decoder
@param image Raw image for which the packets will be listed
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@return Returns a packet iterator that points to the first packet of the tile
@see pi_destroy
*/
opj_pi_iterator_t *pi_create_decode(struct opj_image * image, struct opj_cp * cp, OPJ_UINT32 tileno);



/**
 * Destroys a packet iterator array.
 *
 * @param  p_pi      the packet iterator array to destroy.
 * @param  p_nb_elements  the number of elements in the array.
 */
void pi_destroy(
        opj_pi_iterator_t *p_pi,
        OPJ_UINT32 p_nb_elements);

/**
Modify the packet iterator to point to the next packet
@param pi Packet iterator to modify
@return Returns false if pi pointed to the last packet or else returns true
*/
bool pi_next(opj_pi_iterator_t * pi);


/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __PI_H */
