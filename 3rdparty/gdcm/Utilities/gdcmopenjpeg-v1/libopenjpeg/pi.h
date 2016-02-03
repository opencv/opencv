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

#ifndef __PI_H
#define __PI_H
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
FIXME: documentation
*/
typedef struct opj_pi_resolution {
  int pdx, pdy;
  int pw, ph;
} opj_pi_resolution_t;

/**
FIXME: documentation
*/
typedef struct opj_pi_comp {
  int dx, dy;
  /** number of resolution levels */
  int numresolutions;
  opj_pi_resolution_t *resolutions;
} opj_pi_comp_t;

/** 
Packet iterator 
*/
typedef struct opj_pi_iterator {
	/** Enabling Tile part generation*/
	char tp_on;
	/** precise if the packet has been already used (usefull for progression order change) */
	short int *include;
	/** layer step used to localize the packet in the include vector */
	int step_l;
	/** resolution step used to localize the packet in the include vector */
	int step_r;	
	/** component step used to localize the packet in the include vector */
	int step_c;	
	/** precinct step used to localize the packet in the include vector */
	int step_p;	
	/** component that identify the packet */
	int compno;
	/** resolution that identify the packet */
	int resno;
	/** precinct that identify the packet */
	int precno;
	/** layer that identify the packet */
	int layno;   
	/** 0 if the first packet */
	int first;
	/** progression order change information */
	opj_poc_t poc;
	/** number of components in the image */
	int numcomps;
	/** Components*/
	opj_pi_comp_t *comps;
	int tx0, ty0, tx1, ty1;
	int x, y, dx, dy;
} opj_pi_iterator_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Create a packet iterator for Encoder
@param image Raw image for which the packets will be listed
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@param t2_mode If == 0 In Threshold calculation ,If == 1 Final pass
@return Returns a packet iterator that points to the first packet of the tile
@see pi_destroy
*/
opj_pi_iterator_t *pi_initialise_encode(opj_image_t *image, opj_cp_t *cp, int tileno,J2K_T2_MODE t2_mode);
/**
Modify the packet iterator for enabling tile part generation
@param pi Handle to the packet iterator generated in pi_initialise_encode  
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@param pino Iterator index for pi
@param tpnum Tile part number of the current tile
@param tppos The position of the tile part flag in the progression order
@param t2_mode If == 0 In Threshold calculation ,If == 1 Final pass
@param cur_totnum_tp The total number of tile parts in the current tile
@return Returns true if an error is detected 
*/
bool pi_create_encode(opj_pi_iterator_t *pi, opj_cp_t *cp,int tileno, int pino,int tpnum, int tppos, J2K_T2_MODE t2_mode,int cur_totnum_tp);
/**
Create a packet iterator for Decoder
@param image Raw image for which the packets will be listed
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@return Returns a packet iterator that points to the first packet of the tile
@see pi_destroy
*/
opj_pi_iterator_t *pi_create_decode(opj_image_t * image, opj_cp_t * cp, int tileno);

/**
Destroy a packet iterator
@param pi Previously created packet iterator
@param cp Coding parameters
@param tileno Number that identifies the tile for which the packets were listed
@see pi_create
*/
void pi_destroy(opj_pi_iterator_t *pi, opj_cp_t *cp, int tileno);

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
