/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2006, Mónica Díez García, Image Processing Laboratory, University of Valladolid, Spain
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
Packet iterator : resolution level information 
*/
typedef struct opj_pi_resolution {
/** Size of precints in horizontal axis */
	int pdx;
/** Size of precints in vertical axis */
	int pdy;
/** Size of precints in axial axis */
	int pdz;
/** Number of precints in each axis */
	int prctno[3];				
} opj_pi_resolution_t;

/**
Packet iterator : component information 
*/
typedef struct opj_pi_comp {
/** Size in horizontal axis */
	int dx;
/** Size in vertical axis */
	int dy;
/** Size in axial axis */
	int dz;
/** Number of resolution levels */
	int numresolution[3];			
/** Packet iterator : resolution level information */
	opj_pi_resolution_t *resolutions;
} opj_pi_comp_t;

/** 
Packet iterator 
*/
typedef struct opj_pi_iterator {
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
/**	Packet iterator : component information */
opj_pi_comp_t *comps;
	
	int numcomps;
	int tx0, ty0, tz0;
	int tx1, ty1, tz1;
	int x, y, z;
	int dx, dy, dz;
} opj_pi_iterator_t;

/** @name Funciones generales */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Create a packet iterator
@param volume Raw volume for which the packets will be listed
@param cp Coding parameters
@param tileno Number that identifies the tile for which to list the packets
@return Returns a packet iterator that points to the first packet of the tile
@see pi_destroy
*/
opj_pi_iterator_t *pi_create(opj_volume_t * volume, opj_cp_t * cp, int tileno);

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
