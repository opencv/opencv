/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2002 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 * 
 * JasPer License Version 2.0
 * 
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 * 
 * All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 * 
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 * 
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 * 
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 * 
 * __END_OF_JASPER_LICENSE__
 */

/*
 * Tier-2 Coding Library
 *
 * $Id: jpc_t2cod.h,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

#ifndef JPC_T2COD_H
#define	JPC_T2COD_H

/******************************************************************************\
* Includes.
\******************************************************************************/

#include "jpc_cs.h"

/******************************************************************************\
* Types.
\******************************************************************************/

/* Progression change list. */

typedef struct {

	/* The number of progression changes. */
	int numpchgs;

	/* The maximum number of progression changes that can be accomodated
	  without growing the progression change array. */
	int maxpchgs;

	/* The progression changes. */
	jpc_pchg_t **pchgs;

} jpc_pchglist_t;

/* Packet iterator per-resolution-level information. */

typedef struct {

	/* The number of precincts. */
	int numprcs;

	/* The last layer processed for each precinct. */
	int *prclyrnos;

	/* The precinct width exponent. */
	int prcwidthexpn;

	/* The precinct height exponent. */
	int prcheightexpn;

	/* The number of precincts spanning the resolution level in the horizontal
	  direction. */
	int numhprcs;

} jpc_pirlvl_t;

/* Packet iterator per-component information. */

typedef struct {

	/* The number of resolution levels. */
	int numrlvls;

	/* The per-resolution-level information. */
	jpc_pirlvl_t *pirlvls;

	/* The horizontal sampling period. */
	int hsamp;

	/* The vertical sampling period. */
	int vsamp;

} jpc_picomp_t;

/* Packet iterator class. */

typedef struct {

	/* The number of layers. */
	int numlyrs;

	/* The number of resolution levels. */
	int maxrlvls;

	/* The number of components. */
	int numcomps;

	/* The per-component information. */
	jpc_picomp_t *picomps;

	/* The current component. */
	jpc_picomp_t *picomp;

	/* The current resolution level. */
	jpc_pirlvl_t *pirlvl;

	/* The number of the current component. */
	int compno;

	/* The number of the current resolution level. */
	int rlvlno;

	/* The number of the current precinct. */
	int prcno;

	/* The number of the current layer. */
	int lyrno;

	/* The x-coordinate of the current position. */
	int x;

	/* The y-coordinate of the current position. */
	int y;

	/* The horizontal step size. */
	int xstep;

	/* The vertical step size. */
	int ystep;

	/* The x-coordinate of the top-left corner of the tile on the reference
	  grid. */
	int xstart;

	/* The y-coordinate of the top-left corner of the tile on the reference
	  grid. */
	int ystart;

	/* The x-coordinate of the bottom-right corner of the tile on the
	  reference grid (plus one). */
	int xend;

	/* The y-coordinate of the bottom-right corner of the tile on the
	  reference grid (plus one). */
	int yend;

	/* The current progression change. */
	jpc_pchg_t *pchg;

	/* The progression change list. */
	jpc_pchglist_t *pchglist;

	/* The progression to use in the absense of explicit specification. */
	jpc_pchg_t defaultpchg;

	/* The current progression change number. */
	int pchgno;

	/* Is this the first time in the current progression volume? */
	bool prgvolfirst;

	/* Is the current iterator value valid? */
	bool valid;

	/* The current packet number. */
	int pktno;

} jpc_pi_t;

/******************************************************************************\
* Functions/macros for packet iterators.
\******************************************************************************/

/* Create a packet iterator. */
jpc_pi_t *jpc_pi_create0(void);

/* Destroy a packet iterator. */
void jpc_pi_destroy(jpc_pi_t *pi);

/* Add a progression change to a packet iterator. */
int jpc_pi_addpchg(jpc_pi_t *pi, jpc_pocpchg_t *pchg);

/* Prepare a packet iterator for iteration. */
int jpc_pi_init(jpc_pi_t *pi);

/* Set the iterator to the first packet. */
int jpc_pi_begin(jpc_pi_t *pi);

/* Proceed to the next packet in sequence. */
int jpc_pi_next(jpc_pi_t *pi);

/* Get the index of the current packet. */
#define	jpc_pi_getind(pi)	((pi)->pktno)

/* Get the component number of the current packet. */
#define jpc_pi_cmptno(pi)	(assert(pi->valid), (pi)->compno)

/* Get the resolution level of the current packet. */
#define jpc_pi_rlvlno(pi)	(assert(pi->valid), (pi)->rlvlno)

/* Get the layer number of the current packet. */
#define jpc_pi_lyrno(pi)	(assert(pi->valid), (pi)->lyrno)

/* Get the precinct number of the current packet. */
#define jpc_pi_prcno(pi)	(assert(pi->valid), (pi)->prcno)

/* Get the progression order for the current packet. */
#define jpc_pi_prg(pi)	(assert(pi->valid), (pi)->pchg->prgord)

/******************************************************************************\
* Functions/macros for progression change lists.
\******************************************************************************/

/* Create a progression change list. */
jpc_pchglist_t *jpc_pchglist_create(void);

/* Destroy a progression change list. */
void jpc_pchglist_destroy(jpc_pchglist_t *pchglist);

/* Insert a new element into a progression change list. */
int jpc_pchglist_insert(jpc_pchglist_t *pchglist, int pchgno, jpc_pchg_t *pchg);

/* Remove an element from a progression change list. */
jpc_pchg_t *jpc_pchglist_remove(jpc_pchglist_t *pchglist, int pchgno);

/* Get an element from a progression change list. */
jpc_pchg_t *jpc_pchglist_get(jpc_pchglist_t *pchglist, int pchgno);

/* Copy a progression change list. */
jpc_pchglist_t *jpc_pchglist_copy(jpc_pchglist_t *pchglist);

/* Get the number of elements in a progression change list. */
int jpc_pchglist_numpchgs(jpc_pchglist_t *pchglist);

/******************************************************************************\
* Functions/macros for progression changes.
\******************************************************************************/

/* Destroy a progression change. */
void jpc_pchg_destroy(jpc_pchg_t *pchg);

/* Copy a progression change. */
jpc_pchg_t *jpc_pchg_copy(jpc_pchg_t *pchg);

#endif
