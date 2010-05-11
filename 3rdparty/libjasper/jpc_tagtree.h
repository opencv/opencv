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
 * Tag Tree Library
 *
 * $Id: jpc_tagtree.h,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

#ifndef JPC_TAGTREE_H
#define JPC_TAGTREE_H

/******************************************************************************\
* Includes
\******************************************************************************/

#include <limits.h>
#include <stdio.h>

#include "jpc_bs.h"

/******************************************************************************\
* Constants
\******************************************************************************/

/* The maximum allowable depth for a tag tree. */
#define JPC_TAGTREE_MAXDEPTH	32

/******************************************************************************\
* Types
\******************************************************************************/

/*
 * Tag tree node.
 */

typedef struct jpc_tagtreenode_ {

	/* The parent of this node. */
	struct jpc_tagtreenode_ *parent_;

	/* The value associated with this node. */
	int value_;

	/* The lower bound on the value associated with this node. */
	int low_;

	/* A flag indicating if the value is known exactly. */
	int known_;

} jpc_tagtreenode_t;

/*
 * Tag tree.
 */

typedef struct {

	/* The number of leaves in the horizontal direction. */
	int numleafsh_;

	/* The number of leaves in the vertical direction. */
	int numleafsv_;

	/* The total number of nodes in the tree. */
	int numnodes_;

	/* The nodes. */
	jpc_tagtreenode_t *nodes_;

} jpc_tagtree_t;

/******************************************************************************\
* Functions.
\******************************************************************************/

/* Create a tag tree. */
jpc_tagtree_t *jpc_tagtree_create(int numleafsh, int numleafsv);

/* Destroy a tag tree. */
void jpc_tagtree_destroy(jpc_tagtree_t *tree);

/* Copy data from one tag tree to another. */
void jpc_tagtree_copy(jpc_tagtree_t *dsttree, jpc_tagtree_t *srctree);

/* Reset the tag tree state. */
void jpc_tagtree_reset(jpc_tagtree_t *tree);

/* Set the value associated with a particular leaf node of a tag tree. */
void jpc_tagtree_setvalue(jpc_tagtree_t *tree, jpc_tagtreenode_t *leaf,
  int value);

/* Get a pointer to a particular leaf node. */
jpc_tagtreenode_t *jpc_tagtree_getleaf(jpc_tagtree_t *tree, int n);

/* Invoke the tag tree decoding procedure. */
int jpc_tagtree_decode(jpc_tagtree_t *tree, jpc_tagtreenode_t *leaf,
  int threshold, jpc_bitstream_t *in);

/* Invoke the tag tree encoding procedure. */
int jpc_tagtree_encode(jpc_tagtree_t *tree, jpc_tagtreenode_t *leaf,
  int threshold, jpc_bitstream_t *out);

/* Dump a tag tree (for debugging purposes). */
void jpc_tagtree_dump(jpc_tagtree_t *tree, FILE *out);

#endif
