/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2003 Michael David Adams.
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
 * $Id: jpc_tagtree.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <limits.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "jasper/jas_malloc.h"

#include "jpc_tagtree.h"

/******************************************************************************\
* Prototypes.
\******************************************************************************/

static jpc_tagtree_t *jpc_tagtree_alloc(void);

/******************************************************************************\
* Code for creating and destroying tag trees.
\******************************************************************************/

/* Create a tag tree. */

jpc_tagtree_t *jpc_tagtree_create(int numleafsh, int numleafsv)
{
	int nplh[JPC_TAGTREE_MAXDEPTH];
	int nplv[JPC_TAGTREE_MAXDEPTH];
	jpc_tagtreenode_t *node;
	jpc_tagtreenode_t *parentnode;
	jpc_tagtreenode_t *parentnode0;
	jpc_tagtree_t *tree;
	int i;
	int j;
	int k;
	int numlvls;
	int n;

	assert(numleafsh > 0 && numleafsv > 0);

	if (!(tree = jpc_tagtree_alloc())) {
		return 0;
	}
	tree->numleafsh_ = numleafsh;
	tree->numleafsv_ = numleafsv;

	numlvls = 0;
	nplh[0] = numleafsh;
	nplv[0] = numleafsv;
	do {
		n = nplh[numlvls] * nplv[numlvls];
		nplh[numlvls + 1] = (nplh[numlvls] + 1) / 2;
		nplv[numlvls + 1] = (nplv[numlvls] + 1) / 2;
		tree->numnodes_ += n;
		++numlvls;
	} while (n > 1);

	if (!(tree->nodes_ = jas_malloc(tree->numnodes_ * sizeof(jpc_tagtreenode_t)))) {
		return 0;
	}

	/* Initialize the parent links for all nodes in the tree. */

	node = tree->nodes_;
	parentnode = &tree->nodes_[tree->numleafsh_ * tree->numleafsv_];
	parentnode0 = parentnode;

	for (i = 0; i < numlvls - 1; ++i) {
		for (j = 0; j < nplv[i]; ++j) {
			k = nplh[i];
			while (--k >= 0) {
				node->parent_ = parentnode;
				++node;
				if (--k >= 0) {
					node->parent_ = parentnode;
					++node;
				}
				++parentnode;
			}
			if ((j & 1) || j == nplv[i] - 1) {
				parentnode0 = parentnode;
			} else {
				parentnode = parentnode0;
				parentnode0 += nplh[i];
			}
		}
	}
	node->parent_ = 0;

	/* Initialize the data values to something sane. */

	jpc_tagtree_reset(tree);

	return tree;
}

/* Destroy a tag tree. */

void jpc_tagtree_destroy(jpc_tagtree_t *tree)
{
	if (tree->nodes_) {
		jas_free(tree->nodes_);
	}
	jas_free(tree);
}

static jpc_tagtree_t *jpc_tagtree_alloc()
{
	jpc_tagtree_t *tree;

	if (!(tree = jas_malloc(sizeof(jpc_tagtree_t)))) {
		return 0;
	}
	tree->numleafsh_ = 0;
	tree->numleafsv_ = 0;
	tree->numnodes_ = 0;
	tree->nodes_ = 0;

	return tree;
}

/******************************************************************************\
* Code.
\******************************************************************************/

/* Copy state information from one tag tree to another. */

void jpc_tagtree_copy(jpc_tagtree_t *dsttree, jpc_tagtree_t *srctree)
{
	int n;
	jpc_tagtreenode_t *srcnode;
	jpc_tagtreenode_t *dstnode;

	/* The two tag trees must have similar sizes. */
	assert(srctree->numleafsh_ == dsttree->numleafsh_ &&
	  srctree->numleafsv_ == dsttree->numleafsv_);

	n = srctree->numnodes_;
	srcnode = srctree->nodes_;
	dstnode = dsttree->nodes_;
	while (--n >= 0) {
		dstnode->value_ = srcnode->value_;
		dstnode->low_ = srcnode->low_;
		dstnode->known_ = srcnode->known_;
		++dstnode;
		++srcnode;
	}
}

/* Reset all of the state information associated with a tag tree. */

void jpc_tagtree_reset(jpc_tagtree_t *tree)
{
	int n;
	jpc_tagtreenode_t *node;

	n = tree->numnodes_;
	node = tree->nodes_;

	while (--n >= 0) {
		node->value_ = INT_MAX;
		node->low_ = 0;
		node->known_ = 0;
		++node;
	}
}

/* Set the value associated with the specified leaf node, updating
the other nodes as necessary. */

void jpc_tagtree_setvalue(jpc_tagtree_t *tree, jpc_tagtreenode_t *leaf,
  int value)
{
	jpc_tagtreenode_t *node;

	/* Avoid compiler warnings about unused parameters. */
	tree = 0;

	assert(value >= 0);

	node = leaf;
	while (node && node->value_ > value) {
		node->value_ = value;
		node = node->parent_;
	}
}

/* Get a particular leaf node. */

jpc_tagtreenode_t *jpc_tagtree_getleaf(jpc_tagtree_t *tree, int n)
{
	return &tree->nodes_[n];
}

/* Invoke the tag tree encoding procedure. */

int jpc_tagtree_encode(jpc_tagtree_t *tree, jpc_tagtreenode_t *leaf,
  int threshold, jpc_bitstream_t *out)
{
	jpc_tagtreenode_t *stk[JPC_TAGTREE_MAXDEPTH - 1];
	jpc_tagtreenode_t **stkptr;
	jpc_tagtreenode_t *node;
	int low;

	/* Avoid compiler warnings about unused parameters. */
	tree = 0;

	assert(leaf);
	assert(threshold >= 0);

	/* Traverse to the root of the tree, recording the path taken. */
	stkptr = stk;
	node = leaf;
	while (node->parent_) {
		*stkptr++ = node;
		node = node->parent_;
	}

	low = 0;
	for (;;) {
		if (low > node->low_) {
			/* Deferred propagation of the lower bound downward in
			  the tree. */
			node->low_ = low;
		} else {
			low = node->low_;
		}

		while (low < threshold) {
			if (low >= node->value_) {
				if (!node->known_) {
					if (jpc_bitstream_putbit(out, 1) == EOF) {
						return -1;
					}
					node->known_ = 1;
				}
				break;
			}
			if (jpc_bitstream_putbit(out, 0) == EOF) {
				return -1;
			}
			++low;
		}
		node->low_ = low;
		if (stkptr == stk) {
			break;
		}
		node = *--stkptr;

	}
	return (leaf->low_ < threshold) ? 1 : 0;

}

/* Invoke the tag tree decoding procedure. */

int jpc_tagtree_decode(jpc_tagtree_t *tree, jpc_tagtreenode_t *leaf,
  int threshold, jpc_bitstream_t *in)
{
	jpc_tagtreenode_t *stk[JPC_TAGTREE_MAXDEPTH - 1];
	jpc_tagtreenode_t **stkptr;
	jpc_tagtreenode_t *node;
	int low;
	int ret;

	/* Avoid compiler warnings about unused parameters. */
	tree = 0;

	assert(threshold >= 0);

	/* Traverse to the root of the tree, recording the path taken. */
	stkptr = stk;
	node = leaf;
	while (node->parent_) {
		*stkptr++ = node;
		node = node->parent_;
	}

	low = 0;
	for (;;) {
		if (low > node->low_) {
			node->low_ = low;
		} else {
			low = node->low_;
		}
		while (low < threshold && low < node->value_) {
			if ((ret = jpc_bitstream_getbit(in)) < 0) {
				return -1;
			}
			if (ret) {
				node->value_ = low;
			} else {
				++low;
			}
		}
		node->low_ = low;
		if (stkptr == stk) {
			break;
		}
		node = *--stkptr;
	}

	return (node->value_ < threshold) ? 1 : 0;
}

/******************************************************************************\
* Code for debugging.
\******************************************************************************/

void jpc_tagtree_dump(jpc_tagtree_t *tree, FILE *out)
{
	jpc_tagtreenode_t *node;
	int n;

	node = tree->nodes_;
	n = tree->numnodes_;
	while (--n >= 0) {
		fprintf(out, "node %p, parent %p, value %d, lower %d, known %d\n",
		  (void *) node, (void *) node->parent_, node->value_, node->low_,
		  node->known_);
		++node;
	}
}
