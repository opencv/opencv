/*
 * Copyright (c) 2001-2002, David Janssens
 * Copyright (c) 2003, Yannick Verschueren
 * Copyright (c) 2003,  Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#ifndef __TGT_H
#define __TGT_H

typedef struct tgt_node {
	struct tgt_node *parent;
	int value;
	int low;
	int known;
} tgt_node_t;

typedef struct {
	int numleafsh;
	int numleafsv;
	int numnodes;
	tgt_node_t *nodes;
} tgt_tree_t;

/*
 * Create a tag-tree
 * numleafsh: width of the array of leafs of the tree
 * numleafsv: height of the array of leafs of the tree
 */
tgt_tree_t *tgt_create(int numleafsh, int numleafsv);

/*
 * Reset a tag-tree (set all leafs to 0)
 * tree: tag-tree to reset
 */
void tgt_reset(tgt_tree_t *tree);

/*
 * Destroy a tag-tree, liberating memory
 * tree: tag-tree to destroy
 */
void tgt_destroy(tgt_tree_t *tree);

/*
 * Set the value of a leaf of a tag-tree
 * tree: tag-tree to modify
 * leafno: number that identifies the leaf to modify
 * value: new value of the leaf
 */
void tgt_setvalue(tgt_tree_t *tree, int leafno, int value);

/*
 * Decode the value of a leaf of the tag-tree up to a given threshold
 * leafno: number that identifies the leaf to decode
 * threshold: threshold to use when decoding value of the leaf
 */
int tgt_decode(tgt_tree_t *tree, int leafno, int threshold);

#endif
