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

#ifndef __TGT_H
#define __TGT_H
/**
@file tgt.h
@brief Implementation of a tag-tree coder (TGT)

The functions in TGT.C have for goal to realize a tag-tree coder. The functions in TGT.C
are used by some function in T2.C.
*/

/** @defgroup TGT TGT - Implementation of a tag-tree coder */
/*@{*/

/**
Tag node
*/
typedef struct opj_tgt_node {
  struct opj_tgt_node *parent;
  int value;
  int low;
  int known;
} opj_tgt_node_t;

/**
Tag tree
*/
typedef struct opj_tgt_tree {
  int numleafsh;
  int numleafsv;
  int numnodes;
  opj_tgt_node_t *nodes;
} opj_tgt_tree_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Create a tag-tree
@param numleafsh Width of the array of leafs of the tree
@param numleafsv Height of the array of leafs of the tree
@return Returns a new tag-tree if successful, returns NULL otherwise
*/
opj_tgt_tree_t *tgt_create(int numleafsh, int numleafsv);
/**
Destroy a tag-tree, liberating memory
@param tree Tag-tree to destroy
*/
void tgt_destroy(opj_tgt_tree_t *tree);
/**
Reset a tag-tree (set all leaves to 0)
@param tree Tag-tree to reset
*/
void tgt_reset(opj_tgt_tree_t *tree);
/**
Set the value of a leaf of a tag-tree
@param tree Tag-tree to modify
@param leafno Number that identifies the leaf to modify
@param value New value of the leaf
*/
void tgt_setvalue(opj_tgt_tree_t *tree, int leafno, int value);
/**
Encode the value of a leaf of the tag-tree up to a given threshold
@param bio Pointer to a BIO handle
@param tree Tag-tree to modify
@param leafno Number that identifies the leaf to encode
@param threshold Threshold to use when encoding value of the leaf
*/
void tgt_encode(opj_bio_t *bio, opj_tgt_tree_t *tree, int leafno, int threshold);
/**
Decode the value of a leaf of the tag-tree up to a given threshold
@param bio Pointer to a BIO handle
@param tree Tag-tree to decode
@param leafno Number that identifies the leaf to decode
@param threshold Threshold to use when decoding value of the leaf
@return Returns 1 if the node's value < threshold, returns 0 otherwise
*/
int tgt_decode(opj_bio_t *bio, opj_tgt_tree_t *tree, int leafno, int threshold);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __TGT_H */
