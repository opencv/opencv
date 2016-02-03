/*
 * Copyright (c) 2001-2002, David Janssens
 * Copyright (c) 2003, Yannick Verschueren
 * Copyright (c) 2003, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#include "tgt.h"
#include "bio.h"
#include <stdlib.h>
#include <stdio.h>

/// <summary>
/// Reset tag-tree.
/// </summary>
void tgt_reset(tgt_tree_t *tree)
{
    int i;
    for (i=0; i<tree->numnodes; i++) {
        tree->nodes[i].value=999;
        tree->nodes[i].low=0;
        tree->nodes[i].known=0;
    }
}

/// <summary>
/// Create tag-tree.
/// </summary>
tgt_tree_t *tgt_create(int numleafsh, int numleafsv)
{
    int nplh[32];
    int nplv[32];
    tgt_node_t *node;
    tgt_node_t *parentnode;
    tgt_node_t *parentnode0;
    tgt_tree_t *tree;
    int i, j, k;
    int numlvls;
    int n;

    tree=(tgt_tree_t*)malloc(sizeof(tgt_tree_t));
    tree->numleafsh=numleafsh;
    tree->numleafsv=numleafsv;

    numlvls=0;
    nplh[0]=numleafsh;
    nplv[0]=numleafsv;
    tree->numnodes=0;
    do {
        n=nplh[numlvls]*nplv[numlvls];
        nplh[numlvls+1]=(nplh[numlvls]+1)/2;
        nplv[numlvls+1]=(nplv[numlvls]+1)/2;
        tree->numnodes+=n;
        ++numlvls;
    } while (n>1);

    tree->nodes=(tgt_node_t*)malloc(tree->numnodes*sizeof(tgt_node_t));

    node=tree->nodes;
    parentnode=&tree->nodes[tree->numleafsh*tree->numleafsv];
    parentnode0=parentnode;

    for (i=0; i<numlvls-1; ++i) {
        for (j=0; j<nplv[i]; ++j) {
            k=nplh[i];
            while (--k>=0) {
                node->parent=parentnode;
                ++node;
                if (--k >= 0) {
                    node->parent=parentnode;
                    ++node;
                }
                ++parentnode;
            }
            if ((j&1)||j==nplv[i]-1) {
                parentnode0=parentnode;
            } else {
                parentnode=parentnode0;
                parentnode0+=nplh[i];
            }
        }
    }
    node->parent=0;

    tgt_reset(tree);

    return tree;
}

/// <summary>
/// Destroy tag-tree.
/// </summary>
void tgt_destroy(tgt_tree_t *t) {
    free(t->nodes);
    free(t);
}

/// <summary>
/// Set the value of a leaf of the tag-tree.
/// </summary>
void tgt_setvalue(tgt_tree_t *tree, int leafno, int value) {
    tgt_node_t *node;
    node=&tree->nodes[leafno];
    while (node && node->value>value) {
        node->value=value;
        node=node->parent;
    }
}

/// <summary>
/// Decode the value of a leaf of the tag-tree.
/// </summary>
int tgt_decode(tgt_tree_t *tree, int leafno, int threshold)
{
    tgt_node_t *stk[31];
    tgt_node_t **stkptr;
    tgt_node_t *node;
    int low;

    stkptr=stk;
    node=&tree->nodes[leafno];
    while (node->parent) {
        *stkptr++=node;
        node=node->parent;
    }

    low=0;
    for (;;) {
        if (low>node->low) {
            node->low=low;
        } else {
            low=node->low;
        }
        while (low<threshold && low<node->value) {
            if (bio_read(1)) {
                node->value=low;
            } else {
                ++low;
            }
        }
        node->low=low;
        if (stkptr==stk) {
            break;
        }
        node=*--stkptr;
    }

    return (node->value<threshold)?1:0;
}
