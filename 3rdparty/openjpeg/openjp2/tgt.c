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
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
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

#include "opj_includes.h"

/*
==========================================================
   Tag-tree coder interface
==========================================================
*/

opj_tgt_tree_t *opj_tgt_create(OPJ_UINT32 numleafsh, OPJ_UINT32 numleafsv,
                               opj_event_mgr_t *p_manager)
{
    OPJ_INT32 nplh[32];
    OPJ_INT32 nplv[32];
    opj_tgt_node_t *node = 00;
    opj_tgt_node_t *l_parent_node = 00;
    opj_tgt_node_t *l_parent_node0 = 00;
    opj_tgt_tree_t *tree = 00;
    OPJ_UINT32 i;
    OPJ_INT32  j, k;
    OPJ_UINT32 numlvls;
    OPJ_UINT32 n;

    tree = (opj_tgt_tree_t *) opj_calloc(1, sizeof(opj_tgt_tree_t));
    if (!tree) {
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to create Tag-tree\n");
        return 00;
    }

    tree->numleafsh = numleafsh;
    tree->numleafsv = numleafsv;

    numlvls = 0;
    nplh[0] = (OPJ_INT32)numleafsh;
    nplv[0] = (OPJ_INT32)numleafsv;
    tree->numnodes = 0;
    do {
        n = (OPJ_UINT32)(nplh[numlvls] * nplv[numlvls]);
        nplh[numlvls + 1] = (nplh[numlvls] + 1) / 2;
        nplv[numlvls + 1] = (nplv[numlvls] + 1) / 2;
        tree->numnodes += n;
        ++numlvls;
    } while (n > 1);

    /* ADD */
    if (tree->numnodes == 0) {
        opj_free(tree);
        return 00;
    }

    tree->nodes = (opj_tgt_node_t*) opj_calloc(tree->numnodes,
                  sizeof(opj_tgt_node_t));
    if (!tree->nodes) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to create Tag-tree nodes\n");
        opj_free(tree);
        return 00;
    }
    tree->nodes_size = tree->numnodes * (OPJ_UINT32)sizeof(opj_tgt_node_t);

    node = tree->nodes;
    l_parent_node = &tree->nodes[tree->numleafsh * tree->numleafsv];
    l_parent_node0 = l_parent_node;

    for (i = 0; i < numlvls - 1; ++i) {
        for (j = 0; j < nplv[i]; ++j) {
            k = nplh[i];
            while (--k >= 0) {
                node->parent = l_parent_node;
                ++node;
                if (--k >= 0) {
                    node->parent = l_parent_node;
                    ++node;
                }
                ++l_parent_node;
            }
            if ((j & 1) || j == nplv[i] - 1) {
                l_parent_node0 = l_parent_node;
            } else {
                l_parent_node = l_parent_node0;
                l_parent_node0 += nplh[i];
            }
        }
    }
    node->parent = 0;
    opj_tgt_reset(tree);
    return tree;
}

/**
 * Reinitialises a tag-tree from an existing one.
 *
 * @param       p_tree                          the tree to reinitialize.
 * @param       p_num_leafs_h           the width of the array of leafs of the tree
 * @param       p_num_leafs_v           the height of the array of leafs of the tree
 * @return      a new tag-tree if successful, NULL otherwise
*/
opj_tgt_tree_t *opj_tgt_init(opj_tgt_tree_t * p_tree, OPJ_UINT32 p_num_leafs_h,
                             OPJ_UINT32 p_num_leafs_v, opj_event_mgr_t *p_manager)
{
    OPJ_INT32 l_nplh[32];
    OPJ_INT32 l_nplv[32];
    opj_tgt_node_t *l_node = 00;
    opj_tgt_node_t *l_parent_node = 00;
    opj_tgt_node_t *l_parent_node0 = 00;
    OPJ_UINT32 i;
    OPJ_INT32 j, k;
    OPJ_UINT32 l_num_levels;
    OPJ_UINT32 n;
    OPJ_UINT32 l_node_size;

    if (! p_tree) {
        return 00;
    }

    if ((p_tree->numleafsh != p_num_leafs_h) ||
            (p_tree->numleafsv != p_num_leafs_v)) {
        p_tree->numleafsh = p_num_leafs_h;
        p_tree->numleafsv = p_num_leafs_v;

        l_num_levels = 0;
        l_nplh[0] = (OPJ_INT32)p_num_leafs_h;
        l_nplv[0] = (OPJ_INT32)p_num_leafs_v;
        p_tree->numnodes = 0;
        do {
            n = (OPJ_UINT32)(l_nplh[l_num_levels] * l_nplv[l_num_levels]);
            l_nplh[l_num_levels + 1] = (l_nplh[l_num_levels] + 1) / 2;
            l_nplv[l_num_levels + 1] = (l_nplv[l_num_levels] + 1) / 2;
            p_tree->numnodes += n;
            ++l_num_levels;
        } while (n > 1);

        /* ADD */
        if (p_tree->numnodes == 0) {
            opj_tgt_destroy(p_tree);
            return 00;
        }
        l_node_size = p_tree->numnodes * (OPJ_UINT32)sizeof(opj_tgt_node_t);

        if (l_node_size > p_tree->nodes_size) {
            opj_tgt_node_t* new_nodes = (opj_tgt_node_t*) opj_realloc(p_tree->nodes,
                                        l_node_size);
            if (! new_nodes) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Not enough memory to reinitialize the tag tree\n");
                opj_tgt_destroy(p_tree);
                return 00;
            }
            p_tree->nodes = new_nodes;
            memset(((char *) p_tree->nodes) + p_tree->nodes_size, 0,
                   l_node_size - p_tree->nodes_size);
            p_tree->nodes_size = l_node_size;
        }
        l_node = p_tree->nodes;
        l_parent_node = &p_tree->nodes[p_tree->numleafsh * p_tree->numleafsv];
        l_parent_node0 = l_parent_node;

        for (i = 0; i < l_num_levels - 1; ++i) {
            for (j = 0; j < l_nplv[i]; ++j) {
                k = l_nplh[i];
                while (--k >= 0) {
                    l_node->parent = l_parent_node;
                    ++l_node;
                    if (--k >= 0) {
                        l_node->parent = l_parent_node;
                        ++l_node;
                    }
                    ++l_parent_node;
                }
                if ((j & 1) || j == l_nplv[i] - 1) {
                    l_parent_node0 = l_parent_node;
                } else {
                    l_parent_node = l_parent_node0;
                    l_parent_node0 += l_nplh[i];
                }
            }
        }
        l_node->parent = 0;
    }
    opj_tgt_reset(p_tree);

    return p_tree;
}

void opj_tgt_destroy(opj_tgt_tree_t *p_tree)
{
    if (! p_tree) {
        return;
    }

    if (p_tree->nodes) {
        opj_free(p_tree->nodes);
        p_tree->nodes = 00;
    }
    opj_free(p_tree);
}

void opj_tgt_reset(opj_tgt_tree_t *p_tree)
{
    OPJ_UINT32 i;
    opj_tgt_node_t * l_current_node = 00;;

    if (! p_tree) {
        return;
    }

    l_current_node = p_tree->nodes;
    for (i = 0; i < p_tree->numnodes; ++i) {
        l_current_node->value = 999;
        l_current_node->low = 0;
        l_current_node->known = 0;
        ++l_current_node;
    }
}

void opj_tgt_setvalue(opj_tgt_tree_t *tree, OPJ_UINT32 leafno, OPJ_INT32 value)
{
    opj_tgt_node_t *node;
    node = &tree->nodes[leafno];
    while (node && node->value > value) {
        node->value = value;
        node = node->parent;
    }
}

void opj_tgt_encode(opj_bio_t *bio, opj_tgt_tree_t *tree, OPJ_UINT32 leafno,
                    OPJ_INT32 threshold)
{
    opj_tgt_node_t *stk[31];
    opj_tgt_node_t **stkptr;
    opj_tgt_node_t *node;
    OPJ_INT32 low;

    stkptr = stk;
    node = &tree->nodes[leafno];
    while (node->parent) {
        *stkptr++ = node;
        node = node->parent;
    }

    low = 0;
    for (;;) {
        if (low > node->low) {
            node->low = low;
        } else {
            low = node->low;
        }

        while (low < threshold) {
            if (low >= node->value) {
                if (!node->known) {
                    opj_bio_write(bio, 1, 1);
                    node->known = 1;
                }
                break;
            }
            opj_bio_write(bio, 0, 1);
            ++low;
        }

        node->low = low;
        if (stkptr == stk) {
            break;
        }
        node = *--stkptr;
    }
}

OPJ_UINT32 opj_tgt_decode(opj_bio_t *bio, opj_tgt_tree_t *tree,
                          OPJ_UINT32 leafno, OPJ_INT32 threshold)
{
    opj_tgt_node_t *stk[31];
    opj_tgt_node_t **stkptr;
    opj_tgt_node_t *node;
    OPJ_INT32 low;

    stkptr = stk;
    node = &tree->nodes[leafno];
    while (node->parent) {
        *stkptr++ = node;
        node = node->parent;
    }

    low = 0;
    for (;;) {
        if (low > node->low) {
            node->low = low;
        } else {
            low = node->low;
        }
        while (low < threshold && low < node->value) {
            if (opj_bio_read(bio, 1)) {
                node->value = low;
            } else {
                ++low;
            }
        }
        node->low = low;
        if (stkptr == stk) {
            break;
        }
        node = *--stkptr;
    }

    return (node->value < threshold) ? 1 : 0;
}
