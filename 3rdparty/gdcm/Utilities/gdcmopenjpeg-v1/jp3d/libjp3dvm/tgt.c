/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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
void tgt_tree_dump (FILE *fd, opj_tgt_tree_t * tree){
	int nodesno;

	fprintf(fd, "TGT_TREE {\n");
	fprintf(fd, "  numnodes: %d \n", tree->numnodes);	
	fprintf(fd, "  numleafsh: %d, numleafsv: %d, numleafsz: %d,\n", tree->numleafsh, tree->numleafsv, tree->numleafsz);

	for (nodesno = 0; nodesno < tree->numnodes; nodesno++) {
		fprintf(fd, "tgt_node %d {\n", nodesno);
		fprintf(fd, "  value: %d \n", tree->nodes[nodesno].value);
		fprintf(fd, "  low: %d \n", tree->nodes[nodesno].low);
		fprintf(fd, "  known: %d \n", tree->nodes[nodesno].known);
		if (tree->nodes[nodesno].parent) {
			fprintf(fd, "  parent.value: %d \n", tree->nodes[nodesno].parent->value);
			fprintf(fd, "  parent.low: %d \n", tree->nodes[nodesno].parent->low);
			fprintf(fd, "  parent.known: %d \n", tree->nodes[nodesno].parent->known);
		}
		fprintf(fd, "}\n");

	}
	fprintf(fd, "}\n");

}


opj_tgt_tree_t *tgt_create(int numleafsh, int numleafsv, int numleafsz) {
	
	int nplh[32];
	int nplv[32];
	int nplz[32];
	opj_tgt_node_t *node = NULL;
	opj_tgt_node_t *parentnode = NULL;
	opj_tgt_node_t *parentnode0 = NULL;
	opj_tgt_tree_t *tree = NULL;
	int i, j, k, p, p0;
	int numlvls;
	int n, z = 0;

	tree = (opj_tgt_tree_t *) opj_malloc(sizeof(opj_tgt_tree_t));
	if(!tree) 
		return NULL;
	tree->numleafsh = numleafsh;
	tree->numleafsv = numleafsv;
	tree->numleafsz = numleafsz;

	numlvls = 0;
	nplh[0] = numleafsh;
	nplv[0] = numleafsv;
	nplz[0] = numleafsz;
	tree->numnodes = 0;
	do {
		n = nplh[numlvls] * nplv[numlvls] * nplz[numlvls]; 
		nplh[numlvls + 1] = (nplh[numlvls] + 1) / 2;
		nplv[numlvls + 1] = (nplv[numlvls] + 1) / 2;
		nplz[numlvls + 1] = (nplz[numlvls] + 1) / 2;
		tree->numnodes += n;
		++numlvls;
	} while (n > 1);

	if (tree->numnodes == 0) {
		opj_free(tree);
		return NULL;
	}

	tree->nodes = (opj_tgt_node_t *) opj_malloc(tree->numnodes * sizeof(opj_tgt_node_t));
	if(!tree->nodes) {
		opj_free(tree);
		return NULL;
	}

	node = tree->nodes;
	parentnode = &tree->nodes[tree->numleafsh * tree->numleafsv * tree->numleafsz];
	parentnode0 = parentnode;
		
	p = tree->numleafsh * tree->numleafsv * tree->numleafsz;
	p0 = p;
	n = 0;
	//fprintf(stdout,"\nH %d V %d Z %d numlvls %d nodes %d\n",tree->numleafsh,tree->numleafsv,tree->numleafsz,numlvls,tree->numnodes);
	for (i = 0; i < numlvls - 1; ++i) {
		for (j = 0; j < nplv[i]; ++j) {
			k = nplh[i]*nplz[i];
			while (--k >= 0) {
				node->parent = parentnode;		//fprintf(stdout,"node[%d].parent = node[%d]\n",n,p);
				++node;	++n;		
				if (--k >= 0 && n < p) {
					node->parent = parentnode;	//fprintf(stdout,"node[%d].parent = node[%d]\n",n,p);
					++node;	++n;	
				}
				if (nplz[i] != 1){ //2D operation vs 3D operation
					if (--k >= 0 && n < p) {
						node->parent = parentnode;	//fprintf(stdout,"node[%d].parent = node[%d]\n",n,p);
						++node;	++n;
					}
					if (--k >= 0 && n < p) {
						node->parent = parentnode;	//fprintf(stdout,"node[%d].parent = node[%d]\n",n,p);
						++node;	++n;
					}
				}
				++parentnode; ++p;
			}
			if ((j & 1) || j == nplv[i] - 1) {
				parentnode0 = parentnode;			p0 = p;		//fprintf(stdout,"parent = node[%d] \n",p);
			} else {
				parentnode = parentnode0;			p = p0;		//fprintf(stdout,"parent = node[%d] \n",p);
				parentnode0 += nplh[i]*nplz[i];		p0 += nplh[i]*nplz[i];
			}
		}
	}
	node->parent = 0;

	
	tgt_reset(tree);

	return tree;
}

void tgt_destroy(opj_tgt_tree_t *tree) {
	opj_free(tree->nodes);
	opj_free(tree);
}

void tgt_reset(opj_tgt_tree_t *tree) {
	int i;

	if (NULL == tree)
		return;
	
	for (i = 0; i < tree->numnodes; i++) {
		tree->nodes[i].value = 999;
		tree->nodes[i].low = 0;
		tree->nodes[i].known = 0;
	}
}

void tgt_setvalue(opj_tgt_tree_t *tree, int leafno, int value) {
	opj_tgt_node_t *node;
	node = &tree->nodes[leafno];
	while (node && node->value > value) {
		node->value = value;
		node = node->parent;
	}
}

void tgt_encode(opj_bio_t *bio, opj_tgt_tree_t *tree, int leafno, int threshold) {
	opj_tgt_node_t *stk[31];
	opj_tgt_node_t **stkptr;
	opj_tgt_node_t *node;
	int low;

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
					bio_write(bio, 1, 1);
					node->known = 1;
				}
				break;
			}
			bio_write(bio, 0, 1);
			++low;
		}
		
		node->low = low;
		if (stkptr == stk)
			break;
		node = *--stkptr;
	}
}

int tgt_decode(opj_bio_t *bio, opj_tgt_tree_t *tree, int leafno, int threshold) {
	opj_tgt_node_t *stk[31];
	opj_tgt_node_t **stkptr;
	opj_tgt_node_t *node;
	int low;

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
			if (bio_read(bio, 1)) {
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
