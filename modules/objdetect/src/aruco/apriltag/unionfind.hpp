// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2013-2016, The Regents of The University of Michigan.
//
// This software was developed in the APRIL Robotics Lab under the
// direction of Edwin Olson, ebolson@umich.edu. This software may be
// available under alternative licensing terms; contact the address above.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the Regents of The University of Michigan.
#ifndef _OPENCV_UNIONFIND_HPP_
#define _OPENCV_UNIONFIND_HPP_

#include <stdint.h>
#include <stdlib.h>
namespace cv {
namespace aruco {

typedef struct unionfind unionfind_t;
struct unionfind{
    uint32_t maxid;
    struct ufrec *data;
};

struct ufrec{
    // the parent of this node. If a node's parent is its own index,
    // then it is a root.
    uint32_t parent;

    // for the root of a connected component, the number of components
    // connected to it. For intermediate values, it's not meaningful.
    uint32_t size;
};

static inline unionfind_t *unionfind_create(uint32_t maxid){
    unionfind_t *uf = (unionfind_t*) calloc(1, sizeof(unionfind_t));
    uf->maxid = maxid;
    uf->data = (struct ufrec*) malloc((maxid+1) * sizeof(struct ufrec));
    for (unsigned int i = 0; i <= maxid; i++) {
        uf->data[i].size = 1;
        uf->data[i].parent = i;
    }
    return uf;
}

static inline void unionfind_destroy(unionfind_t *uf){
    free(uf->data);
    free(uf);
}

/*
static inline uint32_t unionfind_get_representative(unionfind_t *uf, uint32_t id)
{
    // base case: a node is its own parent
    if (uf->data[id].parent == id)
        return id;

    // otherwise, recurse
    uint32_t root = unionfind_get_representative(uf, uf->data[id].parent);

    // short circuit the path. [XXX This write prevents tail recursion]
    uf->data[id].parent = root;

    return root;
}
 */

// this one seems to be every-so-slightly faster than the recursive
// version above.
static inline uint32_t unionfind_get_representative(unionfind_t *uf, uint32_t id){
    uint32_t root = id;

    // chase down the root
    while (uf->data[root].parent != root) {
        root = uf->data[root].parent;
    }

    // go back and collapse the tree.
    //
    // XXX: on some of our workloads that have very shallow trees
    // (e.g. image segmentation), we are actually faster not doing
    // this...
    while (uf->data[id].parent != root) {
        uint32_t tmp = uf->data[id].parent;
        uf->data[id].parent = root;
        id = tmp;
    }

    return root;
}

static inline uint32_t unionfind_get_set_size(unionfind_t *uf, uint32_t id){
    uint32_t repid = unionfind_get_representative(uf, id);
    return uf->data[repid].size;
}

static inline uint32_t unionfind_connect(unionfind_t *uf, uint32_t aid, uint32_t bid){
    uint32_t aroot = unionfind_get_representative(uf, aid);
    uint32_t broot = unionfind_get_representative(uf, bid);

    if (aroot == broot)
        return aroot;

    // we don't perform "union by rank", but we perform a similar
    // operation (but probably without the same asymptotic guarantee):
    // We join trees based on the number of *elements* (as opposed to
    // rank) contained within each tree. I.e., we use size as a proxy
    // for rank.  In my testing, it's often *faster* to use size than
    // rank, perhaps because the rank of the tree isn't that critical
    // if there are very few nodes in it.
    uint32_t asize = uf->data[aroot].size;
    uint32_t bsize = uf->data[broot].size;

    // optimization idea: We could shortcut some or all of the tree
    // that is grafted onto the other tree. Pro: those nodes were just
    // read and so are probably in cache. Con: it might end up being
    // wasted effort -- the tree might be grafted onto another tree in
    // a moment!
    if (asize > bsize) {
        uf->data[broot].parent = aroot;
        uf->data[aroot].size += bsize;
        return aroot;
    } else {
        uf->data[aroot].parent = broot;
        uf->data[broot].size += asize;
        return broot;
    }
}
}}
#endif
