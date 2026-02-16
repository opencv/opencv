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

namespace cv {
namespace aruco {

struct UnionFind {
    UnionFind(uint32_t _maxid)
    {
        maxid = _maxid;
        data.resize(maxid+1);
        for (unsigned int i = 0; i <= maxid; i++) {
            data[i].size = 1;
            data[i].parent = i;
        }
    };

    inline uint32_t get_representative(uint32_t id) {
        uint32_t root = id;

        // chase down the root
        while (data[root].parent != root) {
            root = data[root].parent;
        }

        // go back and collapse the tree.
        //
        // XXX: on some of our workloads that have very shallow trees
        // (e.g. image segmentation), we are actually faster not doing
        // this...
        while (data[id].parent != root) {
            uint32_t tmp = data[id].parent;
            data[id].parent = root;
            id = tmp;
        }

        return root;
    }

    inline uint32_t get_set_size(uint32_t id) {
        uint32_t repid = get_representative(id);
        return data[repid].size;
    }

    inline uint32_t connect(uint32_t aid, uint32_t bid) {
        uint32_t aroot = get_representative(aid);
        uint32_t broot = get_representative(bid);

        if (aroot == broot)
            return aroot;

        // we don't perform "union by rank", but we perform a similar
        // operation (but probably without the same asymptotic guarantee):
        // We join trees based on the number of *elements* (as opposed to
        // rank) contained within each tree. I.e., we use size as a proxy
        // for rank.  In my testing, it's often *faster* to use size than
        // rank, perhaps because the rank of the tree isn't that critical
        // if there are very few nodes in it.
        uint32_t asize = data[aroot].size;
        uint32_t bsize = data[broot].size;

        // optimization idea: We could shortcut some or all of the tree
        // that is grafted onto the other tree. Pro: those nodes were just
        // read and so are probably in cache. Con: it might end up being
        // wasted effort -- the tree might be grafted onto another tree in
        // a moment!
        if (asize > bsize) {
            data[broot].parent = aroot;
            data[aroot].size += bsize;
            return aroot;
        } else {
            data[aroot].parent = broot;
            data[broot].size += asize;
            return broot;
        }
    }

    #define DO_UNIONFIND(dx, dy) if (im.data[y*s + dy*s + x + dx] == v) connect(y*w + x, y*w + dy*w + x + dx);
    void do_line(Mat &im, int w, int s, int y) {
        CV_Assert(y+1 < im.rows);
        CV_Assert(!im.empty());

        for (int x = 1; x < w - 1; x++) {
            uint8_t v = im.data[y*s + x];

            if (v == 127)
                continue;

            // (dx,dy) pairs for 8 connectivity:
            //          (REFERENCE) (1, 0)
            // (-1, 1)    (0, 1)    (1, 1)
            //
            DO_UNIONFIND(1, 0);
            DO_UNIONFIND(0, 1);
            if (v == 255) {
                DO_UNIONFIND(-1, 1);
                DO_UNIONFIND(1, 1);
            }
        }
    }
    #undef DO_UNIONFIND

    struct ufrec {
        // the parent of this node. If a node's parent is its own index,
        // then it is a root.
        uint32_t parent;

        // for the root of a connected component, the number of components
        // connected to it. For intermediate values, it's not meaningful.
        uint32_t size;
    };

    uint32_t maxid;
    std::vector<ufrec> data;
};

}}
#endif
