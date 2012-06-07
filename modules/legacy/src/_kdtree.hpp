/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008, Xavier Delacour, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// 2008-05-13, Xavier Delacour <xavier.delacour@gmail.com>

#ifndef __cv_kdtree_h__
#define __cv_kdtree_h__

#include "precomp.hpp"

#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include "assert.h"
#include "math.h"

#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable: 4512) // suppress "assignment operator could not be generated"
#endif

// J.S. Beis and D.G. Lowe. Shape indexing using approximate nearest-neighbor search
// in highdimensional spaces. In Proc. IEEE Conf. Comp. Vision Patt. Recog.,
// pages 1000--1006, 1997. http://citeseer.ist.psu.edu/beis97shape.html
#undef __deref
#undef __valuetype

template < class __valuetype, class __deref >
class CvKDTree {
public:
  typedef __deref deref_type;
  typedef typename __deref::scalar_type scalar_type;
  typedef typename __deref::accum_type accum_type;

private:
  struct node {
    int dim;      // split dimension; >=0 for nodes, -1 for leaves
    __valuetype value;    // if leaf, value of leaf
    int left, right;    // node indices of left and right branches
    scalar_type boundary; // left if deref(value,dim)<=boundary, otherwise right
  };
  typedef std::vector < node > node_array;

  __deref deref;    // requires operator() (__valuetype lhs,int dim)

  node_array nodes;   // node storage
  int point_dim;    // dimension of points (the k in kd-tree)
  int root_node;    // index of root node, -1 if empty tree

  // for given set of point indices, compute dimension of highest variance
  template < class __instype, class __valuector >
  int dimension_of_highest_variance(__instype * first, __instype * last,
            __valuector ctor) {
    assert(last - first > 0);

    accum_type maxvar = -std::numeric_limits < accum_type >::max();
    int maxj = -1;
    for (int j = 0; j < point_dim; ++j) {
      accum_type mean = 0;
      for (__instype * k = first; k < last; ++k)
  mean += deref(ctor(*k), j);
      mean /= last - first;
      accum_type var = 0;
      for (__instype * k = first; k < last; ++k) {
  accum_type diff = accum_type(deref(ctor(*k), j)) - mean;
  var += diff * diff;
      }
      var /= last - first;

      assert(maxj != -1 || var >= maxvar);

      if (var >= maxvar) {
  maxvar = var;
  maxj = j;
      }
    }

    return maxj;
  }

  // given point indices and dimension, find index of median; (almost) modifies [first,last)
  // such that points_in[first,median]<=point[median], points_in(median,last)>point[median].
  // implemented as partial quicksort; expected linear perf.
  template < class __instype, class __valuector >
  __instype * median_partition(__instype * first, __instype * last,
             int dim, __valuector ctor) {
    assert(last - first > 0);
    __instype *k = first + (last - first) / 2;
    median_partition(first, last, k, dim, ctor);
    return k;
  }

  template < class __instype, class __valuector >
  struct median_pr {
    const __instype & pivot;
    int dim;
    __deref deref;
    __valuector ctor;
    median_pr(const __instype & _pivot, int _dim, __deref _deref, __valuector _ctor)
      : pivot(_pivot), dim(_dim), deref(_deref), ctor(_ctor) {
    }
    bool operator() (const __instype & lhs) const {
      return deref(ctor(lhs), dim) <= deref(ctor(pivot), dim);
    }
  };

  template < class __instype, class __valuector >
  void median_partition(__instype * first, __instype * last,
      __instype * k, int dim, __valuector ctor) {
    int pivot = (int)((last - first) / 2);

    std::swap(first[pivot], last[-1]);
    __instype *middle = std::partition(first, last - 1,
               median_pr < __instype, __valuector >
               (last[-1], dim, deref, ctor));
    std::swap(*middle, last[-1]);

    if (middle < k)
      median_partition(middle + 1, last, k, dim, ctor);
    else if (middle > k)
      median_partition(first, middle, k, dim, ctor);
  }

  // insert given points into the tree; return created node
  template < class __instype, class __valuector >
  int insert(__instype * first, __instype * last, __valuector ctor) {
    if (first == last)
      return -1;
    else {

      int dim = dimension_of_highest_variance(first, last, ctor);
      __instype *median = median_partition(first, last, dim, ctor);

      __instype *split = median;
      for (; split != last && deref(ctor(*split), dim) ==
       deref(ctor(*median), dim); ++split);

      if (split == last) { // leaf
  int nexti = -1;
  for (--split; split >= first; --split) {
    int i = (int)nodes.size();
    node & n = *nodes.insert(nodes.end(), node());
    n.dim = -1;
    n.value = ctor(*split);
    n.left = -1;
    n.right = nexti;
    nexti = i;
  }

  return nexti;
      } else { // node
  int i = (int)nodes.size();
  // note that recursive insert may invalidate this ref
  node & n = *nodes.insert(nodes.end(), node());

  n.dim = dim;
  n.boundary = deref(ctor(*median), dim);

  int left = insert(first, split, ctor);
  nodes[i].left = left;
  int right = insert(split, last, ctor);
  nodes[i].right = right;

  return i;
      }
    }
  }

  // run to leaf; linear search for p;
  // if found, remove paths to empty leaves on unwind
  bool remove(int *i, const __valuetype & p) {
    if (*i == -1)
      return false;
    node & n = nodes[*i];
    bool r;

    if (n.dim >= 0) { // node
      if (deref(p, n.dim) <= n.boundary) // left
  r = remove(&n.left, p);
      else // right
  r = remove(&n.right, p);

      // if terminal, remove this node
      if (n.left == -1 && n.right == -1)
  *i = -1;

      return r;
    } else { // leaf
      if (n.value == p) {
  *i = n.right;
  return true;
      } else
  return remove(&n.right, p);
    }
  }

public:
  struct identity_ctor {
    const __valuetype & operator() (const __valuetype & rhs) const {
      return rhs;
    }
  };

  // initialize an empty tree
  CvKDTree(__deref _deref = __deref())
    : deref(_deref), root_node(-1) {
  }
  // given points, initialize a balanced tree
  CvKDTree(__valuetype * first, __valuetype * last, int _point_dim,
     __deref _deref = __deref())
    : deref(_deref) {
    set_data(first, last, _point_dim, identity_ctor());
  }
  // given points, initialize a balanced tree
  template < class __instype, class __valuector >
  CvKDTree(__instype * first, __instype * last, int _point_dim,
     __valuector ctor, __deref _deref = __deref())
    : deref(_deref) {
    set_data(first, last, _point_dim, ctor);
  }

  void set_deref(__deref _deref) {
    deref = _deref;
  }

  void set_data(__valuetype * first, __valuetype * last, int _point_dim) {
    set_data(first, last, _point_dim, identity_ctor());
  }
  template < class __instype, class __valuector >
  void set_data(__instype * first, __instype * last, int _point_dim,
    __valuector ctor) {
    point_dim = _point_dim;
    nodes.clear();
    nodes.reserve(last - first);
    root_node = insert(first, last, ctor);
  }

  int dims() const {
    return point_dim;
  }

  // remove the given point
  bool remove(const __valuetype & p) {
    return remove(&root_node, p);
  }

  void print() const {
    print(root_node);
  }
  void print(int i, int indent = 0) const {
    if (i == -1)
      return;
    for (int j = 0; j < indent; ++j)
      std::cout << " ";
    const node & n = nodes[i];
    if (n.dim >= 0) {
      std::cout << "node " << i << ", left " << nodes[i].left << ", right " <<
  nodes[i].right << ", dim " << nodes[i].dim << ", boundary " <<
  nodes[i].boundary << std::endl;
      print(n.left, indent + 3);
      print(n.right, indent + 3);
    } else
      std::cout << "leaf " << i << ", value = " << nodes[i].value << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  // bbf search
public:
  struct bbf_nn {   // info on found neighbors (approx k nearest)
    const __valuetype *p; // nearest neighbor
    accum_type dist;    // distance from d to query point
    bbf_nn(const __valuetype & _p, accum_type _dist)
      : p(&_p), dist(_dist) {
    }
    bool operator<(const bbf_nn & rhs) const {
      return dist < rhs.dist;
    }
  };
  typedef std::vector < bbf_nn > bbf_nn_pqueue;
private:
  struct bbf_node {   // info on branches not taken
    int node;     // corresponding node
    accum_type dist;    // minimum distance from bounds to query point
    bbf_node(int _node, accum_type _dist)
      : node(_node), dist(_dist) {
    }
    bool operator<(const bbf_node & rhs) const {
      return dist > rhs.dist;
    }
  };
  typedef std::vector < bbf_node > bbf_pqueue;
  mutable bbf_pqueue tmp_pq;

  // called for branches not taken, as bbf walks to leaf;
  // construct bbf_node given minimum distance to bounds of alternate branch
  void pq_alternate(int alt_n, bbf_pqueue & pq, scalar_type dist) const {
    if (alt_n == -1)
      return;

    // add bbf_node for alternate branch in priority queue
    pq.push_back(bbf_node(alt_n, dist));
    std::push_heap(pq.begin(), pq.end());
  }

  // called by bbf to walk to leaf;
  // takes one step down the tree towards query point d
  template < class __desctype >
  int bbf_branch(int i, const __desctype * d, bbf_pqueue & pq) const {
    const node & n = nodes[i];
    // push bbf_node with bounds of alternate branch, then branch
    if (d[n.dim] <= n.boundary) { // left
      pq_alternate(n.right, pq, n.boundary - d[n.dim]);
      return n.left;
    } else {      // right
      pq_alternate(n.left, pq, d[n.dim] - n.boundary);
      return n.right;
    }
  }

  // compute euclidean distance between two points
  template < class __desctype >
  accum_type distance(const __desctype * d, const __valuetype & p) const {
    accum_type dist = 0;
    for (int j = 0; j < point_dim; ++j) {
      accum_type diff = accum_type(d[j]) - accum_type(deref(p, j));
      dist += diff * diff;
    } return (accum_type) sqrt(dist);
  }

  // called per candidate nearest neighbor; constructs new bbf_nn for
  // candidate and adds it to priority queue of all candidates; if
  // queue len exceeds k, drops the point furthest from query point d.
  template < class __desctype >
  void bbf_new_nn(bbf_nn_pqueue & nn_pq, int k,
      const __desctype * d, const __valuetype & p) const {
    bbf_nn nn(p, distance(d, p));
    if ((int) nn_pq.size() < k) {
      nn_pq.push_back(nn);
      std::push_heap(nn_pq.begin(), nn_pq.end());
    } else if (nn_pq[0].dist > nn.dist) {
      std::pop_heap(nn_pq.begin(), nn_pq.end());
      nn_pq.end()[-1] = nn;
      std::push_heap(nn_pq.begin(), nn_pq.end());
    }
    assert(nn_pq.size() < 2 || nn_pq[0].dist >= nn_pq[1].dist);
  }

public:
  // finds (with high probability) the k nearest neighbors of d,
  // searching at most emax leaves/bins.
  // ret_nn_pq is an array containing the (at most) k nearest neighbors
  // (see bbf_nn structure def above).
  template < class __desctype >
  int find_nn_bbf(const __desctype * d,
      int k, int emax,
      bbf_nn_pqueue & ret_nn_pq) const {
    assert(k > 0);
    ret_nn_pq.clear();

    if (root_node == -1)
      return 0;

    // add root_node to bbf_node priority queue;
    // iterate while queue non-empty and emax>0
    tmp_pq.clear();
    tmp_pq.push_back(bbf_node(root_node, 0));
    while (tmp_pq.size() && emax > 0) {

      // from node nearest query point d, run to leaf
      std::pop_heap(tmp_pq.begin(), tmp_pq.end());
      bbf_node bbf(tmp_pq.end()[-1]);
      tmp_pq.erase(tmp_pq.end() - 1);

      int i;
      for (i = bbf.node;
     i != -1 && nodes[i].dim >= 0;
     i = bbf_branch(i, d, tmp_pq));

      if (i != -1) {

  // add points in leaf/bin to ret_nn_pq
  do {
    bbf_new_nn(ret_nn_pq, k, d, nodes[i].value);
  } while (-1 != (i = nodes[i].right));

  --emax;
      }
    }

    tmp_pq.clear();
    return (int)ret_nn_pq.size();
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  // orthogonal range search
private:
  void find_ortho_range(int i, scalar_type * bounds_min,
      scalar_type * bounds_max,
      std::vector < __valuetype > &inbounds) const {
    if (i == -1)
      return;
    const node & n = nodes[i];
    if (n.dim >= 0) { // node
      if (bounds_min[n.dim] <= n.boundary)
  find_ortho_range(n.left, bounds_min, bounds_max, inbounds);
      if (bounds_max[n.dim] > n.boundary)
  find_ortho_range(n.right, bounds_min, bounds_max, inbounds);
    } else { // leaf
      do {
  inbounds.push_back(nodes[i].value);
      } while (-1 != (i = nodes[i].right));
    }
  }
public:
  // return all points that lie within the given bounds; inbounds is cleared
  int find_ortho_range(scalar_type * bounds_min,
           scalar_type * bounds_max,
           std::vector < __valuetype > &inbounds) const {
    inbounds.clear();
    find_ortho_range(root_node, bounds_min, bounds_max, inbounds);
    return (int)inbounds.size();
  }
};

#endif // __cv_kdtree_h__

// Local Variables:
// mode:C++
// End:
