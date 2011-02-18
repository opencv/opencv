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
// Copyright (C) 2009, Xavier Delacour, all rights reserved.
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

// 2009-01-12, Xavier Delacour <xavier.delacour@gmail.com>


// * hash perf could be improved
// * in particular, implement integer only (converted fixed from float input)

// * number of hash functions could be reduced (andoni paper)

// * redundant distance computations could be suppressed

// * rework CvLSHOperations interface-- move some of the loops into it to
// * allow efficient async storage


// Datar, M., Immorlica, N., Indyk, P., and Mirrokni, V. S. 2004. Locality-sensitive hashing 
// scheme based on p-stable distributions. In Proceedings of the Twentieth Annual Symposium on 
// Computational Geometry (Brooklyn, New York, USA, June 08 - 11, 2004). SCG '04. ACM, New York, 
// NY, 253-262. DOI= http://doi.acm.org/10.1145/997817.997857 

#include "precomp.hpp"
#include <math.h>
#include <vector>
#include <algorithm>
#include <limits>

template <class T>
class memory_hash_ops : public CvLSHOperations {
  int d;
  std::vector<T> data;
  std::vector<int> free_data;
  struct node {
    int i, h2, next;
  };
  std::vector<node> nodes;
  std::vector<int> free_nodes;
  std::vector<int> bins;

public:
  memory_hash_ops(int _d, int n) : d(_d) {
    bins.resize(n, -1);
  }

  virtual int vector_add(const void* _p) {
    const T* p = (const T*)_p;
    int i;
    if (free_data.empty()) {
      i = (int)data.size();
      data.insert(data.end(), d, 0);
    } else {
      i = free_data.end()[-1];
      free_data.pop_back();
    }
    std::copy(p, p + d, data.begin() + i);
    return i / d;
  }
  virtual void vector_remove(int i) {
    free_data.push_back(i * d);
  }
  virtual const void* vector_lookup(int i) {
    return &data[i * d];
  }
  virtual void vector_reserve(int n) {
    data.reserve(n * d);
  }
  virtual unsigned int vector_count() {
    return (unsigned)(data.size() / d - free_data.size());
  }

  virtual void hash_insert(lsh_hash h, int /*l*/, int i) {
    int ii;
    if (free_nodes.empty()) {
      ii = (int)nodes.size();
      nodes.push_back(node());
    } else {
      ii = free_nodes.end()[-1];
      free_nodes.pop_back();
    }
    node& n = nodes[ii];
    int h1 = h.h1 % bins.size();
    n.i = i;
    n.h2 = h.h2;
    n.next = bins[h1];
    bins[h1] = ii;
  }
  virtual void hash_remove(lsh_hash h, int /*l*/, int i) {
    int h1 = h.h1 % bins.size();
    for (int ii = bins[h1], iin, iip = -1; ii != -1; iip = ii, ii = iin) {
      iin = nodes[ii].next;
      if (nodes[ii].h2 == h.h2 && nodes[ii].i == i) {
	free_nodes.push_back(ii);
	if (iip == -1)
	  bins[h1] = iin;
	else
	  nodes[iip].next = iin;
      }
    }
  }
  virtual int hash_lookup(lsh_hash h, int /*l*/, int* ret_i, int ret_i_max) {
    int h1 = h.h1 % bins.size();
    int k = 0;
    for (int ii = bins[h1]; ii != -1 && k < ret_i_max; ii = nodes[ii].next)
      if (nodes[ii].h2 == h.h2)
	ret_i[k++] = nodes[ii].i;
    return k;
  }
};

template <class T,int cvtype>
class pstable_l2_func {
  CvMat *a, *b, *r1, *r2;
  int d, k;
  double r;
  pstable_l2_func(const pstable_l2_func& x);
  pstable_l2_func& operator= (const pstable_l2_func& rhs);
public:
  typedef T scalar_type;
  typedef T accum_type;
  pstable_l2_func(int _d, int _k, double _r, CvRNG& rng)
    : d(_d), k(_k), r(_r) {
    assert(sizeof(T) == CV_ELEM_SIZE1(cvtype));
    a = cvCreateMat(k, d, cvtype);
    b = cvCreateMat(k, 1, cvtype);
    r1 = cvCreateMat(k, 1, CV_32SC1);
    r2 = cvCreateMat(k, 1, CV_32SC1);
    cvRandArr(&rng, a, CV_RAND_NORMAL, cvScalar(0), cvScalar(1));
    cvRandArr(&rng, b, CV_RAND_UNI, cvScalar(0), cvScalar(r));
    cvRandArr(&rng, r1, CV_RAND_UNI,
	      cvScalar(std::numeric_limits<int>::min()),
	      cvScalar(std::numeric_limits<int>::max()));
    cvRandArr(&rng, r2, CV_RAND_UNI,
	      cvScalar(std::numeric_limits<int>::min()),
	      cvScalar(std::numeric_limits<int>::max()));
  }
  ~pstable_l2_func() {
    cvReleaseMat(&a);
    cvReleaseMat(&b);
    cvReleaseMat(&r1);
    cvReleaseMat(&r2);
  }

  // * factor all L functions into this (reduces number of matrices to 4 total; 
  // * simpler syntax in lsh_table). give parameter l here that tells us which 
  // * row to use etc.

  lsh_hash operator() (const T* x) const {
    const T* aj = (const T*)a->data.ptr;
    const T* bj = (const T*)b->data.ptr;   

    lsh_hash h;
    h.h1 = h.h2 = 0;
    for (int j = 0; j < k; ++j) {
      accum_type s = 0;
      for (int jj = 0; jj < d; ++jj)
	s += aj[jj] * x[jj];
      s += *bj;
      s = accum_type(s/r);
      int si = int(s);
      h.h1 += r1->data.i[j] * si;
      h.h2 += r2->data.i[j] * si;

      aj += d;
      bj++;
    }
    return h;
  }
  accum_type distance(const T* p, const T* q) const {
    accum_type s = 0;
    for (int j = 0; j < d; ++j) {
      accum_type d1 = p[j] - q[j];
      s += d1 * d1;
    }
    return s;
  }
};

template <class H>
class lsh_table {
public:
  typedef typename H::scalar_type scalar_type;
  typedef typename H::accum_type accum_type;
private:
  std::vector<H*> g;
  CvLSHOperations* ops;
  int d, L, k;
  double r;

  static accum_type comp_dist(const std::pair<int,accum_type>& x,
			      const std::pair<int,accum_type>& y) {
    return x.second < y.second;
  }

  lsh_table(const lsh_table& x);
  lsh_table& operator= (const lsh_table& rhs);
public:
  lsh_table(CvLSHOperations* _ops, int _d, int Lval, int _k, double _r, CvRNG& rng)
    : ops(_ops), d(_d), L(Lval), k(_k), r(_r) {
    g.resize(L);
    for (int j = 0; j < L; ++j)
      g[j] = new H(d, k, r, rng);
  }
  ~lsh_table() {
    for (int j = 0; j < L; ++j)
      delete g[j];
    delete ops;
  }

  int dims() const {
    return d;
  }
  unsigned int size() const {
    return ops->vector_count();
  }

  void add(const scalar_type* data, int n, int* ret_indices = 0) {
    for (int j=0;j<n;++j) {
      const scalar_type* x = data+j*d;
      int i = ops->vector_add(x);
      if (ret_indices)
	ret_indices[j] = i;

      for (int l = 0; l < L; ++l) {
	lsh_hash h = (*g[l])(x);
	ops->hash_insert(h, l, i);
      }
    }
  }
  void remove(const int* indices, int n) {
    for (int j = 0; j < n; ++j) {
      int i = indices[n];
      const scalar_type* x = (const scalar_type*)ops->vector_lookup(i);

      for (int l = 0; l < L; ++l) {
	lsh_hash h = (*g[l])(x);
	ops->hash_remove(h, l, i);
      }
      ops->vector_remove(i);
    }
  }
  void query(const scalar_type* q, int k0, int emax, double* dist, int* results) {
    cv::AutoBuffer<int> tmp(emax);
    typedef std::pair<int, accum_type> dr_type; // * swap int and accum_type here, for naming consistency
    cv::AutoBuffer<dr_type> dr(k0);
    int k1 = 0;

    // * handle k0 >= emax, in which case don't track max distance

    for (int l = 0; l < L && emax > 0; ++l) {
      lsh_hash h = (*g[l])(q);
      int m = ops->hash_lookup(h, l, tmp, emax);
      for (int j = 0; j < m && emax > 0; ++j, --emax) {
	int i = tmp[j];
	const scalar_type* p = (const scalar_type*)ops->vector_lookup(i);
	accum_type pd = (*g[l]).distance(p, q);
	if (k1 < k0) {
	  dr[k1++] = std::make_pair(i, pd);
	  std::push_heap(&dr[0], &dr[k1], comp_dist);
	} else if (pd < dr[0].second) {
	  std::pop_heap(&dr[0], &dr[k0], comp_dist);
	  dr[k0 - 1] = std::make_pair(i, pd);
	  std::push_heap(&dr[0], &dr[k0], comp_dist);
	}
      }
    }

    for (int j = 0; j < k1; ++j)
      dist[j] = dr[j].second, results[j] = dr[j].first;
    std::fill(dist + k1, dist + k0, 0);
    std::fill(results + k1, results + k0, -1);
  }
  void query(const scalar_type* data, int n, int k0, int emax, double* dist, int* results) {
    for (int j = 0; j < n; ++j) {
      query(data, k0, emax, dist, results);
      data += d; // * this may not agree with step for some scalar_types
      dist += k0;
      results += k0;
    }
  }
};

typedef lsh_table<pstable_l2_func<float, CV_32FC1> > lsh_pstable_l2_32f;
typedef lsh_table<pstable_l2_func<double, CV_64FC1> > lsh_pstable_l2_64f;

struct CvLSH {
  int type;
  union {
    lsh_pstable_l2_32f* lsh_32f;
    lsh_pstable_l2_64f* lsh_64f;
  } u;
};

CvLSH* cvCreateLSH(CvLSHOperations* ops, int d, int L, int k, int type, double r, int64 seed) {
  CvLSH* lsh = 0;
  CvRNG rng = cvRNG(seed);

  if (type != CV_32FC1 && type != CV_64FC1)
    CV_Error(CV_StsUnsupportedFormat, "vectors must be either CV_32FC1 or CV_64FC1");
  lsh = new CvLSH;
  lsh->type = type;
  switch (type) {
  case CV_32FC1: lsh->u.lsh_32f = new lsh_pstable_l2_32f(ops, d, L, k, r, rng); break;
  case CV_64FC1: lsh->u.lsh_64f = new lsh_pstable_l2_64f(ops, d, L, k, r, rng); break;
  }

  return lsh;
}

CvLSH* cvCreateMemoryLSH(int d, int n, int L, int k, int type, double r, int64 seed) {
  CvLSHOperations* ops = 0;

  switch (type) {
  case CV_32FC1: ops = new memory_hash_ops<float>(d,n); break;
  case CV_64FC1: ops = new memory_hash_ops<double>(d,n); break;
  }
  return cvCreateLSH(ops, d, L, k, type, r, seed);
}

void cvReleaseLSH(CvLSH** lsh) {
  switch ((*lsh)->type) {
  case CV_32FC1: delete (*lsh)->u.lsh_32f; break;
  case CV_64FC1: delete (*lsh)->u.lsh_64f; break;
  default: assert(0);
  }
  delete *lsh;
  *lsh = 0;
}

unsigned int LSHSize(CvLSH* lsh) {
  switch (lsh->type) {
  case CV_32FC1: return lsh->u.lsh_32f->size(); break;
  case CV_64FC1: return lsh->u.lsh_64f->size(); break;
  default: assert(0);
  }
  return 0;
}


void cvLSHAdd(CvLSH* lsh, const CvMat* data, CvMat* indices) {
  int dims, n;
  int* ret_indices = 0;

  switch (lsh->type) {
  case CV_32FC1: dims = lsh->u.lsh_32f->dims(); break;
  case CV_64FC1: dims = lsh->u.lsh_64f->dims(); break;
  default: assert(0); return;
  }

  n = data->rows;

  if (dims != data->cols)
    CV_Error(CV_StsBadSize, "data must be n x d, where d is what was used to construct LSH");

  if (CV_MAT_TYPE(data->type) != lsh->type)
    CV_Error(CV_StsUnsupportedFormat, "type of data and constructed LSH must agree");
  if (indices) {
    if (CV_MAT_TYPE(indices->type) != CV_32SC1)
      CV_Error(CV_StsUnsupportedFormat, "indices must be CV_32SC1");
    if (indices->rows * indices->cols != n)
      CV_Error(CV_StsBadSize, "indices must be n x 1 or 1 x n for n x d data");
    ret_indices = indices->data.i;
  }

  switch (lsh->type) {
  case CV_32FC1: lsh->u.lsh_32f->add(data->data.fl, n, ret_indices); break;
  case CV_64FC1: lsh->u.lsh_64f->add(data->data.db, n, ret_indices); break;
  default: assert(0); return;
  }
}

void cvLSHRemove(CvLSH* lsh, const CvMat* indices) {
  int n;

  if (CV_MAT_TYPE(indices->type) != CV_32SC1)
    CV_Error(CV_StsUnsupportedFormat, "indices must be CV_32SC1");
  n = indices->rows * indices->cols;
  switch (lsh->type) {
  case CV_32FC1: lsh->u.lsh_32f->remove(indices->data.i, n); break;
  case CV_64FC1: lsh->u.lsh_64f->remove(indices->data.i, n); break;
  default: assert(0); return;
  }
}

void cvLSHQuery(CvLSH* lsh, const CvMat* data, CvMat* indices, CvMat* dist, int k, int emax) {
  int dims;

  switch (lsh->type) {
  case CV_32FC1: dims = lsh->u.lsh_32f->dims(); break;
  case CV_64FC1: dims = lsh->u.lsh_64f->dims(); break;
  default: assert(0); return;
  }

  if (k<1)
    CV_Error(CV_StsOutOfRange, "k must be positive");
  if (CV_MAT_TYPE(data->type) != lsh->type)
    CV_Error(CV_StsUnsupportedFormat, "type of data and constructed LSH must agree");
  if (dims != data->cols)
    CV_Error(CV_StsBadSize, "data must be n x d, where d is what was used to construct LSH");
  if (dist->rows != data->rows || dist->cols != k)
    CV_Error(CV_StsBadSize, "dist must be n x k for n x d data");
  if (dist->rows != indices->rows || dist->cols != indices->cols)
    CV_Error(CV_StsBadSize, "dist and indices must be same size");
  if (CV_MAT_TYPE(dist->type) != CV_64FC1)
    CV_Error(CV_StsUnsupportedFormat, "dist must be CV_64FC1");
  if (CV_MAT_TYPE(indices->type) != CV_32SC1)
    CV_Error(CV_StsUnsupportedFormat, "indices must be CV_32SC1");

  switch (lsh->type) {
  case CV_32FC1: lsh->u.lsh_32f->query(data->data.fl, data->rows,
				       k, emax, dist->data.db, indices->data.i); break;
  case CV_64FC1: lsh->u.lsh_64f->query(data->data.db, data->rows,
				       k, emax, dist->data.db, indices->data.i); break;
  default: assert(0); return;
  }
}
