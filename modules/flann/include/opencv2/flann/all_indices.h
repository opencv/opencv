/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/


#ifndef _OPENCV_ALL_INDICES_H_
#define _OPENCV_ALL_INDICES_H_

#include "opencv2/flann/general.h"

#include "opencv2/flann/nn_index.h"
#include "opencv2/flann/kdtree_index.h"
#include "opencv2/flann/kmeans_index.h"
#include "opencv2/flann/composite_index.h"
#include "opencv2/flann/linear_index.h"
#include "opencv2/flann/autotuned_index.h"

namespace cvflann 
{

template<typename T>
NNIndex<T>* create_index_by_type(const Matrix<T>& dataset, const IndexParams& params)
{
	flann_algorithm_t index_type = params.getIndexType();

	NNIndex<T>* nnIndex;
	switch (index_type) {
	case FLANN_INDEX_LINEAR:
		nnIndex = new LinearIndex<T>(dataset, (const LinearIndexParams&)params);
		break;
	case FLANN_INDEX_KDTREE:
		nnIndex = new KDTreeIndex<T>(dataset, (const KDTreeIndexParams&)params);
		break;
	case FLANN_INDEX_KMEANS:
		nnIndex = new KMeansIndex<T>(dataset, (const KMeansIndexParams&)params);
		break;
	case FLANN_INDEX_COMPOSITE:
		nnIndex = new CompositeIndex<T>(dataset, (const CompositeIndexParams&) params);
		break;
	case FLANN_INDEX_AUTOTUNED:
		nnIndex = new AutotunedIndex<T>(dataset, (const AutotunedIndexParams&) params);
		break;
	default:
		throw FLANNException("Unknown index type");
	}

	return nnIndex;
}

} //namespace cvflann

#endif /* _OPENCV_ALL_INDICES_H_ */
