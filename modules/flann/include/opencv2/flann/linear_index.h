/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
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

#ifndef _OPENCV_LINEARSEARCH_H_
#define _OPENCV_LINEARSEARCH_H_

#include "opencv2/flann/general.h"
#include "opencv2/flann/nn_index.h"


namespace cvflann
{

struct CV_EXPORTS LinearIndexParams : public IndexParams {
	LinearIndexParams() : IndexParams(FLANN_INDEX_LINEAR) {};

	void print() const
	{
		logger().info("Index type: %d\n",(int)algorithm);
	}
};


template <typename ELEM_TYPE, typename DIST_TYPE = typename DistType<ELEM_TYPE>::type >
class LinearIndex : public NNIndex<ELEM_TYPE>
{
	const Matrix<ELEM_TYPE> dataset;
	const LinearIndexParams& index_params;

	LinearIndex(const LinearIndex&);
	LinearIndex& operator=(const LinearIndex&);

public:

	LinearIndex(const Matrix<ELEM_TYPE>& inputData, const LinearIndexParams& params = LinearIndexParams() ) :
		dataset(inputData), index_params(params)
	{
	}

    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_LINEAR;
    }


	size_t size() const
	{
		return dataset.rows;
	}

	size_t veclen() const
	{
		return dataset.cols;
	}


	int usedMemory() const
	{
		return 0;
	}

	void buildIndex()
	{
		/* nothing to do here for linear search */
	}

    void saveIndex(FILE*)
    {
		/* nothing to do here for linear search */
    }


    void loadIndex(FILE*)
    {
		/* nothing to do here for linear search */
    }

	void findNeighbors(ResultSet<ELEM_TYPE>& resultSet, const ELEM_TYPE*, const SearchParams&)
	{
		for (size_t i=0;i<dataset.rows;++i) {
			resultSet.addPoint(dataset[i],(int)i);
		}
	}

	const IndexParams* getParameters() const
	{
		return &index_params;
	}

};

} // namespace cvflann

#endif // _OPENCV_LINEARSEARCH_H_
