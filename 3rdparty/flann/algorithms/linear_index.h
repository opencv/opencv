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

#ifndef LINEARSEARCH_H
#define LINEARSEARCH_H

#include "constants.h"
#include "nn_index.h"

namespace cvflann
{

class LinearIndex : public NNIndex {

	const Matrix<float> dataset;

public:

	LinearIndex(const Matrix<float>& inputData, const LinearIndexParams& params = LinearIndexParams() ) : dataset(inputData)
	{
	}

    flann_algorithm_t getType() const
    {
        return LINEAR;
    }


	int size() const
	{
		return dataset.rows;
	}

	int veclen() const
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

    void saveIndex(FILE* /*stream*/)
    {
		/* nothing to do here for linear search */
    }


    void loadIndex(FILE* /*stream*/)
    {
		/* nothing to do here for linear search */
    }

	void findNeighbors(ResultSet& resultSet, const float* /*vec*/, const SearchParams& /*searchParams*/)
	{
		for (int i=0;i<dataset.rows;++i) {
			resultSet.addPoint(dataset[i],i);
		}
	}

//    Params estimateSearchParams(float precision, Matrix<float>* testset = NULL)
//    {
//        Params params;
//        return params;
//    }

};

}

#endif // LINEARSEARCH_H
