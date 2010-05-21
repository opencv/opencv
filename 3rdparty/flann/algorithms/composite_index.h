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

#ifndef COMPOSITETREE_H
#define COMPOSITETREE_H

#include "constants.h"
#include "nn_index.h"

namespace cvflann
{

class CompositeIndex : public NNIndex
{
	KMeansIndex* kmeans;
	KDTreeIndex* kdtree;

    const Matrix<float> dataset;


public:

	CompositeIndex(const Matrix<float>& inputData, const CompositeIndexParams& params = CompositeIndexParams() ) : dataset(inputData)
	{
		KDTreeIndexParams kdtree_params(params.trees);
		KMeansIndexParams kmeans_params(params.branching, params.iterations, params.centers_init, params.cb_index);

		kdtree = new KDTreeIndex(inputData,kdtree_params);
		kmeans = new KMeansIndex(inputData,kmeans_params);

	}

	virtual ~CompositeIndex()
	{
		delete kdtree;
		delete kmeans;
	}


    flann_algorithm_t getType() const
    {
        return COMPOSITE;
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
		return kmeans->usedMemory()+kdtree->usedMemory();
	}

	void buildIndex()
	{
		logger.info("Building kmeans tree...\n");
		kmeans->buildIndex();
		logger.info("Building kdtree tree...\n");
		kdtree->buildIndex();
	}


    void saveIndex(FILE* /*stream*/)
    {

    }


    void loadIndex(FILE* /*stream*/)
    {

    }

	void findNeighbors(ResultSet& result, const float* vec, const SearchParams& searchParams)
	{
		kmeans->findNeighbors(result,vec,searchParams);
		kdtree->findNeighbors(result,vec,searchParams);
	}


//    Params estimateSearchParams(float precision, Dataset<float>* testset = NULL)
//    {
//        Params params;
//
//        return params;
//    }


};

}

#endif //COMPOSITETREE_H
