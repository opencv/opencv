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

#ifndef OPENCV_FLANN_LINEAR_INDEX_H_
#define OPENCV_FLANN_LINEAR_INDEX_H_

//! @cond IGNORED

#include "nn_index.h"

namespace cvflann
{

struct LinearIndexParams : public IndexParams
{
    LinearIndexParams()
    {
        (* this)["algorithm"] = FLANN_INDEX_LINEAR;
    }
};

template <typename Distance>
class LinearIndex : public NNIndex<Distance>
{
public:

    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;


    LinearIndex(const Matrix<ElementType>& inputData, const IndexParams& params = LinearIndexParams(),
                Distance d = Distance()) :
        dataset_(inputData), index_params_(params), distance_(d)
    {
    }

    LinearIndex(const LinearIndex&);
    LinearIndex& operator=(const LinearIndex&);

    flann_algorithm_t getType() const CV_OVERRIDE
    {
        return FLANN_INDEX_LINEAR;
    }


    size_t size() const CV_OVERRIDE
    {
        return dataset_.rows;
    }

    size_t veclen() const CV_OVERRIDE
    {
        return dataset_.cols;
    }


    int usedMemory() const CV_OVERRIDE
    {
        return 0;
    }

    void buildIndex() CV_OVERRIDE
    {
        /* nothing to do here for linear search */
    }

    void saveIndex(FILE*) CV_OVERRIDE
    {
        /* nothing to do here for linear search */
    }


    void loadIndex(FILE*) CV_OVERRIDE
    {
        /* nothing to do here for linear search */

        index_params_["algorithm"] = getType();
    }

    void findNeighbors(ResultSet<DistanceType>& resultSet, const ElementType* vec, const SearchParams& /*searchParams*/) CV_OVERRIDE
    {
        ElementType* data = dataset_.data;
        for (size_t i = 0; i < dataset_.rows; ++i, data += dataset_.cols) {
            DistanceType dist = distance_(data, vec, dataset_.cols);
            resultSet.addPoint(dist, (int)i);
        }
    }

    IndexParams getParameters() const CV_OVERRIDE
    {
        return index_params_;
    }

private:
    /** The dataset */
    const Matrix<ElementType> dataset_;
    /** Index parameters */
    IndexParams index_params_;
    /** Index distance */
    Distance distance_;

};

}

//! @endcond

#endif // OPENCV_FLANN_LINEAR_INDEX_H_
