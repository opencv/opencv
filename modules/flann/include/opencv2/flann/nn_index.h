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

#ifndef OPENCV_FLANN_NNINDEX_H
#define OPENCV_FLANN_NNINDEX_H

#include <string>

#include "general.h"
#include "matrix.h"
#include "result_set.h"
#include "params.h"

namespace cvflann
{

/**
 * Nearest-neighbour index base class
 */
template <typename Distance>
class NNIndex
{
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

public:

    virtual ~NNIndex() {}

    /**
     * \brief Builds the index
     */
    virtual void buildIndex() = 0;

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    virtual void knnSearch(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, int knn, const SearchParams& params)
    {
        assert(queries.cols == veclen());
        assert(indices.rows >= queries.rows);
        assert(dists.rows >= queries.rows);
        assert(int(indices.cols) >= knn);
        assert(int(dists.cols) >= knn);

#if 0
        KNNResultSet<DistanceType> resultSet(knn);
        for (size_t i = 0; i < queries.rows; i++) {
            resultSet.init(indices[i], dists[i]);
            findNeighbors(resultSet, queries[i], params);
        }
#else
        KNNUniqueResultSet<DistanceType> resultSet(knn);
        for (size_t i = 0; i < queries.rows; i++) {
            resultSet.clear();
            findNeighbors(resultSet, queries[i], params);
            if (get_param(params,"sorted",true)) resultSet.sortAndCopy(indices[i], dists[i], knn);
            else resultSet.copy(indices[i], dists[i], knn);
        }
#endif
    }

    /**
     * \brief Perform radius search
     * \param[in] query The query point
     * \param[out] indices The indinces of the neighbors found within the given radius
     * \param[out] dists The distances to the nearest neighbors found
     * \param[in] radius The radius used for search
     * \param[in] params Search parameters
     * \returns Number of neighbors found
     */
    virtual int radiusSearch(const Matrix<ElementType>& query, Matrix<int>& indices, Matrix<DistanceType>& dists, float radius, const SearchParams& params)
    {
        if (query.rows != 1) {
            fprintf(stderr, "I can only search one feature at a time for range search\n");
            return -1;
        }
        assert(query.cols == veclen());
        assert(indices.cols == dists.cols);

        int n = 0;
        int* indices_ptr = NULL;
        DistanceType* dists_ptr = NULL;
        if (indices.cols > 0) {
            n = (int)indices.cols;
            indices_ptr = indices[0];
            dists_ptr = dists[0];
        }

        RadiusUniqueResultSet<DistanceType> resultSet((DistanceType)radius);
        resultSet.clear();
        findNeighbors(resultSet, query[0], params);
        if (n>0) {
            if (get_param(params,"sorted",true)) resultSet.sortAndCopy(indices_ptr, dists_ptr, n);
            else resultSet.copy(indices_ptr, dists_ptr, n);
        }

        return (int)resultSet.size();
    }

    /**
     * \brief Saves the index to a stream
     * \param stream The stream to save the index to
     */
    virtual void saveIndex(FILE* stream) = 0;

    /**
     * \brief Loads the index from a stream
     * \param stream The stream from which the index is loaded
     */
    virtual void loadIndex(FILE* stream) = 0;

    /**
     * \returns number of features in this index.
     */
    virtual size_t size() const = 0;

    /**
     * \returns The dimensionality of the features in this index.
     */
    virtual size_t veclen() const = 0;

    /**
     * \returns The amount of memory (in bytes) used by the index.
     */
    virtual int usedMemory() const = 0;

    /**
     * \returns The index type (kdtree, kmeans,...)
     */
    virtual flann_algorithm_t getType() const = 0;

    /**
     * \returns The index parameters
     */
    virtual IndexParams getParameters() const = 0;


    /**
     * \brief Method that searches for nearest-neighbours
     */
    virtual void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) = 0;
};

}

#endif //OPENCV_FLANN_NNINDEX_H
