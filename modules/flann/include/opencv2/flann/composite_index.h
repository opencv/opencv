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

#ifndef OPENCV_FLANN_COMPOSITE_INDEX_H_
#define OPENCV_FLANN_COMPOSITE_INDEX_H_

//! @cond IGNORED

#include "nn_index.h"
#include "kdtree_index.h"
#include "kmeans_index.h"

namespace cvflann
{

/**
 * Index parameters for the CompositeIndex.
 */
struct CompositeIndexParams : public IndexParams
{
    CompositeIndexParams(int trees = 4, int branching = 32, int iterations = 11,
                         flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM, float cb_index = 0.2 )
    {
        (*this)["algorithm"] = FLANN_INDEX_KMEANS;
        // number of randomized trees to use (for kdtree)
        (*this)["trees"] = trees;
        // branching factor
        (*this)["branching"] = branching;
        // max iterations to perform in one kmeans clustering (kmeans tree)
        (*this)["iterations"] = iterations;
        // algorithm used for picking the initial cluster centers for kmeans tree
        (*this)["centers_init"] = centers_init;
        // cluster boundary index. Used when searching the kmeans tree
        (*this)["cb_index"] = cb_index;
    }
};


/**
 * This index builds a kd-tree index and a k-means index and performs nearest
 * neighbour search both indexes. This gives a slight boost in search performance
 * as some of the neighbours that are missed by one index are found by the other.
 */
template <typename Distance>
class CompositeIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    /**
     * Index constructor
     * @param inputData dataset containing the points to index
     * @param params Index parameters
     * @param d Distance functor
     */
    CompositeIndex(const Matrix<ElementType>& inputData, const IndexParams& params = CompositeIndexParams(),
                   Distance d = Distance()) : index_params_(params)
    {
        kdtree_index_ = new KDTreeIndex<Distance>(inputData, params, d);
        kmeans_index_ = new KMeansIndex<Distance>(inputData, params, d);

    }

    CompositeIndex(const CompositeIndex&);
    CompositeIndex& operator=(const CompositeIndex&);

    virtual ~CompositeIndex()
    {
        delete kdtree_index_;
        delete kmeans_index_;
    }

    /**
     * @return The index type
     */
    flann_algorithm_t getType() const CV_OVERRIDE
    {
        return FLANN_INDEX_COMPOSITE;
    }

    /**
     * @return Size of the index
     */
    size_t size() const CV_OVERRIDE
    {
        return kdtree_index_->size();
    }

    /**
     * \returns The dimensionality of the features in this index.
     */
    size_t veclen() const CV_OVERRIDE
    {
        return kdtree_index_->veclen();
    }

    /**
     * \returns The amount of memory (in bytes) used by the index.
     */
    int usedMemory() const CV_OVERRIDE
    {
        return kmeans_index_->usedMemory() + kdtree_index_->usedMemory();
    }

    /**
     * \brief Builds the index
     */
    void buildIndex() CV_OVERRIDE
    {
        Logger::info("Building kmeans tree...\n");
        kmeans_index_->buildIndex();
        Logger::info("Building kdtree tree...\n");
        kdtree_index_->buildIndex();
    }

    /**
     * \brief Saves the index to a stream
     * \param stream The stream to save the index to
     */
    void saveIndex(FILE* stream) CV_OVERRIDE
    {
        kmeans_index_->saveIndex(stream);
        kdtree_index_->saveIndex(stream);
    }

    /**
     * \brief Loads the index from a stream
     * \param stream The stream from which the index is loaded
     */
    void loadIndex(FILE* stream) CV_OVERRIDE
    {
        kmeans_index_->loadIndex(stream);
        kdtree_index_->loadIndex(stream);
    }

    /**
     * \returns The index parameters
     */
    IndexParams getParameters() const CV_OVERRIDE
    {
        return index_params_;
    }

    /**
     * \brief Method that searches for nearest-neighbours
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) CV_OVERRIDE
    {
        kmeans_index_->findNeighbors(result, vec, searchParams);
        kdtree_index_->findNeighbors(result, vec, searchParams);
    }

private:
    /** The k-means index */
    KMeansIndex<Distance>* kmeans_index_;

    /** The kd-tree index */
    KDTreeIndex<Distance>* kdtree_index_;

    /** The index parameters */
    const IndexParams index_params_;
};

}

//! @endcond

#endif //OPENCV_FLANN_COMPOSITE_INDEX_H_
