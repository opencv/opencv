/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2011  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2011  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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

#ifndef OPENCV_FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_
#define OPENCV_FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_

#include <algorithm>
#include <string>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>

#include "general.h"
#include "nn_index.h"
#include "dist.h"
#include "matrix.h"
#include "result_set.h"
#include "heap.h"
#include "allocator.h"
#include "random.h"
#include "saving.h"


namespace cvflann
{

struct HierarchicalClusteringIndexParams : public IndexParams
{
    HierarchicalClusteringIndexParams(int branching = 32,
                                      flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
                                      int trees = 4, int leaf_size = 100)
    {
        (*this)["algorithm"] = FLANN_INDEX_HIERARCHICAL;
        // The branching factor used in the hierarchical clustering
        (*this)["branching"] = branching;
        // Algorithm used for picking the initial cluster centers
        (*this)["centers_init"] = centers_init;
        // number of parallel trees to build
        (*this)["trees"] = trees;
        // maximum leaf size
        (*this)["leaf_size"] = leaf_size;
    }
};


/**
 * Hierarchical index
 *
 * Contains a tree constructed through a hierarchical clustering
 * and other information for indexing a set of points for nearest-neighbour matching.
 */
template <typename Distance>
class HierarchicalClusteringIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

private:


    typedef void (HierarchicalClusteringIndex::* centersAlgFunction)(int, int*, int, int*, int&);

    /**
     * The function used for choosing the cluster centers.
     */
    centersAlgFunction chooseCenters;



    /**
     * Chooses the initial centers in the k-means clustering in a random manner.
     *
     * Params:
     *     k = number of centers
     *     vecs = the dataset of points
     *     indices = indices in the dataset
     *     indices_length = length of indices vector
     *
     */
    void chooseCentersRandom(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        UniqueRandom r(indices_length);

        int index;
        for (index=0; index<k; ++index) {
            bool duplicate = true;
            int rnd;
            while (duplicate) {
                duplicate = false;
                rnd = r.next();
                if (rnd<0) {
                    centers_length = index;
                    return;
                }

                centers[index] = indices[rnd];

                for (int j=0; j<index; ++j) {
                    float sq = distance(dataset[centers[index]], dataset[centers[j]], dataset.cols);
                    if (sq<1e-16) {
                        duplicate = true;
                    }
                }
            }
        }

        centers_length = index;
    }


    /**
     * Chooses the initial centers in the k-means using Gonzales' algorithm
     * so that the centers are spaced apart from each other.
     *
     * Params:
     *     k = number of centers
     *     vecs = the dataset of points
     *     indices = indices in the dataset
     * Returns:
     */
    void chooseCentersGonzales(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        int rnd = rand_int(n);
        assert(rnd >=0 && rnd < n);

        centers[0] = indices[rnd];

        int index;
        for (index=1; index<k; ++index) {

            int best_index = -1;
            float best_val = 0;
            for (int j=0; j<n; ++j) {
                float dist = distance(dataset[centers[0]],dataset[indices[j]],dataset.cols);
                for (int i=1; i<index; ++i) {
                    float tmp_dist = distance(dataset[centers[i]],dataset[indices[j]],dataset.cols);
                    if (tmp_dist<dist) {
                        dist = tmp_dist;
                    }
                }
                if (dist>best_val) {
                    best_val = dist;
                    best_index = j;
                }
            }
            if (best_index!=-1) {
                centers[index] = indices[best_index];
            }
            else {
                break;
            }
        }
        centers_length = index;
    }


    /**
     * Chooses the initial centers in the k-means using the algorithm
     * proposed in the KMeans++ paper:
     * Arthur, David; Vassilvitskii, Sergei - k-means++: The Advantages of Careful Seeding
     *
     * Implementation of this function was converted from the one provided in Arthur's code.
     *
     * Params:
     *     k = number of centers
     *     vecs = the dataset of points
     *     indices = indices in the dataset
     * Returns:
     */
    void chooseCentersKMeanspp(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        double currentPot = 0;
        DistanceType* closestDistSq = new DistanceType[n];

        // Choose one random center and set the closestDistSq values
        int index = rand_int(n);
        assert(index >=0 && index < n);
        centers[0] = indices[index];

        for (int i = 0; i < n; i++) {
            closestDistSq[i] = distance(dataset[indices[i]], dataset[indices[index]], dataset.cols);
            currentPot += closestDistSq[i];
        }


        const int numLocalTries = 1;

        // Choose each center
        int centerCount;
        for (centerCount = 1; centerCount < k; centerCount++) {

            // Repeat several trials
            double bestNewPot = -1;
            int bestNewIndex = 0;
            for (int localTrial = 0; localTrial < numLocalTries; localTrial++) {

                // Choose our center - have to be slightly careful to return a valid answer even accounting
                // for possible rounding errors
                double randVal = rand_double(currentPot);
                for (index = 0; index < n-1; index++) {
                    if (randVal <= closestDistSq[index]) break;
                    else randVal -= closestDistSq[index];
                }

                // Compute the new potential
                double newPot = 0;
                for (int i = 0; i < n; i++) newPot += std::min( distance(dataset[indices[i]], dataset[indices[index]], dataset.cols), closestDistSq[i] );

                // Store the best result
                if ((bestNewPot < 0)||(newPot < bestNewPot)) {
                    bestNewPot = newPot;
                    bestNewIndex = index;
                }
            }

            // Add the appropriate center
            centers[centerCount] = indices[bestNewIndex];
            currentPot = bestNewPot;
            for (int i = 0; i < n; i++) closestDistSq[i] = std::min( distance(dataset[indices[i]], dataset[indices[bestNewIndex]], dataset.cols), closestDistSq[i] );
        }

        centers_length = centerCount;

        delete[] closestDistSq;
    }


public:


    /**
     * Index constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the hierarchical k-means algorithm
     */
    HierarchicalClusteringIndex(const Matrix<ElementType>& inputData, const IndexParams& index_params = HierarchicalClusteringIndexParams(),
                                Distance d = Distance())
        : dataset(inputData), params(index_params), root(NULL), indices(NULL), distance(d)
    {
        memoryCounter = 0;

        size_ = dataset.rows;
        veclen_ = dataset.cols;

        branching_ = get_param(params,"branching",32);
        centers_init_ = get_param(params,"centers_init", FLANN_CENTERS_RANDOM);
        trees_ = get_param(params,"trees",4);
        leaf_size_ = get_param(params,"leaf_size",100);

        if (centers_init_==FLANN_CENTERS_RANDOM) {
            chooseCenters = &HierarchicalClusteringIndex::chooseCentersRandom;
        }
        else if (centers_init_==FLANN_CENTERS_GONZALES) {
            chooseCenters = &HierarchicalClusteringIndex::chooseCentersGonzales;
        }
        else if (centers_init_==FLANN_CENTERS_KMEANSPP) {
            chooseCenters = &HierarchicalClusteringIndex::chooseCentersKMeanspp;
        }
        else {
            throw FLANNException("Unknown algorithm for choosing initial centers.");
        }

        trees_ = get_param(params,"trees",4);
        root = new NodePtr[trees_];
        indices = new int*[trees_];
    }

    HierarchicalClusteringIndex(const HierarchicalClusteringIndex&);
    HierarchicalClusteringIndex& operator=(const HierarchicalClusteringIndex&);

    /**
     * Index destructor.
     *
     * Release the memory used by the index.
     */
    virtual ~HierarchicalClusteringIndex()
    {
        if (indices!=NULL) {
            delete[] indices;
        }
    }

    /**
     *  Returns size of index.
     */
    size_t size() const
    {
        return size_;
    }

    /**
     * Returns the length of an index feature.
     */
    size_t veclen() const
    {
        return veclen_;
    }


    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const
    {
        return pool.usedMemory+pool.wastedMemory+memoryCounter;
    }

    /**
     * Builds the index
     */
    void buildIndex()
    {
        if (branching_<2) {
            throw FLANNException("Branching factor must be at least 2");
        }
        for (int i=0; i<trees_; ++i) {
            indices[i] = new int[size_];
            for (size_t j=0; j<size_; ++j) {
                indices[i][j] = j;
            }
            root[i] = pool.allocate<Node>();
            computeClustering(root[i], indices[i], size_, branching_,0);
        }
    }


    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_HIERARCHICAL;
    }


    void saveIndex(FILE* stream)
    {
        save_value(stream, branching_);
        save_value(stream, trees_);
        save_value(stream, centers_init_);
        save_value(stream, leaf_size_);
        save_value(stream, memoryCounter);
        for (int i=0; i<trees_; ++i) {
            save_value(stream, *indices[i], size_);
            save_tree(stream, root[i], i);
        }

    }


    void loadIndex(FILE* stream)
    {
        load_value(stream, branching_);
        load_value(stream, trees_);
        load_value(stream, centers_init_);
        load_value(stream, leaf_size_);
        load_value(stream, memoryCounter);
        indices = new int*[trees_];
        root = new NodePtr[trees_];
        for (int i=0; i<trees_; ++i) {
            indices[i] = new int[size_];
            load_value(stream, *indices[i], size_);
            load_tree(stream, root[i], i);
        }

        params["algorithm"] = getType();
        params["branching"] = branching_;
        params["trees"] = trees_;
        params["centers_init"] = centers_init_;
        params["leaf_size"] = leaf_size_;
    }


    /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     searchParams = parameters that influence the search algorithm (checks)
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams)
    {

        int maxChecks = get_param(searchParams,"checks",32);

        // Priority queue storing intermediate branches in the best-bin-first search
        Heap<BranchSt>* heap = new Heap<BranchSt>(size_);

        std::vector<bool> checked(size_,false);
        int checks = 0;
        for (int i=0; i<trees_; ++i) {
            findNN(root[i], result, vec, checks, maxChecks, heap, checked);
        }

        BranchSt branch;
        while (heap->popMin(branch) && (checks<maxChecks || !result.full())) {
            NodePtr node = branch.node;
            findNN(node, result, vec, checks, maxChecks, heap, checked);
        }
        assert(result.full());

        delete heap;

    }

    IndexParams getParameters() const
    {
        return params;
    }


private:

    /**
     * Struture representing a node in the hierarchical k-means tree.
     */
    struct Node
    {
        /**
         * The cluster center index
         */
        int pivot;
        /**
         * The cluster size (number of points in the cluster)
         */
        int size;
        /**
         * Child nodes (only for non-terminal nodes)
         */
        Node** childs;
        /**
         * Node points (only for terminal nodes)
         */
        int* indices;
        /**
         * Level
         */
        int level;
    };
    typedef Node* NodePtr;



    /**
     * Alias definition for a nicer syntax.
     */
    typedef BranchStruct<NodePtr, DistanceType> BranchSt;



    void save_tree(FILE* stream, NodePtr node, int num)
    {
        save_value(stream, *node);
        if (node->childs==NULL) {
            int indices_offset = node->indices - indices[num];
            save_value(stream, indices_offset);
        }
        else {
            for(int i=0; i<branching_; ++i) {
                save_tree(stream, node->childs[i], num);
            }
        }
    }


    void load_tree(FILE* stream, NodePtr& node, int num)
    {
        node = pool.allocate<Node>();
        load_value(stream, *node);
        if (node->childs==NULL) {
            int indices_offset;
            load_value(stream, indices_offset);
            node->indices = indices[num] + indices_offset;
        }
        else {
            node->childs = pool.allocate<NodePtr>(branching_);
            for(int i=0; i<branching_; ++i) {
                load_tree(stream, node->childs[i], num);
            }
        }
    }




    void computeLabels(int* indices, int indices_length,  int* centers, int centers_length, int* labels, DistanceType& cost)
    {
        cost = 0;
        for (int i=0; i<indices_length; ++i) {
            ElementType* point = dataset[indices[i]];
            DistanceType dist = distance(point, dataset[centers[0]], veclen_);
            labels[i] = 0;
            for (int j=1; j<centers_length; ++j) {
                DistanceType new_dist = distance(point, dataset[centers[j]], veclen_);
                if (dist>new_dist) {
                    labels[i] = j;
                    dist = new_dist;
                }
            }
            cost += dist;
        }
    }

    /**
     * The method responsible with actually doing the recursive hierarchical
     * clustering
     *
     * Params:
     *     node = the node to cluster
     *     indices = indices of the points belonging to the current node
     *     branching = the branching factor to use in the clustering
     *
     * TODO: for 1-sized clusters don't store a cluster center (it's the same as the single cluster point)
     */
    void computeClustering(NodePtr node, int* indices, int indices_length, int branching, int level)
    {
        node->size = indices_length;
        node->level = level;

        if (indices_length < leaf_size_) { // leaf node
            node->indices = indices;
            std::sort(node->indices,node->indices+indices_length);
            node->childs = NULL;
            return;
        }

        std::vector<int> centers(branching);
        std::vector<int> labels(indices_length);

        int centers_length;
        (this->*chooseCenters)(branching, indices, indices_length, &centers[0], centers_length);

        if (centers_length<branching) {
            node->indices = indices;
            std::sort(node->indices,node->indices+indices_length);
            node->childs = NULL;
            return;
        }


        //	assign points to clusters
        DistanceType cost;
        computeLabels(indices, indices_length, &centers[0], centers_length, &labels[0], cost);

        node->childs = pool.allocate<NodePtr>(branching);
        int start = 0;
        int end = start;
        for (int i=0; i<branching; ++i) {
            for (int j=0; j<indices_length; ++j) {
                if (labels[j]==i) {
                    std::swap(indices[j],indices[end]);
                    std::swap(labels[j],labels[end]);
                    end++;
                }
            }

            node->childs[i] = pool.allocate<Node>();
            node->childs[i]->pivot = centers[i];
            node->childs[i]->indices = NULL;
            computeClustering(node->childs[i],indices+start, end-start, branching, level+1);
            start=end;
        }
    }



    /**
     * Performs one descent in the hierarchical k-means tree. The branches not
     * visited are stored in a priority queue.
     *
     * Params:
     *      node = node to explore
     *      result = container for the k-nearest neighbors found
     *      vec = query points
     *      checks = how many points in the dataset have been checked so far
     *      maxChecks = maximum dataset points to checks
     */


    void findNN(NodePtr node, ResultSet<DistanceType>& result, const ElementType* vec, int& checks, int maxChecks,
                Heap<BranchSt>* heap, std::vector<bool>& checked)
    {
        if (node->childs==NULL) {
            if (checks>=maxChecks) {
                if (result.full()) return;
            }
            checks += node->size;
            for (int i=0; i<node->size; ++i) {
                int index = node->indices[i];
                if (!checked[index]) {
                    DistanceType dist = distance(dataset[index], vec, veclen_);
                    result.addPoint(dist, index);
                    checked[index] = true;
                }
            }
        }
        else {
            DistanceType* domain_distances = new DistanceType[branching_];
            int best_index = 0;
            domain_distances[best_index] = distance(vec, dataset[node->childs[best_index]->pivot], veclen_);
            for (int i=1; i<branching_; ++i) {
                domain_distances[i] = distance(vec, dataset[node->childs[i]->pivot], veclen_);
                if (domain_distances[i]<domain_distances[best_index]) {
                    best_index = i;
                }
            }
            for (int i=0; i<branching_; ++i) {
                if (i!=best_index) {
                    heap->insert(BranchSt(node->childs[i],domain_distances[i]));
                }
            }
            delete[] domain_distances;
            findNN(node->childs[best_index],result,vec, checks, maxChecks, heap, checked);
        }
    }

private:


    /**
     * The dataset used by this index
     */
    const Matrix<ElementType> dataset;

    /**
     * Parameters used by this index
     */
    IndexParams params;


    /**
     * Number of features in the dataset.
     */
    size_t size_;

    /**
     * Length of each feature.
     */
    size_t veclen_;

    /**
     * The root node in the tree.
     */
    NodePtr* root;

    /**
     *  Array of indices to vectors in the dataset.
     */
    int** indices;


    /**
     * The distance
     */
    Distance distance;

    /**
     * Pooled memory allocator.
     *
     * Using a pooled memory allocator is more efficient
     * than allocating memory directly when there is a large
     * number small of memory allocations.
     */
    PooledAllocator pool;

    /**
     * Memory occupied by the index.
     */
    int memoryCounter;

    /** index parameters */
    int branching_;
    int trees_;
    flann_centers_init_t centers_init_;
    int leaf_size_;


};

}

#endif /* OPENCV_FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_ */
