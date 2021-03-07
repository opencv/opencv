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

#ifndef OPENCV_FLANN_KMEANS_INDEX_H_
#define OPENCV_FLANN_KMEANS_INDEX_H_

//! @cond IGNORED

#include <algorithm>
#include <map>
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
#include "logger.h"

#define BITS_PER_CHAR 8
#define BITS_PER_BASE 2 // for DNA/RNA sequences
#define BASE_PER_CHAR (BITS_PER_CHAR/BITS_PER_BASE)
#define HISTOS_PER_BASE (1<<BITS_PER_BASE)


namespace cvflann
{

struct KMeansIndexParams : public IndexParams
{
    KMeansIndexParams(int branching = 32, int iterations = 11,
                      flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
                      float cb_index = 0.2, int trees = 1 )
    {
        (*this)["algorithm"] = FLANN_INDEX_KMEANS;
        // branching factor
        (*this)["branching"] = branching;
        // max iterations to perform in one kmeans clustering (kmeans tree)
        (*this)["iterations"] = iterations;
        // algorithm used for picking the initial cluster centers for kmeans tree
        (*this)["centers_init"] = centers_init;
        // cluster boundary index. Used when searching the kmeans tree
        (*this)["cb_index"] = cb_index;
        // number of kmeans trees to search in
        (*this)["trees"] = trees;
    }
};


/**
 * Hierarchical kmeans index
 *
 * Contains a tree constructed through a hierarchical kmeans clustering
 * and other information for indexing a set of points for nearest-neighbour matching.
 */
template <typename Distance>
class KMeansIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    typedef typename Distance::CentersType CentersType;

    typedef typename Distance::is_kdtree_distance is_kdtree_distance;
    typedef typename Distance::is_vector_space_distance is_vector_space_distance;



    typedef void (KMeansIndex::* centersAlgFunction)(int, int*, int, int*, int&);

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
                    DistanceType sq = distance_(dataset_[centers[index]], dataset_[centers[j]], dataset_.cols);
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
        CV_DbgAssert(rnd >=0 && rnd < n);

        centers[0] = indices[rnd];

        int index;
        for (index=1; index<k; ++index) {

            int best_index = -1;
            DistanceType best_val = 0;
            for (int j=0; j<n; ++j) {
                DistanceType dist = distance_(dataset_[centers[0]],dataset_[indices[j]],dataset_.cols);
                for (int i=1; i<index; ++i) {
                    DistanceType tmp_dist = distance_(dataset_[centers[i]],dataset_[indices[j]],dataset_.cols);
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
        CV_DbgAssert(index >=0 && index < n);
        centers[0] = indices[index];

        for (int i = 0; i < n; i++) {
            closestDistSq[i] = distance_(dataset_[indices[i]], dataset_[indices[index]], dataset_.cols);
            closestDistSq[i] = ensureSquareDistance<Distance>( closestDistSq[i] );
            currentPot += closestDistSq[i];
        }


        const int numLocalTries = 1;

        // Choose each center
        int centerCount;
        for (centerCount = 1; centerCount < k; centerCount++) {

            // Repeat several trials
            double bestNewPot = -1;
            int bestNewIndex = -1;
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
                for (int i = 0; i < n; i++) {
                    DistanceType dist = distance_(dataset_[indices[i]], dataset_[indices[index]], dataset_.cols);
                    newPot += std::min( ensureSquareDistance<Distance>(dist), closestDistSq[i] );
                }

                // Store the best result
                if ((bestNewPot < 0)||(newPot < bestNewPot)) {
                    bestNewPot = newPot;
                    bestNewIndex = index;
                }
            }

            // Add the appropriate center
            centers[centerCount] = indices[bestNewIndex];
            currentPot = bestNewPot;
            for (int i = 0; i < n; i++) {
                DistanceType dist = distance_(dataset_[indices[i]], dataset_[indices[bestNewIndex]], dataset_.cols);
                closestDistSq[i] = std::min( ensureSquareDistance<Distance>(dist), closestDistSq[i] );
            }
        }

        centers_length = centerCount;

        delete[] closestDistSq;
    }



public:

    flann_algorithm_t getType() const CV_OVERRIDE
    {
        return FLANN_INDEX_KMEANS;
    }

    template<class CentersContainerType>
    class KMeansDistanceComputer : public cv::ParallelLoopBody
    {
    public:
        KMeansDistanceComputer(Distance _distance, const Matrix<ElementType>& _dataset,
            const int _branching, const int* _indices, const CentersContainerType& _dcenters,
            const size_t _veclen, std::vector<int> &_new_centroids,
            std::vector<DistanceType> &_sq_dists)
            : distance(_distance)
            , dataset(_dataset)
            , branching(_branching)
            , indices(_indices)
            , dcenters(_dcenters)
            , veclen(_veclen)
            , new_centroids(_new_centroids)
            , sq_dists(_sq_dists)
        {
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE
        {
            const int begin = range.start;
            const int end = range.end;

            for( int i = begin; i<end; ++i)
            {
                DistanceType sq_dist(distance(dataset[indices[i]], dcenters[0], veclen));
                int new_centroid(0);
                for (int j=1; j<branching; ++j) {
                    DistanceType new_sq_dist = distance(dataset[indices[i]], dcenters[j], veclen);
                    if (sq_dist>new_sq_dist) {
                        new_centroid = j;
                        sq_dist = new_sq_dist;
                    }
                }
                sq_dists[i] = sq_dist;
                new_centroids[i] = new_centroid;
            }
        }

    private:
        Distance distance;
        const Matrix<ElementType>& dataset;
        const int branching;
        const int* indices;
        const CentersContainerType& dcenters;
        const size_t veclen;
        std::vector<int> &new_centroids;
        std::vector<DistanceType> &sq_dists;
        KMeansDistanceComputer& operator=( const KMeansDistanceComputer & ) { return *this; }
    };

    /**
     * Index constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the hierarchical k-means algorithm
     */
    KMeansIndex(const Matrix<ElementType>& inputData, const IndexParams& params = KMeansIndexParams(),
                Distance d = Distance())
        : dataset_(inputData), index_params_(params), root_(NULL), indices_(NULL), distance_(d)
    {
        memoryCounter_ = 0;

        size_ = dataset_.rows;
        veclen_ = dataset_.cols;

        branching_ = get_param(params,"branching",32);
        trees_ = get_param(params,"trees",1);
        iterations_ = get_param(params,"iterations",11);
        if (iterations_<0) {
            iterations_ = (std::numeric_limits<int>::max)();
        }
        centers_init_  = get_param(params,"centers_init",FLANN_CENTERS_RANDOM);

        if (centers_init_==FLANN_CENTERS_RANDOM) {
            chooseCenters = &KMeansIndex::chooseCentersRandom;
        }
        else if (centers_init_==FLANN_CENTERS_GONZALES) {
            chooseCenters = &KMeansIndex::chooseCentersGonzales;
        }
        else if (centers_init_==FLANN_CENTERS_KMEANSPP) {
            chooseCenters = &KMeansIndex::chooseCentersKMeanspp;
        }
        else {
            FLANN_THROW(cv::Error::StsBadArg, "Unknown algorithm for choosing initial centers.");
        }
        cb_index_ = 0.4f;

        root_ = new KMeansNodePtr[trees_];
        indices_ = new int*[trees_];

        for (int i=0; i<trees_; ++i) {
            root_[i] = NULL;
            indices_[i] = NULL;
        }
    }


    KMeansIndex(const KMeansIndex&);
    KMeansIndex& operator=(const KMeansIndex&);


    /**
     * Index destructor.
     *
     * Release the memory used by the index.
     */
    virtual ~KMeansIndex()
    {
        if (root_ != NULL) {
            free_centers();
            delete[] root_;
        }
        if (indices_!=NULL) {
            free_indices();
            delete[] indices_;
        }
    }

    /**
     *  Returns size of index.
     */
    size_t size() const CV_OVERRIDE
    {
        return size_;
    }

    /**
     * Returns the length of an index feature.
     */
    size_t veclen() const CV_OVERRIDE
    {
        return veclen_;
    }


    void set_cb_index( float index)
    {
        cb_index_ = index;
    }

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const CV_OVERRIDE
    {
        return pool_.usedMemory+pool_.wastedMemory+memoryCounter_;
    }

    /**
     * Builds the index
     */
    void buildIndex() CV_OVERRIDE
    {
        if (branching_<2) {
            FLANN_THROW(cv::Error::StsError, "Branching factor must be at least 2");
        }

        free_indices();

        for (int i=0; i<trees_; ++i) {
            indices_[i] = new int[size_];
            for (size_t j=0; j<size_; ++j) {
                indices_[i][j] = int(j);
            }
            root_[i] = pool_.allocate<KMeansNode>();
            std::memset(root_[i], 0, sizeof(KMeansNode));

            Distance* dummy = NULL;
            computeNodeStatistics(root_[i], indices_[i], (unsigned int)size_, dummy);

            computeClustering(root_[i], indices_[i], (int)size_, branching_,0);
        }
    }


    void saveIndex(FILE* stream) CV_OVERRIDE
    {
        save_value(stream, branching_);
        save_value(stream, iterations_);
        save_value(stream, memoryCounter_);
        save_value(stream, cb_index_);
        save_value(stream, trees_);
        for (int i=0; i<trees_; ++i) {
            save_value(stream, *indices_[i], (int)size_);
            save_tree(stream, root_[i], i);
        }
    }


    void loadIndex(FILE* stream) CV_OVERRIDE
    {
        if (indices_!=NULL) {
            free_indices();
            delete[] indices_;
        }
        if (root_!=NULL) {
            free_centers();
        }

        load_value(stream, branching_);
        load_value(stream, iterations_);
        load_value(stream, memoryCounter_);
        load_value(stream, cb_index_);
        load_value(stream, trees_);

        indices_ = new int*[trees_];
        for (int i=0; i<trees_; ++i) {
            indices_[i] = new int[size_];
            load_value(stream, *indices_[i], size_);
            load_tree(stream, root_[i], i);
        }

        index_params_["algorithm"] = getType();
        index_params_["branching"] = branching_;
        index_params_["trees"] = trees_;
        index_params_["iterations"] = iterations_;
        index_params_["centers_init"] = centers_init_;
        index_params_["cb_index"] = cb_index_;
    }


    /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     searchParams = parameters that influence the search algorithm (checks, cb_index)
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) CV_OVERRIDE
    {

        const int maxChecks = get_param(searchParams,"checks",32);

        if (maxChecks==FLANN_CHECKS_UNLIMITED) {
            findExactNN(root_[0], result, vec);
        }
        else {
            // Priority queue storing intermediate branches in the best-bin-first search
            Heap<BranchSt>* heap = new Heap<BranchSt>((int)size_);

            int checks = 0;
            for (int i=0; i<trees_; ++i) {
                findNN(root_[i], result, vec, checks, maxChecks, heap);
                if ((checks >= maxChecks) && result.full())
                    break;
            }

            BranchSt branch;
            while (heap->popMin(branch) && (checks<maxChecks || !result.full())) {
                KMeansNodePtr node = branch.node;
                findNN(node, result, vec, checks, maxChecks, heap);
            }
            delete heap;

            CV_Assert(result.full());
        }
    }

    /**
     * Clustering function that takes a cut in the hierarchical k-means
     * tree and return the clusters centers of that clustering.
     * Params:
     *     numClusters = number of clusters to have in the clustering computed
     * Returns: number of cluster centers
     */
    int getClusterCenters(Matrix<CentersType>& centers)
    {
        int numClusters = centers.rows;
        if (numClusters<1) {
            FLANN_THROW(cv::Error::StsBadArg, "Number of clusters must be at least 1");
        }

        DistanceType variance;
        KMeansNodePtr* clusters = new KMeansNodePtr[numClusters];

        int clusterCount = getMinVarianceClusters(root_[0], clusters, numClusters, variance);

        Logger::info("Clusters requested: %d, returning %d\n",numClusters, clusterCount);

        for (int i=0; i<clusterCount; ++i) {
            CentersType* center = clusters[i]->pivot;
            for (size_t j=0; j<veclen_; ++j) {
                centers[i][j] = center[j];
            }
        }
        delete[] clusters;

        return clusterCount;
    }

    IndexParams getParameters() const CV_OVERRIDE
    {
        return index_params_;
    }


private:
    /**
     * Structure representing a node in the hierarchical k-means tree.
     */
    struct KMeansNode
    {
        /**
         * The cluster center.
         */
        CentersType* pivot;
        /**
         * The cluster radius.
         */
        DistanceType radius;
        /**
         * The cluster mean radius.
         */
        DistanceType mean_radius;
        /**
         * The cluster variance.
         */
        DistanceType variance;
        /**
         * The cluster size (number of points in the cluster)
         */
        int size;
        /**
         * Child nodes (only for non-terminal nodes)
         */
        KMeansNode** childs;
        /**
         * Node points (only for terminal nodes)
         */
        int* indices;
        /**
         * Level
         */
        int level;
    };
    typedef KMeansNode* KMeansNodePtr;

    /**
     * Alias definition for a nicer syntax.
     */
    typedef BranchStruct<KMeansNodePtr, DistanceType> BranchSt;




    void save_tree(FILE* stream, KMeansNodePtr node, int num)
    {
        save_value(stream, *node);
        save_value(stream, *(node->pivot), (int)veclen_);
        if (node->childs==NULL) {
            int indices_offset = (int)(node->indices - indices_[num]);
            save_value(stream, indices_offset);
        }
        else {
            for(int i=0; i<branching_; ++i) {
                save_tree(stream, node->childs[i], num);
            }
        }
    }


    void load_tree(FILE* stream, KMeansNodePtr& node, int num)
    {
        node = pool_.allocate<KMeansNode>();
        load_value(stream, *node);
        node->pivot = new CentersType[veclen_];
        load_value(stream, *(node->pivot), (int)veclen_);
        if (node->childs==NULL) {
            int indices_offset;
            load_value(stream, indices_offset);
            node->indices = indices_[num] + indices_offset;
        }
        else {
            node->childs = pool_.allocate<KMeansNodePtr>(branching_);
            for(int i=0; i<branching_; ++i) {
                load_tree(stream, node->childs[i], num);
            }
        }
    }


    /**
     * Helper function
     */
    void free_centers(KMeansNodePtr node)
    {
        delete[] node->pivot;
        if (node->childs!=NULL) {
            for (int k=0; k<branching_; ++k) {
                free_centers(node->childs[k]);
            }
        }
    }

    void free_centers()
    {
       if (root_ != NULL) {
           for(int i=0; i<trees_; ++i) {
               if (root_[i] != NULL) {
                   free_centers(root_[i]);
               }
           }
       }
    }

    /**
     * Release the inner elements of indices[]
     */
    void free_indices()
    {
        if (indices_!=NULL) {
            for(int i=0; i<trees_; ++i) {
                if (indices_[i]!=NULL) {
                    delete[] indices_[i];
                    indices_[i] = NULL;
                }
            }
        }
    }

    /**
     * Computes the statistics of a node (mean, radius, variance).
     *
     * Params:
     *     node = the node to use
     *     indices = array of indices of the points belonging to the node
     *     indices_length = number of indices in the array
     */
    void computeNodeStatistics(KMeansNodePtr node, int* indices, unsigned int indices_length)
    {
        DistanceType variance = 0;
        CentersType* mean = new CentersType[veclen_];
        memoryCounter_ += int(veclen_*sizeof(CentersType));

        memset(mean,0,veclen_*sizeof(CentersType));

        for (unsigned int i=0; i<indices_length; ++i) {
            ElementType* vec = dataset_[indices[i]];
            for (size_t j=0; j<veclen_; ++j) {
                mean[j] += vec[j];
            }
            variance += distance_(vec, ZeroIterator<ElementType>(), veclen_);
        }
        float length = static_cast<float>(indices_length);
        for (size_t j=0; j<veclen_; ++j) {
            mean[j] = cvflann::round<CentersType>( mean[j] / static_cast<double>(indices_length) );
        }
        variance /= static_cast<DistanceType>( length );
        variance -= distance_(mean, ZeroIterator<ElementType>(), veclen_);

        DistanceType radius = 0;
        for (unsigned int i=0; i<indices_length; ++i) {
            DistanceType tmp = distance_(mean, dataset_[indices[i]], veclen_);
            if (tmp>radius) {
                radius = tmp;
            }
        }

        node->variance = variance;
        node->radius = radius;
        node->pivot = mean;
    }


    void computeBitfieldNodeStatistics(KMeansNodePtr node, int* indices,
                                       unsigned int indices_length)
    {
        const unsigned int accumulator_veclen = static_cast<unsigned int>(
                                                veclen_*sizeof(CentersType)*BITS_PER_CHAR);

        unsigned long long variance = 0ull;
        CentersType* mean = new CentersType[veclen_];
        memoryCounter_ += int(veclen_*sizeof(CentersType));
        unsigned int* mean_accumulator = new unsigned int[accumulator_veclen];

        memset(mean_accumulator, 0, sizeof(unsigned int)*accumulator_veclen);

        for (unsigned int i=0; i<indices_length; ++i) {
            variance += static_cast<unsigned long long>( ensureSquareDistance<Distance>(
                        distance_(dataset_[indices[i]], ZeroIterator<ElementType>(), veclen_)));
            unsigned char* vec = (unsigned char*)dataset_[indices[i]];
            for (size_t k=0, l=0; k<accumulator_veclen; k+=BITS_PER_CHAR, ++l) {
                mean_accumulator[k]   += (vec[l])    & 0x01;
                mean_accumulator[k+1] += (vec[l]>>1) & 0x01;
                mean_accumulator[k+2] += (vec[l]>>2) & 0x01;
                mean_accumulator[k+3] += (vec[l]>>3) & 0x01;
                mean_accumulator[k+4] += (vec[l]>>4) & 0x01;
                mean_accumulator[k+5] += (vec[l]>>5) & 0x01;
                mean_accumulator[k+6] += (vec[l]>>6) & 0x01;
                mean_accumulator[k+7] += (vec[l]>>7) & 0x01;
            }
        }
        double cnt = static_cast<double>(indices_length);
        unsigned char* char_mean = (unsigned char*)mean;
        for (size_t k=0, l=0; k<accumulator_veclen; k+=BITS_PER_CHAR, ++l) {
            char_mean[l] = static_cast<unsigned char>(
                              (((int)(0.5 + (double)(mean_accumulator[k])   / cnt)))
                            | (((int)(0.5 + (double)(mean_accumulator[k+1]) / cnt))<<1)
                            | (((int)(0.5 + (double)(mean_accumulator[k+2]) / cnt))<<2)
                            | (((int)(0.5 + (double)(mean_accumulator[k+3]) / cnt))<<3)
                            | (((int)(0.5 + (double)(mean_accumulator[k+4]) / cnt))<<4)
                            | (((int)(0.5 + (double)(mean_accumulator[k+5]) / cnt))<<5)
                            | (((int)(0.5 + (double)(mean_accumulator[k+6]) / cnt))<<6)
                            | (((int)(0.5 + (double)(mean_accumulator[k+7]) / cnt))<<7));
        }
        variance = static_cast<unsigned long long>(
                    0.5 + static_cast<double>(variance) / static_cast<double>(indices_length));
        variance -= static_cast<unsigned long long>(
                    ensureSquareDistance<Distance>(
                        distance_(mean, ZeroIterator<ElementType>(), veclen_)));

        DistanceType radius = 0;
        for (unsigned int i=0; i<indices_length; ++i) {
            DistanceType tmp = distance_(mean, dataset_[indices[i]], veclen_);
            if (tmp>radius) {
                radius = tmp;
            }
        }

        node->variance = static_cast<DistanceType>(variance);
        node->radius = radius;
        node->pivot = mean;

        delete[] mean_accumulator;
    }


    void computeDnaNodeStatistics(KMeansNodePtr node, int* indices,
                                       unsigned int indices_length)
    {
        const unsigned int histos_veclen = static_cast<unsigned int>(
                    veclen_*sizeof(CentersType)*(HISTOS_PER_BASE*BASE_PER_CHAR));

        unsigned long long variance = 0ull;
        unsigned int* histograms = new unsigned int[histos_veclen];
        memset(histograms, 0, sizeof(unsigned int)*histos_veclen);

        for (unsigned int i=0; i<indices_length; ++i) {
            variance += static_cast<unsigned long long>( ensureSquareDistance<Distance>(
                        distance_(dataset_[indices[i]], ZeroIterator<ElementType>(), veclen_)));

            unsigned char* vec = (unsigned char*)dataset_[indices[i]];
            for (size_t k=0, l=0; k<histos_veclen; k+=HISTOS_PER_BASE*BASE_PER_CHAR, ++l) {
                histograms[k +     ((vec[l])    & 0x03)]++;
                histograms[k + 4 + ((vec[l]>>2) & 0x03)]++;
                histograms[k + 8 + ((vec[l]>>4) & 0x03)]++;
                histograms[k +12 + ((vec[l]>>6) & 0x03)]++;
            }
        }

        CentersType* mean = new CentersType[veclen_];
        memoryCounter_ += int(veclen_*sizeof(CentersType));
        unsigned char* char_mean = (unsigned char*)mean;
        unsigned int* h = histograms;
        for (size_t k=0, l=0; k<histos_veclen; k+=HISTOS_PER_BASE*BASE_PER_CHAR, ++l) {
            char_mean[l] = (h[k] > h[k+1] ? h[k+2] > h[k+3] ? h[k]   > h[k+2] ? 0x00 : 0x10
                                                            : h[k]   > h[k+3] ? 0x00 : 0x11
                                          : h[k+2] > h[k+3] ? h[k+1] > h[k+2] ? 0x01 : 0x10
                                                            : h[k+1] > h[k+3] ? 0x01 : 0x11)
                         | (h[k+4]>h[k+5] ? h[k+6] > h[k+7] ? h[k+4] > h[k+6] ? 0x00   : 0x1000
                                                            : h[k+4] > h[k+7] ? 0x00   : 0x1100
                                          : h[k+6] > h[k+7] ? h[k+5] > h[k+6] ? 0x0100 : 0x1000
                                                            : h[k+5] > h[k+7] ? 0x0100 : 0x1100)
                         | (h[k+8]>h[k+9] ? h[k+10]>h[k+11] ? h[k+8] >h[k+10] ? 0x00   : 0x100000
                                                            : h[k+8] >h[k+11] ? 0x00   : 0x110000
                                          : h[k+10]>h[k+11] ? h[k+9] >h[k+10] ? 0x010000 : 0x100000
                                                            : h[k+9] >h[k+11] ? 0x010000 : 0x110000)
                         | (h[k+12]>h[k+13] ? h[k+14]>h[k+15] ? h[k+12] >h[k+14] ? 0x00   : 0x10000000
                                                              : h[k+12] >h[k+15] ? 0x00   : 0x11000000
                                            : h[k+14]>h[k+15] ? h[k+13] >h[k+14] ? 0x01000000 : 0x10000000
                                                              : h[k+13] >h[k+15] ? 0x01000000 : 0x11000000);
        }
        variance = static_cast<unsigned long long>(
                    0.5 + static_cast<double>(variance) / static_cast<double>(indices_length));
        variance -= static_cast<unsigned long long>(
                    ensureSquareDistance<Distance>(
                        distance_(mean, ZeroIterator<ElementType>(), veclen_)));

        DistanceType radius = 0;
        for (unsigned int i=0; i<indices_length; ++i) {
            DistanceType tmp = distance_(mean, dataset_[indices[i]], veclen_);
            if (tmp>radius) {
                radius = tmp;
            }
        }

        node->variance = static_cast<DistanceType>(variance);
        node->radius = radius;
        node->pivot = mean;

        delete[] histograms;
    }


    template<typename DistType>
    void computeNodeStatistics(KMeansNodePtr node, int* indices,
                               unsigned int indices_length,
                               const DistType* identifier)
    {
        (void)identifier;
        computeNodeStatistics(node, indices, indices_length);
    }

    void computeNodeStatistics(KMeansNodePtr node, int* indices,
                               unsigned int indices_length,
                               const cvflann::HammingLUT* identifier)
    {
        (void)identifier;
        computeBitfieldNodeStatistics(node, indices, indices_length);
    }

    void computeNodeStatistics(KMeansNodePtr node, int* indices,
                               unsigned int indices_length,
                               const cvflann::Hamming<unsigned char>* identifier)
    {
        (void)identifier;
        computeBitfieldNodeStatistics(node, indices, indices_length);
    }

    void computeNodeStatistics(KMeansNodePtr node, int* indices,
                               unsigned int indices_length,
                               const cvflann::Hamming2<unsigned char>* identifier)
    {
        (void)identifier;
        computeBitfieldNodeStatistics(node, indices, indices_length);
    }

    void computeNodeStatistics(KMeansNodePtr node, int* indices,
                               unsigned int indices_length,
                               const cvflann::DNAmmingLUT* identifier)
    {
        (void)identifier;
        computeDnaNodeStatistics(node, indices, indices_length);
    }

    void computeNodeStatistics(KMeansNodePtr node, int* indices,
                               unsigned int indices_length,
                               const cvflann::DNAmming2<unsigned char>* identifier)
    {
        (void)identifier;
        computeDnaNodeStatistics(node, indices, indices_length);
    }


    void refineClustering(int* indices, int indices_length, int branching, CentersType** centers,
                          std::vector<DistanceType>& radiuses, int* belongs_to, int* count)
    {
        cv::AutoBuffer<double> dcenters_buf(branching*veclen_);
        Matrix<double> dcenters(dcenters_buf.data(), branching, veclen_);

        bool converged = false;
        int iteration = 0;
        while (!converged && iteration<iterations_) {
            converged = true;
            iteration++;

            // compute the new cluster centers
            for (int i=0; i<branching; ++i) {
                memset(dcenters[i],0,sizeof(double)*veclen_);
                radiuses[i] = 0;
            }
            for (int i=0; i<indices_length; ++i) {
                ElementType* vec = dataset_[indices[i]];
                double* center = dcenters[belongs_to[i]];
                for (size_t k=0; k<veclen_; ++k) {
                    center[k] += vec[k];
                }
            }
            for (int i=0; i<branching; ++i) {
                int cnt = count[i];
                for (size_t k=0; k<veclen_; ++k) {
                    dcenters[i][k] /= cnt;
                }
            }

            std::vector<int> new_centroids(indices_length);
            std::vector<DistanceType> sq_dists(indices_length);

            // reassign points to clusters
            KMeansDistanceComputer<Matrix<double> > invoker(
                        distance_, dataset_, branching, indices, dcenters, veclen_, new_centroids, sq_dists);
            parallel_for_(cv::Range(0, (int)indices_length), invoker);

            for (int i=0; i < (int)indices_length; ++i) {
                DistanceType sq_dist(sq_dists[i]);
                int new_centroid(new_centroids[i]);
                if (sq_dist > radiuses[new_centroid]) {
                    radiuses[new_centroid] = sq_dist;
                }
                if (new_centroid != belongs_to[i]) {
                    count[belongs_to[i]]--;
                    count[new_centroid]++;
                    belongs_to[i] = new_centroid;
                    converged = false;
                }
            }

            for (int i=0; i<branching; ++i) {
                // if one cluster converges to an empty cluster,
                // move an element into that cluster
                if (count[i]==0) {
                    int j = (i+1)%branching;
                    while (count[j]<=1) {
                        j = (j+1)%branching;
                    }

                    for (int k=0; k<indices_length; ++k) {
                        if (belongs_to[k]==j) {
                            // for cluster j, we move the furthest element from the center to the empty cluster i
                            if ( distance_(dataset_[indices[k]], dcenters[j], veclen_) == radiuses[j] ) {
                                belongs_to[k] = i;
                                count[j]--;
                                count[i]++;
                                break;
                            }
                        }
                    }
                    converged = false;
                }
            }
        }

       for (int i=0; i<branching; ++i) {
           centers[i] = new CentersType[veclen_];
           memoryCounter_ += (int)(veclen_*sizeof(CentersType));
           for (size_t k=0; k<veclen_; ++k) {
               centers[i][k] = (CentersType)dcenters[i][k];
           }
       }
    }


    void refineBitfieldClustering(int* indices, int indices_length, int branching, CentersType** centers,
                                  std::vector<DistanceType>& radiuses, int* belongs_to, int* count)
    {
        for (int i=0; i<branching; ++i) {
            centers[i] = new CentersType[veclen_];
            memoryCounter_ += (int)(veclen_*sizeof(CentersType));
        }

        const unsigned int accumulator_veclen = static_cast<unsigned int>(
                                                veclen_*sizeof(ElementType)*BITS_PER_CHAR);
        cv::AutoBuffer<unsigned int> dcenters_buf(branching*accumulator_veclen);
        Matrix<unsigned int> dcenters(dcenters_buf.data(), branching, accumulator_veclen);

        bool converged = false;
        int iteration = 0;
        while (!converged && iteration<iterations_) {
            converged = true;
            iteration++;

            // compute the new cluster centers
            for (int i=0; i<branching; ++i) {
                memset(dcenters[i],0,sizeof(unsigned int)*accumulator_veclen);
                radiuses[i] = 0;
            }
            for (int i=0; i<indices_length; ++i) {
                unsigned char* vec = (unsigned char*)dataset_[indices[i]];
                unsigned int* dcenter = dcenters[belongs_to[i]];
                for (size_t k=0, l=0; k<accumulator_veclen; k+=BITS_PER_CHAR, ++l) {
                    dcenter[k]   += (vec[l])    & 0x01;
                    dcenter[k+1] += (vec[l]>>1) & 0x01;
                    dcenter[k+2] += (vec[l]>>2) & 0x01;
                    dcenter[k+3] += (vec[l]>>3) & 0x01;
                    dcenter[k+4] += (vec[l]>>4) & 0x01;
                    dcenter[k+5] += (vec[l]>>5) & 0x01;
                    dcenter[k+6] += (vec[l]>>6) & 0x01;
                    dcenter[k+7] += (vec[l]>>7) & 0x01;
                }
            }
            for (int i=0; i<branching; ++i) {
                double cnt = static_cast<double>(count[i]);
                unsigned int* dcenter = dcenters[i];
                unsigned char* charCenter = (unsigned char*)centers[i];
                for (size_t k=0, l=0; k<accumulator_veclen; k+=BITS_PER_CHAR, ++l) {
                    charCenter[l] = static_cast<unsigned char>(
                                      (((int)(0.5 + (double)(dcenter[k])   / cnt)))
                                    | (((int)(0.5 + (double)(dcenter[k+1]) / cnt))<<1)
                                    | (((int)(0.5 + (double)(dcenter[k+2]) / cnt))<<2)
                                    | (((int)(0.5 + (double)(dcenter[k+3]) / cnt))<<3)
                                    | (((int)(0.5 + (double)(dcenter[k+4]) / cnt))<<4)
                                    | (((int)(0.5 + (double)(dcenter[k+5]) / cnt))<<5)
                                    | (((int)(0.5 + (double)(dcenter[k+6]) / cnt))<<6)
                                    | (((int)(0.5 + (double)(dcenter[k+7]) / cnt))<<7));
                }
            }

            std::vector<int> new_centroids(indices_length);
            std::vector<DistanceType> dists(indices_length);

            // reassign points to clusters
            KMeansDistanceComputer<ElementType**> invoker(
                        distance_, dataset_, branching, indices, centers, veclen_, new_centroids, dists);
            parallel_for_(cv::Range(0, (int)indices_length), invoker);

            for (int i=0; i < indices_length; ++i) {
                DistanceType dist(dists[i]);
                int new_centroid(new_centroids[i]);
                if (dist > radiuses[new_centroid]) {
                    radiuses[new_centroid] = dist;
                }
                if (new_centroid != belongs_to[i]) {
                    count[belongs_to[i]]--;
                    count[new_centroid]++;
                    belongs_to[i] = new_centroid;
                    converged = false;
                }
            }

            for (int i=0; i<branching; ++i) {
                // if one cluster converges to an empty cluster,
                // move an element into that cluster
                if (count[i]==0) {
                    int j = (i+1)%branching;
                    while (count[j]<=1) {
                        j = (j+1)%branching;
                    }

                    for (int k=0; k<indices_length; ++k) {
                        if (belongs_to[k]==j) {
                            // for cluster j, we move the furthest element from the center to the empty cluster i
                            if ( distance_(dataset_[indices[k]], centers[j], veclen_) == radiuses[j] ) {
                                belongs_to[k] = i;
                                count[j]--;
                                count[i]++;
                                break;
                            }
                        }
                    }
                    converged = false;
                }
            }
        }
    }


    void refineDnaClustering(int* indices, int indices_length, int branching, CentersType** centers,
                                  std::vector<DistanceType>& radiuses, int* belongs_to, int* count)
    {
        for (int i=0; i<branching; ++i) {
            centers[i] = new CentersType[veclen_];
            memoryCounter_ += (int)(veclen_*sizeof(CentersType));
        }

        const unsigned int histos_veclen = static_cast<unsigned int>(
                    veclen_*sizeof(CentersType)*(HISTOS_PER_BASE*BASE_PER_CHAR));
        cv::AutoBuffer<unsigned int> histos_buf(branching*histos_veclen);
        Matrix<unsigned int> histos(histos_buf.data(), branching, histos_veclen);

        bool converged = false;
        int iteration = 0;
        while (!converged && iteration<iterations_) {
            converged = true;
            iteration++;

            // compute the new cluster centers
            for (int i=0; i<branching; ++i) {
                memset(histos[i],0,sizeof(unsigned int)*histos_veclen);
                radiuses[i] = 0;
            }
            for (int i=0; i<indices_length; ++i) {
                unsigned char* vec = (unsigned char*)dataset_[indices[i]];
                unsigned int* h = histos[belongs_to[i]];
                for (size_t k=0, l=0; k<histos_veclen; k+=HISTOS_PER_BASE*BASE_PER_CHAR, ++l) {
                    h[k +     ((vec[l])    & 0x03)]++;
                    h[k + 4 + ((vec[l]>>2) & 0x03)]++;
                    h[k + 8 + ((vec[l]>>4) & 0x03)]++;
                    h[k +12 + ((vec[l]>>6) & 0x03)]++;
                }
            }
            for (int i=0; i<branching; ++i) {
                unsigned int* h = histos[i];
                unsigned char* charCenter = (unsigned char*)centers[i];
                for (size_t k=0, l=0; k<histos_veclen; k+=HISTOS_PER_BASE*BASE_PER_CHAR, ++l) {
                    charCenter[l]= (h[k] > h[k+1] ? h[k+2] > h[k+3] ? h[k]   > h[k+2] ? 0x00 : 0x10
                                                                    : h[k]   > h[k+3] ? 0x00 : 0x11
                                                  : h[k+2] > h[k+3] ? h[k+1] > h[k+2] ? 0x01 : 0x10
                                                                    : h[k+1] > h[k+3] ? 0x01 : 0x11)
                                 | (h[k+4]>h[k+5] ? h[k+6] > h[k+7] ? h[k+4] > h[k+6] ? 0x00   : 0x1000
                                                                    : h[k+4] > h[k+7] ? 0x00   : 0x1100
                                                  : h[k+6] > h[k+7] ? h[k+5] > h[k+6] ? 0x0100 : 0x1000
                                                                    : h[k+5] > h[k+7] ? 0x0100 : 0x1100)
                                 | (h[k+8]>h[k+9] ? h[k+10]>h[k+11] ? h[k+8] >h[k+10] ? 0x00   : 0x100000
                                                                    : h[k+8] >h[k+11] ? 0x00   : 0x110000
                                                  : h[k+10]>h[k+11] ? h[k+9] >h[k+10] ? 0x010000 : 0x100000
                                                                    : h[k+9] >h[k+11] ? 0x010000 : 0x110000)
                                 | (h[k+12]>h[k+13] ? h[k+14]>h[k+15] ? h[k+12] >h[k+14] ? 0x00   : 0x10000000
                                                                      : h[k+12] >h[k+15] ? 0x00   : 0x11000000
                                                    : h[k+14]>h[k+15] ? h[k+13] >h[k+14] ? 0x01000000 : 0x10000000
                                                                      : h[k+13] >h[k+15] ? 0x01000000 : 0x11000000);
                }
            }

            std::vector<int> new_centroids(indices_length);
            std::vector<DistanceType> dists(indices_length);

            // reassign points to clusters
            KMeansDistanceComputer<ElementType**> invoker(
                        distance_, dataset_, branching, indices, centers, veclen_, new_centroids, dists);
            parallel_for_(cv::Range(0, (int)indices_length), invoker);

            for (int i=0; i < indices_length; ++i) {
                DistanceType dist(dists[i]);
                int new_centroid(new_centroids[i]);
                if (dist > radiuses[new_centroid]) {
                    radiuses[new_centroid] = dist;
                }
                if (new_centroid != belongs_to[i]) {
                    count[belongs_to[i]]--;
                    count[new_centroid]++;
                    belongs_to[i] = new_centroid;
                    converged = false;
                }
            }

            for (int i=0; i<branching; ++i) {
                // if one cluster converges to an empty cluster,
                // move an element into that cluster
                if (count[i]==0) {
                    int j = (i+1)%branching;
                    while (count[j]<=1) {
                        j = (j+1)%branching;
                    }

                    for (int k=0; k<indices_length; ++k) {
                        if (belongs_to[k]==j) {
                            // for cluster j, we move the furthest element from the center to the empty cluster i
                            if ( distance_(dataset_[indices[k]], centers[j], veclen_) == radiuses[j] ) {
                                belongs_to[k] = i;
                                count[j]--;
                                count[i]++;
                                break;
                            }
                        }
                    }
                    converged = false;
                }
            }
        }
    }


    void computeSubClustering(KMeansNodePtr node, int* indices, int indices_length,
                              int branching, int level, CentersType** centers,
                              std::vector<DistanceType>& radiuses, int* belongs_to, int* count)
    {
        // compute kmeans clustering for each of the resulting clusters
        node->childs = pool_.allocate<KMeansNodePtr>(branching);
        int start = 0;
        int end = start;
        for (int c=0; c<branching; ++c) {
            int s = count[c];

            DistanceType variance = 0;
            DistanceType mean_radius =0;
            for (int i=0; i<indices_length; ++i) {
                if (belongs_to[i]==c) {
                    DistanceType d = distance_(dataset_[indices[i]], ZeroIterator<ElementType>(), veclen_);
                    variance += d;
                    mean_radius += static_cast<DistanceType>( sqrt(d) );
                    std::swap(indices[i],indices[end]);
                    std::swap(belongs_to[i],belongs_to[end]);
                    end++;
                }
            }
            variance /= s;
            mean_radius /= s;
            variance -= distance_(centers[c], ZeroIterator<ElementType>(), veclen_);

            node->childs[c] = pool_.allocate<KMeansNode>();
            std::memset(node->childs[c], 0, sizeof(KMeansNode));
            node->childs[c]->radius = radiuses[c];
            node->childs[c]->pivot = centers[c];
            node->childs[c]->variance = variance;
            node->childs[c]->mean_radius = mean_radius;
            computeClustering(node->childs[c],indices+start, end-start, branching, level+1);
            start=end;
        }
    }


    void computeAnyBitfieldSubClustering(KMeansNodePtr node, int* indices, int indices_length,
                              int branching, int level, CentersType** centers,
                              std::vector<DistanceType>& radiuses, int* belongs_to, int* count)
    {
        // compute kmeans clustering for each of the resulting clusters
        node->childs = pool_.allocate<KMeansNodePtr>(branching);
        int start = 0;
        int end = start;
        for (int c=0; c<branching; ++c) {
            int s = count[c];

            unsigned long long variance = 0ull;
            DistanceType mean_radius =0;
            for (int i=0; i<indices_length; ++i) {
                if (belongs_to[i]==c) {
                    DistanceType d = distance_(dataset_[indices[i]], ZeroIterator<ElementType>(), veclen_);
                    variance += static_cast<unsigned long long>( ensureSquareDistance<Distance>(d) );
                    mean_radius += ensureSimpleDistance<Distance>(d);
                    std::swap(indices[i],indices[end]);
                    std::swap(belongs_to[i],belongs_to[end]);
                    end++;
                }
            }
            mean_radius = static_cast<DistanceType>(
                        0.5f + static_cast<float>(mean_radius) / static_cast<float>(s));
            variance = static_cast<unsigned long long>(
                        0.5 + static_cast<double>(variance) / static_cast<double>(s));
            variance -= static_cast<unsigned long long>(
                        ensureSquareDistance<Distance>(
                            distance_(centers[c], ZeroIterator<ElementType>(), veclen_)));

            node->childs[c] = pool_.allocate<KMeansNode>();
            std::memset(node->childs[c], 0, sizeof(KMeansNode));
            node->childs[c]->radius = radiuses[c];
            node->childs[c]->pivot = centers[c];
            node->childs[c]->variance = static_cast<DistanceType>(variance);
            node->childs[c]->mean_radius = mean_radius;
            computeClustering(node->childs[c],indices+start, end-start, branching, level+1);
            start=end;
        }
    }


    template<typename DistType>
    void refineAndSplitClustering(
            KMeansNodePtr node, int* indices, int indices_length, int branching,
            int level, CentersType** centers, std::vector<DistanceType>& radiuses,
            int* belongs_to, int* count, const DistType* identifier)
    {
        (void)identifier;
        refineClustering(indices, indices_length, branching, centers, radiuses, belongs_to, count);

        computeSubClustering(node, indices, indices_length, branching,
                             level, centers, radiuses, belongs_to, count);
    }


    /**
     * The methods responsible with doing the recursive hierarchical clustering on
     * binary vectors.
     * As some might have heard that KMeans on binary data doesn't make sense,
     * it's worth a little explanation why it actually fairly works. As
     * with the Hierarchical Clustering algortihm, we seed several centers for the
     * current node by picking some of its points. Then in a first pass each point
     * of the node is then related to its closest center. Now let's have a look at
     * the 5 central dimensions of the 9 following points:
     *
     * xxxxxx11100xxxxx (1)
     * xxxxxx11010xxxxx (2)
     * xxxxxx11001xxxxx (3)
     * xxxxxx10110xxxxx (4)
     * xxxxxx10101xxxxx (5)
     * xxxxxx10011xxxxx (6)
     * xxxxxx01110xxxxx (7)
     * xxxxxx01101xxxxx (8)
     * xxxxxx01011xxxxx (9)
     * sum   _____
     * of 1: 66555
     *
     * Even if the barycenter notion doesn't apply, we can set a center
     * xxxxxx11111xxxxx that will better fit the five dimensions we are focusing
     * on for these points.
     *
     * Note that convergence isn't ensured anymore. In practice, using Gonzales
     * as seeding algorithm should be fine for getting convergence ("iterations"
     * value can be set to -1). But with KMeans++ seeding you should definitely
     * set a maximum number of iterations (but make it higher than the "iterations"
     * default value of 11).
     *
     * Params:
     *     node = the node to cluster
     *     indices = indices of the points belonging to the current node
     *     indices_length = number of points in the current node
     *     branching = the branching factor to use in the clustering
     *     level = 0 for the root node, it increases with the subdivision levels
     *     centers = clusters centers to compute
     *     radiuses = radiuses of clusters
     *     belongs_to = LookUp Table returning, for a given indice id, the center id it belongs to
     *     count = array storing the number of indices for a given center id
     *     identifier = dummy pointer on an instance of Distance (use to branch correctly among templates)
     */
    void refineAndSplitClustering(
            KMeansNodePtr node, int* indices, int indices_length, int branching,
            int level, CentersType** centers, std::vector<DistanceType>& radiuses,
            int* belongs_to, int* count, const cvflann::HammingLUT* identifier)
    {
        (void)identifier;
        refineBitfieldClustering(
                    indices, indices_length, branching, centers, radiuses, belongs_to, count);

        computeAnyBitfieldSubClustering(node, indices, indices_length, branching,
                                        level, centers, radiuses, belongs_to, count);
    }


    void refineAndSplitClustering(
            KMeansNodePtr node, int* indices, int indices_length, int branching,
            int level, CentersType** centers, std::vector<DistanceType>& radiuses,
            int* belongs_to, int* count, const cvflann::Hamming<unsigned char>* identifier)
    {
        (void)identifier;
        refineBitfieldClustering(
                    indices, indices_length, branching, centers, radiuses, belongs_to, count);

        computeAnyBitfieldSubClustering(node, indices, indices_length, branching,
                                        level, centers, radiuses, belongs_to, count);
    }


    void refineAndSplitClustering(
            KMeansNodePtr node, int* indices, int indices_length, int branching,
            int level, CentersType** centers, std::vector<DistanceType>& radiuses,
            int* belongs_to, int* count, const cvflann::Hamming2<unsigned char>* identifier)
    {
        (void)identifier;
        refineBitfieldClustering(
                    indices, indices_length, branching, centers, radiuses, belongs_to, count);

        computeAnyBitfieldSubClustering(node, indices, indices_length, branching,
                                        level, centers, radiuses, belongs_to, count);
    }


    void refineAndSplitClustering(
            KMeansNodePtr node, int* indices, int indices_length, int branching,
            int level, CentersType** centers, std::vector<DistanceType>& radiuses,
            int* belongs_to, int* count, const cvflann::DNAmmingLUT* identifier)
    {
        (void)identifier;
        refineDnaClustering(
                    indices, indices_length, branching, centers, radiuses, belongs_to, count);

        computeAnyBitfieldSubClustering(node, indices, indices_length, branching,
                                        level, centers, radiuses, belongs_to, count);
    }


    void refineAndSplitClustering(
            KMeansNodePtr node, int* indices, int indices_length, int branching,
            int level, CentersType** centers, std::vector<DistanceType>& radiuses,
            int* belongs_to, int* count, const cvflann::DNAmming2<unsigned char>* identifier)
    {
        (void)identifier;
        refineDnaClustering(
                    indices, indices_length, branching, centers, radiuses, belongs_to, count);

        computeAnyBitfieldSubClustering(node, indices, indices_length, branching,
                                        level, centers, radiuses, belongs_to, count);
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
    void computeClustering(KMeansNodePtr node, int* indices, int indices_length, int branching, int level)
    {
        node->size = indices_length;
        node->level = level;

        if (indices_length < branching) {
            node->indices = indices;
            std::sort(node->indices,node->indices+indices_length);
            node->childs = NULL;
            return;
        }

        cv::AutoBuffer<int> centers_idx_buf(branching);
        int* centers_idx = centers_idx_buf.data();
        int centers_length;
        (this->*chooseCenters)(branching, indices, indices_length, centers_idx, centers_length);

        if (centers_length<branching) {
            node->indices = indices;
            std::sort(node->indices,node->indices+indices_length);
            node->childs = NULL;
            return;
        }


        std::vector<DistanceType> radiuses(branching);
        cv::AutoBuffer<int> count_buf(branching);
        int* count = count_buf.data();
        for (int i=0; i<branching; ++i) {
            radiuses[i] = 0;
            count[i] = 0;
        }

        //	assign points to clusters
        cv::AutoBuffer<int> belongs_to_buf(indices_length);
        int* belongs_to = belongs_to_buf.data();
        for (int i=0; i<indices_length; ++i) {
            DistanceType sq_dist = distance_(dataset_[indices[i]], dataset_[centers_idx[0]], veclen_);
            belongs_to[i] = 0;
            for (int j=1; j<branching; ++j) {
                DistanceType new_sq_dist = distance_(dataset_[indices[i]], dataset_[centers_idx[j]], veclen_);
                if (sq_dist>new_sq_dist) {
                    belongs_to[i] = j;
                    sq_dist = new_sq_dist;
                }
            }
            if (sq_dist>radiuses[belongs_to[i]]) {
                radiuses[belongs_to[i]] = sq_dist;
            }
            count[belongs_to[i]]++;
        }

        CentersType** centers = new CentersType*[branching];

        Distance* dummy = NULL;
        refineAndSplitClustering(node, indices, indices_length, branching, level,
                                 centers, radiuses, belongs_to, count, dummy);

        delete[] centers;
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


    void findNN(KMeansNodePtr node, ResultSet<DistanceType>& result, const ElementType* vec, int& checks, int maxChecks,
                Heap<BranchSt>* heap)
    {
        // Ignore those clusters that are too far away
        {
            DistanceType bsq = distance_(vec, node->pivot, veclen_);
            DistanceType rsq = node->radius;
            DistanceType wsq = result.worstDist();

            if (isSquareDistance<Distance>())
            {
                DistanceType val = bsq-rsq-wsq;
                if ((val>0) && (val*val > 4*rsq*wsq))
                    return;
            }
            else
            {
                if (bsq-rsq > wsq)
                    return;
            }
        }

        if (node->childs==NULL) {
            if ((checks>=maxChecks) && result.full()) {
                return;
            }
            checks += node->size;
            for (int i=0; i<node->size; ++i) {
                int index = node->indices[i];
                DistanceType dist = distance_(dataset_[index], vec, veclen_);
                result.addPoint(dist, index);
            }
        }
        else {
            DistanceType* domain_distances = new DistanceType[branching_];
            int closest_center = exploreNodeBranches(node, vec, domain_distances, heap);
            delete[] domain_distances;
            findNN(node->childs[closest_center],result,vec, checks, maxChecks, heap);
        }
    }

    /**
     * Helper function that computes the nearest childs of a node to a given query point.
     * Params:
     *     node = the node
     *     q = the query point
     *     distances = array with the distances to each child node.
     * Returns:
     */
    int exploreNodeBranches(KMeansNodePtr node, const ElementType* q, DistanceType* domain_distances, Heap<BranchSt>* heap)
    {

        int best_index = 0;
        domain_distances[best_index] = distance_(q, node->childs[best_index]->pivot, veclen_);
        for (int i=1; i<branching_; ++i) {
            domain_distances[i] = distance_(q, node->childs[i]->pivot, veclen_);
            if (domain_distances[i]<domain_distances[best_index]) {
                best_index = i;
            }
        }

        //		float* best_center = node->childs[best_index]->pivot;
        for (int i=0; i<branching_; ++i) {
            if (i != best_index) {
                domain_distances[i] -= cvflann::round<DistanceType>(
                                        cb_index_*node->childs[i]->variance );

                //				float dist_to_border = getDistanceToBorder(node.childs[i].pivot,best_center,q);
                //				if (domain_distances[i]<dist_to_border) {
                //					domain_distances[i] = dist_to_border;
                //				}
                heap->insert(BranchSt(node->childs[i],domain_distances[i]));
            }
        }

        return best_index;
    }


    /**
     * Function the performs exact nearest neighbor search by traversing the entire tree.
     */
    void findExactNN(KMeansNodePtr node, ResultSet<DistanceType>& result, const ElementType* vec)
    {
        // Ignore those clusters that are too far away
        {
            DistanceType bsq = distance_(vec, node->pivot, veclen_);
            DistanceType rsq = node->radius;
            DistanceType wsq = result.worstDist();

            if (isSquareDistance<Distance>())
            {
                DistanceType val = bsq-rsq-wsq;
                if ((val>0) && (val*val > 4*rsq*wsq))
                    return;
            }
            else
            {
                if (bsq-rsq > wsq)
                    return;
            }
        }


        if (node->childs==NULL) {
            for (int i=0; i<node->size; ++i) {
                int index = node->indices[i];
                DistanceType dist = distance_(dataset_[index], vec, veclen_);
                result.addPoint(dist, index);
            }
        }
        else {
            int* sort_indices = new int[branching_];

            getCenterOrdering(node, vec, sort_indices);

            for (int i=0; i<branching_; ++i) {
                findExactNN(node->childs[sort_indices[i]],result,vec);
            }

            delete[] sort_indices;
        }
    }


    /**
     * Helper function.
     *
     * I computes the order in which to traverse the child nodes of a particular node.
     */
    void getCenterOrdering(KMeansNodePtr node, const ElementType* q, int* sort_indices)
    {
        DistanceType* domain_distances = new DistanceType[branching_];
        for (int i=0; i<branching_; ++i) {
            DistanceType dist = distance_(q, node->childs[i]->pivot, veclen_);

            int j=0;
            while (domain_distances[j]<dist && j<i)
                j++;
            for (int k=i; k>j; --k) {
                domain_distances[k] = domain_distances[k-1];
                sort_indices[k] = sort_indices[k-1];
            }
            domain_distances[j] = dist;
            sort_indices[j] = i;
        }
        delete[] domain_distances;
    }

    /**
     * Method that computes the squared distance from the query point q
     * from inside region with center c to the border between this
     * region and the region with center p
     */
    DistanceType getDistanceToBorder(DistanceType* p, DistanceType* c, DistanceType* q)
    {
        DistanceType sum = 0;
        DistanceType sum2 = 0;

        for (int i=0; i<veclen_; ++i) {
            DistanceType t = c[i]-p[i];
            sum += t*(q[i]-(c[i]+p[i])/2);
            sum2 += t*t;
        }

        return sum*sum/sum2;
    }


    /**
     * Helper function the descends in the hierarchical k-means tree by splitting those clusters that minimize
     * the overall variance of the clustering.
     * Params:
     *     root = root node
     *     clusters = array with clusters centers (return value)
     *     varianceValue = variance of the clustering (return value)
     * Returns:
     */
    int getMinVarianceClusters(KMeansNodePtr root, KMeansNodePtr* clusters, int clusters_length, DistanceType& varianceValue)
    {
        int clusterCount = 1;
        clusters[0] = root;

        DistanceType meanVariance = root->variance*root->size;

        while (clusterCount<clusters_length) {
            DistanceType minVariance = (std::numeric_limits<DistanceType>::max)();
            int splitIndex = -1;

            for (int i=0; i<clusterCount; ++i) {
                if (clusters[i]->childs != NULL) {

                    DistanceType variance = meanVariance - clusters[i]->variance*clusters[i]->size;

                    for (int j=0; j<branching_; ++j) {
                        variance += clusters[i]->childs[j]->variance*clusters[i]->childs[j]->size;
                    }
                    if (variance<minVariance) {
                        minVariance = variance;
                        splitIndex = i;
                    }
                }
            }

            if (splitIndex==-1) break;
            if ( (branching_+clusterCount-1) > clusters_length) break;

            meanVariance = minVariance;

            // split node
            KMeansNodePtr toSplit = clusters[splitIndex];
            clusters[splitIndex] = toSplit->childs[0];
            for (int i=1; i<branching_; ++i) {
                clusters[clusterCount++] = toSplit->childs[i];
            }
        }

        varianceValue = meanVariance/root->size;
        return clusterCount;
    }

private:
    /** The branching factor used in the hierarchical k-means clustering */
    int branching_;

    /** Number of kmeans trees (default is one) */
    int trees_;

    /** Maximum number of iterations to use when performing k-means clustering */
    int iterations_;

    /** Algorithm for choosing the cluster centers */
    flann_centers_init_t centers_init_;

    /**
     * Cluster border index. This is used in the tree search phase when determining
     * the closest cluster to explore next. A zero value takes into account only
     * the cluster centres, a value greater then zero also take into account the size
     * of the cluster.
     */
    float cb_index_;

    /**
     * The dataset used by this index
     */
    const Matrix<ElementType> dataset_;

    /** Index parameters */
    IndexParams index_params_;

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
    KMeansNodePtr* root_;

    /**
     *  Array of indices to vectors in the dataset.
     */
    int** indices_;

    /**
     * The distance
     */
    Distance distance_;

    /**
     * Pooled memory allocator.
     */
    PooledAllocator pool_;

    /**
     * Memory occupied by the index.
     */
    int memoryCounter_;
};

}

//! @endcond

#endif //OPENCV_FLANN_KMEANS_INDEX_H_
