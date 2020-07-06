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

#ifndef OPENCV_FLANN_KDTREE_INDEX_H_
#define OPENCV_FLANN_KDTREE_INDEX_H_

//! @cond IGNORED

#include <algorithm>
#include <map>
#include <cstring>

#include "general.h"
#include "nn_index.h"
#include "dynamic_bitset.h"
#include "matrix.h"
#include "result_set.h"
#include "heap.h"
#include "allocator.h"
#include "random.h"
#include "saving.h"


namespace cvflann
{

struct KDTreeIndexParams : public IndexParams
{
    KDTreeIndexParams(int trees = 4)
    {
        (*this)["algorithm"] = FLANN_INDEX_KDTREE;
        (*this)["trees"] = trees;
    }
};


/**
 * Randomized kd-tree index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 */
template <typename Distance>
class KDTreeIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;


    /**
     * KDTree constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the kdtree algorithm
     */
    KDTreeIndex(const Matrix<ElementType>& inputData, const IndexParams& params = KDTreeIndexParams(),
                Distance d = Distance() ) :
        dataset_(inputData), index_params_(params), distance_(d)
    {
        size_ = dataset_.rows;
        veclen_ = dataset_.cols;

        trees_ = get_param(index_params_,"trees",4);
        tree_roots_ = new NodePtr[trees_];

        // Create a permutable array of indices to the input vectors.
        vind_.resize(size_);
        for (size_t i = 0; i < size_; ++i) {
            vind_[i] = int(i);
        }

        mean_ = new DistanceType[veclen_];
        var_ = new DistanceType[veclen_];
    }


    KDTreeIndex(const KDTreeIndex&);
    KDTreeIndex& operator=(const KDTreeIndex&);

    /**
     * Standard destructor
     */
    ~KDTreeIndex()
    {
        if (tree_roots_!=NULL) {
            delete[] tree_roots_;
        }
        delete[] mean_;
        delete[] var_;
    }

    /**
     * Builds the index
     */
    void buildIndex() CV_OVERRIDE
    {
        /* Construct the randomized trees. */
        for (int i = 0; i < trees_; i++) {
            /* Randomize the order of vectors to allow for unbiased sampling. */
#ifndef OPENCV_FLANN_USE_STD_RAND
            cv::randShuffle(vind_);
#else
            std::random_shuffle(vind_.begin(), vind_.end());
#endif

            tree_roots_[i] = divideTree(&vind_[0], int(size_) );
        }
    }


    flann_algorithm_t getType() const CV_OVERRIDE
    {
        return FLANN_INDEX_KDTREE;
    }


    void saveIndex(FILE* stream) CV_OVERRIDE
    {
        save_value(stream, trees_);
        for (int i=0; i<trees_; ++i) {
            save_tree(stream, tree_roots_[i]);
        }
    }



    void loadIndex(FILE* stream) CV_OVERRIDE
    {
        load_value(stream, trees_);
        if (tree_roots_!=NULL) {
            delete[] tree_roots_;
        }
        tree_roots_ = new NodePtr[trees_];
        for (int i=0; i<trees_; ++i) {
            load_tree(stream,tree_roots_[i]);
        }

        index_params_["algorithm"] = getType();
        index_params_["trees"] = tree_roots_;
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

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const CV_OVERRIDE
    {
        return int(pool_.usedMemory+pool_.wastedMemory+dataset_.rows*sizeof(int));  // pool memory and vind array memory
    }

    /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) CV_OVERRIDE
    {
        int maxChecks = get_param(searchParams,"checks", 32);
        float epsError = 1+get_param(searchParams,"eps",0.0f);

        if (maxChecks==FLANN_CHECKS_UNLIMITED) {
            getExactNeighbors(result, vec, epsError);
        }
        else {
            getNeighbors(result, vec, maxChecks, epsError);
        }
    }

    IndexParams getParameters() const CV_OVERRIDE
    {
        return index_params_;
    }

private:


    /*--------------------- Internal Data Structures --------------------------*/
    struct Node
    {
        /**
         * Dimension used for subdivision.
         */
        int divfeat;
        /**
         * The values used for subdivision.
         */
        DistanceType divval;
        /**
         * The child nodes.
         */
        Node* child1, * child2;
    };
    typedef Node* NodePtr;
    typedef BranchStruct<NodePtr, DistanceType> BranchSt;
    typedef BranchSt* Branch;



    void save_tree(FILE* stream, NodePtr tree)
    {
        save_value(stream, *tree);
        if (tree->child1!=NULL) {
            save_tree(stream, tree->child1);
        }
        if (tree->child2!=NULL) {
            save_tree(stream, tree->child2);
        }
    }


    void load_tree(FILE* stream, NodePtr& tree)
    {
        tree = pool_.allocate<Node>();
        load_value(stream, *tree);
        if (tree->child1!=NULL) {
            load_tree(stream, tree->child1);
        }
        if (tree->child2!=NULL) {
            load_tree(stream, tree->child2);
        }
    }


    /**
     * Create a tree node that subdivides the list of vecs from vind[first]
     * to vind[last].  The routine is called recursively on each sublist.
     * Place a pointer to this new tree node in the location pTree.
     *
     * Params: pTree = the new node to create
     *                  first = index of the first vector
     *                  last = index of the last vector
     */
    NodePtr divideTree(int* ind, int count)
    {
        NodePtr node = pool_.allocate<Node>(); // allocate memory

        /* If too few exemplars remain, then make this a leaf node. */
        if ( count == 1) {
            node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
            node->divfeat = *ind;    /* Store index of this vec. */
        }
        else {
            int idx;
            int cutfeat;
            DistanceType cutval;
            meanSplit(ind, count, idx, cutfeat, cutval);

            node->divfeat = cutfeat;
            node->divval = cutval;
            node->child1 = divideTree(ind, idx);
            node->child2 = divideTree(ind+idx, count-idx);
        }

        return node;
    }


    /**
     * Choose which feature to use in order to subdivide this set of vectors.
     * Make a random choice among those with the highest variance, and use
     * its variance as the threshold value.
     */
    void meanSplit(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval)
    {
        memset(mean_,0,veclen_*sizeof(DistanceType));
        memset(var_,0,veclen_*sizeof(DistanceType));

        /* Compute mean values.  Only the first SAMPLE_MEAN values need to be
            sampled to get a good estimate.
         */
        int cnt = std::min((int)SAMPLE_MEAN+1, count);
        for (int j = 0; j < cnt; ++j) {
            ElementType* v = dataset_[ind[j]];
            for (size_t k=0; k<veclen_; ++k) {
                mean_[k] += v[k];
            }
        }
        for (size_t k=0; k<veclen_; ++k) {
            mean_[k] /= cnt;
        }

        /* Compute variances (no need to divide by count). */
        for (int j = 0; j < cnt; ++j) {
            ElementType* v = dataset_[ind[j]];
            for (size_t k=0; k<veclen_; ++k) {
                DistanceType dist = v[k] - mean_[k];
                var_[k] += dist * dist;
            }
        }
        /* Select one of the highest variance indices at random. */
        cutfeat = selectDivision(var_);
        cutval = mean_[cutfeat];

        int lim1, lim2;
        planeSplit(ind, count, cutfeat, cutval, lim1, lim2);

        if (lim1>count/2) index = lim1;
        else if (lim2<count/2) index = lim2;
        else index = count/2;

        /* If either list is empty, it means that all remaining features
         * are identical. Split in the middle to maintain a balanced tree.
         */
        if ((lim1==count)||(lim2==0)) index = count/2;
    }


    /**
     * Select the top RAND_DIM largest values from v and return the index of
     * one of these selected at random.
     */
    int selectDivision(DistanceType* v)
    {
        int num = 0;
        size_t topind[RAND_DIM];

        /* Create a list of the indices of the top RAND_DIM values. */
        for (size_t i = 0; i < veclen_; ++i) {
            if ((num < RAND_DIM)||(v[i] > v[topind[num-1]])) {
                /* Put this element at end of topind. */
                if (num < RAND_DIM) {
                    topind[num++] = i;            /* Add to list. */
                }
                else {
                    topind[num-1] = i;         /* Replace last element. */
                }
                /* Bubble end value down to right location by repeated swapping. */
                int j = num - 1;
                while (j > 0  &&  v[topind[j]] > v[topind[j-1]]) {
                    std::swap(topind[j], topind[j-1]);
                    --j;
                }
            }
        }
        /* Select a random integer in range [0,num-1], and return that index. */
        int rnd = rand_int(num);
        return (int)topind[rnd];
    }


    /**
     *  Subdivide the list of points by a plane perpendicular on axe corresponding
     *  to the 'cutfeat' dimension at 'cutval' position.
     *
     *  On return:
     *  dataset[ind[0..lim1-1]][cutfeat]<cutval
     *  dataset[ind[lim1..lim2-1]][cutfeat]==cutval
     *  dataset[ind[lim2..count]][cutfeat]>cutval
     */
    void planeSplit(int* ind, int count, int cutfeat, DistanceType cutval, int& lim1, int& lim2)
    {
        /* Move vector indices for left subtree to front of list. */
        int left = 0;
        int right = count-1;
        for (;; ) {
            while (left<=right && dataset_[ind[left]][cutfeat]<cutval) ++left;
            while (left<=right && dataset_[ind[right]][cutfeat]>=cutval) --right;
            if (left>right) break;
            std::swap(ind[left], ind[right]); ++left; --right;
        }
        lim1 = left;
        right = count-1;
        for (;; ) {
            while (left<=right && dataset_[ind[left]][cutfeat]<=cutval) ++left;
            while (left<=right && dataset_[ind[right]][cutfeat]>cutval) --right;
            if (left>right) break;
            std::swap(ind[left], ind[right]); ++left; --right;
        }
        lim2 = left;
    }

    /**
     * Performs an exact nearest neighbor search. The exact search performs a full
     * traversal of the tree.
     */
    void getExactNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, float epsError)
    {
        //		checkID -= 1;  /* Set a different unique ID for each search. */

        if (trees_ > 1) {
            fprintf(stderr,"It doesn't make any sense to use more than one tree for exact search");
        }
        if (trees_>0) {
            searchLevelExact(result, vec, tree_roots_[0], 0.0, epsError);
        }
        CV_Assert(result.full());
    }

    /**
     * Performs the approximate nearest-neighbor search. The search is approximate
     * because the tree traversal is abandoned after a given number of descends in
     * the tree.
     */
    void getNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, int maxCheck, float epsError)
    {
        int i;
        BranchSt branch;

        int checkCount = 0;
        Heap<BranchSt>* heap = new Heap<BranchSt>((int)size_);
        DynamicBitset checked(size_);

        /* Search once through each tree down to root. */
        for (i = 0; i < trees_; ++i) {
            searchLevel(result, vec, tree_roots_[i], 0, checkCount, maxCheck, epsError, heap, checked);
        }

        /* Keep searching other branches from heap until finished. */
        while ( heap->popMin(branch) && (checkCount < maxCheck || !result.full() )) {
            searchLevel(result, vec, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked);
        }

        delete heap;

        CV_Assert(result.full());
    }


    /**
     *  Search starting from a given node of the tree.  Based on any mismatches at
     *  higher levels, all exemplars below this level must have a distance of
     *  at least "mindistsq".
     */
    void searchLevel(ResultSet<DistanceType>& result_set, const ElementType* vec, NodePtr node, DistanceType mindist, int& checkCount, int maxCheck,
                     float epsError, Heap<BranchSt>* heap, DynamicBitset& checked)
    {
        if (result_set.worstDist()<mindist) {
            //			printf("Ignoring branch, too far\n");
            return;
        }

        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            /*  Do not check same node more than once when searching multiple trees.
                Once a vector is checked, we set its location in vind to the
                current checkID.
             */
            int index = node->divfeat;
            if ( checked.test(index) || ((checkCount>=maxCheck)&& result_set.full()) ) return;
            checked.set(index);
            checkCount++;

            DistanceType dist = distance_(dataset_[index], vec, veclen_);
            result_set.addPoint(dist,index);

            return;
        }

        /* Which child branch should be taken first? */
        ElementType val = vec[node->divfeat];
        DistanceType diff = val - node->divval;
        NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
        NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

        /* Create a branch record for the branch not taken.  Add distance
            of this feature boundary (we don't attempt to correct for any
            use of this feature in a parent node, which is unlikely to
            happen and would have only a small effect).  Don't bother
            adding more branches to heap after halfway point, as cost of
            adding exceeds their value.
         */

        DistanceType new_distsq = mindist + distance_.accum_dist(val, node->divval, node->divfeat);
        //		if (2 * checkCount < maxCheck  ||  !result.full()) {
        if ((new_distsq*epsError < result_set.worstDist())||  !result_set.full()) {
            heap->insert( BranchSt(otherChild, new_distsq) );
        }

        /* Call recursively to search next level down. */
        searchLevel(result_set, vec, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
    }

    /**
     * Performs an exact search in the tree starting from a node.
     */
    void searchLevelExact(ResultSet<DistanceType>& result_set, const ElementType* vec, const NodePtr node, DistanceType mindist, const float epsError)
    {
        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            int index = node->divfeat;
            DistanceType dist = distance_(dataset_[index], vec, veclen_);
            result_set.addPoint(dist,index);
            return;
        }

        /* Which child branch should be taken first? */
        ElementType val = vec[node->divfeat];
        DistanceType diff = val - node->divval;
        NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
        NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

        /* Create a branch record for the branch not taken.  Add distance
            of this feature boundary (we don't attempt to correct for any
            use of this feature in a parent node, which is unlikely to
            happen and would have only a small effect).  Don't bother
            adding more branches to heap after halfway point, as cost of
            adding exceeds their value.
         */

        DistanceType new_distsq = mindist + distance_.accum_dist(val, node->divval, node->divfeat);

        /* Call recursively to search next level down. */
        searchLevelExact(result_set, vec, bestChild, mindist, epsError);

        if (new_distsq*epsError<=result_set.worstDist()) {
            searchLevelExact(result_set, vec, otherChild, new_distsq, epsError);
        }
    }


private:

    enum
    {
        /**
         * To improve efficiency, only SAMPLE_MEAN random values are used to
         * compute the mean and variance at each level when building a tree.
         * A value of 100 seems to perform as well as using all values.
         */
        SAMPLE_MEAN = 100,
        /**
         * Top random dimensions to consider
         *
         * When creating random trees, the dimension on which to subdivide is
         * selected at random from among the top RAND_DIM dimensions with the
         * highest variance.  A value of 5 works well.
         */
        RAND_DIM=5
    };


    /**
     * Number of randomized trees that are used
     */
    int trees_;

    /**
     *  Array of indices to vectors in the dataset.
     */
    std::vector<int> vind_;

    /**
     * The dataset used by this index
     */
    const Matrix<ElementType> dataset_;

    IndexParams index_params_;

    size_t size_;
    size_t veclen_;


    DistanceType* mean_;
    DistanceType* var_;


    /**
     * Array of k-d trees used to find neighbours.
     */
    NodePtr* tree_roots_;

    /**
     * Pooled memory allocator.
     *
     * Using a pooled memory allocator is more efficient
     * than allocating memory directly when there is a large
     * number small of memory allocations.
     */
    PooledAllocator pool_;

    Distance distance_;


};   // class KDTreeForest

}

//! @endcond

#endif //OPENCV_FLANN_KDTREE_INDEX_H_
