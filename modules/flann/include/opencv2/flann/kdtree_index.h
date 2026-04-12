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

#include "nn_index.h"
#include "dynamic_bitset.h"
#include "matrix.h"
#include "result_set.h"
#include "heap.h"
#include "allocator.h"
#include "random.h"
#include "saving.h"

#if defined(__clang__) || defined(__GNUC__)
#define CV_RESTRICT __restrict__
#else
#define CV_RESTRICT
#endif

namespace cvflann
{

/**
 * Zero-overhead substitute for DynamicBitset used when duplicate checking
 * is unnecessary (single-tree search).  test() always returns false and
 * set() is a no-op, so the compiler eliminates all related branches.
 */
struct NullDynamicBitset
{
    bool test(size_t) const { return false; }
    void set(size_t) {}
};

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

        // Multi-point leaves (LEAF_MAX_SIZE) benefit low-dimensional trees by
        // reducing depth and improving cache efficiency.  For high-dimensional
        // data the per-point distance cost dominates and the original single-
        // point leaf behaviour is preferable.
        leaf_max_size_ = (veclen_ <= 16) ? (int)LEAF_MAX_SIZE : 1;

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
        const int maxChecks = get_param(searchParams,"checks", 32);
        const float epsError = 1+get_param(searchParams,"eps",0.0f);
        const bool explore_all_trees = get_param(searchParams,"explore_all_trees",false);

        // Dispatch to concrete result-set type so the compiler can inline and
        // eliminate all virtual calls in the hot search loops.
        if (maxChecks==FLANN_CHECKS_UNLIMITED) {
            if (auto* rk = dynamic_cast<KNNUniqueResultSet<DistanceType>*>(&result))
                getExactNeighbors(*rk, vec, epsError);
            else if (auto* rr = dynamic_cast<RadiusUniqueResultSet<DistanceType>*>(&result))
                getExactNeighbors(*rr, vec, epsError);
            else
                getExactNeighbors(result, vec, epsError);
        }
        else {
            if (auto* rk = dynamic_cast<KNNUniqueResultSet<DistanceType>*>(&result))
                getNeighbors(*rk, vec, maxChecks, epsError, explore_all_trees);
            else if (auto* rr = dynamic_cast<RadiusUniqueResultSet<DistanceType>*>(&result))
                getNeighbors(*rr, vec, maxChecks, epsError, explore_all_trees);
            else
                getNeighbors(result, vec, maxChecks, epsError, explore_all_trees);
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
        /**
         * Leaf node storage: array of point indices and their count.
         * Non-null only when child1 == child2 == NULL.
         */
        int* indices;
        int count;
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
        if (tree->child1==NULL && tree->child2==NULL) {
            save_value(stream, tree->indices[0], tree->count);
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
        if (tree->child1==NULL && tree->child2==NULL) {
            tree->indices = pool_.allocate<int>(tree->count);
            load_value(stream, tree->indices[0], tree->count);
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
        if (count <= leaf_max_size_) {
            node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
            node->count = count;
            node->indices = pool_.allocate<int>(count);
            for (int i = 0; i < count; ++i)
                node->indices[i] = ind[i];
        }
        else {
            int idx;
            int cutfeat;
            DistanceType cutval;
            meanSplit(ind, count, idx, cutfeat, cutval);

            node->divfeat = cutfeat;
            node->divval = cutval;
            node->indices = NULL;
            node->count = 0;
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
            Sum(v, veclen_, mean_);
        }
        for (size_t k=0; k<veclen_; ++k) {
            mean_[k] /= cnt;
        }

        /* Compute variances (no need to divide by count). */
        for (int j = 0; j < cnt; ++j) {
            ElementType* v = dataset_[ind[j]];
            Var(v, mean_, veclen_, var_);
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
     *
     * Uses per-dimension lower-bound replacement (not additive accumulation) so the
     * lower bound remains tight even when the same dimension is split multiple times
     * along a path.  dists[d] always holds the current squared-distance contribution
     * of dimension d; it is saved and restored around each "other child" recursion.
     */
    template<typename ResultSetType>
    void getExactNeighbors(ResultSetType& result, const ElementType* vec, float epsError)
    {
        if (trees_ > 1) {
            fprintf(stderr,"It doesn't make any sense to use more than one tree for exact search");
        }
        if (trees_>0) {
            // AutoBuffer uses the stack for small veclen_ (e.g. 3D → 12 bytes),
            // falls back to heap for large dimensions.
            cv::AutoBuffer<DistanceType> dists_buf(veclen_);
            DistanceType* dists = dists_buf.data();
            std::fill(dists, dists + veclen_, DistanceType(0));
            searchLevelExact(result, vec, tree_roots_[0], 0.0, epsError, dists);
        }
        CV_Assert(result.full());
    }

    /**
     * Performs the approximate nearest-neighbor search. The search is approximate
     * because the tree traversal is abandoned after a given number of descends in
     * the tree.
     *
     * When trees_==1, uses NullDynamicBitset to skip duplicate-check overhead
     * entirely (single-tree traversal cannot visit the same point twice).
     */
    template<typename ResultSetType>
    void getNeighbors(ResultSetType& result, const ElementType* vec,
                      int maxCheck, float epsError, bool explore_all_trees = false)
    {
        BranchSt branch;
        int checkCount = 0;

        // Priority queue storing intermediate branches in the best-bin-first search
        const cv::Ptr<Heap<BranchSt>>& heap = Heap<BranchSt>::getPooledInstance(cv::utils::getThreadID(), (int)size_);

        if (trees_ == 1) {
            // Single tree: no duplicate points possible, skip the bitset entirely.
            NullDynamicBitset checked;
            searchLevel(result, vec, tree_roots_[0], 0, checkCount, maxCheck,
                        epsError, heap, checked, explore_all_trees);
            while (heap->popMin(branch) && (checkCount < maxCheck || !result.full())) {
                searchLevel(result, vec, branch.node, branch.mindist, checkCount, maxCheck,
                            epsError, heap, checked, false);
            }
        }
        else {
            DynamicBitset checked(size_);
            for (int i = 0; i < trees_; ++i) {
                searchLevel(result, vec, tree_roots_[i], 0, checkCount, maxCheck,
                            epsError, heap, checked, explore_all_trees);
                if (!explore_all_trees && (checkCount >= maxCheck) && result.full())
                    break;
            }
            while (heap->popMin(branch) && (checkCount < maxCheck || !result.full())) {
                searchLevel(result, vec, branch.node, branch.mindist, checkCount, maxCheck,
                            epsError, heap, checked, false);
            }
        }

        CV_Assert(result.full());
    }


    /**
     *  Search starting from a given node of the tree.  Based on any mismatches at
     *  higher levels, all exemplars below this level must have a distance of
     *  at least "mindistsq".
     *
     *  Templated on ResultSetType and BitsetType so the compiler can inline all
     *  result-set operations and (when BitsetType=NullDynamicBitset) eliminate
     *  the duplicate-check bookkeeping entirely.
     */
    template<typename ResultSetType, typename BitsetType>
    void searchLevel(ResultSetType& result_set, const ElementType* vec, NodePtr node, DistanceType mindist, int& checkCount, int maxCheck,
                     float epsError, const cv::Ptr<Heap<BranchSt>>& heap, BitsetType& checked, bool explore_all_trees = false)
    {
        if (result_set.worstDist()<mindist) {
            return;
        }

        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            /* Accumulate checkCount by leaf size so that maxChecks retains its
             * original meaning (approximately N individual point examinations),
             * regardless of how many points are stored per leaf. */
            if (!explore_all_trees && (checkCount >= maxCheck) && result_set.full()) {
                return;
            }
            checkCount += node->count;
            for (int i = 0; i < node->count; ++i) {
                int index = node->indices[i];
                if (checked.test(index)) continue;
                checked.set(index);
                DistanceType dist = distance_(dataset_[index], vec, veclen_);
                result_set.addPoint(dist, index);
            }
            return;
        }

        /* Which child branch should be taken first? */
        ElementType val = vec[node->divfeat];
        DistanceType diff = val - node->divval;
        NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
        NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

        DistanceType new_distsq = mindist + distance_.accum_dist(val, node->divval, node->divfeat);
        if ((new_distsq*epsError < result_set.worstDist())||  !result_set.full()) {
            heap->insert( BranchSt(otherChild, new_distsq) );
        }

        /* Call recursively to search next level down. */
        searchLevel(result_set, vec, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
    }

    /**
     * Performs an exact search in the tree starting from a node.
     * Templated on ResultSetType to inline all result-set operations.
     */
    template<typename ResultSetType>
    void searchLevelExact(ResultSetType& result_set, const ElementType* vec,
                          const NodePtr node, DistanceType mindist,
                          const float epsError, DistanceType* dists)
    {
        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            for (int i = 0; i < node->count; ++i) {
                int index = node->indices[i];
                DistanceType dist = distance_(dataset_[index], vec, veclen_);
                result_set.addPoint(dist, index);
            }
            return;
        }

        /* Which child branch should be taken first? */
        int idx = node->divfeat;
        ElementType val = vec[idx];
        DistanceType diff = val - node->divval;
        NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
        NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

        /* Per-dimension replacement: subtract the old contribution for this dimension
         * and add the new one.  This keeps mindist tight even when the same dimension
         * is split multiple times along a path (avoids the additive over-accumulation
         * that causes over-pruning and missed neighbours in exact search). */
        DistanceType cut_dist = distance_.accum_dist(val, node->divval, idx);
        DistanceType new_mindist = mindist + cut_dist - dists[idx];

        /* Best child: no boundary crossing, mindist and dists unchanged. */
        searchLevelExact(result_set, vec, bestChild, mindist, epsError, dists);

        /* Other child: save, update, recurse, restore. */
        if (new_mindist * epsError <= result_set.worstDist()) {
            DistanceType old_dist = dists[idx];
            dists[idx] = cut_dist;
            searchLevelExact(result_set, vec, otherChild, new_mindist, epsError, dists);
            dists[idx] = old_dist;
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
        RAND_DIM=5,
        /**
         * Maximum number of points stored in a leaf node.
         * Larger values reduce tree depth and improve cache efficiency.
         */
        LEAF_MAX_SIZE=10
    };

    void Sum(const ElementType* CV_RESTRICT data, size_t len, DistanceType* CV_RESTRICT mean) {
        for (size_t k=0; k<len; ++k) {
            mean[k] += data[k];
        }
    }

    void Var(const ElementType* CV_RESTRICT data, const DistanceType* CV_RESTRICT mean, size_t len, DistanceType*CV_RESTRICT var) {
        for (size_t k=0; k<len; ++k) {
            DistanceType dist = data[k] - mean[k];
            var[k] += dist * dist;
        }
    }

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
    int    leaf_max_size_;


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
