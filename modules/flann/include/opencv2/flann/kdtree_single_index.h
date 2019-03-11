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

#ifndef OPENCV_FLANN_KDTREE_SINGLE_INDEX_H_
#define OPENCV_FLANN_KDTREE_SINGLE_INDEX_H_

#include <algorithm>
#include <map>
#include <cassert>
#include <cstring>

#include "general.h"
#include "nn_index.h"
#include "matrix.h"
#include "result_set.h"
#include "heap.h"
#include "allocator.h"
#include "random.h"
#include "saving.h"

namespace cvflann
{

struct KDTreeSingleIndexParams : public IndexParams
{
    KDTreeSingleIndexParams(int leaf_max_size = 10, bool reorder = true, int dim = -1)
    {
        (*this)["algorithm"] = FLANN_INDEX_KDTREE_SINGLE;
        (*this)["leaf_max_size"] = leaf_max_size;
        (*this)["reorder"] = reorder;
        (*this)["dim"] = dim;
    }
};


/**
 * Randomized kd-tree index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 */
template <typename Distance>
class KDTreeSingleIndex : public NNIndex<Distance>
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
    KDTreeSingleIndex(const Matrix<ElementType>& inputData, const IndexParams& params = KDTreeSingleIndexParams(),
                      Distance d = Distance() ) :
        dataset_(inputData), index_params_(params), distance_(d)
    {
        size_ = dataset_.rows;
        dim_ = dataset_.cols;
        root_node_ = 0;
        int dim_param = get_param(params,"dim",-1);
        if (dim_param>0) dim_ = dim_param;
        leaf_max_size_ = get_param(params,"leaf_max_size",10);
        reorder_ = get_param(params,"reorder",true);

        // Create a permutable array of indices to the input vectors.
        vind_.resize(size_);
        for (size_t i = 0; i < size_; i++) {
            vind_[i] = (int)i;
        }
    }

    KDTreeSingleIndex(const KDTreeSingleIndex&);
    KDTreeSingleIndex& operator=(const KDTreeSingleIndex&);

    /**
     * Standard destructor
     */
    ~KDTreeSingleIndex()
    {
        if (reorder_) delete[] data_.data;
    }

    /**
     * Builds the index
     */
    void buildIndex() CV_OVERRIDE
    {
        computeBoundingBox(root_bbox_);
        root_node_ = divideTree(0, (int)size_, root_bbox_ );   // construct the tree

        if (reorder_) {
            delete[] data_.data;
            data_ = cvflann::Matrix<ElementType>(new ElementType[size_*dim_], size_, dim_);
            for (size_t i=0; i<size_; ++i) {
                for (size_t j=0; j<dim_; ++j) {
                    data_[i][j] = dataset_[vind_[i]][j];
                }
            }
        }
        else {
            data_ = dataset_;
        }
    }

    flann_algorithm_t getType() const CV_OVERRIDE
    {
        return FLANN_INDEX_KDTREE_SINGLE;
    }


    void saveIndex(FILE* stream) CV_OVERRIDE
    {
        save_value(stream, size_);
        save_value(stream, dim_);
        save_value(stream, root_bbox_);
        save_value(stream, reorder_);
        save_value(stream, leaf_max_size_);
        save_value(stream, vind_);
        if (reorder_) {
            save_value(stream, data_);
        }
        save_tree(stream, root_node_);
    }


    void loadIndex(FILE* stream) CV_OVERRIDE
    {
        load_value(stream, size_);
        load_value(stream, dim_);
        load_value(stream, root_bbox_);
        load_value(stream, reorder_);
        load_value(stream, leaf_max_size_);
        load_value(stream, vind_);
        if (reorder_) {
            load_value(stream, data_);
        }
        else {
            data_ = dataset_;
        }
        load_tree(stream, root_node_);


        index_params_["algorithm"] = getType();
        index_params_["leaf_max_size"] = leaf_max_size_;
        index_params_["reorder"] = reorder_;
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
        return dim_;
    }

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const CV_OVERRIDE
    {
        return (int)(pool_.usedMemory+pool_.wastedMemory+dataset_.rows*sizeof(int));  // pool memory and vind array memory
    }


    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    void knnSearch(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, int knn, const SearchParams& params) CV_OVERRIDE
    {
        assert(queries.cols == veclen());
        assert(indices.rows >= queries.rows);
        assert(dists.rows >= queries.rows);
        assert(int(indices.cols) >= knn);
        assert(int(dists.cols) >= knn);

        KNNSimpleResultSet<DistanceType> resultSet(knn);
        for (size_t i = 0; i < queries.rows; i++) {
            resultSet.init(indices[i], dists[i]);
            findNeighbors(resultSet, queries[i], params);
        }
    }

    IndexParams getParameters() const CV_OVERRIDE
    {
        return index_params_;
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
        float epsError = 1+get_param(searchParams,"eps",0.0f);

        std::vector<DistanceType> dists(dim_,0);
        DistanceType distsq = computeInitialDistances(vec, dists);
        searchLevel(result, vec, root_node_, distsq, dists, epsError);
    }

private:


    /*--------------------- Internal Data Structures --------------------------*/
    struct Node
    {
        /**
         * Indices of points in leaf node
         */
        int left, right;
        /**
         * Dimension used for subdivision.
         */
        int divfeat;
        /**
         * The values used for subdivision.
         */
        DistanceType divlow, divhigh;
        /**
         * The child nodes.
         */
        Node* child1, * child2;
    };
    typedef Node* NodePtr;


    struct Interval
    {
        DistanceType low, high;
    };

    typedef std::vector<Interval> BoundingBox;

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


    void computeBoundingBox(BoundingBox& bbox)
    {
        bbox.resize(dim_);
        for (size_t i=0; i<dim_; ++i) {
            bbox[i].low = (DistanceType)dataset_[0][i];
            bbox[i].high = (DistanceType)dataset_[0][i];
        }
        for (size_t k=1; k<dataset_.rows; ++k) {
            for (size_t i=0; i<dim_; ++i) {
                if (dataset_[k][i]<bbox[i].low) bbox[i].low = (DistanceType)dataset_[k][i];
                if (dataset_[k][i]>bbox[i].high) bbox[i].high = (DistanceType)dataset_[k][i];
            }
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
    NodePtr divideTree(int left, int right, BoundingBox& bbox)
    {
        NodePtr node = pool_.allocate<Node>(); // allocate memory

        /* If too few exemplars remain, then make this a leaf node. */
        if ( (right-left) <= leaf_max_size_) {
            node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
            node->left = left;
            node->right = right;

            // compute bounding-box of leaf points
            for (size_t i=0; i<dim_; ++i) {
                bbox[i].low = (DistanceType)dataset_[vind_[left]][i];
                bbox[i].high = (DistanceType)dataset_[vind_[left]][i];
            }
            for (int k=left+1; k<right; ++k) {
                for (size_t i=0; i<dim_; ++i) {
                    if (bbox[i].low>dataset_[vind_[k]][i]) bbox[i].low=(DistanceType)dataset_[vind_[k]][i];
                    if (bbox[i].high<dataset_[vind_[k]][i]) bbox[i].high=(DistanceType)dataset_[vind_[k]][i];
                }
            }
        }
        else {
            int idx;
            int cutfeat;
            DistanceType cutval;
            middleSplit_(&vind_[0]+left, right-left, idx, cutfeat, cutval, bbox);

            node->divfeat = cutfeat;

            BoundingBox left_bbox(bbox);
            left_bbox[cutfeat].high = cutval;
            node->child1 = divideTree(left, left+idx, left_bbox);

            BoundingBox right_bbox(bbox);
            right_bbox[cutfeat].low = cutval;
            node->child2 = divideTree(left+idx, right, right_bbox);

            node->divlow = left_bbox[cutfeat].high;
            node->divhigh = right_bbox[cutfeat].low;

            for (size_t i=0; i<dim_; ++i) {
                bbox[i].low = std::min(left_bbox[i].low, right_bbox[i].low);
                bbox[i].high = std::max(left_bbox[i].high, right_bbox[i].high);
            }
        }

        return node;
    }

    void computeMinMax(int* ind, int count, int dim, ElementType& min_elem, ElementType& max_elem)
    {
        min_elem = dataset_[ind[0]][dim];
        max_elem = dataset_[ind[0]][dim];
        for (int i=1; i<count; ++i) {
            ElementType val = dataset_[ind[i]][dim];
            if (val<min_elem) min_elem = val;
            if (val>max_elem) max_elem = val;
        }
    }

    void middleSplit(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval, const BoundingBox& bbox)
    {
        // find the largest span from the approximate bounding box
        ElementType max_span = bbox[0].high-bbox[0].low;
        cutfeat = 0;
        cutval = (bbox[0].high+bbox[0].low)/2;
        for (size_t i=1; i<dim_; ++i) {
            ElementType span = bbox[i].high-bbox[i].low;
            if (span>max_span) {
                max_span = span;
                cutfeat = i;
                cutval = (bbox[i].high+bbox[i].low)/2;
            }
        }

        // compute exact span on the found dimension
        ElementType min_elem, max_elem;
        computeMinMax(ind, count, cutfeat, min_elem, max_elem);
        cutval = (min_elem+max_elem)/2;
        max_span = max_elem - min_elem;

        // check if a dimension of a largest span exists
        size_t k = cutfeat;
        for (size_t i=0; i<dim_; ++i) {
            if (i==k) continue;
            ElementType span = bbox[i].high-bbox[i].low;
            if (span>max_span) {
                computeMinMax(ind, count, i, min_elem, max_elem);
                span = max_elem - min_elem;
                if (span>max_span) {
                    max_span = span;
                    cutfeat = i;
                    cutval = (min_elem+max_elem)/2;
                }
            }
        }
        int lim1, lim2;
        planeSplit(ind, count, cutfeat, cutval, lim1, lim2);

        if (lim1>count/2) index = lim1;
        else if (lim2<count/2) index = lim2;
        else index = count/2;
    }


    void middleSplit_(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval, const BoundingBox& bbox)
    {
        const float EPS=0.00001f;
        DistanceType max_span = bbox[0].high-bbox[0].low;
        for (size_t i=1; i<dim_; ++i) {
            DistanceType span = bbox[i].high-bbox[i].low;
            if (span>max_span) {
                max_span = span;
            }
        }
        DistanceType max_spread = -1;
        cutfeat = 0;
        for (size_t i=0; i<dim_; ++i) {
            DistanceType span = bbox[i].high-bbox[i].low;
            if (span>(DistanceType)((1-EPS)*max_span)) {
                ElementType min_elem, max_elem;
                computeMinMax(ind, count, cutfeat, min_elem, max_elem);
                DistanceType spread = (DistanceType)(max_elem-min_elem);
                if (spread>max_spread) {
                    cutfeat = (int)i;
                    max_spread = spread;
                }
            }
        }
        // split in the middle
        DistanceType split_val = (bbox[cutfeat].low+bbox[cutfeat].high)/2;
        ElementType min_elem, max_elem;
        computeMinMax(ind, count, cutfeat, min_elem, max_elem);

        if (split_val<min_elem) cutval = (DistanceType)min_elem;
        else if (split_val>max_elem) cutval = (DistanceType)max_elem;
        else cutval = split_val;

        int lim1, lim2;
        planeSplit(ind, count, cutfeat, cutval, lim1, lim2);

        if (lim1>count/2) index = lim1;
        else if (lim2<count/2) index = lim2;
        else index = count/2;
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
        /* If either list is empty, it means that all remaining features
         * are identical. Split in the middle to maintain a balanced tree.
         */
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

    DistanceType computeInitialDistances(const ElementType* vec, std::vector<DistanceType>& dists)
    {
        DistanceType distsq = 0.0;

        for (size_t i = 0; i < dim_; ++i) {
            if (vec[i] < root_bbox_[i].low) {
                dists[i] = distance_.accum_dist(vec[i], root_bbox_[i].low, (int)i);
                distsq += dists[i];
            }
            if (vec[i] > root_bbox_[i].high) {
                dists[i] = distance_.accum_dist(vec[i], root_bbox_[i].high, (int)i);
                distsq += dists[i];
            }
        }

        return distsq;
    }

    /**
     * Performs an exact search in the tree starting from a node.
     */
    void searchLevel(ResultSet<DistanceType>& result_set, const ElementType* vec, const NodePtr node, DistanceType mindistsq,
                     std::vector<DistanceType>& dists, const float epsError)
    {
        /* If this is a leaf node, then do check and return. */
        if ((node->child1 == NULL)&&(node->child2 == NULL)) {
            DistanceType worst_dist = result_set.worstDist();
            for (int i=node->left; i<node->right; ++i) {
                int index = reorder_ ? i : vind_[i];
                DistanceType dist = distance_(vec, data_[index], dim_, worst_dist);
                if (dist<worst_dist) {
                    result_set.addPoint(dist,vind_[i]);
                }
            }
            return;
        }

        /* Which child branch should be taken first? */
        int idx = node->divfeat;
        ElementType val = vec[idx];
        DistanceType diff1 = val - node->divlow;
        DistanceType diff2 = val - node->divhigh;

        NodePtr bestChild;
        NodePtr otherChild;
        DistanceType cut_dist;
        if ((diff1+diff2)<0) {
            bestChild = node->child1;
            otherChild = node->child2;
            cut_dist = distance_.accum_dist(val, node->divhigh, idx);
        }
        else {
            bestChild = node->child2;
            otherChild = node->child1;
            cut_dist = distance_.accum_dist( val, node->divlow, idx);
        }

        /* Call recursively to search next level down. */
        searchLevel(result_set, vec, bestChild, mindistsq, dists, epsError);

        DistanceType dst = dists[idx];
        mindistsq = mindistsq + cut_dist - dst;
        dists[idx] = cut_dist;
        if (mindistsq*epsError<=result_set.worstDist()) {
            searchLevel(result_set, vec, otherChild, mindistsq, dists, epsError);
        }
        dists[idx] = dst;
    }

private:

    /**
     * The dataset used by this index
     */
    const Matrix<ElementType> dataset_;

    IndexParams index_params_;

    int leaf_max_size_;
    bool reorder_;


    /**
     *  Array of indices to vectors in the dataset.
     */
    std::vector<int> vind_;

    Matrix<ElementType> data_;

    size_t size_;
    size_t dim_;

    /**
     * Array of k-d trees used to find neighbours.
     */
    NodePtr root_node_;

    BoundingBox root_bbox_;

    /**
     * Pooled memory allocator.
     *
     * Using a pooled memory allocator is more efficient
     * than allocating memory directly when there is a large
     * number small of memory allocations.
     */
    PooledAllocator pool_;

    Distance distance_;
};   // class KDTree

}

#endif //OPENCV_FLANN_KDTREE_SINGLE_INDEX_H_
