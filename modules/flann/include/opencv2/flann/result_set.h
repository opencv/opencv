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

#ifndef OPENCV_FLANN_RESULTSET_H
#define OPENCV_FLANN_RESULTSET_H

//! @cond IGNORED

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace cvflann
{

/* This record represents a branch point when finding neighbors in
    the tree.  It contains a record of the minimum distance to the query
    point, as well as the node at which the search resumes.
 */

template <typename T, typename DistanceType>
struct BranchStruct
{
    T node;           /* Tree node at which search resumes */
    DistanceType mindist;     /* Minimum distance to query for all nodes below. */

    BranchStruct() {}
    BranchStruct(const T& aNode, DistanceType dist) : node(aNode), mindist(dist) {}

    bool operator<(const BranchStruct<T, DistanceType>& rhs) const
    {
        return mindist<rhs.mindist;
    }
};


template <typename DistanceType>
class ResultSet
{
public:
    virtual ~ResultSet() {}

    virtual bool full() const = 0;

    virtual void addPoint(DistanceType dist, int index) = 0;

    virtual DistanceType worstDist() const = 0;

};

/**
 * KNNSimpleResultSet does not ensure that the element it holds are unique.
 * Is used in those cases where the nearest neighbour algorithm used does not
 * attempt to insert the same element multiple times.
 */
template <typename DistanceType>
class KNNSimpleResultSet : public ResultSet<DistanceType>
{
    int* indices;
    DistanceType* dists;
    int capacity;
    int count;
    DistanceType worst_distance_;

public:
    KNNSimpleResultSet(int capacity_) : capacity(capacity_), count(0)
    {
    }

    void init(int* indices_, DistanceType* dists_)
    {
        indices = indices_;
        dists = dists_;
        count = 0;
        worst_distance_ = (std::numeric_limits<DistanceType>::max)();
        dists[capacity-1] = worst_distance_;
    }

    size_t size() const
    {
        return count;
    }

    bool full() const CV_OVERRIDE
    {
        return count == capacity;
    }


    void addPoint(DistanceType dist, int index) CV_OVERRIDE
    {
        if (dist >= worst_distance_) return;
        int i;
        for (i=count; i>0; --i) {
#ifdef FLANN_FIRST_MATCH
            if ( (dists[i-1]>dist) || ((dist==dists[i-1])&&(indices[i-1]>index)) )
#else
            if (dists[i-1]>dist)
#endif
            {
                if (i<capacity) {
                    dists[i] = dists[i-1];
                    indices[i] = indices[i-1];
                }
            }
            else break;
        }
        if (count < capacity) ++count;
        dists[i] = dist;
        indices[i] = index;
        worst_distance_ = dists[capacity-1];
    }

    DistanceType worstDist() const CV_OVERRIDE
    {
        return worst_distance_;
    }
};

/**
 * K-Nearest neighbour result set. Ensures that the elements inserted are unique
 */
template <typename DistanceType>
class KNNResultSet : public ResultSet<DistanceType>
{
    int* indices;
    DistanceType* dists;
    int capacity;
    int count;
    DistanceType worst_distance_;

public:
    KNNResultSet(int capacity_)
        : indices(NULL), dists(NULL), capacity(capacity_), count(0), worst_distance_(0)
    {
    }

    void init(int* indices_, DistanceType* dists_)
    {
        indices = indices_;
        dists = dists_;
        count = 0;
        worst_distance_ = (std::numeric_limits<DistanceType>::max)();
        dists[capacity-1] = worst_distance_;
    }

    size_t size() const
    {
        return count;
    }

    bool full() const CV_OVERRIDE
    {
        return count == capacity;
    }


    void addPoint(DistanceType dist, int index) CV_OVERRIDE
    {
        CV_DbgAssert(indices);
        CV_DbgAssert(dists);
        if (dist >= worst_distance_) return;
        int i;
        for (i = count; i > 0; --i) {
#ifdef FLANN_FIRST_MATCH
            if ( (dists[i-1]<=dist) && ((dist!=dists[i-1])||(indices[i-1]<=index)) )
#else
            if (dists[i-1]<=dist)
#endif
            {
                // Check for duplicate indices
                for (int j = i; dists[j] == dist && j--;) {
                    if (indices[j] == index) {
                        return;
                    }
                }
                break;
            }
        }

        if (count < capacity) ++count;
        for (int j = count-1; j > i; --j) {
            dists[j] = dists[j-1];
            indices[j] = indices[j-1];
        }
        dists[i] = dist;
        indices[i] = index;
        worst_distance_ = dists[capacity-1];
    }

    DistanceType worstDist() const CV_OVERRIDE
    {
        return worst_distance_;
    }
};


/**
 * A result-set class used when performing a radius based search.
 */
template <typename DistanceType>
class RadiusResultSet : public ResultSet<DistanceType>
{
    DistanceType radius;
    int* indices;
    DistanceType* dists;
    size_t capacity;
    size_t count;

public:
    RadiusResultSet(DistanceType radius_, int* indices_, DistanceType* dists_, int capacity_) :
        radius(radius_), indices(indices_), dists(dists_), capacity(capacity_)
    {
        init();
    }

    ~RadiusResultSet()
    {
    }

    void init()
    {
        count = 0;
    }

    size_t size() const
    {
        return count;
    }

    bool full() const
    {
        return true;
    }

    void addPoint(DistanceType dist, int index)
    {
        if (dist<radius) {
            if ((capacity>0)&&(count < capacity)) {
                dists[count] = dist;
                indices[count] = index;
            }
            count++;
        }
    }

    DistanceType worstDist() const
    {
        return radius;
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the k NN neighbors
 * Faster than KNNResultSet as it uses a binary heap and does not maintain two arrays
 */
template<typename DistanceType>
class UniqueResultSet : public ResultSet<DistanceType>
{
public:
    struct DistIndex
    {
        DistIndex(DistanceType dist, unsigned int index) :
            dist_(dist), index_(index)
        {
        }
        bool operator<(const DistIndex dist_index) const
        {
            return (dist_ < dist_index.dist_) || ((dist_ == dist_index.dist_) && index_ < dist_index.index_);
        }
        DistanceType dist_;
        unsigned int index_;
    };

    /** Default constructor */
    UniqueResultSet() :
        is_full_(false), worst_distance_(std::numeric_limits<DistanceType>::max())
    {
    }

    /** Check the status of the set
     * @return true if we have k NN
     */
    inline bool full() const CV_OVERRIDE
    {
        return is_full_;
    }

    /** Remove all elements in the set
     */
    virtual void clear() = 0;

    /** Copy the set to two C arrays
     * @param indices pointer to a C array of indices
     * @param dist pointer to a C array of distances
     * @param n_neighbors the number of neighbors to copy
     */
    virtual void copy(int* indices, DistanceType* dist, int n_neighbors = -1) const
    {
        if (n_neighbors < 0) {
            for (typename std::set<DistIndex>::const_iterator dist_index = dist_indices_.begin(), dist_index_end =
                     dist_indices_.end(); dist_index != dist_index_end; ++dist_index, ++indices, ++dist) {
                *indices = dist_index->index_;
                *dist = dist_index->dist_;
            }
        }
        else {
            int i = 0;
            for (typename std::set<DistIndex>::const_iterator dist_index = dist_indices_.begin(), dist_index_end =
                     dist_indices_.end(); (dist_index != dist_index_end) && (i < n_neighbors); ++dist_index, ++indices, ++dist, ++i) {
                *indices = dist_index->index_;
                *dist = dist_index->dist_;
            }
        }
    }

    /** Copy the set to two C arrays but sort it according to the distance first
     * @param indices pointer to a C array of indices
     * @param dist pointer to a C array of distances
     * @param n_neighbors the number of neighbors to copy
     */
    virtual void sortAndCopy(int* indices, DistanceType* dist, int n_neighbors = -1) const
    {
        copy(indices, dist, n_neighbors);
    }

    /** The number of neighbors in the set
     * @return
     */
    size_t size() const
    {
        return dist_indices_.size();
    }

    /** The distance of the furthest neighbor
     * If we don't have enough neighbors, it returns the max possible value
     * @return
     */
    inline DistanceType worstDist() const CV_OVERRIDE
    {
        return worst_distance_;
    }
protected:
    /** Flag to say if the set is full */
    bool is_full_;

    /** The worst distance found so far */
    DistanceType worst_distance_;

    /** The best candidates so far */
    std::set<DistIndex> dist_indices_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the k NN neighbors
 * Faster than KNNResultSet as it uses a binary heap and does not maintain two arrays
 */
template<typename DistanceType>
class KNNUniqueResultSet : public UniqueResultSet<DistanceType>
{
public:
    /** Constructor
     * @param capacity the number of neighbors to store at max
     */
    KNNUniqueResultSet(unsigned int capacity) : capacity_(capacity)
    {
        this->is_full_ = false;
        this->clear();
    }

    /** Add a possible candidate to the best neighbors
     * @param dist distance for that neighbor
     * @param index index of that neighbor
     */
    inline void addPoint(DistanceType dist, int index) CV_OVERRIDE
    {
        // Don't do anything if we are worse than the worst
        if (dist >= worst_distance_) return;
        dist_indices_.insert(DistIndex(dist, index));

        if (is_full_) {
            if (dist_indices_.size() > capacity_) {
                dist_indices_.erase(*dist_indices_.rbegin());
                worst_distance_ = dist_indices_.rbegin()->dist_;
            }
        }
        else if (dist_indices_.size() == capacity_) {
            is_full_ = true;
            worst_distance_ = dist_indices_.rbegin()->dist_;
        }
    }

    /** Remove all elements in the set
     */
    void clear() CV_OVERRIDE
    {
        dist_indices_.clear();
        worst_distance_ = std::numeric_limits<DistanceType>::max();
        is_full_ = false;
    }

protected:
    typedef typename UniqueResultSet<DistanceType>::DistIndex DistIndex;
    using UniqueResultSet<DistanceType>::is_full_;
    using UniqueResultSet<DistanceType>::worst_distance_;
    using UniqueResultSet<DistanceType>::dist_indices_;

    /** The number of neighbors to keep */
    unsigned int capacity_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the radius nearest neighbors
 * It is more accurate than RadiusResult as it is not limited in the number of neighbors
 */
template<typename DistanceType>
class RadiusUniqueResultSet : public UniqueResultSet<DistanceType>
{
public:
    /** Constructor
     * @param radius the maximum distance of a neighbor
     */
    RadiusUniqueResultSet(DistanceType radius) :
        radius_(radius)
    {
        is_full_ = true;
    }

    /** Add a possible candidate to the best neighbors
     * @param dist distance for that neighbor
     * @param index index of that neighbor
     */
    void addPoint(DistanceType dist, int index) CV_OVERRIDE
    {
        if (dist <= radius_) dist_indices_.insert(DistIndex(dist, index));
    }

    /** Remove all elements in the set
     */
    inline void clear() CV_OVERRIDE
    {
        dist_indices_.clear();
    }


    /** Check the status of the set
     * @return alwys false
     */
    inline bool full() const CV_OVERRIDE
    {
        return true;
    }

    /** The distance of the furthest neighbor
     * If we don't have enough neighbors, it returns the max possible value
     * @return
     */
    inline DistanceType worstDist() const CV_OVERRIDE
    {
        return radius_;
    }
private:
    typedef typename UniqueResultSet<DistanceType>::DistIndex DistIndex;
    using UniqueResultSet<DistanceType>::dist_indices_;
    using UniqueResultSet<DistanceType>::is_full_;

    /** The furthest distance a neighbor can be */
    DistanceType radius_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that holds the k NN neighbors within a radius distance
 */
template<typename DistanceType>
class KNNRadiusUniqueResultSet : public KNNUniqueResultSet<DistanceType>
{
public:
    /** Constructor
     * @param capacity the number of neighbors to store at max
     * @param radius the maximum distance of a neighbor
     */
    KNNRadiusUniqueResultSet(unsigned int capacity, DistanceType radius)
    {
        this->capacity_ = capacity;
        this->radius_ = radius;
        this->dist_indices_.reserve(capacity_);
        this->clear();
    }

    /** Remove all elements in the set
     */
    void clear()
    {
        dist_indices_.clear();
        worst_distance_ = radius_;
        is_full_ = false;
    }
private:
    using KNNUniqueResultSet<DistanceType>::dist_indices_;
    using KNNUniqueResultSet<DistanceType>::is_full_;
    using KNNUniqueResultSet<DistanceType>::worst_distance_;

    /** The maximum number of neighbors to consider */
    unsigned int capacity_;

    /** The maximum distance of a neighbor */
    DistanceType radius_;
};
}

//! @endcond

#endif //OPENCV_FLANN_RESULTSET_H
