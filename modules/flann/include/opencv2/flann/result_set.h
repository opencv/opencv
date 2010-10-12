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

#ifndef RESULTSET_H
#define RESULTSET_H


#include <algorithm>
#include <limits>
#include <vector>
#include "opencv2/flann/dist.h"

using namespace std;


namespace cvflann
{

/* This record represents a branch point when finding neighbors in
	the tree.  It contains a record of the minimum distance to the query
	point, as well as the node at which the search resumes.
*/

template <typename T>
struct BranchStruct {
	T node;           /* Tree node at which search resumes */
	float mindistsq;     /* Minimum distance to query for all nodes below. */

	bool operator<(const BranchStruct<T>& rhs)
	{
        return mindistsq<rhs.mindistsq;
	}

    static BranchStruct<T> make_branch(const T& aNode, float dist)
    {
        BranchStruct<T> branch;
        branch.node = aNode;
        branch.mindistsq = dist;
        return branch;
    }
};





template <typename ELEM_TYPE>
class ResultSet
{
protected:

public:

	virtual ~ResultSet() {};

	virtual void init(const ELEM_TYPE* target_, int veclen_) = 0;

	virtual int* getNeighbors() = 0;

	virtual float* getDistances() = 0;

	virtual size_t size() const = 0;

	virtual bool full() const = 0;

	virtual bool addPoint(const ELEM_TYPE* point, int index) = 0;

	virtual float worstDist() const = 0;

};


template <typename ELEM_TYPE>
class KNNResultSet : public ResultSet<ELEM_TYPE>
{
	const ELEM_TYPE* target;
	const ELEM_TYPE* target_end;
    int veclen;

	int* indices;
	float* dists;
    int capacity;

	int count;

public:
	KNNResultSet(int capacity_, ELEM_TYPE* target_ = NULL, int veclen_ = 0 ) :
			target(target_), veclen(veclen_), capacity(capacity_), count(0)
	{
		target_end = target + veclen;

        indices = new int[capacity_];
        dists = new float[capacity_];
	}

	~KNNResultSet()
	{
		delete[] indices;
		delete[] dists;
	}

	void init(const ELEM_TYPE* target_, int veclen_)
	{
        target = target_;
        veclen = veclen_;
        target_end = target + veclen;
        count = 0;
	}


	int* getNeighbors()
	{
		return indices;
	}

    float* getDistances()
    {
        return dists;
    }

    size_t size() const
    {
    	return count;
    }

	bool full() const
	{
		return count == capacity;
	}


	bool addPoint(const ELEM_TYPE* point, int index)
	{
		for (int i=0;i<count;++i) {
			if (indices[i]==index) return false;
		}
		float dist = flann_dist(target, target_end, point);

		if (count<capacity) {
			indices[count] = index;
			dists[count] = dist;
			++count;
		}
		else if (dist < dists[count-1] || (dist == dists[count-1] && index < indices[count-1])) {
//         else if (dist < dists[count-1]) {
			indices[count-1] = index;
			dists[count-1] = dist;
		}
		else {
			return false;
		}

		int i = count-1;
		// bubble up
		while (i>=1 && (dists[i]<dists[i-1] || (dists[i]==dists[i-1] && indices[i]<indices[i-1]) ) ) {
//         while (i>=1 && (dists[i]<dists[i-1]) ) {
			swap(indices[i],indices[i-1]);
			swap(dists[i],dists[i-1]);
			i--;
		}

		return true;
	}

	float worstDist() const
	{
		return (count<capacity) ? numeric_limits<float>::max() : dists[count-1];
	}
};


/**
 * A result-set class used when performing a radius based search.
 */
template <typename ELEM_TYPE>
class RadiusResultSet : public ResultSet<ELEM_TYPE>
{
	const ELEM_TYPE* target;
	const ELEM_TYPE* target_end;
    int veclen;

	struct Item {
		int index;
		float dist;

		bool operator<(Item rhs) {
			return dist<rhs.dist;
		}
	};

	vector<Item> items;
	float radius;

	bool sorted;
	int* indices;
	float* dists;
	size_t count;

private:
	void resize_vecs()
	{
		if (items.size()>count) {
			if (indices!=NULL) delete[] indices;
			if (dists!=NULL) delete[] dists;
			count = items.size();
			indices = new int[count];
			dists = new float[count];
		}
	}

public:
	RadiusResultSet(float radius_) :
		radius(radius_), indices(NULL), dists(NULL)
	{
		sorted = false;
		items.reserve(16);
		count = 0;
	}

	~RadiusResultSet()
	{
		if (indices!=NULL) delete[] indices;
		if (dists!=NULL) delete[] dists;
	}

	void init(const ELEM_TYPE* target_, int veclen_)
	{
        target = target_;
        veclen = veclen_;
        target_end = target + veclen;
        items.clear();
        sorted = false;
	}

	int* getNeighbors()
	{
		if (!sorted) {
			sorted = true;
			sort_heap(items.begin(), items.end());
		}
		resize_vecs();
		for (size_t i=0;i<items.size();++i) {
			indices[i] = items[i].index;
		}
		return indices;
	}

    float* getDistances()
    {
		if (!sorted) {
			sorted = true;
			sort_heap(items.begin(), items.end());
		}
		resize_vecs();
		for (size_t i=0;i<items.size();++i) {
			dists[i] = items[i].dist;
		}
        return dists;
    }

    size_t size() const
    {
    	return items.size();
    }

	bool full() const
	{
		return true;
	}

	bool addPoint(const ELEM_TYPE* point, int index)
	{
		Item it;
		it.index = index;
		it.dist = flann_dist(target, target_end, point);
		if (it.dist<=radius) {
			items.push_back(it);
			push_heap(items.begin(), items.end());
            return true;
		}
        return false;
	}

	float worstDist() const
	{
		return radius;
	}

};

} // namespace cvflann

#endif //RESULTSET_H
