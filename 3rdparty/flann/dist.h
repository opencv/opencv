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

#ifndef DIST_H
#define DIST_H

#include <cmath>
using namespace std;

#include "constants.h"

namespace cvflann
{

/**
 * Distance function by default set to the custom distance
 * function. This can be set to a specific distance function
 * for further efficiency.
 */
#define flann_dist custom_dist
//#define flann_dist euclidean_dist


/**
 *  Compute the squared Euclidean distance between two vectors.
 *
 *	This is highly optimised, with loop unrolling, as it is one
 *	of the most expensive inner loops.
 *
 *	The computation of squared root at the end is omitted for
 *	efficiency.
 */
template <typename Iterator1, typename Iterator2>
double euclidean_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double distsq = acc;
	double diff0, diff1, diff2, diff3;
	Iterator1 lastgroup = last1 - 3;

	/* Process 4 items with each loop for efficiency. */
	while (first1 < lastgroup) {
		diff0 = first1[0] - first2[0];
		diff1 = first1[1] - first2[1];
		diff2 = first1[2] - first2[2];
		diff3 = first1[3] - first2[3];
		distsq += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
		first1 += 4;
		first2 += 4;
	}
	/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
	while (first1 < last1) {
		diff0 = *first1++ - *first2++;
		distsq += diff0 * diff0;
	}
	return distsq;
}

/**
 *  Compute the Manhattan (L_1) distance between two vectors.
 *
 *	This is highly optimised, with loop unrolling, as it is one
 *	of the most expensive inner loops.
 */
template <typename Iterator1, typename Iterator2>
double manhattan_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double distsq = acc;
	double diff0, diff1, diff2, diff3;
	Iterator1 lastgroup = last1 - 3;

	/* Process 4 items with each loop for efficiency. */
	while (first1 < lastgroup) {
		diff0 = fabs(first1[0] - first2[0]);
		diff1 = fabs(first1[1] - first2[1]);
		diff2 = fabs(first1[2] - first2[2]);
		diff3 = fabs(first1[3] - first2[3]);
		distsq += diff0 + diff1 + diff2  + diff3;
		first1 += 4;
		first2 += 4;
	}
	/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
	while (first1 < last1) {
		diff0 = fabs(*first1++ - *first2++);
		distsq += diff0;
	}
	return distsq;
}


extern int flann_minkowski_order;
/**
 *  Compute the Minkowski (L_p) distance between two vectors.
 *
 *	This is highly optimised, with loop unrolling, as it is one
 *	of the most expensive inner loops.
 *
 *	The computation of squared root at the end is omitted for
 *	efficiency.
 */
template <typename Iterator1, typename Iterator2>
double minkowski_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double distsq = acc;
	double diff0, diff1, diff2, diff3;
	Iterator1 lastgroup = last1 - 3;

	int p = flann_minkowski_order;

	/* Process 4 items with each loop for efficiency. */
	while (first1 < lastgroup) {
		diff0 = fabs(first1[0] - first2[0]);
		diff1 = fabs(first1[1] - first2[1]);
		diff2 = fabs(first1[2] - first2[2]);
		diff3 = fabs(first1[3] - first2[3]);
		distsq += pow(diff0,p) + pow(diff1,p) + pow(diff2,p)  + pow(diff3,p);
		first1 += 4;
		first2 += 4;
	}
	/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
	while (first1 < last1) {
		diff0 = fabs(*first1++ - *first2++);
		distsq += pow(diff0,p);
	}
	return distsq;
}




extern flann_distance_t flann_distance_type;
/**
 * Custom distance function. The distance computed is dependent on the value
 * of the 'flann_distance_type' global variable.
 *
 * If the last argument 'acc' is passed, the result is accumulated to the value
 * of this argument.
 */
template <typename Iterator1, typename Iterator2>
float custom_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	switch (flann_distance_type) {
	case EUCLIDEAN:
		return (float)euclidean_dist(first1, last1, first2, acc);
	case MANHATTAN:
		return (float)manhattan_dist(first1, last1, first2, acc);
	case MINKOWSKI:
		return (float)minkowski_dist(first1, last1, first2, acc);
	default:
		return (float)euclidean_dist(first1, last1, first2, acc);
	}
}

/*
 * This is a "zero iterator". It basically behaves like a zero filled
 * array to all algorithms that use arrays as iterators (STL style).
 * It's useful when there's a need to compute the distance between feature
 * and origin it and allows for better compiler optimisation than using a
 * zero-filled array.
 */
template <typename T>
struct ZeroIterator {

	T operator*() {
		return 0;
	}

	T operator[](int /*index*/) {
		return 0;
	}

	ZeroIterator<T>& operator ++(int) {
		return *this;
	}

	ZeroIterator<T>& operator+=(int) {
		return *this;
	}

};
extern ZeroIterator<float> zero;

}

#endif //DIST_H
