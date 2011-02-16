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

#ifndef _OPENCV_DIST_H_
#define _OPENCV_DIST_H_

#include <cmath>

#include "opencv2/flann/general.h"

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

CV_EXPORTS double euclidean_dist(const unsigned char* first1, const unsigned char* last1, unsigned char* first2, double acc);


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


CV_EXPORTS int flann_minkowski_order();
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

	int p = flann_minkowski_order();

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


// L_infinity distance (NOT A VALID KD-TREE DISTANCE - NOT DIMENSIONWISE ADDITIVE)
template <typename Iterator1, typename Iterator2>
double max_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double dist = acc;
	Iterator1 lastgroup = last1 - 3;
	double diff0, diff1, diff2, diff3;

	/* Process 4 items with each loop for efficiency. */
	while (first1 < lastgroup) {
		diff0 = fabs(first1[0] - first2[0]);
		diff1 = fabs(first1[1] - first2[1]);
		diff2 = fabs(first1[2] - first2[2]);
		diff3 = fabs(first1[3] - first2[3]);
		if (diff0 > dist) dist = diff0;
		if (diff1 > dist) dist = diff1;
		if (diff2 > dist) dist = diff2;
		if (diff3 > dist) dist = diff3;
		first1 += 4;
		first2 += 4;
	}
	/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
	while (first1 < last1) {
		diff0 = fabs(*first1++ - *first2++);
		dist = (diff0 > dist) ? diff0 : dist;
	}
	return dist;
}


template <typename Iterator1, typename Iterator2>
double hist_intersection_kernel(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
	double kernel = 0;
	Iterator1 lastgroup = last1 - 3;
	double min0, min1, min2, min3;

	/* Process 4 items with each loop for efficiency. */
	while (first1 < lastgroup) {
		min0 = first1[0] < first2[0] ? first1[0] : first2[0];
		min1 = first1[1] < first2[1] ? first1[1] : first2[1];
		min2 = first1[2] < first2[2] ? first1[2] : first2[2];
		min3 = first1[3] < first2[3] ? first1[3] : first2[3];
		kernel += min0 + min1 + min2 + min3;
		first1 += 4;
		first2 += 4;
	}
	/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
	while (first1 < last1) {
		min0 = first1[0] < first2[0] ? first1[0] : first2[0];
		kernel += min0;
		first1++;
		first2++;
	}
	return kernel;
}

template <typename Iterator1, typename Iterator2>
double hist_intersection_dist_sq(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double dist_sq = acc - 2 * hist_intersection_kernel(first1, last1, first2);
	while (first1 < last1) {
		dist_sq += *first1 + *first2;
		first1++;
		first2++;
	}
	return dist_sq;
}


// Hellinger distance
template <typename Iterator1, typename Iterator2>
double hellinger_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double distsq = acc;
	double diff0, diff1, diff2, diff3;
	Iterator1 lastgroup = last1 - 3;

	/* Process 4 items with each loop for efficiency. */
	while (first1 < lastgroup) {
		diff0 = sqrt(first1[0]) - sqrt(first2[0]);
		diff1 = sqrt(first1[1]) - sqrt(first2[1]);
		diff2 = sqrt(first1[2]) - sqrt(first2[2]);
		diff3 = sqrt(first1[3]) - sqrt(first2[3]);
		distsq += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
		first1 += 4;
		first2 += 4;
	}
	/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
	while (first1 < last1) {
		diff0 = sqrt(*first1++) - sqrt(*first2++);
		distsq += diff0 * diff0;
	}
	return distsq;
}


// chi-dsquare distance
template <typename Iterator1, typename Iterator2>
double chi_square_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double dist = acc;

	while (first1 < last1) {
		double sum = *first1 + *first2;
		if (sum > 0) {
			double diff = *first1 - *first2;
			dist += diff * diff / sum;
		}
		first1++;
		first2++;
	}
	return dist;
}


// Kullback–Leibler divergence (NOT SYMMETRIC)
template <typename Iterator1, typename Iterator2>
double kl_divergence(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	double div = acc;

	while (first1 < last1) {
		if (*first2 != 0) {
			double ratio = *first1 / *first2;
			if (ratio > 0) {
				div += *first1 * log(ratio);
			}
		}
		first1++;
		first2++;
	}
	return div;
}




CV_EXPORTS flann_distance_t flann_distance_type();
/**
 * Custom distance function. The distance computed is dependent on the value
 * of the 'flann_distance_type' global variable.
 *
 * If the last argument 'acc' is passed, the result is accumulated to the value
 * of this argument.
 */
template <typename Iterator1, typename Iterator2>
double custom_dist(Iterator1 first1, Iterator1 last1, Iterator2 first2, double acc = 0)
{
	switch (flann_distance_type()) {
	case EUCLIDEAN:
		return euclidean_dist(first1, last1, first2, acc);
	case MANHATTAN:
		return manhattan_dist(first1, last1, first2, acc);
	case MINKOWSKI:
		return minkowski_dist(first1, last1, first2, acc);
	case MAX_DIST:
		return max_dist(first1, last1, first2, acc);
	case HIK:
		return hist_intersection_dist_sq(first1, last1, first2, acc);
	case HELLINGER:
		return hellinger_dist(first1, last1, first2, acc);
	case CS:
		return chi_square_dist(first1, last1, first2, acc);
	case KL:
		return kl_divergence(first1, last1, first2, acc);
	default:
		return euclidean_dist(first1, last1, first2, acc);
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

	T operator[](int) {
		return 0;
	}

	ZeroIterator<T>& operator ++(int) {
		return *this;
	}

	ZeroIterator<T>& operator+=(int) {
		return *this;
	}

};

CV_EXPORTS ZeroIterator<float>& zero();

} // namespace cvflann

#endif //_OPENCV_DIST_H_
