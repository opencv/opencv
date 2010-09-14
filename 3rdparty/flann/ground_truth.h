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

#ifndef GROUND_TRUTH_H
#define GROUND_TRUTH_H

#include "matrix.h"
#include "dist.h"

namespace cvflann
{

template <typename T>
void find_nearest(const Matrix<T>& dataset, T* query, int* matches, int nn, int skip = 0)
{
    int n = nn + skip;

    T* query_end = query + dataset.cols;

    long* match = new long[n];
    T* dists = new T[n];

    dists[0] = flann_dist(query, query_end, dataset[0]);
    match[0] = 0;
    int dcnt = 1;

    for (int i=1;i<dataset.rows;++i) {
        T tmp = flann_dist(query, query_end, dataset[i]);

        if (dcnt<n) {
            match[dcnt] = i;
            dists[dcnt++] = tmp;
        }
        else if (tmp < dists[dcnt-1]) {
            dists[dcnt-1] = tmp;
            match[dcnt-1] = i;
        }

        int j = dcnt-1;
        // bubble up
        while (j>=1 && dists[j]<dists[j-1]) {
            swap(dists[j],dists[j-1]);
            swap(match[j],match[j-1]);
            j--;
        }
    }

    for (int i=0;i<nn;++i) {
        matches[i] = match[i+skip];
    }

    delete[] match;
    delete[] dists;
}


template <typename T>
void compute_ground_truth(const Matrix<T>& dataset, const Matrix<T>& testset, Matrix<int>& matches, int skip=0)
{
    for (int i=0;i<testset.rows;++i) {
        find_nearest(dataset, testset[i], matches[i], matches.cols, skip);
    }
}


}

#endif //GROUND_TRUTH_H
