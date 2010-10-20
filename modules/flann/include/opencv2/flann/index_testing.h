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

#ifndef _OPENCV_TESTING_H_
#define _OPENCV_TESTING_H_

#include <cstring>
#include <cassert>

#include "opencv2/flann/matrix.h"
#include "opencv2/flann/nn_index.h"
#include "opencv2/flann/result_set.h"
#include "opencv2/flann/logger.h"
#include "opencv2/flann/timer.h"


using namespace std;

namespace cvflann
{

CV_EXPORTS int countCorrectMatches(int* neighbors, int* groundTruth, int n);


template <typename ELEM_TYPE>
float computeDistanceRaport(const Matrix<ELEM_TYPE>& inputData, ELEM_TYPE* target, int* neighbors, int* groundTruth, int veclen, int n)
{
	ELEM_TYPE* target_end = target + veclen;
    float ret = 0;
    for (int i=0;i<n;++i) {
        float den = flann_dist(target,target_end, inputData[groundTruth[i]]);
        float num = flann_dist(target,target_end, inputData[neighbors[i]]);

        if (den==0 && num==0) {
            ret += 1;
        }
        else {
            ret += num/den;
        }
    }

    return ret;
}

template <typename ELEM_TYPE>
float search_with_ground_truth(NNIndex<ELEM_TYPE>& index, const Matrix<ELEM_TYPE>& inputData, const Matrix<ELEM_TYPE>& testData, const Matrix<int>& matches, int nn, int checks, float& time, float& dist, int skipMatches)
{
    if (matches.cols<size_t(nn)) {
        logger().info("matches.cols=%d, nn=%d\n",matches.cols,nn);

        throw FLANNException("Ground truth is not computed for as many neighbors as requested");
    }

    KNNResultSet<ELEM_TYPE> resultSet(nn+skipMatches);
    SearchParams searchParams(checks);

    int correct;
    float distR;
    StartStopTimer t;
    int repeats = 0;
    while (t.value<0.2) {
        repeats++;
        t.start();
        correct = 0;
        distR = 0;
        for (size_t i = 0; i < testData.rows; i++) {
            ELEM_TYPE* target = testData[i];
            resultSet.init(target, testData.cols);
            index.findNeighbors(resultSet,target, searchParams);
            int* neighbors = resultSet.getNeighbors();
            neighbors = neighbors+skipMatches;

            correct += countCorrectMatches(neighbors,matches[i], nn);
            distR += computeDistanceRaport(inputData, target,neighbors,matches[i], testData.cols, nn);
        }
        t.stop();
    }
    time = t.value/repeats;


    float precicion = (float)correct/(nn*testData.rows);

    dist = distR/(testData.rows*nn);

    logger().info("%8d %10.4g %10.5g %10.5g %10.5g\n",
            checks, precicion, time, 1000.0 * time / testData.rows, dist);

    return precicion;
}


template <typename ELEM_TYPE>
float test_index_checks(NNIndex<ELEM_TYPE>& index, const Matrix<ELEM_TYPE>& inputData, const Matrix<ELEM_TYPE>& testData, const Matrix<int>& matches,
            int checks, float& precision, int nn = 1, int skipMatches = 0)
{
    logger().info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist\n");
    logger().info("---------------------------------------------------------\n");

    float time = 0;
    float dist = 0;
    precision = search_with_ground_truth(index, inputData, testData, matches, nn, checks, time, dist, skipMatches);

    return time;
}

template <typename ELEM_TYPE>
float test_index_precision(NNIndex<ELEM_TYPE>& index, const Matrix<ELEM_TYPE>& inputData, const Matrix<ELEM_TYPE>& testData, const Matrix<int>& matches,
             float precision, int& checks, int nn = 1, int skipMatches = 0)
{
	const float SEARCH_EPS = 0.001;

    logger().info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist\n");
    logger().info("---------------------------------------------------------\n");

    int c2 = 1;
    float p2;
    int c1 = 1;
    float p1;
    float time;
    float dist;

    p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);

    if (p2>precision) {
        logger().info("Got as close as I can\n");
        checks = c2;
        return time;
    }

    while (p2<precision) {
        c1 = c2;
        p1 = p2;
        c2 *=2;
        p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);
    }

    int cx;
    float realPrecision;
    if (fabs(p2-precision)>SEARCH_EPS) {
        logger().info("Start linear estimation\n");
        // after we got to values in the vecinity of the desired precision
        // use linear approximation get a better estimation

        cx = (c1+c2)/2;
        realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
        while (fabs(realPrecision-precision)>SEARCH_EPS) {

            if (realPrecision<precision) {
                c1 = cx;
            }
            else {
                c2 = cx;
            }
            cx = (c1+c2)/2;
            if (cx==c1) {
                logger().info("Got as close as I can\n");
                break;
            }
            realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
        }

        c2 = cx;
        p2 = realPrecision;

    } else {
        logger().info("No need for linear estimation\n");
        cx = c2;
        realPrecision = p2;
    }

    checks = cx;
    return time;
}


template <typename ELEM_TYPE>
float test_index_precisions(NNIndex<ELEM_TYPE>& index, const Matrix<ELEM_TYPE>& inputData, const Matrix<ELEM_TYPE>& testData, const Matrix<int>& matches,
                    float* precisions, int precisions_length, int nn = 1, int skipMatches = 0, float maxTime = 0)
{
	const float SEARCH_EPS = 0.001;

    // make sure precisions array is sorted
    sort(precisions, precisions+precisions_length);

    int pindex = 0;
    float precision = precisions[pindex];

    logger().info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist");
    logger().info("---------------------------------------------------------");

    int c2 = 1;
    float p2;

    int c1 = 1;
    float p1;

    float time;
    float dist;

    p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);

    // if precision for 1 run down the tree is already
    // better then some of the requested precisions, then
    // skip those
    while (precisions[pindex]<p2 && pindex<precisions_length) {
        pindex++;
    }

    if (pindex==precisions_length) {
        logger().info("Got as close as I can\n");
        return time;
    }

    for (int i=pindex;i<precisions_length;++i) {

        precision = precisions[i];
        while (p2<precision) {
            c1 = c2;
            p1 = p2;
            c2 *=2;
            p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);
            if (maxTime> 0 && time > maxTime && p2<precision) return time;
        }

        int cx;
        float realPrecision;
        if (fabs(p2-precision)>SEARCH_EPS) {
            logger().info("Start linear estimation\n");
            // after we got to values in the vecinity of the desired precision
            // use linear approximation get a better estimation

            cx = (c1+c2)/2;
            realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
            while (fabs(realPrecision-precision)>SEARCH_EPS) {

                if (realPrecision<precision) {
                    c1 = cx;
                }
                else {
                    c2 = cx;
                }
                cx = (c1+c2)/2;
                if (cx==c1) {
                    logger().info("Got as close as I can\n");
                    break;
                }
                realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
            }

            c2 = cx;
            p2 = realPrecision;

        } else {
            logger().info("No need for linear estimation\n");
            cx = c2;
            realPrecision = p2;
        }

    }
    return time;
}

} // namespace cvflann

#endif //_OPENCV_TESTING_H_
