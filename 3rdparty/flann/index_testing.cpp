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

#include "index_testing.h"
#include "result_set.h"
#include "timer.h"
#include "logger.h"
#include "dist.h"
#include "common.h"

#include <algorithm>
#include <math.h>
#include <string.h>
#include <stdlib.h>

namespace cvflann
{

const float SEARCH_EPS = 0.001f;

int countCorrectMatches(int* neighbors, int* groundTruth, int n)
{
    int count = 0;
    for (int i=0;i<n;++i) {
        for (int k=0;k<n;++k) {
            if (neighbors[i]==groundTruth[k]) {
                count++;
                break;
            }
        }
    }
    return count;
}


float computeDistanceRaport(const Matrix<float>& inputData, float* target, int* neighbors, int* groundTruth, int veclen, int n)
{
	float* target_end = target + veclen;
    float ret = 0;
    for (int i=0;i<n;++i) {
        float den = (float)flann_dist(target,target_end, inputData[groundTruth[i]]);
        float num = (float)flann_dist(target,target_end, inputData[neighbors[i]]);

//        printf("den=%g,num=%g\n",den,num);

        if (den==0 && num==0) {
            ret += 1;
        } else {
            ret += num/den;
        }
    }

    return ret;
}

float search_with_ground_truth(NNIndex& index, const Matrix<float>& inputData, const Matrix<float>& testData, const Matrix<int>& matches, int nn, int checks, float& time, float& dist, int skipMatches)
{
    if (matches.cols<nn) {
        logger.info("matches.cols=%d, nn=%d\n",matches.cols,nn);

        throw FLANNException("Ground truth is not computed for as many neighbors as requested");
    }

    KNNResultSet resultSet(nn+skipMatches);
    SearchParams searchParams(checks);

    int correct = 0;
    float distR = 0;
    StartStopTimer t;
    int repeats = 0;
    while (t.value<0.2) {
        repeats++;
        t.start();
        correct = 0;
        distR = 0;
        for (int i = 0; i < testData.rows; i++) {
            float* target = testData[i];
            resultSet.init(target, testData.cols);
            index.findNeighbors(resultSet,target, searchParams);
            int* neighbors = resultSet.getNeighbors();
            neighbors = neighbors+skipMatches;

            correct += countCorrectMatches(neighbors,matches[i], nn);
            distR += computeDistanceRaport(inputData, target,neighbors,matches[i], testData.cols, nn);
        }
        t.stop();
    }
    time = (float)(t.value/repeats);


    float precicion = (float)correct/(nn*testData.rows);

    dist = distR/(testData.rows*nn);

    logger.info("%8d %10.4g %10.5g %10.5g %10.5g\n",
            checks, precicion, time, 1000.0 * time / testData.rows, dist);

    return precicion;
}

void search_for_neighbors(NNIndex& index, const Matrix<float>& testset, Matrix<int>& result,  Matrix<float>& dists, const SearchParams& searchParams, int skip)
{
    assert(testset.rows == result.rows);

    int nn = result.cols;
    KNNResultSet resultSet(nn+skip);


    for (int i = 0; i < testset.rows; i++) {
        float* target = testset[i];
        resultSet.init(target, testset.cols);

        index.findNeighbors(resultSet,target, searchParams);

        int* neighbors = resultSet.getNeighbors();
        float* distances = resultSet.getDistances();
        memcpy(result[i], neighbors+skip, nn*sizeof(int));
        memcpy(dists[i], distances+skip, nn*sizeof(float));
    }

}

float test_index_checks(NNIndex& index, const Matrix<float>& inputData, const Matrix<float>& testData, const Matrix<int>& matches, int checks, float& precision, int nn, int skipMatches)
{
    logger.info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist\n");
    logger.info("---------------------------------------------------------\n");

    float time = 0;
    float dist = 0;
    precision = search_with_ground_truth(index, inputData, testData, matches, nn, checks, time, dist, skipMatches);

    return time;
}


float test_index_precision(NNIndex& index, const Matrix<float>& inputData, const Matrix<float>& testData, const Matrix<int>& matches,
             float precision, int& checks, int nn, int skipMatches)
{
    logger.info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist\n");
    logger.info("---------------------------------------------------------\n");

    int c2 = 1;
    float p2;
    int c1 = 1;
    float p1;
    float time;
    float dist;

    p2 = search_with_ground_truth(index, inputData, testData, matches, nn, c2, time, dist, skipMatches);

    if (p2>precision) {
        logger.info("Got as close as I can\n");
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
        logger.info("Start linear estimation\n");
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
                logger.info("Got as close as I can\n");
                break;
            }
            realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
        }

        c2 = cx;
        p2 = realPrecision;

    } else {
        logger.info("No need for linear estimation\n");
        cx = c2;
        realPrecision = p2;
    }

    checks = cx;
    return time;
}


float test_index_precisions(NNIndex& index, const Matrix<float>& inputData, const Matrix<float>& testData, const Matrix<int>& matches,
                    float* precisions, int precisions_length, int nn, int skipMatches, float maxTime)
{
    // make sure precisions array is sorted
    sort(precisions, precisions+precisions_length);

    int pindex = 0;
    float precision = precisions[pindex];

    logger.info("  Nodes  Precision(%)   Time(s)   Time/vec(ms)  Mean dist");
    logger.info("---------------------------------------------------------");

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
        logger.info("Got as close as I can\n");
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
            logger.info("Start linear estimation\n");
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
                    logger.info("Got as close as I can\n");
                    break;
                }
                realPrecision = search_with_ground_truth(index, inputData, testData, matches, nn, cx, time, dist, skipMatches);
            }

            c2 = cx;
            p2 = realPrecision;

        } else {
            logger.info("No need for linear estimation\n");
            cx = c2;
            realPrecision = p2;
        }

    }
    return time;
}

}
