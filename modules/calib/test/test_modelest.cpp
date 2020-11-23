/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#if 0
#include "_modelest.h"

using namespace std;
using namespace cv;

class BareModelEstimator : public CvModelEstimator2
{
public:
    BareModelEstimator(int modelPoints, CvSize modelSize, int maxBasicSolutions);

    virtual int runKernel( const CvMat*, const CvMat*, CvMat* );
    virtual void computeReprojError( const CvMat*, const CvMat*,
                                     const CvMat*, CvMat* );

    bool checkSubsetPublic( const CvMat* ms1, int count, bool checkPartialSubset );
};

BareModelEstimator::BareModelEstimator(int _modelPoints, CvSize _modelSize, int _maxBasicSolutions)
    :CvModelEstimator2(_modelPoints, _modelSize, _maxBasicSolutions)
{
}

int BareModelEstimator::runKernel( const CvMat*, const CvMat*, CvMat* )
{
    return 0;
}

void BareModelEstimator::computeReprojError( const CvMat*, const CvMat*,
                                             const CvMat*, CvMat* )
{
}

bool BareModelEstimator::checkSubsetPublic( const CvMat* ms1, int count, bool checkPartialSubset )
{
    checkPartialSubsets = checkPartialSubset;
    return checkSubset(ms1, count);
}

class CV_ModelEstimator2_Test : public cvtest::ArrayTest
{
public:
    CV_ModelEstimator2_Test();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );

    bool checkPartialSubsets;
    int usedPointsCount;

    bool checkSubsetResult;
    int generalPositionsCount;
    int maxPointsCount;
};

CV_ModelEstimator2_Test::CV_ModelEstimator2_Test()
{
    generalPositionsCount = get_test_case_count() / 2;
    maxPointsCount = 100;

    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
}

void CV_ModelEstimator2_Test::get_test_array_types_and_sizes( int /*test_case_idx*/,
                                                              vector<vector<Size> > &sizes, vector<vector<int> > &types )
{
    RNG &rng = ts->get_rng();
    checkPartialSubsets = (cvtest::randInt(rng) % 2 == 0);

    int pointsCount = cvtest::randInt(rng) % maxPointsCount;
    usedPointsCount = pointsCount == 0 ? 0 : cvtest::randInt(rng) % pointsCount;

    sizes[INPUT][0] = cvSize(1, pointsCount);
    types[INPUT][0] = CV_64FC2;

    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(1, 1);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_8UC1;
}

void CV_ModelEstimator2_Test::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
    {
        cvtest::ArrayTest::fill_array( test_case_idx, i, j, arr );
        return;
    }

    if (test_case_idx < generalPositionsCount)
    {
        //generate points in a general position (i.e. no three points can lie on the same line.)

        bool isGeneralPosition;
        do
        {
            ArrayTest::fill_array(test_case_idx, i, j, arr);

            //a simple check that the position is general:
            //  for each line check that all other points don't belong to it
            isGeneralPosition = true;
            for (int startPointIndex = 0; startPointIndex < usedPointsCount && isGeneralPosition; startPointIndex++)
            {
                for (int endPointIndex = startPointIndex + 1; endPointIndex < usedPointsCount && isGeneralPosition; endPointIndex++)
                {

                    for (int testPointIndex = 0; testPointIndex < usedPointsCount && isGeneralPosition; testPointIndex++)
                    {
                        if (testPointIndex == startPointIndex || testPointIndex == endPointIndex)
                        {
                            continue;
                        }

                        CV_Assert(arr.type() == CV_64FC2);
                        Point2d tangentVector_1 = arr.at<Point2d>(endPointIndex) - arr.at<Point2d>(startPointIndex);
                        Point2d tangentVector_2 = arr.at<Point2d>(testPointIndex) - arr.at<Point2d>(startPointIndex);

                        const float eps = 1e-4f;
                        //TODO: perhaps it is better to normalize the cross product by norms of the tangent vectors
                        if (fabs(tangentVector_1.cross(tangentVector_2)) < eps)
                        {
                            isGeneralPosition = false;
                        }
                    }
                }
            }
        }
        while(!isGeneralPosition);
    }
    else
    {
        //create points in a degenerate position (there are at least 3 points belonging to the same line)

        ArrayTest::fill_array(test_case_idx, i, j, arr);
        if (usedPointsCount <= 2)
        {
            return;
        }

        RNG &rng = ts->get_rng();
        int startPointIndex, endPointIndex, modifiedPointIndex;
        do
        {
            startPointIndex = cvtest::randInt(rng) % usedPointsCount;
            endPointIndex = cvtest::randInt(rng) % usedPointsCount;
            modifiedPointIndex = checkPartialSubsets ? usedPointsCount - 1 : cvtest::randInt(rng) % usedPointsCount;
        }
        while (startPointIndex == endPointIndex || startPointIndex == modifiedPointIndex || endPointIndex == modifiedPointIndex);

        double startWeight = cvtest::randReal(rng);
        CV_Assert(arr.type() == CV_64FC2);
        arr.at<Point2d>(modifiedPointIndex) = startWeight * arr.at<Point2d>(startPointIndex) + (1.0 - startWeight) * arr.at<Point2d>(endPointIndex);
    }
}


double CV_ModelEstimator2_Test::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}

void CV_ModelEstimator2_Test::prepare_to_validation( int test_case_idx )
{
    test_mat[OUTPUT][0].at<uchar>(0) = checkSubsetResult;
    test_mat[REF_OUTPUT][0].at<uchar>(0) = test_case_idx < generalPositionsCount || usedPointsCount <= 2;
}

void CV_ModelEstimator2_Test::run_func()
{
    //make the input continuous
    Mat input = test_mat[INPUT][0].clone();
    CvMat _input = input;

    RNG &rng = ts->get_rng();
    int modelPoints = cvtest::randInt(rng);
    CvSize modelSize = cvSize(2, modelPoints);
    int maxBasicSolutions = cvtest::randInt(rng);
    BareModelEstimator modelEstimator(modelPoints, modelSize, maxBasicSolutions);
    checkSubsetResult = modelEstimator.checkSubsetPublic(&_input, usedPointsCount, checkPartialSubsets);
}

TEST(Calib3d_ModelEstimator2, accuracy) { CV_ModelEstimator2_Test test; test.safe_run(); }

#endif
