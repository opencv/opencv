///////////////////////////////////////////////////////////////////////////////////////
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.

// This is a implementation of the Logistic Regression algorithm in C++ in OpenCV.

// AUTHOR:
// Rahul Kavi rahulkavi[at]live[at]com
//

// contains a subset of data from the popular Iris Dataset (taken from "http://archive.ics.uci.edu/ml/datasets/Iris")

// # You are free to use, change, or redistribute the code in any way you wish for
// # non-commercial purposes, but please maintain the name of the original author.
// # This code comes with no warranty of any kind.

// #
// # You are free to use, change, or redistribute the code in any way you wish for
// # non-commercial purposes, but please maintain the name of the original author.
// # This code comes with no warranty of any kind.

// # Logistic Regression ALGORITHM


//                           License Agreement
//                For Open Source Computer Vision Library

// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:

//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.

//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.

//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.

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

#include "test_precomp.hpp"

using namespace std;
using namespace cv;


static bool calculateError( const Mat& _p_labels, const Mat& _o_labels, float& error)
{
    error = 0.0f;
    float accuracy = 0.0f;
    Mat _p_labels_temp;
    Mat _o_labels_temp;
    _p_labels.convertTo(_p_labels_temp, CV_32S);
    _o_labels.convertTo(_o_labels_temp, CV_32S);

    CV_Assert(_p_labels_temp.total() == _o_labels_temp.total());
    CV_Assert(_p_labels_temp.rows == _o_labels_temp.rows);

    accuracy = (float)cv::countNonZero(_p_labels_temp == _o_labels_temp)/_p_labels_temp.rows;
    error = 1 - accuracy;
    return true;
}

//--------------------------------------------------------------------------------------------

class CV_LRTest : public cvtest::BaseTest
{
public:
    CV_LRTest() {}
protected:
    virtual void run( int start_from );
};

void CV_LRTest::run( int /*start_from*/ )
{
    // initialize varibles from the popular Iris Dataset
    Mat data = (Mat_<double>(150, 4)<<
        5.1,3.5,1.4,0.2, 4.9,3.0,1.4,0.2, 4.7,3.2,1.3,0.2, 4.6,3.1,1.5,0.2,
        5.0,3.6,1.4,0.2, 5.4,3.9,1.7,0.4, 4.6,3.4,1.4,0.3, 5.0,3.4,1.5,0.2,
        4.4,2.9,1.4,0.2, 4.9,3.1,1.5,0.1, 5.4,3.7,1.5,0.2, 4.8,3.4,1.6,0.2,
        4.8,3.0,1.4,0.1, 4.3,3.0,1.1,0.1, 5.8,4.0,1.2,0.2, 5.7,4.4,1.5,0.4,
        5.4,3.9,1.3,0.4, 5.1,3.5,1.4,0.3, 5.7,3.8,1.7,0.3, 5.1,3.8,1.5,0.3,
        5.4,3.4,1.7,0.2, 5.1,3.7,1.5,0.4, 4.6,3.6,1.0,0.2, 5.1,3.3,1.7,0.5,
        4.8,3.4,1.9,0.2, 5.0,3.0,1.6,0.2, 5.0,3.4,1.6,0.4, 5.2,3.5,1.5,0.2,
        5.2,3.4,1.4,0.2, 4.7,3.2,1.6,0.2, 4.8,3.1,1.6,0.2, 5.4,3.4,1.5,0.4,
        5.2,4.1,1.5,0.1, 5.5,4.2,1.4,0.2, 4.9,3.1,1.5,0.1, 5.0,3.2,1.2,0.2,
        5.5,3.5,1.3,0.2, 4.9,3.1,1.5,0.1, 4.4,3.0,1.3,0.2, 5.1,3.4,1.5,0.2,
        5.0,3.5,1.3,0.3, 4.5,2.3,1.3,0.3, 4.4,3.2,1.3,0.2, 5.0,3.5,1.6,0.6,
        5.1,3.8,1.9,0.4, 4.8,3.0,1.4,0.3, 5.1,3.8,1.6,0.2, 4.6,3.2,1.4,0.2,
        5.3,3.7,1.5,0.2, 5.0,3.3,1.4,0.2, 7.0,3.2,4.7,1.4, 6.4,3.2,4.5,1.5,
        6.9,3.1,4.9,1.5, 5.5,2.3,4.0,1.3, 6.5,2.8,4.6,1.5, 5.7,2.8,4.5,1.3,
        6.3,3.3,4.7,1.6, 4.9,2.4,3.3,1.0, 6.6,2.9,4.6,1.3, 5.2,2.7,3.9,1.4,
        5.0,2.0,3.5,1.0, 5.9,3.0,4.2,1.5, 6.0,2.2,4.0,1.0, 6.1,2.9,4.7,1.4,
        5.6,2.9,3.6,1.3, 6.7,3.1,4.4,1.4, 5.6,3.0,4.5,1.5, 5.8,2.7,4.1,1.0,
        6.2,2.2,4.5,1.5, 5.6,2.5,3.9,1.1, 5.9,3.2,4.8,1.8, 6.1,2.8,4.0,1.3,
        6.3,2.5,4.9,1.5, 6.1,2.8,4.7,1.2, 6.4,2.9,4.3,1.3, 6.6,3.0,4.4,1.4,
        6.8,2.8,4.8,1.4, 6.7,3.0,5.0,1.7, 6.0,2.9,4.5,1.5, 5.7,2.6,3.5,1.0,
        5.5,2.4,3.8,1.1, 5.5,2.4,3.7,1.0, 5.8,2.7,3.9,1.2, 6.0,2.7,5.1,1.6,
        5.4,3.0,4.5,1.5, 6.0,3.4,4.5,1.6, 6.7,3.1,4.7,1.5, 6.3,2.3,4.4,1.3,
        5.6,3.0,4.1,1.3, 5.5,2.5,4.0,1.3, 5.5,2.6,4.4,1.2, 6.1,3.0,4.6,1.4,
        5.8,2.6,4.0,1.2, 5.0,2.3,3.3,1.0, 5.6,2.7,4.2,1.3, 5.7,3.0,4.2,1.2,
        5.7,2.9,4.2,1.3, 6.2,2.9,4.3,1.3, 5.1,2.5,3.0,1.1, 5.7,2.8,4.1,1.3,
        6.3,3.3,6.0,2.5, 5.8,2.7,5.1,1.9, 7.1,3.0,5.9,2.1, 6.3,2.9,5.6,1.8,
        6.5,3.0,5.8,2.2, 7.6,3.0,6.6,2.1, 4.9,2.5,4.5,1.7, 7.3,2.9,6.3,1.8,
        6.7,2.5,5.8,1.8, 7.2,3.6,6.1,2.5, 6.5,3.2,5.1,2.0, 6.4,2.7,5.3,1.9,
        6.8,3.0,5.5,2.1, 5.7,2.5,5.0,2.0, 5.8,2.8,5.1,2.4, 6.4,3.2,5.3,2.3,
        6.5,3.0,5.5,1.8, 7.7,3.8,6.7,2.2, 7.7,2.6,6.9,2.3, 6.0,2.2,5.0,1.5,
        6.9,3.2,5.7,2.3, 5.6,2.8,4.9,2.0, 7.7,2.8,6.7,2.0, 6.3,2.7,4.9,1.8,
        6.7,3.3,5.7,2.1, 7.2,3.2,6.0,1.8, 6.2,2.8,4.8,1.8, 6.1,3.0,4.9,1.8,
        6.4,2.8,5.6,2.1, 7.2,3.0,5.8,1.6, 7.4,2.8,6.1,1.9, 7.9,3.8,6.4,2.0,
        6.4,2.8,5.6,2.2, 6.3,2.8,5.1,1.5, 6.1,2.6,5.6,1.4, 7.7,3.0,6.1,2.3,
        6.3,3.4,5.6,2.4, 6.4,3.1,5.5,1.8, 6.0,3.0,4.8,1.8, 6.9,3.1,5.4,2.1,
        6.7,3.1,5.6,2.4, 6.9,3.1,5.1,2.3, 5.8,2.7,5.1,1.9, 6.8,3.2,5.9,2.3,
        6.7,3.3,5.7,2.5, 6.7,3.0,5.2,2.3, 6.3,2.5,5.0,1.9, 6.5,3.0,5.2,2.0,
        6.2,3.4,5.4,2.3, 5.9,3.0,5.1,1.8);

    Mat labels = (Mat_<int>(150, 1)<< 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          3, 3, 3, 3, 3);

    Mat responses1, responses2;
    float error = 0.0f;

    LogisticRegressionParams params1 = LogisticRegressionParams();

    params1.alpha = 1.0;
    params1.num_iters = 10001;
    params1.norm = LogisticRegression::REG_L2;
    params1.regularized = 1;
    params1.train_method = LogisticRegression::BATCH;
    params1.mini_batch_size = 10;

    // run LR classifier train classifier
    data.convertTo(data, CV_32FC1);
    labels.convertTo(labels, CV_32FC1);
    LogisticRegression lr1(data, labels, params1);

    // predict using the same data
    lr1.predict(data, responses1);

    int test_code = cvtest::TS::OK;

    // calculate error
    if(!calculateError(responses1, labels, error))
    {
        ts->printf(cvtest::TS::LOG, "Bad prediction labels\n" );
        test_code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    else if(error > 0.05f)
    {
        ts->printf(cvtest::TS::LOG, "Bad accuracy of (%f)\n", error);
        test_code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    ts->set_failed_test_info(test_code);
}

//--------------------------------------------------------------------------------------------
class CV_LRTest_SaveLoad : public cvtest::BaseTest
{
public:
    CV_LRTest_SaveLoad(){}
protected:
    virtual void run(int start_from);
};


void CV_LRTest_SaveLoad::run( int /*start_from*/ )
{

    int code = cvtest::TS::OK;

    // initialize varibles from the popular Iris Dataset
    Mat data = (Mat_<double>(150, 4)<<
        5.1,3.5,1.4,0.2, 4.9,3.0,1.4,0.2, 4.7,3.2,1.3,0.2, 4.6,3.1,1.5,0.2,
        5.0,3.6,1.4,0.2, 5.4,3.9,1.7,0.4, 4.6,3.4,1.4,0.3, 5.0,3.4,1.5,0.2,
        4.4,2.9,1.4,0.2, 4.9,3.1,1.5,0.1, 5.4,3.7,1.5,0.2, 4.8,3.4,1.6,0.2,
        4.8,3.0,1.4,0.1, 4.3,3.0,1.1,0.1, 5.8,4.0,1.2,0.2, 5.7,4.4,1.5,0.4,
        5.4,3.9,1.3,0.4, 5.1,3.5,1.4,0.3, 5.7,3.8,1.7,0.3, 5.1,3.8,1.5,0.3,
        5.4,3.4,1.7,0.2, 5.1,3.7,1.5,0.4, 4.6,3.6,1.0,0.2, 5.1,3.3,1.7,0.5,
        4.8,3.4,1.9,0.2, 5.0,3.0,1.6,0.2, 5.0,3.4,1.6,0.4, 5.2,3.5,1.5,0.2,
        5.2,3.4,1.4,0.2, 4.7,3.2,1.6,0.2, 4.8,3.1,1.6,0.2, 5.4,3.4,1.5,0.4,
        5.2,4.1,1.5,0.1, 5.5,4.2,1.4,0.2, 4.9,3.1,1.5,0.1, 5.0,3.2,1.2,0.2,
        5.5,3.5,1.3,0.2, 4.9,3.1,1.5,0.1, 4.4,3.0,1.3,0.2, 5.1,3.4,1.5,0.2,
        5.0,3.5,1.3,0.3, 4.5,2.3,1.3,0.3, 4.4,3.2,1.3,0.2, 5.0,3.5,1.6,0.6,
        5.1,3.8,1.9,0.4, 4.8,3.0,1.4,0.3, 5.1,3.8,1.6,0.2, 4.6,3.2,1.4,0.2,
        5.3,3.7,1.5,0.2, 5.0,3.3,1.4,0.2, 7.0,3.2,4.7,1.4, 6.4,3.2,4.5,1.5,
        6.9,3.1,4.9,1.5, 5.5,2.3,4.0,1.3, 6.5,2.8,4.6,1.5, 5.7,2.8,4.5,1.3,
        6.3,3.3,4.7,1.6, 4.9,2.4,3.3,1.0, 6.6,2.9,4.6,1.3, 5.2,2.7,3.9,1.4,
        5.0,2.0,3.5,1.0, 5.9,3.0,4.2,1.5, 6.0,2.2,4.0,1.0, 6.1,2.9,4.7,1.4,
        5.6,2.9,3.6,1.3, 6.7,3.1,4.4,1.4, 5.6,3.0,4.5,1.5, 5.8,2.7,4.1,1.0,
        6.2,2.2,4.5,1.5, 5.6,2.5,3.9,1.1, 5.9,3.2,4.8,1.8, 6.1,2.8,4.0,1.3,
        6.3,2.5,4.9,1.5, 6.1,2.8,4.7,1.2, 6.4,2.9,4.3,1.3, 6.6,3.0,4.4,1.4,
        6.8,2.8,4.8,1.4, 6.7,3.0,5.0,1.7, 6.0,2.9,4.5,1.5, 5.7,2.6,3.5,1.0,
        5.5,2.4,3.8,1.1, 5.5,2.4,3.7,1.0, 5.8,2.7,3.9,1.2, 6.0,2.7,5.1,1.6,
        5.4,3.0,4.5,1.5, 6.0,3.4,4.5,1.6, 6.7,3.1,4.7,1.5, 6.3,2.3,4.4,1.3,
        5.6,3.0,4.1,1.3, 5.5,2.5,4.0,1.3, 5.5,2.6,4.4,1.2, 6.1,3.0,4.6,1.4,
        5.8,2.6,4.0,1.2, 5.0,2.3,3.3,1.0, 5.6,2.7,4.2,1.3, 5.7,3.0,4.2,1.2,
        5.7,2.9,4.2,1.3, 6.2,2.9,4.3,1.3, 5.1,2.5,3.0,1.1, 5.7,2.8,4.1,1.3,
        6.3,3.3,6.0,2.5, 5.8,2.7,5.1,1.9, 7.1,3.0,5.9,2.1, 6.3,2.9,5.6,1.8,
        6.5,3.0,5.8,2.2, 7.6,3.0,6.6,2.1, 4.9,2.5,4.5,1.7, 7.3,2.9,6.3,1.8,
        6.7,2.5,5.8,1.8, 7.2,3.6,6.1,2.5, 6.5,3.2,5.1,2.0, 6.4,2.7,5.3,1.9,
        6.8,3.0,5.5,2.1, 5.7,2.5,5.0,2.0, 5.8,2.8,5.1,2.4, 6.4,3.2,5.3,2.3,
        6.5,3.0,5.5,1.8, 7.7,3.8,6.7,2.2, 7.7,2.6,6.9,2.3, 6.0,2.2,5.0,1.5,
        6.9,3.2,5.7,2.3, 5.6,2.8,4.9,2.0, 7.7,2.8,6.7,2.0, 6.3,2.7,4.9,1.8,
        6.7,3.3,5.7,2.1, 7.2,3.2,6.0,1.8, 6.2,2.8,4.8,1.8, 6.1,3.0,4.9,1.8,
        6.4,2.8,5.6,2.1, 7.2,3.0,5.8,1.6, 7.4,2.8,6.1,1.9, 7.9,3.8,6.4,2.0,
        6.4,2.8,5.6,2.2, 6.3,2.8,5.1,1.5, 6.1,2.6,5.6,1.4, 7.7,3.0,6.1,2.3,
        6.3,3.4,5.6,2.4, 6.4,3.1,5.5,1.8, 6.0,3.0,4.8,1.8, 6.9,3.1,5.4,2.1,
        6.7,3.1,5.6,2.4, 6.9,3.1,5.1,2.3, 5.8,2.7,5.1,1.9, 6.8,3.2,5.9,2.3,
        6.7,3.3,5.7,2.5, 6.7,3.0,5.2,2.3, 6.3,2.5,5.0,1.9, 6.5,3.0,5.2,2.0,
        6.2,3.4,5.4,2.3, 5.9,3.0,5.1,1.8);

    Mat labels = (Mat_<int>(150, 1)<< 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          3, 3, 3, 3, 3);

    // LogisticRegressionParams params = LogisticRegressionParams();

    Mat responses1, responses2;
    Mat learnt_mat1, learnt_mat2;
    Mat pred_result1, comp_learnt_mats;

    float errorCount = 0.0;

    LogisticRegressionParams params1 = LogisticRegressionParams();
    LogisticRegressionParams params2 = LogisticRegressionParams();

    params1.alpha = 1.0;
    params1.num_iters = 10001;
    params1.norm = LogisticRegression::REG_L2;
    params1.regularized = 1;
    params1.train_method = LogisticRegression::BATCH;
    params1.mini_batch_size = 10;

    data.convertTo(data, CV_32FC1);
    labels.convertTo(labels, CV_32FC1);

    // run LR classifier train classifier
    LogisticRegression lr1(data, labels, params1);
    LogisticRegression lr2(params2);
    learnt_mat1 = lr1.get_learnt_thetas();

    lr1.predict(data, responses1);
    // now save the classifier

    string filename = cv::tempfile(".xml");
    try
    {
      //lr1.save(filename.c_str());
      FileStorage fs;
      fs.open(filename.c_str(),FileStorage::WRITE);
      lr1.write(fs);
      fs.release();
    }

    catch(...)
    {
        ts->printf(cvtest::TS::LOG, "Crash in write method.\n" );
        ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
    }

    try
    {
      //lr2.load(filename.c_str());
      FileStorage fs;
      fs.open(filename.c_str(),FileStorage::READ);
      FileNode fn = fs.root();
      lr2.read(fn);
      fs.release();
    }

    catch(...)
    {
        ts->printf(cvtest::TS::LOG, "Crash in read method.\n");
        ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
    }

    lr2.predict(data, responses2);

    learnt_mat2 = lr2.get_learnt_thetas();

    CV_Assert(responses1.rows == responses2.rows);

    // compare difference in learnt matrices before and after loading from disk
    comp_learnt_mats = (learnt_mat1 == learnt_mat2);
    comp_learnt_mats = comp_learnt_mats.reshape(1, comp_learnt_mats.rows*comp_learnt_mats.cols);
    comp_learnt_mats.convertTo(comp_learnt_mats, CV_32S);
    comp_learnt_mats = comp_learnt_mats/255;

    // compare difference in prediction outputs and stored inputs
    // check if there is any difference between computed learnt mat and retreived mat

    errorCount += 1 - (float)cv::countNonZero(responses1 == responses2)/responses1.rows;
    errorCount += 1 - (float)cv::sum(comp_learnt_mats)[0]/comp_learnt_mats.rows;

    if(errorCount>0)
    {
        ts->printf( cvtest::TS::LOG, "Different prediction results before writing and after reading (errorCount=%d).\n", errorCount );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    remove( filename.c_str() );

    ts->set_failed_test_info( code );
}

TEST(ML_LR, accuracy) { CV_LRTest test; test.safe_run(); }
TEST(ML_LR, save_load) { CV_LRTest_SaveLoad test; test.safe_run(); }
