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
using namespace cv::ml;

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

    accuracy = (float)countNonZero(_p_labels_temp == _o_labels_temp)/_p_labels_temp.rows;
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
    string dataFileName = ts->get_data_path() + "iris.data";
    Ptr<TrainData> tdata = TrainData::loadFromCSV(dataFileName, 0);

    if (tdata.empty()) {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    // run LR classifier train classifier
    Ptr<LogisticRegression> p = LogisticRegression::create();
    p->setLearningRate(1.0);
    p->setIterations(10001);
    p->setRegularization(LogisticRegression::REG_L2);
    p->setTrainMethod(LogisticRegression::BATCH);
    p->setMiniBatchSize(10);
    p->train(tdata);

    // predict using the same data
    Mat responses;
    p->predict(tdata->getSamples(), responses);

    // calculate error
    int test_code = cvtest::TS::OK;
    float error = 0.0f;
    if(!calculateError(responses, tdata->getResponses(), error))
    {
        ts->printf(cvtest::TS::LOG, "Bad prediction labels\n" );
        test_code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    else if(error > 0.05f)
    {
        ts->printf(cvtest::TS::LOG, "Bad accuracy of (%f)\n", error);
        test_code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    {
        FileStorage s("debug.xml", FileStorage::WRITE);
        s << "original" << tdata->getResponses();
        s << "predicted1" << responses;
        s << "learnt" << p->get_learnt_thetas();
        s << "error" << error;
        s.release();
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
    string dataFileName = ts->get_data_path() + "iris.data";
    Ptr<TrainData> tdata = TrainData::loadFromCSV(dataFileName, 0);

    Mat responses1, responses2;
    Mat learnt_mat1, learnt_mat2;

    // train and save the classifier
    String filename = tempfile(".xml");
    try
    {
        // run LR classifier train classifier
        Ptr<LogisticRegression> lr1 = LogisticRegression::create();
        lr1->setLearningRate(1.0);
        lr1->setIterations(10001);
        lr1->setRegularization(LogisticRegression::REG_L2);
        lr1->setTrainMethod(LogisticRegression::BATCH);
        lr1->setMiniBatchSize(10);
        lr1->train(tdata);
        lr1->predict(tdata->getSamples(), responses1);
        learnt_mat1 = lr1->get_learnt_thetas();
        lr1->save(filename);
    }
    catch(...)
    {
        ts->printf(cvtest::TS::LOG, "Crash in write method.\n" );
        ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
    }

    // and load to another
    try
    {
        Ptr<LogisticRegression> lr2 = Algorithm::load<LogisticRegression>(filename);
        lr2->predict(tdata->getSamples(), responses2);
        learnt_mat2 = lr2->get_learnt_thetas();
    }
    catch(...)
    {
        ts->printf(cvtest::TS::LOG, "Crash in write method.\n" );
        ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
    }

    CV_Assert(responses1.rows == responses2.rows);

    // compare difference in learnt matrices before and after loading from disk
    Mat comp_learnt_mats;
    comp_learnt_mats = (learnt_mat1 == learnt_mat2);
    comp_learnt_mats = comp_learnt_mats.reshape(1, comp_learnt_mats.rows*comp_learnt_mats.cols);
    comp_learnt_mats.convertTo(comp_learnt_mats, CV_32S);
    comp_learnt_mats = comp_learnt_mats/255;

    // compare difference in prediction outputs and stored inputs
    // check if there is any difference between computed learnt mat and retreived mat

    float errorCount = 0.0;
    errorCount += 1 - (float)countNonZero(responses1 == responses2)/responses1.rows;
    errorCount += 1 - (float)sum(comp_learnt_mats)[0]/comp_learnt_mats.rows;

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
