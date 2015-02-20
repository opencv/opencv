/*//////////////////////////////////////////////////////////////////////////////////////
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.

// This is a implementation of the Logistic Regression algorithm in C++ in OpenCV.

// AUTHOR:
// Rahul Kavi rahulkavi[at]live[at]com
//

// contains a subset of data from the popular Iris Dataset (taken from
// "http://archive.ics.uci.edu/ml/datasets/Iris")

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
// the use of this software, even if advised of the possibility of such damage.*/

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

static void showImage(const Mat &data, int columns, const String &name)
{
    Mat bigImage;
    for(int i = 0; i < data.rows; ++i)
    {
        bigImage.push_back(data.row(i).reshape(0, columns));
    }
    imshow(name, bigImage.t());
}

static float calculateAccuracyPercent(const Mat &original, const Mat &predicted)
{
    return 100 * (float)countNonZero(original == predicted) / predicted.rows;
}

int main()
{
    const String filename = "../data/data01.xml";
    cout << "**********************************************************************" << endl;
    cout << filename
         << " contains digits 0 and 1 of 20 samples each, collected on an Android device" << endl;
    cout << "Each of the collected images are of size 28 x 28 re-arranged to 1 x 784 matrix"
         << endl;
    cout << "**********************************************************************" << endl;

    Mat data, labels;
    {
        cout << "loading the dataset...";
        FileStorage f;
        if(f.open(filename, FileStorage::READ))
        {
            f["datamat"] >> data;
            f["labelsmat"] >> labels;
            f.release();
        }
        else
        {
            cerr << "file can not be opened: " << filename << endl;
            return 1;
        }
        data.convertTo(data, CV_32F);
        labels.convertTo(labels, CV_32F);
        cout << "read " << data.rows << " rows of data" << endl;
    }

    Mat data_train, data_test;
    Mat labels_train, labels_test;
    for(int i = 0; i < data.rows; i++)
    {
        if(i % 2 == 0)
        {
            data_train.push_back(data.row(i));
            labels_train.push_back(labels.row(i));
        }
        else
        {
            data_test.push_back(data.row(i));
            labels_test.push_back(labels.row(i));
        }
    }
    cout << "training/testing samples count: " << data_train.rows << "/" << data_test.rows << endl;

    // display sample image
    showImage(data_train, 28, "train data");
    showImage(data_test, 28, "test data");

    // simple case with batch gradient
    cout << "training...";
    //! [init]
    Ptr<LogisticRegression> lr1 = LogisticRegression::create();
    lr1->setLearningRate(0.001);
    lr1->setIterations(10);
    lr1->setRegularization(LogisticRegression::REG_L2);
    lr1->setTrainMethod(LogisticRegression::BATCH);
    lr1->setMiniBatchSize(1);
    //! [init]
    lr1->train(data_train, ROW_SAMPLE, labels_train);
    cout << "done!" << endl;

    cout << "predicting...";
    Mat responses;
    lr1->predict(data_test, responses);
    cout << "done!" << endl;

    // show prediction report
    cout << "original vs predicted:" << endl;
    labels_test.convertTo(labels_test, CV_32S);
    cout << labels_test.t() << endl;
    cout << responses.t() << endl;
    cout << "accuracy: " << calculateAccuracyPercent(labels_test, responses) << "%" << endl;

    // save the classfier
    const String saveFilename = "NewLR_Trained.xml";
    cout << "saving the classifier to " << saveFilename << endl;
    lr1->save(saveFilename);

    // load the classifier onto new object
    cout << "loading a new classifier from " << saveFilename << endl;
    Ptr<LogisticRegression> lr2 = StatModel::load<LogisticRegression>(saveFilename);

    // predict using loaded classifier
    cout << "predicting the dataset using the loaded classfier...";
    Mat responses2;
    lr2->predict(data_test, responses2);
    cout << "done!" << endl;

    // calculate accuracy
    cout << labels_test.t() << endl;
    cout << responses2.t() << endl;
    cout << "accuracy: " << calculateAccuracyPercent(labels_test, responses2) << "%" << endl;

    waitKey(0);
    return 0;
}
