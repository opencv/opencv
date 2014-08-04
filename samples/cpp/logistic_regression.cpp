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

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


int main()
{
    Mat data_temp, labels_temp;
    Mat data, labels;

    Mat data_train, data_test;
    Mat labels_train, labels_test;

    Mat responses, result;
    FileStorage fs1, fs2;

    FileStorage f;

    cout<<"*****************************************************************************************"<<endl;
    cout<<"\"data01.xml\" contains digits 0 and 1 of 20 samples each, collected on an Android device"<<endl;
    cout<<"Each of the collected images are of size 28 x 28 re-arranged to 1 x 784 matrix"<<endl;
    cout<<"*****************************************************************************************\n\n"<<endl;

    cout<<"loading the dataset\n"<<endl;

    f.open("data01.xml", FileStorage::READ);

    f["datamat"] >> data_temp;
    f["labelsmat"] >> labels_temp;

    data_temp.convertTo(data, CV_32F);
    labels_temp.convertTo(labels, CV_32F);

    for(int i =0;i<data.rows;i++)
    {
        if(i%2 ==0)
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

    cout<<"training samples per class: "<<data_train.rows/2<<endl;
    cout<<"testing samples per class: "<<data_test.rows/2<<endl;

    // display sample image
    Mat img_disp1 = data_train.row(2).reshape(0,28).t();
    Mat img_disp2 = data_train.row(18).reshape(0,28).t();

    imshow("digit 0", img_disp1);
    imshow("digit 1", img_disp2);

    cout<<"initializing Logisitc Regression Parameters\n"<<endl;

    // LogisticRegressionParams params1 = LogisticRegressionParams(0.001, 10, LogisticRegression::BATCH, LogisticRegression::REG_L2, 1, 1);
    // params1 (above) with batch gradient performs better than mini batch gradient below with same parameters
    LogisticRegressionParams params1 = LogisticRegressionParams(0.001, 10, LogisticRegression::MINI_BATCH, LogisticRegression::REG_L2, 1, 1);

    // however mini batch gradient descent parameters with slower learning rate(below) can be used to get higher accuracy than with parameters mentioned above
    // LogisticRegressionParams params1 = LogisticRegressionParams(0.000001, 10, LogisticRegression::MINI_BATCH, LogisticRegression::REG_L2, 1, 1);

    cout<<"training Logisitc Regression classifier\n"<<endl;

    LogisticRegression lr1(data_train, labels_train, params1);
    lr1.predict(data_test, responses);
    labels_test.convertTo(labels_test, CV_32S);

    cout<<"Original Label ::  Predicted Label"<<endl;
    result = (labels_test == responses)/255;

    for(int i=0;i<labels_test.rows;i++)
    {
        cout<<labels_test.at<int>(i,0)<<" :: "<< responses.at<int>(i,0)<<endl;
    }

    // calculate accuracy
    cout<<"accuracy: "<<((double)cv::sum(result)[0]/result.rows)*100<<"%\n";
    cout<<"saving the classifier"<<endl;

    // save the classfier
    fs1.open("NewLR_Trained.xml",FileStorage::WRITE);
    lr1.write(fs1);
    fs1.release();

    // load the classifier onto new object
    LogisticRegressionParams params2 = LogisticRegressionParams();
    LogisticRegression lr2(params2);
    cout<<"loading a new classifier"<<endl;
    fs2.open("NewLR_Trained.xml",FileStorage::READ);
    FileNode fn2 = fs2.root();
    lr2.read(fn2);
    fs2.release();

    Mat responses2;

    // predict using loaded classifier
    cout<<"predicting the dataset using the loaded classfier\n"<<endl;
    lr2.predict(data_test, responses2);
    // calculate accuracy
    cout<<"accuracy using loaded classifier: "<<100 * (float)cv::countNonZero(labels_test == responses2)/responses2.rows<<"%"<<endl;
    waitKey(0);

    return 0;
}
