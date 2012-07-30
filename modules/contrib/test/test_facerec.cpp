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
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

class CV_FacerecTest : public cvtest::BaseTest
{
public:
    CV_FacerecTest();
    ~CV_FacerecTest();
protected:
  void run(int);
  vector<Mat> loadImages(const int colorCode, const string& person) {
    vector<Mat> images;
    for (int i = 0; i < 5; ++i) {
      const string filename = format("%s_%d.png", person.c_str(), i);
      const string path = cvtest::TS::ptr()->get_data_path() + filename;
      const Mat image = imread(path, colorCode);
      EXPECT_FALSE(image.empty());
      images.push_back(image);
    }
    return images;
  }

  void testMethod(const int colorCode, Ptr<FaceRecognizer> recognizer) {
    const vector<Mat> jias = loadImages(colorCode, "jia");
    const vector<Mat> stefans = loadImages(colorCode, "stefan");

    vector<Mat> train;
    train.insert(train.end(), jias.begin(), jias.begin() + 4);
    train.insert(train.end(), stefans.begin(), stefans.begin() + 4);

    vector<int> labels;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) labels.push_back(i);

    recognizer->train(train, labels);
  
    int prediction;
    double distance;

    recognizer->predict(jias.at(4), prediction, distance);
    ASSERT_EQ(prediction, 0);
    ASSERT_GT(distance, 0);

    recognizer->predict(stefans.at(4), prediction, distance);
    ASSERT_EQ(prediction, 1);
    ASSERT_GT(distance, 0);    
  }
};

CV_FacerecTest::CV_FacerecTest() {}
CV_FacerecTest::~CV_FacerecTest() {}

void CV_FacerecTest::run(int) {
  testMethod(0, createEigenFaceRecognizer(2));
  testMethod(0, createFisherFaceRecognizer(2));
  testMethod(0, createLBPHFaceRecognizer());
  testMethod(1, createEigenFaceRecognizer(2));
  testMethod(1, createFisherFaceRecognizer(2));
  testMethod(1, createLBPHFaceRecognizer());
}

TEST(Contrib_facerec, regression) { CV_FacerecTest test; test.safe_run(); }
