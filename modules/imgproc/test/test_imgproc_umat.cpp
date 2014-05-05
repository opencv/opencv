/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include <string>

using namespace cv;
using namespace std;

class CV_ImgprocUMatTest : public cvtest::BaseTest
{
public:
    CV_ImgprocUMatTest() {}
    ~CV_ImgprocUMatTest() {}
protected:
    void run(int)
    {
        string imgpath = string(ts->get_data_path()) + "shared/lena.png";
        Mat img = imread(imgpath, 1), gray, smallimg, result;
        UMat uimg = img.getUMat(ACCESS_READ), ugray, usmallimg, uresult;

        cvtColor(img, gray, COLOR_BGR2GRAY);
        resize(gray, smallimg, Size(), 0.75, 0.75, INTER_LINEAR);
        equalizeHist(smallimg, result);

        cvtColor(uimg, ugray, COLOR_BGR2GRAY);
        resize(ugray, usmallimg, Size(), 0.75, 0.75, INTER_LINEAR);
        equalizeHist(usmallimg, uresult);

#if 0
        imshow("orig", uimg);
        imshow("small", usmallimg);
        imshow("equalized gray", uresult);
        waitKey();
        destroyWindow("orig");
        destroyWindow("small");
        destroyWindow("equalized gray");
#endif
        ts->set_failed_test_info(cvtest::TS::OK);

        (void)uresult.getMat(ACCESS_READ);
    }
};

TEST(Imgproc_UMat, regression) { CV_ImgprocUMatTest test; test.safe_run(); }
