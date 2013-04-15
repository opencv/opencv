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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Sen Liu, swjutls1987@126.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

#include "opencv2/objdetect/objdetect.hpp"
#include "precomp.hpp"

#ifdef HAVE_OPENCL

using namespace cvtest;
using namespace testing;
using namespace std;
using namespace cv;
extern string workdir;
struct getRect
{
    Rect operator ()(const CvAvgComp &e) const
    {
        return e.rect;
    }
};

PARAM_TEST_CASE(Haar, double, int)
{
    cv::ocl::OclCascadeClassifier cascade, nestedCascade;
    cv::ocl::OclCascadeClassifierBuf cascadebuf;
    cv::CascadeClassifier cpucascade, cpunestedCascade;

    double scale;
    int flags;

    virtual void SetUp()
    {
        scale = GET_PARAM(0);
        flags = GET_PARAM(1);
        string cascadeName = workdir + "../../data/haarcascades/haarcascade_frontalface_alt.xml";

        if( (!cascade.load( cascadeName )) || (!cpucascade.load(cascadeName)) || (!cascadebuf.load( cascadeName )))
        {
            cout << "ERROR: Could not load classifier cascade" << endl;
            return;
        }
    }
};

////////////////////////////////faceDetect/////////////////////////////////////////////////
TEST_P(Haar, FaceDetect)
{
    string imgName = workdir + "lena.jpg";
    Mat img = imread( imgName, 1 );

    if(img.empty())
    {
        std::cout << "Couldn't read " << imgName << std::endl;
        return ;
    }

    vector<Rect> faces, oclfaces;

    Mat gray, smallImg(cvRound (img.rows / scale), cvRound(img.cols / scale), CV_8UC1 );
    MemStorage storage(cvCreateMemStorage(0));
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    cv::ocl::oclMat image;
    CvSeq *_objects;
    image.upload(smallImg);
    _objects = cascade.oclHaarDetectObjects( image, storage, 1.1,
                   3, flags, Size(30, 30), Size(0, 0) );
    vector<CvAvgComp> vecAvgComp;
    Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
    oclfaces.resize(vecAvgComp.size());
    std::transform(vecAvgComp.begin(), vecAvgComp.end(), oclfaces.begin(), getRect());

    cpucascade.detectMultiScale( smallImg, faces,  1.1, 3,
                                 flags,
                                 Size(30, 30), Size(0, 0) );
    EXPECT_EQ(faces.size(), oclfaces.size());
}

TEST_P(Haar, FaceDetectUseBuf)
{
    string imgName = workdir + "lena.jpg";
    Mat img = imread( imgName, 1 );

    if(img.empty())
    {
        std::cout << "Couldn't read " << imgName << std::endl;
        return ;
    }

    vector<Rect> faces, oclfaces;

    Mat gray, smallImg(cvRound (img.rows / scale), cvRound(img.cols / scale), CV_8UC1 );
    MemStorage storage(cvCreateMemStorage(0));
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    cv::ocl::oclMat image;
    image.upload(smallImg);

    cascadebuf.detectMultiScale( image, oclfaces,  1.1, 3,
                                 flags,
                                 Size(30, 30), Size(0, 0) );
    cascadebuf.release();

    cpucascade.detectMultiScale( smallImg, faces,  1.1, 3,
                                 flags,
                                 Size(30, 30), Size(0, 0) );
    EXPECT_EQ(faces.size(), oclfaces.size());
}

INSTANTIATE_TEST_CASE_P(FaceDetect, Haar,
    Combine(Values(1.0),
            Values(CV_HAAR_SCALE_IMAGE, 0)));

#endif // HAVE_OPENCL
