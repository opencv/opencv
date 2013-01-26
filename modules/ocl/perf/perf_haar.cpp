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
extern std::string workdir;
struct getRect
{
    Rect operator ()(const CvAvgComp &e) const
    {
        return e.rect;
    }
};

PARAM_TEST_CASE(HaarTestBase, int, int)
{
    //std::vector<cv::ocl::Info> oclinfo;
    cv::ocl::OclCascadeClassifier cascade, nestedCascade;
    cv::CascadeClassifier cpucascade, cpunestedCascade;
    //    Mat img;

    double scale;
    int index;

    virtual void SetUp()
    {
        scale = 1.0;
        index = 0;
        string cascadeName = "../../../data/haarcascades/haarcascade_frontalface_alt.xml";

        if( (!cascade.load( cascadeName )) || (!cpucascade.load(cascadeName)))
        {
            cout << "ERROR: Could not load classifier cascade" << endl;
            return;
        }
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums>0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
        //cv::ocl::setBinpath("E:\\");
    }
};

////////////////////////////////faceDetect/////////////////////////////////////////////////

struct Haar : HaarTestBase {};

TEST_F(Haar, FaceDetect)
{
    string imgName = workdir + "lena.jpg";
    Mat img = imread( imgName, 1 );

    if(img.empty())
    {
        std::cout << imgName << std::endl;
        return ;
    }

    //int i = 0;
    double t = 0;
    vector<Rect> faces, oclfaces;

    // const static Scalar colors[] =  { CV_RGB(0, 0, 255),
    //                                   CV_RGB(0, 128, 255),
    //                                   CV_RGB(0, 255, 255),
    //                                   CV_RGB(0, 255, 0),
    //                                   CV_RGB(255, 128, 0),
    //                                   CV_RGB(255, 255, 0),
    //                                   CV_RGB(255, 0, 0),
    //                                   CV_RGB(255, 0, 255)
    //                                 } ;

    Mat gray, smallImg(cvRound (img.rows / scale), cvRound(img.cols / scale), CV_8UC1 );
    MemStorage storage(cvCreateMemStorage(0));
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    for(int k = 0; k < LOOP_TIMES; k++)
    {
        cpucascade.detectMultiScale( smallImg, faces,  1.1,
                                     3, 0
                                     | CV_HAAR_SCALE_IMAGE
                                     , Size(30, 30), Size(0, 0) );
    }
    t = (double)cvGetTickCount() - t ;
    printf( "cpudetection time = %g ms\n", t / (LOOP_TIMES * (double)cvGetTickFrequency() * 1000.) );

    cv::ocl::oclMat image;
    CvSeq *_objects=NULL;
    t = (double)cvGetTickCount();
    for(int k = 0; k < LOOP_TIMES; k++)
    {
        image.upload(smallImg);
        _objects = cascade.oclHaarDetectObjects( image, storage, 1.1,
                   3, 0
                   | CV_HAAR_SCALE_IMAGE
                   , Size(30, 30), Size(0, 0) );
    }
    t = (double)cvGetTickCount() - t ;
    printf( "ocldetection time = %g ms\n", t / (LOOP_TIMES * (double)cvGetTickFrequency() * 1000.) );
    vector<CvAvgComp> vecAvgComp;
    Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
    oclfaces.resize(vecAvgComp.size());
    std::transform(vecAvgComp.begin(), vecAvgComp.end(), oclfaces.begin(), getRect());

    //for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    //{
    //	Mat smallImgROI;
    //	Point center;
    //	Scalar color = colors[i%8];
    //	int radius;
    //	center.x = cvRound((r->x + r->width*0.5)*scale);
    //	center.y = cvRound((r->y + r->height*0.5)*scale);
    //	radius = cvRound((r->width + r->height)*0.25*scale);
    //	circle( img, center, radius, color, 3, 8, 0 );
    //}
    //namedWindow("result");
    //imshow("result",img);
    //waitKey(0);
    //destroyAllWindows();

}
#endif // HAVE_OPENCL
