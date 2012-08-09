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

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

PARAM_TEST_CASE(HaarTestBase, int, int)
{
	std::vector<cv::ocl::Info> oclinfo;
    cv::ocl::OclCascadeClassifier cascade, nestedCascade;
	cv::CascadeClassifier cpucascade, cpunestedCascade;
//    Mat img;

    double scale;
    int index;

    virtual void SetUp()
    {
        scale = 1.1;

#if WIN32
        string cascadeName="E:\\opencvbuffer\\trunk\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
#else
        string cascadeName="../data/haarcascades/haarcascade_frontalface_alt.xml";
#endif

        if( (!cascade.load( cascadeName )) || (!cpucascade.load(cascadeName)))
        {
            cout << "ERROR: Could not load classifier cascade" << endl;
            cout << "Usage: facedetect [--cascade=<cascade_path>]\n"
                "   [--nested-cascade[=nested_cascade_path]]\n"
                "   [--scale[=<image scale>\n"
                "   [filename|camera_index]\n" << endl ;

            return;
        }
	int devnums = getDevice(oclinfo);
	CV_Assert(devnums>0);
	//if you want to use undefault device, set it here
	//setDevice(oclinfo[0]);
	cv::ocl::setBinpath("E:\\");
    }
};

////////////////////////////////faceDetect/////////////////////////////////////////////////

struct Haar : HaarTestBase {};

TEST_P(Haar, FaceDetect) 
{    
    for(int index = 1;index < 2; index++)
    {
        Mat img;
        char buff[256];
#if WIN32
        sprintf(buff,"E:\\myDataBase\\%d.jpg",index);
        img = imread( buff, 1 );
#else 
        sprintf(buff,"%d.jpg",index);
        img = imread( buff, 1 );
        std::cout << "Now test " << index << ".jpg" <<std::endl;
#endif
        if(img.empty())
        { 
            std::cout << "Couldn't read test" << index <<".jpg" << std::endl;
            continue;
        }

        int i = 0;
        double t = 0;
        vector<Rect> faces;

        const static Scalar colors[] =  { CV_RGB(0,0,255),
            CV_RGB(0,128,255),
            CV_RGB(0,255,255),
            CV_RGB(0,255,0),
            CV_RGB(255,128,0),
            CV_RGB(255,255,0),
            CV_RGB(255,0,0),
            CV_RGB(255,0,255)} ;

        Mat gray, smallImg(cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
        MemStorage storage(cvCreateMemStorage(0));
        cvtColor( img, gray, CV_BGR2GRAY );
        resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
        equalizeHist( smallImg, smallImg );
        CvMat _image = smallImg;

        Mat tempimg(&_image, false);

        cv::ocl::oclMat image(tempimg);
        CvSeq* _objects;

#if 1
        for(int k= 0; k<10; k++)
        {
            t = (double)cvGetTickCount();
            _objects = cascade.oclHaarDetectObjects( image, storage, 1.1,
                    2, 0
                    |CV_HAAR_SCALE_IMAGE
                    , Size(30,30), Size(0, 0) );

            t = (double)cvGetTickCount() - t ;
            printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        }

#else
        cpucascade.detectMultiScale( image, faces,  1.1,
                2, 0
                |CV_HAAR_SCALE_IMAGE
                , Size(30,30), Size(0, 0) );

#endif
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
        faces.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());

        for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
        { 
            Mat smallImgROI;
            vector<Rect> nestedObjects;
            Point center;
            Scalar color = colors[i%8];
            int radius;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }  

#if WIN32
        sprintf(buff,"E:\\result1\\%d.jpg",index);
        imwrite(buff,img);
#else 
        sprintf(buff,"testdet_%d.jpg",index);
        imwrite(buff,img);
#endif
    }
}


//INSTANTIATE_TEST_CASE_P(HaarTestBase, Haar, Combine(Values(1),
//            Values(1)));


#endif // HAVE_OPENCL
