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

#include "cvtest.h"
#include <string>

using namespace cv;

const int num_detections = 3;
const float true_scores[3] = {-0.383931f, -0.825876f, -0.959934f};
const float score_thr = 0.05f;
const CvRect true_bounding_boxes[3] = {cvRect(0, 45, 362, 452), cvRect(304, 0, 64, 80), cvRect(236, 0, 108, 59)};

class CV_LatentSVMDetectorTest : public CvTest
{
public:
    CV_LatentSVMDetectorTest();
    ~CV_LatentSVMDetectorTest();    
protected:    
    void run(int);
private:
	bool isEqual(CvRect r1, CvRect r2);
};

CV_LatentSVMDetectorTest::CV_LatentSVMDetectorTest(): CvTest( "latentsvmdetector", "cvLatentSvmDetectObjects" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}

CV_LatentSVMDetectorTest::~CV_LatentSVMDetectorTest() {}

bool CV_LatentSVMDetectorTest::isEqual(CvRect r1, CvRect r2)
{
	return ((r1.x == r2.x) && (r1.y == r2.y) && (r1.width == r2.width) && (r1.height == r2.height));
}

void CV_LatentSVMDetectorTest::run( int /* start_from */)
{      
    string img_path = string(ts->get_data_path()) + "latentsvmdetector/cat.jpg";
	string model_path = string(ts->get_data_path()) + "latentsvmdetector/cat.xml";

	IplImage* image = cvLoadImage(img_path.c_str());
	if (!image)
    {
        ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
        return;
    }

	CvLatentSvmDetector* detector = cvLoadLatentSvmDetector(model_path.c_str());
	if (!detector)
	{
		ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
		cvReleaseImage(&image);
		return;
	}

	CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* detections = 0;         
	detections = cvLatentSvmDetectObjects(image, detector, storage);

	if (detections->total != num_detections)
	{
		ts->set_failed_test_info( CvTS::FAIL_MISMATCH );
	}
	else
	{
		ts->set_failed_test_info(CvTS::OK);
		for (int i = 0; i < detections->total; i++)
		{
			CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( detections, i );
			CvRect bounding_box = detection.rect;
			float score = detection.score;
			if ((!isEqual(bounding_box, true_bounding_boxes[i])) || (fabs(score - true_scores[i]) > score_thr))
			{
				ts->set_failed_test_info( CvTS::FAIL_MISMATCH );
				break;
			}
		}
	}

	cvReleaseMemStorage( &storage );
	cvReleaseLatentSvmDetector( &detector );
    cvReleaseImage( &image );
}

CV_LatentSVMDetectorTest latentsvmdetector_test;
