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

#ifdef HAVE_CVCONFIG_H 
#include "cvconfig.h"
#endif

#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

using namespace cv;

const int num_detections = 3;
const float true_scores[3] = {-0.383931f, -0.825876f, -0.959934f};
const float score_thr = 0.05f;
const CvRect true_bounding_boxes[3] = {cvRect(0, 45, 362, 452), cvRect(304, 0, 64, 80), cvRect(236, 0, 108, 59)};

class CV_LatentSVMDetectorTest : public cvtest::BaseTest
{
protected:    
    void run(int);
    bool isEqual(CvRect r1, CvRect r2, int eps);
};

bool CV_LatentSVMDetectorTest::isEqual(CvRect r1, CvRect r2, int eps)
{
    return (std::abs(r1.x - r2.x) <= eps
            && std::abs(r1.y - r2.y) <= eps
            && std::abs(r1.width - r2.width) <= eps
            && std::abs(r1.height - r2.height) <= eps);
}

void CV_LatentSVMDetectorTest::run( int /* start_from */)
{      
    string img_path = string(ts->get_data_path()) + "latentsvmdetector/cat.jpg";
    string model_path = string(ts->get_data_path()) + "latentsvmdetector/models_VOC2007/cat.xml";
    int numThreads = -1;

#ifdef HAVE_TBB
    numThreads = 2;
    tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
    init.initialize(numThreads);
#endif

    IplImage* image = cvLoadImage(img_path.c_str());
    if (!image)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    CvLatentSvmDetector* detector = cvLoadLatentSvmDetector(model_path.c_str());
    if (!detector)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        cvReleaseImage(&image);
        return;
    }

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* detections = 0;         
    detections = cvLatentSvmDetectObjects(image, detector, storage, 0.5f, numThreads);
    if (detections->total != num_detections)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
    }
    else
    {
        ts->set_failed_test_info(cvtest::TS::OK);
        for (int i = 0; i < detections->total; i++)
        {
            CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( detections, i );
            CvRect bounding_box = detection.rect;
            float score = detection.score;
            if ((!isEqual(bounding_box, true_bounding_boxes[i], 1)) || (fabs(score - true_scores[i]) > score_thr))
            {
                ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
                break;
            }
        }
    }
#ifdef HAVE_TBB
    init.terminate();
#endif
    cvReleaseMemStorage( &storage );
    cvReleaseLatentSvmDetector( &detector );
    cvReleaseImage( &image );
}

// Test for c++ version of Latent SVM

class LatentSVMDetectorTest : public cvtest::BaseTest
{
protected:
    void run(int);
};

static void writeDetections( FileStorage& fs, const string& nodeName, const vector<LatentSvmDetector::ObjectDetection>& detections )
{
    fs << nodeName << "[";
    for( size_t i = 0; i < detections.size(); i++ )
    {
        const LatentSvmDetector::ObjectDetection& d = detections[i];
        fs << d.rect.x << d.rect.y << d.rect.width << d.rect.height
           << d.score << d.classID;
    }
    fs << "]";
}

static void readDetections( FileStorage fs, const string& nodeName, vector<LatentSvmDetector::ObjectDetection>& detections )
{
    detections.clear();

    FileNode fn = fs.root()[nodeName];
    FileNodeIterator fni = fn.begin();
    while( fni != fn.end() )
    {
        LatentSvmDetector::ObjectDetection d;
        fni >> d.rect.x >> d.rect.y >> d.rect.width >> d.rect.height
            >> d.score >> d.classID;
        detections.push_back( d );
    }
}

static inline bool isEqual( const LatentSvmDetector::ObjectDetection& d1, const LatentSvmDetector::ObjectDetection& d2, int eps, float threshold)
{
    return (
           std::abs(d1.rect.x - d2.rect.x) <= eps
           && std::abs(d1.rect.y - d2.rect.y) <= eps
           && std::abs(d1.rect.width - d2.rect.width) <= eps
           && std::abs(d1.rect.height - d2.rect.height) <= eps
           && (d1.classID == d2.classID)
           && std::abs(d1.score - d2.score) <= threshold
           );
}

std::ostream& operator << (std::ostream& os, const CvRect& r)
{
    return (os << "[x=" << r.x << ", y=" << r.y << ", w=" << r.width << ", h=" << r.height << "]");
}

bool compareResults( const vector<LatentSvmDetector::ObjectDetection>& calc, const vector<LatentSvmDetector::ObjectDetection>& valid, int eps, float threshold)
{
    if( calc.size() != valid.size() )
        return false;

    for( size_t i = 0; i < calc.size(); i++ )
    {
        const LatentSvmDetector::ObjectDetection& c = calc[i];
        const LatentSvmDetector::ObjectDetection& v = valid[i];
        if( !isEqual(c, v, eps, threshold) )
        {
            std::cerr << "Expected: " << v.rect << " class=" << v.classID << " score=" << v.score << std::endl;
            std::cerr << "Actual:   " << c.rect << " class=" << c.classID << " score=" << c.score << std::endl;
            return false;
        }
    }
    return true;
}

void LatentSVMDetectorTest::run( int /* start_from */)
{
    string img_path_cat = string(ts->get_data_path()) + "latentsvmdetector/cat.jpg";
    string img_path_cars = string(ts->get_data_path()) + "latentsvmdetector/cars.jpg";

    string model_path_cat = string(ts->get_data_path()) + "latentsvmdetector/models_VOC2007/cat.xml";
    string model_path_car = string(ts->get_data_path()) + "latentsvmdetector/models_VOC2007/car.xml";

    string true_res_path = string(ts->get_data_path()) + "latentsvmdetector/results.xml";

    int numThreads = 1;

#ifdef HAVE_TBB
    numThreads = 2;
#endif

    Mat image_cat = imread( img_path_cat );
    Mat image_cars = imread( img_path_cars );
    if( image_cat.empty() || image_cars.empty() )
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    // We will test 2 cases:
    // detector1 - to test case of one class 'cat'
    // detector12 - to test case of two (several) classes 'cat' and car

    // Load detectors
    LatentSvmDetector detector1( vector<string>(1,model_path_cat) );

    vector<string> models_pathes(2);
    models_pathes[0] = model_path_cat;
    models_pathes[1] = model_path_car;
    LatentSvmDetector detector12( models_pathes );

    if( detector1.empty() || detector12.empty() || detector12.getClassCount() != 2 )
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    // 1. Test method detect
    // Run detectors
    vector<LatentSvmDetector::ObjectDetection> detections1_cat, detections12_cat, detections12_cars;
    detector1.detect( image_cat, detections1_cat, 0.5, numThreads );
    detector12.detect( image_cat, detections12_cat, 0.5, numThreads );
    detector12.detect( image_cars, detections12_cars, 0.5, numThreads );

    // Load true results
    FileStorage fs( true_res_path, FileStorage::READ );
    if( fs.isOpened() )
    {
        vector<LatentSvmDetector::ObjectDetection> true_detections1_cat, true_detections12_cat, true_detections12_cars;
        readDetections( fs, "detections1_cat", true_detections1_cat );
        readDetections( fs, "detections12_cat", true_detections12_cat );
        readDetections( fs, "detections12_cars", true_detections12_cars );


        if( !compareResults(detections1_cat, true_detections1_cat, 1, score_thr) )
        {
            std::cerr << "Results of detector1 are invalid on image cat.jpg" << std::endl;
            ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        }
        if( !compareResults(detections12_cat, true_detections12_cat, 1, score_thr) )
        {
            std::cerr << "Results of detector12 are invalid on image cat.jpg" << std::endl;
            ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        }
        if( !compareResults(detections12_cars, true_detections12_cars, 1, score_thr) )
        {
            std::cerr << "Results of detector12 are invalid on image cars.jpg" << std::endl;
            ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        }
    }
    else
    {
        fs.open( true_res_path, FileStorage::WRITE );
        if( fs.isOpened() )
        {
            writeDetections( fs, "detections1_cat", detections1_cat );
            writeDetections( fs, "detections12_cat", detections12_cat );
            writeDetections( fs, "detections12_cars", detections12_cars );
        }
        else
            std::cerr << "File " << true_res_path << " cann't be opened to save test results" << std::endl;
    }

    // 2. Simple tests of other methods
    if( detector1.getClassCount() != 1 || detector1.getClassNames()[0] != "cat" )
    {
        std::cerr << "Incorrect result of method getClassNames() or getClassCount()" << std::endl;
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT);
    }

    detector1.clear();
    if( !detector1.empty() )
    {
        std::cerr << "There is a bug in method clear() or empty()" << std::endl;
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT);
    }

    ts->set_failed_test_info( cvtest::TS::OK);
}

TEST(Objdetect_LatentSVMDetector_c, regression) { CV_LatentSVMDetectorTest test; test.safe_run(); }
TEST(Objdetect_LatentSVMDetector_cpp, regression) { LatentSVMDetectorTest test; test.safe_run(); }
