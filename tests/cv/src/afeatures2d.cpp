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
#include "cvtest.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

const string FEATURES2D_DIR = "features2d";
const string DETECTOR_DIR = FEATURES2D_DIR + "/feature_detectors";
const string DESCRIPTOR_DIR = FEATURES2D_DIR + "/descriptor_extractors";
const string IMAGE_FILENAME = "tsukuba.png";

/****************************************************************************************\
*            Regression tests for feature detectors comparing keypoints.                 *
\****************************************************************************************/

class CV_FeatureDetectorTest : public CvTest
{
public:
    CV_FeatureDetectorTest( const char* testName, const Ptr<FeatureDetector>& _fdetector ) :
        CvTest( testName, "cv::FeatureDetector::detect"), fdetector(_fdetector) {}

protected:
    virtual void run( int start_from )
    {
        const float maxPtDif = 1.f;
        const float maxSizeDif = 1.f;
        const float maxAngleDif = 2.f;
        const float maxResponseDif = 0.1f;

        string imgFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;
        string resFilename = string(ts->get_data_path()) + DETECTOR_DIR + "/" + string(name) + "_res.xml.gz";

        if( fdetector.empty() )
        {
            ts->printf( CvTS::LOG, "Feature detector is empty" );
            ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
            return;
        }

        Mat image = imread( imgFilename, 0 );
        if( image.empty() )
        {
            ts->printf( CvTS::LOG, "image %s can not be read \n", imgFilename.c_str() );
            ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
            return;
        }

        FileStorage fs( resFilename, FileStorage::READ );

        vector<KeyPoint> calcKeypoints;
        fdetector->detect( image, calcKeypoints );

        if( fs.isOpened() ) // compare computed and valid keypoints
        {
            // TODO compare saved feature detector params with current ones
            vector<KeyPoint> validKeypoints;
            read( fs["keypoints"], validKeypoints );
            if( validKeypoints.empty() )
            {
                ts->printf( CvTS::LOG, "Keypoints can nod be read\n" );
                ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
                return;
            }

            int progress = 0, progressCount = validKeypoints.size() * calcKeypoints.size();
            int badPointCount = 0, commonPointCount = max(validKeypoints.size(), calcKeypoints.size());
            for( size_t v = 0; v < validKeypoints.size(); v++ )
            {
                int nearestIdx = -1;
                float minDist = std::numeric_limits<float>::max();

                for( size_t c = 0; c < calcKeypoints.size(); c++ )
                {
                    progress = update_progress( progress, v*calcKeypoints.size() + c, progressCount, 0 );
                    float curDist = norm( calcKeypoints[c].pt - validKeypoints[v].pt );
                    if( curDist < minDist )
                    {
                        minDist = curDist;
                        nearestIdx = c;
                    }
                }

                if( minDist > maxPtDif ||
                    fabs(calcKeypoints[nearestIdx].size - validKeypoints[v].size) > maxSizeDif ||
                    abs(calcKeypoints[nearestIdx].angle - validKeypoints[v].angle) > maxAngleDif ||
                    abs(calcKeypoints[nearestIdx].response - validKeypoints[v].response) > maxResponseDif ||
                    calcKeypoints[nearestIdx].octave != validKeypoints[v].octave

                    // TODO !!!!!!!
                    /*||
                    calcKeypoints[nearestIdx].class_id != validKeypoints[v].class_id*/ )
                {
                    badPointCount++;
                }
            }
            ts->printf( CvTS::LOG, "badPointCount = %d; validPointCount = %d; calcPointCount = %d\n",
                        badPointCount, validKeypoints.size(), calcKeypoints.size() );
            if( badPointCount > 0.9 * commonPointCount )
            {
                ts->printf( CvTS::LOG, "Bad accuracy!\n" );
                ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
                return;
            }
        }
        else // write
        {
            fs.open( resFilename, FileStorage::WRITE );
            if( !fs.isOpened() )
            {
                ts->printf( CvTS::LOG, "file %s can not be opened to write\n", resFilename.c_str() );
                ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
                return;
            }
            else
            {
                fs << "detector_params" << "{";
                fdetector->write( fs );
                fs << "}";

                write( fs, "keypoints", calcKeypoints );
            }
        }
        ts->set_failed_test_info( CvTS::OK );
    }

    Ptr<FeatureDetector> fdetector;
};

CV_FeatureDetectorTest fastTest( "detector_fast", createFeatureDetector("FAST") );
CV_FeatureDetectorTest gfttTest( "detector_gftt", createFeatureDetector("GFTT") );
CV_FeatureDetectorTest harrisTest( "detector_harris", createFeatureDetector("HARRIS") );
CV_FeatureDetectorTest mserTest( "detector_mser", createFeatureDetector("MSER") );
CV_FeatureDetectorTest siftTest( "detector_sift", createFeatureDetector("SIFT") );
CV_FeatureDetectorTest starTest( "detector_star", createFeatureDetector("STAR") );
//CV_FeatureDetectorTest surfTest( "detector_surf", createFeatureDetector("SURF") );

/****************************************************************************************\
*                     Regression tests for descriptor extractors.                        *
\****************************************************************************************/
static void writeMatInBin( const Mat& mat, const string& filename )
{
    FILE* f = fopen( filename.c_str(), "wb");
    if( f )
    {
        int type = mat.type();
        fwrite( (void*)&mat.rows, sizeof(int), 1, f );
        fwrite( (void*)&mat.cols, sizeof(int), 1, f );
        fwrite( (void*)&type, sizeof(int), 1, f );
        fwrite( (void*)&mat.step, sizeof(int), 1, f );
        fwrite( (void*)mat.data, 1, mat.step*mat.rows, f );
        fclose(f);
    }
}

static Mat readMatFromBin( const string& filename )
{
    FILE* f = fopen( filename.c_str(), "rb" );
    if( f )
    {
        int rows, cols, type, step;
        fread( (void*)&rows, sizeof(int), 1, f );
        fread( (void*)&cols, sizeof(int), 1, f );
        fread( (void*)&type, sizeof(int), 1, f );
        fread( (void*)&step, sizeof(int), 1, f );

        uchar* data = (uchar*)cvAlloc(step*rows);
        fread( (void*)data, 1, step*rows, f );
        fclose(f);

        return Mat( rows, cols, type, data );
    }
    return Mat();
}

class CV_DescriptorExtractorTest : public CvTest
{
public:
    CV_DescriptorExtractorTest( const char* testName, float _normDif, const Ptr<DescriptorExtractor>& _dextractor, float _prevTime  ) :
            CvTest( testName, "cv::DescriptorExtractor::compute" ), normDif(_normDif), prevTime(_prevTime), dextractor(_dextractor) {}
protected:
    virtual void createDescriptorExtractor() {}

    void run(int)
    {
        createDescriptorExtractor();

        if( dextractor.empty() )
        {
            ts->printf(CvTS::LOG, "Descriptor extractor is empty\n");
            ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
            return;
        }

        string imgFilename =  string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;
        Mat img = imread( imgFilename, 0 );
        if( img.empty() )
        {
            ts->printf( CvTS::LOG, "image %s can not be read\n", imgFilename.c_str() );
            ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
            return;
        }

        vector<KeyPoint> keypoints;
        FileStorage fs( string(ts->get_data_path()) + FEATURES2D_DIR + "/keypoints.xml.gz", FileStorage::READ );
        if( fs.isOpened() )
            read( fs.getFirstTopLevelNode(), keypoints );
        else
        {
            ts->printf( CvTS::LOG, "Compute and write keypoints\n" );
            fs.open( string(ts->get_data_path()) + FEATURES2D_DIR + "/keypoints.xml.gz", FileStorage::WRITE );
            if( fs.isOpened() )
            {
                SurfFeatureDetector fd;
                fd.detect(img, keypoints);
                write( fs, "keypoints", keypoints );
            }
            else
            {
                ts->printf(CvTS::LOG, "File for writting keypoints can not be opened\n");
                ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
                return;
            }
        }

        Mat calcDescriptors;
        double t = (double)getTickCount();
        dextractor->compute( img, keypoints, calcDescriptors );
        t = getTickCount() - t;
        ts->printf(CvTS::LOG, "\nAverage time of computiting one descriptor = %g ms (previous time = %g ms)\n", t/((double)cvGetTickFrequency()*1000.)/calcDescriptors.rows, prevTime );

        // TODO read and write descriptor extractor parameters and check them
        Mat validDescriptors = readDescriptors();
        if( !validDescriptors.empty() )
        {
            double normVal = norm( calcDescriptors, validDescriptors, NORM_INF );
            ts->printf( CvTS::LOG, "nofm (inf) BTW valid and calculated float descriptors = %f\n", normVal );
            if( normVal > normDif )
                ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
        }
        else
        {
            if( !writeDescriptors( calcDescriptors ) )
            {
                ts->printf( CvTS::LOG, "Descriptors can not be written\n" );
                ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
                return;
            }
        }
    }

    virtual Mat readDescriptors()
    {
        Mat res = readMatFromBin( string(ts->get_data_path()) + DESCRIPTOR_DIR + "/" + string(name) );
        return res;
    }

    virtual bool writeDescriptors( Mat& descs )
    {
        writeMatInBin( descs,  string(ts->get_data_path()) + DESCRIPTOR_DIR + "/" + string(name) );
        return true;
    }

    const float normDif;
    const float prevTime;

    Ptr<DescriptorExtractor> dextractor;
};

template<typename T>
class CV_CalonderDescriptorExtractorTest : public CV_DescriptorExtractorTest
{
public:
    CV_CalonderDescriptorExtractorTest( const char* testName, float _normDif, float _prevTime ) :
            CV_DescriptorExtractorTest( testName, _normDif, Ptr<DescriptorExtractor>(), _prevTime )
    {}

    virtual void createDescriptorExtractor()
    {
        dextractor = new CalonderDescriptorExtractor<T>( string(ts->get_data_path()) + FEATURES2D_DIR + "/calonder_classifier.rtc");
    }
};

//CV_DescriptorExtractorTest siftDescriptorTest( "descriptor_sift", 0.001f,
//                                                createDescriptorExtractor("SIFT"), 8.06652f  );
//CV_DescriptorExtractorTest surfDescriptorTest( "descriptor_surf",  0.004f,
//                                                createDescriptorExtractor("SURF"), 0.147372f );
//CV_DescriptorExtractorTest siftDescriptorTest( "descriptor_opponent_sift", 0.001f,
//                                                createDescriptorExtractor("OpponentSIFT"), 8.06652f  );
//CV_DescriptorExtractorTest surfDescriptorTest( "descriptor_opponent_surf",  0.004f,
//                                                createDescriptorExtractor("OpponentSURF"), 0.147372f );

#if CV_SSE2
CV_CalonderDescriptorExtractorTest<uchar> ucharCalonderTest( "descriptor_calonder_uchar",
                                                             std::numeric_limits<float>::epsilon() + 1,
                                                             0.0132175f );
CV_CalonderDescriptorExtractorTest<float> floatCalonderTest( "descriptor_calonder_float",
                                                             std::numeric_limits<float>::epsilon(),
                                                             0.0221308f );
#endif // CV_SSE2
