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
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

//#define GET_STAT

#define DIST_E              "distE"
#define S_E                 "sE"
#define NO_PAIR_E           "noPairE"
//#define TOTAL_NO_PAIR_E     "totalNoPairE"

#define DETECTOR_NAMES      "detector_names"
#define DETECTORS           "detectors"
#define IMAGE_FILENAMES     "image_filenames"
#define VALIDATION          "validation"
#define FILENAME            "fn"

#define C_SCALE_CASCADE     "scale_cascade"

class CV_DetectorTest : public cvtest::BaseTest
{
public:
    CV_DetectorTest();
protected:
    virtual int prepareData( FileStorage& fs );
    virtual void run( int startFrom );
    virtual string& getValidationFilename();

    virtual void readDetector( const FileNode& fn ) = 0;
    virtual void writeDetector( FileStorage& fs, int di ) = 0;
    int runTestCase( int detectorIdx, vector<vector<Rect> >& objects );
    virtual int detectMultiScale( int di, const Mat& img, vector<Rect>& objects ) = 0;
    int validate( int detectorIdx, vector<vector<Rect> >& objects );

    struct
    {
        float dist;
        float s;
        float noPair;
        //float totalNoPair;
    } eps;
    vector<string> detectorNames;
    vector<string> detectorFilenames;
    vector<string> imageFilenames;
    vector<Mat> images;
    string validationFilename;
    string configFilename;
    FileStorage validationFS;
    bool write_results;
};

CV_DetectorTest::CV_DetectorTest()
{
    configFilename = "dummy";
    write_results = false;
}

string& CV_DetectorTest::getValidationFilename()
{
    return validationFilename;
}

int CV_DetectorTest::prepareData( FileStorage& _fs )
{
    if( !_fs.isOpened() )
        test_case_count = -1;
    else
    {
        FileNode fn = _fs.getFirstTopLevelNode();

        fn[DIST_E] >> eps.dist;
        fn[S_E] >> eps.s;
        fn[NO_PAIR_E] >> eps.noPair;
//        fn[TOTAL_NO_PAIR_E] >> eps.totalNoPair;

        // read detectors
        if( fn[DETECTOR_NAMES].node->data.seq != 0 )
        {
            FileNodeIterator it = fn[DETECTOR_NAMES].begin();
            for( ; it != fn[DETECTOR_NAMES].end(); )
            {
                string _name;
                it >> _name;
                detectorNames.push_back(_name);
                readDetector(fn[DETECTORS][_name]);
            }
        }
        test_case_count = (int)detectorNames.size();

        // read images filenames and images
        string dataPath = ts->get_data_path();
        if( fn[IMAGE_FILENAMES].node->data.seq != 0 )
        {
            for( FileNodeIterator it = fn[IMAGE_FILENAMES].begin(); it != fn[IMAGE_FILENAMES].end(); )
            {
                string filename;
                it >> filename;
                imageFilenames.push_back(filename);
                Mat img = imread( dataPath+filename, 1 );
                images.push_back( img );
            }
        }
    }
    return cvtest::TS::OK;
}

void CV_DetectorTest::run( int )
{
    string dataPath = ts->get_data_path();
    string vs_filename = dataPath + getValidationFilename();

    write_results = !validationFS.open( vs_filename, FileStorage::READ );

    int code;
    if( !write_results )
    {
        code = prepareData( validationFS );
    }
    else
    {
        FileStorage fs0(dataPath + configFilename, FileStorage::READ );
        code = prepareData(fs0);
    }

    if( code < 0 )
    {
        ts->set_failed_test_info( code );
        return;
    }

    if( write_results )
    {
        validationFS.release();
        validationFS.open( vs_filename, FileStorage::WRITE );
        validationFS << FileStorage::getDefaultObjectName(validationFilename) << "{";

        validationFS << DIST_E << eps.dist;
        validationFS << S_E << eps.s;
        validationFS << NO_PAIR_E << eps.noPair;
    //    validationFS << TOTAL_NO_PAIR_E << eps.totalNoPair;

        // write detector names
        validationFS << DETECTOR_NAMES << "[";
        vector<string>::const_iterator nit = detectorNames.begin();
        for( ; nit != detectorNames.end(); ++nit )
        {
            validationFS << *nit;
        }
        validationFS << "]"; // DETECTOR_NAMES

        // write detectors
        validationFS << DETECTORS << "{";
        assert( detectorNames.size() == detectorFilenames.size() );
        nit = detectorNames.begin();
        for( int di = 0; nit != detectorNames.end(); ++nit, di++ )
        {
            validationFS << *nit << "{";
            writeDetector( validationFS, di );
            validationFS << "}";
        }
        validationFS << "}";

        // write image filenames
        validationFS << IMAGE_FILENAMES << "[";
        vector<string>::const_iterator it = imageFilenames.begin();
        for( int ii = 0; it != imageFilenames.end(); ++it, ii++ )
        {
            char buf[10];
            sprintf( buf, "%s%d", "img_", ii );
            cvWriteComment( validationFS.fs, buf, 0 );
            validationFS << *it;
        }
        validationFS << "]"; // IMAGE_FILENAMES

        validationFS << VALIDATION << "{";
    }

    int progress = 0;
    for( int di = 0; di < test_case_count; di++ )
    {
        progress = update_progress( progress, di, test_case_count, 0 );
        if( write_results )
            validationFS << detectorNames[di] << "{";
        vector<vector<Rect> > objects;
        int temp_code = runTestCase( di, objects );

        if (!write_results && temp_code == cvtest::TS::OK)
            temp_code = validate( di, objects );

        if (temp_code != cvtest::TS::OK)
            code = temp_code;

        if( write_results )
            validationFS << "}"; // detectorNames[di]
    }

    if( write_results )
    {
        validationFS << "}"; // VALIDATION
        validationFS << "}"; // getDefaultObjectName
    }

    if ( test_case_count <= 0 || imageFilenames.size() <= 0 )
    {
        ts->printf( cvtest::TS::LOG, "validation file is not determined or not correct" );
        code = cvtest::TS::FAIL_INVALID_TEST_DATA;
    }
    ts->set_failed_test_info( code );
}

int CV_DetectorTest::runTestCase( int detectorIdx, vector<vector<Rect> >& objects )
{
    string dataPath = ts->get_data_path(), detectorFilename;
    if( !detectorFilenames[detectorIdx].empty() )
        detectorFilename = dataPath + detectorFilenames[detectorIdx];

    for( int ii = 0; ii < (int)imageFilenames.size(); ++ii )
    {
        vector<Rect> imgObjects;
        Mat image = images[ii];
        if( image.empty() )
        {
            char msg[30];
            sprintf( msg, "%s %d %s", "image ", ii, " can not be read" );
            ts->printf( cvtest::TS::LOG, msg );
            return cvtest::TS::FAIL_INVALID_TEST_DATA;
        }
        int code = detectMultiScale( detectorIdx, image, imgObjects );
        if( code != cvtest::TS::OK )
            return code;

        objects.push_back( imgObjects );

        if( write_results )
        {
            char buf[10];
            sprintf( buf, "%s%d", "img_", ii );
            string imageIdxStr = buf;
            validationFS << imageIdxStr << "[:";
            for( vector<Rect>::const_iterator it = imgObjects.begin();
                    it != imgObjects.end(); ++it )
            {
                validationFS << it->x << it->y << it->width << it->height;
            }
            validationFS << "]"; // imageIdxStr
        }
    }
    return cvtest::TS::OK;
}


bool isZero( uchar i ) {return i == 0;}

int CV_DetectorTest::validate( int detectorIdx, vector<vector<Rect> >& objects )
{
    assert( imageFilenames.size() == objects.size() );
    int imageIdx = 0;
    int totalNoPair = 0, totalValRectCount = 0;

    for( vector<vector<Rect> >::const_iterator it = objects.begin();
        it != objects.end(); ++it, imageIdx++ ) // for image
    {
        Size imgSize = images[imageIdx].size();
        float dist = min(imgSize.height, imgSize.width) * eps.dist;
        float wDiff = imgSize.width * eps.s;
        float hDiff = imgSize.height * eps.s;

        int noPair = 0;

        // read validation rectangles
        char buf[10];
        sprintf( buf, "%s%d", "img_", imageIdx );
        string imageIdxStr = buf;
        FileNode node = validationFS.getFirstTopLevelNode()[VALIDATION][detectorNames[detectorIdx]][imageIdxStr];
        vector<Rect> valRects;
        if( node.node->data.seq != 0 )
        {
            for( FileNodeIterator it2 = node.begin(); it2 != node.end(); )
            {
                Rect r;
                it2 >> r.x >> r.y >> r.width >> r.height;
                valRects.push_back(r);
            }
        }
        totalValRectCount += (int)valRects.size();

        // compare rectangles
        vector<uchar> map(valRects.size(), 0);
        for( vector<Rect>::const_iterator cr = it->begin();
            cr != it->end(); ++cr )
        {
            // find nearest rectangle
            Point2f cp1 = Point2f( cr->x + (float)cr->width/2.0f, cr->y + (float)cr->height/2.0f );
            int minIdx = -1, vi = 0;
            float minDist = (float)norm( Point(imgSize.width, imgSize.height) );
            for( vector<Rect>::const_iterator vr = valRects.begin();
                vr != valRects.end(); ++vr, vi++ )
            {
                Point2f cp2 = Point2f( vr->x + (float)vr->width/2.0f, vr->y + (float)vr->height/2.0f );
                float curDist = (float)norm(cp1-cp2);
                if( curDist < minDist )
                {
                    minIdx = vi;
                    minDist = curDist;
                }
            }
            if( minIdx == -1 )
            {
                noPair++;
            }
            else
            {
                Rect vr = valRects[minIdx];
                if( map[minIdx] != 0 || (minDist > dist) || (abs(cr->width - vr.width) > wDiff) ||
                                                        (abs(cr->height - vr.height) > hDiff) )
                    noPair++;
                else
                    map[minIdx] = 1;
            }
        }
        noPair += (int)count_if( map.begin(), map.end(), isZero );
        totalNoPair += noPair;

        EXPECT_LE(noPair, cvRound(valRects.size()*eps.noPair)+1)
            << "detector " << detectorNames[detectorIdx] << " has overrated count of rectangles without pair on "
            << imageFilenames[imageIdx] << " image";

        if (::testing::Test::HasFailure())
            break;
    }

    EXPECT_LE(totalNoPair, cvRound(totalValRectCount*eps./*total*/noPair)+1)
        << "detector " << detectorNames[detectorIdx] << " has overrated count of rectangles without pair on all images set";

    if (::testing::Test::HasFailure())
        return cvtest::TS::FAIL_BAD_ACCURACY;

    return cvtest::TS::OK;
}

//----------------------------------------------- CascadeDetectorTest -----------------------------------
class CV_CascadeDetectorTest : public CV_DetectorTest
{
public:
    CV_CascadeDetectorTest();
protected:
    virtual void readDetector( const FileNode& fn );
    virtual void writeDetector( FileStorage& fs, int di );
    virtual int detectMultiScale( int di, const Mat& img, vector<Rect>& objects );
    virtual int detectMultiScale_C( const string& filename, int di, const Mat& img, vector<Rect>& objects );
    vector<int> flags;
};

CV_CascadeDetectorTest::CV_CascadeDetectorTest()
{
    validationFilename = "cascadeandhog/cascade.xml";
    configFilename = "cascadeandhog/_cascade.xml";
}

void CV_CascadeDetectorTest::readDetector( const FileNode& fn )
{
    string filename;
    int flag;
    fn[FILENAME] >> filename;
    detectorFilenames.push_back(filename);
    fn[C_SCALE_CASCADE] >> flag;
    if( flag )
        flags.push_back( 0 );
    else
        flags.push_back( CV_HAAR_SCALE_IMAGE );
}

void CV_CascadeDetectorTest::writeDetector( FileStorage& fs, int di )
{
    int sc = flags[di] & CV_HAAR_SCALE_IMAGE ? 0 : 1;
    fs << FILENAME << detectorFilenames[di];
    fs << C_SCALE_CASCADE << sc;
}


int CV_CascadeDetectorTest::detectMultiScale_C( const string& filename,
                                                int di, const Mat& img,
                                                vector<Rect>& objects )
{
    Ptr<CvHaarClassifierCascade> c_cascade = cvLoadHaarClassifierCascade(filename.c_str(), cvSize(0,0));
    Ptr<CvMemStorage> storage = cvCreateMemStorage();

    if( c_cascade.empty() )
    {
        ts->printf( cvtest::TS::LOG, "cascade %s can not be opened");
        return cvtest::TS::FAIL_INVALID_TEST_DATA;
    }
    Mat grayImg;
    cvtColor( img, grayImg, CV_BGR2GRAY );
    equalizeHist( grayImg, grayImg );

    CvMat c_gray = grayImg;
    CvSeq* rs = cvHaarDetectObjects(&c_gray, c_cascade, storage, 1.1, 3, flags[di] );

    objects.clear();
    for( int i = 0; i < rs->total; i++ )
    {
        Rect r = *(Rect*)cvGetSeqElem(rs, i);
        objects.push_back(r);
    }

    return cvtest::TS::OK;
}

int CV_CascadeDetectorTest::detectMultiScale( int di, const Mat& img,
                                              vector<Rect>& objects)
{
    string dataPath = ts->get_data_path(), filename;
    filename = dataPath + detectorFilenames[di];
    const string pattern = "haarcascade_frontalface_default.xml";

    if( filename.size() >= pattern.size() &&
        strcmp(filename.c_str() + (filename.size() - pattern.size()),
              pattern.c_str()) == 0 )
        return detectMultiScale_C(filename, di, img, objects);

    CascadeClassifier cascade( filename );
    if( cascade.empty() )
    {
        ts->printf( cvtest::TS::LOG, "cascade %s can not be opened");
        return cvtest::TS::FAIL_INVALID_TEST_DATA;
    }
    Mat grayImg;
    cvtColor( img, grayImg, CV_BGR2GRAY );
    equalizeHist( grayImg, grayImg );
    cascade.detectMultiScale( grayImg, objects, 1.1, 3, flags[di] );
    return cvtest::TS::OK;
}

//----------------------------------------------- HOGDetectorTest -----------------------------------
class CV_HOGDetectorTest : public CV_DetectorTest
{
public:
    CV_HOGDetectorTest();
protected:
    virtual void readDetector( const FileNode& fn );
    virtual void writeDetector( FileStorage& fs, int di );
    virtual int detectMultiScale( int di, const Mat& img, vector<Rect>& objects );
};

CV_HOGDetectorTest::CV_HOGDetectorTest()
{
    validationFilename = "cascadeandhog/hog.xml";
}

void CV_HOGDetectorTest::readDetector( const FileNode& fn )
{
    string filename;
    if( fn[FILENAME].node->data.seq != 0 )
        fn[FILENAME] >> filename;
    detectorFilenames.push_back( filename);
}

void CV_HOGDetectorTest::writeDetector( FileStorage& fs, int di )
{
    fs << FILENAME << detectorFilenames[di];
}

int CV_HOGDetectorTest::detectMultiScale( int di, const Mat& img,
                                              vector<Rect>& objects)
{
    HOGDescriptor hog;
    if( detectorFilenames[di].empty() )
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    else
        assert(0);
    hog.detectMultiScale(img, objects);
    return cvtest::TS::OK;
}

//----------------------------------------------- HOGDetectorReadWriteTest -----------------------------------
TEST(Objdetect_HOGDetectorReadWrite, regression)
{
    // Inspired by bug #2607
    Mat img;
    img = imread(cvtest::TS::ptr()->get_data_path() + "/cascadeandhog/images/karen-and-rob.png");
    ASSERT_FALSE(img.empty());

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    string tempfilename = cv::tempfile(".xml");
    FileStorage fs(tempfilename, FileStorage::WRITE);
    hog.write(fs, "myHOG");

    fs.open(tempfilename, FileStorage::READ);
    remove(tempfilename.c_str());

    FileNode n = fs["opencv_storage"]["myHOG"];

    ASSERT_NO_THROW(hog.read(n));
}



TEST(Objdetect_CascadeDetector, regression) { CV_CascadeDetectorTest test; test.safe_run(); }
TEST(Objdetect_HOGDetector, regression) { CV_HOGDetectorTest test; test.safe_run(); }
