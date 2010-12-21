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
#include "cvchessboardgenerator.h"

#include <limits>
#include <numeric>

using namespace std;
using namespace cv;

#define _L2_ERR

void show_points( const Mat& gray, const Mat& u, const vector<Point2f>& v, Size pattern_size, bool was_found )
{
    Mat rgb( gray.size(), CV_8U);
    merge(vector<Mat>(3, gray), rgb);
        
    for(size_t i = 0; i < v.size(); i++ )
        circle( rgb, v[i], 3, CV_RGB(255, 0, 0), CV_FILLED);            

    if( !u.empty() )
    {
        const Point2f* u_data = u.ptr<Point2f>();
        size_t count = u.cols * u.rows;
        for(size_t i = 0; i < count; i++ )
            circle( rgb, u_data[i], 3, CV_RGB(0, 255, 0), CV_FILLED);
    }
    if (!v.empty())
    {
        Mat corners((int)v.size(), 1, CV_32FC2, (void*)&v[0]);     
        drawChessboardCorners( rgb, pattern_size, corners, was_found );
    }
    //namedWindow( "test", 0 ); imshow( "test", rgb ); waitKey(0);
}


enum Pattern { CHESSBOARD, CIRCLES_GRID };

class CV_ChessboardDetectorTest : public CvTest
{
public:
    CV_ChessboardDetectorTest( Pattern pattern, const char* testName, const char* funcName );
protected:
    void run(int);
    void run_batch(const string& filename);
    bool checkByGenerator();

    Pattern pattern;
};

CV_ChessboardDetectorTest::CV_ChessboardDetectorTest( Pattern _pattern, const char* testName, const char* funcName ):
    CvTest( testName, funcName )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    pattern = _pattern;
}

double calcError(const vector<Point2f>& v, const Mat& u)
{
    int count_exp = u.cols * u.rows;
    const Point2f* u_data = u.ptr<Point2f>();

    double err = numeric_limits<double>::max();
    for( int k = 0; k < 2; ++k )
    {
        double err1 = 0;
        for( int j = 0; j < count_exp; ++j )
        {
            int j1 = k == 0 ? j : count_exp - j - 1;
            double dx = fabs( v[j].x - u_data[j1].x );
            double dy = fabs( v[j].y - u_data[j1].y );

#if defined(_L2_ERR)
            err1 += dx*dx + dy*dy;
#else
            dx = MAX( dx, dy );
            if( dx > err1 )
                err1 = dx;
#endif //_L2_ERR
            //printf("dx = %f\n", dx);
        }
        //printf("\n");
        err = min(err, err1);
    }
    
#if defined(_L2_ERR)
    err = sqrt(err/count_exp);
#endif //_L2_ERR
    
    return err;
}

const double rough_success_error_level = 2.5;
const double precise_success_error_level = 2;


/* ///////////////////// chess_corner_test ///////////////////////// */
void CV_ChessboardDetectorTest::run( int /*start_from */)
{
    /*if (!checkByGenerator())
        return;*/
    switch( pattern )
    {
        case CHESSBOARD:
            checkByGenerator();
            run_batch("chessboard_list.dat");
            run_batch("chessboard_list_subpixel.dat");
            break;
        case CIRCLES_GRID:
            run_batch("circles_list.dat");
            break;
    }
}

void CV_ChessboardDetectorTest::run_batch( const string& filename )
{
    CvTS& ts = *this->ts;
    ts.set_failed_test_info( CvTS::OK );

    ts.printf(CvTS::LOG, "\nRunning batch %s\n", filename.c_str());
//#define WRITE_POINTS 1
#ifndef WRITE_POINTS    
    double max_rough_error = 0, max_precise_error = 0;
#endif
    string folder;
    switch( pattern )
    {
        case CHESSBOARD:
            folder = string(ts.get_data_path()) + "cameracalibration/";
            break;
        case CIRCLES_GRID:
            folder = string(ts.get_data_path()) + "cameracalibration/circles/";
            break;
    }

    FileStorage fs( folder + filename, FileStorage::READ );
    FileNode board_list = fs["boards"];
        
    if( !fs.isOpened() || board_list.empty() || !board_list.isSeq() || board_list.size() % 2 != 0 )
    {
        ts.printf( CvTS::LOG, "%s can not be readed or is not valid\n", (folder + filename).c_str() );
        ts.printf( CvTS::LOG, "fs.isOpened=%d, board_list.empty=%d, board_list.isSeq=%d,board_list.size()%2=%d\n", 
            fs.isOpened(), (int)board_list.empty(), board_list.isSeq(), board_list.size()%2);
        ts.set_failed_test_info( CvTS::FAIL_MISSING_TEST_DATA );        
        return;
    }

    int progress = 0;
    int max_idx = board_list.node->data.seq->total/2;
    double sum_error = 0.0;
    int count = 0;

    for(int idx = 0; idx < max_idx; ++idx )
    {
        ts.update_context( this, idx, true );
        
        /* read the image */
        string img_file = board_list[idx * 2];                    
        Mat gray = imread( folder + img_file, 0);
                
        if( gray.empty() )
        {
            ts.printf( CvTS::LOG, "one of chessboard images can't be read: %s\n", img_file.c_str() );
            ts.set_failed_test_info( CvTS::FAIL_MISSING_TEST_DATA );
            continue;
        }

        string filename = folder + (string)board_list[idx * 2 + 1];
        Mat expected;
        {
            CvMat *u = (CvMat*)cvLoad( filename.c_str() );
            if(!u )
            {                
                ts.printf( CvTS::LOG, "one of chessboard corner files can't be read: %s\n", filename.c_str() ); 
                ts.set_failed_test_info( CvTS::FAIL_MISSING_TEST_DATA );
                continue;                
            }
            expected = Mat(u, true);
            cvReleaseMat( &u );
        }                
        size_t count_exp = static_cast<size_t>(expected.cols * expected.rows);                
        Size pattern_size = expected.size();

        vector<Point2f> v;
        bool result;
        switch( pattern )
        {
            case CHESSBOARD:
                result = findChessboardCorners(gray, pattern_size, v, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
                break;
            case CIRCLES_GRID:
                result = findCirclesGrid(gray, pattern_size, v);
                break;
        }
        show_points( gray, Mat(), v, pattern_size, result );
        if( !result || v.size() != count_exp )
        {
            ts.printf( CvTS::LOG, "chessboard is not found in %s\n", img_file.c_str() );
            ts.set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
            continue;
        }

#ifndef WRITE_POINTS
        double err = calcError(v, expected);
#if 0
        if( err > rough_success_error_level )
        {
            ts.printf( CvTS::LOG, "bad accuracy of corner guesses\n" );
            ts.set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
            continue;
        }
#endif
        max_rough_error = MAX( max_rough_error, err );
#endif
        if( pattern == CHESSBOARD )
            cornerSubPix( gray, v, Size(5, 5), Size(-1,-1), TermCriteria(TermCriteria::EPS|TermCriteria::MAX_ITER, 30, 0.1));
        //find4QuadCornerSubpix(gray, v, Size(5, 5));
        show_points( gray, expected, v, pattern_size, result  );

#ifndef WRITE_POINTS
//        printf("called find4QuadCornerSubpix\n");
        err = calcError(v, expected);
        sum_error += err;
        count++;
#if 1
        if( err > precise_success_error_level )
        {
            ts.printf( CvTS::LOG, "Image %s: bad accuracy of adjusted corners %f\n", img_file.c_str(), err ); 
            ts.set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
            continue;
        }
#endif
        ts.printf(CvTS::LOG, "Error on %s is %f\n", img_file.c_str(), err);
        max_precise_error = MAX( max_precise_error, err );
#else
        Mat mat_v(pattern_size, CV_32FC2, (void*)&v[0]);
        CvMat cvmat_v = mat_v;
        cvSave( filename.c_str(), &cvmat_v );
#endif
        progress = update_progress( progress, idx, max_idx, 0 );
    }    
    
    sum_error /= count;
    ts.printf(CvTS::LOG, "Average error is %f\n", sum_error);
}

double calcErrorMinError(const Size& cornSz, const vector<Point2f>& corners_found, const vector<Point2f>& corners_generated)
{
    Mat m1(cornSz, CV_32FC2, (Point2f*)&corners_generated[0]);    
    Mat m2; flip(m1, m2, 0);

    Mat m3; flip(m1, m3, 1); m3 = m3.t(); flip(m3, m3, 1);
    
    Mat m4 = m1.t(); flip(m4, m4, 1);

    double min1 =  min(calcError(corners_found, m1), calcError(corners_found, m2));    
    double min2 =  min(calcError(corners_found, m3), calcError(corners_found, m4));    
    return min(min1, min2);
}

bool validateData(const ChessBoardGenerator& cbg, const Size& imgSz, 
                  const vector<Point2f>& corners_generated)
{
    Size cornersSize = cbg.cornersSize();
    Mat_<Point2f> mat(cornersSize.height, cornersSize.width, (Point2f*)&corners_generated[0]);

    double minNeibDist = std::numeric_limits<double>::max();
    double tmp = 0;
    for(int i = 1; i < mat.rows - 2; ++i)
        for(int j = 1; j < mat.cols - 2; ++j)
        {
            const Point2f& cur = mat(i, j);
            
            tmp = norm( cur - mat(i + 1, j + 1) );
            if (tmp < minNeibDist)
                tmp = minNeibDist;

            tmp = norm( cur - mat(i - 1, j + 1 ) );
            if (tmp < minNeibDist)
                tmp = minNeibDist;

            tmp = norm( cur - mat(i + 1, j - 1) );
            if (tmp < minNeibDist)
                tmp = minNeibDist;

            tmp = norm( cur - mat(i - 1, j - 1) );
            if (tmp < minNeibDist)
                tmp = minNeibDist;
        }

    const double threshold = 0.25;
    double cbsize = (max(cornersSize.width, cornersSize.height) + 1) * minNeibDist;
    int imgsize =  min(imgSz.height, imgSz.width);    
    return imgsize * threshold < cbsize;
}

bool CV_ChessboardDetectorTest::checkByGenerator()
{   
    bool res = true;
    //theRNG() = 0x58e6e895b9913160;
    //cv::DefaultRngAuto dra;
    //theRNG() = *ts->get_rng();

    Mat bg(Size(800, 600), CV_8UC3, Scalar::all(255));  
    randu(bg, Scalar::all(0), Scalar::all(255)); 
    GaussianBlur(bg, bg, Size(7,7), 3.0); 
            
    Mat_<float> camMat(3, 3);
    camMat << 300.f, 0.f, bg.cols/2.f, 0, 300.f, bg.rows/2.f, 0.f, 0.f, 1.f;
    
    Mat_<float> distCoeffs(1, 5);
    distCoeffs << 1.2f, 0.2f, 0.f, 0.f, 0.f;

    const Size sizes[] = { Size(6, 6), Size(8, 6), Size(11, 12),  Size(5, 4) };
    const size_t sizes_num = sizeof(sizes)/sizeof(sizes[0]);                
    const int test_num = 16;    
    int progress = 0;
    for(int i = 0; i < test_num; ++i)
    {          
        progress = update_progress( progress, i, test_num, 0 );
        ChessBoardGenerator cbg(sizes[i % sizes_num]);

        vector<Point2f> corners_generated;

        Mat cb = cbg(bg, camMat, distCoeffs, corners_generated);

        if(!validateData(cbg, cb.size(), corners_generated))
        {
            ts->printf( CvTS::LOG, "Chess board skipped - too small" );
            continue;               
        }

        /*cb = cb * 0.8 + Scalar::all(30);            
        GaussianBlur(cb, cb, Size(3, 3), 0.8);     */
        //cv::addWeighted(cb, 0.8, bg, 0.2, 20, cb); 
        //cv::namedWindow("CB"); cv::imshow("CB", cb); cv::waitKey();
                               
        vector<Point2f> corners_found;
        int flags = i % 8; // need to check branches for all flags
        bool found = findChessboardCorners(cb, cbg.cornersSize(), corners_found, flags);
        if (!found)        
        {            
            ts->printf( CvTS::LOG, "Chess board corners not found\n" );
            ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
            res = false;
            continue;          
        }

        double err = calcErrorMinError(cbg.cornersSize(), corners_found, corners_generated);            
        if( err > rough_success_error_level )
        {
            ts->printf( CvTS::LOG, "bad accuracy of corner guesses" );
            ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
            res = false;
            continue;
        }        
    }  

    /* ***** negative ***** */
    {        
        vector<Point2f> corners_found;
        bool found = findChessboardCorners(bg, Size(8, 7), corners_found);
        if (found)
            res = false;

        ChessBoardGenerator cbg(Size(8, 7));

        vector<Point2f> cg;
        Mat cb = cbg(bg, camMat, distCoeffs, cg);        

        found = findChessboardCorners(cb, Size(3, 4), corners_found);
        if (found)
            res = false;        

        Point2f c = std::accumulate(cg.begin(), cg.end(), Point2f(), plus<Point2f>()) * (1.f/cg.size());

        Mat_<double> aff(2, 3);
        aff << 1.0, 0.0, -(double)c.x, 0.0, 1.0, 0.0;
        Mat sh;
        warpAffine(cb, sh, aff, cb.size());        

        found = findChessboardCorners(sh, cbg.cornersSize(), corners_found);
        if (found)
            res = false;        
        
        vector< vector<Point> > cnts(1);
        vector<Point>& cnt = cnts[0];
        cnt.push_back(cg[  0]); cnt.push_back(cg[0+2]); 
        cnt.push_back(cg[7+0]); cnt.push_back(cg[7+2]);                
        cv::drawContours(cb, cnts, -1, Scalar::all(128), CV_FILLED);

        found = findChessboardCorners(cb, cbg.cornersSize(), corners_found);
        if (found)
            res = false;

        cv::drawChessboardCorners(cb, cbg.cornersSize(), Mat(corners_found), found);
    }
    
    return res;
}

CV_ChessboardDetectorTest chessboard_detector_test ( CHESSBOARD, "chessboard-detector", "cvFindChessboardCorners" );
CV_ChessboardDetectorTest circlesgrid_detector_test ( CIRCLES_GRID, "circlesgrid-detector", "findCirclesGrid" );

/* End of file. */
