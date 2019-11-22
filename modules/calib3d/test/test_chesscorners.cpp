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
#include "test_chessboardgenerator.hpp"

#include <functional>

namespace opencv_test { namespace {

#define _L2_ERR

//#define DEBUG_CHESSBOARD

#ifdef DEBUG_CHESSBOARD
void show_points( const Mat& gray, const Mat& expected, const vector<Point2f>& actual, bool was_found )
{
    Mat rgb( gray.size(), CV_8U);
    merge(vector<Mat>(3, gray), rgb);

    for(size_t i = 0; i < actual.size(); i++ )
        circle( rgb, actual[i], 5, Scalar(0, 0, 200), 1, LINE_AA);

    if( !expected.empty() )
    {
        const Point2f* u_data = expected.ptr<Point2f>();
        size_t count = expected.cols * expected.rows;
        for(size_t i = 0; i < count; i++ )
            circle(rgb, u_data[i], 4, Scalar(0, 240, 0), 1, LINE_AA);
    }
    putText(rgb, was_found ? "FOUND !!!" : "NOT FOUND", Point(5, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 240, 0));
    imshow( "test", rgb ); while ((uchar)waitKey(0) != 'q') {};
}
#else
#define show_points(...)
#endif

enum Pattern { CHESSBOARD,CHESSBOARD_SB,CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID};

class CV_ChessboardDetectorTest : public cvtest::BaseTest
{
public:
    CV_ChessboardDetectorTest( Pattern pattern, int algorithmFlags = 0 );
protected:
    void run(int);
    void run_batch(const string& filename);
    bool checkByGenerator();
    bool checkByGeneratorHighAccuracy();

    // wraps calls based on the given pattern
    bool findChessboardCornersWrapper(InputArray image, Size patternSize, OutputArray corners,int flags);

    Pattern pattern;
    int algorithmFlags;
};

CV_ChessboardDetectorTest::CV_ChessboardDetectorTest( Pattern _pattern, int _algorithmFlags )
{
    pattern = _pattern;
    algorithmFlags = _algorithmFlags;
}

double calcError(const vector<Point2f>& v, const Mat& u)
{
    int count_exp = u.cols * u.rows;
    const Point2f* u_data = u.ptr<Point2f>();

    double err = std::numeric_limits<double>::max();
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
    ts->set_failed_test_info( cvtest::TS::OK );

    /*if (!checkByGenerator())
        return;*/
    switch( pattern )
    {
        case CHESSBOARD_SB:
            checkByGeneratorHighAccuracy();      // not supported by CHESSBOARD
            /* fallthrough */
        case CHESSBOARD:
            checkByGenerator();
            if (ts->get_err_code() != cvtest::TS::OK)
            {
                break;
            }

            run_batch("negative_list.dat");
            if (ts->get_err_code() != cvtest::TS::OK)
            {
                break;
            }

            run_batch("chessboard_list.dat");
            if (ts->get_err_code() != cvtest::TS::OK)
            {
                break;
            }

            run_batch("chessboard_list_subpixel.dat");
            break;
        case CIRCLES_GRID:
            run_batch("circles_list.dat");
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            run_batch("acircles_list.dat");
            break;
    }
}

void CV_ChessboardDetectorTest::run_batch( const string& filename )
{
    ts->printf(cvtest::TS::LOG, "\nRunning batch %s\n", filename.c_str());
//#define WRITE_POINTS 1
#ifndef WRITE_POINTS
    double max_rough_error = 0, max_precise_error = 0;
#endif
    string folder;
    switch( pattern )
    {
        case CHESSBOARD:
        case CHESSBOARD_SB:
            folder = string(ts->get_data_path()) + "cv/cameracalibration/";
            break;
        case CIRCLES_GRID:
            folder = string(ts->get_data_path()) + "cv/cameracalibration/circles/";
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            folder = string(ts->get_data_path()) + "cv/cameracalibration/asymmetric_circles/";
            break;
    }

    FileStorage fs( folder + filename, FileStorage::READ );
    FileNode board_list = fs["boards"];

    if( !fs.isOpened() || board_list.empty() || !board_list.isSeq() || board_list.size() % 2 != 0 )
    {
        ts->printf( cvtest::TS::LOG, "%s can not be read or is not valid\n", (folder + filename).c_str() );
        ts->printf( cvtest::TS::LOG, "fs.isOpened=%d, board_list.empty=%d, board_list.isSeq=%d,board_list.size()%2=%d\n",
            fs.isOpened(), (int)board_list.empty(), board_list.isSeq(), board_list.size()%2);
        ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
        return;
    }

    int progress = 0;
    int max_idx = (int)board_list.size()/2;
    double sum_error = 0.0;
    int count = 0;

    for(int idx = 0; idx < max_idx; ++idx )
    {
        ts->update_context( this, idx, true );

        /* read the image */
        String img_file = board_list[idx * 2];
        Mat gray = imread( folder + img_file, 0);

        if( gray.empty() )
        {
            ts->printf( cvtest::TS::LOG, "one of chessboard images can't be read: %s\n", img_file.c_str() );
            ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
            return;
        }

        String _filename = folder + (String)board_list[idx * 2 + 1];
        bool doesContatinChessboard;
        Mat expected;
        {
            FileStorage fs1(_filename, FileStorage::READ);
            fs1["corners"] >> expected;
            fs1["isFound"] >> doesContatinChessboard;
            fs1.release();
        }
        size_t count_exp = static_cast<size_t>(expected.cols * expected.rows);
        Size pattern_size = expected.size();

        vector<Point2f> v;
        int flags = 0;
        switch( pattern )
        {
            case CHESSBOARD:
                flags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
                break;
            case CIRCLES_GRID:
            case CHESSBOARD_SB:
            case ASYMMETRIC_CIRCLES_GRID:
            default:
                flags = 0;
        }
        bool result = findChessboardCornersWrapper(gray, pattern_size,v,flags);
        if(result ^ doesContatinChessboard || (doesContatinChessboard && v.size() != count_exp))
        {
            ts->printf( cvtest::TS::LOG, "chessboard is detected incorrectly in %s\n", img_file.c_str() );
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            show_points( gray, expected, v, result  );
            return;
        }

        if( result )
        {

#ifndef WRITE_POINTS
            double err = calcError(v, expected);
            max_rough_error = MAX( max_rough_error, err );
#endif
            if( pattern == CHESSBOARD )
                cornerSubPix( gray, v, Size(5, 5), Size(-1,-1), TermCriteria(TermCriteria::EPS|TermCriteria::MAX_ITER, 30, 0.1));
            //find4QuadCornerSubpix(gray, v, Size(5, 5));
            show_points( gray, expected, v, result  );
#ifndef WRITE_POINTS
    //        printf("called find4QuadCornerSubpix\n");
            err = calcError(v, expected);
            sum_error += err;
            count++;
            if( err > precise_success_error_level )
            {
                ts->printf( cvtest::TS::LOG, "Image %s: bad accuracy of adjusted corners %f\n", img_file.c_str(), err );
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                return;
            }
            ts->printf(cvtest::TS::LOG, "Error on %s is %f\n", img_file.c_str(), err);
            max_precise_error = MAX( max_precise_error, err );
#endif
        }
        else
        {
            show_points( gray, Mat(), v, result );
        }

#ifdef WRITE_POINTS
        Mat mat_v(pattern_size, CV_32FC2, (void*)&v[0]);
        FileStorage fs(_filename, FileStorage::WRITE);
        fs << "isFound" << result;
        fs << "corners" << mat_v;
        fs.release();
#endif
        progress = update_progress( progress, idx, max_idx, 0 );
    }

    if (count != 0)
        sum_error /= count;
    ts->printf(cvtest::TS::LOG, "Average error is %f (%d patterns have been found)\n", sum_error, count);
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

            tmp = cv::norm(cur - mat(i + 1, j + 1)); // TODO cvtest
            if (tmp < minNeibDist)
                minNeibDist = tmp;

            tmp = cv::norm(cur - mat(i - 1, j + 1)); // TODO cvtest
            if (tmp < minNeibDist)
                minNeibDist = tmp;

            tmp = cv::norm(cur - mat(i + 1, j - 1)); // TODO cvtest
            if (tmp < minNeibDist)
                minNeibDist = tmp;

            tmp = cv::norm(cur - mat(i - 1, j - 1)); // TODO cvtest
            if (tmp < minNeibDist)
                minNeibDist = tmp;
        }

    const double threshold = 0.25;
    double cbsize = (max(cornersSize.width, cornersSize.height) + 1) * minNeibDist;
    int imgsize =  min(imgSz.height, imgSz.width);
    return imgsize * threshold < cbsize;
}

bool CV_ChessboardDetectorTest::findChessboardCornersWrapper(InputArray image, Size patternSize, OutputArray corners,int flags)
{
    switch(pattern)
    {
    case CHESSBOARD:
        return findChessboardCorners(image,patternSize,corners,flags);
    case CHESSBOARD_SB:
        // check default settings until flags have been specified
        return findChessboardCornersSB(image,patternSize,corners,0);
    case ASYMMETRIC_CIRCLES_GRID:
        flags |= CALIB_CB_ASYMMETRIC_GRID | algorithmFlags;
        return findCirclesGrid(image, patternSize,corners,flags);
    case CIRCLES_GRID:
        flags |= CALIB_CB_SYMMETRIC_GRID;
        return findCirclesGrid(image, patternSize,corners,flags);
    default:
        ts->printf( cvtest::TS::LOG, "Internal Error: unsupported chessboard pattern" );
        ts->set_failed_test_info( cvtest::TS::FAIL_GENERIC);
    }
    return false;
}

bool CV_ChessboardDetectorTest::checkByGenerator()
{
    bool res = true;

    //theRNG() = 0x58e6e895b9913160;
    //cv::DefaultRngAuto dra;
    //theRNG() = *ts->get_rng();

    Mat bg(Size(800, 600), CV_8UC3, Scalar::all(255));
    randu(bg, Scalar::all(0), Scalar::all(255));
    GaussianBlur(bg, bg, Size(5, 5), 0.0);

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
        SCOPED_TRACE(cv::format("test_num=%d", test_num));

        progress = update_progress( progress, i, test_num, 0 );
        ChessBoardGenerator cbg(sizes[i % sizes_num]);

        vector<Point2f> corners_generated;

        Mat cb = cbg(bg, camMat, distCoeffs, corners_generated);

        if(!validateData(cbg, cb.size(), corners_generated))
        {
            ts->printf( cvtest::TS::LOG, "Chess board skipped - too small" );
            continue;
        }

        /*cb = cb * 0.8 + Scalar::all(30);
        GaussianBlur(cb, cb, Size(3, 3), 0.8);     */
        //cv::addWeighted(cb, 0.8, bg, 0.2, 20, cb);
        //cv::namedWindow("CB"); cv::imshow("CB", cb); cv::waitKey();

        vector<Point2f> corners_found;
        int flags = i % 8; // need to check branches for all flags
        bool found = findChessboardCornersWrapper(cb, cbg.cornersSize(), corners_found, flags);
        if (!found)
        {
            ts->printf( cvtest::TS::LOG, "Chess board corners not found\n" );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            res = false;
            return res;
        }

        double err = calcErrorMinError(cbg.cornersSize(), corners_found, corners_generated);
        EXPECT_LE(err, rough_success_error_level) << "bad accuracy of corner guesses";
#if 0
        if (err >= rough_success_error_level)
        {
            imshow("cb", cb);
            Mat cb_corners = cb.clone();
            cv::drawChessboardCorners(cb_corners, cbg.cornersSize(), Mat(corners_found), found);
            imshow("corners", cb_corners);
            waitKey(0);
        }
#endif
    }

    /* ***** negative ***** */
    {
        vector<Point2f> corners_found;
        bool found = findChessboardCornersWrapper(bg, Size(8, 7), corners_found,0);
        if (found)
            res = false;

        ChessBoardGenerator cbg(Size(8, 7));

        vector<Point2f> cg;
        Mat cb = cbg(bg, camMat, distCoeffs, cg);

        found = findChessboardCornersWrapper(cb, Size(3, 4), corners_found,0);
        if (found)
            res = false;

        Point2f c = std::accumulate(cg.begin(), cg.end(), Point2f(), std::plus<Point2f>()) * (1.f/cg.size());

        Mat_<double> aff(2, 3);
        aff << 1.0, 0.0, -(double)c.x, 0.0, 1.0, 0.0;
        Mat sh;
        warpAffine(cb, sh, aff, cb.size());

        found = findChessboardCornersWrapper(sh, cbg.cornersSize(), corners_found,0);
        if (found)
            res = false;

        vector< vector<Point> > cnts(1);
        vector<Point>& cnt = cnts[0];
        cnt.push_back(cg[  0]); cnt.push_back(cg[0+2]);
        cnt.push_back(cg[7+0]); cnt.push_back(cg[7+2]);
        cv::drawContours(cb, cnts, -1, Scalar::all(128), FILLED);

        found = findChessboardCornersWrapper(cb, cbg.cornersSize(), corners_found,0);
        if (found)
            res = false;

        cv::drawChessboardCorners(cb, cbg.cornersSize(), Mat(corners_found), found);
    }

    return res;
}

// generates artificial checkerboards using warpPerspective which supports
// subpixel rendering. The transformation is found by transferring corners to
// the camera image using a virtual plane.
bool CV_ChessboardDetectorTest::checkByGeneratorHighAccuracy()
{
    // draw 2D pattern
    cv::Size pattern_size(6,5);
    int cell_size = 80;
    bool bwhite = true;
    cv::Mat image = cv::Mat::ones((pattern_size.height+3)*cell_size,(pattern_size.width+3)*cell_size,CV_8UC1)*255;
    cv::Mat pimage = image(Rect(cell_size,cell_size,(pattern_size.width+1)*cell_size,(pattern_size.height+1)*cell_size));
    pimage = 0;
    for(int row=0;row<=pattern_size.height;++row)
    {
        int y = int(cell_size*row+0.5F);
        bool bwhite2 = bwhite;
        for(int col=0;col<=pattern_size.width;++col)
        {
            if(bwhite2)
            {
                int x = int(cell_size*col+0.5F);
                pimage(cv::Rect(x,y,cell_size,cell_size)) = 255;
            }
            bwhite2 = !bwhite2;

        }
        bwhite = !bwhite;
    }

    // generate 2d points
    std::vector<Point2f> pts1,pts2,pts1_all,pts2_all;
    std::vector<Point3f> pts3d;
    for(int row=0;row<pattern_size.height;++row)
    {
        int y = int(cell_size*(row+2));
        for(int col=0;col<pattern_size.width;++col)
        {
            int x = int(cell_size*(col+2));
            pts1_all.push_back(cv::Point2f(x-0.5F,y-0.5F));
        }
    }

    // back project chessboard corners to a virtual plane
    double fx = 500;
    double fy = 500;
    cv::Point2f center(250,250);
    double fxi = 1.0/fx;
    double fyi = 1.0/fy;
    for(auto &&pt : pts1_all)
    {
        // calc camera ray
        cv::Vec3f ray(float((pt.x-center.x)*fxi),float((pt.y-center.y)*fyi),1.0F);
        ray /= cv::norm(ray);

        // intersect ray with virtual plane
        cv::Scalar plane(0,0,1,-1);
        cv::Vec3f n(float(plane(0)),float(plane(1)),float(plane(2)));
        cv::Point3f p0(0,0,0);

        cv::Point3f l0(0,0,0);    // camera center in world coordinates
        p0.z = float(-plane(3)/plane(2));
        double val1 = ray.dot(n);
        if(val1 == 0)
        {
            ts->printf( cvtest::TS::LOG, "Internal Error: ray and plane are parallel" );
            ts->set_failed_test_info( cvtest::TS::FAIL_GENERIC);
            return false;
        }
        pts3d.push_back(Point3f(ray/val1*cv::Vec3f((p0-l0)).dot(n))+l0);
    }

    // generate multiple rotations
    for(int i=15;i<90;i=i+15)
    {
        // project 3d points to new camera
        Vec3f rvec(0.0F,0.05F,float(float(i)/180.0*CV_PI));
        Vec3f tvec(0,0,0);
        cv::Mat k = (cv::Mat_<double>(3,3) << fx/2,0,center.x*2, 0,fy/2,center.y, 0,0,1);
        cv::projectPoints(pts3d,rvec,tvec,k,cv::Mat(),pts2_all);

        // get perspective transform using four correspondences and wrap original image
        pts1.clear();
        pts2.clear();
        pts1.push_back(pts1_all[0]);
        pts1.push_back(pts1_all[pattern_size.width-1]);
        pts1.push_back(pts1_all[pattern_size.width*pattern_size.height-1]);
        pts1.push_back(pts1_all[pattern_size.width*(pattern_size.height-1)]);
        pts2.push_back(pts2_all[0]);
        pts2.push_back(pts2_all[pattern_size.width-1]);
        pts2.push_back(pts2_all[pattern_size.width*pattern_size.height-1]);
        pts2.push_back(pts2_all[pattern_size.width*(pattern_size.height-1)]);
        Mat m2 = getPerspectiveTransform(pts1,pts2);
        Mat out(image.size(),image.type());
        warpPerspective(image,out,m2,out.size());

        // find checkerboard
        vector<Point2f> corners_found;
        bool found = findChessboardCornersWrapper(out,pattern_size,corners_found,0);
        if (!found)
        {
            ts->printf( cvtest::TS::LOG, "Chess board corners not found\n" );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return false;
        }
        double err = calcErrorMinError(pattern_size,corners_found,pts2_all);
        if(err > 0.08)
        {
            ts->printf( cvtest::TS::LOG, "bad accuracy of corner guesses" );
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return false;
        }
        //cv::cvtColor(out,out,cv::COLOR_GRAY2BGR);
        //cv::drawChessboardCorners(out,pattern_size,corners_found,true);
        //cv::imshow("img",out);
        //cv::waitKey(-1);
    }
    return true;
}

TEST(Calib3d_ChessboardDetector, accuracy) {  CV_ChessboardDetectorTest test( CHESSBOARD ); test.safe_run(); }
TEST(Calib3d_ChessboardDetector2, accuracy) {  CV_ChessboardDetectorTest test( CHESSBOARD_SB ); test.safe_run(); }
TEST(Calib3d_CirclesPatternDetector, accuracy) { CV_ChessboardDetectorTest test( CIRCLES_GRID ); test.safe_run(); }
TEST(Calib3d_AsymmetricCirclesPatternDetector, accuracy) { CV_ChessboardDetectorTest test( ASYMMETRIC_CIRCLES_GRID ); test.safe_run(); }
#ifdef HAVE_OPENCV_FLANN
TEST(Calib3d_AsymmetricCirclesPatternDetectorWithClustering, accuracy) { CV_ChessboardDetectorTest test( ASYMMETRIC_CIRCLES_GRID, CALIB_CB_CLUSTERING ); test.safe_run(); }
#endif

TEST(Calib3d_CirclesPatternDetectorWithClustering, accuracy)
{
    cv::String dataDir = string(TS::ptr()->get_data_path()) + "cv/cameracalibration/circles/";

    cv::Mat expected;
    FileStorage fs(dataDir + "circles_corners15.dat", FileStorage::READ);
    fs["corners"] >> expected;
    fs.release();

    cv::Mat image = cv::imread(dataDir + "circles15.png");

    std::vector<Point2f> centers;
    cv::findCirclesGrid(image, Size(10, 8), centers, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING);
    ASSERT_EQ(expected.total(), centers.size());

    double error = calcError(centers, expected);
    ASSERT_LE(error, precise_success_error_level);
}

}} // namespace
/* End of file. */
