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

#include <limits>

#include "cvtest.h"
#include "cvchessboardgenerator.h"

using namespace cv;

class CV_ChessboardSubpixelTest : public CvTest
{
public:
    CV_ChessboardSubpixelTest();

protected:
    Mat intrinsic_matrix_;
    Mat distortion_coeffs_;
    Size image_size_;
    
    void run(int);
    void generateIntrinsicParams();
};


int calcDistance(const vector<Point2f>& set1, const vector<Point2f>& set2, double& mean_dist)
{
    if(set1.size() != set2.size())
    {
        return 0;
    }
    
    std::vector<int> indices;
    double sum_dist = 0.0;
    for(size_t i = 0; i < set1.size(); i++)
    {
        double min_dist = std::numeric_limits<double>::max();
        int min_idx = -1;
        
        for(int j = 0; j < (int)set2.size(); j++)
        {
            double dist = norm(set1[i] - set2[j]);
            if(dist < min_dist)
            {
                min_idx = j;
                min_dist = dist;
            }
        }
        
        // check validity of min_idx
        if(min_idx == -1)
        {
            return 0;
        }
        std::vector<int>::iterator it = std::find(indices.begin(), indices.end(), min_idx);
        if(it != indices.end()) 
        {
            // there are two points in set1 corresponding to the same point in set2
            return 0;
        }
        indices.push_back(min_idx);
        
//        printf("dist %d = %f\n", (int)i, min_dist);
        
        sum_dist += min_dist*min_dist;
    }
    
    mean_dist = sqrt(sum_dist/set1.size());
//    printf("sum_dist = %f, set1.size() = %d, mean_dist = %f\n", sum_dist, (int)set1.size(), mean_dist);
    
    return 1;
}

CV_ChessboardSubpixelTest::CV_ChessboardSubpixelTest():
    CvTest( "chessboard-subpixel", "cvFindCornerSubPix" ), 
    intrinsic_matrix_(Size(3, 3), CV_64FC1), distortion_coeffs_(Size(1, 4), CV_64FC1), 
    image_size_(640, 480)
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE | CvTS::TIMING_MODE;
}

/* ///////////////////// chess_corner_test ///////////////////////// */
void CV_ChessboardSubpixelTest::run( int )
{
    int code = CvTS::OK;
    int  progress = 0;

    CvRNG* rng = ts->get_rng();
    
    const int runs_count = 20;
    const int max_pattern_size = 8;
    const int min_pattern_size = 5;
    Mat bg(image_size_, CV_8UC1);
    bg = Scalar(0);
    
    double sum_dist = 0.0;
    int count = 0;
    for(int i = 0; i < runs_count; i++)
    {
        const int pattern_width = min_pattern_size + cvRandInt(rng) % (max_pattern_size - min_pattern_size);
        const int pattern_height = min_pattern_size + cvRandInt(rng) % (max_pattern_size - min_pattern_size);
        Size pattern_size;
        if(pattern_width > pattern_height)
        {
            pattern_size = Size(pattern_height, pattern_width);
        }
        else
        {
            pattern_size = Size(pattern_width, pattern_height);
        }
        ChessBoardGenerator gen_chessboard(Size(pattern_size.width + 1, pattern_size.height + 1));
        
        // generates intrinsic camera and distortion matrices
        generateIntrinsicParams();
        
        vector<Point2f> corners;
        Mat chessboard_image = gen_chessboard(bg, intrinsic_matrix_, distortion_coeffs_, corners); 
        
        vector<Point2f> test_corners;
        bool result = findChessboardCorners(chessboard_image, pattern_size, test_corners, 15);
        if(!result)
        {
#if 0
            ts->printf(CvTS::LOG, "Warning: chessboard was not detected! Writing image to test.jpg\n");
            ts->printf(CvTS::LOG, "Size = %d, %d\n", pattern_size.width, pattern_size.height);
            ts->printf(CvTS::LOG, "Intrinsic params: fx = %f, fy = %f, cx = %f, cy = %f\n", 
                       intrinsic_matrix_.at<double>(0, 0), intrinsic_matrix_.at<double>(1, 1), 
                       intrinsic_matrix_.at<double>(0, 2), intrinsic_matrix_.at<double>(1, 2));
            ts->printf(CvTS::LOG, "Distortion matrix: %f, %f, %f, %f, %f\n", 
                       distortion_coeffs_.at<double>(0, 0), distortion_coeffs_.at<double>(0, 1),
                       distortion_coeffs_.at<double>(0, 2), distortion_coeffs_.at<double>(0, 3), 
                       distortion_coeffs_.at<double>(0, 4));
            
            imwrite("test.jpg", chessboard_image);
#endif
            continue;
        }
        
        double dist1 = 0.0;
        int ret = calcDistance(corners, test_corners, dist1);
        if(ret == 0)
        {
            ts->printf(CvTS::LOG, "findChessboardCorners returns invalid corner coordinates!\n");
            code = CvTS::FAIL_INVALID_OUTPUT;
            break;            
        }
        
        IplImage chessboard_image_header = chessboard_image;
        cvFindCornerSubPix(&chessboard_image_header, (CvPoint2D32f*)&test_corners[0], 
            (int)test_corners.size(), cvSize(3, 3), cvSize(1, 1), cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER,300,0.1));
        find4QuadCornerSubpix(chessboard_image, test_corners, Size(5, 5));
        
        double dist2 = 0.0;
        ret = calcDistance(corners, test_corners, dist2);
        if(ret == 0)
        {
            ts->printf(CvTS::LOG, "findCornerSubpix returns invalid corner coordinates!\n");
            code = CvTS::FAIL_INVALID_OUTPUT;
            break;            
        }
        
        ts->printf(CvTS::LOG, "Error after findChessboardCorners: %f, after findCornerSubPix: %f\n", 
                   dist1, dist2);
        sum_dist += dist2;
        count++;
        
        const double max_reduce_factor = 0.8;
        if(dist1 < dist2*max_reduce_factor)
        {
            ts->printf(CvTS::LOG, "findCornerSubPix increases average error!\n");
            code = CvTS::FAIL_INVALID_OUTPUT;
            break;
        }
        
        progress = update_progress( progress, i-1, runs_count, 0 );
    }
    sum_dist /= count;
    ts->printf(CvTS::LOG, "Average error after findCornerSubpix: %f\n", sum_dist); 
        
    if( code < 0 )
        ts->set_failed_test_info( code );
}
                                        
void CV_ChessboardSubpixelTest::generateIntrinsicParams()
{
    CvRNG* rng = ts->get_rng();
    const double max_focus_length = 1000.0;
    const double max_focus_diff = 5.0;
    
    double fx = cvRandReal(rng)*max_focus_length;
    double fy = fx + cvRandReal(rng)*max_focus_diff;
    double cx = image_size_.width/2;
    double cy = image_size_.height/2;
    
    double k1 = 0.5*cvRandReal(rng);
    double k2 = 0.05*cvRandReal(rng);
    double p1 = 0.05*cvRandReal(rng);
    double p2 = 0.05*cvRandReal(rng);
    double k3 = 0.0;
    
    intrinsic_matrix_ = (Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    distortion_coeffs_ = (Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
}

CV_ChessboardSubpixelTest chessboard_subpixel_test;

/* End of file. */
