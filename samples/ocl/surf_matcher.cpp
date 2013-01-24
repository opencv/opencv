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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

//#define USE_CPU_DESCRIPTOR // use cpu descriptor extractor until ocl descriptor extractor is fixed
//#define USE_CPU_BFMATCHER
void help();

void help()
{
    cout << "\nThis program demonstrates using SURF_OCL features detector and descriptor extractor" << endl;
    cout << "\nUsage:\n\tsurf_matcher --left <image1> --right <image2>" << endl;
}


////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
    if (argc != 5 && argc != 1)
    {
        help();
        return -1;
    }
    vector<cv::ocl::Info> info;
    if(!cv::ocl::getDevice(info))
    {
        cout << "Error: Did not find a valid OpenCL device!" << endl;
        return -1;
    }
    Mat cpu_img1, cpu_img2, cpu_img1_grey, cpu_img2_grey;
    oclMat img1, img2;
    if(argc != 5)
    {
        cpu_img1 = imread("o.png");
        cvtColor(cpu_img1, cpu_img1_grey, CV_BGR2GRAY);
        img1     = cpu_img1_grey;
        CV_Assert(!img1.empty());

        cpu_img2 = imread("r2.png");
        cvtColor(cpu_img2, cpu_img2_grey, CV_BGR2GRAY);
        img2     = cpu_img2_grey;
    }
    else
    {
        for (int i = 1; i < argc; ++i)
        {
            if (string(argv[i]) == "--left")
            {
                cpu_img1 = imread(argv[++i]);
                cvtColor(cpu_img1, cpu_img1_grey, CV_BGR2GRAY);
                img1     = cpu_img1_grey;
                CV_Assert(!img1.empty());
            }
            else if (string(argv[i]) == "--right")
            {
                cpu_img2 = imread(argv[++i]);
                cvtColor(cpu_img2, cpu_img2_grey, CV_BGR2GRAY);
                img2     = cpu_img2_grey;
            }
            else if (string(argv[i]) == "--help")
            {
                help();
                return -1;
            }
        }
    }

    SURF_OCL surf;
    //surf.hessianThreshold = 400.f;
    //surf.extended = false;

    // detecting keypoints & computing descriptors
    oclMat keypoints1GPU, keypoints2GPU;
    oclMat descriptors1GPU, descriptors2GPU;

    // downloading results
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;


#ifndef USE_CPU_DESCRIPTOR
    surf(img1, oclMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, oclMat(), keypoints2GPU, descriptors2GPU);

    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);


#ifdef USE_CPU_BFMATCHER
    //BFMatcher
    BFMatcher matcher(cv::NORM_L2);
    matcher.match(Mat(descriptors1GPU), Mat(descriptors2GPU), matches);
#else
    BruteForceMatcher_OCL_base matcher(BruteForceMatcher_OCL_base::L2Dist);
    matcher.match(descriptors1GPU, descriptors2GPU, matches);
#endif

#else
    surf(img1, oclMat(), keypoints1GPU);
    surf(img2, oclMat(), keypoints2GPU);
    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);

    // use SURF_OCL to detect keypoints and use SURF to extract descriptors
    SURF surf_cpu;
    Mat descriptors1, descriptors2;
    surf_cpu(cpu_img1, Mat(), keypoints1, descriptors1, true);
    surf_cpu(cpu_img2, Mat(), keypoints2, descriptors2, true);
    matcher.match(descriptors1, descriptors2, matches);
#endif
    cout << "OCL: FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "OCL: FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for( size_t i = 0; i < keypoints1.size(); i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2.5*min_dist )
    std::vector< DMatch > good_matches;

    for( size_t i = 0; i < keypoints1.size(); i++ )
    {
        if( matches[i].distance < 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }

    // drawing the results
    Mat img_matches;
    drawMatches( cpu_img1, keypoints1, cpu_img2, keypoints2,
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, CV_RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( cpu_img1.cols, 0 );
    obj_corners[2] = cvPoint( cpu_img1.cols, cpu_img1.rows ); obj_corners[3] = cvPoint( 0, cpu_img1.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( (float)cpu_img1.cols, 0), scene_corners[1] + Point2f( (float)cpu_img1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( (float)cpu_img1.cols, 0), scene_corners[2] + Point2f( (float)cpu_img1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( (float)cpu_img1.cols, 0), scene_corners[3] + Point2f( (float)cpu_img1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( (float)cpu_img1.cols, 0), scene_corners[0] + Point2f( (float)cpu_img1.cols, 0), Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    namedWindow("ocl surf matches", 0);
    imshow("ocl surf matches", img_matches);
    waitKey(0);

    return 0;
}
