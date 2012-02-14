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
#include "opencv2/core/core.hpp"
#include "precomp.hpp"

#include <iostream>

using namespace cv;

void downsamplePoints( const Mat& src, Mat& dst, size_t count )
{
    CV_Assert( count >= 2 );
    CV_Assert( src.cols == 1 || src.rows == 1 );
    CV_Assert( src.total() >= count );
    CV_Assert( src.type() == CV_8UC3);

    dst.create( 1, count, CV_8UC3 );
    //TODO: optimize by exploiting symmetry in the distance matrix
    Mat dists( src.total(), src.total(), CV_32FC1, Scalar(0) );
    if( dists.empty() )
        std::cerr << "Such big matrix cann't be created." << std::endl;

    for( int i = 0; i < dists.rows; i++ )
    {
        for( int j = i; j < dists.cols; j++ )
        {
            float dist = (float)norm(src.at<Point3_<uchar> >(i) - src.at<Point3_<uchar> >(j));
            dists.at<float>(j, i) = dists.at<float>(i, j) = dist;
        }
    }

    double maxVal;
    Point maxLoc;
    minMaxLoc(dists, 0, &maxVal, 0, &maxLoc);

    dst.at<Point3_<uchar> >(0) = src.at<Point3_<uchar> >(maxLoc.x);
    dst.at<Point3_<uchar> >(1) = src.at<Point3_<uchar> >(maxLoc.y);

    Mat activedDists( 0, dists.cols, dists.type() );
    Mat candidatePointsMask( 1, dists.cols, CV_8UC1, Scalar(255) );
    activedDists.push_back( dists.row(maxLoc.y) );
    candidatePointsMask.at<uchar>(0, maxLoc.y) = 0;

    for( size_t i = 2; i < count; i++ )
    {
        activedDists.push_back(dists.row(maxLoc.x));
        candidatePointsMask.at<uchar>(0, maxLoc.x) = 0;

        Mat minDists;
        reduce( activedDists, minDists, 0, CV_REDUCE_MIN );
        minMaxLoc( minDists, 0, &maxVal, 0, &maxLoc, candidatePointsMask );
        dst.at<Point3_<uchar> >(i) = src.at<Point3_<uchar> >(maxLoc.x);
    }
}

void cv::generateColors( std::vector<Scalar>& colors, size_t count, size_t factor )
{
    if( count < 1 )
        return;

    colors.resize(count);

    if( count == 1 )
    {
        colors[0] = Scalar(0,0,255); // red
        return;
    }
    if( count == 2 )
    {
        colors[0] = Scalar(0,0,255); // red
        colors[1] = Scalar(0,255,0); // green
        return;
    }

    // Generate a set of colors in RGB space. A size of the set is severel times (=factor) larger then
    // the needed count of colors.
    Mat bgr( 1, count*factor, CV_8UC3 );
    randu( bgr, 0, 256 );

    // Convert the colors set to Lab space.
    // Distances between colors in this space correspond a human perception.
    Mat lab;
    cvtColor( bgr, lab, CV_BGR2Lab);

    // Subsample colors from the generated set so that
    // to maximize the minimum distances between each other.
    // Douglas-Peucker algorithm is used for this.
    Mat lab_subset;
    downsamplePoints( lab, lab_subset, count );

    // Convert subsampled colors back to RGB
    Mat bgr_subset;
    cvtColor( lab_subset, bgr_subset, CV_Lab2BGR );

    CV_Assert( bgr_subset.total() == count );
    for( size_t i = 0; i < count; i++ )
    {
        Point3_<uchar> c = bgr_subset.at<Point3_<uchar> >(i);
        colors[i] = Scalar(c.x, c.y, c.z);
    }
}
