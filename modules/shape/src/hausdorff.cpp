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

#include "precomp.hpp"

/*
 * Hausdorff distance for a set of points
 */
namespace cv
{
    static float _apply(const Mat &set1, const Mat &set2, int distType)
    {
        float max_dist = -1;
        switch (distType)
        {
        case DIST_L2:
            for (int r=0; r<set1.cols; r++)
            {
                Point2f diff = set1.at<Point2f>(0,r)-set2.at<Point2f>(0,0);
                float min_dist = std::sqrt(diff.x*diff.x + diff.y*diff.y);
                for (int c=1; c<set2.cols; c++)
                {
                    diff = set1.at<Point2f>(0,r)-set2.at<Point2f>(0,c);
                    float dist = std::sqrt(diff.x*diff.x + diff.y*diff.y);
                    min_dist = std::min(dist,min_dist);
                }
                max_dist = std::max(min_dist, max_dist);
            }
            break;

        case DIST_L1:
            for (int r=0; r<set1.cols; r++)
            {
                Point2f diff = set1.at<Point2f>(0,r)-set2.at<Point2f>(0,0);
                float min_dist = std::sqrt(diff.x*diff.x + diff.y*diff.y);
                for (int c=1; c<set2.cols; c++)
                {
                    diff = set1.at<Point2f>(0,r)-set2.at<Point2f>(0,c);
                    float dist = std::fabs(diff.x) + std::fabs(diff.y);
                    min_dist = std::min(dist, min_dist);
                }
                max_dist = std::max(min_dist, max_dist);
            }
            break;

        default:
            CV_Error(-206, "The available flags are: DIST_L1, DIST_L2");
            break;
        }
        return max_dist;
    }

    float hausdorff(InputArray _set1, InputArray _set2, int distType)
    {
        Mat set1=_set1.getMat(), set2=_set2.getMat();
        CV_Assert((set1.channels()==2) & (set1.cols>0));
        CV_Assert((set1.channels()==2) & (set2.cols>0));
        return _apply(set1, set2, distType);
    }
}




