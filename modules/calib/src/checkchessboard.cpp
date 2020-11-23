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
#include "opencv2/imgproc/imgproc_c.h"
#include "calib3d_c_api.h"

#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

static void icvGetQuadrangleHypotheses(const std::vector<std::vector< cv::Point > > & contours, const std::vector< cv::Vec4i > & hierarchy, std::vector<std::pair<float, int> >& quads, int class_id)
{
    const float min_aspect_ratio = 0.3f;
    const float max_aspect_ratio = 3.0f;
    const float min_box_size = 10.0f;

    typedef std::vector< std::vector< cv::Point > >::const_iterator iter_t;
    iter_t i;
    for (i = contours.begin(); i != contours.end(); ++i)
    {
        const iter_t::difference_type idx = i - contours.begin();
        if (hierarchy.at(idx)[3] != -1)
            continue; // skip holes

        const std::vector< cv::Point > & c = *i;
        cv::RotatedRect box = cv::minAreaRect(c);

        float box_size = MAX(box.size.width, box.size.height);
        if(box_size < min_box_size)
        {
            continue;
        }

        float aspect_ratio = box.size.width/MAX(box.size.height, 1);
        if(aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio)
        {
            continue;
        }

        quads.push_back(std::pair<float, int>(box_size, class_id));
    }
}

static void countClasses(const std::vector<std::pair<float, int> >& pairs, size_t idx1, size_t idx2, std::vector<int>& counts)
{
    counts.assign(2, 0);
    for(size_t i = idx1; i != idx2; i++)
    {
        counts[pairs[i].second]++;
    }
}

inline bool less_pred(const std::pair<float, int>& p1, const std::pair<float, int>& p2)
{
    return p1.first < p2.first;
}

static void fillQuads(Mat & white, Mat & black, double white_thresh, double black_thresh, vector<pair<float, int> > & quads)
{
    Mat thresh;
    {
        vector< vector<Point> > contours;
        vector< Vec4i > hierarchy;
        threshold(white, thresh, white_thresh, 255, THRESH_BINARY);
        findContours(thresh, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        icvGetQuadrangleHypotheses(contours, hierarchy, quads, 1);
    }

    {
        vector< vector<Point> > contours;
        vector< Vec4i > hierarchy;
        threshold(black, thresh, black_thresh, 255, THRESH_BINARY_INV);
        findContours(thresh, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        icvGetQuadrangleHypotheses(contours, hierarchy, quads, 0);
    }
}

static bool checkQuads(vector<pair<float, int> > & quads, const cv::Size & size)
{
    const size_t min_quads_count = size.width*size.height/2;
    std::sort(quads.begin(), quads.end(), less_pred);

    // now check if there are many hypotheses with similar sizes
    // do this by floodfill-style algorithm
    const float size_rel_dev = 0.4f;

    for(size_t i = 0; i < quads.size(); i++)
    {
        size_t j = i + 1;
        for(; j < quads.size(); j++)
        {
            if(quads[j].first/quads[i].first > 1.0f + size_rel_dev)
            {
                break;
            }
        }

        if(j + 1 > min_quads_count + i)
        {
            // check the number of black and white squares
            std::vector<int> counts;
            countClasses(quads, i, j, counts);
            const int black_count = cvRound(ceil(size.width/2.0)*ceil(size.height/2.0));
            const int white_count = cvRound(floor(size.width/2.0)*floor(size.height/2.0));
            if(counts[0] < black_count*0.75 ||
               counts[1] < white_count*0.75)
            {
                continue;
            }
            return true;
        }
    }
    return false;
}

// does a fast check if a chessboard is in the input image. This is a workaround to
// a problem of cvFindChessboardCorners being slow on images with no chessboard
// - src: input image
// - size: chessboard size
// Returns 1 if a chessboard can be in this image and findChessboardCorners should be called,
// 0 if there is no chessboard, -1 in case of error
int cvCheckChessboard(IplImage* src, CvSize size)
{
    cv::Mat img = cv::cvarrToMat(src);
    return (int)cv::checkChessboard(img, size);
}

bool cv::checkChessboard(InputArray _img, Size size)
{
    Mat img = _img.getMat();
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);

    const int erosion_count = 1;
    const float black_level = 20.f;
    const float white_level = 130.f;
    const float black_white_gap = 70.f;

    Mat white;
    Mat black;
    erode(img, white, Mat(), Point(-1, -1), erosion_count);
    dilate(img, black, Mat(), Point(-1, -1), erosion_count);

    bool result = false;
    for(float thresh_level = black_level; thresh_level < white_level && !result; thresh_level += 20.0f)
    {
        vector<pair<float, int> > quads;
        fillQuads(white, black, thresh_level + black_white_gap, thresh_level, quads);
        if (checkQuads(quads, size))
            result = true;
    }
    return result;
}

// does a fast check if a chessboard is in the input image. This is a workaround to
// a problem of cvFindChessboardCorners being slow on images with no chessboard
// - src: input binary image
// - size: chessboard size
// Returns 1 if a chessboard can be in this image and findChessboardCorners should be called,
// 0 if there is no chessboard, -1 in case of error
int checkChessboardBinary(const cv::Mat & img, const cv::Size & size)
{
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);

    Mat white = img.clone();
    Mat black = img.clone();

    int result = 0;
    for ( int erosion_count = 0; erosion_count <= 3; erosion_count++ )
    {
        if ( 1 == result )
            break;

        if ( 0 != erosion_count ) // first iteration keeps original images
        {
            erode(white, white, Mat(), Point(-1, -1), 1);
            dilate(black, black, Mat(), Point(-1, -1), 1);
        }

        vector<pair<float, int> > quads;
        fillQuads(white, black, 128, 128, quads);
        if (checkQuads(quads, size))
            result = 1;
    }
    return result;
}
