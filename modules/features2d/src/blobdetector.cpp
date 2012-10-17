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

#include "precomp.hpp"
#include <iterator>

//#define DEBUG_BLOB_DETECTOR

#ifdef DEBUG_BLOB_DETECTOR
#  include "opencv2/opencv_modules.hpp"
#  ifdef HAVE_OPENCV_HIGHGUI
#    include "opencv2/highgui/highgui.hpp"
#  else
#    undef DEBUG_BLOB_DETECTOR
#  endif
#endif

using namespace cv;

/*
*  SimpleBlobDetector
*/
SimpleBlobDetector::Params::Params()
{
    thresholdStep = 10;
    minThreshold = 50;
    maxThreshold = 220;
    minRepeatability = 2;
    minDistBetweenBlobs = 10;

    filterByColor = true;
    blobColor = 0;

    filterByArea = true;
    minArea = 25;
    maxArea = 5000;

    filterByCircularity = false;
    minCircularity = 0.8f;
    maxCircularity = std::numeric_limits<float>::max();

    filterByInertia = true;
    //minInertiaRatio = 0.6;
    minInertiaRatio = 0.1f;
    maxInertiaRatio = std::numeric_limits<float>::max();

    filterByConvexity = true;
    //minConvexity = 0.8;
    minConvexity = 0.95f;
    maxConvexity = std::numeric_limits<float>::max();
}

void SimpleBlobDetector::Params::read(const cv::FileNode& fn )
{
    thresholdStep = fn["thresholdStep"];
    minThreshold = fn["minThreshold"];
    maxThreshold = fn["maxThreshold"];

    minRepeatability = (size_t)(int)fn["minRepeatability"];
    minDistBetweenBlobs = fn["minDistBetweenBlobs"];

    filterByColor = (int)fn["filterByColor"] != 0 ? true : false;
    blobColor = (uchar)(int)fn["blobColor"];

    filterByArea = (int)fn["filterByArea"] != 0 ? true : false;
    minArea = fn["minArea"];
    maxArea = fn["maxArea"];

    filterByCircularity = (int)fn["filterByCircularity"] != 0 ? true : false;
    minCircularity = fn["minCircularity"];
    maxCircularity = fn["maxCircularity"];

    filterByInertia = (int)fn["filterByInertia"] != 0 ? true : false;
    minInertiaRatio = fn["minInertiaRatio"];
    maxInertiaRatio = fn["maxInertiaRatio"];

    filterByConvexity = (int)fn["filterByConvexity"] != 0 ? true : false;
    minConvexity = fn["minConvexity"];
    maxConvexity = fn["maxConvexity"];
}

void SimpleBlobDetector::Params::write(cv::FileStorage& fs) const
{
    fs << "thresholdStep" << thresholdStep;
    fs << "minThreshold" << minThreshold;
    fs << "maxThreshold" << maxThreshold;

    fs << "minRepeatability" << (int)minRepeatability;
    fs << "minDistBetweenBlobs" << minDistBetweenBlobs;

    fs << "filterByColor" << (int)filterByColor;
    fs << "blobColor" << (int)blobColor;

    fs << "filterByArea" << (int)filterByArea;
    fs << "minArea" << minArea;
    fs << "maxArea" << maxArea;

    fs << "filterByCircularity" << (int)filterByCircularity;
    fs << "minCircularity" << minCircularity;
    fs << "maxCircularity" << maxCircularity;

    fs << "filterByInertia" << (int)filterByInertia;
    fs << "minInertiaRatio" << minInertiaRatio;
    fs << "maxInertiaRatio" << maxInertiaRatio;

    fs << "filterByConvexity" << (int)filterByConvexity;
    fs << "minConvexity" << minConvexity;
    fs << "maxConvexity" << maxConvexity;
}

SimpleBlobDetector::SimpleBlobDetector(const SimpleBlobDetector::Params &parameters) :
params(parameters)
{
}

void SimpleBlobDetector::read( const cv::FileNode& fn )
{
    params.read(fn);
}

void SimpleBlobDetector::write( cv::FileStorage& fs ) const
{
    params.write(fs);
}

void SimpleBlobDetector::findBlobs(const cv::Mat &image, const cv::Mat &binaryImage, vector<Center> &centers) const
{
    (void)image;
    centers.clear();

    vector < vector<Point> > contours;
    Mat tmpBinaryImage = binaryImage.clone();
    findContours(tmpBinaryImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
    //  Mat keypointsImage;
    //  cvtColor( binaryImage, keypointsImage, CV_GRAY2RGB );
    //
    //  Mat contoursImage;
    //  cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
    //  drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
    //  imshow("contours", contoursImage );
#endif

    for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
    {
        Center center;
        center.confidence = 1;
        Moments moms = moments(Mat(contours[contourIdx]));
        if (params.filterByArea)
        {
            double area = moms.m00;
            if (area < params.minArea || area >= params.maxArea)
                continue;
        }

        if (params.filterByCircularity)
        {
            double area = moms.m00;
            double perimeter = arcLength(Mat(contours[contourIdx]), true);
            double ratio = 4 * CV_PI * area / (perimeter * perimeter);
            if (ratio < params.minCircularity || ratio >= params.maxCircularity)
                continue;
        }

        if (params.filterByInertia)
        {
            double denominator = sqrt(pow(2 * moms.mu11, 2) + pow(moms.mu20 - moms.mu02, 2));
            const double eps = 1e-2;
            double ratio;
            if (denominator > eps)
            {
                double cosmin = (moms.mu20 - moms.mu02) / denominator;
                double sinmin = 2 * moms.mu11 / denominator;
                double cosmax = -cosmin;
                double sinmax = -sinmin;

                double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
                double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
                ratio = imin / imax;
            }
            else
            {
                ratio = 1;
            }

            if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
                continue;

            center.confidence = ratio * ratio;
        }

        if (params.filterByConvexity)
        {
            vector < Point > hull;
            convexHull(Mat(contours[contourIdx]), hull);
            double area = contourArea(Mat(contours[contourIdx]));
            double hullArea = contourArea(Mat(hull));
            double ratio = area / hullArea;
            if (ratio < params.minConvexity || ratio >= params.maxConvexity)
                continue;
        }

        center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

        if (params.filterByColor)
        {
            if (binaryImage.at<uchar> (cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
                continue;
        }

        //compute blob radius
        {
            vector<double> dists;
            for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
            {
                Point2d pt = contours[contourIdx][pointIdx];
                dists.push_back(norm(center.location - pt));
            }
            std::sort(dists.begin(), dists.end());
            center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        }

        centers.push_back(center);

#ifdef DEBUG_BLOB_DETECTOR
        //    circle( keypointsImage, center.location, 1, Scalar(0,0,255), 1 );
#endif
    }
#ifdef DEBUG_BLOB_DETECTOR
    //  imshow("bk", keypointsImage );
    //  waitKey();
#endif
}

void SimpleBlobDetector::detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat&) const
{
    //TODO: support mask
    keypoints.clear();
    Mat grayscaleImage;
    if (image.channels() == 3)
        cvtColor(image, grayscaleImage, CV_BGR2GRAY);
    else
        grayscaleImage = image;

    vector < vector<Center> > centers;
    for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
    {
        Mat binarizedImage;
        threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);

#ifdef DEBUG_BLOB_DETECTOR
        //    Mat keypointsImage;
        //    cvtColor( binarizedImage, keypointsImage, CV_GRAY2RGB );
#endif

        vector < Center > curCenters;
        findBlobs(grayscaleImage, binarizedImage, curCenters);
        vector < vector<Center> > newCenters;
        for (size_t i = 0; i < curCenters.size(); i++)
        {
#ifdef DEBUG_BLOB_DETECTOR
            //      circle(keypointsImage, curCenters[i].location, curCenters[i].radius, Scalar(0,0,255),-1);
#endif

            bool isNew = true;
            for (size_t j = 0; j < centers.size(); j++)
            {
                double dist = norm(centers[j][ centers[j].size() / 2 ].location - curCenters[i].location);
                isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][ centers[j].size() / 2 ].radius && dist >= curCenters[i].radius;
                if (!isNew)
                {
                    centers[j].push_back(curCenters[i]);

                    size_t k = centers[j].size() - 1;
                    while( k > 0 && centers[j][k].radius < centers[j][k-1].radius )
                    {
                        centers[j][k] = centers[j][k-1];
                        k--;
                    }
                    centers[j][k] = curCenters[i];

                    break;
                }
            }
            if (isNew)
            {
                newCenters.push_back(vector<Center> (1, curCenters[i]));
                //centers.push_back(vector<Center> (1, curCenters[i]));
            }
        }
        std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));

#ifdef DEBUG_BLOB_DETECTOR
        //    imshow("binarized", keypointsImage );
        //waitKey();
#endif
    }

    for (size_t i = 0; i < centers.size(); i++)
    {
        if (centers[i].size() < params.minRepeatability)
            continue;
        Point2d sumPoint(0, 0);
        double normalizer = 0;
        for (size_t j = 0; j < centers[i].size(); j++)
        {
            sumPoint += centers[i][j].confidence * centers[i][j].location;
            normalizer += centers[i][j].confidence;
        }
        sumPoint *= (1. / normalizer);
        KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius));
        keypoints.push_back(kpt);
    }

#ifdef DEBUG_BLOB_DETECTOR
    namedWindow("keypoints", CV_WINDOW_NORMAL);
    Mat outImg = image.clone();
    for(size_t i=0; i<keypoints.size(); i++)
    {
        circle(outImg, keypoints[i].pt, keypoints[i].size, Scalar(255, 0, 255), -1);
    }
    //drawKeypoints(image, keypoints, outImg);
    imshow("keypoints", outImg);
    waitKey();
#endif
}
