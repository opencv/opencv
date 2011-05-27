/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

#include "exposure_compensate.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;


Ptr<ExposureCompensator> ExposureCompensator::createDefault(int type)
{
    if (type == NO)
        return new NoExposureCompensator();
    if (type == OVERLAP)
        return new OverlapExposureCompensator();
    if (type == SEGMENT)
        return new SegmentExposureCompensator();
    CV_Error(CV_StsBadArg, "unsupported exposure compensation method");
    return NULL;
}


void OverlapExposureCompensator::feed(const vector<Point> &corners, const vector<Mat> &images, 
                                      const vector<Mat> &masks)
{
    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());
    Mat_<int> N(num_images, num_images); N.setTo(0);
    Mat_<double> I(num_images, num_images); I.setTo(0);

    Rect dst_roi = resultRoi(corners, images);
    Mat subimg1, subimg2;
    Mat_<uchar> submask1, submask2, intersect;

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = i; j < num_images; ++j)
        {
            Rect roi;
            if (overlapRoi(corners[i], corners[j], images[i].size(), images[j].size(), roi))
            {
                subimg1 = images[i](Rect(roi.tl() - corners[i], roi.br() - corners[i]));
                subimg2 = images[j](Rect(roi.tl() - corners[j], roi.br() - corners[j]));

                submask1 = masks[i](Rect(roi.tl() - corners[i], roi.br() - corners[i]));
                submask2 = masks[j](Rect(roi.tl() - corners[j], roi.br() - corners[j]));
                intersect = submask1 & submask2;

                N(i, j) = N(j, i) = countNonZero(intersect);

                double Isum1 = 0, Isum2 = 0;
                for (int y = 0; y < roi.height; ++y)
                {
                    const Point3_<uchar>* r1 = subimg1.ptr<Point3_<uchar> >(y);
                    const Point3_<uchar>* r2 = subimg2.ptr<Point3_<uchar> >(y);
                    for (int x = 0; x < roi.width; ++x)
                    {
                        if (intersect(y, x))
                        {
                            Isum1 += sqrt(static_cast<double>(sqr(r1[x].x) + sqr(r1[x].y) + sqr(r1[x].z)));
                            Isum2 += sqrt(static_cast<double>(sqr(r2[x].x) + sqr(r2[x].y) + sqr(r2[x].z)));
                        }
                    }
                }
                I(i, j) = Isum1 / max(N(i, j), 1);
                I(j, i) = Isum2 / max(N(i, j), 1);
            }
        }
    }

    double alpha = 0.01;
    double beta = 100;

    Mat_<double> A(num_images, num_images); A.setTo(0);
    Mat_<double> b(num_images, 1); b.setTo(0);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            b(i, 0) += beta * N(i, j);
            A(i, i) += beta * N(i, j);
            if (j == i) continue;
            A(i, i) += 2 * alpha * I(i, j) * I(i, j) * N(i, j);
            A(i, j) -= 2 * alpha * I(i, j) * I(j, i) * N(i, j);
        }
    }

    solve(A, b, gains_);
}


void OverlapExposureCompensator::apply(int index, Point /*corner*/, Mat &image, const Mat &/*mask*/)
{
    image *= gains_(index, 0);
}


void SegmentExposureCompensator::feed(const vector<Point> &/*corners*/, const vector<Mat> &/*images*/, 
                                      const vector<Mat> &/*masks*/)
{
}


void SegmentExposureCompensator::apply(int /*index*/, Point /*corner*/, Mat &/*image*/, const Mat &/*mask*/)
{
}