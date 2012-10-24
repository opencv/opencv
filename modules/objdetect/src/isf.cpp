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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#include <precomp.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>


template<typename T>
struct Decimate {
    int shrinkage;

    Decimate(const int sr) : shrinkage(sr) {}

    void operator()(const cv::Mat& in, cv::Mat& out) const
    {
        int cols = in.cols / shrinkage;
        int rows = in.rows / shrinkage;
        out.create(rows, cols, in.type());

        CV_Assert(cols * shrinkage == in.cols);
        CV_Assert(rows * shrinkage == in.rows);

        for (int outIdx_y = 0; outIdx_y < rows; ++outIdx_y)
        {
            T* outPtr = out.ptr<T>(outIdx_y);
            for (int outIdx_x = 0; outIdx_x < cols; ++outIdx_x)
            {
                // do desimate
                int inIdx_y = outIdx_y * shrinkage;
                int inIdx_x = outIdx_x * shrinkage;
                int sum = 0;

                for (int y = inIdx_y; y < inIdx_y + shrinkage; ++y)
                    for (int x = inIdx_x; x < inIdx_x + shrinkage; ++x)
                        sum += in.at<T>(y, x);

                sum /= shrinkage * shrinkage;
                outPtr[outIdx_x] = cv::saturate_cast<T>(sum);
            }
        }
    }

};

void cv::IntegralChannels::createHogBins(const cv::Mat gray, std::vector<cv::Mat>& integrals, int bins) const
{
    CV_Assert(gray.type() == CV_8UC1);
    int h = gray.rows;
    int w = gray.cols;
    CV_Assert(!(w % shrinkage) && !(h % shrinkage));

    Decimate<uchar> decimate(shrinkage);

    cv::Mat df_dx, df_dy, mag, angle;
    cv::Sobel(gray, df_dx, CV_32F, 1, 0, 3, 0.125);
    cv::Sobel(gray, df_dy, CV_32F, 0, 1, 3, 0.125);

    cv::cartToPolar(df_dx, df_dy, mag, angle, true);
    mag *= (1.f / sqrt(2));

    cv::Mat nmag;
    mag.convertTo(nmag, CV_8UC1);

    angle /= 60.f;

    std::vector<cv::Mat> hist;
    for (int bin = 0; bin < bins; ++bin)
        hist.push_back(cv::Mat::zeros(h, w, CV_8UC1));

    for (int y = 0; y < h; ++y)
    {
        uchar* magnitude = nmag.ptr<uchar>(y);
        float* ang = angle.ptr<float>(y);

        for (int x = 0; x < w; ++x)
        {
            hist[ (int)ang[x] ].ptr<uchar>(y)[x] = magnitude[x];
        }
    }

    for(int i = 0; i < bins; ++i)
    {
        cv::Mat shrunk, sum;
        decimate(hist[i], shrunk);
        cv::integral(shrunk, sum, cv::noArray(), CV_32S);
        integrals.push_back(sum);
    }

    cv::Mat shrMag;
    decimate(nmag, shrMag);
    cv::integral(shrMag, mag, cv::noArray(), CV_32S);
    integrals.push_back(mag);
}

void cv::IntegralChannels::createLuvBins(const cv::Mat frame, std::vector<cv::Mat>& integrals) const
{
    CV_Assert(frame.type() == CV_8UC3);
    CV_Assert(!(frame.cols % shrinkage) && !(frame.rows % shrinkage));

    Decimate<uchar> decimate(shrinkage);

    cv::Mat luv;
    cv::cvtColor(frame, luv, CV_BGR2Luv);

    std::vector<cv::Mat> splited;
    split(luv, splited);

    for (size_t i = 0; i < splited.size(); ++i)
    {
        cv::Mat shrunk, sum;
        decimate(splited[i], shrunk);
        cv::integral(shrunk, sum, cv::noArray(), CV_32S);
        integrals.push_back(sum);
    }
}