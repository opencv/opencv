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

#include "perf_precomp.hpp"

using namespace std;
using namespace testing;
using namespace perf;

DEF_PARAM_TEST_1(Image, string);

struct GreedyLabeling
{
    struct dot
    {
        int x;
        int y;

        static dot make(int i, int j)
        {
            dot d; d.x = i; d.y = j;
            return d;
        }
    };

    struct InInterval
    {
        InInterval(const int& _lo, const int& _hi) : lo(-_lo), hi(_hi) {}
        const int lo, hi;

        bool operator() (const unsigned char a, const unsigned char b) const
        {
            int d = a - b;
            return lo <= d && d <= hi;
        }

    private:
        InInterval& operator=(const InInterval&);


    };

    GreedyLabeling(cv::Mat img)
    : image(img), _labels(image.size(), CV_32SC1, cv::Scalar::all(-1)) {stack = new dot[image.cols * image.rows];}

    ~GreedyLabeling(){delete[] stack;}

    void operator() (cv::Mat labels) const
    {
        labels.setTo(cv::Scalar::all(-1));
        InInterval inInt(0, 2);
        int cc = -1;

        int* dist_labels = (int*)labels.data;
        int pitch = static_cast<int>(labels.step1());

        unsigned char* source = (unsigned char*)image.data;
        int width = image.cols;
        int height = image.rows;

        for (int j = 0; j < image.rows; ++j)
            for (int i = 0; i < image.cols; ++i)
            {
                if (dist_labels[j * pitch + i] != -1) continue;

                dot* top = stack;
                dot p = dot::make(i, j);
                cc++;

                dist_labels[j * pitch + i] = cc;

                while (top >= stack)
                {
                    int*  dl = &dist_labels[p.y * pitch + p.x];
                    unsigned char* sp = &source[p.y * image.step1() + p.x];

                    dl[0] = cc;

                    //right
                    if( p.x < (width - 1) && dl[ +1] == -1 && inInt(sp[0], sp[+1]))
                        *top++ = dot::make(p.x + 1, p.y);

                    //left
                    if( p.x > 0 && dl[-1] == -1 && inInt(sp[0], sp[-1]))
                        *top++ = dot::make(p.x - 1, p.y);

                    //bottom
                    if( p.y < (height - 1) && dl[+pitch] == -1 && inInt(sp[0], sp[+image.step1()]))
                        *top++ = dot::make(p.x, p.y + 1);

                    //top
                    if( p.y > 0 && dl[-pitch] == -1 && inInt(sp[0], sp[-static_cast<int>(image.step1())]))
                        *top++ = dot::make(p.x, p.y - 1);

                    p = *--top;
                }
            }
    }

    cv::Mat image;
    cv::Mat _labels;
    dot* stack;
};

PERF_TEST_P(Image, DISABLED_Labeling_ConnectivityMask,
            Values<string>("gpu/labeling/aloe-disp.png"))
{
    declare.time(1.0);

    const cv::Mat image = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    if (PERF_RUN_CUDA())
    {
        cv::cuda::GpuMat d_image(image);
        cv::cuda::GpuMat mask;

        TEST_CYCLE() cv::cuda::connectivityMask(d_image, mask, cv::Scalar::all(0), cv::Scalar::all(2));

        CUDA_SANITY_CHECK(mask);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

PERF_TEST_P(Image, DISABLED_Labeling_ConnectedComponents,
            Values<string>("gpu/labeling/aloe-disp.png"))
{
    declare.time(1.0);

    const cv::Mat image = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    if (PERF_RUN_CUDA())
    {
        cv::cuda::GpuMat d_mask;
        cv::cuda::connectivityMask(cv::cuda::GpuMat(image), d_mask, cv::Scalar::all(0), cv::Scalar::all(2));

        cv::cuda::GpuMat components;

        TEST_CYCLE() cv::cuda::labelComponents(d_mask, components);

        CUDA_SANITY_CHECK(components);
    }
    else
    {
        GreedyLabeling host(image);

        TEST_CYCLE() host(host._labels);

        cv::Mat components = host._labels;
        CPU_SANITY_CHECK(components);
    }
}
