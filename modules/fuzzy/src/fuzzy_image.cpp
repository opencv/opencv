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
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
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

using namespace cv;

void ft::createKernel(InputArray A, InputArray B, OutputArray kernel, const int chn)
{
    Mat AMat = A.getMat();
    Mat BMat = B.getMat();
    Mat kernelOneChannel = AMat * BMat;
    std::vector<Mat> channels;

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(kernelOneChannel);
    }

    merge(channels, kernel);
}

void ft::createKernel(int function, int radius, cv::OutputArray kernel, const int chn)
{
    int basicFunctionWidth = 2 * radius + 1;
    Mat kernelOneChannel;
    Mat A(1, basicFunctionWidth, CV_32F, 0.0f);
    std::vector<Mat> channels;

    A.at<float>(0, radius) = 1;

    if (function == ft::LINEAR)
    {
        float a = 1.0f / radius;

        for (int i = 1; i < radius; i++)
        {
            float previous = A.at<float>(0, i - 1);
            float current =  previous + a;

            A.at<float>(0, i) = current;
            A.at<float>(0, (2 * radius) - i) = current;
        }

        mulTransposed(A, kernelOneChannel, true);
    }

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(kernelOneChannel);
    }

    merge(channels, kernel);
}

void ft::inpaint(const cv::Mat &image, const cv::Mat &mask, cv::Mat &output, int radius, int function, int algorithm)
{
    if (algorithm == ft::ONE_STEP)
    {
        Mat kernel;
        ft::createKernel(function, radius, kernel, image.channels());

        Mat processingInput;
        image.convertTo(processingInput, CV_32F);

        Mat processingOutput;
        ft::FT02D_process(processingInput, kernel, processingOutput, mask);

        processingInput.copyTo(processingOutput, mask);

        output = processingOutput;
    }
    else if (algorithm == ft::MULTI_STEP)
    {
        Mat kernel;
        Mat processingOutput;
        Mat outpuMask;
        int state = 0;
        int currentRadius = radius;

        Mat processingInput;
        image.convertTo(processingInput, CV_32F);

        Mat processingMask;
        cvtColor(mask, processingMask, COLOR_BGR2GRAY);

        do
        {
            ft::createKernel(function, currentRadius, kernel, image.channels());

            state = ft::FT02D_iteration(processingInput, kernel, processingOutput, processingMask, outpuMask, true);

            currentRadius++;
        }
        while(state != 0);

        processingInput.copyTo(processingOutput, mask);

        output = processingOutput;
    }
    else if (algorithm == ft::ITERATIVE)
    {
        Mat kernel;
        Mat processingOutput;
        Mat maskOutput;
        int state = 0;
        int currentRadius = radius;

        Mat originalImage;
        image.convertTo(originalImage, CV_32F);

        Mat processingInput;
        image.convertTo(processingInput, CV_32F);

        Mat processingMask;
        cvtColor(mask, processingMask, COLOR_BGR2GRAY);

        do
        {
            ft::createKernel(function, currentRadius, kernel, image.channels());

            Mat invMask = 1 - processingMask;

            state = ft::FT02D_iteration(processingInput, kernel, processingOutput, processingMask, maskOutput, false);

            maskOutput.copyTo(processingMask);
            processingOutput.copyTo(processingInput, invMask);

            currentRadius++;
        }
        while(state != 0);

        output = processingInput;
    }
}

void ft::filter(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output)
{
    Mat mask = Mat::ones(image.size(), CV_8U);

    ft::FT02D_process(image, kernel, output, mask);
}
