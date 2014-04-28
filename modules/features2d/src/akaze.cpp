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
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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

/*
OpenCV wrapper of reference implementation of
[1] Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces.
Pablo F. Alcantarilla, J. Nuevo and Adrien Bartoli.
In British Machine Vision Conference (BMVC), Bristol, UK, September 2013
http://www.robesafe.com/personal/pablo.alcantarilla/papers/Alcantarilla13bmvc.pdf
@author Eugene Khvedchenya <ekhvedchenya@gmail.com>
*/

#include "precomp.hpp"
#include "akaze/AKAZEFeatures.h"

namespace cv
{

    AKAZE::AKAZE(int _descriptor, int _descriptor_size, int _descriptor_channels)
        : descriptor_channels(_descriptor_channels)
        , descriptor(_descriptor)
        , descriptor_size(_descriptor_size)
    {

    }

    AKAZE::~AKAZE()
    {

    }

    // returns the descriptor size in bytes
    int AKAZE::descriptorSize() const
    {
        if (descriptor < MLDB_UPRIGHT)
        {
            return 64;
        }
        else
        {
            // We use the full length binary descriptor -> 486 bits
            if (descriptor_size == 0)
            {
                int t = (6 + 36 + 120) * descriptor_channels;
                return (int)ceil(t / 8.);
            }
            else
            {
                // We use the random bit selection length binary descriptor
                return (int)ceil(descriptor_size / 8.);
            }
        }
    }

    // returns the descriptor type
    int AKAZE::descriptorType() const
    {
        if (descriptor < MLDB_UPRIGHT)
        {
            return CV_32F;
        }
        else
        {
            return CV_8U;
        }
    }

    // returns the default norm type
    int AKAZE::defaultNorm() const
    {
        if (descriptor < MLDB_UPRIGHT)
        {
            return NORM_L2;
        }
        else
        {
            return NORM_HAMMING;
        }
    }


    void AKAZE::operator()(InputArray image, InputArray mask,
        std::vector<KeyPoint>& keypoints,
        OutputArray descriptors,
        bool useProvidedKeypoints) const
    {
        cv::Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        cv::Mat& desc = descriptors.getMatRef();

        AKAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;

        AKAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);

        if (!useProvidedKeypoints)
        {
            impl.Feature_Detection(keypoints);
        }

        if (!mask.empty())
        {
            cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
        }

        impl.Compute_Descriptors(keypoints, desc);

        CV_Assert((!desc.rows || desc.cols == descriptorSize()));
        CV_Assert((!desc.rows || (desc.type() == descriptorType())));
    }

    void AKAZE::detectImpl(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask) const
    {
        cv::Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        AKAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;

        AKAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);
        impl.Feature_Detection(keypoints);

        if (!mask.empty())
        {
            cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
        }
    }

    void AKAZE::computeImpl(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) const
    {
        cv::Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        cv::Mat& desc = descriptors.getMatRef();

        AKAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;

        AKAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);
        impl.Compute_Descriptors(keypoints, desc);

        CV_Assert((!desc.rows || desc.cols == descriptorSize()));
        CV_Assert((!desc.rows || (desc.type() == descriptorType())));
    }
}