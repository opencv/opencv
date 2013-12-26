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

#include "opencv2/ts/gpu_perf.hpp"
#include "opencv2/core/gpumat.hpp"

#include "cvconfig.h"

using namespace cv;

namespace perf
{
    Mat readImage(const string& fileName, int flags)
    {
        return imread(perf::TestBase::getDataPath(fileName), flags);
    }

    void PrintTo(const CvtColorInfo& info, std::ostream* os)
    {
        static const char* str[] =
        {
            "BGR2BGRA",
            "BGRA2BGR",
            "BGR2RGBA",
            "RGBA2BGR",
            "BGR2RGB",
            "BGRA2RGBA",

            "BGR2GRAY",
            "RGB2GRAY",
            "GRAY2BGR",
            "GRAY2BGRA",
            "BGRA2GRAY",
            "RGBA2GRAY",

            "BGR2BGR565",
            "RGB2BGR565",
            "BGR5652BGR",
            "BGR5652RGB",
            "BGRA2BGR565",
            "RGBA2BGR565",
            "BGR5652BGRA",
            "BGR5652RGBA",

            "GRAY2BGR565",
            "BGR5652GRAY",

            "BGR2BGR555",
            "RGB2BGR555",
            "BGR5552BGR",
            "BGR5552RGB",
            "BGRA2BGR555",
            "RGBA2BGR555",
            "BGR5552BGRA",
            "BGR5552RGBA",

            "GRAY2BGR555",
            "BGR5552GRAY",

            "BGR2XYZ",
            "RGB2XYZ",
            "XYZ2BGR",
            "XYZ2RGB",

            "BGR2YCrCb",
            "RGB2YCrCb",
            "YCrCb2BGR",
            "YCrCb2RGB",

            "BGR2HSV",
            "RGB2HSV",

            "",
            "",

            "BGR2Lab",
            "RGB2Lab",

            "BayerBG2BGR",
            "BayerGB2BGR",
            "BayerRG2BGR",
            "BayerGR2BGR",

            "BGR2Luv",
            "RGB2Luv",

            "BGR2HLS",
            "RGB2HLS",

            "HSV2BGR",
            "HSV2RGB",

            "Lab2BGR",
            "Lab2RGB",
            "Luv2BGR",
            "Luv2RGB",

            "HLS2BGR",
            "HLS2RGB",

            "BayerBG2BGR_VNG",
            "BayerGB2BGR_VNG",
            "BayerRG2BGR_VNG",
            "BayerGR2BGR_VNG",

            "BGR2HSV_FULL",
            "RGB2HSV_FULL",
            "BGR2HLS_FULL",
            "RGB2HLS_FULL",

            "HSV2BGR_FULL",
            "HSV2RGB_FULL",
            "HLS2BGR_FULL",
            "HLS2RGB_FULL",

            "LBGR2Lab",
            "LRGB2Lab",
            "LBGR2Luv",
            "LRGB2Luv",

            "Lab2LBGR",
            "Lab2LRGB",
            "Luv2LBGR",
            "Luv2LRGB",

            "BGR2YUV",
            "RGB2YUV",
            "YUV2BGR",
            "YUV2RGB",

            "BayerBG2GRAY",
            "BayerGB2GRAY",
            "BayerRG2GRAY",
            "BayerGR2GRAY",

            //YUV 4:2:0 formats family
            "YUV2RGB_NV12",
            "YUV2BGR_NV12",
            "YUV2RGB_NV21",
            "YUV2BGR_NV21",

            "YUV2RGBA_NV12",
            "YUV2BGRA_NV12",
            "YUV2RGBA_NV21",
            "YUV2BGRA_NV21",

            "YUV2RGB_YV12",
            "YUV2BGR_YV12",
            "YUV2RGB_IYUV",
            "YUV2BGR_IYUV",

            "YUV2RGBA_YV12",
            "YUV2BGRA_YV12",
            "YUV2RGBA_IYUV",
            "YUV2BGRA_IYUV",

            "YUV2GRAY_420",

            //YUV 4:2:2 formats family
            "YUV2RGB_UYVY",
            "YUV2BGR_UYVY",
            "YUV2RGB_VYUY",
            "YUV2BGR_VYUY",

            "YUV2RGBA_UYVY",
            "YUV2BGRA_UYVY",
            "YUV2RGBA_VYUY",
            "YUV2BGRA_VYUY",

            "YUV2RGB_YUY2",
            "YUV2BGR_YUY2",
            "YUV2RGB_YVYU",
            "YUV2BGR_YVYU",

            "YUV2RGBA_YUY2",
            "YUV2BGRA_YUY2",
            "YUV2RGBA_YVYU",
            "YUV2BGRA_YVYU",

            "YUV2GRAY_UYVY",
            "YUV2GRAY_YUY2",

            // alpha premultiplication
            "RGBA2mRGBA",
            "mRGBA2RGBA",

            "COLORCVT_MAX"
        };

        *os << str[info.code];
    }

    static void printOsInfo()
    {
    #if defined _WIN32
    #   if defined _WIN64
            printf("[----------]\n[ GPU INFO ] \tRun on OS Windows x64.\n[----------]\n"), fflush(stdout);
    #   else
            printf("[----------]\n[ GPU INFO ] \tRun on OS Windows x32.\n[----------]\n"), fflush(stdout);
    #   endif
    #elif defined linux
    #   if defined _LP64
            printf("[----------]\n[ GPU INFO ] \tRun on OS Linux x64.\n[----------]\n"), fflush(stdout);
    #   else
            printf("[----------]\n[ GPU INFO ] \tRun on OS Linux x32.\n[----------]\n"), fflush(stdout);
    #   endif
    #elif defined __APPLE__
    #   if defined _LP64
            printf("[----------]\n[ GPU INFO ] \tRun on OS Apple x64.\n[----------]\n"), fflush(stdout);
    #   else
            printf("[----------]\n[ GPU INFO ] \tRun on OS Apple x32.\n[----------]\n"), fflush(stdout);
    #   endif
    #endif

    }

    void printCudaInfo()
    {
        printOsInfo();
        for (int i = 0; i < cv::gpu::getCudaEnabledDeviceCount(); i++)
            cv::gpu::printCudaDeviceInfo(i);
    }

    struct KeypointIdxCompare
    {
        std::vector<cv::KeyPoint>* keypoints;

        explicit KeypointIdxCompare(std::vector<cv::KeyPoint>* _keypoints) : keypoints(_keypoints) {}

        bool operator ()(size_t i1, size_t i2) const
        {
            cv::KeyPoint kp1 = (*keypoints)[i1];
            cv::KeyPoint kp2 = (*keypoints)[i2];
            if (kp1.pt.x != kp2.pt.x)
                return kp1.pt.x < kp2.pt.x;
            if (kp1.pt.y != kp2.pt.y)
                return kp1.pt.y < kp2.pt.y;
            if (kp1.response != kp2.response)
                return kp1.response < kp2.response;
            return kp1.octave < kp2.octave;
        }
    };

    void sortKeyPoints(std::vector<cv::KeyPoint>& keypoints, cv::InputOutputArray _descriptors)
    {
        std::vector<size_t> indexies(keypoints.size());
        for (size_t i = 0; i < indexies.size(); ++i)
            indexies[i] = i;

        std::sort(indexies.begin(), indexies.end(), KeypointIdxCompare(&keypoints));

        std::vector<cv::KeyPoint> new_keypoints;
        cv::Mat new_descriptors;

        new_keypoints.resize(keypoints.size());

        cv::Mat descriptors;
        if (_descriptors.needed())
        {
            descriptors = _descriptors.getMat();
            new_descriptors.create(descriptors.size(), descriptors.type());
        }

        for (size_t i = 0; i < indexies.size(); ++i)
        {
            size_t new_idx = indexies[i];
            new_keypoints[i] = keypoints[new_idx];
            if (!new_descriptors.empty())
                descriptors.row((int) new_idx).copyTo(new_descriptors.row((int) i));
        }

        keypoints.swap(new_keypoints);
        if (_descriptors.needed())
            new_descriptors.copyTo(_descriptors);
    }
}
