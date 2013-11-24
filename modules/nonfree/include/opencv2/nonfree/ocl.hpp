/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
// License Agreement
// For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistribution's of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistribution's in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * The name of the copyright holders may not be used to endorse or promote products
// derived from this software without specific prior written permission.
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

#ifndef __OPENCV_NONFREE_OCL_HPP__
#define __OPENCV_NONFREE_OCL_HPP__

#include "opencv2/ocl.hpp"

namespace cv
{
    namespace ocl
    {
        //! Speeded up robust features, port from CUDA module.
        ////////////////////////////////// SURF //////////////////////////////////////////

        class CV_EXPORTS SURF_OCL
        {
        public:
            enum KeypointLayout
            {
                X_ROW = 0,
                Y_ROW,
                LAPLACIAN_ROW,
                OCTAVE_ROW,
                SIZE_ROW,
                ANGLE_ROW,
                HESSIAN_ROW,
                ROWS_COUNT
            };

            //! the default constructor
            SURF_OCL();
            //! the full constructor taking all the necessary parameters
            explicit SURF_OCL(double _hessianThreshold, int _nOctaves = 4,
                              int _nOctaveLayers = 2, bool _extended = false, float _keypointsRatio = 0.01f, bool _upright = false);

            //! returns the descriptor size in float's (64 or 128)
            int descriptorSize() const;
            //! upload host keypoints to device memory
            void uploadKeypoints(const std::vector<cv::KeyPoint> &keypoints, oclMat &keypointsocl);
            //! download keypoints from device to host memory
            void downloadKeypoints(const oclMat &keypointsocl, std::vector<KeyPoint> &keypoints);
            //! download descriptors from device to host memory
            void downloadDescriptors(const oclMat &descriptorsocl, std::vector<float> &descriptors);
            //! finds the keypoints using fast hessian detector used in SURF
            //! supports CV_8UC1 images
            //! keypoints will have nFeature cols and 6 rows
            //! keypoints.ptr<float>(X_ROW)[i] will contain x coordinate of i'th feature
            //! keypoints.ptr<float>(Y_ROW)[i] will contain y coordinate of i'th feature
            //! keypoints.ptr<float>(LAPLACIAN_ROW)[i] will contain laplacian sign of i'th feature
            //! keypoints.ptr<float>(OCTAVE_ROW)[i] will contain octave of i'th feature
            //! keypoints.ptr<float>(SIZE_ROW)[i] will contain size of i'th feature
            //! keypoints.ptr<float>(ANGLE_ROW)[i] will contain orientation of i'th feature
            //! keypoints.ptr<float>(HESSIAN_ROW)[i] will contain response of i'th feature
            void operator()(const oclMat &img, const oclMat &mask, oclMat &keypoints);
            //! finds the keypoints and computes their descriptors.
            //! Optionally it can compute descriptors for the user-provided keypoints and recompute keypoints direction
            void operator()(const oclMat &img, const oclMat &mask, oclMat &keypoints, oclMat &descriptors,
                            bool useProvidedKeypoints = false);
            void operator()(const oclMat &img, const oclMat &mask, std::vector<KeyPoint> &keypoints);
            void operator()(const oclMat &img, const oclMat &mask, std::vector<KeyPoint> &keypoints, oclMat &descriptors,
                            bool useProvidedKeypoints = false);
            void operator()(const oclMat &img, const oclMat &mask, std::vector<KeyPoint> &keypoints, std::vector<float> &descriptors,
                            bool useProvidedKeypoints = false);

            void releaseMemory();

            // SURF parameters
            float hessianThreshold;
            int nOctaves;
            int nOctaveLayers;
            bool extended;
            bool upright;
            //! max keypoints = min(keypointsRatio * img.size().area(), 65535)
            float keypointsRatio;
            oclMat sum, mask1, maskSum, intBuffer;
            oclMat det, trace;
            oclMat maxPosBuffer;

        };
    }
}

#endif //__OPENCV_NONFREE_OCL_HPP__
