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

#ifndef __OPENCV_SUPERRES_OPTICAL_FLOW_HPP__
#define __OPENCV_SUPERRES_OPTICAL_FLOW_HPP__

#include "opencv2/core.hpp"

namespace cv
{
    namespace superres
    {

//! @addtogroup superres
//! @{

        class CV_EXPORTS DenseOpticalFlowExt : public cv::Algorithm
        {
        public:
            virtual void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2 = noArray()) = 0;
            virtual void collectGarbage() = 0;
        };


        class CV_EXPORTS FarnebackOpticalFlow : public virtual DenseOpticalFlowExt
        {
        public:
            CV_PURE_PROPERTY(double, PyrScale)
            CV_PURE_PROPERTY(int, LevelsNumber)
            CV_PURE_PROPERTY(int, WindowSize)
            CV_PURE_PROPERTY(int, Iterations)
            CV_PURE_PROPERTY(int, PolyN)
            CV_PURE_PROPERTY(double, PolySigma)
            CV_PURE_PROPERTY(int, Flags)
        };
        CV_EXPORTS Ptr<FarnebackOpticalFlow> createOptFlow_Farneback();
        CV_EXPORTS Ptr<FarnebackOpticalFlow> createOptFlow_Farneback_CUDA();


//        CV_EXPORTS Ptr<DenseOpticalFlowExt> createOptFlow_Simple();


        class CV_EXPORTS DualTVL1OpticalFlow : public virtual DenseOpticalFlowExt
        {
        public:
            CV_PURE_PROPERTY(double, Tau)
            CV_PURE_PROPERTY(double, Lambda)
            CV_PURE_PROPERTY(double, Theta)
            CV_PURE_PROPERTY(int, ScalesNumber)
            CV_PURE_PROPERTY(int, WarpingsNumber)
            CV_PURE_PROPERTY(double, Epsilon)
            CV_PURE_PROPERTY(int, Iterations)
            CV_PURE_PROPERTY(bool, UseInitialFlow)
        };
        CV_EXPORTS Ptr<DualTVL1OpticalFlow> createOptFlow_DualTVL1();
        CV_EXPORTS Ptr<DualTVL1OpticalFlow> createOptFlow_DualTVL1_CUDA();


        class CV_EXPORTS BroxOpticalFlow : public virtual DenseOpticalFlowExt
        {
        public:
            //! @brief Flow smoothness
            CV_PURE_PROPERTY(double, Alpha)
            //! @brief Gradient constancy importance
            CV_PURE_PROPERTY(double, Gamma)
            //! @brief Pyramid scale factor
            CV_PURE_PROPERTY(double, ScaleFactor)
            //! @brief Number of lagged non-linearity iterations (inner loop)
            CV_PURE_PROPERTY(int, InnerIterations)
            //! @brief Number of warping iterations (number of pyramid levels)
            CV_PURE_PROPERTY(int, OuterIterations)
            //! @brief Number of linear system solver iterations
            CV_PURE_PROPERTY(int, SolverIterations)
        };
        CV_EXPORTS Ptr<BroxOpticalFlow> createOptFlow_Brox_CUDA();


        class PyrLKOpticalFlow : public virtual DenseOpticalFlowExt
        {
        public:
            CV_PURE_PROPERTY(int, WindowSize)
            CV_PURE_PROPERTY(int, MaxLevel)
            CV_PURE_PROPERTY(int, Iterations)
        };
        CV_EXPORTS Ptr<PyrLKOpticalFlow> createOptFlow_PyrLK_CUDA();

//! @}

    }
}

#endif // __OPENCV_SUPERRES_OPTICAL_FLOW_HPP__
