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
            /** @see setPyrScale */
            virtual double getPyrScale() const = 0;
            /** @copybrief getPyrScale @see getPyrScale */
            virtual void setPyrScale(double val) = 0;
            /** @see setLevelsNumber */
            virtual int getLevelsNumber() const = 0;
            /** @copybrief getLevelsNumber @see getLevelsNumber */
            virtual void setLevelsNumber(int val) = 0;
            /** @see setWindowSize */
            virtual int getWindowSize() const = 0;
            /** @copybrief getWindowSize @see getWindowSize */
            virtual void setWindowSize(int val) = 0;
            /** @see setIterations */
            virtual int getIterations() const = 0;
            /** @copybrief getIterations @see getIterations */
            virtual void setIterations(int val) = 0;
            /** @see setPolyN */
            virtual int getPolyN() const = 0;
            /** @copybrief getPolyN @see getPolyN */
            virtual void setPolyN(int val) = 0;
            /** @see setPolySigma */
            virtual double getPolySigma() const = 0;
            /** @copybrief getPolySigma @see getPolySigma */
            virtual void setPolySigma(double val) = 0;
            /** @see setFlags */
            virtual int getFlags() const = 0;
            /** @copybrief getFlags @see getFlags */
            virtual void setFlags(int val) = 0;
        };
        CV_EXPORTS Ptr<FarnebackOpticalFlow> createOptFlow_Farneback();
        CV_EXPORTS Ptr<FarnebackOpticalFlow> createOptFlow_Farneback_CUDA();


//        CV_EXPORTS Ptr<DenseOpticalFlowExt> createOptFlow_Simple();


        class CV_EXPORTS DualTVL1OpticalFlow : public virtual DenseOpticalFlowExt
        {
        public:
            /** @see setTau */
            virtual double getTau() const = 0;
            /** @copybrief getTau @see getTau */
            virtual void setTau(double val) = 0;
            /** @see setLambda */
            virtual double getLambda() const = 0;
            /** @copybrief getLambda @see getLambda */
            virtual void setLambda(double val) = 0;
            /** @see setTheta */
            virtual double getTheta() const = 0;
            /** @copybrief getTheta @see getTheta */
            virtual void setTheta(double val) = 0;
            /** @see setScalesNumber */
            virtual int getScalesNumber() const = 0;
            /** @copybrief getScalesNumber @see getScalesNumber */
            virtual void setScalesNumber(int val) = 0;
            /** @see setWarpingsNumber */
            virtual int getWarpingsNumber() const = 0;
            /** @copybrief getWarpingsNumber @see getWarpingsNumber */
            virtual void setWarpingsNumber(int val) = 0;
            /** @see setEpsilon */
            virtual double getEpsilon() const = 0;
            /** @copybrief getEpsilon @see getEpsilon */
            virtual void setEpsilon(double val) = 0;
            /** @see setIterations */
            virtual int getIterations() const = 0;
            /** @copybrief getIterations @see getIterations */
            virtual void setIterations(int val) = 0;
            /** @see setUseInitialFlow */
            virtual bool getUseInitialFlow() const = 0;
            /** @copybrief getUseInitialFlow @see getUseInitialFlow */
            virtual void setUseInitialFlow(bool val) = 0;
        };
        CV_EXPORTS Ptr<DualTVL1OpticalFlow> createOptFlow_DualTVL1();
        CV_EXPORTS Ptr<DualTVL1OpticalFlow> createOptFlow_DualTVL1_CUDA();


        class CV_EXPORTS BroxOpticalFlow : public virtual DenseOpticalFlowExt
        {
        public:
            //! @brief Flow smoothness
            /** @see setAlpha */
            virtual double getAlpha() const = 0;
            /** @copybrief getAlpha @see getAlpha */
            virtual void setAlpha(double val) = 0;
            //! @brief Gradient constancy importance
            /** @see setGamma */
            virtual double getGamma() const = 0;
            /** @copybrief getGamma @see getGamma */
            virtual void setGamma(double val) = 0;
            //! @brief Pyramid scale factor
            /** @see setScaleFactor */
            virtual double getScaleFactor() const = 0;
            /** @copybrief getScaleFactor @see getScaleFactor */
            virtual void setScaleFactor(double val) = 0;
            //! @brief Number of lagged non-linearity iterations (inner loop)
            /** @see setInnerIterations */
            virtual int getInnerIterations() const = 0;
            /** @copybrief getInnerIterations @see getInnerIterations */
            virtual void setInnerIterations(int val) = 0;
            //! @brief Number of warping iterations (number of pyramid levels)
            /** @see setOuterIterations */
            virtual int getOuterIterations() const = 0;
            /** @copybrief getOuterIterations @see getOuterIterations */
            virtual void setOuterIterations(int val) = 0;
            //! @brief Number of linear system solver iterations
            /** @see setSolverIterations */
            virtual int getSolverIterations() const = 0;
            /** @copybrief getSolverIterations @see getSolverIterations */
            virtual void setSolverIterations(int val) = 0;
        };
        CV_EXPORTS Ptr<BroxOpticalFlow> createOptFlow_Brox_CUDA();


        class PyrLKOpticalFlow : public virtual DenseOpticalFlowExt
        {
        public:
            /** @see setWindowSize */
            virtual int getWindowSize() const = 0;
            /** @copybrief getWindowSize @see getWindowSize */
            virtual void setWindowSize(int val) = 0;
            /** @see setMaxLevel */
            virtual int getMaxLevel() const = 0;
            /** @copybrief getMaxLevel @see getMaxLevel */
            virtual void setMaxLevel(int val) = 0;
            /** @see setIterations */
            virtual int getIterations() const = 0;
            /** @copybrief getIterations @see getIterations */
            virtual void setIterations(int val) = 0;
        };
        CV_EXPORTS Ptr<PyrLKOpticalFlow> createOptFlow_PyrLK_CUDA();

//! @}

    }
}

#endif // __OPENCV_SUPERRES_OPTICAL_FLOW_HPP__
