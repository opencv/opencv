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

#ifndef OPENCV_CUDAOPTFLOW_HPP
#define OPENCV_CUDAOPTFLOW_HPP

#ifndef __cplusplus
#  error cudaoptflow.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"

/**
  @addtogroup cuda
  @{
    @defgroup cudaoptflow Optical Flow
  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudaoptflow
//! @{

//
// Interface
//

/** @brief Base interface for dense optical flow algorithms.
 */
class CV_EXPORTS DenseOpticalFlow : public Algorithm
{
public:
    /** @brief Calculates a dense optical flow.

    @param I0 first input image.
    @param I1 second input image of the same size and the same type as I0.
    @param flow computed flow image that has the same size as I0 and type CV_32FC2.
    @param stream Stream for the asynchronous version.
     */
    virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow, Stream& stream = Stream::Null()) = 0;
};

/** @brief Base interface for sparse optical flow algorithms.
 */
class CV_EXPORTS SparseOpticalFlow : public Algorithm
{
public:
    /** @brief Calculates a sparse optical flow.

    @param prevImg First input image.
    @param nextImg Second input image of the same size and the same type as prevImg.
    @param prevPts Vector of 2D points for which the flow needs to be found.
    @param nextPts Output vector of 2D points containing the calculated new positions of input features in the second image.
    @param status Output status vector. Each element of the vector is set to 1 if the
                  flow for the corresponding features has been found. Otherwise, it is set to 0.
    @param err Optional output vector that contains error response for each point (inverse confidence).
    @param stream Stream for the asynchronous version.
     */
    virtual void calc(InputArray prevImg, InputArray nextImg,
                      InputArray prevPts, InputOutputArray nextPts,
                      OutputArray status,
                      OutputArray err = cv::noArray(),
                      Stream& stream = Stream::Null()) = 0;
};

//
// BroxOpticalFlow
//

/** @brief Class computing the optical flow for two images using Brox et al Optical Flow algorithm (@cite Brox2004).
 */
class CV_EXPORTS BroxOpticalFlow : public DenseOpticalFlow
{
public:
    virtual double getFlowSmoothness() const = 0;
    virtual void setFlowSmoothness(double alpha) = 0;

    virtual double getGradientConstancyImportance() const = 0;
    virtual void setGradientConstancyImportance(double gamma) = 0;

    virtual double getPyramidScaleFactor() const = 0;
    virtual void setPyramidScaleFactor(double scale_factor) = 0;

    //! number of lagged non-linearity iterations (inner loop)
    virtual int getInnerIterations() const = 0;
    virtual void setInnerIterations(int inner_iterations) = 0;

    //! number of warping iterations (number of pyramid levels)
    virtual int getOuterIterations() const = 0;
    virtual void setOuterIterations(int outer_iterations) = 0;

    //! number of linear system solver iterations
    virtual int getSolverIterations() const = 0;
    virtual void setSolverIterations(int solver_iterations) = 0;

    static Ptr<BroxOpticalFlow> create(
            double alpha = 0.197,
            double gamma = 50.0,
            double scale_factor = 0.8,
            int inner_iterations = 5,
            int outer_iterations = 150,
            int solver_iterations = 10);
};

//
// PyrLKOpticalFlow
//

/** @brief Class used for calculating a sparse optical flow.

The class can calculate an optical flow for a sparse feature set using the
iterative Lucas-Kanade method with pyramids.

@sa calcOpticalFlowPyrLK

@note
   -   An example of the Lucas Kanade optical flow algorithm can be found at
        opencv_source_code/samples/gpu/pyrlk_optical_flow.cpp
 */
class CV_EXPORTS SparsePyrLKOpticalFlow : public SparseOpticalFlow
{
public:
    virtual Size getWinSize() const = 0;
    virtual void setWinSize(Size winSize) = 0;

    virtual int getMaxLevel() const = 0;
    virtual void setMaxLevel(int maxLevel) = 0;

    virtual int getNumIters() const = 0;
    virtual void setNumIters(int iters) = 0;

    virtual bool getUseInitialFlow() const = 0;
    virtual void setUseInitialFlow(bool useInitialFlow) = 0;

    static Ptr<SparsePyrLKOpticalFlow> create(
            Size winSize = Size(21, 21),
            int maxLevel = 3,
            int iters = 30,
            bool useInitialFlow = false);
};

/** @brief Class used for calculating a dense optical flow.

The class can calculate an optical flow for a dense optical flow using the
iterative Lucas-Kanade method with pyramids.
 */
class CV_EXPORTS DensePyrLKOpticalFlow : public DenseOpticalFlow
{
public:
    virtual Size getWinSize() const = 0;
    virtual void setWinSize(Size winSize) = 0;

    virtual int getMaxLevel() const = 0;
    virtual void setMaxLevel(int maxLevel) = 0;

    virtual int getNumIters() const = 0;
    virtual void setNumIters(int iters) = 0;

    virtual bool getUseInitialFlow() const = 0;
    virtual void setUseInitialFlow(bool useInitialFlow) = 0;

    static Ptr<DensePyrLKOpticalFlow> create(
            Size winSize = Size(13, 13),
            int maxLevel = 3,
            int iters = 30,
            bool useInitialFlow = false);
};

//
// FarnebackOpticalFlow
//

/** @brief Class computing a dense optical flow using the Gunnar Farneback’s algorithm.
 */
class CV_EXPORTS FarnebackOpticalFlow : public DenseOpticalFlow
{
public:
    virtual int getNumLevels() const = 0;
    virtual void setNumLevels(int numLevels) = 0;

    virtual double getPyrScale() const = 0;
    virtual void setPyrScale(double pyrScale) = 0;

    virtual bool getFastPyramids() const = 0;
    virtual void setFastPyramids(bool fastPyramids) = 0;

    virtual int getWinSize() const = 0;
    virtual void setWinSize(int winSize) = 0;

    virtual int getNumIters() const = 0;
    virtual void setNumIters(int numIters) = 0;

    virtual int getPolyN() const = 0;
    virtual void setPolyN(int polyN) = 0;

    virtual double getPolySigma() const = 0;
    virtual void setPolySigma(double polySigma) = 0;

    virtual int getFlags() const = 0;
    virtual void setFlags(int flags) = 0;

    static Ptr<FarnebackOpticalFlow> create(
            int numLevels = 5,
            double pyrScale = 0.5,
            bool fastPyramids = false,
            int winSize = 13,
            int numIters = 10,
            int polyN = 5,
            double polySigma = 1.1,
            int flags = 0);
};

//
// OpticalFlowDual_TVL1
//

/** @brief Implementation of the Zach, Pock and Bischof Dual TV-L1 Optical Flow method.
 *
 * @sa C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
 * @sa Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
 */
class CV_EXPORTS OpticalFlowDual_TVL1 : public DenseOpticalFlow
{
public:
    /**
     * Time step of the numerical scheme.
     */
    virtual double getTau() const = 0;
    virtual void setTau(double tau) = 0;

    /**
     * Weight parameter for the data term, attachment parameter.
     * This is the most relevant parameter, which determines the smoothness of the output.
     * The smaller this parameter is, the smoother the solutions we obtain.
     * It depends on the range of motions of the images, so its value should be adapted to each image sequence.
     */
    virtual double getLambda() const = 0;
    virtual void setLambda(double lambda) = 0;

    /**
     * Weight parameter for (u - v)^2, tightness parameter.
     * It serves as a link between the attachment and the regularization terms.
     * In theory, it should have a small value in order to maintain both parts in correspondence.
     * The method is stable for a large range of values of this parameter.
     */
    virtual double getGamma() const = 0;
    virtual void setGamma(double gamma) = 0;

    /**
     * parameter used for motion estimation. It adds a variable allowing for illumination variations
     * Set this parameter to 1. if you have varying illumination.
     * See: Chambolle et al, A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging
     * Journal of Mathematical imaging and vision, may 2011 Vol 40 issue 1, pp 120-145
     */
    virtual double getTheta() const = 0;
    virtual void setTheta(double theta) = 0;

    /**
     * Number of scales used to create the pyramid of images.
     */
    virtual int getNumScales() const = 0;
    virtual void setNumScales(int nscales) = 0;

    /**
     * Number of warpings per scale.
     * Represents the number of times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale.
     * This is a parameter that assures the stability of the method.
     * It also affects the running time, so it is a compromise between speed and accuracy.
     */
    virtual int getNumWarps() const = 0;
    virtual void setNumWarps(int warps) = 0;

    /**
     * Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.
     * A small value will yield more accurate solutions at the expense of a slower convergence.
     */
    virtual double getEpsilon() const = 0;
    virtual void setEpsilon(double epsilon) = 0;

    /**
     * Stopping criterion iterations number used in the numerical scheme.
     */
    virtual int getNumIterations() const = 0;
    virtual void setNumIterations(int iterations) = 0;

    virtual double getScaleStep() const = 0;
    virtual void setScaleStep(double scaleStep) = 0;

    virtual bool getUseInitialFlow() const = 0;
    virtual void setUseInitialFlow(bool useInitialFlow) = 0;

    static Ptr<OpticalFlowDual_TVL1> create(
            double tau = 0.25,
            double lambda = 0.15,
            double theta = 0.3,
            int nscales = 5,
            int warps = 5,
            double epsilon = 0.01,
            int iterations = 300,
            double scaleStep = 0.8,
            double gamma = 0.0,
            bool useInitialFlow = false);
};

//! @}

}} // namespace cv { namespace cuda {

#endif /* OPENCV_CUDAOPTFLOW_HPP */
