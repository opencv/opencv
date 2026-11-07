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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_CALIB3D_PRIVATE_HPP
#define OPENCV_CALIB3D_PRIVATE_HPP

#ifndef __OPENCV_BUILD
#  error this is a private header which should not be used from outside of the OpenCV library
#endif

#include "opencv2/core.hpp"

//! @cond IGNORED

namespace cv
{

// C++ version of the CvLevMarq solver.
// Main differences between LMSolver and CvLevMarq:
// 1. Damping Factor ($\lambda$) Adjustment
// LevMarq: Uses a static, simple scaling approach. If an iteration reduces the error, $\lambda$ is divided by 10. If the error increases, $\lambda$ is multiplied by 10 (repeatedly within the same step, up to 16 times) until a descending step is found.
// LMSolver: Uses an advanced adaptive gain ratio $R = \frac{\Delta S_{\text{actual}}}{\Delta S_{\text{predicted}}}$. If the step is very successful ($R > 0.75$), $\lambda$ is aggressively halved or set to 0. If the step is poor ($R < 0.25$), $\lambda$ is dynamically multiplied by a computed factor $\nu$ derived from the exact residual mismatch.
// 2. Diagonal Augmentation Strategy
// LevMarq: Directly scales the updated main diagonal of $J^T J$ by $1 + \lambda$ on every step.
// LMSolver: Caches a reference diagonal $D = \operatorname{diag}(J^T J)$ at the start of every accepted step and augments the diagonal by adding $\lambda \cdot D$.
// 3. Stopping Criteria
// LevMarq: Stops when a maximum iteration count is reached, or when the relative $L_2$ norm of the parameter update step falls below $\epsilon$ ($\frac{|\Delta x|_2}{|x|_2} < \epsilon$).
// LMSolver: Checks both the infinity norm of the parameter updates ($|\Delta x|\infty < \epsilon$) and the infinity norm of the residual errors ($|r|\infty < \epsilon$) independently.
class CV_EXPORTS LevMarq
{
public:
    LevMarq();
    LevMarq( int nparams, int nerrs, TermCriteria criteria=
              TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,30,DBL_EPSILON),
              bool completeSymmFlag=false );
    void init( int nparams, int nerrs, TermCriteria criteria=
              TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,30,DBL_EPSILON),
              bool completeSymmFlag=false );
    bool update( Mat1d& param, Mat1d& J, Mat1d& err );
    bool updateAlt( Mat1d& param, Mat1d& JtJ, Mat1d& JtErr, double*& errNorm );

    void step();
    enum { DONE=0, STARTED=1, CALC_J=2, CHECK_ERR=3 };

    Mat1b mask;
    Mat1d prevParam;
    Mat1d param;
    Mat1d J;
    Mat1d err;
    Mat1d JtJ;
    Mat1d JtJN;
    Mat1d JtErr;
    Mat1d JtJV;
    Mat1d JtJW;
    double prevErrNorm, errNorm;
    int lambdaLg10;
    TermCriteria criteria;
    int state;
    int iters;
    bool completeSymmFlag;
    int solveMethod;
};

} // namespace cv

//! @endcond

#endif // OPENCV_CALIB3D_PRIVATE_HPP
