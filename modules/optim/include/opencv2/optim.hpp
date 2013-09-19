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

#ifndef __OPENCV_OPTIM_HPP__
#define __OPENCV_OPTIM_HPP__

#include "opencv2/core.hpp"

namespace cv{namespace optim
{
class CV_EXPORTS Solver : public Algorithm
{
public:
    class CV_EXPORTS Function
    {
    public:
       virtual ~Function() {}
       //! ndim - dimensionality
       virtual double calc(const double* x) const = 0;
    };

    virtual Ptr<Function> getFunction() const = 0;
    virtual void setFunction(const Ptr<Function>& f) = 0;

    virtual TermCriteria getTermCriteria() const = 0;
    virtual void setTermCriteria(const TermCriteria& termcrit) = 0;

    // x contain the initial point before the call and the minima position (if algorithm converged) after. x is assumed to be (something that
    // after getMat() will return) row-vector or column-vector. *It's size  and should
    // be consisted with previous dimensionality data given, if any (otherwise, it determines dimensionality)*
    virtual double minimize(InputOutputArray x) = 0;
};

//! downhill simplex class
class CV_EXPORTS DownhillSolver : public Solver
{
public:
    //! returns row-vector, even if the column-vector was given
    virtual void getInitStep(OutputArray step) const=0;
    //!This should be called at least once before the first call to minimize() and step is assumed to be (something that
    //! after getMat() will return) row-vector or column-vector. *It's dimensionality determines the dimensionality of a problem.*
    virtual void setInitStep(InputArray step)=0;
};

// both minRange & minError are specified by termcrit.epsilon; In addition, user may specify the number of iterations that the algorithm does.
CV_EXPORTS_W Ptr<DownhillSolver> createDownhillSolver(const Ptr<Solver::Function>& f=Ptr<Solver::Function>(),
        InputArray initStep=Mat_<double>(1,1,0.0),
        TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5000,0.000001));

//!the return codes for solveLP() function
enum
{
    SOLVELP_UNBOUNDED    = -2, //problem is unbounded (target function can achieve arbitrary high values)
    SOLVELP_UNFEASIBLE    = -1, //problem is unfeasible (there are no points that satisfy all the constraints imposed)
    SOLVELP_SINGLE    = 0, //there is only one maximum for target function
    SOLVELP_MULTI    = 1 //there are multiple maxima for target function - the arbitrary one is returned
};

CV_EXPORTS_W int solveLP(const Mat& Func, const Mat& Constr, Mat& z);
CV_EXPORTS_W void denoise_TVL1(const std::vector<Mat>& observations,Mat& result, double lambda=1.0, int niters=30);
}}// cv

#endif
