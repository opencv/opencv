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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

#include "precomp.hpp"
#include <stdio.h>

namespace cv {

LevMarq::Settings::Settings():
    jacobiScaling(false),
    upDouble(true),
    useStepQuality(true),
    clampDiagonal(true),
    stepNormInf(false),
    checkRelEnergyChange(true),
    checkMinGradient(true),
    checkStepNorm(true),
    geodesic(false),
    hGeo(1e-4),
    geoScale(0.5),
    stepNormTolerance(1e-6),
    relEnergyDeltaTolerance(1e-6),
    minGradientTolerance(1e-6),
    smallEnergyTolerance(0), // not used by default
    maxIterations(500),
    initialLambda(0.0001),
    initialLmUpFactor(2.0),
    initialLmDownFactor(3.0)
{ }


// from Ceres, equation energy change:
// eq. energy = 1/2 * (residuals + J * step)^2 =
// 1/2 * ( residuals^2 + 2 * residuals^T * J * step + (J*step)^T * J * step)
// eq. energy change = 1/2 * residuals^2 - eq. energy =
// 1/2 * residuals^2 - 1/2 * ( residuals^2 + 2 * residuals^T * J * step + (J*step)^T * J * step) =
// 1/2 * ( residuals^2 - residuals^2 - 2 * residuals^T * J * step - (J*step)^T * J * step) =
// - 1/2 * ( 2 * residuals^T * J * step + (J*step)^T * J * step) =
// - 1/2 * ( 2 * residuals^T * J + (J*step)^T * J ) * step =
// - 1/2 * ( 2 * residuals^T * J + step^T * J^T * J ) * step =
// - 1/2 * step^T * ( 2 * J^t * residuals + J^T * J * step ) =
// - 1/2 * step^T * ( 2 * J^t * residuals + (J^T * J + LMDiag - LMDiag) * step ) =
// - 1/2 * step^T * ( 2 * J^t * residuals + (J^T * J + LMDiag) * step - LMDiag * step ) =
// - 1/2 * step^T * ( 2 * J^t * residuals - J^T * residuals - LMDiag * step ) =
// - 1/2 * step^T * ( J^t * residuals - LMDiag * step ) =
// - 1/2 * x^T * ( jtb - LMDiag * x )
static double calcJacCostChangeLm(const cv::Mat_<double>& jtb, const cv::Mat_<double>& x, const cv::Mat_<double>& lmDiag)
{
    return -0.5 * cv::sum(x.mul(jtb - lmDiag.mul(x)))[0];
}


// calculates J^T*rvv where rvv is second directional derivative of the function in direction v
// rvv = (f(x0 + v*h) - f(x0))/h - J*v)/h
// where v is a LevMarq equation solution
// J^T*rvv = J^T*((f(x0 + v*h) - f(x0))/h - J*v)/h =
// J^T*(f(x0 + v*h) - f(x0) - J*v*h)/h^2 =
// (J^T*f(x0 + v*h) - J^T*f(x0) - J^T*J*v*h)/h^2 =
// < using (J^T*J + lmdiag) * v = J^T*b, also f(x0 + v*h) = b_v, f(x0) = b >
// (J^T*b_v - J^T*b - (J^T*J + lmdiag - lmdiag)*v*h)/h^2 =
// (J^T*b_v - J^T*b + J^t*b*h + lmdiag*v*h)/h^2 =
// (J^T*b_v - J^t*b*(1 - h) + lmdiag*v*h)/h^2
static void calcJtrvv(const Mat_<double>& jtbv, const Mat_<double>& jtb, const Mat_<double>& lmdiag,
                      const Mat_<double>& v, const double hGeo,
                      Mat_<double>& jtrvv)
{
    jtrvv = (jtbv + jtb * (hGeo - 1.0) + lmdiag.mul(v, hGeo)) / (hGeo * hGeo);
}


LevMarq::Report detail::LevMarqBase::optimize()
{
    if (settings.geodesic && !backend->enableGeo())
    {
        CV_LOG_INFO(NULL, "The backend does not support geodesic acceleration, please turn off the corresponding option");
        return LevMarq::Report(false, 0, 0); // not found
    }

    // this function sets probe vars to current X
    backend->prepareVars();

    double energy = 0.0;
    if (!backend->calcFunc(energy, /*calcEnergy*/ true, /*calcJacobian*/ true) || energy < 0)
    {
        CV_LOG_INFO(NULL, "Error while calculating energy function");
        return LevMarq::Report(false, 0, 0); // not found
    }

    double oldEnergy = energy;

    CV_LOG_DEBUG(NULL, "#s" << " energy: " << energy);

    // diagonal clamping options
    const double minDiag = 1e-6;
    const double maxDiag = 1e32;
    // limit lambda inflation
    const double maxLambda = 1e32;

    // finish reasons
    bool tooLong = false; // => not found
    bool bigLambda = false; // => not found
    bool smallGradient = false; // => found
    bool smallStep = false; // => found
    bool smallEnergyDelta = false; // => found
    bool smallEnergy = false; // => found

    // column scale inverted, for jacobian scaling
    Mat_<double> di;

    // do the jacobian conditioning improvement used in Ceres
    if (settings.jacobiScaling)
    {
        // L2-normalize each jacobian column
        // vec d = {d_j = sum(J_ij^2) for each column j of J} = get_diag{ J^T * J }
        // di = { 1/(1+sqrt(d_j)) }, extra +1 to avoid div by zero
        Mat_<double> ds;
        const Mat_<double> diag = backend->getDiag();
        cv::sqrt(diag, ds);
        di = 1.0 / (ds + 1.0);
    }

    double lmUpFactor = settings.initialLmUpFactor;
    double lambdaLevMarq = settings.initialLambda;

    unsigned int iter = 0;
    bool done = false;
    while (!done)
    {
        // At this point we should have jtj and jtb built

        CV_LOG_DEBUG(NULL, "#LM#s" << " energy: " << energy);

        // do the jacobian conditioning improvement used in Ceres
        if (settings.jacobiScaling)
        {
            backend->doJacobiScaling(di);
        }

        const Mat_<double> jtb = backend->getJtb();

        double gradientMax = cv::norm(jtb, NORM_INF);

        // Save original diagonal of jtj matrix for LevMarq
        const Mat_<double> diag = backend->getDiag();

        // Solve using LevMarq and get delta transform
        bool enoughLm = false;

        while (!enoughLm && !done)
        {
            // form LevMarq matrix
            Mat_<double> lmDiag, jtjDiag;
            lmDiag = diag * lambdaLevMarq;
            if (settings.clampDiagonal)
                lmDiag = cv::min(cv::max(lmDiag, minDiag), maxDiag);
            jtjDiag = lmDiag + diag;
            backend->setDiag(jtjDiag);

            CV_LOG_DEBUG(NULL, "linear decompose...");

            bool decomposed = backend->decompose();

            CV_LOG_DEBUG(NULL, (decomposed ? "OK" : "FAIL"));

            CV_LOG_DEBUG(NULL, "linear solve...");
            // use double or convert everything to float
            Mat_<double> x((int)jtb.rows, 1);
            bool solved = decomposed && backend->solveDecomposed(jtb, x);

            CV_LOG_DEBUG(NULL, (solved ? "OK" : "FAIL"));

            double costChange = 0.0;
            double jacCostChange = 0.0;
            double stepQuality = 0.0;
            double xNorm = 0.0;
            if (solved)
            {
                // what energy drop should be according to local model estimation
                jacCostChange = calcJacCostChangeLm(jtb, x, lmDiag);

                // x norm
                xNorm = cv::norm(x, settings.stepNormInf ? NORM_INF : NORM_L2SQR);

                // undo jacobi scaling
                if (settings.jacobiScaling)
                {
                    x = x.mul(di);
                }

                if (settings.geodesic)
                {
                    backend->currentOplusX(x * settings.hGeo, /*geo*/ true);

                    Mat_<double> jtbv(jtb.rows, 1);
                    if (backend->calcJtbv(jtbv))
                    {
                        Mat_<double> jtrvv(jtb.rows, 1);
                        calcJtrvv(jtbv, jtb, lmDiag, x, settings.hGeo, jtrvv);

                        Mat_<double> xgeo((int)jtb.rows, 1);
                        bool geoSolved = backend->solveDecomposed(jtrvv, xgeo);

                        if (geoSolved)
                        {
                            double truncerr = sqrt(xgeo.dot(xgeo) / x.dot(x));
                            bool geoIsGood = (truncerr < 1.0);
                            if (geoIsGood)
                                x += xgeo * settings.geoScale;

                            CV_LOG_DEBUG(NULL, "Geo truncerr: " << truncerr << (geoIsGood ? ", use it" : ", skip it") );
                        }
                        else
                        {
                            CV_LOG_DEBUG(NULL, "Geo: failed to solve");
                        }
                    }
                }

                // calc energy with current delta x
                backend->currentOplusX(x, /*geo*/ false);

                bool success = backend->calcFunc(energy, /*calcEnergy*/ true, /*calcJacobian*/ false);
                if (!success || energy < 0 || std::isnan(energy))
                {
                    CV_LOG_INFO(NULL, "Error while calculating energy function");
                    return LevMarq::Report(false, iter, oldEnergy); // not found
                }

                costChange = oldEnergy - energy;

                stepQuality = costChange / jacCostChange;

                CV_LOG_DEBUG(NULL, "#LM#" << iter
                    << " energy: " << energy
                    << " deltaEnergy: " << costChange
                    << " deltaEqEnergy: " << jacCostChange
                    << " max(J^T*b): " << gradientMax
                    << (settings.stepNormInf ? " normInf(x): " : " norm2(x): ") << xNorm
                    << " deltaEnergy/energy: " << costChange / energy);
            }

            // zero cost change is treated like an algorithm failure if checkRelEnergyChange is off
            if (!solved || costChange < 0 || (!settings.checkRelEnergyChange && abs(costChange) < DBL_EPSILON))
            {
                // failed to optimize, increase lambda and repeat

                lambdaLevMarq *= lmUpFactor;
                if (settings.upDouble)
                    lmUpFactor *= 2.0;

                CV_LOG_DEBUG(NULL, "LM goes up, lambda: " << lambdaLevMarq << ", old energy: " << oldEnergy);
            }
            else
            {
                // optimized successfully, decrease lambda and set variables for next iteration
                enoughLm = true;

                if (settings.useStepQuality)
                    lambdaLevMarq *= std::max(1.0 / settings.initialLmDownFactor, 1.0 - pow(2.0 * stepQuality - 1.0, 3));
                else
                    lambdaLevMarq *= 1.0 / settings.initialLmDownFactor;
                lmUpFactor = settings.initialLmUpFactor;

                // Once set, these flags will be activated until next successful LM iteration - this is not a bug
                smallGradient = (gradientMax < settings.minGradientTolerance);
                smallStep = (xNorm < settings.stepNormTolerance);
                smallEnergyDelta = (costChange / energy < settings.relEnergyDeltaTolerance);
                smallEnergy = (energy < settings.smallEnergyTolerance);

                backend->acceptProbe();

                CV_LOG_DEBUG(NULL, "#" << iter << " energy: " << energy);

                oldEnergy = energy;

                CV_LOG_DEBUG(NULL, "LM goes down, lambda: " << lambdaLevMarq << " step quality: " << stepQuality);
            }

            iter++;

            tooLong = (iter >= settings.maxIterations);
            bigLambda = (lambdaLevMarq >= maxLambda);

            done = tooLong || bigLambda;
            done = done || (settings.checkMinGradient && smallGradient);
            done = done || (settings.checkStepNorm && smallStep);
            done = done || (settings.checkRelEnergyChange && smallEnergyDelta);
            done = done || (smallEnergy);
        }

        // calc jacobian for next iteration
        if (!done)
        {
            double dummy;
            if (!backend->calcFunc(dummy, /*calcEnergy*/ false, /*calcJacobian*/ true))
            {
                CV_LOG_INFO(NULL, "Error while calculating jacobian");
                return LevMarq::Report(false, iter, oldEnergy); // not found
            }
        }
    }

    bool found = (smallGradient || smallStep || smallEnergyDelta || smallEnergy);

    CV_LOG_DEBUG(NULL, "Finished: " << (found ? "" : "not ") << "found");
    std::string fr = "Finish reason: ";
    if (settings.checkMinGradient && smallGradient)
    {
        CV_LOG_DEBUG(NULL, fr + "gradient max val dropped below threshold");
    }
    if (settings.checkStepNorm && smallStep)
    {
        CV_LOG_DEBUG(NULL, fr + "step size dropped below threshold");
    }
    if (settings.checkRelEnergyChange && smallEnergyDelta)
    {
        CV_LOG_DEBUG(NULL, fr + "relative energy change between iterations dropped below threshold");
    }
    if (smallEnergy)
    {
        CV_LOG_DEBUG(NULL, fr + "energy dropped below threshold");
    }
    if (tooLong)
    {
        CV_LOG_DEBUG(NULL, fr + "max number of iterations reached");
    }
    if (bigLambda)
    {
        CV_LOG_DEBUG(NULL, fr + "lambda has grown above the threshold, the trust region is too small");
    }

    return LevMarq::Report(found, iter, oldEnergy);
}


struct LevMarqDenseLinearBackend : public detail::LevMarqBackend
{
    // all variables including fixed ones
    size_t nVars;
    size_t allVars;
    Mat_<double> jtj, jtb;
    Mat_<double> probeX, currentX;
    // for oplus operation
    Mat_<double> delta;

    // "Long" callback: f(x, &b, &J) -> bool
    // Produces jacobian and residuals for each energy term
    LevMarq::LongCallback cb;
    // "Normal" callback: f(x, &jtb, &jtj, &energy) -> bool
    // Produces J^T*J and J^T*b directly instead of J and b
    LevMarq::NormalCallback cb_alt;

    Mat_<uchar> mask;
    // full matrices containing all vars including fixed ones
    // used only when mask is not empty
    Mat_<double> jtjFull, jtbFull;

    // used only with long callback
    Mat_<double> jLong, bLong;
    size_t nerrs;
    // used only with alt. callback
    // What part of symmetric matrix is to copy to another part
    bool LtoR;
    // What method to use for linear system solving
    int solveMethod;

    // for geodesic acceleration
    bool useGeo;
    // x0 + v*h variable
    Mat_<double> geoX;
    // J^T*rvv vector
    Mat_<double> jtrvv;

    LevMarqDenseLinearBackend(int nvars_, LevMarq::LongCallback callback_, InputArray mask_, int nerrs_, int solveMethod_) :
        LevMarqDenseLinearBackend(noArray(), nvars_, callback_, nullptr, nerrs_, false, mask_, solveMethod_)
    { }
    LevMarqDenseLinearBackend(int nvars_, LevMarq::NormalCallback callback_, InputArray mask_, bool LtoR_, int solveMethod_) :
        LevMarqDenseLinearBackend(noArray(), nvars_, nullptr, callback_, 0, LtoR_, mask_, solveMethod_)
    { }
    LevMarqDenseLinearBackend(InputOutputArray param_, LevMarq::LongCallback callback_, InputArray mask_, int nerrs_, int solveMethod_) :
        LevMarqDenseLinearBackend(param_, 0, callback_, nullptr, nerrs_, false, mask_, solveMethod_)
    { }
    LevMarqDenseLinearBackend(InputOutputArray param_, LevMarq::NormalCallback callback_, InputArray mask_, bool LtoR_, int solveMethod_) :
        LevMarqDenseLinearBackend(param_, 0, nullptr, callback_, 0, LtoR_, mask_, solveMethod_)
    { }

    LevMarqDenseLinearBackend(InputOutputArray currentX_, int nvars,
        LevMarq::LongCallback cb_ = nullptr,
        LevMarq::NormalCallback cb_alt_ = nullptr,
        size_t nerrs_ = 0,
        bool LtoR_ = false,
        InputArray mask_ = noArray(),
        int solveMethod_ = DECOMP_SVD) :
        LevMarqBackend(),
        // these fields will be initialized at prepareVars()
        jtj(),
        jtb(),
        probeX(),
        delta(),
        jtjFull(),
        jtbFull(),
        jLong(),
        bLong(),
        useGeo()
    {
        if (!currentX_.empty())
        {
            CV_Assert(currentX_.type() == CV_64F);
            CV_Assert(currentX_.rows() == 1 || currentX_.cols() == 1);
            this->allVars = currentX_.size().area();
            this->currentX = currentX_.getMat().reshape(1, (int)this->allVars);
        }
        else
        {
            CV_Assert(nvars > 0);
            this->allVars = nvars;
            this->currentX = Mat_<double>((int)this->allVars, 1);
        }

        CV_Assert((cb_ || cb_alt_) && !(cb_ && cb_alt_));
        this->cb = cb_;
        this->cb_alt = cb_alt_;

        this->nerrs = nerrs_;
        this->LtoR = LtoR_;
        this->solveMethod = solveMethod_;

        if (!mask_.empty())
        {
            CV_Assert(mask_.depth() == CV_8U || mask_.depth() == CV_8S || mask_.depth() == CV_Bool);
            CV_Assert(mask_.size() == currentX_.size());
            int maskSize = mask_.size().area();
            this->mask.create(maskSize, 1);
            mask_.copyTo(this->mask);
        }
        else
            this->mask = Mat_<uchar>();

        this->nVars = this->mask.empty() ? this->allVars : countNonZero(this->mask);
        CV_Assert(this->nVars > 0);
    }

    virtual bool enableGeo() CV_OVERRIDE
    {
        useGeo = true;
        return true;
    }

    virtual void prepareVars() CV_OVERRIDE
    {
        probeX = currentX.clone();
        delta = Mat_<double>((int)allVars, 1);

        // Allocate vars for use with mask
        if (!mask.empty())
        {
            jtjFull = Mat_<double>((int)allVars, (int)allVars);
            jtbFull = Mat_<double>((int)allVars, 1);
            jtj = Mat_<double>((int)nVars, (int)nVars);
            jtb = Mat_<double>((int)nVars, 1);
        }
        else
        {
            jtj = Mat_<double>((int)nVars, (int)nVars);
            jtb = Mat_<double>((int)nVars, 1);
            jtjFull = jtj;
            jtbFull = jtb;
        }

        if (nerrs)
        {
            jLong = Mat_<double>((int)nerrs, (int)allVars);
            bLong = Mat_<double>((int)nerrs, 1, CV_64F);
        }

        if (useGeo)
        {
            geoX = currentX.clone();
            jtrvv = jtb.clone();
        }
    }

    // adds x to current variables and writes result to probe vars
    virtual void currentOplusX(const Mat_<double>& x, bool geo) CV_OVERRIDE
    {
        // 'unpack' the param delta
        int j = 0;
        if (!mask.empty())
        {
            for (int i = 0; i < (int)allVars; i++)
            {
                delta.at<double>(i) = (mask.at<uchar>(i) != 0) ? x(j++) : 0.0;
            }
        }
        else
            delta = x;

        if (geo)
        {
            if (useGeo)
            {
                geoX = currentX + delta;
            }
            else
            {
                CV_Error(cv::Error::StsBadArg, "Geodesic acceleration is disabled");
            }
        }
        else
        {
            probeX = currentX + delta;
        }
    }

    virtual void acceptProbe() CV_OVERRIDE
    {
        probeX.copyTo(currentX);
    }

    static void subMatrix(const Mat_<double>& src, Mat_<double>& dst, const Mat_<uchar>& mask)
    {
        CV_Assert(src.type() == CV_64F && dst.type() == CV_64F);
        int m = src.rows, n = src.cols;
        int i1 = 0, j1 = 0;
        for (int i = 0; i < m; i++)
        {
            if (mask(i))
            {
                const double* srcptr = src[i];
                double* dstptr = dst[i1++];

                for (int j = j1 = 0; j < n; j++)
                {
                    if (n < m || mask(j))
                        dstptr[j1++] = srcptr[j];
                }
            }
        }
    }

    virtual bool calcFunc(double& energy, bool calcEnergy = true, bool calcJacobian = false) CV_OVERRIDE
    {
        Mat_<double> xd = probeX;

        double sd = 0.0;
        if (calcJacobian)
        {
            jtbFull.setZero();
            jtjFull.setZero();

            if (!cb_alt)
            {
                jLong.setZero();
            }
        }

        if (cb_alt)
        {
            bool r = calcJacobian ? cb_alt(xd, jtbFull, jtjFull, sd) : cb_alt(xd, noArray(), noArray(), sd);
            if (!r)
                return false;
        }
        else
        {
            bLong.setZero();
            bool r = calcJacobian ? cb(xd, bLong, jLong) : cb(xd, bLong, noArray());
            if (!r)
                return false;
        }

        if (calcJacobian)
        {
            if (cb_alt)
            {
                completeSymm(jtjFull, LtoR);
            }
            else
            {
                mulTransposed(jLong, jtjFull, true);
                gemm(jLong, bLong, 1, noArray(), 0, jtbFull, GEMM_1_T);
            }
        }

        if (calcEnergy)
        {
            if (cb_alt)
            {
                energy = sd;
            }
            else
            {
                energy = norm(bLong, NORM_L2SQR);
            }
        }

        if (!mask.empty())
        {
            subMatrix(jtjFull, jtj, mask);
            subMatrix(jtbFull, jtb, mask);
        }
        else
        {
            jtj = jtjFull;
            jtb = jtbFull;
        }

        return true;
    }

    virtual const Mat_<double> getDiag() CV_OVERRIDE
    {
        return jtj.diag().clone();
    }

    virtual const Mat_<double> getJtb() CV_OVERRIDE
    {
        return jtb;
    }

    virtual void setDiag(const Mat_<double>& d) CV_OVERRIDE
    {
        d.copyTo(jtj.diag());
    }

    virtual void doJacobiScaling(const Mat_<double>& di) CV_OVERRIDE
    {
        // J := J * d_inv, d_inv = make_diag(di)
        // J^T*J := (J * d_inv)^T * J * d_inv = diag(di) * (J^T * J) * diag(di) = eltwise_mul(J^T*J, di*di^T)
        // J^T*b := (J * d_inv)^T * b = d_inv^T * J^T*b = eltwise_mul(J^T*b, di)
        // scaling J^T*J
        for (int i = 0; i < (int)nVars; i++)
        {
            double* jtjrow = jtj.ptr<double>(i);
            for (int j = 0; j < (int)nVars; j++)
            {
                jtjrow[j] *= di(i) * di(j);
            }
        }
        // scaling J^T*b
        for (int i = 0; i < (int)nVars; i++)
        {
            jtb(i) *= di(i);
        }
    }

    virtual bool decompose() CV_OVERRIDE
    {
        //TODO: do the real decomposition
        return true;
    }

    virtual bool solveDecomposed(const Mat_<double>& right, Mat_<double>& x) CV_OVERRIDE
    {
        return cv::solve(jtj, -right, x, solveMethod);
    }

    // calculates J^T*f(geo)
    virtual bool calcJtbv(Mat_<double>& jtbv) CV_OVERRIDE
    {
        if (cb_alt)
        {
            CV_Error(cv::Error::StsNotImplemented, "Geodesic acceleration is not implemented for normal callbacks, please use \"long\" callbacks");
        }
        else
        {
            Mat_<double> b_v = bLong.clone();
            bool r = cb(geoX, b_v, noArray());
            if (!r)
                return false;

            Mat_<double> jLongFiltered(jLong.rows, (int)nVars);
            int ctr = 0;
            for (int i = 0; i < (int)allVars; i++)
            {
                if (mask.empty() || mask(i))
                {
                    jLong.col(i).copyTo(jLongFiltered.col(ctr));
                    ctr++;
                }
            }

            jtbv = jLongFiltered.t() * b_v;
            return true;
        }
    }
};

class LevMarq::Impl : public detail::LevMarqBase
{
public:
    Impl(const Ptr<detail::LevMarqBackend>& backend_, const LevMarq::Settings& settings_) :
        LevMarqBase(backend_, settings_)
    { }

    Report run(InputOutputArray param)
    {
        CV_Assert(!param.empty() && (param.type() == CV_64F) && (param.rows() == 1 || param.cols() == 1));
        backend.dynamicCast<LevMarqDenseLinearBackend>()->currentX = param.getMat().reshape(1, param.size().area());
        return optimize();
    }
};


LevMarq::LevMarq(int nvars, LongCallback callback, const Settings& settings, InputArray mask,
                 MatrixType matrixType, VariableType paramType, int nerrs, int solveMethod)
{
    if (matrixType != MatrixType::AUTO && matrixType != MatrixType::DENSE)
        CV_Error(cv::Error::StsNotImplemented, "General purpuse sparse solver for LevMarq is not implemented yet");
    if (paramType != VariableType::LINEAR)
        CV_Error(cv::Error::StsNotImplemented, "SO(3) and SE(3) params for LevMarq are not implemented yet");

    auto backend = makePtr<LevMarqDenseLinearBackend>(nvars, callback, mask, nerrs, solveMethod);
    pImpl = makePtr<LevMarq::Impl>(backend, settings);
}

LevMarq::LevMarq(int nvars, NormalCallback callback, const Settings& settings, InputArray mask,
                 MatrixType matrixType, VariableType paramType, bool LtoR, int solveMethod)
{
    if (matrixType != MatrixType::AUTO && matrixType != MatrixType::DENSE)
        CV_Error(cv::Error::StsNotImplemented, "General purpuse sparse solver for LevMarq is not implemented yet");
    if (paramType != VariableType::LINEAR)
        CV_Error(cv::Error::StsNotImplemented, "SO(3) and SE(3) params for LevMarq are not implemented yet");

    auto backend = makePtr<LevMarqDenseLinearBackend>(nvars, callback, mask, LtoR, solveMethod);
    pImpl = makePtr<LevMarq::Impl>(backend, settings);
}

LevMarq::LevMarq(InputOutputArray param, LongCallback callback, const Settings& settings, InputArray mask,
                 MatrixType matrixType, VariableType paramType, int nerrs, int solveMethod)
{
    if (matrixType != MatrixType::AUTO && matrixType != MatrixType::DENSE)
        CV_Error(cv::Error::StsNotImplemented, "General purpuse sparse solver for LevMarq is not implemented yet");
    if (paramType != VariableType::LINEAR)
        CV_Error(cv::Error::StsNotImplemented, "SO(3) and SE(3) params for LevMarq are not implemented yet");

    auto backend = makePtr<LevMarqDenseLinearBackend>(param, callback, mask, nerrs, solveMethod);
    pImpl = makePtr<LevMarq::Impl>(backend, settings);
}

LevMarq::LevMarq(InputOutputArray param, NormalCallback callback, const Settings& settings, InputArray mask,
                 MatrixType matrixType, VariableType paramType, bool LtoR, int solveMethod)
{
    if (matrixType != MatrixType::AUTO && matrixType != MatrixType::DENSE)
        CV_Error(cv::Error::StsNotImplemented, "General purpuse sparse solver for LevMarq is not implemented yet");
    if (paramType != VariableType::LINEAR)
        CV_Error(cv::Error::StsNotImplemented, "SO(3) and SE(3) params for LevMarq are not implemented yet");

    auto backend = makePtr<LevMarqDenseLinearBackend>(param, callback, mask, LtoR, solveMethod);
    pImpl = makePtr<LevMarq::Impl>(backend, settings);
}

LevMarq::Report LevMarq::optimize()
{
    return pImpl->optimize();
}

LevMarq::Report LevMarq::run(InputOutputArray param)
{
    return pImpl->run(param);
}

}
