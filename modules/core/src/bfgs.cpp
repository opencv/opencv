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
// Copyright (C) 2018, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv
{
    class BFGSSolverImpl : public BFGSSolver
    {
    public:
        Ptr<Function> getFunction() const;
        void setFunction(const Ptr<Function>& f);
        TermCriteria getTermCriteria() const;
        BFGSSolverImpl();
        void setTermCriteria(const TermCriteria& termcrit);
        double minimize(InputOutputArray x);

    protected:
        Ptr<MinProblemSolver::Function> _Function;
        TermCriteria _termcrit;

    private:
        static int cstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy, double& stp,
                         const double fp, const double dp, bool& brackt, const double stpmin, const double stpmax, int& info);
    };

    BFGSSolverImpl::BFGSSolverImpl() {
        _Function = Ptr<Function>();
    }

    Ptr<MinProblemSolver::Function> BFGSSolverImpl::getFunction() const {
        return _Function;
    }

    void BFGSSolverImpl::setFunction(const Ptr<Function>& f) {
        _Function = f;
    }

    TermCriteria BFGSSolverImpl::getTermCriteria() const {
        return _termcrit;
    }

    void BFGSSolverImpl::setTermCriteria(const TermCriteria& termcrit) {
        CV_Assert(
            (termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) && termcrit.epsilon > 0 && termcrit.maxCount > 0) ||
            ((termcrit.type == TermCriteria::MAX_ITER) && termcrit.maxCount > 0)
        );
        _termcrit = termcrit;
    }

    double BFGSSolverImpl::minimize(InputOutputArray) {
        CV_Assert(_Function.empty() == false);
        // TODO
        return 0;
    }

    Ptr<BFGSSolver> BFGSSolver::create(const Ptr<MinProblemSolver::Function>& f, TermCriteria termcrit) {
        Ptr<BFGSSolver> bfgs = makePtr<BFGSSolverImpl>();
        bfgs->setFunction(f);
        bfgs->setTermCriteria(termcrit);
        return bfgs;
    }

    int BFGSSolverImpl::cstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy, double& stp,
                              const double fp, const double dp, bool& brackt, const double stpmin, const double stpmax, int& info) {
        info = 0;
        bool bound = false;

        if ((brackt & ((stp <= std::min(stx, sty)) | (stp >= std::max(stx, sty)))) | (dx * (stp - stx) >= 0.0) | (stpmax < stpmin)) {
            return -1;
        }

        double sgnd = dp * (dx / std::abs(dx));
        double stpf = 0;
        double stpc = 0;
        double stpq = 0;

        if (fp > fx) {
            info = 1;
            bound = true;
            double theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
            double s = std::max(theta, std::max(dx, dp));
            double gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp < stx)
                gamma = -gamma;
            double p = (gamma - dx) + theta;
            double q = ((gamma - dx) + gamma) + dp;
            double r = p / q;
            stpc = stx + r * (stp - stx);
            stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.) * (stp - stx);
            if (std::abs(stpc - stx) < std::abs(stpq - stx))
                stpf = stpc;
            else
                stpf = stpc + (stpq - stpc) / 2;
            brackt = true;
        } else if (sgnd < 0.0) {
            info = 2;
            bound = false;
            double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            double s = std::max(theta, std::max(dx, dp));
            double gamma = s * std::sqrt((theta / s) * (theta / s)  - (dx / s) * (dp / s));
            if (stp > stx)
                gamma = -gamma;
            double p = (gamma - dp) + theta;
            double q = ((gamma - dp) + gamma) + dx;
            double r = p / q;
            stpc = stp + r * (stx - stp);
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            if (std::abs(stpc - stp) > std::abs(stpq - stp))
                stpf = stpc;
            else
                stpf = stpq;
            brackt = true;
        } else if (std::abs(dp) < std::abs(dx)) {
            info = 3;
            bound = 1;
            double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            double s = std::max(theta, std::max( dx, dp));
            double gamma = s * std::sqrt(std::max(static_cast<double>(0.), (theta / s) * (theta / s) - (dx / s) * (dp / s)));
            if (stp > stx)
                gamma = -gamma;
            double p = (gamma - dp) + theta;
            double q = (gamma + (dx - dp)) + gamma;
            double r = p / q;
            if ((r < 0.0) & (gamma != 0.0)) {
                stpc = stp + r * (stx - stp);
            } else if (stp > stx) {
                stpc = stpmax;
            } else {
                stpc = stpmin;
            }
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            if (brackt) {
                if (std::abs(stp - stpc) < std::abs(stp - stpq)) {
                    stpf = stpc;
                } else {
                    stpf = stpq;
                }
            } else {
                if (std::abs(stp - stpc) > std::abs(stp - stpq)) {
                    stpf = stpc;
                } else {
                    stpf = stpq;
                }
            }
        } else {
            info = 4;
            bound = false;
            if (brackt) {
                double theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
                double s = std::max(theta, std::max(dy, dp));
                double gamma = s * std::sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
                if (stp > sty)
                    gamma = -gamma;
                double p = (gamma - dp) + theta;
                double q = ((gamma - dp) + gamma) + dy;
                double r = p / q;
                stpc = stp + r * (sty - stp);
                stpf = stpc;
            } else if (stp > stx)
                stpf = stpmax;
            else {
                stpf = stpmin;
            }
        }
        if (fp > fx) {
            sty = stp;
            fy = fp;
            dy = dp;
        } else {
            if (sgnd < 0.0) {
                sty = stx;
                fy = fx;
                dy = dx;
            }
            stx = stp;
            fx = fp;
            dx = dp;
        }
        stpf = std::min(stpmax, stpf);
        stpf = std::max(stpmin, stpf);
        stp = stpf;
        if (brackt & bound) {
            if (sty > stx) {
                stp = std::min(stx + static_cast<double>(0.66) * (sty - stx), stp);
            } else {
                stp = std::max(stx + static_cast<double>(0.66) * (sty - stx), stp);
            }
        }

        return 0;
    }
};
