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
    class LBFGSSolverImpl : public LBFGSSolver
    {
    public:
        Ptr<Function> getFunction() const;
        void setFunction(const Ptr<Function>& f);
        TermCriteria getTermCriteria() const;
        LBFGSSolverImpl();
        void setTermCriteria(const TermCriteria& termcrit);
        double minimize(InputOutputArray x);

    private:
        double linesearch(const Mat &x, const Mat &searchDir, double a);
        int cvsrch(Mat &x, double f, Mat &g, double &stp, Mat &s);
        static int cstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy, double& stp,
                         const double fp, const double dp, bool& brackt, const double stpmin, const double stpmax, int& info);

    protected:
        Ptr<MinProblemSolver::Function> _Function;
        TermCriteria _termcrit;
    };

    LBFGSSolverImpl::LBFGSSolverImpl() {
        _Function = Ptr<Function>();
    }

    Ptr<MinProblemSolver::Function> LBFGSSolverImpl::getFunction() const {
        return _Function;
    }

    void LBFGSSolverImpl::setFunction(const Ptr<Function>& f) {
        _Function = f;
    }

    TermCriteria LBFGSSolverImpl::getTermCriteria() const {
        return _termcrit;
    }

    void LBFGSSolverImpl::setTermCriteria(const TermCriteria& termcrit) {
        CV_Assert(
            (termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) && termcrit.epsilon > 0 && termcrit.maxCount > 0) ||
            ((termcrit.type == TermCriteria::MAX_ITER) && termcrit.maxCount > 0)
        );
        _termcrit = termcrit;
    }

    double LBFGSSolverImpl::linesearch(const Mat &x, const Mat &searchDir, double a) {
        double ak = a;
        double fval = _Function->calc(x.ptr<double>());
        Mat g;
        x.copyTo(g);
        _Function->getGradient(x.ptr<double>(), g.ptr<double>());

        Mat s;
        searchDir.copyTo(s);
        Mat xx;
        x.copyTo(xx);

        cvsrch(xx, fval, g, ak, s);

        return ak;
    }

    int LBFGSSolverImpl::cvsrch(Mat &x, double f, Mat &g, double &stp, Mat &s) {
        int info = 0;
        int infoc = 1;
        const double xtol = 1e-15;
        const double ftol = 1e-4;
        const double gtol = 1e-2;
        const double stpmin = 1e-15;
        const double stpmax = 1e15;
        const double xtrapf = 4;
        const int maxfev = 20;
        int nfev = 0;

        double dginit = g.dot(s);
        if (dginit >= 0.0)
            return -1;

        bool brackt = false;
        bool stage1 = true;

        double finit = f;
        double dgtest = ftol * dginit;
        double width = stpmax - stpmin;
        double width1 = 2 * width;
        Mat wa;
        x.copyTo(wa);

        double stx = 0.0;
        double fx = finit;
        double dgx = dginit;
        double sty = 0.0;
        double fy = finit;
        double dgy = dginit;
        double stmin;
        double stmax;

        while (true) {
            if (brackt) {
                stmin = std::min(stx, sty);
                stmax = std::max(stx, sty);
            } else {
                stmin = stx;
                stmax = stp + xtrapf * (stp - stx);
            }

            stp = std::max(stp, stpmin);
            stp = std::min(stp, stpmax);

            if ((brackt && ((stp <= stmin) || (stp >= stmax))) || (nfev >= maxfev - 1 ) || (infoc == 0)
                || (brackt && ((stmax - stmin) <= (xtol * stmax)))) {
                stp = stx;
            }

            x = wa + stp * s;
            f = _Function->calc(x.ptr<double>());
            _Function->getGradient(x.ptr<double>(), g.ptr<double>());
            nfev++;
            double dg = g.dot(s);
            double ftest1 = finit + stp * dgtest;

            if ((brackt & ((stp <= stmin) | (stp >= stmax))) | (infoc == 0))
                info = 6;

            if ((stp == stpmax) & (f <= ftest1) & (dg <= dgtest))
                info = 5;

            if ((stp == stpmin) & ((f > ftest1) | (dg >= dgtest)))
                info = 4;

            if (nfev >= maxfev)
                info = 3;

            if (brackt & (stmax - stmin <= xtol * stmax))
                info = 2;

            if ((f <= ftest1) & (std::abs(dg) <= gtol * (-dginit)))
                info = 1;

            if (info != 0)
                return -1;

            if (stage1 & (f <= ftest1) & (dg >= std::min(ftol, gtol) * dginit))
                stage1 = false;

            if (stage1 & (f <= fx) & (f > ftest1)) {
                double fm = f - stp * dgtest;
                double fxm = fx - stx * dgtest;
                double fym = fy - sty * dgtest;
                double dgm = dg - dgtest;
                double dgxm = dgx - dgtest;
                double dgym = dgy - dgtest;
                cstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax, infoc);
                fx = fxm + stx * dgtest;
                fy = fym + sty * dgtest;
                dgx = dgxm + dgtest;
                dgy = dgym + dgtest;
            } else {
                cstep(stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin, stmax, infoc);
            }

            if (brackt) {
                if (std::abs(sty - stx) >= 0.66 * width1)
                    stp = stx + 0.5 * (sty - stx);
                width1 = width;
                width = std::abs(sty - stx);
            }
        }

        return 0;
    }

    int LBFGSSolverImpl::cstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy, double& stp,
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

    double LBFGSSolverImpl::minimize(InputOutputArray argument) {
        CV_Assert(_Function.empty() == false);
        Mat x0 = argument.getMat();
        const int m = 10;
        const int DIM = argument.size().height;
        Mat sVector(DIM, m, CV_64F, 0.0);
        Mat yVector(DIM, m, CV_64F, 0.0);
        Mat alpha(m, 1, CV_64F, 0.0);
        Mat grad(DIM, 1, CV_64F, 0.0), q(DIM, 1, CV_64F, 0.0), grad_old(DIM, 1, CV_64F, 0.0), s(DIM, 1, CV_64F, 0.0), y(DIM, 1, CV_64F, 0.0);
        _Function->getGradient(x0.ptr<double>(), grad.ptr<double>());
        Mat x_old;
        x0.copyTo(x_old);
        double H0k = 1;
        for (int iter = 0; iter < _termcrit.maxCount; ++iter) {
            const double relativeEpsilon = _termcrit.epsilon * std::max(1.0, cv::norm(x0, NORM_L2));

            if (cv::norm(grad, NORM_L2) < relativeEpsilon)
                break;

            grad.copyTo(q);
            const int k = std::min(m, iter);

            for (int j = k - 1; j >= 0; --j) {
                const double rho = 1.0 / sVector.col(j).dot(yVector.col(j));
                alpha.at<double>(j) = rho * sVector.col(j).dot(q);
                q = q - alpha.at<double>(j) * yVector.col(j);
            }

            q = H0k * q;

            for (int j = 0; j < k; ++j) {
                const double rho = 1.0 / sVector.col(j).dot(yVector.col(j));
                const double beta = rho * yVector.col(j).dot(q);
                q = q + sVector.col(j) * (alpha.at<double>(j) - beta);
            }

            double descent = -grad.dot(q);
            double alpha_init =  1.0 / cv::norm(grad, NORM_L2);
            if (descent > -0.0001 * relativeEpsilon) {
                q = -1 * grad;
                iter = 0;
                alpha_init = 1.0;
            }

            const double rate = linesearch(x0, -q, alpha_init);

            x0 = x0 - rate * q;

            grad.copyTo(grad_old);
            _Function->getGradient(x0.ptr<double>(), grad.ptr<double>());

            s = x0 - x_old;
            y = grad - grad_old;

            if (iter < m) {
                sVector.col(iter) = s;
                yVector.col(iter) = y;
            } else {
                for (int i = 0; i < m - 1; ++i)
                    sVector.col(i) = sVector.col(m - i - 1);
                sVector.col(m - 1) = s;
                for (int i = 0; i < m - 1; ++i)
                    yVector.col(i) = yVector.col(m - i - 1);
                yVector.col(m - 1) = y;
            }

            H0k = y.dot(s) / y.dot(y);

            x0.copyTo(x_old);
        }
        return _Function->calc(x0.ptr<double>());
    }

    Ptr<LBFGSSolver> LBFGSSolver::create(const Ptr<MinProblemSolver::Function>& f, TermCriteria termcrit) {
        Ptr<LBFGSSolver> lbfgs = makePtr<LBFGSSolverImpl>();
        lbfgs->setFunction(f);
        lbfgs->setTermCriteria(termcrit);
        return lbfgs;
    }
};
