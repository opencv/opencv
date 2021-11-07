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

/*
   This is a translation to C++ from the Matlab's LMSolve package by Miroslav Balda.
   Here is the original copyright:
   ============================================================================

   Copyright (c) 2007, Miroslav Balda
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in
         the documentation and/or other materials provided with the distribution

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

namespace cv {

static void subMatrix(const Mat& src, Mat& dst,
                      const Mat& mask)
{
    CV_Assert(src.type() == CV_64F && dst.type() == CV_64F);
    int m = src.rows, n = src.cols;
    int i1 = 0, j1 = 0;
    for(int i = 0; i < m; i++)
    {
        if(mask.at<uchar>(i))
        {
            const double* srcptr = src.ptr<double>(i);
            double* dstptr = dst.ptr<double>(i1++);

            for(int j = j1 = 0; j < n; j++)
            {
                if(n < m || mask.at<uchar>(j))
                    dstptr[j1++] = srcptr[j];
            }
        }
    }
}

class LMSolverImpl CV_FINAL : public LMSolver
{
public:
    LMSolverImpl(const Ptr<LMSolver::Callback>& _cb, int _maxIters, double _eps = FLT_EPSILON)
        : cb(_cb), eps(_eps), maxIters(_maxIters)
    {
    }

    int run(InputOutputArray param0) const CV_OVERRIDE
    {
        return LMSolver::run(param0, noArray(), 0,
            TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, maxIters, eps), DECOMP_SVD,
            [&](Mat& param, Mat* err, Mat* J)->bool
            {
                return cb->compute(param, err ? _OutputArray(*err) : _OutputArray(),
                                   J ? _OutputArray(*J) : _OutputArray());
            });
    }

    void setMaxIters(int iters) CV_OVERRIDE { CV_Assert(iters > 0); maxIters = iters; }
    int getMaxIters() const CV_OVERRIDE { return maxIters; }

    Ptr<LMSolver::Callback> cb;
    double eps;
    int maxIters;
};


Ptr<LMSolver> LMSolver::create(const Ptr<LMSolver::Callback>& cb, int maxIters, double eps)
{
    return makePtr<LMSolverImpl>(cb, maxIters, eps);
}

static int LMSolver_run(InputOutputArray _param0, InputArray _mask,
                        int nerrs, const TermCriteria& termcrit,
                        int solveMethod, bool LtoR,
                        std::function<bool (Mat&, Mat*, Mat*)>* cb,
                        std::function<bool (Mat&, Mat*, Mat*, double*)>* cb_alt)
{
    //DEBUG
    //static int ctr = 0;
    //ctr++;
    //std::cout << "old ctr: " << ctr << std::endl;

    int lambdaLg10 = -3;
    Mat mask = _mask.getMat();
    Mat param0 = _param0.getMat();
    Mat x, xd, r, rd, J, A, Ap, v, temp_d, d, Am, vm, dm;
    int p0type = param0.type();
    int maxIters = termcrit.type & TermCriteria::COUNT ? termcrit.maxCount : 1000;
    double epsx = termcrit.type & TermCriteria::EPS ? termcrit.epsilon : 0, epsf = epsx;

    CV_Assert( (param0.cols == 1 || param0.rows == 1) && (p0type == CV_32F || p0type == CV_64F));
    CV_Assert( cb || cb_alt );

    int lx = param0.rows + param0.cols - 1;
    param0.convertTo(x, CV_64F);
    d.create(lx, 1, CV_64F);

    CV_Assert(!mask.data ||
              (mask.depth() == CV_8U &&
               (mask.cols == 1 || mask.rows == 1) &&
               (mask.rows + mask.cols - 1 == lx)));
    int lxm = mask.data ? countNonZero(mask) : lx;
    if (lxm < lx) {
        Am.create(lxm, lxm, CV_64F);
        vm.create(lxm, 1, CV_64F);
    }

    if( x.cols != 1 )
        transpose(x, x);

    A.create(lx, lx, CV_64F);
    v.create(lx, 1, CV_64F);

    if (nerrs > 0) {
        J.create(nerrs, lx, CV_64F);
        r.create(nerrs, 1, CV_64F);
        rd.create(nerrs, 1, CV_64F);
    }

    double S = 0;
    int nfJ = 1;
    if (cb_alt) {
        if( !(*cb_alt)(x, &v, &A, &S) )
            return -1;
        completeSymm(A, LtoR);
    } else {
        if( !(*cb)(x, &r, &J) )
            return -1;
        S = norm(r, NORM_L2SQR);
        mulTransposed(J, A, true);
        gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
    }

    int i, iter = 0;

    for( ;; )
    {
        CV_Assert( A.type() == CV_64F && A.rows == lx );
        A.copyTo(Ap);
        double lambda = exp(lambdaLg10*log(10.));
        for( i = 0; i < lx; i++ )
            Ap.at<double>(i, i) *= (1 + lambda);
        if (lxm < lx) {
            // remove masked-out rows & cols from JtJ and JtErr
            subMatrix(Ap, Am, mask);
            subMatrix(v, vm, mask);
            solve(Am, vm, dm, solveMethod);
            int j = 0;
            // 'unpack' the param delta
            for(i = j = 0; i < lx; i++)
                d.at<double>(i) = mask.at<uchar>(i) != 0 ? dm.at<double>(j++) : 0.;
        } else {
            solve(Ap, v, d, solveMethod);
        }
        subtract(x, d, xd);

        //DEBUG
        //if (ctr == 1003)
        //{
        //    std::cout << "S: " << S << std::endl;
        //    std::cout << "Ap.diag(): " << Ap.diag() << std::endl;
        //    std::cout << "lambda: " << lambda << std::endl;
        //    std::cout << "v: " << v << std::endl;
        //    std::cout << "d: " << d << std::endl;
        //}

        double Sd = 0.;

        if (cb_alt) {
            if( !(*cb_alt)(xd, 0, 0, &Sd) )
                return -1;
        } else {
            if( !(*cb)(xd, &rd, 0) )
                return -1;
            Sd = norm(rd, NORM_L2SQR);
        }

        nfJ++;
        if( Sd < S )
        {
            nfJ++;
            S = Sd;
            lambdaLg10 = MAX(lambdaLg10-1, -16);
            iter++;
            std::swap(x, xd);
            if (cb_alt) {
                v.setZero();
                A.setZero();
                Sd = 0.;
                if( !(*cb_alt)(x, &v, &A, &Sd) )
                    return -1;
                completeSymm(A, LtoR);
            } else {
                r.setZero();
                J.setZero();
                if( !(*cb)(x, &r, &J) )
                    return -1;
                mulTransposed(J, A, true);
                gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
            }
        } else {
            iter += lambdaLg10 == 16;
            lambdaLg10 = MIN(lambdaLg10+1, 16);
        }

        bool proceed = iter < maxIters && norm(d, NORM_INF) >= epsx && S >= epsf*epsf;

        //DEBUG
        //printf("%c %d %d, err=%g, lg10(lambda)=%d\n",
        //       (proceed ? ' ' : '*'), iter, nfJ, S, lambdaLg10);
        /*
        if(lxm < lx)
        {
            printf("lambda=%g. delta:", lambda);
            int j;
            for(i = j = 0; i < lx; i++) {
                double delta = d.at<double>(i);
                j += delta != 0;
                if(j < 10)
                    printf(" %.2g", delta);
            }
            printf("\n");
            printf("%c %d %d, err=%g, param[0]=%g, d[0]=%g, lg10(lambda)=%d\n",
                   (proceed ? ' ' : '*'), iter, nfJ, S, x.at<double>(0), d.at<double>(0), lambdaLg10);
        }
        */
        if(!proceed)
            break;
    }

    if( param0.size() != x.size() )
        transpose(x, x);

    x.convertTo(param0, p0type);
    if( iter == maxIters )
        iter = -iter;

    return iter;
}

int LMSolver::run(InputOutputArray param, InputArray mask, int nerrs,
                  const TermCriteria& termcrit, int solveMethod,
                  LMSolver::LongCallback cb)
{
    return LMSolver_run(param, mask, nerrs, termcrit, solveMethod, true, &cb, 0);
}

int LMSolver::runAlt(InputOutputArray param, InputArray mask,
                     const TermCriteria& termcrit, int solveMethod, bool LtoR,
                     LMSolver::AltCallback cb_alt)
{
    return LMSolver_run(param, mask, 0, termcrit, solveMethod, LtoR, 0, &cb_alt);
}

}
