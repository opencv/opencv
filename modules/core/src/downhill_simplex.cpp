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

/*#define dprintf(x) printf x
#define print_matrix(x) print(x)*/

#define dprintf(x)
#define print_matrix(x)

/*

****Error Message********************************************************************************************************************

Downhill Simplex method in OpenCV dev 3.0.0 getting this error:

OpenCV Error: Assertion failed (dims <= 2 && data && (unsigned)i0 < (unsigned)(s ize.p[0] * size.p[1])
&& elemSize() == (((((DataType<_Tp>::type) & ((512 - 1) << 3)) >> 3) + 1) << ((((sizeof(size_t)/4+1)16384|0x3a50)
>> ((DataType<_Tp>::typ e) & ((1 << 3) - 1))2) & 3))) in Mat::at,
file C:\builds\master_PackSlave-w in32-vc12-shared\opencv\modules\core\include\opencv2/core/mat.inl.hpp, line 893

****Problem and Possible Fix*********************************************************************************************************

DownhillSolverImpl::innerDownhillSimplex something looks broken here:
Mat_<double> coord_sum(1,ndim,0.0),buf(1,ndim,0.0),y(1,ndim,0.0);
nfunk = 0;
for(i=0;i<ndim+1;++i)
{
y(i) = f->calc(p[i]);
}

y has only ndim elements, while the loop goes over ndim+1

Edited the following for possible fix:

Replaced y(1,ndim,0.0) ------> y(1,ndim+1,0.0)

***********************************************************************************************************************************

The code below was used in tesing the source code.
Created by @SareeAlnaghy

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <opencv2\optim\optim.hpp>

using namespace std;
using namespace cv;

void test(Ptr<optim::DownhillSolver> MinProblemSolver, Ptr<optim::MinProblemSolver::Function> ptr_F, Mat &P, Mat &step)
{
try{

MinProblemSolver->setFunction(ptr_F);
MinProblemSolver->setInitStep(step);
double res = MinProblemSolver->minimize(P);

cout << "res " << res << endl;
}
catch (exception e)
{
cerr << "Error:: " << e.what() << endl;
}
}

int main()
{

class DistanceToLines :public optim::MinProblemSolver::Function {
public:
double calc(const double* x)const{

return x[0] * x[0] + x[1] * x[1];

}
};

Mat P = (Mat_<double>(1, 2) << 1.0, 1.0);
Mat step = (Mat_<double>(2, 1) << -0.5, 0.5);

Ptr<optim::MinProblemSolver::Function> ptr_F(new DistanceToLines());
Ptr<optim::DownhillSolver> MinProblemSolver = optim::createDownhillSolver();

test(MinProblemSolver, ptr_F, P, step);

system("pause");
return 0;
}

****Suggesttion for imporving Simplex implentation***************************************************************************************

Currently the downhilll simplex method outputs the function value that is minimized. It should also return the coordinate points where the
function is minimized. This is very useful in many applications such as using back projection methods to find a point of intersection of
multiple lines in three dimensions as not all lines intersect in three dimensions.

*/

namespace cv
{

class DownhillSolverImpl : public DownhillSolver
{
public:
    DownhillSolverImpl()
    {
        _Function=Ptr<Function>();
        _step=Mat_<double>();
    }

    void getInitStep(OutputArray step) const { _step.copyTo(step); }
    void setInitStep(InputArray step)
    {
        // set dimensionality and make a deep copy of step
        Mat m = step.getMat();
        dprintf(("m.cols=%d\nm.rows=%d\n", m.cols, m.rows));
        CV_Assert( std::min(m.cols, m.rows) == 1 && m.type() == CV_64FC1 );
        if( m.rows == 1 )
            m.copyTo(_step);
        else
            transpose(m, _step);
    }

    Ptr<MinProblemSolver::Function> getFunction() const { return _Function; }

    void setFunction(const Ptr<Function>& f) { _Function=f; }

    TermCriteria getTermCriteria() const { return _termcrit; }

    void setTermCriteria( const TermCriteria& termcrit )
    {
        CV_Assert( termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) &&
                   termcrit.epsilon > 0 &&
                   termcrit.maxCount > 0 );
        _termcrit=termcrit;
    }

    double minimize( InputOutputArray x_ )
    {
        dprintf(("hi from minimize\n"));
        CV_Assert( !_Function.empty() );
        dprintf(("termcrit:\n\ttype: %d\n\tmaxCount: %d\n\tEPS: %g\n",_termcrit.type,_termcrit.maxCount,_termcrit.epsilon));
        dprintf(("step\n"));
        print_matrix(_step);

        Mat x = x_.getMat();
        Mat_<double> simplex;

        createInitialSimplex(x, simplex, _step);
        int count = 0;
        double res = innerDownhillSimplex(simplex,_termcrit.epsilon, _termcrit.epsilon,
                                          count, _Function, _termcrit.maxCount);
        dprintf(("%d iterations done\n",count));

        if( !x.empty() )
        {
            Mat simplex_0m(x.rows, x.cols, CV_64F, simplex.ptr<double>());
            simplex_0m.convertTo(x, x.type());
        }
        else
        {
            int x_type = x_.fixedType() ? x_.type() : CV_64F;
            simplex.row(0).convertTo(x_, x_type);
        }
        return res;
    }
protected:
    Ptr<MinProblemSolver::Function> _Function;
    TermCriteria _termcrit;
    Mat _step;

    inline void updateCoordSum(const Mat_<double>& p, Mat_<double>& coord_sum)
    {
        int i, j, m = p.rows, n = p.cols;
        double* coord_sum_ = coord_sum.ptr<double>();
        CV_Assert( coord_sum.cols == n && coord_sum.rows == 1 );

        for( j = 0; j < n; j++ )
            coord_sum_[j] = 0.;

        for( i = 0; i < m; i++ )
        {
            const double* p_i = p.ptr<double>(i);
            for( j = 0; j < n; j++ )
                coord_sum_[j] += p_i[j];
        }
    }

    inline void createInitialSimplex( const Mat& x0, Mat_<double>& simplex, Mat& step )
    {
        int i, j, ndim = step.cols;
        Mat x = x0;
        if( x0.empty() )
            x = Mat::zeros(1, ndim, CV_64F);
        CV_Assert( (x.cols == 1 && x.rows == ndim) || (x.cols == ndim && x.rows == 1) );
        CV_Assert( x.type() == CV_32F || x.type() == CV_64F );

        simplex.create(ndim + 1, ndim);
        Mat simplex_0m(x.rows, x.cols, CV_64F, simplex.ptr<double>());

        x.convertTo(simplex_0m, CV_64F);
        double* simplex_0 = simplex.ptr<double>();
        const double* step_ = step.ptr<double>();
        for( i = 1; i <= ndim; i++ )
        {
            double* simplex_i = simplex.ptr<double>(i);
            for( j = 0; j < ndim; j++ )
                simplex_i[j] = simplex_0[j];
            simplex_i[i-1] += 0.5*step_[i-1];
        }
        for( j = 0; j < ndim; j++ )
            simplex_0[j] -= 0.5*step_[j];

        dprintf(("this is simplex\n"));
        print_matrix(simplex);
    }

    /*
     Performs the actual minimization of MinProblemSolver::Function f (after the initialization was done)

     The matrix p[ndim+1][1..ndim] represents ndim+1 vertices that
     form a simplex - each row is an ndim vector.
     On output, nfunk gives the number of function evaluations taken.
    */
    double innerDownhillSimplex( Mat_<double>& p,double MinRange,double MinError, int& nfunk,
                                 const Ptr<MinProblemSolver::Function>& f, int nmax )
    {
        int i, j, ndim = p.cols;
        Mat_<double> coord_sum(1, ndim), buf(1, ndim), y(1, ndim+1);
        double* y_ = y.ptr<double>();

        nfunk = 0;

        for( i = 0; i <= ndim; i++ )
            y_[i] = f->calc(p[i]);

        nfunk = ndim+1;
        updateCoordSum(p, coord_sum);

        for (;;)
        {
            /*  find highest (worst), next-to-worst, and lowest
             (best) points by going through all of them. */
            int ilo = 0, ihi, inhi;
            if( y_[0] > y_[1] )
            {
                ihi = 0; inhi = 1;
            }
            else
            {
                ihi = 1; inhi = 0;
            }
            for( i = 0; i <= ndim; i++ )
            {
                double yval = y_[i];
                if (yval <= y_[ilo])
                    ilo = i;
                if (yval > y_[ihi])
                {
                    inhi = ihi;
                    ihi = i;
                }
                else if (yval > y_[inhi] && i != ihi)
                    inhi = i;
            }
            CV_Assert( ilo != ihi && ilo != inhi && ihi != inhi );
            dprintf(("this is y on iteration %d:\n",nfunk));
            print_matrix(y);

            /* check stop criterion */
            double error = fabs(y_[ihi] - y_[ilo]);
            double range = 0;
            for( j = 0; j < ndim; j++ )
            {
                double minval, maxval;
                minval = maxval = p(0, j);
                for( i = 1; i <= ndim; i++ )
                {
                    double pval = p(i, j);
                    minval = std::min(minval, pval);
                    maxval = std::max(maxval, pval);
                }
                range = std::max(range, fabs(maxval - minval));
            }

            if( range <= MinRange || error <= MinError || nfunk >= nmax )
            {
                /* Put best point and value in first slot. */
                std::swap(y(0), y(ilo));
                for( j = 0; j < ndim; j++ )
                {
                    std::swap(p(0, j), p(ilo, j));
                }
                break;
            }
            nfunk += 2;

            double ylo = y_[ilo], ynhi = y_[inhi];
            /* Begin a new iteration. First, reflect the worst point about the centroid of others */
            double ytry = tryNewPoint(p, y, coord_sum, f, ihi, -1.0, buf);
            if( ytry <= ylo )
            {
                /* If that's better than the best point, go twice as far in that direction */
                ytry = tryNewPoint(p, y, coord_sum, f, ihi, 2.0, buf);
            }
            else if( ytry >= ynhi )
            {
                /* The new point is worse than the second-highest,
                   do not go so far in that direction */
                double ysave = y(ihi);
                ytry = tryNewPoint(p, y, coord_sum, f, ihi, 0.5, buf);
                if (ytry >= ysave)
                {
                    /* Can't seem to improve things. Contract the simplex to good point
                       in hope to find a simplex landscape. */
                    for( i = 0; i <= ndim; i++ )
                    {
                        if (i != ilo)
                        {
                            for( j = 0; j < ndim; j++ )
                                p(i,j) = 0.5*(p(i,j) + p(ilo,j));
                            y(i)=f->calc(p.ptr<double>(i));
                        }
                    }
                    nfunk += ndim;
                    updateCoordSum(p, coord_sum);
                }
            }
            else --(nfunk); /* correct nfunk */
            dprintf(("this is simplex on iteration %d\n",nfunk));
            print_matrix(p);
        } /* go to next iteration. */
        return y(0);
    }

    inline double tryNewPoint(Mat_<double>& p, Mat_<double>& y, Mat_<double>& coord_sum,
                              const Ptr<MinProblemSolver::Function>& f, int ihi,
                              double fac, Mat_<double>& ptry)
    {
        int j, ndim = p.cols;

        double fac1 = (1.0 - fac)/ndim;
        double fac2 = fac1 - fac;
        double* p_ihi = p.ptr<double>(ihi);
        double* ptry_ = ptry.ptr<double>();
        double* coord_sum_ = coord_sum.ptr<double>();

        for( j = 0; j < ndim; j++ )
            ptry_[j] = coord_sum_[j]*fac1 - p_ihi[j]*fac2;

        double ytry = f->calc(ptry_);
        if (ytry < y(ihi))
        {
            y(ihi) = ytry;
            for( j = 0; j < ndim; j++ )
                p_ihi[j] = ptry_[j];
            updateCoordSum(p, coord_sum);
        }

        return ytry;
    }
};


// both minRange & minError are specified by termcrit.epsilon;
// In addition, user may specify the number of iterations that the algorithm does.
Ptr<DownhillSolver> DownhillSolver::create( const Ptr<MinProblemSolver::Function>& f,
                                            InputArray initStep, TermCriteria termcrit )
{
    Ptr<DownhillSolver> DS = makePtr<DownhillSolverImpl>();
    DS->setFunction(f);
    DS->setInitStep(initStep);
    DS->setTermCriteria(termcrit);
    return DS;
}

}
