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
#include "debug.hpp"
#include "opencv2/core/core_c.h"

/*

****Error Message********************************************************************************************************************

Downhill Simplex method in OpenCV dev 3.0.0 getting this error:

OpenCV Error: Assertion failed (dims <= 2 && data && (unsigned)i0 < (unsigned)(s ize.p[0] * size.p[1])
&& elemSize() == (((((DataType<_Tp>::type) & ((512 - 1) << 3)) >> 3) + 1) << ((((sizeof(size_t)/4+1)16384|0x3a50)
>> ((DataType<_Tp>::typ e) & ((1 << 3) - 1))2) & 3))) in cv::Mat::at,
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

void test(Ptr<optim::DownhillSolver> solver, Ptr<optim::Solver::Function> ptr_F, Mat &P, Mat &step)
{
try{

solver->setFunction(ptr_F);
solver->setInitStep(step);
double res = solver->minimize(P);

cout << "res " << res << endl;
}
catch (exception e)
{
cerr << "Error:: " << e.what() << endl;
}
}

int main()
{

class DistanceToLines :public optim::Solver::Function {
public:
double calc(const double* x)const{

return x[0] * x[0] + x[1] * x[1];

}
};

Mat P = (Mat_<double>(1, 2) << 1.0, 1.0);
Mat step = (Mat_<double>(2, 1) << -0.5, 0.5);

Ptr<optim::Solver::Function> ptr_F(new DistanceToLines());
Ptr<optim::DownhillSolver> solver = optim::createDownhillSolver();

test(solver, ptr_F, P, step);

system("pause");
return 0;
}

****Suggesttion for imporving Simplex implentation***************************************************************************************

Currently the downhilll simplex method outputs the function value that is minimized. It should also return the coordinate points where the
function is minimized. This is very useful in many applications such as using back projection methods to find a point of intersection of
multiple lines in three dimensions as not all lines intersect in three dimensions.

*/





namespace cv{namespace optim{

    class DownhillSolverImpl : public DownhillSolver
    {
    public:
        void getInitStep(OutputArray step) const;
        void setInitStep(InputArray step);
        Ptr<Function> getFunction() const;
        void setFunction(const Ptr<Function>& f);
        TermCriteria getTermCriteria() const;
        DownhillSolverImpl();
        void setTermCriteria(const TermCriteria& termcrit);
        double minimize(InputOutputArray x);
    protected:
        Ptr<Solver::Function> _Function;
        TermCriteria _termcrit;
        Mat _step;
        Mat_<double> buf_x;

    private:
        inline void createInitialSimplex(Mat_<double>& simplex,Mat& step);
        inline double innerDownhillSimplex(cv::Mat_<double>& p,double MinRange,double MinError,int& nfunk,
                const Ptr<Solver::Function>& f,int nmax);
        inline double tryNewPoint(Mat_<double>& p,Mat_<double>& y,Mat_<double>& coord_sum,const Ptr<Solver::Function>& f,int ihi,
                double fac,Mat_<double>& ptry);
    };

    double DownhillSolverImpl::tryNewPoint(
        Mat_<double>& p,
        Mat_<double>& y,
        Mat_<double>&  coord_sum,
        const Ptr<Solver::Function>& f,
        int      ihi,
        double   fac,
        Mat_<double>& ptry
        )
    {
        int ndim=p.cols;
        int j;
        double fac1,fac2,ytry;

        fac1=(1.0-fac)/ndim;
        fac2=fac1-fac;
        for (j=0;j<ndim;j++)
        {
            ptry(j)=coord_sum(j)*fac1-p(ihi,j)*fac2;
        }
        ytry=f->calc((double*)ptry.data);
        if (ytry < y(ihi))
        {
            y(ihi)=ytry;
            for (j=0;j<ndim;j++)
            {
                coord_sum(j) += ptry(j)-p(ihi,j);
                p(ihi,j)=ptry(j);
            }
        }

        return ytry;
    }

    /*
    Performs the actual minimization of Solver::Function f (after the initialization was done)

    The matrix p[ndim+1][1..ndim] represents ndim+1 vertices that
    form a simplex - each row is an ndim vector.
    On output, nfunk gives the number of function evaluations taken.
    */
    double DownhillSolverImpl::innerDownhillSimplex(
        cv::Mat_<double>&   p,
        double     MinRange,
        double     MinError,
        int&       nfunk,
        const Ptr<Solver::Function>& f,
        int nmax
        )
    {
        int ndim=p.cols;
        double res;
        int i,ihi,ilo,inhi,j,mpts=ndim+1;
        double error, range,ysave,ytry;
        Mat_<double> coord_sum(1,ndim,0.0),buf(1,ndim,0.0),y(1,ndim+1,0.0);

        nfunk = 0;

        for(i=0;i<ndim+1;++i)
        {
            y(i) = f->calc(p[i]);
        }

        nfunk = ndim+1;

        reduce(p,coord_sum,0,CV_REDUCE_SUM);

        for (;;)
        {
            ilo=0;
            /*  find highest (worst), next-to-worst, and lowest
                (best) points by going through all of them. */
            ihi = y(0)>y(1) ? (inhi=1,0) : (inhi=0,1);
            for (i=0;i<mpts;i++)
            {
                if (y(i) <= y(ilo))
                    ilo=i;
                if (y(i) > y(ihi))
                {
                    inhi=ihi;
                    ihi=i;
                }
                else if (y(i) > y(inhi) && i != ihi)
                    inhi=i;
            }

            /* check stop criterion */
            error=fabs(y(ihi)-y(ilo));
            range=0;
            for(i=0;i<ndim;++i)
            {
                double min = p(0,i);
                double max = p(0,i);
                double d;
                for(j=1;j<=ndim;++j)
                {
                    if( min > p(j,i) ) min = p(j,i);
                    if( max < p(j,i) ) max = p(j,i);
                }
                d = fabs(max-min);
                if(range < d) range = d;
            }

            if(range <= MinRange || error <= MinError)
            { /* Put best point and value in first slot. */
                std::swap(y(0),y(ilo));
                for (i=0;i<ndim;i++)
                {
                    std::swap(p(0,i),p(ilo,i));
                }
                break;
            }

            if (nfunk >= nmax){
                dprintf(("nmax exceeded\n"));
                return y(ilo);
            }
            nfunk += 2;
            /*Begin a new iteration. First, reflect the worst point about the centroid of others */
            ytry = tryNewPoint(p,y,coord_sum,f,ihi,-1.0,buf);
            if (ytry <= y(ilo))
            { /*If that's better than the best point, go twice as far in that direction*/
                ytry = tryNewPoint(p,y,coord_sum,f,ihi,2.0,buf);
            }
            else if (ytry >= y(inhi))
            {   /* The new point is worse than the second-highest, but better
                  than the worst so do not go so far in that direction */
                ysave = y(ihi);
                ytry = tryNewPoint(p,y,coord_sum,f,ihi,0.5,buf);
                if (ytry >= ysave)
                { /* Can't seem to improve things. Contract the simplex to good point
               in hope to find a simplex landscape. */
                    for (i=0;i<mpts;i++)
                    {
                        if (i != ilo)
                        {
                            for (j=0;j<ndim;j++)
                            {
                                p(i,j) = coord_sum(j) = 0.5*(p(i,j)+p(ilo,j));
                            }
                            y(i)=f->calc((double*)coord_sum.data);
                        }
                    }
                    nfunk += ndim;
                    reduce(p,coord_sum,0,CV_REDUCE_SUM);
                }
            } else --(nfunk); /* correct nfunk */
            dprintf(("this is simplex on iteration %d\n",nfunk));
            print_matrix(p);
        } /* go to next iteration. */
        res = y(0);

        return res;
    }

    void DownhillSolverImpl::createInitialSimplex(Mat_<double>& simplex,Mat& step){
        for(int i=1;i<=step.cols;++i)
        {
            simplex.row(0).copyTo(simplex.row(i));
            simplex(i,i-1)+= 0.5*step.at<double>(0,i-1);
        }
        simplex.row(0) -= 0.5*step;

        dprintf(("this is simplex\n"));
        print_matrix(simplex);
    }

    double DownhillSolverImpl::minimize(InputOutputArray x){
        dprintf(("hi from minimize\n"));
        CV_Assert(_Function.empty()==false);
        dprintf(("termcrit:\n\ttype: %d\n\tmaxCount: %d\n\tEPS: %g\n",_termcrit.type,_termcrit.maxCount,_termcrit.epsilon));
        dprintf(("step\n"));
        print_matrix(_step);

        Mat x_mat=x.getMat();
        CV_Assert(MIN(x_mat.rows,x_mat.cols)==1);
        CV_Assert(MAX(x_mat.rows,x_mat.cols)==_step.cols);
        CV_Assert(x_mat.type()==CV_64FC1);

        Mat_<double> proxy_x;

        if(x_mat.rows>1){
            buf_x.create(1,_step.cols);
            Mat_<double> proxy(_step.cols,1,(double*)buf_x.data);
            x_mat.copyTo(proxy);
            proxy_x=buf_x;
        }else{
            proxy_x=x_mat;
        }

        int count=0;
        int ndim=_step.cols;
        Mat_<double> simplex=Mat_<double>(ndim+1,ndim,0.0);

        simplex.row(0).copyTo(proxy_x);
        createInitialSimplex(simplex,_step);
        double res = innerDownhillSimplex(
                simplex,_termcrit.epsilon, _termcrit.epsilon, count,_Function,_termcrit.maxCount);
        simplex.row(0).copyTo(proxy_x);

        dprintf(("%d iterations done\n",count));

        if(x_mat.rows>1){
            Mat(x_mat.rows, 1, CV_64F, (double*)proxy_x.data).copyTo(x);
        }
        return res;
    }
    DownhillSolverImpl::DownhillSolverImpl(){
        _Function=Ptr<Function>();
        _step=Mat_<double>();
    }
    Ptr<Solver::Function> DownhillSolverImpl::getFunction()const{
        return _Function;
    }
    void DownhillSolverImpl::setFunction(const Ptr<Function>& f){
        _Function=f;
    }
    TermCriteria DownhillSolverImpl::getTermCriteria()const{
        return _termcrit;
    }
    void DownhillSolverImpl::setTermCriteria(const TermCriteria& termcrit){
        CV_Assert(termcrit.type==(TermCriteria::MAX_ITER+TermCriteria::EPS) && termcrit.epsilon>0 && termcrit.maxCount>0);
        _termcrit=termcrit;
    }
    // both minRange & minError are specified by termcrit.epsilon; In addition, user may specify the number of iterations that the algorithm does.
    Ptr<DownhillSolver> createDownhillSolver(const Ptr<Solver::Function>& f, InputArray initStep, TermCriteria termcrit){
        DownhillSolver *DS=new DownhillSolverImpl();
        DS->setFunction(f);
        DS->setInitStep(initStep);
        DS->setTermCriteria(termcrit);
        return Ptr<DownhillSolver>(DS);
    }
    void DownhillSolverImpl::getInitStep(OutputArray step)const{
        _step.copyTo(step);
    }
    void DownhillSolverImpl::setInitStep(InputArray step){
        //set dimensionality and make a deep copy of step
        Mat m=step.getMat();
        dprintf(("m.cols=%d\nm.rows=%d\n",m.cols,m.rows));
        CV_Assert(MIN(m.cols,m.rows)==1 && m.type()==CV_64FC1);
        if(m.rows==1){
            m.copyTo(_step);
        }else{
            transpose(m,_step);
        }
    }
}}
