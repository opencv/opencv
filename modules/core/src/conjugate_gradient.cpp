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
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
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

#define dprintf(x)
#define print_matrix(x)

namespace cv
{
    double MinProblemSolver::Function::getGradientEps() const { return 1e-3; }
    void MinProblemSolver::Function::getGradient(const double* x, double* grad)
    {
        double eps = getGradientEps();
        int i, n = getDims();
        AutoBuffer<double> x_buf(n);
        double* x_ = x_buf.data();
        for( i = 0; i < n; i++ )
            x_[i] = x[i];
        for( i = 0; i < n; i++ )
        {
            x_[i] = x[i] + eps;
            double y1 = calc(x_);
            x_[i] = x[i] - eps;
            double y0 = calc(x_);
            grad[i] = (y1 - y0)/(2*eps);
            x_[i] = x[i];
        }
    }

#define SEC_METHOD_ITERATIONS 4
#define INITIAL_SEC_METHOD_SIGMA 0.1
    class ConjGradSolverImpl CV_FINAL : public ConjGradSolver
    {
    public:
        Ptr<Function> getFunction() const CV_OVERRIDE;
        void setFunction(const Ptr<Function>& f) CV_OVERRIDE;
        TermCriteria getTermCriteria() const CV_OVERRIDE;
        ConjGradSolverImpl();
        void setTermCriteria(const TermCriteria& termcrit) CV_OVERRIDE;
        double minimize(InputOutputArray x) CV_OVERRIDE;
    protected:
        Ptr<MinProblemSolver::Function> _Function;
        TermCriteria _termcrit;
        Mat_<double> d,r,buf_x,r_old;
        Mat_<double> minimizeOnTheLine_buf1,minimizeOnTheLine_buf2;
    private:
        static void minimizeOnTheLine(Ptr<MinProblemSolver::Function> _f,Mat_<double>& x,const Mat_<double>& d,Mat_<double>& buf1,Mat_<double>& buf2);
    };

    void ConjGradSolverImpl::minimizeOnTheLine(Ptr<MinProblemSolver::Function> _f,Mat_<double>& x,const Mat_<double>& d,Mat_<double>& buf1,
            Mat_<double>& buf2){
        double sigma=INITIAL_SEC_METHOD_SIGMA;
        buf1=0.0;
        buf2=0.0;

        dprintf(("before minimizeOnTheLine\n"));
        dprintf(("x:\n"));
        print_matrix(x);
        dprintf(("d:\n"));
        print_matrix(d);

        for(int i=0;i<SEC_METHOD_ITERATIONS;i++){
            _f->getGradient((double*)x.data,(double*)buf1.data);
            dprintf(("buf1:\n"));
            print_matrix(buf1);
            x=x+sigma*d;
            _f->getGradient((double*)x.data,(double*)buf2.data);
            dprintf(("buf2:\n"));
            print_matrix(buf2);
            double d1=buf1.dot(d), d2=buf2.dot(d);
            if((d1-d2)==0){
                break;
            }
            double alpha=-sigma*d1/(d2-d1);
            dprintf(("(buf2.dot(d)-buf1.dot(d))=%f\nalpha=%f\n",(buf2.dot(d)-buf1.dot(d)),alpha));
            x=x+(alpha-sigma)*d;
            sigma=-alpha;
        }

        dprintf(("after minimizeOnTheLine\n"));
        print_matrix(x);
    }

    double ConjGradSolverImpl::minimize(InputOutputArray x){
        CV_Assert(_Function.empty()==false);
        dprintf(("termcrit:\n\ttype: %d\n\tmaxCount: %d\n\tEPS: %g\n",_termcrit.type,_termcrit.maxCount,_termcrit.epsilon));

        Mat x_mat=x.getMat();
        CV_Assert(MIN(x_mat.rows,x_mat.cols)==1);
        int ndim=MAX(x_mat.rows,x_mat.cols);
        CV_Assert(x_mat.type()==CV_64FC1);

        if(d.cols!=ndim){
            d.create(1,ndim);
            r.create(1,ndim);
            r_old.create(1,ndim);
            minimizeOnTheLine_buf1.create(1,ndim);
            minimizeOnTheLine_buf2.create(1,ndim);
        }

        Mat_<double> proxy_x;
        if(x_mat.rows>1){
            buf_x.create(1,ndim);
            Mat_<double> proxy(ndim,1,buf_x.ptr<double>());
            x_mat.copyTo(proxy);
            proxy_x=buf_x;
        }else{
            proxy_x=x_mat;
        }
        _Function->getGradient(proxy_x.ptr<double>(),d.ptr<double>());
        d*=-1.0;
        d.copyTo(r);

        //here everything goes. check that everything is set properly
        dprintf(("proxy_x\n"));print_matrix(proxy_x);
        dprintf(("d first time\n"));print_matrix(d);
        dprintf(("r\n"));print_matrix(r);

        for(int count=0;count<_termcrit.maxCount;count++){
            minimizeOnTheLine(_Function,proxy_x,d,minimizeOnTheLine_buf1,minimizeOnTheLine_buf2);
            r.copyTo(r_old);
            _Function->getGradient(proxy_x.ptr<double>(),r.ptr<double>());
            r*=-1.0;
            double r_norm_sq=norm(r);
            if(_termcrit.type==(TermCriteria::MAX_ITER+TermCriteria::EPS) && r_norm_sq<_termcrit.epsilon){
                break;
            }
            r_norm_sq=r_norm_sq*r_norm_sq;
            double beta=MAX(0.0,(r_norm_sq-r.dot(r_old))/r_norm_sq);
            d=r+beta*d;
        }



        if(x_mat.rows>1){
            Mat(ndim, 1, CV_64F, proxy_x.ptr<double>()).copyTo(x);
        }
        return _Function->calc(proxy_x.ptr<double>());
    }

    ConjGradSolverImpl::ConjGradSolverImpl(){
        _Function=Ptr<Function>();
    }
    Ptr<MinProblemSolver::Function> ConjGradSolverImpl::getFunction()const{
        return _Function;
    }
    void ConjGradSolverImpl::setFunction(const Ptr<Function>& f){
        _Function=f;
    }
    TermCriteria ConjGradSolverImpl::getTermCriteria()const{
        return _termcrit;
    }
    void ConjGradSolverImpl::setTermCriteria(const TermCriteria& termcrit){
        CV_Assert((termcrit.type==(TermCriteria::MAX_ITER+TermCriteria::EPS) && termcrit.epsilon>0 && termcrit.maxCount>0) ||
                ((termcrit.type==TermCriteria::MAX_ITER) && termcrit.maxCount>0));
        _termcrit=termcrit;
    }
    // both minRange & minError are specified by termcrit.epsilon; In addition, user may specify the number of iterations that the algorithm does.
    Ptr<ConjGradSolver> ConjGradSolver::create(const Ptr<MinProblemSolver::Function>& f, TermCriteria termcrit){
        Ptr<ConjGradSolver> CG = makePtr<ConjGradSolverImpl>();
        CG->setFunction(f);
        CG->setTermCriteria(termcrit);
        return CG;
    }
}
