#include "precomp.hpp"
#include "debug.hpp"

namespace cv{namespace optim{

    class ConjGradSolverImpl : public ConjGradSolver
    {
    public:
        Ptr<Function> getFunction() const;
        void setFunction(const Ptr<Function>& f);
        TermCriteria getTermCriteria() const;
        ConjGradSolverImpl();
        void setTermCriteria(const TermCriteria& termcrit);
        double minimize(InputOutputArray x);
    protected:
        Ptr<Solver::Function> _Function;
        TermCriteria _termcrit;
        Mat_<double> d,r,buf_x,r_old;
    private:
    };

    double ConjGradSolverImpl::minimize(InputOutputArray x){
        CV_Assert(_Function.empty()==false);
        dprintf(("termcrit:\n\ttype: %d\n\tmaxCount: %d\n\tEPS: %g\n",_termcrit.type,_termcrit.maxCount,_termcrit.epsilon));

        Mat x_mat=x.getMat();
        CV_Assert(MIN(x_mat.rows,x_mat.cols)==1);
        int ndim=MAX(x_mat.rows,x_mat.cols);
        CV_Assert(x_mat.type()==CV_64FC1);

        d.create(1,ndim);
        r.create(1,ndim);
        r_old.create(1,ndim);

        Mat_<double> proxy_x;
        if(x_mat.rows>1){
            buf_x.create(1,ndim);
            Mat_<double> proxy(ndim,1,(double*)buf_x.data);
            x_mat.copyTo(proxy);
            proxy_x=buf_x;
        }else{
            proxy_x=x_mat;
        }

        //here everything goes. check that everything is setted properly

        if(x_mat.rows>1){
            Mat(ndim, 1, CV_64F, (double*)proxy_x.data).copyTo(x);
        }
        return 0.0;
    }
    ConjGradSolverImpl::ConjGradSolverImpl(){
        _Function=Ptr<Function>();
    }
    Ptr<Solver::Function> ConjGradSolverImpl::getFunction()const{
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
    Ptr<ConjGradSolver> createConjGradSolver(const Ptr<Solver::Function>& f, TermCriteria termcrit){
        ConjGradSolver *CG=new ConjGradSolverImpl();
        CG->setFunction(f);
        CG->setTermCriteria(termcrit);
        return Ptr<ConjGradSolver>(CG);
    }
}}
                               
