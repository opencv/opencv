#include <precomp.hpp>
#define ALEX_DEBUG
#include <debug.hpp>

namespace cv{namespace optim{
    static double _cvDownhillSimplex_( 
        double*    x,
        double*    step,
        int        ndim, 
        double     MinRange,
        double     MinError,
        int        *nfunk,
        const Ptr<Solver::Function>& f,
        int nmax
        );

    double DownhillSolver::minimize(InputOutputArray x){
        dprintf(("hi from minimize\n"));
        if(_F.empty()){
            dprintf(("F is empty\n"));
        }else{
            dprintf(("F is not empty\n"));
        }
        dprintf(("termcrit:\n\ttype: %d\n\tmaxCount: %d\n\tEPS: %g\n",_termcrit.type,_termcrit.maxCount,_termcrit.epsilon));
        dprintf(("step\n"));
        print_matrix(_step);

        int count;
        Mat x_mat=x.getMat();
        CV_Assert(MIN(x_mat.rows,x_mat.cols)==1);
        CV_Assert(MAX(x_mat.rows,x_mat.cols)==_step.cols);
        CV_Assert(x_mat.type()==CV_64FC1);

        Mat proxy_x;
        double *raw_x;

        if(x_mat.rows>1){
            proxy_x=x_mat.t();
            raw_x=(double*)proxy_x.data;
        }else{
            raw_x=(double*)x_mat.data;
        }

        double res=_cvDownhillSimplex_(raw_x,(double*)_step.data,_step.cols,_termcrit.epsilon,_termcrit.epsilon,&count,_F,_termcrit.maxCount);
        dprintf(("%d iterations done\n",count));

        if(x_mat.rows>1){
            for(Mat_<double>::iterator it1=x_mat.begin<double>(),it2=proxy_x.begin<double>();it1<x_mat.end<double>();it1++,it2++){
                *it1=*it2;
            }
        }
        return res;
    }
    DownhillSolver::DownhillSolver(){
        _F=Ptr<Function>();
        _termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5000,0.000001);
        _step=Mat_<double>();
    }
    Ptr<Solver::Function> DownhillSolver::getFunction()const{
        return _F;
    }
    void DownhillSolver::setFunction(const Ptr<Function>& f){
        _F=f;
    }
    TermCriteria DownhillSolver::getTermCriteria()const{
        return _termcrit;
    }
    void DownhillSolver::setTermCriteria(const TermCriteria& termcrit){
        if(termcrit.type==(TermCriteria::MAX_ITER+TermCriteria::EPS) && termcrit.epsilon>0 && termcrit.maxCount>0){
            _termcrit=termcrit;
        }
    }
    // both minRange & minError are specified by termcrit.epsilon; In addition, user may specify the number of iterations that the algorithm does.
    Ptr<DownhillSolver> createDownhillSolver(const Ptr<Solver::Function>& f, InputArray initStep, TermCriteria termcrit){
        DownhillSolver *DS=new DownhillSolver();
        DS->setFunction(f);
        cv::Mat step=initStep.getMat();
        DS->setInitStep(initStep);
        DS->setTermCriteria(termcrit);
        return Ptr<DownhillSolver>(DS);
    }
    void DownhillSolver::getInitStep(OutputArray step)const{
        step.create(1,_step.cols,CV_64FC1);
        _step.copyTo(step);
    }
    void DownhillSolver::setInitStep(InputArray step){
        //set dimensionality and make a deep copy of step
        Mat m=step.getMat();
        if(MIN(m.cols,m.rows)==1 && m.type()==CV_64FC1){
            int ndim=MAX(m.cols,m.rows);
            if(ndim!=_step.cols){
                _step=Mat_<double>(1,ndim);
            }
            for(Mat_<double>::iterator it1=m.begin<double>(),it2=_step.begin<double>();it1<m.end<double>();it1++,it2++){
                *it2=*it1;
            }
        }
    }

    /*============== OPTIMIZATION FUNCTIONS ===================== */
    static void _cvDownhillSimplexInit(double** p, int n, double* dp);
    /*    
        Extrapolates by a factor fac through the face of the simplex across from the high point, tries
        it, and replaces the high point if the new point is better. 
    */
    static double _cvDHSTry(
        double** p,
        double* y,
        double* psum,
        int ndim,
        const Ptr<Solver::Function>& f,
        int ihi,
        double fac
        );
    static double _cvDownhillSimplex(
        double**   p, 
        int        ndim, 
        double     MinRange,
        double     MinError,
        int        *nfunk,
        const Ptr<Solver::Function>& f,
        int nmax
        );
    //A small number
#define TINY 1.0e-10 
    //Maximum allowed number of function evaluations. 
//#define NMAX 5000 
#define BUF_SIZE 32
    static double _cvDHSTry(    
        double** p, 
        double*  y, 
        double*  psum, 
        int      ndim,
        const Ptr<Solver::Function>& f,
        int      ihi, 
        double   fac)
    {
        int j;
        double  buf[BUF_SIZE];
        double fac1,fac2,ytry,*ptry;
        
        if(ndim>BUF_SIZE)
        {
    ////        ptry=(double*)cvAlloc(ndim*sizeof(double));
        }
        else
        {
            ptry=buf;
        }
        
        fac1=(1.0-fac)/ndim;
        fac2=fac1-fac;
        for (j=0;j<ndim;j++) 
        {
            ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
        }
        ytry=f->calc(ptry); /* Evaluate the function at the trial point. */
        if (ytry < y[ihi]) 
        { /* If it's better than the highest, then replace the highest. */
            y[ihi]=ytry;
            for (j=0;j<ndim;j++) 
            {
                psum[j] += ptry[j]-p[ihi][j];
                p[ihi][j]=ptry[j];
            }
        }
        
        if(ndim>BUF_SIZE)
        {
    ////        cvFree((void**)&ptry);   
        }
        
        return ytry;
    }

    /*
    Multidimensional minimization of the function func(x) 

    The matrix p[ndim+1][1..ndim] is input. 
    Its ndim+1 rows are ndim-dimensional vectors which are the vertices of
    the starting simplex. 
    On output,
    nfunk gives the number of function evaluations taken.
    */
        
#define GET_PSUM \
        for (j=0;j<ndim;j++) {\
        for (sum=0.0,i=0;i<mpts;i++) sum += p[i][j];\
            psum[j]=sum;}

    double _cvDownhillSimplex( 
        double**   p, 
        int        ndim, 
        double     MinRange,
        double     MinError,
        int        *nfunk,
        const Ptr<Solver::Function>& f,
        int nmax
        )
    {
        double res;
        int i,ihi,ilo,inhi,j,mpts=ndim+1;
        double error, range,sum,ysave,ytry,*psum;
        double buf[BUF_SIZE];
        double buf2[BUF_SIZE];

        double* y;

        *nfunk = 0;
        
        if(ndim+1 > BUF_SIZE)
        {
    ////        y = (double*)cvAlloc((ndim+1) * sizeof(double));;
        }
        else
        {
            y=buf;
        }

        /* calc first */
        for(i=0;i<ndim+1;++i)
        {
            y[i] = f->calc(p[i]);
        }

        if(ndim+1 > BUF_SIZE)
        {
    ////        psum = (double*)cvAlloc((ndim+1) * sizeof(double));
        }
        else
        {
            psum=buf2;
        }
        
        *nfunk = ndim+1;
        
        GET_PSUM
        
        for (;;) 
        {
            ilo=0;
            /* First we must determine which point is the highest (worst), next-highest, and lowest
                (best), by looping over the points in the simplex. */
            ihi = y[0]>y[1] ? (inhi=1,0) : (inhi=0,1);
            for (i=0;i<mpts;i++) 
            {
                if (y[i] <= y[ilo]) 
                    ilo=i;
                if (y[i] > y[ihi]) 
                {
                    inhi=ihi;
                    ihi=i;
                } 
                else if (y[i] > y[inhi] && i != ihi) 
                    inhi=i;
            }

            error=fabs(y[ihi]-y[ilo]);
            /* new tolerance calculation */
            range=0;
            for(i=0;i<ndim;++i)
            {
                double min = p[0][i];
                double max = p[0][i];
                double d;
                for(j=1;j<=ndim;++j)
                {
                    if( min > p[j][i] ) min = p[j][i];
                    if( max < p[j][i] ) max = p[j][i];
                }
                d = fabs(max-min);
                if(range < d) range = d;
            }

            /* Compute the fractional range from highest to lowest and return if satisfactory. */
            if(range <= MinRange || error <= MinError) 
            { /* If returning, put best point and value in slot 1. */
                _SWAP(double,y[0],y[ilo])
                for (i=0;i<ndim;i++) 
                {
                    _SWAP(double,p[0][i],p[ilo][i])
                }
                break;
            }

            if (nfunk[0] >= nmax){
                dprintf(("nmax exceeded\n"));
                return y[ilo];
            }
            nfunk[0] += 2;
            /*Begin a new iteration. First extrapolate by a factor - 1 through the face of the simplex
                across from the high point, i.e., re ect the simplex from the high point. */
            ytry = _cvDHSTry(p,y,psum,ndim,f,ihi,-1.0);
            if (ytry <= y[ilo])
            { /*Gives a result better than the best point, so try an additional extrapolation by a
                factor 2. */
                ytry = _cvDHSTry(p,y,psum,ndim,f,ihi,2.0);
            }
            else if (ytry >= y[inhi]) 
            {   /* The re ected point is worse than the second-highest, so look for an intermediate
                    lower point, i.e., do a one-dimensional contraction. */
                ysave = y[ihi];
                ytry = _cvDHSTry(p,y,psum,ndim,f,ihi,0.5);
                if (ytry >= ysave) 
                { /* Can't seem to get rid of that high point. Better
                    contract around the lowest (best) point. */
                    for (i=0;i<mpts;i++) 
                    {
                        if (i != ilo) 
                        {
                            for (j=0;j<ndim;j++)
                            {
                                p[i][j] = psum[j] = 0.5*(p[i][j]+p[ilo][j]);
                            }
                            y[i]=f->calc(psum);
                        }
                    }
                    nfunk[0] += ndim; /* Keep track of function evaluations. */
                    GET_PSUM /* Recompute psum. */
                }
            } else --(nfunk[0]); /* Correct the evaluation count. */
        } /* Go back for the test of doneness and the next iteration. */
        res = y[0];
        
        if(ndim+1 > BUF_SIZE)
        {
    ////        cvFree((void**)&psum);
    ////        cvFree((void**)&y);
        }

        return res;
    } /* DownhillSimplex*/

    void _cvDownhillSimplexInit(double** p, int n, double* dp)
    {
        long i,j;
        for(i=1;i<=n;++i)
        {
            for(j=0;j<n;++j) p[i][j]=p[0][j];
            p[i][(i-1)] += 0.5*dp[i-1];// * 0.5 * (2*(i%2)-1);
        }
        for(j=0;j<n;++j) p[0][j] -= 0.5 * dp[j];
    }/*frInitForSimplexMethod*/
    /* this function chose direction by gradient but not by max axes */
    double _cvDownhillSimplex_(
        double*    x,
        double*    step,
        int        ndim, 
        double     MinRange,
        double     MinError,
        int        *nfunk,
        const Ptr<Solver::Function>& f,
        int nmax
        )
    {
#define MAXDIM 64
        double  E;
        double* p[MAXDIM+1];
        double  pbuf[(MAXDIM+1)*MAXDIM];
        int     i;    

    ////    assert(ndim<MAXDIM);
        if(ndim>=MAXDIM) return 0;

        for(i=0;i<MAXDIM+1;++i)
        {/* init parameters */
            p[i] = pbuf+MAXDIM*i;
        }

        for(i=0;i<ndim;++i)p[0][i]=x[i];
        _cvDownhillSimplexInit(p,ndim,step);
        E = _cvDownhillSimplex( p, ndim, MinRange, MinError, nfunk,f,nmax );
        for(i=0;i<ndim;++i)x[i]=p[0][i];
        return E;
#undef MAXDIM
    }/* _cvDownhillSimplex_ */


    /*double   x[7];
    double   var_angle = 5;
    double   steps[6] = {var_angle,var_angle,var_angle,var_angle,100,100};

    double res = _cvDownhillSimplex_(
                        x,
                        steps,
                        6,
                        0.00001,
                        0.00001,
                        &count,
                        ::Error,
                        (void*)&points_str);*/
}}
//WORKFLOW: eval->2tests->memory_alloc->API_tests
