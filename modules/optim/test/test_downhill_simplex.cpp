#include "test_precomp.hpp"
#include <cstdlib>
#include <cmath>
#include <algorithm>

static void mytest(cv::Ptr<cv::optim::DownhillSolver> solver,cv::Ptr<cv::optim::Solver::Function> ptr_F,cv::Mat& x,cv::Mat& step,
        cv::Mat& etalon_x,double etalon_res){
    solver->setFunction(ptr_F);
    int ndim=MAX(step.cols,step.rows);
    solver->setInitStep(step);
    cv::Mat settedStep;
    solver->getInitStep(settedStep);
    ASSERT_TRUE(settedStep.rows==1 && settedStep.cols==ndim);
    ASSERT_TRUE(std::equal(step.begin<double>(),step.end<double>(),settedStep.begin<double>()));
    std::cout<<"step setted:\n\t"<<step<<std::endl;
    double res=solver->minimize(x);
    std::cout<<"res:\n\t"<<res<<std::endl;
    std::cout<<"x:\n\t"<<x<<std::endl;
    std::cout<<"etalon_res:\n\t"<<etalon_res<<std::endl;
    std::cout<<"etalon_x:\n\t"<<etalon_x<<std::endl;
    double tol=solver->getTermCriteria().epsilon;
    ASSERT_TRUE(std::abs(res-etalon_res)<tol);
    /*for(cv::Mat_<double>::iterator it1=x.begin<double>(),it2=etalon_x.begin<double>();it1!=x.end<double>();it1++,it2++){
        ASSERT_TRUE(std::abs((*it1)-(*it2))<tol);
    }*/
    std::cout<<"--------------------------\n";
}

class SphereF:public cv::optim::Solver::Function{
public:
    double calc(const double* x)const{
        return x[0]*x[0]+x[1]*x[1];
    }
};
class RosenbrockF:public cv::optim::Solver::Function{
    double calc(const double* x)const{
        return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])+(1-x[0])*(1-x[0]);
    }
};

TEST(Optim_Downhill, regression_basic){
    cv::Ptr<cv::optim::DownhillSolver> solver=cv::optim::createDownhillSolver();
#if 1
    {
        cv::Ptr<cv::optim::Solver::Function> ptr_F(new SphereF());
        cv::Mat x=(cv::Mat_<double>(1,2)<<1.0,1.0),
            step=(cv::Mat_<double>(2,1)<<-0.5,-0.5),
            etalon_x=(cv::Mat_<double>(1,2)<<-0.0,0.0);
        double etalon_res=0.0;
        mytest(solver,ptr_F,x,step,etalon_x,etalon_res);
    }
#endif
#if 1
    {
        cv::Ptr<cv::optim::Solver::Function> ptr_F(new RosenbrockF());
        cv::Mat x=(cv::Mat_<double>(2,1)<<0.0,0.0),
            step=(cv::Mat_<double>(2,1)<<0.5,+0.5),
            etalon_x=(cv::Mat_<double>(2,1)<<1.0,1.0);
        double etalon_res=0.0;
        mytest(solver,ptr_F,x,step,etalon_x,etalon_res);
    }
#endif
}
