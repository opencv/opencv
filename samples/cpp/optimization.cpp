#include "opencv2/core/core_c.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cvdef.h"
#include <utility>
#include <iostream>

class RosenbrockF_CG : public cv::MinProblemSolver::Function {
    int getDims() const { return 2; }

    double calc(const double* x)const{
        return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])+(1-x[0])*(1-x[0]);
    }

    void getGradient(const double* x,double* grad){
        grad[0]=-2*(1-x[0])-400*(x[1]-x[0]*x[0])*x[0];
        grad[1]=200*(x[1]-x[0]*x[0]);
    }
};

template<typename T>
static std::pair<double, double> get_error(cv::Ptr<T> solver) {
    cv::Mat x = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
    cv::Mat etalon_x = (cv::Mat_<double>(2, 1) << 1.0, 1.0);
    cv::Ptr<cv::MinProblemSolver::Function> ptr_F(new RosenbrockF_CG());
    const double e1 = static_cast<double>(cv::getTickCount());
    solver->setFunction(ptr_F);
    const double duration = (cv::getTickCount() - e1) / cv::getTickFrequency();
    solver->minimize(x);
    double err = 0;

    for (cv::Mat_<double>::iterator it1 = x.begin<double>(), it2 = etalon_x.begin<double>(); it1 != x.end<double>(); it1++, it2++) {
        err += std::abs((*it1) - (*it2));
    }

    return std::make_pair(err, duration);
}

template<>
std::pair<double, double> get_error(cv::Ptr<cv::DownhillSolver> solver) {
    cv::Mat x = (cv::Mat_<double>(2, 1) << 0.0, 0.0);
    cv::Mat etalon_x = (cv::Mat_<double>(2, 1) << 1.0, 1.0);
    cv::Ptr<cv::MinProblemSolver::Function> ptr_F(new RosenbrockF_CG());
    const double e1 = static_cast<double>(cv::getTickCount());
    solver->setFunction(ptr_F);
    cv::Mat step = (cv::Mat_<double>(2,1)<<-0.5,-0.5);
    solver->setInitStep(step);
    const double duration = (cv::getTickCount() - e1) / cv::getTickFrequency();
    solver->minimize(x);
    double err = 0;

    for (cv::Mat_<double>::iterator it1 = x.begin<double>(), it2 = etalon_x.begin<double>(); it1 != x.end<double>(); it1++, it2++) {
        err += std::abs((*it1) - (*it2));
    }

    return std::make_pair(err, duration);
}

template<typename T>
static void print_result(const char *name, cv::Ptr<T> solver) {
    std::pair<double, double> res = get_error(solver);
    std::cout << name << ": time " << res.second << " sec" << "; error " << res.first << std::endl;
}

int main() {
    cv::Ptr<cv::ConjGradSolver> solver_cg = cv::ConjGradSolver::create();
    cv::Ptr<cv::DownhillSolver> solver_dh = cv::DownhillSolver::create();
    cv::Ptr<cv::BFGSSolver> solver_bfgs = cv::BFGSSolver::create();
    cv::Ptr<cv::LBFGSSolver> solver_lbfgs = cv::LBFGSSolver::create();
    print_result("Conjugate Gradients", solver_cg);
    print_result("Downhill", solver_dh);
    print_result("BFGS", solver_bfgs);
    print_result("L-BFGS", solver_bfgs);

    return 0;
}
