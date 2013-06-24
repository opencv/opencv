#include "precomp.hpp"

namespace cv{namespace optim{

double LPSolver::solve(const Function& F,const Constraints& C, OutputArray result)const{
    printf("call to solve\n");

    //TODO: sanity check and throw exception, if appropriate

    //TODO: copy A,b,z
    
    //TODO: run simplex algo

    return 0.0;
}

double LPSolver::LPFunction::calc(InputArray args)const{
    printf("call to LPFunction::calc()\n");
    return 0.0;
}

}}
