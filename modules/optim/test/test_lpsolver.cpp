#include "test_precomp.hpp"
#include "opencv2/optim.hpp"

TEST(Optim_LpSolver, regression_basic){
    cv::Mat A,B,z,etalon_z;

    if(true){
    //cormen's example #1
    A=(cv::Mat_<double>(1,3)<<3,1,2);
    B=(cv::Mat_<double>(3,4)<<1,1,3,30,2,2,5,24,4,1,2,36);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::optim::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(1,3)<<8,4,0);
    ASSERT_EQ(cv::countNonZero(z!=etalon_z),0);
    }

    if(true){
    //cormen's example #2
    A=(cv::Mat_<double>(1,2)<<18,12.5);
    B=(cv::Mat_<double>(3,3)<<1,1,20,1,0,20,0,1,16);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::optim::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(1,2)<<20,0);
    ASSERT_EQ(cv::countNonZero(z!=etalon_z),0);
    }

    if(true){
    //cormen's example #3
    A=(cv::Mat_<double>(1,2)<<5,-3);
    B=(cv::Mat_<double>(2,3)<<1,-1,1,2,1,2);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::optim::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(1,2)<<1,0);
    ASSERT_EQ(cv::countNonZero(z!=etalon_z),0);
    }
}

TEST(Optim_LpSolver, regression_init_unfeasible){
    cv::Mat A,B,z,etalon_z;

    if(true){
    //cormen's example #4 - unfeasible
    A=(cv::Mat_<double>(1,3)<<-1,-1,-1);
    B=(cv::Mat_<double>(2,4)<<-2,-7.5,-3,-10000,-20,-5,-10,-30000);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::optim::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(1,3)<<1250,1000,0);
    ASSERT_EQ(cv::countNonZero(z!=etalon_z),0);
    }
}

TEST(Optim_LpSolver, regression_absolutely_unfeasible){
    cv::Mat A,B,z,etalon_z;

    if(true){
    //trivial absolutely unfeasible example
    A=(cv::Mat_<double>(1,1)<<1);
    B=(cv::Mat_<double>(2,2)<<1,-1);
    std::cout<<"here A goes\n"<<A<<"\n";
    int res=cv::optim::solveLP(A,B,z);
    ASSERT_EQ(res,-1);
    }
}

TEST(Optim_LpSolver, regression_multiple_solutions){
    cv::Mat A,B,z,etalon_z;

    if(true){
    //trivial example with multiple solutions
    A=(cv::Mat_<double>(1,2)<<1,1);
    B=(cv::Mat_<double>(1,3)<<1,1,1);
    std::cout<<"here A goes\n"<<A<<"\n";
    int res=cv::optim::solveLP(A,B,z);
    printf("res=%d\n",res);
    printf("scalar %g\n",z.dot(A));
    std::cout<<"here z goes\n"<<z<<"\n";
    ASSERT_EQ(res,1);
    ASSERT_EQ(z.dot(A),1);
    }

    if(false){
    //cormen's example from chapter about initialize_simplex
    //online solver told it has inf many solutions, but I'm not sure
    A=(cv::Mat_<double>(1,2)<<2,-1);
    B=(cv::Mat_<double>(2,3)<<2,-1,2,1,-5,-4);
    std::cout<<"here A goes\n"<<A<<"\n";
    int res=cv::optim::solveLP(A,B,z);
    printf("res=%d\n",res);
    printf("scalar %g\n",z.dot(A));
    std::cout<<"here z goes\n"<<z<<"\n";
    ASSERT_EQ(res,1);
    }
}

TEST(Optim_LpSolver, regression_cycling){
    cv::Mat A,B,z,etalon_z;

    if(true){
    //example with cycling from http://people.orie.cornell.edu/miketodd/or630/SimplexCyclingExample.pdf
    A=(cv::Mat_<double>(1,4)<<10,-57,-9,-24);
    B=(cv::Mat_<double>(3,5)<<0.5,-5.5,-2.5,9,0,0.5,-1.5,-0.5,1,0,1,0,0,0,1);
    std::cout<<"here A goes\n"<<A<<"\n";
    int res=cv::optim::solveLP(A,B,z);
    printf("res=%d\n",res);
    printf("scalar %g\n",z.dot(A));
    std::cout<<"here z goes\n"<<z<<"\n";
    ASSERT_EQ(z.dot(A),1);
    //ASSERT_EQ(res,1);
    }
}

//TODO
// get optimal solution from initial (0,0,...,0) - DONE
// milestone: pass first test (wo initial solution) - DONE
    //
    // ??how_check_multiple_solutions & pass_test - DONE
    // Blands_rule - DONE
    // (assert, assign) - DONE
    //
    // (&1tests on cycling)
    // make_more_clear
    // wrap in OOP
    //
    // non-trivial tests
    // pull-request
    //
    // study hill and other algos
    //
// ??how to get smallest l2 norm
// FUTURE: compress&debug-> more_tests(Cormen) -> readNumRecipes-> fast&stable || hill_climbing
