#include "test_precomp.hpp"
#include "opencv2/optim.hpp"

TEST(Optim_LpSolver, regression)
{
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
    if(false){
    //cormen's example #4 - unfeasible
    A=(cv::Mat_<double>(1,3)<<-1,-1,-1);
    B=(cv::Mat_<double>(2,4)<<-2,-7.5,-3,-10000,-20,-5,-10,-30000);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::optim::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(1,2)<<1,0);
    ASSERT_EQ(cv::countNonZero(z!=etalon_z),0);
    }
}

//TODO
// get optimal solution from initial (0,0,...,0) - DONE
// milestone: pass first test (wo initial solution) - DONE
    // learn how to get initial solution
    // Blands_rule
    // 1_more_test & make_more_clear
    // -> **contact_Vadim**: min_l2_norm, init_optional_fsbl_check, error_codes, comment_style-too_many?, copyTo temp headers
// ??how to get smallest l2 norm
// FUTURE: compress&debug-> more_tests(Cormen) -> readNumRecipes-> fast&stable || hill_climbing
