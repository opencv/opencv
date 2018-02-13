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
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Core_LPSolver, regression_basic){
    cv::Mat A,B,z,etalon_z;

#if 1
    //cormen's example #1
    A=(cv::Mat_<double>(3,1)<<3,1,2);
    B=(cv::Mat_<double>(3,4)<<1,1,3,30,2,2,5,24,4,1,2,36);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(3,1)<<8,4,0);
    ASSERT_LT(cvtest::norm(z, etalon_z, cv::NORM_L1), 1e-12);
#endif

#if 1
    //cormen's example #2
    A=(cv::Mat_<double>(1,2)<<18,12.5);
    B=(cv::Mat_<double>(3,3)<<1,1,20,1,0,20,0,1,16);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(2,1)<<20,0);
    ASSERT_LT(cvtest::norm(z, etalon_z, cv::NORM_L1), 1e-12);
#endif

#if 1
    //cormen's example #3
    A=(cv::Mat_<double>(1,2)<<5,-3);
    B=(cv::Mat_<double>(2,3)<<1,-1,1,2,1,2);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(2,1)<<1,0);
    ASSERT_LT(cvtest::norm(z, etalon_z, cv::NORM_L1), 1e-12);
#endif
}

TEST(Core_LPSolver, regression_init_unfeasible){
    cv::Mat A,B,z,etalon_z;

#if 1
    //cormen's example #4 - unfeasible
    A=(cv::Mat_<double>(1,3)<<-1,-1,-1);
    B=(cv::Mat_<double>(2,4)<<-2,-7.5,-3,-10000,-20,-5,-10,-30000);
    std::cout<<"here A goes\n"<<A<<"\n";
    cv::solveLP(A,B,z);
    std::cout<<"here z goes\n"<<z<<"\n";
    etalon_z=(cv::Mat_<double>(3,1)<<1250,1000,0);
    ASSERT_LT(cvtest::norm(z, etalon_z, cv::NORM_L1), 1e-12);
#endif
}

TEST(DISABLED_Core_LPSolver, regression_absolutely_unfeasible){
    cv::Mat A,B,z,etalon_z;

#if 1
    //trivial absolutely unfeasible example
    A=(cv::Mat_<double>(1,1)<<1);
    B=(cv::Mat_<double>(2,2)<<1,-1);
    std::cout<<"here A goes\n"<<A<<"\n";
    int res=cv::solveLP(A,B,z);
    ASSERT_EQ(res,-1);
#endif
}

TEST(Core_LPSolver, regression_multiple_solutions){
    cv::Mat A,B,z,etalon_z;

#if 1
    //trivial example with multiple solutions
    A=(cv::Mat_<double>(2,1)<<1,1);
    B=(cv::Mat_<double>(1,3)<<1,1,1);
    std::cout<<"here A goes\n"<<A<<"\n";
    int res=cv::solveLP(A,B,z);
    printf("res=%d\n",res);
    printf("scalar %g\n",z.dot(A));
    std::cout<<"here z goes\n"<<z<<"\n";
    ASSERT_EQ(res,1);
    ASSERT_LT(fabs(z.dot(A) - 1), DBL_EPSILON);
#endif
}

TEST(Core_LPSolver, regression_cycling){
    cv::Mat A,B,z,etalon_z;

#if 1
    //example with cycling from http://people.orie.cornell.edu/miketodd/or630/SimplexCyclingExample.pdf
    A=(cv::Mat_<double>(4,1)<<10,-57,-9,-24);
    B=(cv::Mat_<double>(3,5)<<0.5,-5.5,-2.5,9,0,0.5,-1.5,-0.5,1,0,1,0,0,0,1);
    std::cout<<"here A goes\n"<<A<<"\n";
    int res=cv::solveLP(A,B,z);
    printf("res=%d\n",res);
    printf("scalar %g\n",z.dot(A));
    std::cout<<"here z goes\n"<<z<<"\n";
    ASSERT_LT(fabs(z.dot(A) - 1), DBL_EPSILON);
    //ASSERT_EQ(res,1);
#endif
}

}} // namespace
