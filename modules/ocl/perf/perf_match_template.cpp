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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <iomanip>
#ifdef HAVE_OPENCL
using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

#ifndef MWC_TEST_UTILITY
#define MWC_TEST_UTILITY
//////// Utility
#ifndef DIFFERENT_SIZES
#else
#undef DIFFERENT_SIZES
#endif
#define DIFFERENT_SIZES testing::Values(cv::Size(256, 256), cv::Size(3000, 3000))

// Param class
#ifndef IMPLEMENT_PARAM_CLASS
#define IMPLEMENT_PARAM_CLASS(name, type) \
class name \
{ \
public: \
    name ( type arg = type ()) : val_(arg) {} \
    operator type () const {return val_;} \
private: \
    type val_; \
}; \
    inline void PrintTo( name param, std::ostream* os) \
{ \
    *os << #name <<  "(" << testing::PrintToString(static_cast< type >(param)) << ")"; \
}

IMPLEMENT_PARAM_CLASS(Channels, int)
#endif // IMPLEMENT_PARAM_CLASS
#endif // MWC_TEST_UTILITY

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate
#define ALL_TEMPLATE_METHODS testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR), TemplateMethod(cv::TM_CCOEFF), TemplateMethod(cv::TM_SQDIFF_NORMED), TemplateMethod(cv::TM_CCORR_NORMED), TemplateMethod(cv::TM_CCOEFF_NORMED))

IMPLEMENT_PARAM_CLASS(TemplateSize, cv::Size);

const char *TEMPLATE_METHOD_NAMES[6] = {"TM_SQDIFF", "TM_SQDIFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_CCOEFF", "TM_CCOEFF_NORMED"};

PARAM_TEST_CASE(MatchTemplate, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;
    //vector<cv::ocl::Info> oclinfo;
    
    virtual void SetUp()
    {
        size = GET_PARAM(0);
        templ_size = GET_PARAM(1);
        cn = GET_PARAM(2);
        method = GET_PARAM(3);
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
    }
};
struct MatchTemplate8U : MatchTemplate {};

TEST_P(MatchTemplate8U, Performance)
{
    std::cout << "Method: " << TEMPLATE_METHOD_NAMES[method] << std::endl;
    std::cout << "Image Size: (" << size.width << ", " << size.height << ")" << std::endl;
    std::cout << "Template Size: (" << templ_size.width << ", " << templ_size.height << ")" << std::endl;
    std::cout << "Channels: " << cn << std::endl;
    
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_8U, cn));
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_8U, cn));
    cv::Mat dst_gold;
    cv::ocl::oclMat dst;
    
    
    
    
    
    double totalgputick = 0;
    double totalgputick_kernel = 0;
    
    double t1 = 0;
    double t2 = 0;
    
    for(int j = 0; j < LOOP_TIMES + 1; j ++)
    {
    
        t1 = (double)cvGetTickCount();//gpu start1
        
        cv::ocl::oclMat ocl_image = cv::ocl::oclMat(image);//upload
        cv::ocl::oclMat ocl_templ = cv::ocl::oclMat(templ);//upload
        
        t2 = (double)cvGetTickCount(); //kernel
        cv::ocl::matchTemplate(ocl_image, ocl_templ, dst, method);
        t2 = (double)cvGetTickCount() - t2;//kernel
        
        cv::Mat cpu_dst;
        dst.download(cpu_dst); //download
        
        t1 = (double)cvGetTickCount() - t1;//gpu end1
        
        if(j == 0)
        {
            continue;
        }
        
        totalgputick = t1 + totalgputick;
        totalgputick_kernel = t2 + totalgputick_kernel;
        
    }
    
    cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    
    
}


struct MatchTemplate32F : MatchTemplate {};
TEST_P(MatchTemplate32F, Performance)
{
    std::cout << "Method: " << TEMPLATE_METHOD_NAMES[method] << std::endl;
    std::cout << "Image Size: (" << size.width << ", " << size.height << ")" << std::endl;
    std::cout << "Template Size: (" << templ_size.width << ", " << templ_size.height << ")" << std::endl;
    std::cout << "Channels: " << cn << std::endl;
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_32F, cn));
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_32F, cn));
    
    cv::Mat dst_gold;
    cv::ocl::oclMat dst;
    
    
    
    
    double totalgputick = 0;
    double totalgputick_kernel = 0;
    
    double t1 = 0;
    double t2 = 0;
    
    for(int j = 0; j < LOOP_TIMES; j ++)
    {
    
        t1 = (double)cvGetTickCount();//gpu start1
        
        cv::ocl::oclMat ocl_image = cv::ocl::oclMat(image);//upload
        cv::ocl::oclMat ocl_templ = cv::ocl::oclMat(templ);//upload
        
        t2 = (double)cvGetTickCount(); //kernel
        cv::ocl::matchTemplate(ocl_image, ocl_templ, dst, method);
        t2 = (double)cvGetTickCount() - t2;//kernel
        
        cv::Mat cpu_dst;
        dst.download(cpu_dst); //download
        
        t1 = (double)cvGetTickCount() - t1;//gpu end1
        
        totalgputick = t1 + totalgputick;
        
        totalgputick_kernel = t2 + totalgputick_kernel;
        
    }
    
    cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    
    
    
}


INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplate8U,
                        testing::Combine(
                            testing::Values(cv::Size(1280, 1024), cv::Size(MWIDTH, MHEIGHT), cv::Size(1800, 1500)),
                            testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16))/*, TemplateSize(cv::Size(30, 30))*/),
                            testing::Values(Channels(1), Channels(4)/*, Channels(3)*/),
                            ALL_TEMPLATE_METHODS
                        )
                       );

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplate32F, testing::Combine(
                            testing::Values(cv::Size(1280, 1024), cv::Size(MWIDTH, MHEIGHT), cv::Size(1800, 1500)),
                            testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16))/*, TemplateSize(cv::Size(30, 30))*/),
                            testing::Values(Channels(1), Channels(4) /*, Channels(3)*/),
                            testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))));

#endif //HAVE_OPENCL