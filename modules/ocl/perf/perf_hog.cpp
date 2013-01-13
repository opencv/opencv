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
//    Peng Xiao, pengxiao@multicorewareinc.com
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
extern std::string workdir;

#ifndef MWC_TEST_UTILITY
#define MWC_TEST_UTILITY

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

#endif // IMPLEMENT_PARAM_CLASS
#endif // MWC_TEST_UTILITY

IMPLEMENT_PARAM_CLASS(WinSizw48, bool);

PARAM_TEST_CASE(HOG, WinSizw48, bool)
{
    bool is48;
    vector<float> detector;
    virtual void SetUp()
    {
        is48 = GET_PARAM(0);
        
        if(is48)
        {
            detector = cv::ocl::HOGDescriptor::getPeopleDetector48x96();
        }
        else
        {
            detector = cv::ocl::HOGDescriptor::getPeopleDetector64x128();
        }
    }
};

TEST_P(HOG, Performance)
{
    cv::Mat img = readImage(workdir + "lena.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());
    
    // define HOG related arguments
    float scale = 1.05f;
    //int nlevels = 13;
    int gr_threshold = 8;
    float hit_threshold = 1.4f;
    //bool hit_threshold_auto = true;
    
    int win_width = is48 ? 48 : 64;
    int win_stride_width = 8;
    int win_stride_height = 8;
    
    bool gamma_corr = true;
    
    Size win_size(win_width, win_width * 2); //(64, 128) or (48, 96)
    Size win_stride(win_stride_width, win_stride_height);
    
    cv::ocl::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
                                   cv::ocl::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
                                   cv::ocl::HOGDescriptor::DEFAULT_NLEVELS);
                                   
    gpu_hog.setSVMDetector(detector);
    
    double totalgputick = 0;
    double totalgputick_kernel = 0;
    
    double t1 = 0;
    double t2 = 0;
    
    for(int j = 0; j < LOOP_TIMES + 1; j ++)
    {
        t1 = (double)cvGetTickCount();//gpu start1
        
        ocl::oclMat d_src(img);//upload
        
        t2 = (double)cvGetTickCount(); //kernel
        
        vector<Rect> found;
        gpu_hog.detectMultiScale(d_src, found, hit_threshold, win_stride,
                                 Size(0, 0), scale, gr_threshold);
                                 
        t2 = (double)cvGetTickCount() - t2;//kernel
        
        // no download time for HOG
        
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


INSTANTIATE_TEST_CASE_P(GPU_ObjDetect, HOG, testing::Combine(testing::Values(WinSizw48(false), WinSizw48(true)), testing::Values(false)));

#endif  //Have opencl