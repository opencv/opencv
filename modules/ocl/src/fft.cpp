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
#include <iomanip>
#include "precomp.hpp"

#ifdef HAVE_CLAMDFFT

using namespace cv;
using namespace cv::ocl;
using namespace std;

#if !defined (HAVE_OPENCL)
void cv::ocl::dft(const oclMat& src, oclMat& dst, int flags) { throw_nogpu(); }
#else

#include <clAmdFft.h>

namespace cv{ namespace ocl {
    enum FftType
    {
        C2R = 1, // complex to complex
        R2C = 2, // real to opencl HERMITIAN_INTERLEAVED
        C2C = 3  // opencl HERMITIAN_INTERLEAVED to real
    };
    struct FftPlan
    {
        friend void fft_setup();
        friend void fft_teardown();
        ~FftPlan();
    protected:
        FftPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type);
        const Size dft_size;
        const int src_step, dst_step;
        const int flags;
        const FftType type;
        clAmdFftPlanHandle plHandle;
        static vector<FftPlan*> planStore;
        static bool started;
        static clAmdFftSetupData * setupData;
    public:
        // return a baked plan-> 
        // if there is one matched plan, return it
        // if not, bake a new one, put it into the planStore and return it.
        static clAmdFftPlanHandle getPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type);
    };
}}
bool cv::ocl::FftPlan::started = false;
vector<cv::ocl::FftPlan*> cv::ocl::FftPlan::planStore = vector<cv::ocl::FftPlan*>();
clAmdFftSetupData * cv::ocl::FftPlan::setupData = 0;

void cv::ocl::fft_setup()
{
    if(FftPlan::started)
    {
        return;
    }
    FftPlan::setupData = new clAmdFftSetupData;
    openCLSafeCall(clAmdFftInitSetupData( FftPlan::setupData ));
    FftPlan::started = true;
}
void cv::ocl::fft_teardown()
{
    if(!FftPlan::started)
    {
        return;
    }
    delete FftPlan::setupData;
    for(int i = 0; i < FftPlan::planStore.size(); i ++)
    {
        delete FftPlan::planStore[i];
    }
    FftPlan::planStore.clear();
    openCLSafeCall( clAmdFftTeardown( ) );
    FftPlan::started = false;
}

// bake a new plan
cv::ocl::FftPlan::FftPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type)
    : dft_size(_dft_size), src_step(_src_step), dst_step(_dst_step), flags(_flags), type(_type), plHandle(0)
{
    if(!FftPlan::started)
    {
        // implicitly do fft setup
        fft_setup();
    }

    bool is_1d_input	= (_dft_size.height == 1);
    int is_row_dft		= flags & DFT_ROWS;
    int is_scaled_dft		= flags & DFT_SCALE;
    int is_inverse			= flags & DFT_INVERSE;

    clAmdFftResultLocation	place;
    clAmdFftLayout			inLayout;
    clAmdFftLayout			outLayout;
    clAmdFftDim				dim = is_1d_input||is_row_dft ? CLFFT_1D : CLFFT_2D;

    size_t batchSize		 = is_row_dft?dft_size.height : 1;
    size_t clLengthsIn[ 3 ]  = {1, 1, 1};
    size_t clStridesIn[ 3 ]  = {1, 1, 1};
    size_t clLengthsOut[ 3 ] = {1, 1, 1};
    size_t clStridesOut[ 3 ] = {1, 1, 1};
    clLengthsIn[0]			 = dft_size.width;
    clLengthsIn[1]			 = is_row_dft ? 1 : dft_size.height;
    clStridesIn[0]			 = 1;
    clStridesOut[0]			 = 1;

    switch(_type)
    {
    case C2C:
        inLayout        = CLFFT_COMPLEX_INTERLEAVED;
        outLayout       = CLFFT_COMPLEX_INTERLEAVED;
        clStridesIn[1]  = src_step / sizeof(std::complex<float>);
        clStridesOut[1] = clStridesIn[1];
        break;
    case R2C:
        CV_Assert(!is_row_dft); // this is not supported yet
        inLayout        = CLFFT_REAL;
        outLayout       = CLFFT_HERMITIAN_INTERLEAVED;
        clStridesIn[1]  = src_step / sizeof(float);
        clStridesOut[1] = dst_step / sizeof(std::complex<float>);
        break;
    case C2R:
        CV_Assert(!is_row_dft); // this is not supported yet
        inLayout        = CLFFT_HERMITIAN_INTERLEAVED;
        outLayout       = CLFFT_REAL;
        clStridesIn[1]  = src_step / sizeof(std::complex<float>);
        clStridesOut[1] = dst_step / sizeof(float);
        break;
    default:
        //std::runtime_error("does not support this convertion!");
        cout << "Does not support this convertion!" << endl;
        throw exception();
        break;
    }

    clStridesIn[2]  = is_row_dft ? clStridesIn[1]  : dft_size.width * clStridesIn[1];
    clStridesOut[2] = is_row_dft ? clStridesOut[1] : dft_size.width * clStridesOut[1];

    openCLSafeCall( clAmdFftCreateDefaultPlan( &plHandle, Context::getContext()->impl->clContext, dim, clLengthsIn ) );

    openCLSafeCall( clAmdFftSetResultLocation( plHandle, CLFFT_OUTOFPLACE ) );
    openCLSafeCall( clAmdFftSetLayout( plHandle, inLayout, outLayout ) );
    openCLSafeCall( clAmdFftSetPlanBatchSize( plHandle, batchSize ) );

    openCLSafeCall( clAmdFftSetPlanInStride  ( plHandle, dim, clStridesIn ) );
    openCLSafeCall( clAmdFftSetPlanOutStride ( plHandle, dim, clStridesOut ) );
    openCLSafeCall( clAmdFftSetPlanDistance  ( plHandle, clStridesIn[ dim ], clStridesIn[ dim ]) );
    openCLSafeCall( clAmdFftBakePlan( plHandle, 1, &(Context::getContext()->impl->clCmdQueue), NULL, NULL ) );
}
cv::ocl::FftPlan::~FftPlan()
{
    for(int i = 0; i < planStore.size(); i ++)
    {
        if(planStore[i]->plHandle == plHandle)
        {
            planStore.erase(planStore.begin()+ i);
        }
    }
    openCLSafeCall( clAmdFftDestroyPlan( &plHandle ) );
}

clAmdFftPlanHandle cv::ocl::FftPlan::getPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type)
{
    // go through search
    for(int i = 0; i < planStore.size(); i ++)
    {
        FftPlan * plan = planStore[i];
        if(
            plan->dft_size.width == _dft_size.width && 
            plan->dft_size.height == _dft_size.height &&
            plan->flags == _flags &&
            plan->src_step == _src_step &&
            plan->dst_step == _dst_step &&
            plan->type == _type
            )
        {
            return plan->plHandle;
        }
    }
    // no baked plan is found
    FftPlan *newPlan = new FftPlan(_dft_size, _src_step, _dst_step, _flags, _type);
    planStore.push_back(newPlan);
    return newPlan->plHandle;
}

void cv::ocl::dft(const oclMat& src, oclMat& dst, Size dft_size, int flags) 
{
    if(dft_size == Size(0,0))
    {
        dft_size = src.size();
    }
    // check if the given dft size is of optimal dft size
    CV_Assert(dft_size.area() == getOptimalDFTSize(dft_size.area()));

    // similar assertions with cuda module
    CV_Assert(src.type() == CV_32F || src.type() == CV_32FC2);

    // we don't support DFT_SCALE flag
    CV_Assert(!(DFT_SCALE & flags));

    bool is_1d_input	= (src.rows == 1);
    int is_row_dft		= flags & DFT_ROWS;
    int is_scaled_dft		= flags & DFT_SCALE;
    int is_inverse			= flags & DFT_INVERSE;
    bool is_complex_input	= src.channels() == 2;
    bool is_complex_output	= !(flags & DFT_REAL_OUTPUT);

    // We don't support real-to-real transform
    CV_Assert(is_complex_input || is_complex_output);
    FftType type = (FftType)(is_complex_input << 0 | is_complex_output << 1);

    switch(type)
    {
    case C2C:
        dst.create(src.rows, src.cols, CV_32FC2);
        break;
    case R2C:
        CV_Assert(!is_row_dft); // this is not supported yet
        dst.create(src.rows, src.cols/2 + 1, CV_32FC2);
        break;
    case C2R:
        CV_Assert(dft_size.width / 2 + 1 == src.cols && dft_size.height == src.rows);
        CV_Assert(!is_row_dft); // this is not supported yet
        dst.create(src.rows, dft_size.width, CV_32FC1);
        break;
    default:
        //std::runtime_error("does not support this convertion!");
        cout << "Does not support this convertion!" << endl;
        throw exception();
        break;
    }
    clAmdFftPlanHandle plHandle = FftPlan::getPlan(dft_size, src.step, dst.step, flags, type);

    //get the buffersize
    size_t buffersize=0;
    openCLSafeCall( clAmdFftGetTmpBufSize(plHandle, &buffersize ) );

    //allocate the intermediate buffer	
    cl_mem clMedBuffer=NULL;
    if (buffersize)
    {
        cl_int medstatus;
        clMedBuffer = clCreateBuffer ( src.clCxt->impl->clContext, CL_MEM_READ_WRITE, buffersize, 0, &medstatus);
        openCLSafeCall( medstatus );
    }
    openCLSafeCall( clAmdFftEnqueueTransform( plHandle, 
        is_inverse?CLFFT_BACKWARD:CLFFT_FORWARD, 
        1, 
        &src.clCxt->impl->clCmdQueue, 
        0, NULL, NULL, 
        (cl_mem*)&src.data, (cl_mem*)&dst.data, clMedBuffer ) );
    openCLSafeCall( clFinish(src.clCxt->impl->clCmdQueue) );
    if(clMedBuffer)
    {
        openCLFree(clMedBuffer);
    }
}

#endif
#endif //HAVE_CLAMDFFT
