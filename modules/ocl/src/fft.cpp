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
//     and/or other materials provided with the distribution.
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

using namespace cv;
using namespace cv::ocl;

#if !defined HAVE_CLAMDFFT
void cv::ocl::dft(const oclMat&, oclMat&, Size, int)
{
    CV_Error(CV_OpenCLNoAMDBlasFft, "OpenCL DFT is not implemented");
}
namespace cv { namespace ocl {
    void fft_teardown();
}}
void cv::ocl::fft_teardown(){}
#else
#include "opencv2/ocl/cl_runtime/clamdfft_runtime.hpp"
namespace cv
{
    namespace ocl
    {
        void fft_setup();
        void fft_teardown();
        enum FftType
        {
            C2R = 1, // complex to complex
            R2C = 2, // real to opencl HERMITIAN_INTERLEAVED
            C2C = 3  // opencl HERMITIAN_INTERLEAVED to real
        };
        struct FftPlan
        {
        protected:
            clAmdFftPlanHandle plHandle;
            FftPlan& operator=(const FftPlan&);
        public:
            FftPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type);
            ~FftPlan();
            inline clAmdFftPlanHandle getPlanHandle() { return plHandle; }

            const Size dft_size;
            const int src_step, dst_step;
            const int flags;
            const FftType type;
        };
        class PlanCache
        {
        protected:
            PlanCache();
            ~PlanCache();
            static PlanCache* planCache;

            bool started;
            vector<FftPlan *> planStore;
            clAmdFftSetupData *setupData;
        public:
            friend void fft_setup();
            friend void fft_teardown();

            static PlanCache* getPlanCache()
            {
                if (NULL == planCache)
                    planCache = new PlanCache();
                return planCache;
            }
            // return a baked plan->
            // if there is one matched plan, return it
            // if not, bake a new one, put it into the planStore and return it.
            static FftPlan* getPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type);

            // remove a single plan from the store
            // return true if the plan is successfully removed
            // else
            static bool removePlan(clAmdFftPlanHandle );
        };
    }
}
PlanCache* PlanCache::planCache = NULL;

void cv::ocl::fft_setup()
{
    PlanCache& pCache = *PlanCache::getPlanCache();
    if(pCache.started)
    {
        return;
    }
    if (pCache.setupData == NULL)
        pCache.setupData = new clAmdFftSetupData;
    openCLSafeCall(clAmdFftInitSetupData( pCache.setupData ));
    pCache.started = true;
}
void cv::ocl::fft_teardown()
{
    PlanCache& pCache = *PlanCache::getPlanCache();

    if(!pCache.started)
        return;

    for(size_t i = 0; i < pCache.planStore.size(); i ++)
        delete pCache.planStore[i];
    pCache.planStore.clear();

    try
    {
        openCLSafeCall( clAmdFftTeardown( ) );
    }
    catch (const std::bad_alloc &)
    { }

    delete pCache.setupData; pCache.setupData = NULL;
    pCache.started = false;
}

// bake a new plan
cv::ocl::FftPlan::FftPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type)
    : plHandle(0), dft_size(_dft_size), src_step(_src_step), dst_step(_dst_step), flags(_flags), type(_type)
{
    fft_setup();

    bool is_1d_input    = (_dft_size.height == 1);
    int is_row_dft        = flags & DFT_ROWS;
    int is_scaled_dft   = flags & DFT_SCALE;
    int is_inverse        = flags & DFT_INVERSE;

    //clAmdFftResultLocation    place;
    clAmdFftLayout            inLayout;
    clAmdFftLayout            outLayout;
    clAmdFftDim                dim = is_1d_input || is_row_dft ? CLFFT_1D : CLFFT_2D;

    size_t batchSize         = is_row_dft ? dft_size.height : 1;
    size_t clLengthsIn[ 3 ]  = {1, 1, 1};
    size_t clStridesIn[ 3 ]  = {1, 1, 1};
    //size_t clLengthsOut[ 3 ] = {1, 1, 1};
    size_t clStridesOut[ 3 ] = {1, 1, 1};
    clLengthsIn[0]             = dft_size.width;
    clLengthsIn[1]             = is_row_dft ? 1 : dft_size.height;
    clStridesIn[0]             = 1;
    clStridesOut[0]             = 1;

    switch(_type)
    {
    case C2C:
        inLayout        = CLFFT_COMPLEX_INTERLEAVED;
        outLayout       = CLFFT_COMPLEX_INTERLEAVED;
        clStridesIn[1]  = src_step / sizeof(std::complex<float>);
        clStridesOut[1] = clStridesIn[1];
        break;
    case R2C:
        inLayout        = CLFFT_REAL;
        outLayout       = CLFFT_HERMITIAN_INTERLEAVED;
        clStridesIn[1]  = src_step / sizeof(float);
        clStridesOut[1] = dst_step / sizeof(std::complex<float>);
        break;
    case C2R:
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

    openCLSafeCall( clAmdFftCreateDefaultPlan( &plHandle, *(cl_context*)getClContextPtr(), dim, clLengthsIn ) );

    openCLSafeCall( clAmdFftSetResultLocation( plHandle, CLFFT_OUTOFPLACE ) );
    openCLSafeCall( clAmdFftSetLayout( plHandle, inLayout, outLayout ) );
    openCLSafeCall( clAmdFftSetPlanBatchSize( plHandle, batchSize ) );

    openCLSafeCall( clAmdFftSetPlanInStride  ( plHandle, dim, clStridesIn ) );
    openCLSafeCall( clAmdFftSetPlanOutStride ( plHandle, dim, clStridesOut ) );
    openCLSafeCall( clAmdFftSetPlanDistance  ( plHandle, clStridesIn[ dim ], clStridesOut[ dim ]) );

    float scale_ = is_scaled_dft ? 1.f / _dft_size.area() : 1.f;
    openCLSafeCall( clAmdFftSetPlanScale  ( plHandle, is_inverse ? CLFFT_BACKWARD : CLFFT_FORWARD, scale_ ) );

    //ready to bake
    openCLSafeCall( clAmdFftBakePlan( plHandle, 1, (cl_command_queue*)getClCommandQueuePtr(), NULL, NULL ) );
}
cv::ocl::FftPlan::~FftPlan()
{
    openCLSafeCall( clAmdFftDestroyPlan( &plHandle ) );
}

cv::ocl::PlanCache::PlanCache()
    : started(false),
      planStore(vector<cv::ocl::FftPlan *>()),
      setupData(NULL)
{
}

cv::ocl::PlanCache::~PlanCache()
{
    fft_teardown();
}

FftPlan* cv::ocl::PlanCache::getPlan(Size _dft_size, int _src_step, int _dst_step, int _flags, FftType _type)
{
    PlanCache& pCache = *PlanCache::getPlanCache();
    vector<FftPlan *>& pStore = pCache.planStore;
    // go through search
    for(size_t i = 0; i < pStore.size(); i ++)
    {
        FftPlan *plan = pStore[i];
        if(
            plan->dft_size.width == _dft_size.width &&
            plan->dft_size.height == _dft_size.height &&
            plan->flags == _flags &&
            plan->src_step == _src_step &&
            plan->dst_step == _dst_step &&
            plan->type == _type
            )
        {
            return plan;
        }
    }
    // no baked plan is found
    FftPlan *newPlan = new FftPlan(_dft_size, _src_step, _dst_step, _flags, _type);
    pStore.push_back(newPlan);
    return newPlan;
}

bool cv::ocl::PlanCache::removePlan(clAmdFftPlanHandle plHandle)
{
    PlanCache& pCache = *PlanCache::getPlanCache();
    vector<FftPlan *>& pStore = pCache.planStore;
    for(size_t i = 0; i < pStore.size(); i ++)
    {
        if(pStore[i]->getPlanHandle() == plHandle)
        {
            pStore.erase(pStore.begin() + i);
            delete pStore[i];
            return true;
        }
    }
    return false;
}

void cv::ocl::dft(const oclMat &src, oclMat &dst, Size dft_size, int flags)
{
    if(dft_size == Size(0, 0))
    {
        dft_size = src.size();
    }
    // check if the given dft size is of optimal dft size
    CV_Assert(dft_size.area() == getOptimalDFTSize(dft_size.area()));

    // the two flags are not compatible
    CV_Assert( !((flags & DFT_SCALE) && (flags & DFT_ROWS)) );

    // similar assertions with cuda module
    CV_Assert(src.type() == CV_32F || src.type() == CV_32FC2);

    //bool is_1d_input    = (src.rows == 1);
    //int is_row_dft        = flags & DFT_ROWS;
    //int is_scaled_dft        = flags & DFT_SCALE;
    int is_inverse = flags & DFT_INVERSE;
    bool is_complex_input = src.channels() == 2;
    bool is_complex_output = !(flags & DFT_REAL_OUTPUT);


    // We don't support real-to-real transform
    CV_Assert(is_complex_input || is_complex_output);
    FftType type = (FftType)(is_complex_input << 0 | is_complex_output << 1);

    switch(type)
    {
    case C2C:
        dst.create(src.rows, src.cols, CV_32FC2);
        break;
    case R2C:
        dst.create(src.rows, src.cols / 2 + 1, CV_32FC2);
        break;
    case C2R:
        CV_Assert(dft_size.width / 2 + 1 == src.cols && dft_size.height == src.rows);
        dst.create(src.rows, dft_size.width, CV_32FC1);
        break;
    default:
        //std::runtime_error("does not support this convertion!");
        cout << "Does not support this convertion!" << endl;
        throw exception();
        break;
    }
    clAmdFftPlanHandle plHandle = PlanCache::getPlan(dft_size, src.step, dst.step, flags, type)->getPlanHandle();

    //get the buffersize
    size_t buffersize = 0;
    openCLSafeCall( clAmdFftGetTmpBufSize(plHandle, &buffersize ) );

    //allocate the intermediate buffer
    // TODO, bind this with the current FftPlan
    cl_mem clMedBuffer = NULL;
    if (buffersize)
    {
        cl_int medstatus;
        clMedBuffer = clCreateBuffer ( *(cl_context*)(src.clCxt->getOpenCLContextPtr()), CL_MEM_READ_WRITE, buffersize, 0, &medstatus);
        openCLSafeCall( medstatus );
    }
    cl_command_queue clq = *(cl_command_queue*)(src.clCxt->getOpenCLCommandQueuePtr());
    openCLSafeCall( clAmdFftEnqueueTransform( plHandle,
        is_inverse ? CLFFT_BACKWARD : CLFFT_FORWARD,
        1,
        &clq,
        0, NULL, NULL,
        (cl_mem *)&src.data, (cl_mem *)&dst.data, clMedBuffer ) );
    openCLSafeCall( clFinish(clq) );
    if(clMedBuffer)
    {
        openCLFree(clMedBuffer);
    }
    //fft_teardown();
}

#endif
