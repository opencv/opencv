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
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma, jin@multicorewareinc.com
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

namespace cv
{
    namespace ocl
    {
        typedef struct _contant_struct
        {
            cl_float c_Tb;
            cl_float c_TB;
            cl_float c_Tg;
            cl_float c_varInit;
            cl_float c_varMin;
            cl_float c_varMax;
            cl_float c_tau;
            cl_uchar c_shadowVal;
        }contant_struct;

        cl_mem cl_constants = NULL;
        float c_TB;
    }
}

#if defined _MSC_VER
#define snprintf sprintf_s
#endif

namespace cv { namespace ocl { namespace device
{
    namespace mog
    {
        void mog_ocl(const oclMat& frame, int cn, oclMat& fgmask, oclMat& weight, oclMat& sortKey, oclMat& mean, oclMat& var,
            int nmixtures, float varThreshold, float learningRate, float backgroundRatio, float noiseSigma);

        void getBackgroundImage_ocl(int cn, const oclMat& weight, const oclMat& mean, oclMat& dst, int nmixtures, float backgroundRatio);

        void loadConstants(float Tb, float TB, float Tg, float varInit, float varMin, float varMax, float tau,
                            unsigned char shadowVal);

        void mog2_ocl(const oclMat& frame, int cn, oclMat& fgmask, oclMat& modesUsed, oclMat& weight, oclMat& variance, oclMat& mean,
                      float alphaT, float prune, bool detectShadows, int nmixtures);

        void getBackgroundImage2_ocl(int cn, const oclMat& modesUsed, const oclMat& weight, const oclMat& mean, oclMat& dst, int nmixtures);
    }
}}}

namespace mog
{
    const int defaultNMixtures = 5;
    const int defaultHistory = 200;
    const float defaultBackgroundRatio = 0.7f;
    const float defaultVarThreshold = 2.5f * 2.5f;
    const float defaultNoiseSigma = 30.0f * 0.5f;
    const float defaultInitialWeight = 0.05f;
}
void cv::ocl::BackgroundSubtractor::operator()(const oclMat&, oclMat&, float)
{

}
cv::ocl::BackgroundSubtractor::~BackgroundSubtractor()
{

}

cv::ocl::MOG::MOG(int nmixtures) :
frameSize_(0, 0), frameType_(0), nframes_(0)
{
    nmixtures_ = std::min(nmixtures > 0 ? nmixtures : mog::defaultNMixtures, 8);
    history = mog::defaultHistory;
    varThreshold = mog::defaultVarThreshold;
    backgroundRatio = mog::defaultBackgroundRatio;
    noiseSigma = mog::defaultNoiseSigma;
}

void cv::ocl::MOG::initialize(cv::Size frameSize, int frameType)
{
    CV_Assert(frameType == CV_8UC1 || frameType == CV_8UC3 || frameType == CV_8UC4);

    frameSize_ = frameSize;
    frameType_ = frameType;

    int ch = CV_MAT_CN(frameType);
    int work_ch = ch;

    // for each gaussian mixture of each pixel bg model we store
    // the mixture sort key (w/sum_of_variances), the mixture weight (w),
    // the mean (nchannels values) and
    // the diagonal covariance matrix (another nchannels values)

    weight_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
    sortKey_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
    mean_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC(work_ch));
    var_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC(work_ch));

    weight_.setTo(cv::Scalar::all(0));
    sortKey_.setTo(cv::Scalar::all(0));
    mean_.setTo(cv::Scalar::all(0));
    var_.setTo(cv::Scalar::all(0));

    nframes_ = 0;
}

void cv::ocl::MOG::operator()(const cv::ocl::oclMat& frame, cv::ocl::oclMat& fgmask, float learningRate)
{
    using namespace cv::ocl::device::mog;

    CV_Assert(frame.depth() == CV_8U);

    int ch = frame.oclchannels();
    int work_ch = ch;

    if (nframes_ == 0 || learningRate >= 1.0 || frame.size() != frameSize_ || work_ch != mean_.oclchannels())
        initialize(frame.size(), frame.type());

    fgmask.create(frameSize_, CV_8UC1);

    ++nframes_;
    learningRate = learningRate >= 0.0f && nframes_ > 1 ? learningRate : 1.0f / std::min(nframes_, history);
    CV_Assert(learningRate >= 0.0f);

    mog_ocl(frame, ch, fgmask, weight_, sortKey_, mean_, var_, nmixtures_,
        varThreshold, learningRate, backgroundRatio, noiseSigma);
}

void cv::ocl::MOG::getBackgroundImage(oclMat& backgroundImage) const
{
    using namespace cv::ocl::device::mog;

    backgroundImage.create(frameSize_, frameType_);

    cv::ocl::device::mog::getBackgroundImage_ocl(backgroundImage.oclchannels(), weight_, mean_, backgroundImage, nmixtures_, backgroundRatio);
}

void cv::ocl::MOG::release()
{
    frameSize_ = Size(0, 0);
    frameType_ = 0;
    nframes_ = 0;

    weight_.release();
    sortKey_.release();
    mean_.release();
    var_.release();
    clReleaseMemObject(cl_constants);
}

static void mog_withoutLearning(const oclMat& frame, int cn, oclMat& fgmask, oclMat& weight, oclMat& mean, oclMat& var,
    int nmixtures, float varThreshold, float backgroundRatio)
{
    Context* clCxt = Context::getContext();

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {frame.cols, frame.rows, 1};

    int frame_step = (int)(frame.step/frame.elemSize());
    int fgmask_step = (int)(fgmask.step/fgmask.elemSize());
    int weight_step = (int)(weight.step/weight.elemSize());
    int mean_step = (int)(mean.step/mean.elemSize());
    int var_step = (int)(var.step/var.elemSize());

    int fgmask_offset_y = (int)(fgmask.offset/fgmask.step);
    int fgmask_offset_x = (int)(fgmask.offset%fgmask.step);
    fgmask_offset_x = fgmask_offset_x/(int)fgmask.elemSize();

    int frame_offset_y = (int)(frame.offset/frame.step);
    int frame_offset_x = (int)(frame.offset%frame.step);
    frame_offset_x = frame_offset_x/(int)frame.elemSize();

    char build_option[50];
    if(cn == 1)
    {
        snprintf(build_option, 50, "-D CN1 -D NMIXTURES=%d", nmixtures);
    }else
    {
        snprintf(build_option, 50, "-D NMIXTURES=%d", nmixtures);
    }

    String kernel_name = "mog_withoutLearning_kernel";
    std::vector<std::pair<size_t, const void*> > args;

    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&frame.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&fgmask.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&weight.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&mean.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&var.data));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame.cols));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&weight_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&mean_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&var_step));

    args.push_back(std::make_pair(sizeof(cl_float), (void*)&varThreshold));
    args.push_back(std::make_pair(sizeof(cl_float), (void*)&backgroundRatio));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_offset_y));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_offset_y));

    openCLExecuteKernel(clCxt, &bgfg_mog, kernel_name, global_thread, local_thread, args, -1, -1, build_option);
}


static void mog_withLearning(const oclMat& frame, int cn, oclMat& fgmask_raw, oclMat& weight, oclMat& sortKey, oclMat& mean, oclMat& var,
    int nmixtures, float varThreshold, float backgroundRatio, float learningRate, float minVar)
{
    Context* clCxt = Context::getContext();

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {frame.cols, frame.rows, 1};

    oclMat fgmask(fgmask_raw.size(), CV_32SC1);

    int frame_step = (int)(frame.step/frame.elemSize());
    int fgmask_step = (int)(fgmask.step/fgmask.elemSize());
    int weight_step = (int)(weight.step/weight.elemSize());
    int sortKey_step = (int)(sortKey.step/sortKey.elemSize());
    int mean_step = (int)(mean.step/mean.elemSize());
    int var_step = (int)(var.step/var.elemSize());

    int fgmask_offset_y = (int)(fgmask.offset/fgmask.step);
    int fgmask_offset_x = (int)(fgmask.offset%fgmask.step);
    fgmask_offset_x = fgmask_offset_x/(int)fgmask.elemSize();

    int frame_offset_y = (int)(frame.offset/frame.step);
    int frame_offset_x = (int)(frame.offset%frame.step);
    frame_offset_x = frame_offset_x/(int)frame.elemSize();

    char build_option[50];
    if(cn == 1)
    {
        snprintf(build_option, 50, "-D CN1 -D NMIXTURES=%d", nmixtures);
    }else
    {
        snprintf(build_option, 50, "-D NMIXTURES=%d", nmixtures);
    }

    String kernel_name = "mog_withLearning_kernel";
    std::vector<std::pair<size_t, const void*> > args;

    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&frame.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&fgmask.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&weight.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&sortKey.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&mean.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&var.data));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame.cols));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&weight_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&sortKey_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&mean_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&var_step));

    args.push_back(std::make_pair(sizeof(cl_float), (void*)&varThreshold));
    args.push_back(std::make_pair(sizeof(cl_float), (void*)&backgroundRatio));
    args.push_back(std::make_pair(sizeof(cl_float), (void*)&learningRate));
    args.push_back(std::make_pair(sizeof(cl_float), (void*)&minVar));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_offset_y));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_offset_y));

    openCLExecuteKernel(clCxt, &bgfg_mog, kernel_name, global_thread, local_thread, args, -1, -1, build_option);
    fgmask.convertTo(fgmask, CV_8U);
    fgmask.copyTo(fgmask_raw);
}

void cv::ocl::device::mog::mog_ocl(const oclMat& frame, int cn, oclMat& fgmask, oclMat& weight, oclMat& sortKey, oclMat& mean, oclMat& var,
    int nmixtures, float varThreshold, float learningRate, float backgroundRatio, float noiseSigma)
{
    const float minVar = noiseSigma * noiseSigma;

    if(learningRate > 0.0f)
        mog_withLearning(frame, cn, fgmask, weight, sortKey, mean, var, nmixtures,
                         varThreshold, backgroundRatio, learningRate, minVar);
    else
        mog_withoutLearning(frame, cn, fgmask, weight, mean, var, nmixtures, varThreshold, backgroundRatio);
}

void cv::ocl::device::mog::getBackgroundImage_ocl(int cn, const oclMat& weight, const oclMat& mean, oclMat& dst, int nmixtures, float backgroundRatio)
{
    Context* clCxt = Context::getContext();

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {dst.cols, dst.rows, 1};

    int weight_step = (int)(weight.step/weight.elemSize());
    int mean_step = (int)(mean.step/mean.elemSize());
    int dst_step = (int)(dst.step/dst.elemSize());

    char build_option[50];
    if(cn == 1)
    {
        snprintf(build_option, 50, "-D CN1 -D NMIXTURES=%d", nmixtures);
    }else
    {
        snprintf(build_option, 50, "-D NMIXTURES=%d", nmixtures);
    }

    String kernel_name = "getBackgroundImage_kernel";
    std::vector<std::pair<size_t, const void*> > args;

    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&weight.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&mean.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&dst.data));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&dst.cols));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&weight_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&mean_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&dst_step));

    args.push_back(std::make_pair(sizeof(cl_float), (void*)&backgroundRatio));

    openCLExecuteKernel(clCxt, &bgfg_mog, kernel_name, global_thread, local_thread, args, -1, -1, build_option);
}

void cv::ocl::device::mog::loadConstants(float Tb, float TB, float Tg, float varInit, float varMin, float varMax, float tau, unsigned char shadowVal)
{
    varMin = cv::min(varMin, varMax);
    varMax = cv::max(varMin, varMax);

    c_TB = TB;

    _contant_struct *constants = new _contant_struct;
    constants->c_Tb = Tb;
    constants->c_TB = TB;
    constants->c_Tg = Tg;
    constants->c_varInit = varInit;
    constants->c_varMin = varMin;
    constants->c_varMax = varMax;
    constants->c_tau = tau;
    constants->c_shadowVal = shadowVal;

    cl_constants = load_constant(*((cl_context*)getClContextPtr()), *((cl_command_queue*)getClCommandQueuePtr()),
        (void *)constants, sizeof(_contant_struct));
}

void cv::ocl::device::mog::mog2_ocl(const oclMat& frame, int cn, oclMat& fgmaskRaw, oclMat& modesUsed, oclMat& weight, oclMat& variance,
                                oclMat& mean, float alphaT, float prune, bool detectShadows, int nmixtures)
{
    oclMat fgmask(fgmaskRaw.size(), CV_32SC1);

    Context* clCxt = Context::getContext();

    const float alpha1 = 1.0f - alphaT;

    cl_int detectShadows_flag = 0;
    if(detectShadows)
        detectShadows_flag = 1;

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {frame.cols, frame.rows, 1};

    int frame_step = (int)(frame.step/frame.elemSize());
    int fgmask_step = (int)(fgmask.step/fgmask.elemSize());
    int weight_step = (int)(weight.step/weight.elemSize());
    int modesUsed_step = (int)(modesUsed.step/modesUsed.elemSize());
    int mean_step = (int)(mean.step/mean.elemSize());
    int var_step = (int)(variance.step/variance.elemSize());

    int fgmask_offset_y = (int)(fgmask.offset/fgmask.step);
    int fgmask_offset_x = (int)(fgmask.offset%fgmask.step);
    fgmask_offset_x = fgmask_offset_x/(int)fgmask.elemSize();

    int frame_offset_y = (int)(frame.offset/frame.step);
    int frame_offset_x = (int)(frame.offset%frame.step);
    frame_offset_x = frame_offset_x/(int)frame.elemSize();

    String kernel_name = "mog2_kernel";
    std::vector<std::pair<size_t, const void*> > args;

    char build_option[50];
    if(cn == 1)
    {
        snprintf(build_option, 50, "-D CN1 -D NMIXTURES=%d", nmixtures);
    }else
    {
        snprintf(build_option, 50, "-D NMIXTURES=%d", nmixtures);
    }

    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&frame.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&fgmask.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&weight.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&mean.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&modesUsed.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&variance.data));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame.cols));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&weight_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&mean_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&modesUsed_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&var_step));

    args.push_back(std::make_pair(sizeof(cl_float), (void*)&alphaT));
    args.push_back(std::make_pair(sizeof(cl_float), (void*)&alpha1));
    args.push_back(std::make_pair(sizeof(cl_float), (void*)&prune));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&detectShadows_flag));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&fgmask_offset_y));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&frame_offset_y));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&cl_constants));

    openCLExecuteKernel(clCxt, &bgfg_mog, kernel_name, global_thread, local_thread, args, -1, -1, build_option);

    fgmask.convertTo(fgmask, CV_8U);
    fgmask.copyTo(fgmaskRaw);
}

void cv::ocl::device::mog::getBackgroundImage2_ocl(int cn, const oclMat& modesUsed, const oclMat& weight, const oclMat& mean, oclMat& dst, int nmixtures)
{
    Context* clCxt = Context::getContext();

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {modesUsed.cols, modesUsed.rows, 1};

    int weight_step = (int)(weight.step/weight.elemSize());
    int modesUsed_step = (int)(modesUsed.step/modesUsed.elemSize());
    int mean_step = (int)(mean.step/mean.elemSize());
    int dst_step = (int)(dst.step/dst.elemSize());

    int dst_y = (int)(dst.offset/dst.step);
    int dst_x = (int)(dst.offset%dst.step);
    dst_x = dst_x/(int)dst.elemSize();

    String kernel_name = "getBackgroundImage2_kernel";
    std::vector<std::pair<size_t, const void*> > args;

    char build_option[50];
    if(cn == 1)
    {
        snprintf(build_option, 50, "-D CN1 -D NMIXTURES=%d", nmixtures);
    }else
    {
        snprintf(build_option, 50, "-D NMIXTURES=%d", nmixtures);
    }

    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&modesUsed.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&weight.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&mean.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_float), (void*)&c_TB));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&modesUsed.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&modesUsed.cols));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&modesUsed_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&weight_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&mean_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&dst_step));

    args.push_back(std::make_pair(sizeof(cl_int), (void*)&dst_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void*)&dst_y));

    openCLExecuteKernel(clCxt, &bgfg_mog, kernel_name, global_thread, local_thread, args, -1, -1, build_option);
}

/////////////////////////////////////////////////////////////////
// MOG2

namespace mog2
{
    // default parameters of gaussian background detection algorithm
    const int defaultHistory = 500; // Learning rate; alpha = 1/defaultHistory2
    const float defaultVarThreshold = 4.0f * 4.0f;
    const int defaultNMixtures = 5; // maximal number of Gaussians in mixture
    const float defaultBackgroundRatio = 0.9f; // threshold sum of weights for background test
    const float defaultVarThresholdGen = 3.0f * 3.0f;
    const float defaultVarInit = 15.0f; // initial variance for new components
    const float defaultVarMax = 5.0f * defaultVarInit;
    const float defaultVarMin = 4.0f;

    // additional parameters
    const float defaultfCT = 0.05f; // complexity reduction prior constant 0 - no reduction of number of components
    const unsigned char defaultnShadowDetection = 127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
    const float defaultfTau = 0.5f; // Tau - shadow threshold, see the paper for explanation
}

cv::ocl::MOG2::MOG2(int nmixtures) : frameSize_(0, 0), frameType_(0), nframes_(0)
{
    nmixtures_ = nmixtures > 0 ? nmixtures : mog2::defaultNMixtures;

    history = mog2::defaultHistory;
    varThreshold = mog2::defaultVarThreshold;
    bShadowDetection = true;

    backgroundRatio = mog2::defaultBackgroundRatio;
    fVarInit = mog2::defaultVarInit;
    fVarMax  = mog2::defaultVarMax;
    fVarMin = mog2::defaultVarMin;

    varThresholdGen = mog2::defaultVarThresholdGen;
    fCT = mog2::defaultfCT;
    nShadowDetection =  mog2::defaultnShadowDetection;
    fTau = mog2::defaultfTau;
}

void cv::ocl::MOG2::initialize(cv::Size frameSize, int frameType)
{
    using namespace cv::ocl::device::mog;
    CV_Assert(frameType == CV_8UC1 || frameType == CV_8UC3 || frameType == CV_8UC4);

    frameSize_ = frameSize;
    frameType_ = frameType;
    nframes_ = 0;

    int ch = CV_MAT_CN(frameType);
    int work_ch = ch;

    // for each gaussian mixture of each pixel bg model we store ...
    // the mixture weight (w),
    // the mean (nchannels values) and
    // the covariance
    weight_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
    weight_.setTo(Scalar::all(0));

    variance_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
    variance_.setTo(Scalar::all(0));

    mean_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC(work_ch)); //4 channels
    mean_.setTo(Scalar::all(0));

    //make the array for keeping track of the used modes per pixel - all zeros at start
    bgmodelUsedModes_.create(frameSize_, CV_32FC1);
    bgmodelUsedModes_.setTo(cv::Scalar::all(0));

    loadConstants(varThreshold, backgroundRatio, varThresholdGen, fVarInit, fVarMin, fVarMax, fTau, nShadowDetection);
}

void cv::ocl::MOG2::operator()(const oclMat& frame, oclMat& fgmask, float learningRate)
{
    using namespace cv::ocl::device::mog;

    int ch = frame.oclchannels();
    int work_ch = ch;

    if (nframes_ == 0 || learningRate >= 1.0f || frame.size() != frameSize_ || work_ch != mean_.oclchannels())
        initialize(frame.size(), frame.type());

    fgmask.create(frameSize_, CV_8UC1);
    fgmask.setTo(cv::Scalar::all(0));

    ++nframes_;
    learningRate = learningRate >= 0.0f && nframes_ > 1 ? learningRate : 1.0f / std::min(2 * nframes_, history);
    CV_Assert(learningRate >= 0.0f);

    mog2_ocl(frame, frame.oclchannels(), fgmask, bgmodelUsedModes_, weight_, variance_, mean_, learningRate, -learningRate * fCT, bShadowDetection, nmixtures_);
}

void cv::ocl::MOG2::getBackgroundImage(oclMat& backgroundImage) const
{
    using namespace cv::ocl::device::mog;

    backgroundImage.create(frameSize_, frameType_);

    cv::ocl::device::mog::getBackgroundImage2_ocl(backgroundImage.oclchannels(), bgmodelUsedModes_, weight_, mean_, backgroundImage, nmixtures_);
}

void cv::ocl::MOG2::release()
{
    frameSize_ = Size(0, 0);
    frameType_ = 0;
    nframes_ = 0;

    weight_.release();
    variance_.release();
    mean_.release();

    bgmodelUsedModes_.release();
}
