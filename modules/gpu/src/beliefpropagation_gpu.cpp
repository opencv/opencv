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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
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
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

cv::gpu::StereoBeliefPropagation_GPU::StereoBeliefPropagation_GPU(int, int, int) { throw_nogpu(); }
cv::gpu::StereoBeliefPropagation_GPU::StereoBeliefPropagation_GPU(int, int, int, float, float, float) { throw_nogpu(); }

void cv::gpu::StereoBeliefPropagation_GPU::operator()(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::StereoBeliefPropagation_GPU::operator()(const GpuMat&, const GpuMat&, GpuMat&, const CudaStream&) { throw_nogpu(); }

bool cv::gpu::StereoBeliefPropagation_GPU::checkIfGpuCallReasonable() { throw_nogpu(); return false; }

#else /* !defined (HAVE_CUDA) */

static const float DEFAULT_DISC_COST   = 1.7f;
static const float DEFAULT_DATA_COST   = 10.0f;
static const float DEFAULT_LAMBDA_COST = 0.07f;

typedef DevMem2D_<float> DevMem2Df;

namespace cv { namespace gpu { namespace impl {
    extern "C" void load_constants(int ndisp, float disc_cost, float data_cost, float lambda);
    extern "C" void comp_data_caller(const DevMem2D& l, const DevMem2D& r, DevMem2Df mdata, const cudaStream_t& stream);
    extern "C" void data_down_kernel_caller(int dst_cols, int dst_rows, int src_rows, const DevMem2Df& src, DevMem2Df dst, const cudaStream_t& stream);
    extern "C" void level_up(int dst_idx, int dst_cols, int dst_rows, int src_rows, DevMem2Df* mu, DevMem2Df* md, DevMem2Df* ml, DevMem2Df* mr, const cudaStream_t& stream);
    extern "C" void call_all_iterations(int cols, int rows, int iters, DevMem2Df& u, DevMem2Df& d, DevMem2Df& l, DevMem2Df& r, const DevMem2Df& data, const cudaStream_t& stream);
    extern "C" void output_caller(const DevMem2Df& u, const DevMem2Df& d, const DevMem2Df& l, const DevMem2Df& r, const DevMem2Df& data, DevMem2D disp, const cudaStream_t& stream);
}}}

cv::gpu::StereoBeliefPropagation_GPU::StereoBeliefPropagation_GPU(int ndisp_, int iters_, int levels_)
 : ndisp(ndisp_), iters(iters_), levels(levels_), disc_cost(DEFAULT_DISC_COST), data_cost(DEFAULT_DATA_COST), lambda(DEFAULT_LAMBDA_COST), datas(levels_) 
{
    const int max_supported_ndisp = 1 << (sizeof(unsigned char) * 8);

    CV_Assert(0 < ndisp && ndisp <= max_supported_ndisp);
    CV_Assert(ndisp % 8 == 0);
}

cv::gpu::StereoBeliefPropagation_GPU::StereoBeliefPropagation_GPU(int ndisp_, int iters_, int levels_, float disc_cost_, float data_cost_, float lambda_)
    : ndisp(ndisp_), iters(iters_), levels(levels_), disc_cost(disc_cost_), data_cost(data_cost_), lambda(lambda_), datas(levels_) 
{
    const int max_supported_ndisp = 1 << (sizeof(unsigned char) * 8);

    CV_Assert(0 < ndisp && ndisp <= max_supported_ndisp);
    CV_Assert(ndisp % 8 == 0);
}

static void stereo_bp_gpu_operator(int ndisp, int iters, int levels, float disc_cost, float data_cost, float lambda, 
                                   GpuMat& u, GpuMat& d, GpuMat& l, GpuMat& r, 
                                   GpuMat& u2, GpuMat& d2, GpuMat& l2, GpuMat& r2, 
                                   vector<GpuMat>& datas, 
                                   const GpuMat& left, const GpuMat& right, GpuMat& disp, 
                                   const cudaStream_t& stream)
{
    CV_DbgAssert(left.cols == right.cols && left.rows == right.rows && left.type() == right.type() && left.type() == CV_8U);

    const Scalar zero = Scalar::all(0);

    int rows = left.rows;
    int cols = left.cols;

    int divisor = (int)pow(2.f, levels - 1.0f);
    int lowest_cols = cols / divisor;
    int lowest_rows = rows / divisor;
    const int min_image_dim_size = 20;
    CV_Assert(min(lowest_cols, lowest_rows) > min_image_dim_size);    

    disp.create(rows, cols, CV_8U);

    u.create(rows * ndisp, cols, CV_32F);  
    d.create(rows * ndisp, cols, CV_32F);  
    l.create(rows * ndisp, cols, CV_32F);  
    r.create(rows * ndisp, cols, CV_32F);

    if (levels & 1)
    {
        u = zero; //can clear less area
        d = zero;
        l = zero;
        r = zero;
    }

    if (levels > 1)
    {
        int less_rows = (rows + 1) / 2;
        int less_cols = (cols + 1) / 2;

        u2.create(less_rows * ndisp, less_cols, CV_32F);
        d2.create(less_rows * ndisp, less_cols, CV_32F);
        l2.create(less_rows * ndisp, less_cols, CV_32F);
        r2.create(less_rows * ndisp, less_cols, CV_32F);

        if ((levels & 1) == 0)
        {
            u2 = zero;
            d2 = zero;
            l2 = zero;
            r2 = zero;    
        }
    }       

    impl::load_constants(ndisp, disc_cost, data_cost, lambda);
     
    vector<int> cols_all(levels);
    vector<int> rows_all(levels);
    vector<int> iters_all(levels);

    cols_all[0] = cols;
    rows_all[0] = rows;
    iters_all[0] = iters;

    datas[0].create(rows * ndisp, cols, CV_32F);
    //datas[0] = Scalar(data_cost); //DOTO did in kernel, but not sure if correct

    impl::comp_data_caller(left, right, datas.front(), stream);

    for (int i = 1; i < levels; i++) 
    {
        cols_all[i] = (cols_all[i-1] + 1)/2;
        rows_all[i] = (rows_all[i-1] + 1)/2;

        // this is difference from Felzenszwalb algorithm
        // we reduce iters num for each next level
        iters_all[i] = max(2 * iters_all[i-1] / 3, 1);

        datas[i].create(rows_all[i] * ndisp, cols_all[i], CV_32F);               

        impl::data_down_kernel_caller(cols_all[i], rows_all[i], rows_all[i-1], datas[i-1], datas[i], stream);
    }
    
    DevMem2D_<float> mus[] = {u, u2}; 
    DevMem2D_<float> mds[] = {d, d2};
    DevMem2D_<float> mrs[] = {r, r2}; 
    DevMem2D_<float> mls[] = {l, l2};

    int mem_idx = (levels & 1) ? 0 : 1;

    for (int i = levels - 1; i >= 0; i--) // for lower level we have already computed messages by setting to zero
    {                        
        if (i != levels - 1) 
            impl::level_up(mem_idx, cols_all[i], rows_all[i], rows_all[i+1], mus, mds, mls, mrs, stream);

        impl::call_all_iterations(cols_all[i], rows_all[i], iters_all[i], mus[mem_idx], mds[mem_idx], mls[mem_idx], mrs[mem_idx], datas[i], stream);

        mem_idx = (mem_idx + 1) & 1;
    }

    impl::output_caller(u, d, l, r, datas.front(), disp, stream);
}

void cv::gpu::StereoBeliefPropagation_GPU::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp)
{    
    ::stereo_bp_gpu_operator(ndisp, iters, levels, disc_cost, data_cost, lambda, u, d, l, r, u2, d2, l2, r2, datas, left, right, disp, 0);
}

void cv::gpu::StereoBeliefPropagation_GPU::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp, const CudaStream& stream)
{
    ::stereo_bp_gpu_operator(ndisp, iters, levels, disc_cost, data_cost, lambda, u, d, l, r, u2, d2, l2, r2, datas, left, right, disp, StreamAccessor::getStream(stream));
}

bool cv::gpu::StereoBeliefPropagation_GPU::checkIfGpuCallReasonable()
{
    if (0 == getCudaEnabledDeviceCount())
        return false;

    int device = getDevice();

    int minor, major;
    getComputeCapability(device, &major, &minor);
    int numSM = getNumberOfSMs(device);

    if (major > 1 || numSM > 16)
        return true;

    return false;
}

#endif /* !defined (HAVE_CUDA) */
