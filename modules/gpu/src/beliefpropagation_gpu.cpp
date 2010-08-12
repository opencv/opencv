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
// any express or bpied warranties, including, but not limited to, the bpied
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

cv::gpu::StereoBeliefPropagation::StereoBeliefPropagation(int, int, int, int) { throw_nogpu(); }
cv::gpu::StereoBeliefPropagation::StereoBeliefPropagation(int, int, int, float, float, float, float, int) { throw_nogpu(); }

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace bp 
{
    void load_constants(int ndisp, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump);
    void comp_data(int msg_type, const DevMem2D& l, const DevMem2D& r, int channels, DevMem2D mdata, const cudaStream_t& stream);
    void data_step_down(int dst_cols, int dst_rows, int src_rows, int msg_type, const DevMem2D& src, DevMem2D dst, const cudaStream_t& stream);
    void level_up_messages(int dst_idx, int dst_cols, int dst_rows, int src_rows, int msg_type, DevMem2D* mus, DevMem2D* mds, DevMem2D* mls, DevMem2D* mrs, const cudaStream_t& stream);
    void calc_all_iterations(int cols, int rows, int iters, int msg_type, DevMem2D& u, DevMem2D& d, DevMem2D& l, DevMem2D& r, const DevMem2D& data, const cudaStream_t& stream);
    void output(int msg_type, const DevMem2D& u, const DevMem2D& d, const DevMem2D& l, const DevMem2D& r, const DevMem2D& data, DevMem2D disp, const cudaStream_t& stream);
}}}

namespace
{
    const float DEFAULT_MAX_DATA_TERM = 10.0f;
    const float DEFAULT_DATA_WEIGHT = 0.07f;
    const float DEFAULT_MAX_DISC_TERM = 1.7f;
    const float DEFAULT_DISC_SINGLE_JUMP = 1.0f;
}

cv::gpu::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp_, int iters_, int levels_, int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_), 
      max_data_term(DEFAULT_MAX_DATA_TERM), data_weight(DEFAULT_DATA_WEIGHT), 
      max_disc_term(DEFAULT_MAX_DISC_TERM), disc_single_jump(DEFAULT_DISC_SINGLE_JUMP), 
      msg_type(msg_type_), datas(levels_)
{
}

cv::gpu::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp_, int iters_, int levels_, float max_data_term_, float data_weight_, float max_disc_term_, float disc_single_jump_, int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_), 
      max_data_term(max_data_term_), data_weight(data_weight_), 
      max_disc_term(max_disc_term_), disc_single_jump(disc_single_jump_), 
      msg_type(msg_type_), datas(levels_)
{
}

static void stereo_bp_gpu_operator(int& ndisp, int& iters, int& levels, 
                                   float& max_data_term, float& data_weight, float& max_disc_term, float& disc_single_jump,
                                   int& msg_type,
                                   GpuMat& u, GpuMat& d, GpuMat& l, GpuMat& r,
                                   GpuMat& u2, GpuMat& d2, GpuMat& l2, GpuMat& r2,
                                   vector<GpuMat>& datas, GpuMat& out,
                                   const GpuMat& left, const GpuMat& right, GpuMat& disp,
                                   const cudaStream_t& stream)
{
    CV_DbgAssert(0 < ndisp && 0 < iters && 0 < levels
        && (msg_type == CV_32F || msg_type == CV_16S)
        && left.rows == right.rows && left.cols == right.cols && left.type() == right.type());

    CV_Assert((left.type() == CV_8UC1 || left.type() == CV_8UC3));

    const Scalar zero = Scalar::all(0);

    const float scale = ((msg_type == CV_32F) ? 1.0f : 10.0f);

    int rows = left.rows;
    int cols = left.cols;

    int divisor = (int)pow(2.f, levels - 1.0f);
    int lowest_cols = cols / divisor;
    int lowest_rows = rows / divisor;
    const int min_image_dim_size = 2;
    CV_Assert(min(lowest_cols, lowest_rows) > min_image_dim_size);    

    u.create(rows * ndisp, cols, msg_type);
    d.create(rows * ndisp, cols, msg_type);
    l.create(rows * ndisp, cols, msg_type);
    r.create(rows * ndisp, cols, msg_type);

    if (levels & 1)
    {
        //can clear less area
        u = zero;
        d = zero;
        l = zero;
        r = zero;
    }

    if (levels > 1)
    {
        int less_rows = (rows + 1) / 2;
        int less_cols = (cols + 1) / 2;

        u2.create(less_rows * ndisp, less_cols, msg_type);
        d2.create(less_rows * ndisp, less_cols, msg_type);
        l2.create(less_rows * ndisp, less_cols, msg_type);
        r2.create(less_rows * ndisp, less_cols, msg_type);

        if ((levels & 1) == 0)
        {
            u2 = zero;
            d2 = zero;
            l2 = zero;
            r2 = zero;
        }
    }

    bp::load_constants(ndisp, max_data_term, scale * data_weight, scale * max_disc_term, scale * disc_single_jump);

    datas.resize(levels);

    AutoBuffer<int> buf(levels << 1);

    int* cols_all = buf;
    int* rows_all = cols_all + levels;

    cols_all[0] = cols;
    rows_all[0] = rows;

    datas[0].create(rows * ndisp, cols, msg_type);

    bp::comp_data(msg_type, left, right, left.channels(), datas.front(), stream);

    for (int i = 1; i < levels; i++)
    {
        cols_all[i] = (cols_all[i-1] + 1) / 2;
        rows_all[i] = (rows_all[i-1] + 1) / 2;

        datas[i].create(rows_all[i] * ndisp, cols_all[i], msg_type);

        bp::data_step_down(cols_all[i], rows_all[i], rows_all[i-1], msg_type, datas[i-1], datas[i], stream);
    }

    DevMem2D mus[] = {u, u2};
    DevMem2D mds[] = {d, d2};
    DevMem2D mrs[] = {r, r2};
    DevMem2D mls[] = {l, l2};

    int mem_idx = (levels & 1) ? 0 : 1;

    for (int i = levels - 1; i >= 0; i--)
    {
        // for lower level we have already computed messages by setting to zero
        if (i != levels - 1)
            bp::level_up_messages(mem_idx, cols_all[i], rows_all[i], rows_all[i+1], msg_type, mus, mds, mls, mrs, stream);

        bp::calc_all_iterations(cols_all[i], rows_all[i], iters, msg_type, mus[mem_idx], mds[mem_idx], mls[mem_idx], mrs[mem_idx], datas[i], stream);

        mem_idx = (mem_idx + 1) & 1;
    }

    if (disp.empty())
        disp.create(rows, cols, CV_16S);

    out = ((disp.type() == CV_16S) ? disp : GpuMat(rows, cols, CV_16S));
    out = zero;
    
    bp::output(msg_type, u, d, l, r, datas.front(), disp, stream);

    if (disp.type() != CV_16S)
        out.convertTo(disp, disp.type());
}

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp)
{
    ::stereo_bp_gpu_operator(ndisp, iters, levels, max_data_term, data_weight, max_disc_term, disc_single_jump, msg_type, u, d, l, r, u2, d2, l2, r2, datas, out, left, right, disp, 0);
}

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp, const Stream& stream)
{
    ::stereo_bp_gpu_operator(ndisp, iters, levels, max_data_term, data_weight, max_disc_term, disc_single_jump, msg_type, u, d, l, r, u2, d2, l2, r2, datas, out, left, right, disp, StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
