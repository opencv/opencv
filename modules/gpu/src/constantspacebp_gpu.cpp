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

cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int, int, int, int, int) { throw_nogpu(); }
cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int, int, int, int, float, float, float, float, int) { throw_nogpu(); }

void cv::gpu::StereoConstantSpaceBP::operator()(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::StereoConstantSpaceBP::operator()(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace csbp 
{   
    void load_constants(int ndisp, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump, 
                        const DevMem2D& left, const DevMem2D& right, const DevMem2D& temp/*, const DevMem2D& temp2*/);

    void init_data_cost(int rows, int cols, const DevMem2D& disp_selected_pyr, const DevMem2D& data_cost_selected,
                        size_t msg_step, int msg_type, int h, int w, int level, int nr_plane, int ndisp, int channels, 
                        const cudaStream_t& stream);
    
    void compute_data_cost(const DevMem2D& disp_selected_pyr, const DevMem2D& data_cost, size_t msg_step1, size_t msg_step2, int msg_type,
                           int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, const cudaStream_t& stream);

    void init_message(const DevMem2D& u_new, const DevMem2D& d_new, const DevMem2D& l_new, const DevMem2D& r_new, 
                      const DevMem2D& u_cur, const DevMem2D& d_cur, const DevMem2D& l_cur, const DevMem2D& r_cur, 
                      const DevMem2D& selected_disp_pyr_new, const DevMem2D& selected_disp_pyr_cur, 
                      const DevMem2D& data_cost_selected, const DevMem2D& data_cost, size_t msg_step1, size_t msg_step2, int msg_type, 
                      int h, int w, int nr_plane, int h2, int w2, int nr_plane2, const cudaStream_t& stream);

    void calc_all_iterations(const DevMem2D& u, const DevMem2D& d, const DevMem2D& l, const DevMem2D& r, const DevMem2D& data_cost_selected, 
                             const DevMem2D& selected_disp_pyr_cur, size_t msg_step, int msg_type, int h, int w, int nr_plane, int iters, 
                             const cudaStream_t& stream);

    void compute_disp(const DevMem2D& u, const DevMem2D& d, const DevMem2D& l, const DevMem2D& r, const DevMem2D& data_cost_selected, 
                      const DevMem2D& disp_selected, size_t msg_step, int msg_type, const DevMem2D& disp, int nr_plane, 
                      const cudaStream_t& stream);

}}}

namespace
{
    const float DEFAULT_MAX_DATA_TERM = 10.0f;
    const float DEFAULT_DATA_WEIGHT = 0.07f;
    const float DEFAULT_MAX_DISC_TERM = 1.7f;
    const float DEFAULT_DISC_SINGLE_JUMP = 1.0f;
}

cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp_, int iters_, int levels_, int nr_plane_,
                                                      int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_), nr_plane(nr_plane_), 
      max_data_term(DEFAULT_MAX_DATA_TERM), data_weight(DEFAULT_DATA_WEIGHT), 
      max_disc_term(DEFAULT_MAX_DISC_TERM), disc_single_jump(DEFAULT_DISC_SINGLE_JUMP),
      msg_type(msg_type_)
{  
}

cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp_, int iters_, int levels_, int nr_plane_,
                                                      float max_data_term_, float data_weight_, float max_disc_term_, float disc_single_jump_,
                                                      int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_), nr_plane(nr_plane_), 
      max_data_term(max_data_term_), data_weight(data_weight_), 
      max_disc_term(max_disc_term_), disc_single_jump(disc_single_jump_),
      msg_type(msg_type_)
{   
}

static void stereo_csbp_gpu_operator(int& ndisp, int& iters, int& levels, int& nr_plane, 
                                     float& max_data_term, float& data_weight, float& max_disc_term, float& disc_single_jump,
                                     int& msg_type,
                                     GpuMat u[2], GpuMat d[2], GpuMat l[2], GpuMat r[2],
                                     GpuMat disp_selected_pyr[2], GpuMat& data_cost, GpuMat& data_cost_selected,
                                     GpuMat& temp, GpuMat& out,
                                     const GpuMat& left, const GpuMat& right, GpuMat& disp,
                                     const cudaStream_t& stream)
{
    CV_DbgAssert(0 < ndisp && 0 < iters && 0 < levels && 0 < nr_plane
        && (msg_type == CV_32F || msg_type == CV_16S)
        && left.rows == right.rows && left.cols == right.cols && left.type() == right.type());

    CV_Assert(levels <= 8 && (left.type() == CV_8UC1 || left.type() == CV_8UC3));    

    const Scalar zero = Scalar::all(0);

    const float scale = ((msg_type == CV_32F) ? 1.0f : 10.0f);

    const size_t type_size = ((msg_type == CV_32F) ? sizeof(float) : sizeof(short));

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Init

    int rows = left.rows;
    int cols = left.cols;

    levels = min(levels, int(log((double)ndisp) / log(2.0)));

    AutoBuffer<int> buf(levels * 4);

    int* cols_pyr = buf;
    int* rows_pyr = cols_pyr + levels;
    int* nr_plane_pyr = rows_pyr + levels;
    int* step_pyr = nr_plane_pyr + levels;

    cols_pyr[0] = cols;
    rows_pyr[0] = rows;
    nr_plane_pyr[0] = nr_plane;

    const int n = 64;
    step_pyr[0] = alignSize(cols * type_size, n) / type_size;
    for (int i = 1; i < levels; i++)
    {
        cols_pyr[i] = (cols_pyr[i-1] + 1) / 2;
        rows_pyr[i] = (rows_pyr[i-1] + 1) / 2;

        nr_plane_pyr[i] = nr_plane_pyr[i-1] * 2;

        step_pyr[i] = alignSize(cols_pyr[i] * type_size, n) / type_size;
    }

    Size msg_size(step_pyr[0], rows * nr_plane_pyr[0]);
    Size data_cost_size(step_pyr[0], rows * nr_plane_pyr[0] * 2);

    u[0].create(msg_size, msg_type);
    d[0].create(msg_size, msg_type);
    l[0].create(msg_size, msg_type);
    r[0].create(msg_size, msg_type);
    
    u[1].create(msg_size, msg_type);
    d[1].create(msg_size, msg_type);
    l[1].create(msg_size, msg_type);
    r[1].create(msg_size, msg_type);
    
    disp_selected_pyr[0].create(msg_size, msg_type);    
    disp_selected_pyr[1].create(msg_size, msg_type);

    data_cost.create(data_cost_size, msg_type);
    data_cost_selected.create(msg_size, msg_type);

    step_pyr[0] = data_cost.step / type_size;

    Size temp_size = data_cost_size;
    if (data_cost_size.width * data_cost_size.height < static_cast<size_t>(step_pyr[levels - 1]) * rows_pyr[levels - 1] * ndisp)
    {
        temp_size = Size(step_pyr[levels - 1], rows_pyr[levels - 1] * ndisp);
    }

    temp.create(temp_size, msg_type);

    ////////////////////////////////////////////////////////////////////////////
    // Compute

    csbp::load_constants(ndisp, max_data_term, scale * data_weight, scale * max_disc_term, scale * disc_single_jump, 
        left, right, temp);

    l[0] = zero;
    d[0] = zero;
    r[0] = zero;
    u[0] = zero;

    l[1] = zero;
    d[1] = zero;
    r[1] = zero;
    u[1] = zero;

    data_cost = zero;
    data_cost_selected = zero;

    int cur_idx = 0;

    for (int i = levels - 1; i >= 0; i--)
    {
        if (i == levels - 1)
        {
            csbp::init_data_cost(left.rows, left.cols, disp_selected_pyr[cur_idx], data_cost_selected, 
                step_pyr[i], msg_type, rows_pyr[i], cols_pyr[i], i, nr_plane_pyr[i], ndisp, left.channels(), stream);
        }
        else
        {
            csbp::compute_data_cost(disp_selected_pyr[cur_idx], data_cost, step_pyr[i], step_pyr[i+1], msg_type, 
                left.rows, left.cols, rows_pyr[i], cols_pyr[i], rows_pyr[i+1], i, nr_plane_pyr[i+1], left.channels(), stream);

            int new_idx = (cur_idx + 1) & 1;

            csbp::init_message(u[new_idx], d[new_idx], l[new_idx], r[new_idx],
                               u[cur_idx], d[cur_idx], l[cur_idx], r[cur_idx],
                               disp_selected_pyr[new_idx], disp_selected_pyr[cur_idx],
                               data_cost_selected, data_cost, step_pyr[i], step_pyr[i+1], msg_type, 
                               rows_pyr[i], cols_pyr[i], nr_plane_pyr[i],
                               rows_pyr[i+1], cols_pyr[i+1], nr_plane_pyr[i+1], stream);

            cur_idx = new_idx;
        }

        csbp::calc_all_iterations(u[cur_idx], d[cur_idx], l[cur_idx], r[cur_idx], 
                                  data_cost_selected, disp_selected_pyr[cur_idx], step_pyr[i], msg_type, 
                                  rows_pyr[i], cols_pyr[i], nr_plane_pyr[i], iters, stream);
    }

    if (disp.empty())
        disp.create(rows, cols, CV_16S);

    out = ((disp.type() == CV_16S) ? disp : GpuMat(rows, cols, CV_16S));
    out = zero;
    
    csbp::compute_disp(u[cur_idx], d[cur_idx], l[cur_idx], r[cur_idx], 
                       data_cost_selected, disp_selected_pyr[cur_idx], step_pyr[0], msg_type, out, nr_plane_pyr[0], stream);

    if (disp.type() != CV_16S)
        out.convertTo(disp, disp.type());
}

void cv::gpu::StereoConstantSpaceBP::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp)
{
    ::stereo_csbp_gpu_operator(ndisp, iters, levels, nr_plane, max_data_term, data_weight, max_disc_term, disc_single_jump, msg_type,
                               u, d, l, r, disp_selected_pyr, data_cost, data_cost_selected, temp/*, temp2*/, out, left, right, disp, 0);
}

void cv::gpu::StereoConstantSpaceBP::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp, const Stream& stream)
{
    ::stereo_csbp_gpu_operator(ndisp, iters, levels, nr_plane, max_data_term, data_weight, max_disc_term, disc_single_jump, msg_type,
                               u, d, l, r, disp_selected_pyr, data_cost, data_cost_selected, temp/*, temp2*/, out, left, right, disp, 
                               StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
