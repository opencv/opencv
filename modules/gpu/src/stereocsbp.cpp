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

void cv::gpu::StereoConstantSpaceBP::estimateRecommendedParams(int, int, int&, int&, int&, int&) { throw_nogpu(); }

cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int, int, int, int, int) { throw_nogpu(); }
cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int, int, int, int, float, float, float, float, int, int) { throw_nogpu(); }

void cv::gpu::StereoConstantSpaceBP::operator()(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

BEGIN_OPENCV_DEVICE_NAMESPACE

namespace stereocsbp
{
    void load_constants(int ndisp, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump, int min_disp_th,
        const DevMem2Db& left, const DevMem2Db& right, const DevMem2Db& temp);

    template<class T>
    void init_data_cost(int rows, int cols, T* disp_selected_pyr, T* data_cost_selected, size_t msg_step,
                int h, int w, int level, int nr_plane, int ndisp, int channels, bool use_local_init_data_cost, cudaStream_t stream);

    template<class T>
    void compute_data_cost(const T* disp_selected_pyr, T* data_cost, size_t msg_step1, size_t msg_step2,
                           int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, cudaStream_t stream);

    template<class T>
    void init_message(T* u_new, T* d_new, T* l_new, T* r_new,
                      const T* u_cur, const T* d_cur, const T* l_cur, const T* r_cur,
                      T* selected_disp_pyr_new, const T* selected_disp_pyr_cur,
                      T* data_cost_selected, const T* data_cost, size_t msg_step1, size_t msg_step2,
                      int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream);

    template<class T>
    void calc_all_iterations(T* u, T* d, T* l, T* r, const T* data_cost_selected,
        const T* selected_disp_pyr_cur, size_t msg_step, int h, int w, int nr_plane, int iters, cudaStream_t stream);

    template<class T> 
    void compute_disp(const T* u, const T* d, const T* l, const T* r, const T* data_cost_selected, const T* disp_selected, size_t msg_step,
        const DevMem2D_<short>& disp, int nr_plane, cudaStream_t stream);
}

END_OPENCV_DEVICE_NAMESPACE

using namespace OPENCV_DEVICE_NAMESPACE_ stereocsbp;

namespace
{
    const float DEFAULT_MAX_DATA_TERM = 30.0f;
    const float DEFAULT_DATA_WEIGHT = 1.0f;
    const float DEFAULT_MAX_DISC_TERM = 160.0f;
    const float DEFAULT_DISC_SINGLE_JUMP = 10.0f;
}

void cv::gpu::StereoConstantSpaceBP::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane)
{
    ndisp = (int) ((float) width / 3.14f);
    if ((ndisp & 1) != 0) 
        ndisp++;

    int mm = ::max(width, height);
    iters = mm / 100 + ((mm > 1200)? - 4 : 4);

    levels = (int)::log(static_cast<double>(mm)) * 2 / 3;
    if (levels == 0) levels++;

    nr_plane = (int) ((float) ndisp / std::pow(2.0, levels + 1));
}

cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp_, int iters_, int levels_, int nr_plane_,
                                                      int msg_type_)

    : ndisp(ndisp_), iters(iters_), levels(levels_), nr_plane(nr_plane_),
      max_data_term(DEFAULT_MAX_DATA_TERM), data_weight(DEFAULT_DATA_WEIGHT),
      max_disc_term(DEFAULT_MAX_DISC_TERM), disc_single_jump(DEFAULT_DISC_SINGLE_JUMP), min_disp_th(0),
      msg_type(msg_type_), use_local_init_data_cost(true)
{
    CV_Assert(msg_type_ == CV_32F || msg_type_ == CV_16S);
}

cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp_, int iters_, int levels_, int nr_plane_,
                                                      float max_data_term_, float data_weight_, float max_disc_term_, float disc_single_jump_,
                                                      int min_disp_th_, int msg_type_)
    : ndisp(ndisp_), iters(iters_), levels(levels_), nr_plane(nr_plane_),
      max_data_term(max_data_term_), data_weight(data_weight_),
      max_disc_term(max_disc_term_), disc_single_jump(disc_single_jump_), min_disp_th(min_disp_th_),
      msg_type(msg_type_), use_local_init_data_cost(true)
{
    CV_Assert(msg_type_ == CV_32F || msg_type_ == CV_16S);
}

template<class T>
static void csbp_operator(StereoConstantSpaceBP& rthis, GpuMat u[2], GpuMat d[2], GpuMat l[2], GpuMat r[2],
                          GpuMat disp_selected_pyr[2], GpuMat& data_cost, GpuMat& data_cost_selected,
                          GpuMat& temp, GpuMat& out, const GpuMat& left, const GpuMat& right, GpuMat& disp, Stream& stream)
{
    CV_DbgAssert(0 < rthis.ndisp && 0 < rthis.iters && 0 < rthis.levels && 0 < rthis.nr_plane
        && left.rows == right.rows && left.cols == right.cols && left.type() == right.type());

    CV_Assert(rthis.levels <= 8 && (left.type() == CV_8UC1 || left.type() == CV_8UC3 || left.type() == CV_8UC4));

    const Scalar zero = Scalar::all(0);

    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Init

    int rows = left.rows;
    int cols = left.cols;

    rthis.levels = min(rthis.levels, int(log((double)rthis.ndisp) / log(2.0)));
    int levels = rthis.levels;

    AutoBuffer<int> buf(levels * 4);

    int* cols_pyr = buf;
    int* rows_pyr = cols_pyr + levels;
    int* nr_plane_pyr = rows_pyr + levels;
    int* step_pyr = nr_plane_pyr + levels;

    cols_pyr[0] = cols;
    rows_pyr[0] = rows;
    nr_plane_pyr[0] = rthis.nr_plane;

    const int n = 64;
    step_pyr[0] = static_cast<int>(alignSize(cols * sizeof(T), n) / sizeof(T));
    for (int i = 1; i < levels; i++)
    {
        cols_pyr[i] = (cols_pyr[i-1] + 1) / 2;
        rows_pyr[i] = (rows_pyr[i-1] + 1) / 2;

        nr_plane_pyr[i] = nr_plane_pyr[i-1] * 2;

        step_pyr[i] = static_cast<int>(alignSize(cols_pyr[i] * sizeof(T), n) / sizeof(T));
    }

    Size msg_size(step_pyr[0], rows * nr_plane_pyr[0]);
    Size data_cost_size(step_pyr[0], rows * nr_plane_pyr[0] * 2);

    u[0].create(msg_size, DataType<T>::type);
    d[0].create(msg_size, DataType<T>::type);
    l[0].create(msg_size, DataType<T>::type);
    r[0].create(msg_size, DataType<T>::type);

    u[1].create(msg_size, DataType<T>::type);
    d[1].create(msg_size, DataType<T>::type);
    l[1].create(msg_size, DataType<T>::type);
    r[1].create(msg_size, DataType<T>::type);

    disp_selected_pyr[0].create(msg_size, DataType<T>::type);
    disp_selected_pyr[1].create(msg_size, DataType<T>::type);

    data_cost.create(data_cost_size, DataType<T>::type);
    data_cost_selected.create(msg_size, DataType<T>::type);

    step_pyr[0] = static_cast<int>(data_cost.step / sizeof(T));

    Size temp_size = data_cost_size;
    if (data_cost_size.width * data_cost_size.height < step_pyr[levels - 1] * rows_pyr[levels - 1] * rthis.ndisp)
        temp_size = Size(step_pyr[levels - 1], rows_pyr[levels - 1] * rthis.ndisp);

    temp.create(temp_size, DataType<T>::type);

    ////////////////////////////////////////////////////////////////////////////
    // Compute

    load_constants(rthis.ndisp, rthis.max_data_term, rthis.data_weight, rthis.max_disc_term, rthis.disc_single_jump, rthis.min_disp_th, left, right, temp);

    if (stream)
    {
        stream.enqueueMemSet(l[0], zero);
        stream.enqueueMemSet(d[0], zero);
        stream.enqueueMemSet(r[0], zero);
        stream.enqueueMemSet(u[0], zero);
        
        stream.enqueueMemSet(l[1], zero);
        stream.enqueueMemSet(d[1], zero);
        stream.enqueueMemSet(r[1], zero);
        stream.enqueueMemSet(u[1], zero);

        stream.enqueueMemSet(data_cost, zero);
        stream.enqueueMemSet(data_cost_selected, zero);
    }
    else
    {
        l[0].setTo(zero);
        d[0].setTo(zero);
        r[0].setTo(zero);
        u[0].setTo(zero);

        l[1].setTo(zero);
        d[1].setTo(zero);
        r[1].setTo(zero);
        u[1].setTo(zero);

        data_cost.setTo(zero);
        data_cost_selected.setTo(zero);
    }

    int cur_idx = 0;

    for (int i = levels - 1; i >= 0; i--)
    {
        if (i == levels - 1)
        {
            init_data_cost(left.rows, left.cols, disp_selected_pyr[cur_idx].ptr<T>(), data_cost_selected.ptr<T>(),
                step_pyr[i], rows_pyr[i], cols_pyr[i], i, nr_plane_pyr[i], rthis.ndisp, left.channels(), rthis.use_local_init_data_cost, cudaStream);
        }
        else
        {
            compute_data_cost(disp_selected_pyr[cur_idx].ptr<T>(), data_cost.ptr<T>(), step_pyr[i], step_pyr[i+1],
                left.rows, left.cols, rows_pyr[i], cols_pyr[i], rows_pyr[i+1], i, nr_plane_pyr[i+1], left.channels(), cudaStream);

            int new_idx = (cur_idx + 1) & 1;

            init_message(u[new_idx].ptr<T>(), d[new_idx].ptr<T>(), l[new_idx].ptr<T>(), r[new_idx].ptr<T>(),
                         u[cur_idx].ptr<T>(), d[cur_idx].ptr<T>(), l[cur_idx].ptr<T>(), r[cur_idx].ptr<T>(),
                         disp_selected_pyr[new_idx].ptr<T>(), disp_selected_pyr[cur_idx].ptr<T>(),
                         data_cost_selected.ptr<T>(), data_cost.ptr<T>(), step_pyr[i], step_pyr[i+1], rows_pyr[i],
                         cols_pyr[i], nr_plane_pyr[i], rows_pyr[i+1], cols_pyr[i+1], nr_plane_pyr[i+1], cudaStream);

            cur_idx = new_idx;
        }

        calc_all_iterations(u[cur_idx].ptr<T>(), d[cur_idx].ptr<T>(), l[cur_idx].ptr<T>(), r[cur_idx].ptr<T>(),
                            data_cost_selected.ptr<T>(), disp_selected_pyr[cur_idx].ptr<T>(), step_pyr[i],
                            rows_pyr[i], cols_pyr[i], nr_plane_pyr[i], rthis.iters, cudaStream);
    }

    if (disp.empty())
        disp.create(rows, cols, CV_16S);

    out = ((disp.type() == CV_16S) ? disp : (out.create(rows, cols, CV_16S), out));

    if (stream)
        stream.enqueueMemSet(out, zero);
    else
        out.setTo(zero);

    compute_disp(u[cur_idx].ptr<T>(), d[cur_idx].ptr<T>(), l[cur_idx].ptr<T>(), r[cur_idx].ptr<T>(),
                 data_cost_selected.ptr<T>(), disp_selected_pyr[cur_idx].ptr<T>(), step_pyr[0], out, nr_plane_pyr[0], cudaStream);

    if (disp.type() != CV_16S)
    {
        if (stream)
            stream.enqueueConvert(out, disp, disp.type());
        else
            out.convertTo(disp, disp.type());
    }
}


typedef void (*csbp_operator_t)(StereoConstantSpaceBP& rthis, GpuMat u[2], GpuMat d[2], GpuMat l[2], GpuMat r[2],
                                     GpuMat disp_selected_pyr[2], GpuMat& data_cost, GpuMat& data_cost_selected,
                                     GpuMat& temp, GpuMat& out, const GpuMat& left, const GpuMat& right, GpuMat& disp, Stream& stream);

const static csbp_operator_t operators[] = {0, 0, 0, csbp_operator<short>, 0, csbp_operator<float>, 0, 0};

void cv::gpu::StereoConstantSpaceBP::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp, Stream& stream)
{
    CV_Assert(msg_type == CV_32F || msg_type == CV_16S);
    operators[msg_type](*this, u, d, l, r, disp_selected_pyr, data_cost, data_cost_selected, temp, out, left, right, disp, stream);
}

#endif /* !defined (HAVE_CUDA) */
