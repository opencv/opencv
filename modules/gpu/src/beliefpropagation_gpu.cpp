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
void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }

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

namespace
{
    class StereoBeliefPropagationImpl
    {
    public:
        StereoBeliefPropagationImpl(StereoBeliefPropagation& rthis_,
                                    GpuMat& u_, GpuMat& d_, GpuMat& l_, GpuMat& r_,
                                    GpuMat& u2_, GpuMat& d2_, GpuMat& l2_, GpuMat& r2_,
                                    vector<GpuMat>& datas_, GpuMat& out_)
            : rthis(rthis_), u(u_), d(d_), l(l_), r(r_), u2(u2_), d2(d2_), l2(l2_), r2(r2_), datas(datas_), out(out_),
              zero(Scalar::all(0)), scale(rthis_.msg_type == CV_32F ? 1.0f : 10.0f)
        {
            CV_DbgAssert(0 < rthis.ndisp && 0 < rthis.iters && 0 < rthis.levels);
            CV_Assert(rthis.msg_type == CV_32F || rthis.msg_type == CV_16S);

            if (rthis.msg_type == CV_16S)
                CV_Assert((1 << (rthis.levels - 1)) * scale * rthis.max_data_term < numeric_limits<short>::max());
        }

        void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp, const cudaStream_t& stream)
        {
            CV_DbgAssert(left.rows == right.rows && left.cols == right.cols && left.type() == right.type());
            CV_Assert(left.type() == CV_8UC1 || left.type() == CV_8UC3);

            rows = left.rows; 
            cols = left.cols;

            int divisor = (int)pow(2.f, rthis.levels - 1.0f);
            int lowest_cols = cols / divisor;
            int lowest_rows = rows / divisor;
            const int min_image_dim_size = 2;
            CV_Assert(min(lowest_cols, lowest_rows) > min_image_dim_size);

            init();

            datas[0].create(rows * rthis.ndisp, cols, rthis.msg_type);

            bp::comp_data(rthis.msg_type, left, right, left.channels(), datas[0], stream);

            calcBP(disp, stream);
        }
        
        void operator()(const GpuMat& data, GpuMat& disp, const cudaStream_t& stream)
        {
            CV_Assert((data.type() == rthis.msg_type) && (data.rows % rthis.ndisp == 0));
            
            rows = data.rows / rthis.ndisp;
            cols = data.cols;
            
            int divisor = (int)pow(2.f, rthis.levels - 1.0f);
            int lowest_cols = cols / divisor;
            int lowest_rows = rows / divisor;
            const int min_image_dim_size = 2;
            CV_Assert(min(lowest_cols, lowest_rows) > min_image_dim_size);

            init();

            datas[0] = data;

            calcBP(disp, stream);
        }
    private:
        void init()
        {
            u.create(rows * rthis.ndisp, cols, rthis.msg_type);
            d.create(rows * rthis.ndisp, cols, rthis.msg_type);
            l.create(rows * rthis.ndisp, cols, rthis.msg_type);
            r.create(rows * rthis.ndisp, cols, rthis.msg_type);

            if (rthis.levels & 1)
            {
                //can clear less area
                u = zero;
                d = zero;
                l = zero;
                r = zero;
            }

            if (rthis.levels > 1)
            {
                int less_rows = (rows + 1) / 2;
                int less_cols = (cols + 1) / 2;

                u2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);
                d2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);
                l2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);
                r2.create(less_rows * rthis.ndisp, less_cols, rthis.msg_type);

                if ((rthis.levels & 1) == 0)
                {
                    u2 = zero;
                    d2 = zero;
                    l2 = zero;
                    r2 = zero;
                }
            }

            bp::load_constants(rthis.ndisp, rthis.max_data_term, scale * rthis.data_weight, scale * rthis.max_disc_term, scale * rthis.disc_single_jump);

            datas.resize(rthis.levels);

            cols_all.resize(rthis.levels);
            rows_all.resize(rthis.levels);

            cols_all[0] = cols;
            rows_all[0] = rows;           
        }

        void calcBP(GpuMat& disp, const cudaStream_t& stream)
        {
            for (int i = 1; i < rthis.levels; ++i)
            {
                cols_all[i] = (cols_all[i-1] + 1) / 2;
                rows_all[i] = (rows_all[i-1] + 1) / 2;

                datas[i].create(rows_all[i] * rthis.ndisp, cols_all[i], rthis.msg_type);

                bp::data_step_down(cols_all[i], rows_all[i], rows_all[i-1], rthis.msg_type, datas[i-1], datas[i], stream);
            }

            DevMem2D mus[] = {u, u2};
            DevMem2D mds[] = {d, d2};
            DevMem2D mrs[] = {r, r2};
            DevMem2D mls[] = {l, l2};

            int mem_idx = (rthis.levels & 1) ? 0 : 1;

            for (int i = rthis.levels - 1; i >= 0; --i)
            {
                // for lower level we have already computed messages by setting to zero
                if (i != rthis.levels - 1)
                    bp::level_up_messages(mem_idx, cols_all[i], rows_all[i], rows_all[i+1], rthis.msg_type, mus, mds, mls, mrs, stream);

                bp::calc_all_iterations(cols_all[i], rows_all[i], rthis.iters, rthis.msg_type, mus[mem_idx], mds[mem_idx], mls[mem_idx], mrs[mem_idx], datas[i], stream);

                mem_idx = (mem_idx + 1) & 1;
            }

            if (disp.empty())
                disp.create(rows, cols, CV_16S);

            out = ((disp.type() == CV_16S) ? disp : (out.create(rows, cols, CV_16S), out));
            out = zero;

            bp::output(rthis.msg_type, u, d, l, r, datas.front(), disp, stream);

            if (disp.type() != CV_16S)
                out.convertTo(disp, disp.type());
        }

        StereoBeliefPropagation& rthis;

        GpuMat& u;
        GpuMat& d;
        GpuMat& l;
        GpuMat& r;

        GpuMat& u2;
        GpuMat& d2;
        GpuMat& l2;
        GpuMat& r2;

        vector<GpuMat>& datas;
        GpuMat& out;

        const Scalar zero;
        const float scale;

        int rows, cols;

        vector<int> cols_all, rows_all;
    };
}

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp)
{
    ::StereoBeliefPropagationImpl impl(*this, u, d, l, r, u2, d2, l2, r2, datas, out);
    impl(left, right, disp, 0);
}

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disp, Stream& stream)
{
    ::StereoBeliefPropagationImpl impl(*this, u, d, l, r, u2, d2, l2, r2, datas, out);
    impl(left, right, disp, StreamAccessor::getStream(stream));
}

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat& data, GpuMat& disp) 
{ 
    ::StereoBeliefPropagationImpl impl(*this, u, d, l, r, u2, d2, l2, r2, datas, out);
    impl(data, disp, 0);
}

void cv::gpu::StereoBeliefPropagation::operator()(const GpuMat& data, GpuMat& disp, Stream& stream) 
{ 
    ::StereoBeliefPropagationImpl impl(*this, u, d, l, r, u2, d2, l2, r2, datas, out);
    impl(data, disp, StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
