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
//     and/or other materials provided with the distribution.
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
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::StereoConstantSpaceBP::estimateRecommendedParams(int, int, int&, int&, int&, int&) { throw_no_cuda(); }

Ptr<cuda::StereoConstantSpaceBP> cv::cuda::createStereoConstantSpaceBP(int, int, int, int, int) { throw_no_cuda(); return Ptr<cuda::StereoConstantSpaceBP>(); }

#else /* !defined (HAVE_CUDA) */

#include "cuda/stereocsbp.hpp"

namespace
{
    class StereoCSBPImpl : public cuda::StereoConstantSpaceBP
    {
    public:
        StereoCSBPImpl(int ndisp, int iters, int levels, int nr_plane, int msg_type);

        void compute(InputArray left, InputArray right, OutputArray disparity);
        void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream);
        void compute(InputArray data, OutputArray disparity, Stream& stream);

        int getMinDisparity() const { return min_disp_th_; }
        void setMinDisparity(int minDisparity) { min_disp_th_ = minDisparity; }

        int getNumDisparities() const { return ndisp_; }
        void setNumDisparities(int numDisparities) { ndisp_ = numDisparities; }

        int getBlockSize() const { return 0; }
        void setBlockSize(int /*blockSize*/) {}

        int getSpeckleWindowSize() const { return 0; }
        void setSpeckleWindowSize(int /*speckleWindowSize*/) {}

        int getSpeckleRange() const { return 0; }
        void setSpeckleRange(int /*speckleRange*/) {}

        int getDisp12MaxDiff() const { return 0; }
        void setDisp12MaxDiff(int /*disp12MaxDiff*/) {}

        int getNumIters() const { return iters_; }
        void setNumIters(int iters) { iters_ = iters; }

        int getNumLevels() const { return levels_; }
        void setNumLevels(int levels) { levels_ = levels; }

        double getMaxDataTerm() const { return max_data_term_; }
        void setMaxDataTerm(double max_data_term) { max_data_term_ = (float) max_data_term; }

        double getDataWeight() const { return data_weight_; }
        void setDataWeight(double data_weight) { data_weight_ = (float) data_weight; }

        double getMaxDiscTerm() const { return max_disc_term_; }
        void setMaxDiscTerm(double max_disc_term) { max_disc_term_ = (float) max_disc_term; }

        double getDiscSingleJump() const { return disc_single_jump_; }
        void setDiscSingleJump(double disc_single_jump) { disc_single_jump_ = (float) disc_single_jump; }

        int getMsgType() const { return msg_type_; }
        void setMsgType(int msg_type) { msg_type_ = msg_type; }

        int getNrPlane() const { return nr_plane_; }
        void setNrPlane(int nr_plane) { nr_plane_ = nr_plane; }

        bool getUseLocalInitDataCost() const { return use_local_init_data_cost_; }
        void setUseLocalInitDataCost(bool use_local_init_data_cost) { use_local_init_data_cost_ = use_local_init_data_cost; }

    private:
        int min_disp_th_;
        int ndisp_;
        int iters_;
        int levels_;
        float max_data_term_;
        float data_weight_;
        float max_disc_term_;
        float disc_single_jump_;
        int msg_type_;
        int nr_plane_;
        bool use_local_init_data_cost_;

        GpuMat mbuf_;
        GpuMat temp_;
        GpuMat outBuf_;
    };

    const float DEFAULT_MAX_DATA_TERM = 30.0f;
    const float DEFAULT_DATA_WEIGHT = 1.0f;
    const float DEFAULT_MAX_DISC_TERM = 160.0f;
    const float DEFAULT_DISC_SINGLE_JUMP = 10.0f;

    StereoCSBPImpl::StereoCSBPImpl(int ndisp, int iters, int levels, int nr_plane, int msg_type) :
        min_disp_th_(0), ndisp_(ndisp), iters_(iters), levels_(levels),
        max_data_term_(DEFAULT_MAX_DATA_TERM), data_weight_(DEFAULT_DATA_WEIGHT),
        max_disc_term_(DEFAULT_MAX_DISC_TERM), disc_single_jump_(DEFAULT_DISC_SINGLE_JUMP),
        msg_type_(msg_type), nr_plane_(nr_plane), use_local_init_data_cost_(true)
    {
    }

    void StereoCSBPImpl::compute(InputArray left, InputArray right, OutputArray disparity)
    {
        compute(left, right, disparity, Stream::Null());
    }

    void StereoCSBPImpl::compute(InputArray _left, InputArray _right, OutputArray disp, Stream& _stream)
    {
        using namespace cv::cuda::device::stereocsbp;

        CV_Assert( msg_type_ == CV_32F || msg_type_ == CV_16S );
        CV_Assert( 0 < ndisp_ && 0 < iters_ && 0 < levels_ && 0 < nr_plane_ && levels_ <= 8 );

        GpuMat left = _left.getGpuMat();
        GpuMat right = _right.getGpuMat();

        CV_Assert( left.type() == CV_8UC1 || left.type() == CV_8UC3 || left.type() == CV_8UC4 );
        CV_Assert( left.size() == right.size() && left.type() == right.type() );

        cudaStream_t stream = StreamAccessor::getStream(_stream);

        ////////////////////////////////////////////////////////////////////////////////////////////
        // Init

        int rows = left.rows;
        int cols = left.cols;

        levels_ = std::min(levels_, int(log((double)ndisp_) / log(2.0)));

        // compute sizes
        AutoBuffer<int> buf(levels_ * 3);
        int* cols_pyr = buf.data();
        int* rows_pyr = cols_pyr + levels_;
        int* nr_plane_pyr = rows_pyr + levels_;

        cols_pyr[0]     = cols;
        rows_pyr[0]     = rows;
        nr_plane_pyr[0] = nr_plane_;

        for (int i = 1; i < levels_; i++)
        {
            cols_pyr[i]     = cols_pyr[i-1] / 2;
            rows_pyr[i]     = rows_pyr[i-1] / 2;
            nr_plane_pyr[i] = nr_plane_pyr[i-1] * 2;
        }

        GpuMat u[2], d[2], l[2], r[2], disp_selected_pyr[2], data_cost, data_cost_selected;

        //allocate buffers
        int buffers_count = 10; // (up + down + left + right + disp_selected_pyr) * 2
        buffers_count += 2; //  data_cost has twice more rows than other buffers, what's why +2, not +1;
        buffers_count += 1; //  data_cost_selected
        mbuf_.create(rows * nr_plane_ * buffers_count, cols, msg_type_);

        data_cost          = mbuf_.rowRange(0, rows * nr_plane_ * 2);
        data_cost_selected = mbuf_.rowRange(data_cost.rows, data_cost.rows + rows * nr_plane_);

        for(int k = 0; k < 2; ++k) // in/out
        {
            GpuMat sub1 = mbuf_.rowRange(data_cost.rows + data_cost_selected.rows, mbuf_.rows);
            GpuMat sub2 = sub1.rowRange((k+0)*sub1.rows/2, (k+1)*sub1.rows/2);

            GpuMat *buf_ptrs[] = { &u[k], &d[k], &l[k], &r[k], &disp_selected_pyr[k] };
            for(int _r = 0; _r < 5; ++_r)
            {
                *buf_ptrs[_r] = sub2.rowRange(_r * sub2.rows/5, (_r+1) * sub2.rows/5);
                CV_DbgAssert( buf_ptrs[_r]->cols == cols && buf_ptrs[_r]->rows == rows * nr_plane_ );
            }
        };

        size_t elem_step = mbuf_.step / mbuf_.elemSize();

        Size temp_size = data_cost.size();
        if ((size_t)temp_size.area() < elem_step * rows_pyr[levels_ - 1] * ndisp_)
            temp_size = Size(static_cast<int>(elem_step), rows_pyr[levels_ - 1] * ndisp_);

        temp_.create(temp_size, msg_type_);

        ////////////////////////////////////////////////////////////////////////////
        // Compute

        l[0].setTo(0, _stream);
        d[0].setTo(0, _stream);
        r[0].setTo(0, _stream);
        u[0].setTo(0, _stream);

        l[1].setTo(0, _stream);
        d[1].setTo(0, _stream);
        r[1].setTo(0, _stream);
        u[1].setTo(0, _stream);

        data_cost.setTo(0, _stream);
        data_cost_selected.setTo(0, _stream);

        int cur_idx = 0;

        if (msg_type_ == CV_32F)
        {
            for (int i = levels_ - 1; i >= 0; i--)
            {
                if (i == levels_ - 1)
                {
                    init_data_cost(left.ptr<uchar>(), right.ptr<uchar>(), temp_.ptr<uchar>(), left.step, left.rows, left.cols, disp_selected_pyr[cur_idx].ptr<float>(), data_cost_selected.ptr<float>(),
                        elem_step, rows_pyr[i], cols_pyr[i], i, nr_plane_pyr[i], ndisp_, left.channels(), data_weight_, max_data_term_, min_disp_th_, use_local_init_data_cost_, stream);
                }
                else
                {
                    compute_data_cost(left.ptr<uchar>(), right.ptr<uchar>(), left.step, disp_selected_pyr[cur_idx].ptr<float>(), data_cost.ptr<float>(), elem_step,
                        left.rows, left.cols, rows_pyr[i], cols_pyr[i], rows_pyr[i+1], i, nr_plane_pyr[i+1], left.channels(), data_weight_, max_data_term_, min_disp_th_, stream);

                    int new_idx = (cur_idx + 1) & 1;

                    init_message(temp_.ptr<uchar>(),
                                 u[new_idx].ptr<float>(), d[new_idx].ptr<float>(), l[new_idx].ptr<float>(), r[new_idx].ptr<float>(),
                                 u[cur_idx].ptr<float>(), d[cur_idx].ptr<float>(), l[cur_idx].ptr<float>(), r[cur_idx].ptr<float>(),
                                 disp_selected_pyr[new_idx].ptr<float>(), disp_selected_pyr[cur_idx].ptr<float>(),
                                 data_cost_selected.ptr<float>(), data_cost.ptr<float>(), elem_step, rows_pyr[i],
                                 cols_pyr[i], nr_plane_pyr[i], rows_pyr[i+1], cols_pyr[i+1], nr_plane_pyr[i+1], stream);

                    cur_idx = new_idx;
                }

                calc_all_iterations(temp_.ptr<uchar>(), u[cur_idx].ptr<float>(), d[cur_idx].ptr<float>(), l[cur_idx].ptr<float>(), r[cur_idx].ptr<float>(),
                                    data_cost_selected.ptr<float>(), disp_selected_pyr[cur_idx].ptr<float>(), elem_step,
                                    rows_pyr[i], cols_pyr[i], nr_plane_pyr[i], iters_, max_disc_term_, disc_single_jump_, stream);
            }
        }
        else
        {
            for (int i = levels_ - 1; i >= 0; i--)
            {
                if (i == levels_ - 1)
                {
                    init_data_cost(left.ptr<uchar>(), right.ptr<uchar>(), temp_.ptr<uchar>(), left.step, left.rows, left.cols, disp_selected_pyr[cur_idx].ptr<short>(), data_cost_selected.ptr<short>(),
                        elem_step, rows_pyr[i], cols_pyr[i], i, nr_plane_pyr[i], ndisp_, left.channels(), data_weight_, max_data_term_, min_disp_th_, use_local_init_data_cost_, stream);
                }
                else
                {
                    compute_data_cost(left.ptr<uchar>(), right.ptr<uchar>(), left.step, disp_selected_pyr[cur_idx].ptr<short>(), data_cost.ptr<short>(), elem_step,
                        left.rows, left.cols, rows_pyr[i], cols_pyr[i], rows_pyr[i+1], i, nr_plane_pyr[i+1], left.channels(), data_weight_, max_data_term_, min_disp_th_, stream);

                    int new_idx = (cur_idx + 1) & 1;

                    init_message(temp_.ptr<uchar>(),
                                 u[new_idx].ptr<short>(), d[new_idx].ptr<short>(), l[new_idx].ptr<short>(), r[new_idx].ptr<short>(),
                                 u[cur_idx].ptr<short>(), d[cur_idx].ptr<short>(), l[cur_idx].ptr<short>(), r[cur_idx].ptr<short>(),
                                 disp_selected_pyr[new_idx].ptr<short>(), disp_selected_pyr[cur_idx].ptr<short>(),
                                 data_cost_selected.ptr<short>(), data_cost.ptr<short>(), elem_step, rows_pyr[i],
                                 cols_pyr[i], nr_plane_pyr[i], rows_pyr[i+1], cols_pyr[i+1], nr_plane_pyr[i+1], stream);

                    cur_idx = new_idx;
                }

                calc_all_iterations(temp_.ptr<uchar>(), u[cur_idx].ptr<short>(), d[cur_idx].ptr<short>(), l[cur_idx].ptr<short>(), r[cur_idx].ptr<short>(),
                                    data_cost_selected.ptr<short>(), disp_selected_pyr[cur_idx].ptr<short>(), elem_step,
                                    rows_pyr[i], cols_pyr[i], nr_plane_pyr[i], iters_, max_disc_term_, disc_single_jump_, stream);
            }
        }

        const int dtype = disp.fixedType() ? disp.type() : CV_16SC1;

        disp.create(rows, cols, dtype);
        GpuMat out = disp.getGpuMat();

        if (dtype != CV_16SC1)
        {
            outBuf_.create(rows, cols, CV_16SC1);
            out = outBuf_;
        }

        out.setTo(0, _stream);

        if (msg_type_ == CV_32F)
        {
            compute_disp(u[cur_idx].ptr<float>(), d[cur_idx].ptr<float>(), l[cur_idx].ptr<float>(), r[cur_idx].ptr<float>(),
                         data_cost_selected.ptr<float>(), disp_selected_pyr[cur_idx].ptr<float>(), elem_step, out, nr_plane_pyr[0], stream);
        }
        else
        {
            compute_disp(u[cur_idx].ptr<short>(), d[cur_idx].ptr<short>(), l[cur_idx].ptr<short>(), r[cur_idx].ptr<short>(),
                         data_cost_selected.ptr<short>(), disp_selected_pyr[cur_idx].ptr<short>(), elem_step, out, nr_plane_pyr[0], stream);
        }

        if (dtype != CV_16SC1)
            out.convertTo(disp, dtype, _stream);
    }

    void StereoCSBPImpl::compute(InputArray /*data*/, OutputArray /*disparity*/, Stream& /*stream*/)
    {
        CV_Error(Error::StsNotImplemented, "Not implemented");
    }
}

Ptr<cuda::StereoConstantSpaceBP> cv::cuda::createStereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane, int msg_type)
{
    return makePtr<StereoCSBPImpl>(ndisp, iters, levels, nr_plane, msg_type);
}

void cv::cuda::StereoConstantSpaceBP::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane)
{
    ndisp = (int) ((float) width / 3.14f);
    if ((ndisp & 1) != 0)
        ndisp++;

    int mm = std::max(width, height);
    iters = mm / 100 + ((mm > 1200)? - 4 : 4);

    levels = (int)::log(static_cast<double>(mm)) * 2 / 3;
    if (levels == 0) levels++;

    nr_plane = (int) ((float) ndisp / std::pow(2.0, levels + 1));
}

#endif /* !defined (HAVE_CUDA) */
