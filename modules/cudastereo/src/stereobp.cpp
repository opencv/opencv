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

void cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(int, int, int&, int&, int&) { throw_no_cuda(); }

Ptr<cuda::StereoBeliefPropagation> cv::cuda::createStereoBeliefPropagation(int, int, int, int) { throw_no_cuda(); return Ptr<cuda::StereoBeliefPropagation>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace stereobp
    {
        void load_constants(int ndisp, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump);
        template<typename T, typename D>
        void comp_data_gpu(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream);
        template<typename T>
        void data_step_down_gpu(int dst_cols, int dst_rows, int src_cols, int src_rows, const PtrStepSzb& src, const PtrStepSzb& dst, cudaStream_t stream);
        template <typename T>
        void level_up_messages_gpu(int dst_idx, int dst_cols, int dst_rows, int src_rows, PtrStepSzb* mus, PtrStepSzb* mds, PtrStepSzb* mls, PtrStepSzb* mrs, cudaStream_t stream);
        template <typename T>
        void calc_all_iterations_gpu(int cols, int rows, int iters, const PtrStepSzb& u, const PtrStepSzb& d,
            const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, cudaStream_t stream);
        template <typename T>
        void output_gpu(const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data,
            const PtrStepSz<short>& disp, cudaStream_t stream);
    }
}}}

namespace
{
    class StereoBPImpl : public cuda::StereoBeliefPropagation
    {
    public:
        StereoBPImpl(int ndisp, int iters, int levels, int msg_type);

        void compute(InputArray left, InputArray right, OutputArray disparity);
        void compute(InputArray left, InputArray right, OutputArray disparity, Stream& stream);
        void compute(InputArray data, OutputArray disparity, Stream& stream);

        int getMinDisparity() const { return 0; }
        void setMinDisparity(int /*minDisparity*/) {}

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

    private:
        void init(Stream& stream);
        void calcBP(OutputArray disp, Stream& stream);

        int ndisp_;
        int iters_;
        int levels_;
        float max_data_term_;
        float data_weight_;
        float max_disc_term_;
        float disc_single_jump_;
        int msg_type_;

        float scale_;
        int rows_, cols_;
        std::vector<int> cols_all_, rows_all_;
        GpuMat u_, d_, l_, r_, u2_, d2_, l2_, r2_;
        std::vector<GpuMat> datas_;
        GpuMat outBuf_;
    };

    const float DEFAULT_MAX_DATA_TERM = 10.0f;
    const float DEFAULT_DATA_WEIGHT = 0.07f;
    const float DEFAULT_MAX_DISC_TERM = 1.7f;
    const float DEFAULT_DISC_SINGLE_JUMP = 1.0f;

    StereoBPImpl::StereoBPImpl(int ndisp, int iters, int levels, int msg_type) :
        ndisp_(ndisp), iters_(iters), levels_(levels),
        max_data_term_(DEFAULT_MAX_DATA_TERM), data_weight_(DEFAULT_DATA_WEIGHT),
        max_disc_term_(DEFAULT_MAX_DISC_TERM), disc_single_jump_(DEFAULT_DISC_SINGLE_JUMP),
        msg_type_(msg_type)
    {
    }

    void StereoBPImpl::compute(InputArray left, InputArray right, OutputArray disparity)
    {
        compute(left, right, disparity, Stream::Null());
    }

    void StereoBPImpl::compute(InputArray _left, InputArray _right, OutputArray disparity, Stream& stream)
    {
        using namespace cv::cuda::device::stereobp;

        typedef void (*comp_data_t)(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& data, cudaStream_t stream);
        static const comp_data_t comp_data_callers[2][5] =
        {
            {0, comp_data_gpu<unsigned char, short>, 0, comp_data_gpu<uchar3, short>, comp_data_gpu<uchar4, short>},
            {0, comp_data_gpu<unsigned char, float>, 0, comp_data_gpu<uchar3, float>, comp_data_gpu<uchar4, float>}
        };

        scale_ = msg_type_ == CV_32F ? 1.0f : 10.0f;

        CV_Assert( 0 < ndisp_ && 0 < iters_ && 0 < levels_ );
        CV_Assert( msg_type_ == CV_32F || msg_type_ == CV_16S );
        CV_Assert( msg_type_ == CV_32F || (1 << (levels_ - 1)) * scale_ * max_data_term_ < std::numeric_limits<short>::max() );

        GpuMat left = _left.getGpuMat();
        GpuMat right = _right.getGpuMat();

        CV_Assert( left.type() == CV_8UC1 || left.type() == CV_8UC3 || left.type() == CV_8UC4 );
        CV_Assert( left.size() == right.size() && left.type() == right.type() );

        rows_ = left.rows;
        cols_ = left.cols;

        const int divisor = (int) pow(2.f, levels_ - 1.0f);
        const int lowest_cols = cols_ / divisor;
        const int lowest_rows = rows_ / divisor;
        const int min_image_dim_size = 2;
        CV_Assert( std::min(lowest_cols, lowest_rows) > min_image_dim_size );

        init(stream);

        datas_[0].create(rows_ * ndisp_, cols_, msg_type_);

        comp_data_callers[msg_type_ == CV_32F][left.channels()](left, right, datas_[0], StreamAccessor::getStream(stream));

        calcBP(disparity, stream);
    }

    void StereoBPImpl::compute(InputArray _data, OutputArray disparity, Stream& stream)
    {
        scale_ = msg_type_ == CV_32F ? 1.0f : 10.0f;

        CV_Assert( 0 < ndisp_ && 0 < iters_ && 0 < levels_ );
        CV_Assert( msg_type_ == CV_32F || msg_type_ == CV_16S );
        CV_Assert( msg_type_ == CV_32F || (1 << (levels_ - 1)) * scale_ * max_data_term_ < std::numeric_limits<short>::max() );

        GpuMat data = _data.getGpuMat();

        CV_Assert( (data.type() == msg_type_) && (data.rows % ndisp_ == 0) );

        rows_ = data.rows / ndisp_;
        cols_ = data.cols;

        const int divisor = (int) pow(2.f, levels_ - 1.0f);
        const int lowest_cols = cols_ / divisor;
        const int lowest_rows = rows_ / divisor;
        const int min_image_dim_size = 2;
        CV_Assert( std::min(lowest_cols, lowest_rows) > min_image_dim_size );

        init(stream);

        data.copyTo(datas_[0], stream);

        calcBP(disparity, stream);
    }

    void StereoBPImpl::init(Stream& stream)
    {
        using namespace cv::cuda::device::stereobp;

        u_.create(rows_ * ndisp_, cols_, msg_type_);
        d_.create(rows_ * ndisp_, cols_, msg_type_);
        l_.create(rows_ * ndisp_, cols_, msg_type_);
        r_.create(rows_ * ndisp_, cols_, msg_type_);

        if (levels_ & 1)
        {
            //can clear less area
            u_.setTo(0, stream);
            d_.setTo(0, stream);
            l_.setTo(0, stream);
            r_.setTo(0, stream);
        }

        if (levels_ > 1)
        {
            int less_rows = (rows_ + 1) / 2;
            int less_cols = (cols_ + 1) / 2;

            u2_.create(less_rows * ndisp_, less_cols, msg_type_);
            d2_.create(less_rows * ndisp_, less_cols, msg_type_);
            l2_.create(less_rows * ndisp_, less_cols, msg_type_);
            r2_.create(less_rows * ndisp_, less_cols, msg_type_);

            if ((levels_ & 1) == 0)
            {
                u2_.setTo(0, stream);
                d2_.setTo(0, stream);
                l2_.setTo(0, stream);
                r2_.setTo(0, stream);
            }
        }

        load_constants(ndisp_, max_data_term_, scale_ * data_weight_, scale_ * max_disc_term_, scale_ * disc_single_jump_);

        datas_.resize(levels_);

        cols_all_.resize(levels_);
        rows_all_.resize(levels_);

        cols_all_[0] = cols_;
        rows_all_[0] = rows_;
    }

    void StereoBPImpl::calcBP(OutputArray disp, Stream& _stream)
    {
        using namespace cv::cuda::device::stereobp;

        typedef void (*data_step_down_t)(int dst_cols, int dst_rows, int src_cols, int src_rows, const PtrStepSzb& src, const PtrStepSzb& dst, cudaStream_t stream);
        static const data_step_down_t data_step_down_callers[2] =
        {
            data_step_down_gpu<short>, data_step_down_gpu<float>
        };

        typedef void (*level_up_messages_t)(int dst_idx, int dst_cols, int dst_rows, int src_rows, PtrStepSzb* mus, PtrStepSzb* mds, PtrStepSzb* mls, PtrStepSzb* mrs, cudaStream_t stream);
        static const level_up_messages_t level_up_messages_callers[2] =
        {
            level_up_messages_gpu<short>, level_up_messages_gpu<float>
        };

        typedef void (*calc_all_iterations_t)(int cols, int rows, int iters, const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, cudaStream_t stream);
        static const calc_all_iterations_t calc_all_iterations_callers[2] =
        {
            calc_all_iterations_gpu<short>, calc_all_iterations_gpu<float>
        };

        typedef void (*output_t)(const PtrStepSzb& u, const PtrStepSzb& d, const PtrStepSzb& l, const PtrStepSzb& r, const PtrStepSzb& data, const PtrStepSz<short>& disp, cudaStream_t stream);
        static const output_t output_callers[2] =
        {
            output_gpu<short>, output_gpu<float>
        };

        const int funcIdx = msg_type_ == CV_32F;

        cudaStream_t stream = StreamAccessor::getStream(_stream);

        for (int i = 1; i < levels_; ++i)
        {
            cols_all_[i] = (cols_all_[i-1] + 1) / 2;
            rows_all_[i] = (rows_all_[i-1] + 1) / 2;

            datas_[i].create(rows_all_[i] * ndisp_, cols_all_[i], msg_type_);

            data_step_down_callers[funcIdx](cols_all_[i], rows_all_[i], cols_all_[i-1], rows_all_[i-1], datas_[i-1], datas_[i], stream);
        }

        PtrStepSzb mus[] = {u_, u2_};
        PtrStepSzb mds[] = {d_, d2_};
        PtrStepSzb mrs[] = {r_, r2_};
        PtrStepSzb mls[] = {l_, l2_};

        int mem_idx = (levels_ & 1) ? 0 : 1;

        for (int i = levels_ - 1; i >= 0; --i)
        {
            // for lower level we have already computed messages by setting to zero
            if (i != levels_ - 1)
                level_up_messages_callers[funcIdx](mem_idx, cols_all_[i], rows_all_[i], rows_all_[i+1], mus, mds, mls, mrs, stream);

            calc_all_iterations_callers[funcIdx](cols_all_[i], rows_all_[i], iters_, mus[mem_idx], mds[mem_idx], mls[mem_idx], mrs[mem_idx], datas_[i], stream);

            mem_idx = (mem_idx + 1) & 1;
        }

        const int dtype = disp.fixedType() ? disp.type() : CV_16SC1;

        disp.create(rows_, cols_, dtype);
        GpuMat out = disp.getGpuMat();

        if (dtype != CV_16SC1)
        {
            outBuf_.create(rows_, cols_, CV_16SC1);
            out = outBuf_;
        }

        out.setTo(0, _stream);

        output_callers[funcIdx](u_, d_, l_, r_, datas_.front(), out, stream);

        if (dtype != CV_16SC1)
            out.convertTo(disp, dtype, _stream);
    }
}

Ptr<cuda::StereoBeliefPropagation> cv::cuda::createStereoBeliefPropagation(int ndisp, int iters, int levels, int msg_type)
{
    return makePtr<StereoBPImpl>(ndisp, iters, levels, msg_type);
}

void cv::cuda::StereoBeliefPropagation::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels)
{
    ndisp = width / 4;
    if ((ndisp & 1) != 0)
        ndisp++;

    int mm = std::max(width, height);
    iters = mm / 100 + 2;

    levels = (int)(::log(static_cast<double>(mm)) + 1) * 4 / 5;
    if (levels == 0) levels++;
}

#endif /* !defined (HAVE_CUDA) */
