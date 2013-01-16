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

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

class cv::gpu::FGDStatModel::Impl
{
};

cv::gpu::FGDStatModel::Params::Params() { throw_nogpu(); }

cv::gpu::FGDStatModel::FGDStatModel(int) { throw_nogpu(); }
cv::gpu::FGDStatModel::FGDStatModel(const cv::gpu::GpuMat&, const Params&, int) { throw_nogpu(); }
cv::gpu::FGDStatModel::~FGDStatModel() {}
void cv::gpu::FGDStatModel::create(const cv::gpu::GpuMat&, const Params&) { throw_nogpu(); }
void cv::gpu::FGDStatModel::release() {}
int cv::gpu::FGDStatModel::update(const cv::gpu::GpuMat&) { throw_nogpu(); return 0; }

#else

#include "fgd_bgfg_common.hpp"

namespace
{
    class BGPixelStat
    {
    public:
        void create(cv::Size size, const cv::gpu::FGDStatModel::Params& params, int out_cn);
        void release();

        void setTrained();

        operator bgfg::BGPixelStat();

    private:
        cv::gpu::GpuMat Pbc_;
        cv::gpu::GpuMat Pbcc_;
        cv::gpu::GpuMat is_trained_st_model_;
        cv::gpu::GpuMat is_trained_dyn_model_;

        cv::gpu::GpuMat ctable_Pv_;
        cv::gpu::GpuMat ctable_Pvb_;
        cv::gpu::GpuMat ctable_v_;

        cv::gpu::GpuMat cctable_Pv_;
        cv::gpu::GpuMat cctable_Pvb_;
        cv::gpu::GpuMat cctable_v1_;
        cv::gpu::GpuMat cctable_v2_;
    };

    void BGPixelStat::create(cv::Size size, const cv::gpu::FGDStatModel::Params& params, int out_cn)
    {
        cv::gpu::ensureSizeIsEnough(size, CV_32FC1, Pbc_);
        Pbc_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(size, CV_32FC1, Pbcc_);
        Pbcc_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(size, CV_8UC1, is_trained_st_model_);
        is_trained_st_model_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(size, CV_8UC1, is_trained_dyn_model_);
        is_trained_dyn_model_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(params.N2c * size.height, size.width, CV_32FC1, ctable_Pv_);
        ctable_Pv_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(params.N2c * size.height, size.width, CV_32FC1, ctable_Pvb_);
        ctable_Pvb_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(params.N2c * size.height, size.width, CV_8UC(out_cn), ctable_v_);
        ctable_v_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_32FC1, cctable_Pv_);
        cctable_Pv_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_32FC1, cctable_Pvb_);
        cctable_Pvb_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_8UC(out_cn), cctable_v1_);
        cctable_v1_.setTo(cv::Scalar::all(0));

        cv::gpu::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_8UC(out_cn), cctable_v2_);
        cctable_v2_.setTo(cv::Scalar::all(0));
    }

    void BGPixelStat::release()
    {
        Pbc_.release();
        Pbcc_.release();
        is_trained_st_model_.release();
        is_trained_dyn_model_.release();

        ctable_Pv_.release();
        ctable_Pvb_.release();
        ctable_v_.release();

        cctable_Pv_.release();
        cctable_Pvb_.release();
        cctable_v1_.release();
        cctable_v2_.release();
    }

    void BGPixelStat::setTrained()
    {
        is_trained_st_model_.setTo(cv::Scalar::all(1));
        is_trained_dyn_model_.setTo(cv::Scalar::all(1));
    }

    BGPixelStat::operator bgfg::BGPixelStat()
    {
        bgfg::BGPixelStat stat;

        stat.rows_ = Pbc_.rows;

        stat.Pbc_data_ = Pbc_.data;
        stat.Pbc_step_ = Pbc_.step;

        stat.Pbcc_data_ = Pbcc_.data;
        stat.Pbcc_step_ = Pbcc_.step;

        stat.is_trained_st_model_data_ = is_trained_st_model_.data;
        stat.is_trained_st_model_step_ = is_trained_st_model_.step;

        stat.is_trained_dyn_model_data_ = is_trained_dyn_model_.data;
        stat.is_trained_dyn_model_step_ = is_trained_dyn_model_.step;

        stat.ctable_Pv_data_ = ctable_Pv_.data;
        stat.ctable_Pv_step_ = ctable_Pv_.step;

        stat.ctable_Pvb_data_ = ctable_Pvb_.data;
        stat.ctable_Pvb_step_ = ctable_Pvb_.step;

        stat.ctable_v_data_ = ctable_v_.data;
        stat.ctable_v_step_ = ctable_v_.step;

        stat.cctable_Pv_data_ = cctable_Pv_.data;
        stat.cctable_Pv_step_ = cctable_Pv_.step;

        stat.cctable_Pvb_data_ = cctable_Pvb_.data;
        stat.cctable_Pvb_step_ = cctable_Pvb_.step;

        stat.cctable_v1_data_ = cctable_v1_.data;
        stat.cctable_v1_step_ = cctable_v1_.step;

        stat.cctable_v2_data_ = cctable_v2_.data;
        stat.cctable_v2_step_ = cctable_v2_.step;

        return stat;
    }
}

class cv::gpu::FGDStatModel::Impl
{
public:
    Impl(cv::gpu::GpuMat& background, cv::gpu::GpuMat& foreground, std::vector< std::vector<cv::Point> >& foreground_regions, int out_cn);
    ~Impl();

    void create(const cv::gpu::GpuMat& firstFrame, const cv::gpu::FGDStatModel::Params& params);
    void release();

    int update(const cv::gpu::GpuMat& curFrame);

private:
    Impl(const Impl&);
    Impl& operator=(const Impl&);

    int out_cn_;

    cv::gpu::FGDStatModel::Params params_;

    cv::gpu::GpuMat& background_;
    cv::gpu::GpuMat& foreground_;
    std::vector< std::vector<cv::Point> >& foreground_regions_;

    cv::Mat h_foreground_;

    cv::gpu::GpuMat prevFrame_;
    cv::gpu::GpuMat Ftd_;
    cv::gpu::GpuMat Fbd_;
    BGPixelStat stat_;

    cv::gpu::GpuMat hist_;
    cv::gpu::GpuMat histBuf_;

    cv::gpu::GpuMat countBuf_;

    cv::gpu::GpuMat buf_;
    cv::gpu::GpuMat filterBuf_;
    cv::gpu::GpuMat filterBrd_;

    cv::Ptr<cv::gpu::FilterEngine_GPU> dilateFilter_;
    cv::Ptr<cv::gpu::FilterEngine_GPU> erodeFilter_;

    CvMemStorage* storage_;
};

cv::gpu::FGDStatModel::Impl::Impl(cv::gpu::GpuMat& background, cv::gpu::GpuMat& foreground, std::vector< std::vector<cv::Point> >& foreground_regions, int out_cn) :
    out_cn_(out_cn), background_(background), foreground_(foreground), foreground_regions_(foreground_regions)
{
    CV_Assert( out_cn_ == 3 || out_cn_ == 4 );

    storage_ = cvCreateMemStorage();
    CV_Assert( storage_ != 0 );
}

cv::gpu::FGDStatModel::Impl::~Impl()
{
    cvReleaseMemStorage(&storage_);
}

namespace
{
    void copyChannels(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, int dst_cn = -1)
    {
        const int src_cn = src.channels();

        if (dst_cn < 0)
            dst_cn = src_cn;

        cv::gpu::ensureSizeIsEnough(src.size(), CV_MAKE_TYPE(src.depth(), dst_cn), dst);

        if (src_cn == dst_cn)
            src.copyTo(dst);
        else
        {
            static const int cvt_codes[4][4] =
            {
                {-1, -1, cv::COLOR_GRAY2BGR, cv::COLOR_GRAY2BGRA},
                {-1, -1, -1, -1},
                {cv::COLOR_BGR2GRAY, -1, -1, cv::COLOR_BGR2BGRA},
                {cv::COLOR_BGRA2GRAY, -1, cv::COLOR_BGRA2BGR, -1}
            };

            const int cvt_code = cvt_codes[src_cn - 1][dst_cn - 1];
            CV_DbgAssert( cvt_code >= 0 );

            cv::gpu::cvtColor(src, dst, cvt_code, dst_cn);
        }
    }
}

void cv::gpu::FGDStatModel::Impl::create(const cv::gpu::GpuMat& firstFrame, const cv::gpu::FGDStatModel::Params& params)
{
    CV_Assert(firstFrame.type() == CV_8UC3 || firstFrame.type() == CV_8UC4);

    params_ = params;

    cv::gpu::ensureSizeIsEnough(firstFrame.size(), CV_8UC1, foreground_);

    copyChannels(firstFrame, background_, out_cn_);

    copyChannels(firstFrame, prevFrame_);

    cv::gpu::ensureSizeIsEnough(firstFrame.size(), CV_8UC1, Ftd_);
    cv::gpu::ensureSizeIsEnough(firstFrame.size(), CV_8UC1, Fbd_);

    stat_.create(firstFrame.size(), params_, out_cn_);
    bgfg::setBGPixelStat(stat_);

    if (params_.perform_morphing > 0)
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + params_.perform_morphing * 2, 1 + params_.perform_morphing * 2));
        cv::Point anchor(params_.perform_morphing, params_.perform_morphing);

        dilateFilter_ = cv::gpu::createMorphologyFilter_GPU(cv::MORPH_DILATE, CV_8UC1, kernel, filterBuf_, anchor);
        erodeFilter_ = cv::gpu::createMorphologyFilter_GPU(cv::MORPH_ERODE, CV_8UC1, kernel, filterBuf_, anchor);
    }
}

void cv::gpu::FGDStatModel::Impl::release()
{
    background_.release();
    foreground_.release();

    prevFrame_.release();
    Ftd_.release();
    Fbd_.release();
    stat_.release();

    hist_.release();
    histBuf_.release();

    countBuf_.release();

    buf_.release();
    filterBuf_.release();
    filterBrd_.release();
}

/////////////////////////////////////////////////////////////////////////
// changeDetection

namespace
{
    void calcDiffHistogram(const cv::gpu::GpuMat& prevFrame, const cv::gpu::GpuMat& curFrame, cv::gpu::GpuMat& hist, cv::gpu::GpuMat& histBuf)
    {
        typedef void (*func_t)(cv::gpu::PtrStepSzb prevFrame, cv::gpu::PtrStepSzb curFrame, unsigned int* hist0, unsigned int* hist1, unsigned int* hist2, unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2, bool cc20, cudaStream_t stream);
        static const func_t funcs[4][4] =
        {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,bgfg::calcDiffHistogram_gpu<uchar3, uchar3>,bgfg::calcDiffHistogram_gpu<uchar3, uchar4>},
            {0,0,bgfg::calcDiffHistogram_gpu<uchar4, uchar3>,bgfg::calcDiffHistogram_gpu<uchar4, uchar4>}
        };

        hist.create(3, 256, CV_32SC1);
        histBuf.create(3, bgfg::PARTIAL_HISTOGRAM_COUNT * bgfg::HISTOGRAM_BIN_COUNT, CV_32SC1);

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1](
                    prevFrame, curFrame,
                    hist.ptr<unsigned int>(0), hist.ptr<unsigned int>(1), hist.ptr<unsigned int>(2),
                    histBuf.ptr<unsigned int>(0), histBuf.ptr<unsigned int>(1), histBuf.ptr<unsigned int>(2),
                    cv::gpu::deviceSupports(cv::gpu::FEATURE_SET_COMPUTE_20), 0);
    }

    void calcRelativeVariance(unsigned int hist[3 * 256], double relativeVariance[3][bgfg::HISTOGRAM_BIN_COUNT])
    {
        std::memset(relativeVariance, 0, 3 * bgfg::HISTOGRAM_BIN_COUNT * sizeof(double));

        for (int thres = bgfg::HISTOGRAM_BIN_COUNT - 2; thres >= 0; --thres)
        {
            cv::Vec3d sum(0.0, 0.0, 0.0);
            cv::Vec3d sqsum(0.0, 0.0, 0.0);
            cv::Vec3i count(0, 0, 0);

            for (int j = thres; j < bgfg::HISTOGRAM_BIN_COUNT; ++j)
            {
                sum[0]   += static_cast<double>(j) * hist[j];
                sqsum[0] += static_cast<double>(j * j) * hist[j];
                count[0] += hist[j];

                sum[1]   += static_cast<double>(j) * hist[j + 256];
                sqsum[1] += static_cast<double>(j * j) * hist[j + 256];
                count[1] += hist[j + 256];

                sum[2]   += static_cast<double>(j) * hist[j + 512];
                sqsum[2] += static_cast<double>(j * j) * hist[j + 512];
                count[2] += hist[j + 512];
            }

            count[0] = std::max(count[0], 1);
            count[1] = std::max(count[1], 1);
            count[2] = std::max(count[2], 1);

            cv::Vec3d my(
                sum[0] / count[0],
                sum[1] / count[1],
                sum[2] / count[2]
            );

            relativeVariance[0][thres] = std::sqrt(sqsum[0] / count[0] - my[0] * my[0]);
            relativeVariance[1][thres] = std::sqrt(sqsum[1] / count[1] - my[1] * my[1]);
            relativeVariance[2][thres] = std::sqrt(sqsum[2] / count[2] - my[2] * my[2]);
        }
    }

    void calcDiffThreshMask(const cv::gpu::GpuMat& prevFrame, const cv::gpu::GpuMat& curFrame, cv::Vec3d bestThres, cv::gpu::GpuMat& changeMask)
    {
        typedef void (*func_t)(cv::gpu::PtrStepSzb prevFrame, cv::gpu::PtrStepSzb curFrame, uchar3 bestThres, cv::gpu::PtrStepSzb changeMask, cudaStream_t stream);
        static const func_t funcs[4][4] =
        {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,bgfg::calcDiffThreshMask_gpu<uchar3, uchar3>,bgfg::calcDiffThreshMask_gpu<uchar3, uchar4>},
            {0,0,bgfg::calcDiffThreshMask_gpu<uchar4, uchar3>,bgfg::calcDiffThreshMask_gpu<uchar4, uchar4>}
        };

        changeMask.setTo(cv::Scalar::all(0));

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1](prevFrame, curFrame, make_uchar3((uchar)bestThres[0], (uchar)bestThres[1], (uchar)bestThres[2]), changeMask, 0);
    }

    // performs change detection for Foreground detection algorithm
    void changeDetection(const cv::gpu::GpuMat& prevFrame, const cv::gpu::GpuMat& curFrame, cv::gpu::GpuMat& changeMask, cv::gpu::GpuMat& hist, cv::gpu::GpuMat& histBuf)
    {
        calcDiffHistogram(prevFrame, curFrame, hist, histBuf);

        unsigned int histData[3 * 256];
        cv::Mat h_hist(3, 256, CV_32SC1, histData);
        hist.download(h_hist);

        double relativeVariance[3][bgfg::HISTOGRAM_BIN_COUNT];
        calcRelativeVariance(histData, relativeVariance);

        // Find maximum:
        cv::Vec3d bestThres(10.0, 10.0, 10.0);
        for (int i = 0; i < bgfg::HISTOGRAM_BIN_COUNT; ++i)
        {
            bestThres[0] = std::max(bestThres[0], relativeVariance[0][i]);
            bestThres[1] = std::max(bestThres[1], relativeVariance[1][i]);
            bestThres[2] = std::max(bestThres[2], relativeVariance[2][i]);
        }

        calcDiffThreshMask(prevFrame, curFrame, bestThres, changeMask);
    }
}

/////////////////////////////////////////////////////////////////////////
// bgfgClassification

namespace
{
    int bgfgClassification(const cv::gpu::GpuMat& prevFrame, const cv::gpu::GpuMat& curFrame,
                           const cv::gpu::GpuMat& Ftd, const cv::gpu::GpuMat& Fbd,
                           cv::gpu::GpuMat& foreground, cv::gpu::GpuMat& countBuf,
                           const cv::gpu::FGDStatModel::Params& params, int out_cn)
    {
        typedef void (*func_t)(cv::gpu::PtrStepSzb prevFrame, cv::gpu::PtrStepSzb curFrame, cv::gpu::PtrStepSzb Ftd, cv::gpu::PtrStepSzb Fbd, cv::gpu::PtrStepSzb foreground,
                               int deltaC, int deltaCC, float alpha2, int N1c, int N1cc, cudaStream_t stream);
        static const func_t funcs[4][4][4] =
        {
            {
                {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}
            },
            {
                {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}
            },
            {
                {0,0,0,0}, {0,0,0,0},
                {0,0,bgfg::bgfgClassification_gpu<uchar3, uchar3, uchar3>,bgfg::bgfgClassification_gpu<uchar3, uchar3, uchar4>},
                {0,0,bgfg::bgfgClassification_gpu<uchar3, uchar4, uchar3>,bgfg::bgfgClassification_gpu<uchar3, uchar4, uchar4>}
            },
            {
                {0,0,0,0}, {0,0,0,0},
                {0,0,bgfg::bgfgClassification_gpu<uchar4, uchar3, uchar3>,bgfg::bgfgClassification_gpu<uchar4, uchar3, uchar4>},
                {0,0,bgfg::bgfgClassification_gpu<uchar4, uchar4, uchar3>,bgfg::bgfgClassification_gpu<uchar4, uchar4, uchar4>}
            }
        };

        const int deltaC  = cvRound(params.delta * 256 / params.Lc);
        const int deltaCC = cvRound(params.delta * 256 / params.Lcc);

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1][out_cn - 1](prevFrame, curFrame, Ftd, Fbd, foreground, deltaC, deltaCC, params.alpha2, params.N1c, params.N1cc, 0);

        int count = cv::gpu::countNonZero(foreground, countBuf);

        cv::gpu::multiply(foreground, cv::Scalar::all(255), foreground);

        return count;
    }
}

/////////////////////////////////////////////////////////////////////////
// smoothForeground

namespace
{
    void morphology(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst, cv::gpu::GpuMat& filterBrd, int brd, cv::Ptr<cv::gpu::FilterEngine_GPU>& filter, cv::Scalar brdVal)
    {
        cv::gpu::copyMakeBorder(src, filterBrd, brd, brd, brd, brd, cv::BORDER_CONSTANT, brdVal);
        filter->apply(filterBrd(cv::Rect(brd, brd, src.cols, src.rows)), dst, cv::Rect(0, 0, src.cols, src.rows));
    }

    void smoothForeground(cv::gpu::GpuMat& foreground, cv::gpu::GpuMat& filterBrd, cv::gpu::GpuMat& buf,
                          cv::Ptr<cv::gpu::FilterEngine_GPU>& erodeFilter, cv::Ptr<cv::gpu::FilterEngine_GPU>& dilateFilter,
                          const cv::gpu::FGDStatModel::Params& params)
    {
        const int brd = params.perform_morphing;

        const cv::Scalar erodeBrdVal = cv::Scalar::all(UCHAR_MAX);
        const cv::Scalar dilateBrdVal = cv::Scalar::all(0);

        // MORPH_OPEN
        morphology(foreground, buf, filterBrd, brd, erodeFilter, erodeBrdVal);
        morphology(buf, foreground, filterBrd, brd, dilateFilter, dilateBrdVal);

        // MORPH_CLOSE
        morphology(foreground, buf, filterBrd, brd, dilateFilter, dilateBrdVal);
        morphology(buf, foreground, filterBrd, brd, erodeFilter, erodeBrdVal);
    }
}

/////////////////////////////////////////////////////////////////////////
// findForegroundRegions

namespace
{
    void seqToContours(CvSeq* _ccontours, CvMemStorage* storage, cv::OutputArrayOfArrays _contours)
    {
        cv::Seq<CvSeq*> all_contours(cvTreeToNodeSeq(_ccontours, sizeof(CvSeq), storage));

        size_t total = all_contours.size();

        _contours.create((int) total, 1, 0, -1, true);

        cv::SeqIterator<CvSeq*> it = all_contours.begin();
        for (size_t i = 0; i < total; ++i, ++it)
        {
            CvSeq* c = *it;
            ((CvContour*)c)->color = (int)i;
            _contours.create((int)c->total, 1, CV_32SC2, (int)i, true);
            cv::Mat ci = _contours.getMat((int)i);
            CV_Assert( ci.isContinuous() );
            cvCvtSeqToArray(c, ci.data);
        }
    }

    int findForegroundRegions(cv::gpu::GpuMat& d_foreground, cv::Mat& h_foreground, std::vector< std::vector<cv::Point> >& foreground_regions,
                              CvMemStorage* storage, const cv::gpu::FGDStatModel::Params& params)
    {
        int region_count = 0;

        // Discard under-size foreground regions:

        d_foreground.download(h_foreground);
        IplImage ipl_foreground = h_foreground;
        CvSeq* first_seq = 0;

        cvFindContours(&ipl_foreground, storage, &first_seq, sizeof(CvContour), CV_RETR_LIST);

        for (CvSeq* seq = first_seq; seq; seq = seq->h_next)
        {
            CvContour* cnt = reinterpret_cast<CvContour*>(seq);

            if (cnt->rect.width * cnt->rect.height < params.minArea || (params.is_obj_without_holes && CV_IS_SEQ_HOLE(seq)))
            {
                // Delete under-size contour:
                CvSeq* prev_seq = seq->h_prev;
                if (prev_seq)
                {
                    prev_seq->h_next = seq->h_next;

                    if (seq->h_next)
                        seq->h_next->h_prev = prev_seq;
                }
                else
                {
                    first_seq = seq->h_next;

                    if (seq->h_next)
                        seq->h_next->h_prev = NULL;
                }
            }
            else
            {
                region_count++;
            }
        }

        seqToContours(first_seq, storage, foreground_regions);
        h_foreground.setTo(0);

        cv::drawContours(h_foreground, foreground_regions, -1, cv::Scalar::all(255), -1);

        d_foreground.upload(h_foreground);

        return region_count;
    }
}

/////////////////////////////////////////////////////////////////////////
// updateBackgroundModel

namespace
{
    void updateBackgroundModel(const cv::gpu::GpuMat& prevFrame, const cv::gpu::GpuMat& curFrame, const cv::gpu::GpuMat& Ftd, const cv::gpu::GpuMat& Fbd,
                               const cv::gpu::GpuMat& foreground, cv::gpu::GpuMat& background,
                               const cv::gpu::FGDStatModel::Params& params)
    {
        typedef void (*func_t)(cv::gpu::PtrStepSzb prevFrame, cv::gpu::PtrStepSzb curFrame, cv::gpu::PtrStepSzb Ftd, cv::gpu::PtrStepSzb Fbd,
                               cv::gpu::PtrStepSzb foreground, cv::gpu::PtrStepSzb background,
                               int deltaC, int deltaCC, float alpha1, float alpha2, float alpha3, int N1c, int N1cc, int N2c, int N2cc, float T, cudaStream_t stream);
        static const func_t funcs[4][4][4] =
        {
            {
                {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}
            },
            {
                {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0}
            },
            {
                {0,0,0,0}, {0,0,0,0},
                {0,0,bgfg::updateBackgroundModel_gpu<uchar3, uchar3, uchar3>,bgfg::updateBackgroundModel_gpu<uchar3, uchar3, uchar4>},
                {0,0,bgfg::updateBackgroundModel_gpu<uchar3, uchar4, uchar3>,bgfg::updateBackgroundModel_gpu<uchar3, uchar4, uchar4>}
            },
            {
                {0,0,0,0}, {0,0,0,0},
                {0,0,bgfg::updateBackgroundModel_gpu<uchar4, uchar3, uchar3>,bgfg::updateBackgroundModel_gpu<uchar4, uchar3, uchar4>},
                {0,0,bgfg::updateBackgroundModel_gpu<uchar4, uchar4, uchar3>,bgfg::updateBackgroundModel_gpu<uchar4, uchar4, uchar4>}
            }
        };

        const int deltaC  = cvRound(params.delta * 256 / params.Lc);
        const int deltaCC = cvRound(params.delta * 256 / params.Lcc);

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1][background.channels() - 1](
                    prevFrame, curFrame, Ftd, Fbd, foreground, background,
                    deltaC, deltaCC, params.alpha1, params.alpha2, params.alpha3, params.N1c, params.N1cc, params.N2c, params.N2cc, params.T,
                    0);
    }
}

/////////////////////////////////////////////////////////////////////////
// Impl::update

int cv::gpu::FGDStatModel::Impl::update(const cv::gpu::GpuMat& curFrame)
{
    CV_Assert(curFrame.type() == CV_8UC3 || curFrame.type() == CV_8UC4);
    CV_Assert(curFrame.size() == prevFrame_.size());

    cvClearMemStorage(storage_);
    foreground_regions_.clear();
    foreground_.setTo(cv::Scalar::all(0));

    changeDetection(prevFrame_, curFrame, Ftd_, hist_, histBuf_);
    changeDetection(background_, curFrame, Fbd_, hist_, histBuf_);

    int FG_pixels_count = bgfgClassification(prevFrame_, curFrame, Ftd_, Fbd_, foreground_, countBuf_, params_, out_cn_);

    if (params_.perform_morphing > 0)
        smoothForeground(foreground_, filterBrd_, buf_, erodeFilter_, dilateFilter_, params_);

    int region_count = 0;
    if (params_.minArea > 0 || params_.is_obj_without_holes)
        region_count = findForegroundRegions(foreground_, h_foreground_, foreground_regions_, storage_, params_);

    // Check ALL BG update condition:
    const double BGFG_FGD_BG_UPDATE_TRESH = 0.5;
    if (static_cast<double>(FG_pixels_count) / Ftd_.size().area() > BGFG_FGD_BG_UPDATE_TRESH)
        stat_.setTrained();

    updateBackgroundModel(prevFrame_, curFrame, Ftd_, Fbd_, foreground_, background_, params_);

    copyChannels(curFrame, prevFrame_);

    return region_count;
}

namespace
{
    // Default parameters of foreground detection algorithm:
    const int BGFG_FGD_LC  = 128;
    const int BGFG_FGD_N1C = 15;
    const int BGFG_FGD_N2C = 25;

    const int BGFG_FGD_LCC   = 64;
    const int BGFG_FGD_N1CC = 25;
    const int BGFG_FGD_N2CC = 40;

    // Background reference image update parameter:
    const float BGFG_FGD_ALPHA_1 = 0.1f;

    // stat model update parameter
    // 0.002f ~ 1K frame(~45sec), 0.005 ~ 18sec (if 25fps and absolutely static BG)
    const float BGFG_FGD_ALPHA_2 = 0.005f;

    // start value for alpha parameter (to fast initiate statistic model)
    const float BGFG_FGD_ALPHA_3 = 0.1f;

    const float BGFG_FGD_DELTA = 2.0f;

    const float BGFG_FGD_T = 0.9f;

    const float BGFG_FGD_MINAREA= 15.0f;
}

cv::gpu::FGDStatModel::Params::Params()
{
    Lc      = BGFG_FGD_LC;
    N1c     = BGFG_FGD_N1C;
    N2c     = BGFG_FGD_N2C;

    Lcc     = BGFG_FGD_LCC;
    N1cc    = BGFG_FGD_N1CC;
    N2cc    = BGFG_FGD_N2CC;

    delta   = BGFG_FGD_DELTA;

    alpha1  = BGFG_FGD_ALPHA_1;
    alpha2  = BGFG_FGD_ALPHA_2;
    alpha3  = BGFG_FGD_ALPHA_3;

    T       = BGFG_FGD_T;
    minArea = BGFG_FGD_MINAREA;

    is_obj_without_holes = true;
    perform_morphing     = 1;
}

cv::gpu::FGDStatModel::FGDStatModel(int out_cn)
{
    impl_.reset(new Impl(background, foreground, foreground_regions, out_cn));
}

cv::gpu::FGDStatModel::FGDStatModel(const cv::gpu::GpuMat& firstFrame, const Params& params, int out_cn)
{
    impl_.reset(new Impl(background, foreground, foreground_regions, out_cn));
    create(firstFrame, params);
}

cv::gpu::FGDStatModel::~FGDStatModel()
{
}

void cv::gpu::FGDStatModel::create(const cv::gpu::GpuMat& firstFrame, const Params& params)
{
    impl_->create(firstFrame, params);
}

void cv::gpu::FGDStatModel::release()
{
    impl_->release();
}

int cv::gpu::FGDStatModel::update(const cv::gpu::GpuMat& curFrame)
{
    return impl_->update(curFrame);
}

#endif // HAVE_CUDA
