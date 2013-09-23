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

#if !defined(HAVE_CUDA) || defined(CUDA_DISABLER) || !defined(HAVE_OPENCV_IMGPROC) || !defined(HAVE_OPENCV_CUDAARITHM) || !defined(HAVE_OPENCV_CUDAIMGPROC)

cv::cuda::FGDParams::FGDParams() { throw_no_cuda(); }

Ptr<cuda::BackgroundSubtractorFGD> cv::cuda::createBackgroundSubtractorFGD(const FGDParams&) { throw_no_cuda(); return Ptr<cuda::BackgroundSubtractorFGD>(); }

#else

#include "cuda/fgd.hpp"
#include "opencv2/imgproc/imgproc_c.h"

/////////////////////////////////////////////////////////////////////////
// FGDParams

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

cv::cuda::FGDParams::FGDParams()
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

/////////////////////////////////////////////////////////////////////////
// copyChannels

namespace
{
    void copyChannels(const GpuMat& src, GpuMat& dst, int dst_cn = -1)
    {
        const int src_cn = src.channels();

        if (dst_cn < 0)
            dst_cn = src_cn;

        cuda::ensureSizeIsEnough(src.size(), CV_MAKE_TYPE(src.depth(), dst_cn), dst);

        if (src_cn == dst_cn)
        {
            src.copyTo(dst);
        }
        else
        {
            static const int cvt_codes[4][4] =
            {
                {-1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
                {-1, -1, -1, -1},
                {COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
                {COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1}
            };

            const int cvt_code = cvt_codes[src_cn - 1][dst_cn - 1];
            CV_DbgAssert( cvt_code >= 0 );

            cuda::cvtColor(src, dst, cvt_code, dst_cn);
        }
    }
}

/////////////////////////////////////////////////////////////////////////
// changeDetection

namespace
{
    void calcDiffHistogram(const GpuMat& prevFrame, const GpuMat& curFrame, GpuMat& hist, GpuMat& histBuf)
    {
        typedef void (*func_t)(PtrStepSzb prevFrame, PtrStepSzb curFrame,
                               unsigned int* hist0, unsigned int* hist1, unsigned int* hist2,
                               unsigned int* partialBuf0, unsigned int* partialBuf1, unsigned int* partialBuf2,
                               bool cc20, cudaStream_t stream);
        static const func_t funcs[4][4] =
        {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,fgd::calcDiffHistogram_gpu<uchar3, uchar3>,fgd::calcDiffHistogram_gpu<uchar3, uchar4>},
            {0,0,fgd::calcDiffHistogram_gpu<uchar4, uchar3>,fgd::calcDiffHistogram_gpu<uchar4, uchar4>}
        };

        hist.create(3, 256, CV_32SC1);
        histBuf.create(3, fgd::PARTIAL_HISTOGRAM_COUNT * fgd::HISTOGRAM_BIN_COUNT, CV_32SC1);

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1](
                    prevFrame, curFrame,
                    hist.ptr<unsigned int>(0), hist.ptr<unsigned int>(1), hist.ptr<unsigned int>(2),
                    histBuf.ptr<unsigned int>(0), histBuf.ptr<unsigned int>(1), histBuf.ptr<unsigned int>(2),
                    deviceSupports(FEATURE_SET_COMPUTE_20), 0);
    }

    void calcRelativeVariance(unsigned int hist[3 * 256], double relativeVariance[3][fgd::HISTOGRAM_BIN_COUNT])
    {
        std::memset(relativeVariance, 0, 3 * fgd::HISTOGRAM_BIN_COUNT * sizeof(double));

        for (int thres = fgd::HISTOGRAM_BIN_COUNT - 2; thres >= 0; --thres)
        {
            Vec3d sum(0.0, 0.0, 0.0);
            Vec3d sqsum(0.0, 0.0, 0.0);
            Vec3i count(0, 0, 0);

            for (int j = thres; j < fgd::HISTOGRAM_BIN_COUNT; ++j)
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

            Vec3d my(
                sum[0] / count[0],
                sum[1] / count[1],
                sum[2] / count[2]
            );

            relativeVariance[0][thres] = std::sqrt(sqsum[0] / count[0] - my[0] * my[0]);
            relativeVariance[1][thres] = std::sqrt(sqsum[1] / count[1] - my[1] * my[1]);
            relativeVariance[2][thres] = std::sqrt(sqsum[2] / count[2] - my[2] * my[2]);
        }
    }

    void calcDiffThreshMask(const GpuMat& prevFrame, const GpuMat& curFrame, Vec3d bestThres, GpuMat& changeMask)
    {
        typedef void (*func_t)(PtrStepSzb prevFrame, PtrStepSzb curFrame, uchar3 bestThres, PtrStepSzb changeMask, cudaStream_t stream);
        static const func_t funcs[4][4] =
        {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,fgd::calcDiffThreshMask_gpu<uchar3, uchar3>,fgd::calcDiffThreshMask_gpu<uchar3, uchar4>},
            {0,0,fgd::calcDiffThreshMask_gpu<uchar4, uchar3>,fgd::calcDiffThreshMask_gpu<uchar4, uchar4>}
        };

        changeMask.setTo(Scalar::all(0));

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1](prevFrame, curFrame,
                                                                 make_uchar3((uchar)bestThres[0], (uchar)bestThres[1], (uchar)bestThres[2]),
                                                                 changeMask, 0);
    }

    // performs change detection for Foreground detection algorithm
    void changeDetection(const GpuMat& prevFrame, const GpuMat& curFrame, GpuMat& changeMask, GpuMat& hist, GpuMat& histBuf)
    {
        calcDiffHistogram(prevFrame, curFrame, hist, histBuf);

        unsigned int histData[3 * 256];
        Mat h_hist(3, 256, CV_32SC1, histData);
        hist.download(h_hist);

        double relativeVariance[3][fgd::HISTOGRAM_BIN_COUNT];
        calcRelativeVariance(histData, relativeVariance);

        // Find maximum:
        Vec3d bestThres(10.0, 10.0, 10.0);
        for (int i = 0; i < fgd::HISTOGRAM_BIN_COUNT; ++i)
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
    int bgfgClassification(const GpuMat& prevFrame, const GpuMat& curFrame,
                           const GpuMat& Ftd, const GpuMat& Fbd,
                           GpuMat& foreground, GpuMat& countBuf,
                           const FGDParams& params, int out_cn)
    {
        typedef void (*func_t)(PtrStepSzb prevFrame, PtrStepSzb curFrame, PtrStepSzb Ftd, PtrStepSzb Fbd, PtrStepSzb foreground,
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
                {0,0,fgd::bgfgClassification_gpu<uchar3, uchar3, uchar3>,fgd::bgfgClassification_gpu<uchar3, uchar3, uchar4>},
                {0,0,fgd::bgfgClassification_gpu<uchar3, uchar4, uchar3>,fgd::bgfgClassification_gpu<uchar3, uchar4, uchar4>}
            },
            {
                {0,0,0,0}, {0,0,0,0},
                {0,0,fgd::bgfgClassification_gpu<uchar4, uchar3, uchar3>,fgd::bgfgClassification_gpu<uchar4, uchar3, uchar4>},
                {0,0,fgd::bgfgClassification_gpu<uchar4, uchar4, uchar3>,fgd::bgfgClassification_gpu<uchar4, uchar4, uchar4>}
            }
        };

        const int deltaC  = cvRound(params.delta * 256 / params.Lc);
        const int deltaCC = cvRound(params.delta * 256 / params.Lcc);

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1][out_cn - 1](prevFrame, curFrame, Ftd, Fbd, foreground,
                                                                             deltaC, deltaCC, params.alpha2,
                                                                             params.N1c, params.N1cc, 0);

        int count = cuda::countNonZero(foreground, countBuf);

        cuda::multiply(foreground, Scalar::all(255), foreground);

        return count;
    }
}

/////////////////////////////////////////////////////////////////////////
// smoothForeground

#ifdef HAVE_OPENCV_CUDAFILTERS

namespace
{
    void morphology(const GpuMat& src, GpuMat& dst, GpuMat& filterBrd, int brd, Ptr<cuda::Filter>& filter, Scalar brdVal)
    {
        cuda::copyMakeBorder(src, filterBrd, brd, brd, brd, brd, BORDER_CONSTANT, brdVal);
        filter->apply(filterBrd(Rect(brd, brd, src.cols, src.rows)), dst);
    }

    void smoothForeground(GpuMat& foreground, GpuMat& filterBrd, GpuMat& buf,
                          Ptr<cuda::Filter>& erodeFilter, Ptr<cuda::Filter>& dilateFilter,
                          const FGDParams& params)
    {
        const int brd = params.perform_morphing;

        const Scalar erodeBrdVal = Scalar::all(UCHAR_MAX);
        const Scalar dilateBrdVal = Scalar::all(0);

        // MORPH_OPEN
        morphology(foreground, buf, filterBrd, brd, erodeFilter, erodeBrdVal);
        morphology(buf, foreground, filterBrd, brd, dilateFilter, dilateBrdVal);

        // MORPH_CLOSE
        morphology(foreground, buf, filterBrd, brd, dilateFilter, dilateBrdVal);
        morphology(buf, foreground, filterBrd, brd, erodeFilter, erodeBrdVal);
    }
}

#endif

/////////////////////////////////////////////////////////////////////////
// findForegroundRegions

namespace
{
    void seqToContours(CvSeq* _ccontours, CvMemStorage* storage, OutputArrayOfArrays _contours)
    {
        Seq<CvSeq*> all_contours(cvTreeToNodeSeq(_ccontours, sizeof(CvSeq), storage));

        size_t total = all_contours.size();

        _contours.create((int) total, 1, 0, -1, true);

        SeqIterator<CvSeq*> it = all_contours.begin();
        for (size_t i = 0; i < total; ++i, ++it)
        {
            CvSeq* c = *it;
            ((CvContour*)c)->color = (int)i;
            _contours.create((int)c->total, 1, CV_32SC2, (int)i, true);
            Mat ci = _contours.getMat((int)i);
            CV_Assert( ci.isContinuous() );
            cvCvtSeqToArray(c, ci.data);
        }
    }

    int findForegroundRegions(GpuMat& d_foreground, Mat& h_foreground, std::vector< std::vector<Point> >& foreground_regions,
                              CvMemStorage* storage, const FGDParams& params)
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

        drawContours(h_foreground, foreground_regions, -1, Scalar::all(255), -1);

        d_foreground.upload(h_foreground);

        return region_count;
    }
}

/////////////////////////////////////////////////////////////////////////
// updateBackgroundModel

namespace
{
    void updateBackgroundModel(const GpuMat& prevFrame, const GpuMat& curFrame, const GpuMat& Ftd, const GpuMat& Fbd,
                               const GpuMat& foreground, GpuMat& background,
                               const FGDParams& params)
    {
        typedef void (*func_t)(PtrStepSzb prevFrame, PtrStepSzb curFrame, PtrStepSzb Ftd, PtrStepSzb Fbd,
                               PtrStepSzb foreground, PtrStepSzb background,
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
                {0,0,fgd::updateBackgroundModel_gpu<uchar3, uchar3, uchar3>,fgd::updateBackgroundModel_gpu<uchar3, uchar3, uchar4>},
                {0,0,fgd::updateBackgroundModel_gpu<uchar3, uchar4, uchar3>,fgd::updateBackgroundModel_gpu<uchar3, uchar4, uchar4>}
            },
            {
                {0,0,0,0}, {0,0,0,0},
                {0,0,fgd::updateBackgroundModel_gpu<uchar4, uchar3, uchar3>,fgd::updateBackgroundModel_gpu<uchar4, uchar3, uchar4>},
                {0,0,fgd::updateBackgroundModel_gpu<uchar4, uchar4, uchar3>,fgd::updateBackgroundModel_gpu<uchar4, uchar4, uchar4>}
            }
        };

        const int deltaC  = cvRound(params.delta * 256 / params.Lc);
        const int deltaCC = cvRound(params.delta * 256 / params.Lcc);

        funcs[prevFrame.channels() - 1][curFrame.channels() - 1][background.channels() - 1](
                    prevFrame, curFrame, Ftd, Fbd, foreground, background,
                    deltaC, deltaCC, params.alpha1, params.alpha2, params.alpha3,
                    params.N1c, params.N1cc, params.N2c, params.N2cc, params.T,
                    0);
    }
}


namespace
{
    class BGPixelStat
    {
    public:
        void create(Size size, const FGDParams& params);

        void setTrained();

        operator fgd::BGPixelStat();

    private:
        GpuMat Pbc_;
        GpuMat Pbcc_;
        GpuMat is_trained_st_model_;
        GpuMat is_trained_dyn_model_;

        GpuMat ctable_Pv_;
        GpuMat ctable_Pvb_;
        GpuMat ctable_v_;

        GpuMat cctable_Pv_;
        GpuMat cctable_Pvb_;
        GpuMat cctable_v1_;
        GpuMat cctable_v2_;
    };

    void BGPixelStat::create(Size size, const FGDParams& params)
    {
        cuda::ensureSizeIsEnough(size, CV_32FC1, Pbc_);
        Pbc_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(size, CV_32FC1, Pbcc_);
        Pbcc_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(size, CV_8UC1, is_trained_st_model_);
        is_trained_st_model_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(size, CV_8UC1, is_trained_dyn_model_);
        is_trained_dyn_model_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(params.N2c * size.height, size.width, CV_32FC1, ctable_Pv_);
        ctable_Pv_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(params.N2c * size.height, size.width, CV_32FC1, ctable_Pvb_);
        ctable_Pvb_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(params.N2c * size.height, size.width, CV_8UC4, ctable_v_);
        ctable_v_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_32FC1, cctable_Pv_);
        cctable_Pv_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_32FC1, cctable_Pvb_);
        cctable_Pvb_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_8UC4, cctable_v1_);
        cctable_v1_.setTo(Scalar::all(0));

        cuda::ensureSizeIsEnough(params.N2cc * size.height, size.width, CV_8UC4, cctable_v2_);
        cctable_v2_.setTo(Scalar::all(0));
    }

    void BGPixelStat::setTrained()
    {
        is_trained_st_model_.setTo(Scalar::all(1));
        is_trained_dyn_model_.setTo(Scalar::all(1));
    }

    BGPixelStat::operator fgd::BGPixelStat()
    {
        fgd::BGPixelStat stat;

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

    class FGDImpl : public cuda::BackgroundSubtractorFGD
    {
    public:
        explicit FGDImpl(const FGDParams& params);
        ~FGDImpl();

        void apply(InputArray image, OutputArray fgmask, double learningRate=-1);

        void getBackgroundImage(OutputArray backgroundImage) const;

        void getForegroundRegions(OutputArrayOfArrays foreground_regions);

    private:
        void initialize(const GpuMat& firstFrame);

        FGDParams params_;
        Size frameSize_;

        GpuMat background_;
        GpuMat foreground_;
        std::vector< std::vector<Point> > foreground_regions_;

        Mat h_foreground_;

        GpuMat prevFrame_;
        GpuMat Ftd_;
        GpuMat Fbd_;
        BGPixelStat stat_;

        GpuMat hist_;
        GpuMat histBuf_;

        GpuMat countBuf_;

        GpuMat buf_;
        GpuMat filterBrd_;

#ifdef HAVE_OPENCV_CUDAFILTERS
        Ptr<cuda::Filter> dilateFilter_;
        Ptr<cuda::Filter> erodeFilter_;
#endif

        CvMemStorage* storage_;
    };

    FGDImpl::FGDImpl(const FGDParams& params) : params_(params), frameSize_(0, 0)
    {
        storage_ = cvCreateMemStorage();
        CV_Assert( storage_ != 0 );
    }

    FGDImpl::~FGDImpl()
    {
        cvReleaseMemStorage(&storage_);
    }

    void FGDImpl::apply(InputArray _frame, OutputArray fgmask, double)
    {
        GpuMat curFrame = _frame.getGpuMat();

        if (curFrame.size() != frameSize_)
        {
            initialize(curFrame);
            return;
        }

        CV_Assert( curFrame.type() == CV_8UC3 || curFrame.type() == CV_8UC4 );
        CV_Assert( curFrame.size() == prevFrame_.size() );

        cvClearMemStorage(storage_);
        foreground_regions_.clear();
        foreground_.setTo(Scalar::all(0));

        changeDetection(prevFrame_, curFrame, Ftd_, hist_, histBuf_);
        changeDetection(background_, curFrame, Fbd_, hist_, histBuf_);

        int FG_pixels_count = bgfgClassification(prevFrame_, curFrame, Ftd_, Fbd_, foreground_, countBuf_, params_, 4);

#ifdef HAVE_OPENCV_CUDAFILTERS
        if (params_.perform_morphing > 0)
            smoothForeground(foreground_, filterBrd_, buf_, erodeFilter_, dilateFilter_, params_);
#endif

        if (params_.minArea > 0 || params_.is_obj_without_holes)
            findForegroundRegions(foreground_, h_foreground_, foreground_regions_, storage_, params_);

        // Check ALL BG update condition:
        const double BGFG_FGD_BG_UPDATE_TRESH = 0.5;
        if (static_cast<double>(FG_pixels_count) / Ftd_.size().area() > BGFG_FGD_BG_UPDATE_TRESH)
            stat_.setTrained();

        updateBackgroundModel(prevFrame_, curFrame, Ftd_, Fbd_, foreground_, background_, params_);

        copyChannels(curFrame, prevFrame_, 4);

        foreground_.copyTo(fgmask);
    }

    void FGDImpl::getBackgroundImage(OutputArray backgroundImage) const
    {
        cuda::cvtColor(background_, backgroundImage, COLOR_BGRA2BGR);
    }

    void FGDImpl::getForegroundRegions(OutputArrayOfArrays dst)
    {
        size_t total = foreground_regions_.size();

        dst.create((int) total, 1, 0, -1, true);

        for (size_t i = 0; i < total; ++i)
        {
            std::vector<Point>& c = foreground_regions_[i];

            dst.create((int) c.size(), 1, CV_32SC2, (int) i, true);
            Mat ci = dst.getMat((int) i);

            Mat(ci.size(), ci.type(), &c[0]).copyTo(ci);
        }
    }

    void FGDImpl::initialize(const GpuMat& firstFrame)
    {
        CV_Assert( firstFrame.type() == CV_8UC3 || firstFrame.type() == CV_8UC4 );

        frameSize_ = firstFrame.size();

        cuda::ensureSizeIsEnough(firstFrame.size(), CV_8UC1, foreground_);

        copyChannels(firstFrame, background_, 4);
        copyChannels(firstFrame, prevFrame_, 4);

        cuda::ensureSizeIsEnough(firstFrame.size(), CV_8UC1, Ftd_);
        cuda::ensureSizeIsEnough(firstFrame.size(), CV_8UC1, Fbd_);

        stat_.create(firstFrame.size(), params_);
        fgd::setBGPixelStat(stat_);

#ifdef HAVE_OPENCV_CUDAFILTERS
        if (params_.perform_morphing > 0)
        {
            Mat kernel = getStructuringElement(MORPH_RECT, Size(1 + params_.perform_morphing * 2, 1 + params_.perform_morphing * 2));
            Point anchor(params_.perform_morphing, params_.perform_morphing);

            dilateFilter_ = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, kernel, anchor);
            erodeFilter_ = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, kernel, anchor);
        }
#endif
    }
}

Ptr<cuda::BackgroundSubtractorFGD> cv::cuda::createBackgroundSubtractorFGD(const FGDParams& params)
{
    return makePtr<FGDImpl>(params);
}

#endif // HAVE_CUDA
