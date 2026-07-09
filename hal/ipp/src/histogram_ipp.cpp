// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utils/tls.hpp>
#include "precomp_ipp.hpp"

#if IPP_VERSION_X100 >= 700

#define IPP_HISTOGRAM_PARALLEL 1

namespace cv { namespace ipp { unsigned long long getIppTopFeatures(); } }

using namespace cv;

namespace {

typedef IppStatus(CV_STDCALL * IppiHistogram_C1)(const void* pSrc, int srcStep,
    IppiSize roiSize, Ipp32u* pHist, const IppiHistogramSpec* pSpec, Ipp8u* pBuffer);

static IppiHistogram_C1 getIppiHistogramFunction_C1(int type)
{
    IppiHistogram_C1 ippFunction =
        (type == CV_8UC1) ? (IppiHistogram_C1)ippiHistogram_8u_C1R :
        (type == CV_16UC1) ? (IppiHistogram_C1)ippiHistogram_16u_C1R :
        (type == CV_32FC1) ? (IppiHistogram_C1)ippiHistogram_32f_C1R :
        NULL;

    return ippFunction;
}

class ipp_calcHistParallelTLS
{
public:
    ipp_calcHistParallelTLS() {}

    IppAutoBuffer<IppiHistogramSpec> spec;
    IppAutoBuffer<Ipp8u>  buffer;
    IppAutoBuffer<Ipp32u> thist;
};

class ipp_calcHistParallel: public ParallelLoopBody
{
public:
    ipp_calcHistParallel(const Mat &src, Mat &hist, Ipp32s histSize, const float *ranges, bool uniform, bool &ok):
        ParallelLoopBody(), m_src(src), m_hist(hist), m_ok(ok)
    {
        ok = true;

        m_uniform        = uniform;
        m_ranges         = ranges;
        m_histSize       = histSize;
        m_type           = ippiGetDataType(src.type());
        m_levelsNum      = histSize+1;
        ippiHistogram_C1 = getIppiHistogramFunction_C1(src.type());
        m_fullRoi    = ippiSize(src.size());
        m_bufferSize = 0;
        m_specSize   = 0;
        if(!ippiHistogram_C1)
        {
            ok = false;
            return;
        }

        if(ippiHistogramGetBufferSize(m_type, m_fullRoi, &m_levelsNum, 1, 1, &m_specSize, &m_bufferSize) < 0)
        {
            ok = false;
            return;
        }

        hist.setTo(0);
    }

    virtual void operator() (const Range & range) const CV_OVERRIDE
    {
        if(!m_ok)
            return;

        ipp_calcHistParallelTLS *pTls = m_tls.get();

        IppiSize roi = {m_src.cols, range.end - range.start };
        bool     mtLoop = false;
        if(m_fullRoi.height != roi.height)
            mtLoop = true;

        if(!pTls->spec)
        {
            pTls->spec.allocate(m_specSize);
            if(!pTls->spec.get())
            {
                m_ok = false;
                return;
            }

            pTls->buffer.allocate(m_bufferSize);
            if(!pTls->buffer.get() && m_bufferSize)
            {
                m_ok = false;
                return;
            }

            if(m_uniform)
            {
                if(ippiHistogramUniformInit(m_type, (Ipp32f*)&m_ranges[0], (Ipp32f*)&m_ranges[1], (Ipp32s*)&m_levelsNum, 1, pTls->spec) < 0)
                {
                    m_ok = false;
                    return;
                }
            }
            else
            {
                if(ippiHistogramInit(m_type, (const Ipp32f**)&m_ranges, (Ipp32s*)&m_levelsNum, 1, pTls->spec) < 0)
                {
                    m_ok = false;
                    return;
                }
            }

            pTls->thist.allocate(m_histSize*sizeof(Ipp32u));
        }

        if(CV_INSTRUMENT_FUN_IPP(ippiHistogram_C1, m_src.ptr(range.start), (int)m_src.step, roi, pTls->thist, pTls->spec, pTls->buffer) < 0)
        {
            m_ok = false;
            return;
        }

        if(mtLoop)
        {
            for(int i = 0; i < m_histSize; i++)
                CV_XADD((int*)(m_hist.ptr(i)), *(int*)((Ipp32u*)pTls->thist + i));
        }
        else
            ippiCopy_32s_C1R((Ipp32s*)pTls->thist.get(), sizeof(Ipp32u), (Ipp32s*)m_hist.ptr(), (int)m_hist.step, ippiSize(1, m_histSize));
    }

private:
    const Mat      &m_src;
    Mat            &m_hist;
    Ipp32s          m_histSize;
    const float    *m_ranges;
    bool            m_uniform;

    IppiHistogram_C1    ippiHistogram_C1;
    IppiSize            m_fullRoi;
    IppDataType         m_type;
    Ipp32s              m_levelsNum;
    int                 m_bufferSize;
    int                 m_specSize;

    mutable Mutex                    m_syncMutex;
    TLSData<ipp_calcHistParallelTLS> m_tls;

    volatile bool &m_ok;
    const ipp_calcHistParallel & operator = (const ipp_calcHistParallel & );
};

static bool ipp_calchist(const Mat &image, Mat &hist, int histSize, const float** ranges, bool uniform, bool accumulate)
{
#if IPP_VERSION_X100 < 201801
    // No SSE42 optimization for uniform 32f
    if(uniform && image.depth() == CV_32F && cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

    // IPP_DISABLE_HISTOGRAM - https://github.com/opencv/opencv/issues/11544
    // and https://github.com/opencv/opencv/issues/21595
    if ((uniform && (ranges[0][1] - ranges[0][0]) != histSize) || abs(ranges[0][0]) != cvFloor(ranges[0][0]))
        return false;

    Mat ihist = hist;
    if(accumulate)
        ihist.create(1, &histSize, CV_32S);

    bool  ok      = true;
    int   threads = ippiSuggestThreadsNum(image.cols, image.rows, image.elemSize(), (1+((double)ihist.total()/image.total()))*2);
    Range range(0, image.rows);
    ipp_calcHistParallel invoker(image, ihist, histSize, ranges[0], uniform, ok);
    if(!ok)
        return false;

    if(IPP_HISTOGRAM_PARALLEL && threads > 1)
        parallel_for_(range, invoker, threads*2);
    else
        invoker(range);

    if(ok)
    {
        if(accumulate)
        {
            IppiSize histRoi = ippiSize(1, histSize);
            IppAutoBuffer<Ipp32f> fhist(histSize*sizeof(Ipp32f));
            CV_INSTRUMENT_FUN_IPP(ippiConvert_32s32f_C1R, (Ipp32s*)ihist.ptr(), (int)ihist.step, (Ipp32f*)fhist, sizeof(Ipp32f), histRoi);
            CV_INSTRUMENT_FUN_IPP(ippiAdd_32f_C1IR, (Ipp32f*)fhist, sizeof(Ipp32f), (Ipp32f*)hist.ptr(), (int)hist.step, histRoi);
        }
        else
            CV_INSTRUMENT_FUN_IPP(ippiConvert_32s32f_C1R, (Ipp32s*)ihist.ptr(), (int)ihist.step, (Ipp32f*)hist.ptr(), (int)hist.step, ippiSize(1, histSize));
    }
    return ok;
}

} // namespace

int ipp_hal_calcHist(const uchar* src_data, size_t src_step, int src_type, int src_width, int src_height,
                     float* hist_data, int hist_size, const float** ranges, bool uniform, bool accumulate)
{
    CV_HAL_CHECK_USE_IPP();

    Mat image(src_height, src_width, src_type, (void*)src_data, src_step);
    Mat hist(1, &hist_size, CV_32F, (void*)hist_data);

    if(ipp_calchist(image, hist, hist_size, ranges, uniform, accumulate))
        return CV_HAL_ERROR_OK;

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif
