// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "stat.hpp"

namespace cv
{

template<typename _Tp, typename _Rt>
void batchDistL1_(const _Tp* src1, const _Tp* src2, size_t step2,
                  int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normL1<_Tp, _Rt>(src1, src2 + step2*i, len);
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normL1<_Tp, _Rt>(src1, src2 + step2*i, len) : val0;
    }
}

template<typename _Tp, typename _Rt>
void batchDistL2Sqr_(const _Tp* src1, const _Tp* src2, size_t step2,
                     int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len);
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len) : val0;
    }
}

template<typename _Tp, typename _Rt>
void batchDistL2_(const _Tp* src1, const _Tp* src2, size_t step2,
                  int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = std::sqrt(normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len));
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? std::sqrt(normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len)) : val0;
    }
}

static void batchDistHamming(const uchar* src1, const uchar* src2, size_t step2,
                             int nvecs, int len, int* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
             dist[i] = hal::normHamming(src1, src2 + step2*i, len);
    }
    else
    {
        int val0 = INT_MAX;
        for( int i = 0; i < nvecs; i++ )
        {
            if (mask[i])
                dist[i] = hal::normHamming(src1, src2 + step2*i, len);
            else
                dist[i] = val0;
        }
    }
}

static void batchDistHamming2(const uchar* src1, const uchar* src2, size_t step2,
                              int nvecs, int len, int* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = hal::normHamming(src1, src2 + step2*i, len, 2);
    }
    else
    {
        int val0 = INT_MAX;
        for( int i = 0; i < nvecs; i++ )
        {
            if (mask[i])
                dist[i] = hal::normHamming(src1, src2 + step2*i, len, 2);
            else
                dist[i] = val0;
        }
    }
}

static void batchDistL1_8u32s(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, int* dist, const uchar* mask)
{
    batchDistL1_<uchar, int>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL1_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL1_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_8u32s(const uchar* src1, const uchar* src2, size_t step2,
                                  int nvecs, int len, int* dist, const uchar* mask)
{
    batchDistL2Sqr_<uchar, int>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                                  int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2Sqr_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL1_32f(const float* src1, const float* src2, size_t step2,
                             int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL1_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_32f(const float* src1, const float* src2, size_t step2,
                                int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2Sqr_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2_32f(const float* src1, const float* src2, size_t step2,
                             int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

typedef void (*BatchDistFunc)(const uchar* src1, const uchar* src2, size_t step2,
                              int nvecs, int len, uchar* dist, const uchar* mask);


struct BatchDistInvoker : public ParallelLoopBody
{
    BatchDistInvoker( const Mat& _src1, const Mat& _src2,
                      Mat& _dist, Mat& _nidx, int _K,
                      const Mat& _mask, int _update,
                      BatchDistFunc _func)
    {
        src1 = &_src1;
        src2 = &_src2;
        dist = &_dist;
        nidx = &_nidx;
        K = _K;
        mask = &_mask;
        update = _update;
        func = _func;
    }

    void operator()(const Range& range) const
    {
        AutoBuffer<int> buf(src2->rows);
        int* bufptr = buf;

        for( int i = range.start; i < range.end; i++ )
        {
            func(src1->ptr(i), src2->ptr(), src2->step, src2->rows, src2->cols,
                 K > 0 ? (uchar*)bufptr : dist->ptr(i), mask->data ? mask->ptr(i) : 0);

            if( K > 0 )
            {
                int* nidxptr = nidx->ptr<int>(i);
                // since positive float's can be compared just like int's,
                // we handle both CV_32S and CV_32F cases with a single branch
                int* distptr = (int*)dist->ptr(i);

                int j, k;

                for( j = 0; j < src2->rows; j++ )
                {
                    int d = bufptr[j];
                    if( d < distptr[K-1] )
                    {
                        for( k = K-2; k >= 0 && distptr[k] > d; k-- )
                        {
                            nidxptr[k+1] = nidxptr[k];
                            distptr[k+1] = distptr[k];
                        }
                        nidxptr[k+1] = j + update;
                        distptr[k+1] = d;
                    }
                }
            }
        }
    }

    const Mat *src1;
    const Mat *src2;
    Mat *dist;
    Mat *nidx;
    const Mat *mask;
    int K;
    int update;
    BatchDistFunc func;
};

}

void cv::batchDistance( InputArray _src1, InputArray _src2,
                        OutputArray _dist, int dtype, OutputArray _nidx,
                        int normType, int K, InputArray _mask,
                        int update, bool crosscheck )
{
    CV_INSTRUMENT_REGION()

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();
    int type = src1.type();
    CV_Assert( type == src2.type() && src1.cols == src2.cols &&
               (type == CV_32F || type == CV_8U));
    CV_Assert( _nidx.needed() == (K > 0) );

    if( dtype == -1 )
    {
        dtype = normType == NORM_HAMMING || normType == NORM_HAMMING2 ? CV_32S : CV_32F;
    }
    CV_Assert( (type == CV_8U && dtype == CV_32S) || dtype == CV_32F);

    K = std::min(K, src2.rows);

    _dist.create(src1.rows, (K > 0 ? K : src2.rows), dtype);
    Mat dist = _dist.getMat(), nidx;
    if( _nidx.needed() )
    {
        _nidx.create(dist.size(), CV_32S);
        nidx = _nidx.getMat();
    }

    if( update == 0 && K > 0 )
    {
        dist = Scalar::all(dtype == CV_32S ? (double)INT_MAX : (double)FLT_MAX);
        nidx = Scalar::all(-1);
    }

    if( crosscheck )
    {
        CV_Assert( K == 1 && update == 0 && mask.empty() );
        Mat tdist, tidx;
        batchDistance(src2, src1, tdist, dtype, tidx, normType, K, mask, 0, false);

        // if an idx-th element from src1 appeared to be the nearest to i-th element of src2,
        // we update the minimum mutual distance between idx-th element of src1 and the whole src2 set.
        // As a result, if nidx[idx] = i*, it means that idx-th element of src1 is the nearest
        // to i*-th element of src2 and i*-th element of src2 is the closest to idx-th element of src1.
        // If nidx[idx] = -1, it means that there is no such ideal couple for it in src2.
        // This O(N) procedure is called cross-check and it helps to eliminate some false matches.
        if( dtype == CV_32S )
        {
            for( int i = 0; i < tdist.rows; i++ )
            {
                int idx = tidx.at<int>(i);
                int d = tdist.at<int>(i), d0 = dist.at<int>(idx);
                if( d < d0 )
                {
                    dist.at<int>(idx) = d;
                    nidx.at<int>(idx) = i + update;
                }
            }
        }
        else
        {
            for( int i = 0; i < tdist.rows; i++ )
            {
                int idx = tidx.at<int>(i);
                float d = tdist.at<float>(i), d0 = dist.at<float>(idx);
                if( d < d0 )
                {
                    dist.at<float>(idx) = d;
                    nidx.at<int>(idx) = i + update;
                }
            }
        }
        return;
    }

    BatchDistFunc func = 0;
    if( type == CV_8U )
    {
        if( normType == NORM_L1 && dtype == CV_32S )
            func = (BatchDistFunc)batchDistL1_8u32s;
        else if( normType == NORM_L1 && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL1_8u32f;
        else if( normType == NORM_L2SQR && dtype == CV_32S )
            func = (BatchDistFunc)batchDistL2Sqr_8u32s;
        else if( normType == NORM_L2SQR && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL2Sqr_8u32f;
        else if( normType == NORM_L2 && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL2_8u32f;
        else if( normType == NORM_HAMMING && dtype == CV_32S )
            func = (BatchDistFunc)batchDistHamming;
        else if( normType == NORM_HAMMING2 && dtype == CV_32S )
            func = (BatchDistFunc)batchDistHamming2;
    }
    else if( type == CV_32F && dtype == CV_32F )
    {
        if( normType == NORM_L1 )
            func = (BatchDistFunc)batchDistL1_32f;
        else if( normType == NORM_L2SQR )
            func = (BatchDistFunc)batchDistL2Sqr_32f;
        else if( normType == NORM_L2 )
            func = (BatchDistFunc)batchDistL2_32f;
    }

    if( func == 0 )
        CV_Error_(CV_StsUnsupportedFormat,
                  ("The combination of type=%d, dtype=%d and normType=%d is not supported",
                   type, dtype, normType));

    parallel_for_(Range(0, src1.rows),
                  BatchDistInvoker(src1, src2, dist, nidx, K, mask, update, func));
}
