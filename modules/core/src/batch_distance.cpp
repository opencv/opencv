// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "stat.hpp"
#include <opencv2/core/hal/hal.hpp>

namespace cv
{

template<typename _Tp, typename _Rt>
static void batchDistL1_(const void* _src1, const void* _src2, size_t step2,
                         int nvecs, int len, void* _dist, const uchar* mask)
{
    const _Tp* src1 = (const _Tp*)_src1;
    const _Tp* src2 = (const _Tp*)_src2;
    _Rt* dist = (_Rt*)_dist;
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
static void batchDistL2Sqr_(const void* _src1, const void* _src2, size_t step2,
                            int nvecs, int len, void* _dist, const uchar* mask)
{
    const _Tp* src1 = (const _Tp*)_src1;
    const _Tp* src2 = (const _Tp*)_src2;
    _Rt* dist = (_Rt*)_dist;
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

template<>
void batchDistL2Sqr_<float, float>(const void* _src1, const void* _src2, size_t step2,
                                   int nvecs, int len, void* _dist, const uchar* mask)
{
    const float* src1 = (const float*)_src1;
    const float* src2 = (const float*)_src2;
    float* dist = (float*)_dist;
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = hal::normL2Sqr_(src1, src2 + step2*i, len);
    }
    else
    {
        float val0 = std::numeric_limits<float>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? hal::normL2Sqr_(src1, src2 + step2*i, len) : val0;
    }
}

template<typename _Tp, typename _Rt>
static void batchDistL2_(const void* _src1, const void* _src2, size_t step2,
                         int nvecs, int len, void* _dist, const uchar* mask)
{
    const _Tp* src1 = (const _Tp*)_src1;
    const _Tp* src2 = (const _Tp*)_src2;
    _Rt* dist = (_Rt*)_dist;
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

template<>
void batchDistL2_<float, float>(const void* _src1, const void* _src2, size_t step2,
                                int nvecs, int len, void* _dist, const uchar* mask)
{
    const float* src1 = (const float*)_src1;
    const float* src2 = (const float*)_src2;
    float* dist = (float*)_dist;
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = std::sqrt(hal::normL2Sqr_(src1, src2 + step2*i, len));
    }
    else
    {
        float val0 = std::numeric_limits<float>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? std::sqrt(hal::normL2Sqr_(src1, src2 + step2*i, len)) : val0;
    }
}

static void batchDistHamming(const void* _src1, const void* _src2, size_t step2,
                             int nvecs, int len, void* _dist, const uchar* mask)
{
    const uchar* src1 = (const uchar*)_src1;
    const uchar* src2 = (const uchar*)_src2;
    int* dist = (int*)_dist;
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

static void batchDistHamming2(const void* _src1, const void* _src2, size_t step2,
                              int nvecs, int len, void* _dist, const uchar* mask)
{
    const uchar* src1 = (const uchar*)_src1;
    const uchar* src2 = (const uchar*)_src2;
    int* dist = (int*)_dist;
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

typedef void (*BatchDistFunc)(const void* src1, const void* src2, size_t step2,
                              int nvecs, int len, void* dist, const uchar* mask);


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

    void operator()(const Range& range) const CV_OVERRIDE
    {
        AutoBuffer<int> buf(src2->rows);
        int* bufptr = buf.data();

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

// Both nearest-neighbor searches use the same symmetric pairwise distances.
// Split the larger descriptor set so the partial minima occupy the smaller side.
template<typename T>
struct BatchDistCrossCheckInvoker : public ParallelLoopBody
{
    BatchDistCrossCheckInvoker(const Mat& _src1, const Mat& _src2,
                              Mat& _qdist, Mat& _qidx, Mat& _tdist, Mat& _tidx,
                              BatchDistFunc _func, bool _splitQueries)
        : src1(_src1), src2(_src2), qdist(_qdist), qidx(_qidx),
          tdist(_tdist), tidx(_tidx), func(_func), splitQueries(_splitQueries)
    {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        const int stripes = splitQueries ? tdist.rows : qdist.rows;
        AutoBuffer<T> buf(splitQueries ? src2.rows : (src2.rows - 1) / stripes + 1);
        for (int stripe = range.start; stripe < range.end; ++stripe)
        {
            const int qbegin = splitQueries ? (int)((int64)src1.rows * stripe / stripes) : 0;
            const int qend = splitQueries ? (int)((int64)src1.rows * (stripe + 1) / stripes) : src1.rows;
            const int tbegin = splitQueries ? 0 : (int)((int64)src2.rows * stripe / stripes);
            const int tend = splitQueries ? src2.rows : (int)((int64)src2.rows * (stripe + 1) / stripes);
            T* qd = qdist.ptr<T>(splitQueries ? 0 : stripe);
            int* qi = qidx.ptr<int>(splitQueries ? 0 : stripe);
            T* td = tdist.ptr<T>(splitQueries ? stripe : 0);
            int* ti = tidx.ptr<int>(splitQueries ? stripe : 0);

            for (int q = qbegin; q < qend; ++q)
            {
                func(src1.ptr(q), src2.ptr(tbegin), src2.step, tend - tbegin,
                     src1.cols, buf.data(), 0);
                T best = qd[q];
                int bestIdx = qi[q];
                for (int t = tbegin; t < tend; ++t)
                {
                    const T d = buf[t - tbegin];
                    if (d < best)
                    {
                        best = d;
                        bestIdx = t;
                    }
                    if (d < td[t])
                    {
                        td[t] = d;
                        ti[t] = q;
                    }
                }
                qd[q] = best;
                qi[q] = bestIdx;
            }
        }
    }

    const Mat& src1;
    const Mat& src2;
    Mat& qdist;
    Mat& qidx;
    Mat& tdist;
    Mat& tidx;
    BatchDistFunc func;
    bool splitQueries;
};

template<typename T>
static void batchDistCrossCheck(const Mat& src1, const Mat& src2,
                               Mat& dist, Mat& nidx, BatchDistFunc func)
{
    const bool splitQueries = src1.rows >= src2.rows;
    const int stripes = std::min(std::max(src1.rows, src2.rows), std::max(1, getNumThreads()));
    const T maxDist = std::numeric_limits<T>::max();
    Mat qdist(splitQueries ? 1 : stripes, src1.rows, dist.type(), Scalar::all(maxDist));
    Mat qidx(qdist.size(), CV_32S, Scalar::all(-1));
    Mat tdist(splitQueries ? stripes : 1, src2.rows, dist.type(), Scalar::all(maxDist));
    Mat tidx(tdist.size(), CV_32S, Scalar::all(-1));
    parallel_for_(Range(0, stripes),
                  BatchDistCrossCheckInvoker<T>(src1, src2, qdist, qidx, tdist, tidx, func, splitQueries));

    Mat& partialDist = splitQueries ? tdist : qdist;
    Mat& partialIdx = splitQueries ? tidx : qidx;
    T* best = partialDist.ptr<T>();
    int* bestIdx = partialIdx.ptr<int>();
    // Ordered stripes and strict comparisons retain the first index on ties.
    for (int stripe = 1; stripe < stripes; ++stripe)
    {
        const T* pd = partialDist.ptr<T>(stripe);
        const int* pi = partialIdx.ptr<int>(stripe);
        for (int j = 0; j < partialDist.cols; ++j)
        {
            if (pd[j] < best[j])
            {
                best[j] = pd[j];
                bestIdx[j] = pi[j];
            }
        }
    }

    // Preserve the distances of rejected matches as well as the mutual indices.
    const T* td = tdist.ptr<T>();
    const int* ti = tidx.ptr<int>();
    const int* qi = qidx.ptr<int>();
    for (int t = 0; t < src2.rows; ++t)
    {
        const int q = ti[t];
        if (q >= 0 && td[t] < dist.at<T>(q))
        {
            dist.at<T>(q) = td[t];
            nidx.at<int>(q) = t;
        }
    }
    for (int q = 0; q < src1.rows; ++q)
    {
        if (qi[q] < 0 || ti[qi[q]] != q)
            nidx.at<int>(q) = -1;
    }
}

}

void cv::batchDistance( InputArray _src1, InputArray _src2,
                        OutputArray _dist, int dtype, OutputArray _nidx,
                        int normType, int K, InputArray _mask,
                        int update, bool crosscheck )
{
    CV_INSTRUMENT_REGION();

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
        CV_Assert(!nidx.empty());
    }

    BatchDistFunc func = 0;
    if( type == CV_8U )
    {
        if( normType == NORM_L1 && dtype == CV_32S )
            func = batchDistL1_<uchar, int>;
        else if( normType == NORM_L1 && dtype == CV_32F )
            func = batchDistL1_<uchar, float>;
        else if( normType == NORM_L2SQR && dtype == CV_32S )
            func = batchDistL2Sqr_<uchar, int>;
        else if( normType == NORM_L2SQR && dtype == CV_32F )
            func = batchDistL2Sqr_<uchar, float>;
        else if( normType == NORM_L2 && dtype == CV_32F )
            func = batchDistL2_<uchar, float>;
        else if( normType == NORM_HAMMING && dtype == CV_32S )
            func = batchDistHamming;
        else if( normType == NORM_HAMMING2 && dtype == CV_32S )
            func = batchDistHamming2;
    }
    else if( type == CV_32F && dtype == CV_32F )
    {
        if( normType == NORM_L1 )
            func = batchDistL1_<float, float>;
        else if( normType == NORM_L2SQR )
            func = batchDistL2Sqr_<float, float>;
        else if( normType == NORM_L2 )
            func = batchDistL2_<float, float>;
    }

    if( func == 0 )
        CV_Error_(cv::Error::StsUnsupportedFormat,
                  ("The combination of type=%d, dtype=%d and normType=%d is not supported",
                   type, dtype, normType));

    if (crosscheck)
    {
        if (dtype == CV_32S)
            batchDistCrossCheck<int>(src1, src2, dist, nidx, func);
        else
            batchDistCrossCheck<float>(src1, src2, dist, nidx, func);
        return;
    }

    parallel_for_(Range(0, src1.rows),
                  BatchDistInvoker(src1, src2, dist, nidx, K, mask, update, func));
}
