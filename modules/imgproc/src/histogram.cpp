/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//            Intel License Agreement
//        For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "opencl_kernels.hpp"

namespace cv
{

////////////////// Helper functions //////////////////////

static const size_t OUT_OF_RANGE = (size_t)1 << (sizeof(size_t)*8 - 2);

static void
calcHistLookupTables_8u( const Mat& hist, const SparseMat& shist,
                         int dims, const float** ranges, const double* uniranges,
                         bool uniform, bool issparse, std::vector<size_t>& _tab )
{
    const int low = 0, high = 256;
    int i, j;
    _tab.resize((high-low)*dims);
    size_t* tab = &_tab[0];

    if( uniform )
    {
        for( i = 0; i < dims; i++ )
        {
            double a = uniranges[i*2];
            double b = uniranges[i*2+1];
            int sz = !issparse ? hist.size[i] : shist.size(i);
            size_t step = !issparse ? hist.step[i] : 1;

            for( j = low; j < high; j++ )
            {
                int idx = cvFloor(j*a + b);
                size_t written_idx;
                if( (unsigned)idx < (unsigned)sz )
                    written_idx = idx*step;
                else
                    written_idx = OUT_OF_RANGE;

                tab[i*(high - low) + j - low] = written_idx;
            }
        }
    }
    else
    {
        for( i = 0; i < dims; i++ )
        {
            int limit = std::min(cvCeil(ranges[i][0]), high);
            int idx = -1, sz = !issparse ? hist.size[i] : shist.size(i);
            size_t written_idx = OUT_OF_RANGE;
            size_t step = !issparse ? hist.step[i] : 1;

            for(j = low;;)
            {
                for( ; j < limit; j++ )
                    tab[i*(high - low) + j - low] = written_idx;

                if( (unsigned)(++idx) < (unsigned)sz )
                {
                    limit = std::min(cvCeil(ranges[i][idx+1]), high);
                    written_idx = idx*step;
                }
                else
                {
                    for( ; j < high; j++ )
                        tab[i*(high - low) + j - low] = OUT_OF_RANGE;
                    break;
                }
            }
        }
    }
}


static void histPrepareImages( const Mat* images, int nimages, const int* channels,
                               const Mat& mask, int dims, const int* histSize,
                               const float** ranges, bool uniform,
                               std::vector<uchar*>& ptrs, std::vector<int>& deltas,
                               Size& imsize, std::vector<double>& uniranges )
{
    int i, j, c;
    CV_Assert( channels != 0 || nimages == dims );

    imsize = images[0].size();
    int depth = images[0].depth(), esz1 = (int)images[0].elemSize1();
    bool isContinuous = true;

    ptrs.resize(dims + 1);
    deltas.resize((dims + 1)*2);

    for( i = 0; i < dims; i++ )
    {
        if(!channels)
        {
            j = i;
            c = 0;
            CV_Assert( images[j].channels() == 1 );
        }
        else
        {
            c = channels[i];
            CV_Assert( c >= 0 );
            for( j = 0; j < nimages; c -= images[j].channels(), j++ )
                if( c < images[j].channels() )
                    break;
            CV_Assert( j < nimages );
        }

        CV_Assert( images[j].size() == imsize && images[j].depth() == depth );
        if( !images[j].isContinuous() )
            isContinuous = false;
        ptrs[i] = images[j].data + c*esz1;
        deltas[i*2] = images[j].channels();
        deltas[i*2+1] = (int)(images[j].step/esz1 - imsize.width*deltas[i*2]);
    }

    if( mask.data )
    {
        CV_Assert( mask.size() == imsize && mask.channels() == 1 );
        isContinuous = isContinuous && mask.isContinuous();
        ptrs[dims] = mask.data;
        deltas[dims*2] = 1;
        deltas[dims*2 + 1] = (int)(mask.step/mask.elemSize1());
    }

#ifndef HAVE_TBB
    if( isContinuous )
    {
        imsize.width *= imsize.height;
        imsize.height = 1;
    }
#endif

    if( !ranges )
    {
        CV_Assert( depth == CV_8U );

        uniranges.resize( dims*2 );
        for( i = 0; i < dims; i++ )
        {
            uniranges[i*2] = histSize[i]/256.;
            uniranges[i*2+1] = 0;
        }
    }
    else if( uniform )
    {
        uniranges.resize( dims*2 );
        for( i = 0; i < dims; i++ )
        {
            CV_Assert( ranges[i] && ranges[i][0] < ranges[i][1] );
            double low = ranges[i][0], high = ranges[i][1];
            double t = histSize[i]/(high - low);
            uniranges[i*2] = t;
            uniranges[i*2+1] = -t*low;
        }
    }
    else
    {
        for( i = 0; i < dims; i++ )
        {
            size_t n = histSize[i];
            for(size_t k = 0; k < n; k++ )
                CV_Assert( ranges[i][k] < ranges[i][k+1] );
        }
    }
}


////////////////////////////////// C A L C U L A T E    H I S T O G R A M ////////////////////////////////////
#ifdef HAVE_TBB
enum {one = 1, two, three}; // array elements number

template<typename T>
class calcHist1D_Invoker
{
public:
    calcHist1D_Invoker( const std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                        Mat& hist, const double* _uniranges, int sz, int dims,
                        Size& imageSize )
        : mask_(_ptrs[dims]),
          mstep_(_deltas[dims*2 + 1]),
          imageWidth_(imageSize.width),
          histogramSize_(hist.size()), histogramType_(hist.type()),
          globalHistogram_((tbb::atomic<int>*)hist.data)
    {
        p_[0] = ((T**)&_ptrs[0])[0];
        step_[0] = (&_deltas[0])[1];
        d_[0] = (&_deltas[0])[0];
        a_[0] = (&_uniranges[0])[0];
        b_[0] = (&_uniranges[0])[1];
        size_[0] = sz;
    }

    void operator()( const BlockedRange& range ) const
    {
        T* p0 = p_[0] + range.begin() * (step_[0] + imageWidth_*d_[0]);
        uchar* mask = mask_ + range.begin()*mstep_;

        for( int row = range.begin(); row < range.end(); row++, p0 += step_[0] )
        {
            if( !mask_ )
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0] )
                {
                    int idx = cvFloor(*p0*a_[0] + b_[0]);
                    if( (unsigned)idx < (unsigned)size_[0] )
                    {
                        globalHistogram_[idx].fetch_and_add(1);
                    }
                }
            }
            else
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0] )
                {
                    if( mask[x] )
                    {
                        int idx = cvFloor(*p0*a_[0] + b_[0]);
                        if( (unsigned)idx < (unsigned)size_[0] )
                        {
                            globalHistogram_[idx].fetch_and_add(1);
                        }
                    }
                }
                mask += mstep_;
            }
        }
    }

private:
    calcHist1D_Invoker operator=(const calcHist1D_Invoker&);

    T* p_[one];
    uchar* mask_;
    int step_[one];
    int d_[one];
    int mstep_;
    double a_[one];
    double b_[one];
    int size_[one];
    int imageWidth_;
    Size histogramSize_;
    int histogramType_;
    tbb::atomic<int>* globalHistogram_;
};

template<typename T>
class calcHist2D_Invoker
{
public:
    calcHist2D_Invoker( const std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                        Mat& hist, const double* _uniranges, const int* size,
                        int dims, Size& imageSize, size_t* hstep )
        : mask_(_ptrs[dims]),
          mstep_(_deltas[dims*2 + 1]),
          imageWidth_(imageSize.width),
          histogramSize_(hist.size()), histogramType_(hist.type()),
          globalHistogram_(hist.data)
    {
        p_[0] = ((T**)&_ptrs[0])[0]; p_[1] = ((T**)&_ptrs[0])[1];
        step_[0] = (&_deltas[0])[1]; step_[1] = (&_deltas[0])[3];
        d_[0] = (&_deltas[0])[0];    d_[1] = (&_deltas[0])[2];
        a_[0] = (&_uniranges[0])[0]; a_[1] = (&_uniranges[0])[2];
        b_[0] = (&_uniranges[0])[1]; b_[1] = (&_uniranges[0])[3];
        size_[0] = size[0];          size_[1] = size[1];
        hstep_[0] = hstep[0];
    }

    void operator()(const BlockedRange& range) const
    {
        T* p0 = p_[0] + range.begin()*(step_[0] + imageWidth_*d_[0]);
        T* p1 = p_[1] + range.begin()*(step_[1] + imageWidth_*d_[1]);
        uchar* mask = mask_ + range.begin()*mstep_;

        for( int row = range.begin(); row < range.end(); row++, p0 += step_[0], p1 += step_[1] )
        {
            if( !mask_ )
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1] )
                {
                    int idx0 = cvFloor(*p0*a_[0] + b_[0]);
                    int idx1 = cvFloor(*p1*a_[1] + b_[1]);
                    if( (unsigned)idx0 < (unsigned)size_[0] && (unsigned)idx1 < (unsigned)size_[1] )
                        ( (tbb::atomic<int>*)(globalHistogram_ + hstep_[0]*idx0) )[idx1].fetch_and_add(1);
                }
            }
            else
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1] )
                {
                    if( mask[x] )
                    {
                        int idx0 = cvFloor(*p0*a_[0] + b_[0]);
                        int idx1 = cvFloor(*p1*a_[1] + b_[1]);
                        if( (unsigned)idx0 < (unsigned)size_[0] && (unsigned)idx1 < (unsigned)size_[1] )
                            ((tbb::atomic<int>*)(globalHistogram_ + hstep_[0]*idx0))[idx1].fetch_and_add(1);
                    }
                }
                mask += mstep_;
            }
        }
    }

private:
    calcHist2D_Invoker operator=(const calcHist2D_Invoker&);

    T* p_[two];
    uchar* mask_;
    int step_[two];
    int d_[two];
    int mstep_;
    double a_[two];
    double b_[two];
    int size_[two];
    const int imageWidth_;
    size_t hstep_[one];
    Size histogramSize_;
    int histogramType_;
    uchar* globalHistogram_;
};


template<typename T>
class calcHist3D_Invoker
{
public:
    calcHist3D_Invoker( const std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                        Size imsize, Mat& hist, const double* uniranges, int _dims,
                        size_t* hstep, int* size )
        : mask_(_ptrs[_dims]),
          mstep_(_deltas[_dims*2 + 1]),
          imageWidth_(imsize.width),
          globalHistogram_(hist.data)
    {
        p_[0] = ((T**)&_ptrs[0])[0]; p_[1] = ((T**)&_ptrs[0])[1]; p_[2] = ((T**)&_ptrs[0])[2];
        step_[0] = (&_deltas[0])[1]; step_[1] = (&_deltas[0])[3]; step_[2] = (&_deltas[0])[5];
        d_[0] = (&_deltas[0])[0];    d_[1] = (&_deltas[0])[2];    d_[2] = (&_deltas[0])[4];
        a_[0] = uniranges[0];        a_[1] = uniranges[2];        a_[2] = uniranges[4];
        b_[0] = uniranges[1];        b_[1] = uniranges[3];        b_[2] = uniranges[5];
        size_[0] = size[0];          size_[1] = size[1];          size_[2] = size[2];
        hstep_[0] = hstep[0];        hstep_[1] = hstep[1];
    }

    void operator()( const BlockedRange& range ) const
    {
        T* p0 = p_[0] + range.begin()*(imageWidth_*d_[0] + step_[0]);
        T* p1 = p_[1] + range.begin()*(imageWidth_*d_[1] + step_[1]);
        T* p2 = p_[2] + range.begin()*(imageWidth_*d_[2] + step_[2]);
        uchar* mask = mask_ + range.begin()*mstep_;

        for( int i = range.begin(); i < range.end(); i++, p0 += step_[0], p1 += step_[1], p2 += step_[2] )
        {
            if( !mask_ )
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1], p2 += d_[2] )
                {
                    int idx0 = cvFloor(*p0*a_[0] + b_[0]);
                    int idx1 = cvFloor(*p1*a_[1] + b_[1]);
                    int idx2 = cvFloor(*p2*a_[2] + b_[2]);
                    if( (unsigned)idx0 < (unsigned)size_[0] &&
                            (unsigned)idx1 < (unsigned)size_[1] &&
                            (unsigned)idx2 < (unsigned)size_[2] )
                    {
                        ( (tbb::atomic<int>*)(globalHistogram_ + hstep_[0]*idx0 + hstep_[1]*idx1) )[idx2].fetch_and_add(1);
                    }
                }
            }
            else
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1], p2 += d_[2] )
                {
                    if( mask[x] )
                    {
                        int idx0 = cvFloor(*p0*a_[0] + b_[0]);
                        int idx1 = cvFloor(*p1*a_[1] + b_[1]);
                        int idx2 = cvFloor(*p2*a_[2] + b_[2]);
                        if( (unsigned)idx0 < (unsigned)size_[0] &&
                                (unsigned)idx1 < (unsigned)size_[1] &&
                                (unsigned)idx2 < (unsigned)size_[2] )
                        {
                            ( (tbb::atomic<int>*)(globalHistogram_ + hstep_[0]*idx0 + hstep_[1]*idx1) )[idx2].fetch_and_add(1);
                        }
                    }
                }
                mask += mstep_;
            }
        }
    }

    static bool isFit( const Mat& histogram, const Size imageSize )
    {
        return ( imageSize.width * imageSize.height >= 320*240
                 && histogram.total() >= 8*8*8 );
    }

private:
    calcHist3D_Invoker operator=(const calcHist3D_Invoker&);

    T* p_[three];
    uchar* mask_;
    int step_[three];
    int d_[three];
    const int mstep_;
    double a_[three];
    double b_[three];
    int size_[three];
    int imageWidth_;
    size_t hstep_[two];
    uchar* globalHistogram_;
};

class CalcHist1D_8uInvoker
{
public:
    CalcHist1D_8uInvoker( const std::vector<uchar*>& ptrs, const std::vector<int>& deltas,
                          Size imsize, Mat& hist, int dims, const std::vector<size_t>& tab,
                          tbb::mutex* lock )
        : mask_(ptrs[dims]),
          mstep_(deltas[dims*2 + 1]),
          imageWidth_(imsize.width),
          imageSize_(imsize),
          histSize_(hist.size()), histType_(hist.type()),
          tab_((size_t*)&tab[0]),
          histogramWriteLock_(lock),
          globalHistogram_(hist.data)
    {
        p_[0] = (&ptrs[0])[0];
        step_[0] = (&deltas[0])[1];
        d_[0] = (&deltas[0])[0];
    }

    void operator()( const BlockedRange& range ) const
    {
        int localHistogram[256] = { 0, };
        uchar* mask = mask_;
        uchar* p0 = p_[0];
        int x;
        tbb::mutex::scoped_lock lock;

        if( !mask_ )
        {
            int n = (imageWidth_ - 4) / 4 + 1;
            int tail = imageWidth_ - n*4;

            int xN = 4*n;
            p0 += (xN*d_[0] + tail*d_[0] + step_[0]) * range.begin();
        }
        else
        {
            p0 += (imageWidth_*d_[0] + step_[0]) * range.begin();
            mask += mstep_*range.begin();
        }

        for( int i = range.begin(); i < range.end(); i++, p0 += step_[0] )
        {
            if( !mask_ )
            {
                if( d_[0] == 1 )
                {
                    for( x = 0; x <= imageWidth_ - 4; x += 4 )
                    {
                        int t0 = p0[x], t1 = p0[x+1];
                        localHistogram[t0]++; localHistogram[t1]++;
                        t0 = p0[x+2]; t1 = p0[x+3];
                        localHistogram[t0]++; localHistogram[t1]++;
                    }
                    p0 += x;
                }
                else
                {
                    for( x = 0; x <= imageWidth_ - 4; x += 4 )
                    {
                        int t0 = p0[0], t1 = p0[d_[0]];
                        localHistogram[t0]++; localHistogram[t1]++;
                        p0 += d_[0]*2;
                        t0 = p0[0]; t1 = p0[d_[0]];
                        localHistogram[t0]++; localHistogram[t1]++;
                        p0 += d_[0]*2;
                    }
                }

                for( ; x < imageWidth_; x++, p0 += d_[0] )
                {
                    localHistogram[*p0]++;
                }
            }
            else
            {
                for( x = 0; x < imageWidth_; x++, p0 += d_[0] )
                {
                    if( mask[x] )
                    {
                        localHistogram[*p0]++;
                    }
                }
                mask += mstep_;
            }
        }

        lock.acquire(*histogramWriteLock_);
        for(int i = 0; i < 256; i++ )
        {
            size_t hidx = tab_[i];
            if( hidx < OUT_OF_RANGE )
            {
                *(int*)((globalHistogram_ + hidx)) += localHistogram[i];
            }
        }
        lock.release();
    }

    static bool isFit( const Mat& histogram, const Size imageSize )
    {
        return ( histogram.total() >= 8
                && imageSize.width * imageSize.height >= 160*120 );
    }

private:
    uchar* p_[one];
    uchar* mask_;
    int mstep_;
    int step_[one];
    int d_[one];
    int imageWidth_;
    Size imageSize_;
    Size histSize_;
    int histType_;
    size_t* tab_;
    tbb::mutex* histogramWriteLock_;
    uchar* globalHistogram_;
};

class CalcHist2D_8uInvoker
{
public:
    CalcHist2D_8uInvoker( const std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                          Size imsize, Mat& hist, int dims, const std::vector<size_t>& _tab,
                          tbb::mutex* lock )
        : mask_(_ptrs[dims]),
          mstep_(_deltas[dims*2 + 1]),
          imageWidth_(imsize.width),
          histSize_(hist.size()), histType_(hist.type()),
          tab_((size_t*)&_tab[0]),
          histogramWriteLock_(lock),
          globalHistogram_(hist.data)
    {
        p_[0] = (uchar*)(&_ptrs[0])[0]; p_[1] = (uchar*)(&_ptrs[0])[1];
        step_[0] = (&_deltas[0])[1];    step_[1] = (&_deltas[0])[3];
        d_[0] = (&_deltas[0])[0];       d_[1] = (&_deltas[0])[2];
    }

    void operator()( const BlockedRange& range ) const
    {
        uchar* p0 = p_[0] + range.begin()*(step_[0] + imageWidth_*d_[0]);
        uchar* p1 = p_[1] + range.begin()*(step_[1] + imageWidth_*d_[1]);
        uchar* mask = mask_ + range.begin()*mstep_;

        Mat localHist = Mat::zeros(histSize_, histType_);
        uchar* localHistData = localHist.data;
        tbb::mutex::scoped_lock lock;

        for(int i = range.begin(); i < range.end(); i++, p0 += step_[0], p1 += step_[1])
        {
            if( !mask_ )
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1] )
                {
                    size_t idx = tab_[*p0] + tab_[*p1 + 256];
                    if( idx < OUT_OF_RANGE )
                    {
                        ++*(int*)(localHistData + idx);
                    }
                }
            }
            else
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1] )
                {
                    size_t idx;
                    if( mask[x] && (idx = tab_[*p0] + tab_[*p1 + 256]) < OUT_OF_RANGE )
                    {
                        ++*(int*)(localHistData + idx);
                    }
                }
                mask += mstep_;
            }
        }

        lock.acquire(*histogramWriteLock_);
        for(int i = 0; i < histSize_.width*histSize_.height; i++)
        {
            ((int*)globalHistogram_)[i] += ((int*)localHistData)[i];
        }
        lock.release();
    }

    static bool isFit( const Mat& histogram, const Size imageSize )
    {
        return ( (histogram.total() > 4*4 &&  histogram.total() <= 116*116
                  && imageSize.width * imageSize.height >= 320*240)
                 || (histogram.total() > 116*116 && imageSize.width * imageSize.height >= 1280*720) );
    }

private:
    uchar* p_[two];
    uchar* mask_;
    int step_[two];
    int d_[two];
    int mstep_;
    int imageWidth_;
    Size histSize_;
    int histType_;
    size_t* tab_;
    tbb::mutex* histogramWriteLock_;
    uchar* globalHistogram_;
};

class CalcHist3D_8uInvoker
{
public:
    CalcHist3D_8uInvoker( const std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                          Size imsize, Mat& hist, int dims, const std::vector<size_t>& tab )
        : mask_(_ptrs[dims]),
          mstep_(_deltas[dims*2 + 1]),
          histogramSize_(hist.size.p), histogramType_(hist.type()),
          imageWidth_(imsize.width),
          tab_((size_t*)&tab[0]),
          globalHistogram_(hist.data)
    {
        p_[0] = (uchar*)(&_ptrs[0])[0]; p_[1] = (uchar*)(&_ptrs[0])[1]; p_[2] = (uchar*)(&_ptrs[0])[2];
        step_[0] = (&_deltas[0])[1];    step_[1] = (&_deltas[0])[3];    step_[2] = (&_deltas[0])[5];
        d_[0] = (&_deltas[0])[0];       d_[1] = (&_deltas[0])[2];       d_[2] = (&_deltas[0])[4];
    }

    void operator()( const BlockedRange& range ) const
    {
        uchar* p0 = p_[0] + range.begin()*(step_[0] + imageWidth_*d_[0]);
        uchar* p1 = p_[1] + range.begin()*(step_[1] + imageWidth_*d_[1]);
        uchar* p2 = p_[2] + range.begin()*(step_[2] + imageWidth_*d_[2]);
        uchar* mask = mask_ + range.begin()*mstep_;

        for(int i = range.begin(); i < range.end(); i++, p0 += step_[0], p1 += step_[1], p2 += step_[2] )
        {
            if( !mask_ )
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1], p2 += d_[2] )
                {
                    size_t idx = tab_[*p0] + tab_[*p1 + 256] + tab_[*p2 + 512];
                    if( idx < OUT_OF_RANGE )
                    {
                        ( *(tbb::atomic<int>*)(globalHistogram_ + idx) ).fetch_and_add(1);
                    }
                }
            }
            else
            {
                for( int x = 0; x < imageWidth_; x++, p0 += d_[0], p1 += d_[1], p2 += d_[2] )
                {
                    size_t idx;
                    if( mask[x] && (idx = tab_[*p0] + tab_[*p1 + 256] + tab_[*p2 + 512]) < OUT_OF_RANGE )
                    {
                        (*(tbb::atomic<int>*)(globalHistogram_ + idx)).fetch_and_add(1);
                    }
                }
                mask += mstep_;
            }
        }
    }

    static bool isFit( const Mat& histogram, const Size imageSize )
    {
        return ( histogram.total() >= 128*128*128
                 && imageSize.width * imageSize.width >= 320*240 );
    }

private:
    uchar* p_[three];
    uchar* mask_;
    int mstep_;
    int step_[three];
    int d_[three];
    int* histogramSize_;
    int histogramType_;
    int imageWidth_;
    size_t* tab_;
    uchar* globalHistogram_;
};

static void
callCalcHist2D_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                   Size imsize, Mat& hist, int dims,  std::vector<size_t>& _tab )
{
    int grainSize = imsize.height / tbb::task_scheduler_init::default_num_threads();
    tbb::mutex histogramWriteLock;

    CalcHist2D_8uInvoker body(_ptrs, _deltas, imsize, hist, dims, _tab, &histogramWriteLock);
    parallel_for(BlockedRange(0, imsize.height, grainSize), body);
}

static void
callCalcHist3D_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                   Size imsize, Mat& hist, int dims,  std::vector<size_t>& _tab )
{
    CalcHist3D_8uInvoker body(_ptrs, _deltas, imsize, hist, dims, _tab);
    parallel_for(BlockedRange(0, imsize.height), body);
}
#endif

template<typename T> static void
calcHist_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
           Size imsize, Mat& hist, int dims, const float** _ranges,
           const double* _uniranges, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int i, x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    int size[CV_MAX_DIM];
    size_t hstep[CV_MAX_DIM];

    for( i = 0; i < dims; i++ )
    {
        size[i] = hist.size[i];
        hstep[i] = hist.step[i];
    }

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];

        if( dims == 1 )
        {
#ifdef HAVE_TBB
            calcHist1D_Invoker<T> body(_ptrs, _deltas, hist, _uniranges, size[0], dims, imsize);
            parallel_for(BlockedRange(0, imsize.height), body);
#else
            double a = uniranges[0], b = uniranges[1];
            int sz = size[0], d0 = deltas[0], step0 = deltas[1];
            const T* p0 = (const T*)ptrs[0];

            for( ; imsize.height--; p0 += step0, mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++, p0 += d0 )
                    {
                        int idx = cvFloor(*p0*a + b);
                        if( (unsigned)idx < (unsigned)sz )
                            ((int*)H)[idx]++;
                    }
                else
                    for( x = 0; x < imsize.width; x++, p0 += d0 )
                        if( mask[x] )
                        {
                            int idx = cvFloor(*p0*a + b);
                            if( (unsigned)idx < (unsigned)sz )
                                ((int*)H)[idx]++;
                        }
            }
#endif //HAVE_TBB
            return;
        }
        else if( dims == 2 )
        {
#ifdef HAVE_TBB
            calcHist2D_Invoker<T> body(_ptrs, _deltas, hist, _uniranges, size, dims, imsize, hstep);
            parallel_for(BlockedRange(0, imsize.height), body);
#else
            double a0 = uniranges[0], b0 = uniranges[1], a1 = uniranges[2], b1 = uniranges[3];
            int sz0 = size[0], sz1 = size[1];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3];
            size_t hstep0 = hstep[0];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];

            for( ; imsize.height--; p0 += step0, p1 += step1, mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                    {
                        int idx0 = cvFloor(*p0*a0 + b0);
                        int idx1 = cvFloor(*p1*a1 + b1);
                        if( (unsigned)idx0 < (unsigned)sz0 && (unsigned)idx1 < (unsigned)sz1 )
                            ((int*)(H + hstep0*idx0))[idx1]++;
                    }
                else
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                        if( mask[x] )
                        {
                            int idx0 = cvFloor(*p0*a0 + b0);
                            int idx1 = cvFloor(*p1*a1 + b1);
                            if( (unsigned)idx0 < (unsigned)sz0 && (unsigned)idx1 < (unsigned)sz1 )
                                ((int*)(H + hstep0*idx0))[idx1]++;
                        }
            }
#endif //HAVE_TBB
            return;
        }
        else if( dims == 3 )
        {
#ifdef HAVE_TBB
            if( calcHist3D_Invoker<T>::isFit(hist, imsize) )
            {
                calcHist3D_Invoker<T> body(_ptrs, _deltas, imsize, hist, uniranges, dims, hstep, size);
                parallel_for(BlockedRange(0, imsize.height), body);
                return;
            }
#endif
            double a0 = uniranges[0], b0 = uniranges[1],
                   a1 = uniranges[2], b1 = uniranges[3],
                   a2 = uniranges[4], b2 = uniranges[5];
            int sz0 = size[0], sz1 = size[1], sz2 = size[2];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3],
                d2 = deltas[4], step2 = deltas[5];
            size_t hstep0 = hstep[0], hstep1 = hstep[1];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];
            const T* p2 = (const T*)ptrs[2];

            for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                    {
                        int idx0 = cvFloor(*p0*a0 + b0);
                        int idx1 = cvFloor(*p1*a1 + b1);
                        int idx2 = cvFloor(*p2*a2 + b2);
                        if( (unsigned)idx0 < (unsigned)sz0 &&
                            (unsigned)idx1 < (unsigned)sz1 &&
                            (unsigned)idx2 < (unsigned)sz2 )
                            ((int*)(H + hstep0*idx0 + hstep1*idx1))[idx2]++;
                    }
                else
                    for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                        if( mask[x] )
                        {
                            int idx0 = cvFloor(*p0*a0 + b0);
                            int idx1 = cvFloor(*p1*a1 + b1);
                            int idx2 = cvFloor(*p2*a2 + b2);
                            if( (unsigned)idx0 < (unsigned)sz0 &&
                               (unsigned)idx1 < (unsigned)sz1 &&
                               (unsigned)idx2 < (unsigned)sz2 )
                                ((int*)(H + hstep0*idx0 + hstep1*idx1))[idx2]++;
                        }
            }
        }
        else
        {
            for( ; imsize.height--; mask += mstep )
            {
                if( !mask )
                    for( x = 0; x < imsize.width; x++ )
                    {
                        uchar* Hptr = H;
                        for( i = 0; i < dims; i++ )
                        {
                            int idx = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                            if( (unsigned)idx >= (unsigned)size[i] )
                                break;
                            ptrs[i] += deltas[i*2];
                            Hptr += idx*hstep[i];
                        }

                        if( i == dims )
                            ++*((int*)Hptr);
                        else
                            for( ; i < dims; i++ )
                                ptrs[i] += deltas[i*2];
                    }
                else
                    for( x = 0; x < imsize.width; x++ )
                    {
                        uchar* Hptr = H;
                        i = 0;
                        if( mask[x] )
                            for( ; i < dims; i++ )
                            {
                                int idx = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                                if( (unsigned)idx >= (unsigned)size[i] )
                                    break;
                                ptrs[i] += deltas[i*2];
                                Hptr += idx*hstep[i];
                            }

                        if( i == dims )
                            ++*((int*)Hptr);
                        else
                            for( ; i < dims; i++ )
                                ptrs[i] += deltas[i*2];
                    }
                for( i = 0; i < dims; i++ )
                    ptrs[i] += deltas[i*2 + 1];
            }
        }
    }
    else
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; mask += mstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                uchar* Hptr = H;
                i = 0;

                if( !mask || mask[x] )
                    for( ; i < dims; i++ )
                    {
                        float v = (float)*ptrs[i];
                        const float* R = ranges[i];
                        int idx = -1, sz = size[i];

                        while( v >= R[idx+1] && ++idx < sz )
                            ; // nop

                        if( (unsigned)idx >= (unsigned)sz )
                            break;

                        ptrs[i] += deltas[i*2];
                        Hptr += idx*hstep[i];
                    }

                if( i == dims )
                    ++*((int*)Hptr);
                else
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}


static void
calcHist_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
             Size imsize, Mat& hist, int dims, const float** _ranges,
             const double* _uniranges, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    std::vector<size_t> _tab;

    calcHistLookupTables_8u( hist, SparseMat(), dims, _ranges, _uniranges, uniform, false, _tab );
    const size_t* tab = &_tab[0];

    if( dims == 1 )
    {
#ifdef HAVE_TBB
        if( CalcHist1D_8uInvoker::isFit(hist, imsize) )
        {
            int treadsNumber = tbb::task_scheduler_init::default_num_threads();
            int grainSize = imsize.height/treadsNumber;
            tbb::mutex histogramWriteLock;

            CalcHist1D_8uInvoker body(_ptrs, _deltas, imsize, hist, dims, _tab, &histogramWriteLock);
            parallel_for(BlockedRange(0, imsize.height, grainSize), body);
            return;
        }
#endif
        int d0 = deltas[0], step0 = deltas[1];
        int matH[256] = { 0, };
        const uchar* p0 = (const uchar*)ptrs[0];

        for( ; imsize.height--; p0 += step0, mask += mstep )
        {
            if( !mask )
            {
                if( d0 == 1 )
                {
                    for( x = 0; x <= imsize.width - 4; x += 4 )
                    {
                        int t0 = p0[x], t1 = p0[x+1];
                        matH[t0]++; matH[t1]++;
                        t0 = p0[x+2]; t1 = p0[x+3];
                        matH[t0]++; matH[t1]++;
                    }
                    p0 += x;
                }
                else
                    for( x = 0; x <= imsize.width - 4; x += 4 )
                    {
                        int t0 = p0[0], t1 = p0[d0];
                        matH[t0]++; matH[t1]++;
                        p0 += d0*2;
                        t0 = p0[0]; t1 = p0[d0];
                        matH[t0]++; matH[t1]++;
                        p0 += d0*2;
                    }

                for( ; x < imsize.width; x++, p0 += d0 )
                    matH[*p0]++;
            }
            else
                for( x = 0; x < imsize.width; x++, p0 += d0 )
                    if( mask[x] )
                        matH[*p0]++;
        }

        for(int i = 0; i < 256; i++ )
        {
            size_t hidx = tab[i];
            if( hidx < OUT_OF_RANGE )
                *(int*)(H + hidx) += matH[i];
        }
    }
    else if( dims == 2 )
    {
#ifdef HAVE_TBB
        if( CalcHist2D_8uInvoker::isFit(hist, imsize) )
        {
            callCalcHist2D_8u(_ptrs, _deltas, imsize, hist, dims, _tab);
            return;
        }
#endif
        int d0 = deltas[0], step0 = deltas[1],
            d1 = deltas[2], step1 = deltas[3];
        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];

        for( ; imsize.height--; p0 += step0, p1 += step1, mask += mstep )
        {
            if( !mask )
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                {
                    size_t idx = tab[*p0] + tab[*p1 + 256];
                    if( idx < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
            else
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                {
                    size_t idx;
                    if( mask[x] && (idx = tab[*p0] + tab[*p1 + 256]) < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
        }
    }
    else if( dims == 3 )
    {
#ifdef HAVE_TBB
        if( CalcHist3D_8uInvoker::isFit(hist, imsize) )
        {
            callCalcHist3D_8u(_ptrs, _deltas, imsize, hist, dims, _tab);
            return;
        }
#endif
        int d0 = deltas[0], step0 = deltas[1],
            d1 = deltas[2], step1 = deltas[3],
            d2 = deltas[4], step2 = deltas[5];

        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];
        const uchar* p2 = (const uchar*)ptrs[2];

        for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, mask += mstep )
        {
            if( !mask )
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                {
                    size_t idx = tab[*p0] + tab[*p1 + 256] + tab[*p2 + 512];
                    if( idx < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
            else
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                {
                    size_t idx;
                    if( mask[x] && (idx = tab[*p0] + tab[*p1 + 256] + tab[*p2 + 512]) < OUT_OF_RANGE )
                        ++*(int*)(H + idx);
                }
        }
    }
    else
    {
        for( ; imsize.height--; mask += mstep )
        {
            if( !mask )
                for( x = 0; x < imsize.width; x++ )
                {
                    uchar* Hptr = H;
                    int i = 0;
                    for( ; i < dims; i++ )
                    {
                        size_t idx = tab[*ptrs[i] + i*256];
                        if( idx >= OUT_OF_RANGE )
                            break;
                        Hptr += idx;
                        ptrs[i] += deltas[i*2];
                    }

                    if( i == dims )
                        ++*((int*)Hptr);
                    else
                        for( ; i < dims; i++ )
                            ptrs[i] += deltas[i*2];
                }
            else
                for( x = 0; x < imsize.width; x++ )
                {
                    uchar* Hptr = H;
                    int i = 0;
                    if( mask[x] )
                        for( ; i < dims; i++ )
                        {
                            size_t idx = tab[*ptrs[i] + i*256];
                            if( idx >= OUT_OF_RANGE )
                                break;
                            Hptr += idx;
                            ptrs[i] += deltas[i*2];
                        }

                    if( i == dims )
                        ++*((int*)Hptr);
                    else
                        for( ; i < dims; i++ )
                            ptrs[i] += deltas[i*2];
                }
            for(int i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}

#if defined HAVE_IPP && !defined HAVE_IPP_ICV_ONLY

class IPPCalcHistInvoker :
    public ParallelLoopBody
{
public:
    IPPCalcHistInvoker(const Mat & _src, Mat & _hist, AutoBuffer<Ipp32s> & _levels, Ipp32s _histSize, Ipp32s _low, Ipp32s _high, bool * _ok) :
        ParallelLoopBody(), src(&_src), hist(&_hist), levels(&_levels), histSize(_histSize), low(_low), high(_high), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator() (const Range & range) const
    {
        Mat phist(hist->size(), hist->type(), Scalar::all(0));

        IppStatus status = ippiHistogramEven_8u_C1R(
            src->data + src->step * range.start, (int)src->step, ippiSize(src->cols, range.end - range.start),
            (Ipp32s *)phist.data, (Ipp32s *)*levels, histSize, low, high);

        if (status < 0)
        {
            *ok = false;
            return;
        }

        for (int i = 0; i < histSize; ++i)
            CV_XADD((int *)(hist->data + i * hist->step), *(int *)(phist.data + i * phist.step));
    }

private:
    const Mat * src;
    Mat * hist;
    AutoBuffer<Ipp32s> * levels;
    Ipp32s histSize, low, high;
    bool * ok;

    const IPPCalcHistInvoker & operator = (const IPPCalcHistInvoker & );
};

#endif

}

void cv::calcHist( const Mat* images, int nimages, const int* channels,
                   InputArray _mask, OutputArray _hist, int dims, const int* histSize,
                   const float** ranges, bool uniform, bool accumulate )
{
    Mat mask = _mask.getMat();

    CV_Assert(dims > 0 && histSize);

    uchar* histdata = _hist.getMat().data;
    _hist.create(dims, histSize, CV_32F);
    Mat hist = _hist.getMat(), ihist = hist;
    ihist.flags = (ihist.flags & ~CV_MAT_TYPE_MASK)|CV_32S;

#if defined HAVE_IPP && !defined HAVE_IPP_ICV_ONLY
    if (nimages == 1 && images[0].type() == CV_8UC1 && dims == 1 && channels &&
            channels[0] == 0 && mask.empty() && images[0].dims <= 2 &&
            !accumulate && uniform)
    {
        ihist.setTo(Scalar::all(0));
        AutoBuffer<Ipp32s> levels(histSize[0] + 1);

        bool ok = true;
        const Mat & src = images[0];
        int nstripes = std::min<int>(8, static_cast<int>(src.total() / (1 << 16)));
#ifdef HAVE_CONCURRENCY
        nstripes = 1;
#endif
        IPPCalcHistInvoker invoker(src, ihist, levels, histSize[0] + 1, (Ipp32s)ranges[0][0], (Ipp32s)ranges[0][1], &ok);
        Range range(0, src.rows);
        parallel_for_(range, invoker, nstripes);

        if (ok)
        {
            ihist.convertTo(hist, CV_32F);
            return;
        }
        setIppErrorStatus();
    }
#endif

    if( !accumulate || histdata != hist.data )
        hist = Scalar(0.);
    else
        hist.convertTo(ihist, CV_32S);

    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;

    CV_Assert( !mask.data || mask.type() == CV_8UC1 );
    histPrepareImages( images, nimages, channels, mask, dims, hist.size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;

    int depth = images[0].depth();

    if( depth == CV_8U )
        calcHist_8u(ptrs, deltas, imsize, ihist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_16U )
        calcHist_<ushort>(ptrs, deltas, imsize, ihist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_32F )
        calcHist_<float>(ptrs, deltas, imsize, ihist, dims, ranges, _uniranges, uniform );
    else
        CV_Error(CV_StsUnsupportedFormat, "");

    ihist.convertTo(hist, CV_32F);
}


namespace cv
{

template<typename T> static void
calcSparseHist_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                 Size imsize, SparseMat& hist, int dims, const float** _ranges,
                 const double* _uniranges, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    const int* size = hist.hdr->size;
    int idx[CV_MAX_DIM];

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];

        for( ; imsize.height--; mask += mstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                i = 0;
                if( !mask || mask[x] )
                    for( ; i < dims; i++ )
                    {
                        idx[i] = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                        if( (unsigned)idx[i] >= (unsigned)size[i] )
                            break;
                        ptrs[i] += deltas[i*2];
                    }

                if( i == dims )
                    ++*(int*)hist.ptr(idx, true);
                else
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
            }
            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; mask += mstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                i = 0;

                if( !mask || mask[x] )
                    for( ; i < dims; i++ )
                    {
                        float v = (float)*ptrs[i];
                        const float* R = ranges[i];
                        int j = -1, sz = size[i];

                        while( v >= R[j+1] && ++j < sz )
                            ; // nop

                        if( (unsigned)j >= (unsigned)sz )
                            break;
                        ptrs[i] += deltas[i*2];
                        idx[i] = j;
                    }

                if( i == dims )
                    ++*(int*)hist.ptr(idx, true);
                else
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}


static void
calcSparseHist_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                   Size imsize, SparseMat& hist, int dims, const float** _ranges,
                   const double* _uniranges, bool uniform )
{
    uchar** ptrs = (uchar**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int x;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    int idx[CV_MAX_DIM];
    std::vector<size_t> _tab;

    calcHistLookupTables_8u( Mat(), hist, dims, _ranges, _uniranges, uniform, true, _tab );
    const size_t* tab = &_tab[0];

    for( ; imsize.height--; mask += mstep )
    {
        for( x = 0; x < imsize.width; x++ )
        {
            int i = 0;
            if( !mask || mask[x] )
                for( ; i < dims; i++ )
                {
                    size_t hidx = tab[*ptrs[i] + i*256];
                    if( hidx >= OUT_OF_RANGE )
                        break;
                    ptrs[i] += deltas[i*2];
                    idx[i] = (int)hidx;
                }

            if( i == dims )
                ++*(int*)hist.ptr(idx,true);
            else
                for( ; i < dims; i++ )
                    ptrs[i] += deltas[i*2];
        }
        for(int i = 0; i < dims; i++ )
            ptrs[i] += deltas[i*2 + 1];
    }
}


static void calcHist( const Mat* images, int nimages, const int* channels,
                      const Mat& mask, SparseMat& hist, int dims, const int* histSize,
                      const float** ranges, bool uniform, bool accumulate, bool keepInt )
{
    size_t i, N;

    if( !accumulate )
        hist.create(dims, histSize, CV_32F);
    else
    {
        SparseMatIterator it = hist.begin();
        for( i = 0, N = hist.nzcount(); i < N; i++, ++it )
        {
            Cv32suf* val = (Cv32suf*)it.ptr;
            val->i = cvRound(val->f);
        }
    }

    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;

    CV_Assert( !mask.data || mask.type() == CV_8UC1 );
    histPrepareImages( images, nimages, channels, mask, dims, hist.hdr->size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;

    int depth = images[0].depth();
    if( depth == CV_8U )
        calcSparseHist_8u(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_16U )
        calcSparseHist_<ushort>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, uniform );
    else if( depth == CV_32F )
        calcSparseHist_<float>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, uniform );
    else
        CV_Error(CV_StsUnsupportedFormat, "");

    if( !keepInt )
    {
        SparseMatIterator it = hist.begin();
        for( i = 0, N = hist.nzcount(); i < N; i++, ++it )
        {
            Cv32suf* val = (Cv32suf*)it.ptr;
            val->f = (float)val->i;
        }
    }
}

#ifdef HAVE_OPENCL

enum
{
    BINS = 256
};

static bool ocl_calcHist1(InputArray _src, OutputArray _hist, int ddepth = CV_32S)
{
    int compunits = ocl::Device::getDefault().maxComputeUnits();
    size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();
    Size size = _src.size();
    bool use16 = size.width % 16 == 0 && _src.offset() % 16 == 0 && _src.step() % 16 == 0;

    ocl::Kernel k1("calculate_histogram", ocl::imgproc::histogram_oclsrc,
                   format("-D BINS=%d -D HISTS_COUNT=%d -D WGS=%d -D cn=%d",
                          BINS, compunits, wgs, use16 ? 16 : 1));
    if (k1.empty())
        return false;

    _hist.create(BINS, 1, ddepth);
    UMat src = _src.getUMat(), ghist(1, BINS * compunits, CV_32SC1),
            hist = ddepth == CV_32S ? _hist.getUMat() : UMat(BINS, 1, CV_32SC1);

    k1.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::PtrWriteOnly(ghist), (int)src.total());

    size_t globalsize = compunits * wgs;
    if (!k1.run(1, &globalsize, &wgs, false))
        return false;

    ocl::Kernel k2("merge_histogram", ocl::imgproc::histogram_oclsrc,
                   format("-D BINS=%d -D HISTS_COUNT=%d -D WGS=%d", BINS, compunits, (int)wgs));
    if (k2.empty())
        return false;

    k2.args(ocl::KernelArg::PtrReadOnly(ghist), ocl::KernelArg::PtrWriteOnly(hist));
    if (!k2.run(1, &wgs, &wgs, false))
        return false;

    if (hist.depth() != ddepth)
        hist.convertTo(_hist, ddepth);
    else
        _hist.getUMatRef() = hist;

    return true;
}

static bool ocl_calcHist(InputArrayOfArrays images, OutputArray hist)
{
    std::vector<UMat> v;
    images.getUMatVector(v);

    return ocl_calcHist1(v[0], hist, CV_32F);
}

#endif

}

void cv::calcHist( const Mat* images, int nimages, const int* channels,
               InputArray _mask, SparseMat& hist, int dims, const int* histSize,
               const float** ranges, bool uniform, bool accumulate )
{
    Mat mask = _mask.getMat();
    calcHist( images, nimages, channels, mask, hist, dims, histSize,
              ranges, uniform, accumulate, false );
}


void cv::calcHist( InputArrayOfArrays images, const std::vector<int>& channels,
                   InputArray mask, OutputArray hist,
                   const std::vector<int>& histSize,
                   const std::vector<float>& ranges,
                   bool accumulate )
{
    CV_OCL_RUN(images.total() == 1 && channels.size() == 1 && images.channels(0) == 1 &&
               channels[0] == 0 && images.isUMatVector() && mask.empty() && !accumulate &&
               histSize.size() == 1 && histSize[0] == BINS && ranges.size() == 2 &&
               ranges[0] == 0 && ranges[1] == BINS,
               ocl_calcHist(images, hist))

    int i, dims = (int)histSize.size(), rsz = (int)ranges.size(), csz = (int)channels.size();
    int nimages = (int)images.total();

    CV_Assert(nimages > 0 && dims > 0);
    CV_Assert(rsz == dims*2 || (rsz == 0 && images.depth(0) == CV_8U));
    CV_Assert(csz == 0 || csz == dims);
    float* _ranges[CV_MAX_DIM];
    if( rsz > 0 )
    {
        for( i = 0; i < rsz/2; i++ )
            _ranges[i] = (float*)&ranges[i*2];
    }

    AutoBuffer<Mat> buf(nimages);
    for( i = 0; i < nimages; i++ )
        buf[i] = images.getMat(i);

    calcHist(&buf[0], nimages, csz ? &channels[0] : 0,
            mask, hist, dims, &histSize[0], rsz ? (const float**)_ranges : 0,
            true, accumulate);
}


/////////////////////////////////////// B A C K   P R O J E C T ////////////////////////////////////

namespace cv
{

template<typename T, typename BT> static void
calcBackProj_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
               Size imsize, const Mat& hist, int dims, const float** _ranges,
               const double* _uniranges, float scale, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int i, x;
    BT* bproj = (BT*)_ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    int size[CV_MAX_DIM];
    size_t hstep[CV_MAX_DIM];

    for( i = 0; i < dims; i++ )
    {
        size[i] = hist.size[i];
        hstep[i] = hist.step[i];
    }

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];

        if( dims == 1 )
        {
            double a = uniranges[0], b = uniranges[1];
            int sz = size[0], d0 = deltas[0], step0 = deltas[1];
            const T* p0 = (const T*)ptrs[0];

            for( ; imsize.height--; p0 += step0, bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++, p0 += d0 )
                {
                    int idx = cvFloor(*p0*a + b);
                    bproj[x] = (unsigned)idx < (unsigned)sz ? saturate_cast<BT>(((float*)H)[idx]*scale) : 0;
                }
            }
        }
        else if( dims == 2 )
        {
            double a0 = uniranges[0], b0 = uniranges[1],
                   a1 = uniranges[2], b1 = uniranges[3];
            int sz0 = size[0], sz1 = size[1];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3];
            size_t hstep0 = hstep[0];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];

            for( ; imsize.height--; p0 += step0, p1 += step1, bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
                {
                    int idx0 = cvFloor(*p0*a0 + b0);
                    int idx1 = cvFloor(*p1*a1 + b1);
                    bproj[x] = (unsigned)idx0 < (unsigned)sz0 &&
                               (unsigned)idx1 < (unsigned)sz1 ?
                        saturate_cast<BT>(((float*)(H + hstep0*idx0))[idx1]*scale) : 0;
                }
            }
        }
        else if( dims == 3 )
        {
            double a0 = uniranges[0], b0 = uniranges[1],
                   a1 = uniranges[2], b1 = uniranges[3],
                   a2 = uniranges[4], b2 = uniranges[5];
            int sz0 = size[0], sz1 = size[1], sz2 = size[2];
            int d0 = deltas[0], step0 = deltas[1],
                d1 = deltas[2], step1 = deltas[3],
                d2 = deltas[4], step2 = deltas[5];
            size_t hstep0 = hstep[0], hstep1 = hstep[1];
            const T* p0 = (const T*)ptrs[0];
            const T* p1 = (const T*)ptrs[1];
            const T* p2 = (const T*)ptrs[2];

            for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
                {
                    int idx0 = cvFloor(*p0*a0 + b0);
                    int idx1 = cvFloor(*p1*a1 + b1);
                    int idx2 = cvFloor(*p2*a2 + b2);
                    bproj[x] = (unsigned)idx0 < (unsigned)sz0 &&
                               (unsigned)idx1 < (unsigned)sz1 &&
                               (unsigned)idx2 < (unsigned)sz2 ?
                        saturate_cast<BT>(((float*)(H + hstep0*idx0 + hstep1*idx1))[idx2]*scale) : 0;
                }
            }
        }
        else
        {
            for( ; imsize.height--; bproj += bpstep )
            {
                for( x = 0; x < imsize.width; x++ )
                {
                    uchar* Hptr = H;
                    for( i = 0; i < dims; i++ )
                    {
                        int idx = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                        if( (unsigned)idx >= (unsigned)size[i] || (_ranges && *ptrs[i] >= _ranges[i][1]))
                            break;
                        ptrs[i] += deltas[i*2];
                        Hptr += idx*hstep[i];
                    }

                    if( i == dims )
                        bproj[x] = saturate_cast<BT>(*(float*)Hptr*scale);
                    else
                    {
                        bproj[x] = 0;
                        for( ; i < dims; i++ )
                            ptrs[i] += deltas[i*2];
                    }
                }
                for( i = 0; i < dims; i++ )
                    ptrs[i] += deltas[i*2 + 1];
            }
        }
    }
    else
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                uchar* Hptr = H;
                for( i = 0; i < dims; i++ )
                {
                    float v = (float)*ptrs[i];
                    const float* R = ranges[i];
                    int idx = -1, sz = size[i];

                    while( v >= R[idx+1] && ++idx < sz )
                        ; // nop

                    if( (unsigned)idx >= (unsigned)sz )
                        break;

                    ptrs[i] += deltas[i*2];
                    Hptr += idx*hstep[i];
                }

                if( i == dims )
                    bproj[x] = saturate_cast<BT>(*(float*)Hptr*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}


static void
calcBackProj_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                 Size imsize, const Mat& hist, int dims, const float** _ranges,
                 const double* _uniranges, float scale, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int i, x;
    uchar* bproj = _ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    std::vector<size_t> _tab;

    calcHistLookupTables_8u( hist, SparseMat(), dims, _ranges, _uniranges, uniform, false, _tab );
    const size_t* tab = &_tab[0];

    if( dims == 1 )
    {
        int d0 = deltas[0], step0 = deltas[1];
        uchar matH[256] = {0};
        const uchar* p0 = (const uchar*)ptrs[0];

        for( i = 0; i < 256; i++ )
        {
            size_t hidx = tab[i];
            if( hidx < OUT_OF_RANGE )
                matH[i] = saturate_cast<uchar>(*(float*)(H + hidx)*scale);
        }

        for( ; imsize.height--; p0 += step0, bproj += bpstep )
        {
            if( d0 == 1 )
            {
                for( x = 0; x <= imsize.width - 4; x += 4 )
                {
                    uchar t0 = matH[p0[x]], t1 = matH[p0[x+1]];
                    bproj[x] = t0; bproj[x+1] = t1;
                    t0 = matH[p0[x+2]]; t1 = matH[p0[x+3]];
                    bproj[x+2] = t0; bproj[x+3] = t1;
                }
                p0 += x;
            }
            else
                for( x = 0; x <= imsize.width - 4; x += 4 )
                {
                    uchar t0 = matH[p0[0]], t1 = matH[p0[d0]];
                    bproj[x] = t0; bproj[x+1] = t1;
                    p0 += d0*2;
                    t0 = matH[p0[0]]; t1 = matH[p0[d0]];
                    bproj[x+2] = t0; bproj[x+3] = t1;
                    p0 += d0*2;
                }

            for( ; x < imsize.width; x++, p0 += d0 )
                bproj[x] = matH[*p0];
        }
    }
    else if( dims == 2 )
    {
        int d0 = deltas[0], step0 = deltas[1],
            d1 = deltas[2], step1 = deltas[3];
        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];

        for( ; imsize.height--; p0 += step0, p1 += step1, bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1 )
            {
                size_t idx = tab[*p0] + tab[*p1 + 256];
                bproj[x] = idx < OUT_OF_RANGE ? saturate_cast<uchar>(*(float*)(H + idx)*scale) : 0;
            }
        }
    }
    else if( dims == 3 )
    {
        int d0 = deltas[0], step0 = deltas[1],
        d1 = deltas[2], step1 = deltas[3],
        d2 = deltas[4], step2 = deltas[5];
        const uchar* p0 = (const uchar*)ptrs[0];
        const uchar* p1 = (const uchar*)ptrs[1];
        const uchar* p2 = (const uchar*)ptrs[2];

        for( ; imsize.height--; p0 += step0, p1 += step1, p2 += step2, bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++, p0 += d0, p1 += d1, p2 += d2 )
            {
                size_t idx = tab[*p0] + tab[*p1 + 256] + tab[*p2 + 512];
                bproj[x] = idx < OUT_OF_RANGE ? saturate_cast<uchar>(*(float*)(H + idx)*scale) : 0;
            }
        }
    }
    else
    {
        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                uchar* Hptr = H;
                for( i = 0; i < dims; i++ )
                {
                    size_t idx = tab[*ptrs[i] + i*256];
                    if( idx >= OUT_OF_RANGE )
                        break;
                    ptrs[i] += deltas[i*2];
                    Hptr += idx;
                }

                if( i == dims )
                    bproj[x] = saturate_cast<uchar>(*(float*)Hptr*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }
            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}

}

void cv::calcBackProject( const Mat* images, int nimages, const int* channels,
                          InputArray _hist, OutputArray _backProject,
                          const float** ranges, double scale, bool uniform )
{
    Mat hist = _hist.getMat();
    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;
    int dims = hist.dims == 2 && hist.size[1] == 1 ? 1 : hist.dims;

    CV_Assert( dims > 0 && hist.data );
    _backProject.create( images[0].size(), images[0].depth() );
    Mat backProject = _backProject.getMat();
    histPrepareImages( images, nimages, channels, backProject, dims, hist.size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;

    int depth = images[0].depth();
    if( depth == CV_8U )
        calcBackProj_8u(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, (float)scale, uniform);
    else if( depth == CV_16U )
        calcBackProj_<ushort, ushort>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, (float)scale, uniform );
    else if( depth == CV_32F )
        calcBackProj_<float, float>(ptrs, deltas, imsize, hist, dims, ranges, _uniranges, (float)scale, uniform );
    else
        CV_Error(CV_StsUnsupportedFormat, "");
}


namespace cv
{

template<typename T, typename BT> static void
calcSparseBackProj_( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                     Size imsize, const SparseMat& hist, int dims, const float** _ranges,
                     const double* _uniranges, float scale, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x;
    BT* bproj = (BT*)_ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    const int* size = hist.hdr->size;
    int idx[CV_MAX_DIM];
    const SparseMat_<float>& hist_ = (const SparseMat_<float>&)hist;

    if( uniform )
    {
        const double* uniranges = &_uniranges[0];
        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                for( i = 0; i < dims; i++ )
                {
                    idx[i] = cvFloor(*ptrs[i]*uniranges[i*2] + uniranges[i*2+1]);
                    if( (unsigned)idx[i] >= (unsigned)size[i] )
                        break;
                    ptrs[i] += deltas[i*2];
                }

                if( i == dims )
                    bproj[x] = saturate_cast<BT>(hist_(idx)*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }
            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
    else
    {
        // non-uniform histogram
        const float* ranges[CV_MAX_DIM];
        for( i = 0; i < dims; i++ )
            ranges[i] = &_ranges[i][0];

        for( ; imsize.height--; bproj += bpstep )
        {
            for( x = 0; x < imsize.width; x++ )
            {
                for( i = 0; i < dims; i++ )
                {
                    float v = (float)*ptrs[i];
                    const float* R = ranges[i];
                    int j = -1, sz = size[i];

                    while( v >= R[j+1] && ++j < sz )
                        ; // nop

                    if( (unsigned)j >= (unsigned)sz )
                        break;
                    idx[i] = j;
                    ptrs[i] += deltas[i*2];
                }

                if( i == dims )
                    bproj[x] = saturate_cast<BT>(hist_(idx)*scale);
                else
                {
                    bproj[x] = 0;
                    for( ; i < dims; i++ )
                        ptrs[i] += deltas[i*2];
                }
            }

            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}


static void
calcSparseBackProj_8u( std::vector<uchar*>& _ptrs, const std::vector<int>& _deltas,
                       Size imsize, const SparseMat& hist, int dims, const float** _ranges,
                       const double* _uniranges, float scale, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x;
    uchar* bproj = _ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    std::vector<size_t> _tab;
    int idx[CV_MAX_DIM];

    calcHistLookupTables_8u( Mat(), hist, dims, _ranges, _uniranges, uniform, true, _tab );
    const size_t* tab = &_tab[0];

    for( ; imsize.height--; bproj += bpstep )
    {
        for( x = 0; x < imsize.width; x++ )
        {
            for( i = 0; i < dims; i++ )
            {
                size_t hidx = tab[*ptrs[i] + i*256];
                if( hidx >= OUT_OF_RANGE )
                    break;
                idx[i] = (int)hidx;
                ptrs[i] += deltas[i*2];
            }

            if( i == dims )
                bproj[x] = saturate_cast<uchar>(hist.value<float>(idx)*scale);
            else
            {
                bproj[x] = 0;
                for( ; i < dims; i++ )
                    ptrs[i] += deltas[i*2];
            }
        }
        for( i = 0; i < dims; i++ )
            ptrs[i] += deltas[i*2 + 1];
    }
}

}

void cv::calcBackProject( const Mat* images, int nimages, const int* channels,
                          const SparseMat& hist, OutputArray _backProject,
                          const float** ranges, double scale, bool uniform )
{
    std::vector<uchar*> ptrs;
    std::vector<int> deltas;
    std::vector<double> uniranges;
    Size imsize;
    int dims = hist.dims();

    CV_Assert( dims > 0 );
    _backProject.create( images[0].size(), images[0].depth() );
    Mat backProject = _backProject.getMat();
    histPrepareImages( images, nimages, channels, backProject,
                       dims, hist.hdr->size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;

    int depth = images[0].depth();
    if( depth == CV_8U )
        calcSparseBackProj_8u(ptrs, deltas, imsize, hist, dims, ranges,
                              _uniranges, (float)scale, uniform);
    else if( depth == CV_16U )
        calcSparseBackProj_<ushort, ushort>(ptrs, deltas, imsize, hist, dims, ranges,
                                          _uniranges, (float)scale, uniform );
    else if( depth == CV_32F )
        calcSparseBackProj_<float, float>(ptrs, deltas, imsize, hist, dims, ranges,
                                          _uniranges, (float)scale, uniform );
    else
        CV_Error(CV_StsUnsupportedFormat, "");
}

#ifdef HAVE_OPENCL

namespace cv {

static void getUMatIndex(const std::vector<UMat> & um, int cn, int & idx, int & cnidx)
{
    int totalChannels = 0;
    for (size_t i = 0, size = um.size(); i < size; ++i)
    {
        int ccn = um[i].channels();
        totalChannels += ccn;

        if (totalChannels == cn)
        {
            idx = (int)(i + 1);
            cnidx = 0;
            return;
        }
        else if (totalChannels > cn)
        {
            idx = (int)i;
            cnidx = i == 0 ? cn : (cn - totalChannels + ccn);
            return;
        }
    }

    idx = cnidx = -1;
}

static bool ocl_calcBackProject( InputArrayOfArrays _images, std::vector<int> channels,
                                 InputArray _hist, OutputArray _dst,
                                 const std::vector<float>& ranges,
                                 float scale, size_t histdims )
{
    std::vector<UMat> images;
    _images.getUMatVector(images);

    size_t nimages = images.size(), totalcn = images[0].channels();

    CV_Assert(nimages > 0);
    Size size = images[0].size();
    int depth = images[0].depth();

    //kernels are valid for this type only
    if (depth != CV_8U)
        return false;

    for (size_t i = 1; i < nimages; ++i)
    {
        const UMat & m = images[i];
        totalcn += m.channels();
        CV_Assert(size == m.size() && depth == m.depth());
    }

    std::sort(channels.begin(), channels.end());
    for (size_t i = 0; i < histdims; ++i)
        CV_Assert(channels[i] < (int)totalcn);

    if (histdims == 1)
    {
        int idx, cnidx;
        getUMatIndex(images, channels[0], idx, cnidx);
        CV_Assert(idx >= 0);
        UMat im = images[idx];

        String opts = format("-D histdims=1 -D scn=%d", im.channels());
        ocl::Kernel lutk("calcLUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (lutk.empty())
            return false;

        size_t lsize = 256;
        UMat lut(1, (int)lsize, CV_32SC1), hist = _hist.getUMat(), uranges(ranges, true);

        lutk.args(ocl::KernelArg::ReadOnlyNoSize(hist), hist.rows,
                  ocl::KernelArg::PtrWriteOnly(lut), scale, ocl::KernelArg::PtrReadOnly(uranges));
        if (!lutk.run(1, &lsize, NULL, false))
            return false;

        ocl::Kernel mapk("LUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (mapk.empty())
            return false;

        _dst.create(size, depth);
        UMat dst = _dst.getUMat();

        im.offset += cnidx;
        mapk.args(ocl::KernelArg::ReadOnlyNoSize(im), ocl::KernelArg::PtrReadOnly(lut),
                  ocl::KernelArg::WriteOnly(dst));

        size_t globalsize[2] = { size.width, size.height };
        return mapk.run(2, globalsize, NULL, false);
    }
    else if (histdims == 2)
    {
        int idx0, idx1, cnidx0, cnidx1;
        getUMatIndex(images, channels[0], idx0, cnidx0);
        getUMatIndex(images, channels[1], idx1, cnidx1);
        CV_Assert(idx0 >= 0 && idx1 >= 0);
        UMat im0 = images[idx0], im1 = images[idx1];

        // Lut for the first dimension
        String opts = format("-D histdims=2 -D scn1=%d -D scn2=%d", im0.channels(), im1.channels());
        ocl::Kernel lutk1("calcLUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (lutk1.empty())
            return false;

        size_t lsize = 256;
        UMat lut(1, (int)lsize<<1, CV_32SC1), uranges(ranges, true), hist = _hist.getUMat();

        lutk1.args(hist.rows, ocl::KernelArg::PtrWriteOnly(lut), (int)0, ocl::KernelArg::PtrReadOnly(uranges), (int)0);
        if (!lutk1.run(1, &lsize, NULL, false))
            return false;

        // lut for the second dimension
        ocl::Kernel lutk2("calcLUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (lutk2.empty())
            return false;

        lut.offset += lsize * sizeof(int);
        lutk2.args(hist.cols, ocl::KernelArg::PtrWriteOnly(lut), (int)256, ocl::KernelArg::PtrReadOnly(uranges), (int)2);
        if (!lutk2.run(1, &lsize, NULL, false))
            return false;

        // perform lut
        ocl::Kernel mapk("LUT", ocl::imgproc::calc_back_project_oclsrc, opts);
        if (mapk.empty())
            return false;

        _dst.create(size, depth);
        UMat dst = _dst.getUMat();

        im0.offset += cnidx0;
        im1.offset += cnidx1;
        mapk.args(ocl::KernelArg::ReadOnlyNoSize(im0), ocl::KernelArg::ReadOnlyNoSize(im1),
               ocl::KernelArg::ReadOnlyNoSize(hist), ocl::KernelArg::PtrReadOnly(lut), scale, ocl::KernelArg::WriteOnly(dst));

        size_t globalsize[2] = { size.width, size.height };
        return mapk.run(2, globalsize, NULL, false);
    }
    return false;
}

}

#endif

void cv::calcBackProject( InputArrayOfArrays images, const std::vector<int>& channels,
                          InputArray hist, OutputArray dst,
                          const std::vector<float>& ranges,
                          double scale )
{
    Size histSize = hist.size();
#ifdef HAVE_OPENCL
    bool _1D = histSize.height == 1 || histSize.width == 1;
    size_t histdims = _1D ? 1 : hist.dims();
#endif

    CV_OCL_RUN(dst.isUMat() && hist.type() == CV_32FC1 &&
               histdims <= 2 && ranges.size() == histdims * 2 && histdims == channels.size(),
               ocl_calcBackProject(images, channels, hist, dst, ranges, (float)scale, histdims))

    Mat H0 = hist.getMat(), H;
    int hcn = H0.channels();

    if( hcn > 1 )
    {
        CV_Assert( H0.isContinuous() );
        int hsz[CV_CN_MAX+1];
        memcpy(hsz, &H0.size[0], H0.dims*sizeof(hsz[0]));
        hsz[H0.dims] = hcn;
        H = Mat(H0.dims+1, hsz, H0.depth(), H0.data);
    }
    else
        H = H0;

    bool _1d = H.rows == 1 || H.cols == 1;
    int i, dims = H.dims, rsz = (int)ranges.size(), csz = (int)channels.size();
    int nimages = (int)images.total();

    CV_Assert(nimages > 0);
    CV_Assert(rsz == dims*2 || (rsz == 2 && _1d) || (rsz == 0 && images.depth(0) == CV_8U));
    CV_Assert(csz == 0 || csz == dims || (csz == 1 && _1d));

    float* _ranges[CV_MAX_DIM];
    if( rsz > 0 )
    {
        for( i = 0; i < rsz/2; i++ )
            _ranges[i] = (float*)&ranges[i*2];
    }

    AutoBuffer<Mat> buf(nimages);
    for( i = 0; i < nimages; i++ )
        buf[i] = images.getMat(i);

    calcBackProject(&buf[0], nimages, csz ? &channels[0] : 0,
        hist, dst, rsz ? (const float**)_ranges : 0, scale, true);
}


////////////////// C O M P A R E   H I S T O G R A M S ////////////////////////

double cv::compareHist( InputArray _H1, InputArray _H2, int method )
{
    Mat H1 = _H1.getMat(), H2 = _H2.getMat();
    const Mat* arrays[] = {&H1, &H2, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    double result = 0;
    int j, len = (int)it.size;

    CV_Assert( H1.type() == H2.type() && H1.depth() == CV_32F );

    double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;

    CV_Assert( it.planes[0].isContinuous() && it.planes[1].isContinuous() );

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        const float* h1 = (const float*)it.planes[0].data;
        const float* h2 = (const float*)it.planes[1].data;
        len = it.planes[0].rows*it.planes[0].cols*H1.channels();

        if( (method == CV_COMP_CHISQR) || (method == CV_COMP_CHISQR_ALT))
        {
            for( j = 0; j < len; j++ )
            {
                double a = h1[j] - h2[j];
                double b = (method == CV_COMP_CHISQR) ? h1[j] : h1[j] + h2[j];
                if( fabs(b) > DBL_EPSILON )
                    result += a*a/b;
            }
        }
        else if( method == CV_COMP_CORREL )
        {
            for( j = 0; j < len; j++ )
            {
                double a = h1[j];
                double b = h2[j];

                s12 += a*b;
                s1 += a;
                s11 += a*a;
                s2 += b;
                s22 += b*b;
            }
        }
        else if( method == CV_COMP_INTERSECT )
        {
            for( j = 0; j < len; j++ )
                result += std::min(h1[j], h2[j]);
        }
        else if( method == CV_COMP_BHATTACHARYYA )
        {
            for( j = 0; j < len; j++ )
            {
                double a = h1[j];
                double b = h2[j];
                result += std::sqrt(a*b);
                s1 += a;
                s2 += b;
            }
        }
        else
            CV_Error( CV_StsBadArg, "Unknown comparison method" );
    }

    if( method == CV_COMP_CHISQR_ALT )
        result *= 2;
    else if( method == CV_COMP_CORREL )
    {
        size_t total = H1.total();
        double scale = 1./total;
        double num = s12 - s1*s2*scale;
        double denom2 = (s11 - s1*s1*scale)*(s22 - s2*s2*scale);
        result = std::abs(denom2) > DBL_EPSILON ? num/std::sqrt(denom2) : 1.;
    }
    else if( method == CV_COMP_BHATTACHARYYA )
    {
        s1 *= s2;
        s1 = fabs(s1) > FLT_EPSILON ? 1./std::sqrt(s1) : 1.;
        result = std::sqrt(std::max(1. - result*s1, 0.));
    }

    return result;
}


double cv::compareHist( const SparseMat& H1, const SparseMat& H2, int method )
{
    double result = 0;
    int i, dims = H1.dims();

    CV_Assert( dims > 0 && dims == H2.dims() && H1.type() == H2.type() && H1.type() == CV_32F );
    for( i = 0; i < dims; i++ )
        CV_Assert( H1.size(i) == H2.size(i) );

    const SparseMat *PH1 = &H1, *PH2 = &H2;
    if( PH1->nzcount() > PH2->nzcount() && method != CV_COMP_CHISQR && method != CV_COMP_CHISQR_ALT)
        std::swap(PH1, PH2);

    SparseMatConstIterator it = PH1->begin();
    int N1 = (int)PH1->nzcount(), N2 = (int)PH2->nzcount();

    if( (method == CV_COMP_CHISQR) || (method == CV_COMP_CHISQR_ALT) )
    {
        for( i = 0; i < N1; i++, ++it )
        {
            float v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            float v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            double a = v1 - v2;
            double b = (method == CV_COMP_CHISQR) ? v1 : v1 + v2;
            if( fabs(b) > DBL_EPSILON )
                result += a*a/b;
        }
    }
    else if( method == CV_COMP_CORREL )
    {
        double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;

        for( i = 0; i < N1; i++, ++it )
        {
            double v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            s12 += v1*PH2->value<float>(node->idx, (size_t*)&node->hashval);
            s1 += v1;
            s11 += v1*v1;
        }

        it = PH2->begin();
        for( i = 0; i < N2; i++, ++it )
        {
            double v2 = it.value<float>();
            s2 += v2;
            s22 += v2*v2;
        }

        size_t total = 1;
        for( i = 0; i < H1.dims(); i++ )
            total *= H1.size(i);
        double scale = 1./total;
        double num = s12 - s1*s2*scale;
        double denom2 = (s11 - s1*s1*scale)*(s22 - s2*s2*scale);
        result = std::abs(denom2) > DBL_EPSILON ? num/std::sqrt(denom2) : 1.;
    }
    else if( method == CV_COMP_INTERSECT )
    {
        for( i = 0; i < N1; i++, ++it )
        {
            float v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            float v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            if( v2 )
                result += std::min(v1, v2);
        }
    }
    else if( method == CV_COMP_BHATTACHARYYA )
    {
        double s1 = 0, s2 = 0;

        for( i = 0; i < N1; i++, ++it )
        {
            double v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            double v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            result += std::sqrt(v1*v2);
            s1 += v1;
        }

        it = PH2->begin();
        for( i = 0; i < N2; i++, ++it )
            s2 += it.value<float>();

        s1 *= s2;
        s1 = fabs(s1) > FLT_EPSILON ? 1./std::sqrt(s1) : 1.;
        result = std::sqrt(std::max(1. - result*s1, 0.));
    }
    else
        CV_Error( CV_StsBadArg, "Unknown comparison method" );

    if( method == CV_COMP_CHISQR_ALT )
        result *= 2;

    return result;
}


const int CV_HIST_DEFAULT_TYPE = CV_32F;

/* Creates new histogram */
CvHistogram *
cvCreateHist( int dims, int *sizes, CvHistType type, float** ranges, int uniform )
{
    CvHistogram *hist = 0;

    if( (unsigned)dims > CV_MAX_DIM )
        CV_Error( CV_BadOrder, "Number of dimensions is out of range" );

    if( !sizes )
        CV_Error( CV_HeaderIsNull, "Null <sizes> pointer" );

    hist = (CvHistogram *)cvAlloc( sizeof( CvHistogram ));
    hist->type = CV_HIST_MAGIC_VAL + ((int)type & 1);
    if (uniform) hist->type|= CV_HIST_UNIFORM_FLAG;
    hist->thresh2 = 0;
    hist->bins = 0;
    if( type == CV_HIST_ARRAY )
    {
        hist->bins = cvInitMatNDHeader( &hist->mat, dims, sizes,
                                        CV_HIST_DEFAULT_TYPE );
        cvCreateData( hist->bins );
    }
    else if( type == CV_HIST_SPARSE )
        hist->bins = cvCreateSparseMat( dims, sizes, CV_HIST_DEFAULT_TYPE );
    else
        CV_Error( CV_StsBadArg, "Invalid histogram type" );

    if( ranges )
        cvSetHistBinRanges( hist, ranges, uniform );

    return hist;
}


/* Creates histogram wrapping header for given array */
CV_IMPL CvHistogram*
cvMakeHistHeaderForArray( int dims, int *sizes, CvHistogram *hist,
                          float *data, float **ranges, int uniform )
{
    if( !hist )
        CV_Error( CV_StsNullPtr, "Null histogram header pointer" );

    if( !data )
        CV_Error( CV_StsNullPtr, "Null data pointer" );

    hist->thresh2 = 0;
    hist->type = CV_HIST_MAGIC_VAL;
    hist->bins = cvInitMatNDHeader( &hist->mat, dims, sizes, CV_HIST_DEFAULT_TYPE, data );

    if( ranges )
    {
        if( !uniform )
            CV_Error( CV_StsBadArg, "Only uniform bin ranges can be used here "
                                    "(to avoid memory allocation)" );
        cvSetHistBinRanges( hist, ranges, uniform );
    }

    return hist;
}


CV_IMPL void
cvReleaseHist( CvHistogram **hist )
{
    if( !hist )
        CV_Error( CV_StsNullPtr, "" );

    if( *hist )
    {
        CvHistogram* temp = *hist;

        if( !CV_IS_HIST(temp))
            CV_Error( CV_StsBadArg, "Invalid histogram header" );
        *hist = 0;

        if( CV_IS_SPARSE_HIST( temp ))
            cvReleaseSparseMat( (CvSparseMat**)&temp->bins );
        else
        {
            cvReleaseData( temp->bins );
            temp->bins = 0;
        }

        if( temp->thresh2 )
            cvFree( &temp->thresh2 );
        cvFree( &temp );
    }
}

CV_IMPL void
cvClearHist( CvHistogram *hist )
{
    if( !CV_IS_HIST(hist) )
        CV_Error( CV_StsBadArg, "Invalid histogram header" );
    cvZero( hist->bins );
}


// Clears histogram bins that are below than threshold
CV_IMPL void
cvThreshHist( CvHistogram* hist, double thresh )
{
    if( !CV_IS_HIST(hist) )
        CV_Error( CV_StsBadArg, "Invalid histogram header" );

    if( !CV_IS_SPARSE_MAT(hist->bins) )
    {
        CvMat mat;
        cvGetMat( hist->bins, &mat, 0, 1 );
        cvThreshold( &mat, &mat, thresh, 0, CV_THRESH_TOZERO );
    }
    else
    {
        CvSparseMat* mat = (CvSparseMat*)hist->bins;
        CvSparseMatIterator iterator;
        CvSparseNode *node;

        for( node = cvInitSparseMatIterator( mat, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ))
        {
            float* val = (float*)CV_NODE_VAL( mat, node );
            if( *val <= thresh )
                *val = 0;
        }
    }
}


// Normalizes histogram (make sum of the histogram bins == factor)
CV_IMPL void
cvNormalizeHist( CvHistogram* hist, double factor )
{
    double sum = 0;

    if( !CV_IS_HIST(hist) )
        CV_Error( CV_StsBadArg, "Invalid histogram header" );

    if( !CV_IS_SPARSE_HIST(hist) )
    {
        CvMat mat;
        cvGetMat( hist->bins, &mat, 0, 1 );
        sum = cvSum( &mat ).val[0];
        if( fabs(sum) < DBL_EPSILON )
            sum = 1;
        cvScale( &mat, &mat, factor/sum, 0 );
    }
    else
    {
        CvSparseMat* mat = (CvSparseMat*)hist->bins;
        CvSparseMatIterator iterator;
        CvSparseNode *node;
        float scale;

        for( node = cvInitSparseMatIterator( mat, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ))
        {
            sum += *(float*)CV_NODE_VAL(mat,node);
        }

        if( fabs(sum) < DBL_EPSILON )
            sum = 1;
        scale = (float)(factor/sum);

        for( node = cvInitSparseMatIterator( mat, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ))
        {
            *(float*)CV_NODE_VAL(mat,node) *= scale;
        }
    }
}


// Retrieves histogram global min, max and their positions
CV_IMPL void
cvGetMinMaxHistValue( const CvHistogram* hist,
                      float *value_min, float* value_max,
                      int* idx_min, int* idx_max )
{
    double minVal, maxVal;
    int dims, size[CV_MAX_DIM];

    if( !CV_IS_HIST(hist) )
        CV_Error( CV_StsBadArg, "Invalid histogram header" );

    dims = cvGetDims( hist->bins, size );

    if( !CV_IS_SPARSE_HIST(hist) )
    {
        CvMat mat;
        CvPoint minPt, maxPt;

        cvGetMat( hist->bins, &mat, 0, 1 );
        cvMinMaxLoc( &mat, &minVal, &maxVal, &minPt, &maxPt );

        if( dims == 1 )
        {
            if( idx_min )
                *idx_min = minPt.y + minPt.x;
            if( idx_max )
                *idx_max = maxPt.y + maxPt.x;
        }
        else if( dims == 2 )
        {
            if( idx_min )
                idx_min[0] = minPt.y, idx_min[1] = minPt.x;
            if( idx_max )
                idx_max[0] = maxPt.y, idx_max[1] = maxPt.x;
        }
        else if( idx_min || idx_max )
        {
            int imin = minPt.y*mat.cols + minPt.x;
            int imax = maxPt.y*mat.cols + maxPt.x;

            for(int i = dims - 1; i >= 0; i-- )
            {
                if( idx_min )
                {
                    int t = imin / size[i];
                    idx_min[i] = imin - t*size[i];
                    imin = t;
                }

                if( idx_max )
                {
                    int t = imax / size[i];
                    idx_max[i] = imax - t*size[i];
                    imax = t;
                }
            }
        }
    }
    else
    {
        CvSparseMat* mat = (CvSparseMat*)hist->bins;
        CvSparseMatIterator iterator;
        CvSparseNode *node;
        int minv = INT_MAX;
        int maxv = INT_MIN;
        CvSparseNode* minNode = 0;
        CvSparseNode* maxNode = 0;
        const int *_idx_min = 0, *_idx_max = 0;
        Cv32suf m;

        for( node = cvInitSparseMatIterator( mat, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ))
        {
            int value = *(int*)CV_NODE_VAL(mat,node);
            value = CV_TOGGLE_FLT(value);
            if( value < minv )
            {
                minv = value;
                minNode = node;
            }

            if( value > maxv )
            {
                maxv = value;
                maxNode = node;
            }
        }

        if( minNode )
        {
            _idx_min = CV_NODE_IDX(mat,minNode);
            _idx_max = CV_NODE_IDX(mat,maxNode);
            m.i = CV_TOGGLE_FLT(minv); minVal = m.f;
            m.i = CV_TOGGLE_FLT(maxv); maxVal = m.f;
        }
        else
        {
            minVal = maxVal = 0;
        }

        for(int i = 0; i < dims; i++ )
        {
            if( idx_min )
                idx_min[i] = _idx_min ? _idx_min[i] : -1;
            if( idx_max )
                idx_max[i] = _idx_max ? _idx_max[i] : -1;
        }
    }

    if( value_min )
        *value_min = (float)minVal;

    if( value_max )
        *value_max = (float)maxVal;
}


// Compares two histograms using one of a few methods
CV_IMPL double
cvCompareHist( const CvHistogram* hist1,
               const CvHistogram* hist2,
               int method )
{
    int i;
    int size1[CV_MAX_DIM], size2[CV_MAX_DIM], total = 1;

    if( !CV_IS_HIST(hist1) || !CV_IS_HIST(hist2) )
        CV_Error( CV_StsBadArg, "Invalid histogram header[s]" );

    if( CV_IS_SPARSE_MAT(hist1->bins) != CV_IS_SPARSE_MAT(hist2->bins))
        CV_Error(CV_StsUnmatchedFormats, "One of histograms is sparse and other is not");

    if( !CV_IS_SPARSE_MAT(hist1->bins) )
    {
        cv::Mat H1 = cv::cvarrToMat(hist1->bins);
        cv::Mat H2 = cv::cvarrToMat(hist2->bins);
        return cv::compareHist(H1, H2, method);
    }

    int dims1 = cvGetDims( hist1->bins, size1 );
    int dims2 = cvGetDims( hist2->bins, size2 );

    if( dims1 != dims2 )
        CV_Error( CV_StsUnmatchedSizes,
                 "The histograms have different numbers of dimensions" );

    for( i = 0; i < dims1; i++ )
    {
        if( size1[i] != size2[i] )
            CV_Error( CV_StsUnmatchedSizes, "The histograms have different sizes" );
        total *= size1[i];
    }

    double result = 0;
    CvSparseMat* mat1 = (CvSparseMat*)(hist1->bins);
    CvSparseMat* mat2 = (CvSparseMat*)(hist2->bins);
    CvSparseMatIterator iterator;
    CvSparseNode *node1, *node2;

    if( mat1->heap->active_count > mat2->heap->active_count && method != CV_COMP_CHISQR && method != CV_COMP_CHISQR_ALT)
    {
        CvSparseMat* t;
        CV_SWAP( mat1, mat2, t );
    }

    if( (method == CV_COMP_CHISQR) || (method == CV_COMP_CHISQR_ALT) )
    {
        for( node1 = cvInitSparseMatIterator( mat1, &iterator );
             node1 != 0; node1 = cvGetNextSparseNode( &iterator ))
        {
            double v1 = *(float*)CV_NODE_VAL(mat1,node1);
            uchar* node2_data = cvPtrND( mat2, CV_NODE_IDX(mat1,node1), 0, 0, &node1->hashval );
            double v2 = node2_data ? *(float*)node2_data : 0.f;
            double a = v1 - v2;
            double b = (method == CV_COMP_CHISQR) ? v1 : v1 + v2;
            if( fabs(b) > DBL_EPSILON )
                result += a*a/b;
        }
    }
    else if( method == CV_COMP_CORREL )
    {
        double s1 = 0, s11 = 0;
        double s2 = 0, s22 = 0;
        double s12 = 0;
        double num, denom2, scale = 1./total;

        for( node1 = cvInitSparseMatIterator( mat1, &iterator );
             node1 != 0; node1 = cvGetNextSparseNode( &iterator ))
        {
            double v1 = *(float*)CV_NODE_VAL(mat1,node1);
            uchar* node2_data = cvPtrND( mat2, CV_NODE_IDX(mat1,node1),
                                        0, 0, &node1->hashval );
            if( node2_data )
            {
                double v2 = *(float*)node2_data;
                s12 += v1*v2;
            }
            s1 += v1;
            s11 += v1*v1;
        }

        for( node2 = cvInitSparseMatIterator( mat2, &iterator );
             node2 != 0; node2 = cvGetNextSparseNode( &iterator ))
        {
            double v2 = *(float*)CV_NODE_VAL(mat2,node2);
            s2 += v2;
            s22 += v2*v2;
        }

        num = s12 - s1*s2*scale;
        denom2 = (s11 - s1*s1*scale)*(s22 - s2*s2*scale);
        result = fabs(denom2) > DBL_EPSILON ? num/sqrt(denom2) : 1;
    }
    else if( method == CV_COMP_INTERSECT )
    {
        for( node1 = cvInitSparseMatIterator( mat1, &iterator );
             node1 != 0; node1 = cvGetNextSparseNode( &iterator ))
        {
            float v1 = *(float*)CV_NODE_VAL(mat1,node1);
            uchar* node2_data = cvPtrND( mat2, CV_NODE_IDX(mat1,node1),
                                         0, 0, &node1->hashval );
            if( node2_data )
            {
                float v2 = *(float*)node2_data;
                if( v1 <= v2 )
                    result += v1;
                else
                    result += v2;
            }
        }
    }
    else if( method == CV_COMP_BHATTACHARYYA )
    {
        double s1 = 0, s2 = 0;

        for( node1 = cvInitSparseMatIterator( mat1, &iterator );
             node1 != 0; node1 = cvGetNextSparseNode( &iterator ))
        {
            double v1 = *(float*)CV_NODE_VAL(mat1,node1);
            uchar* node2_data = cvPtrND( mat2, CV_NODE_IDX(mat1,node1),
                                         0, 0, &node1->hashval );
            s1 += v1;
            if( node2_data )
            {
                double v2 = *(float*)node2_data;
                result += sqrt(v1 * v2);
            }
        }

        for( node1 = cvInitSparseMatIterator( mat2, &iterator );
             node1 != 0; node1 = cvGetNextSparseNode( &iterator ))
        {
            double v2 = *(float*)CV_NODE_VAL(mat2,node1);
            s2 += v2;
        }

        s1 *= s2;
        s1 = fabs(s1) > FLT_EPSILON ? 1./sqrt(s1) : 1.;
        result = 1. - result*s1;
        result = sqrt(MAX(result,0.));
    }
    else
        CV_Error( CV_StsBadArg, "Unknown comparison method" );

    if( method == CV_COMP_CHISQR_ALT )
        result *= 2;

    return result;
}

// copies one histogram to another
CV_IMPL void
cvCopyHist( const CvHistogram* src, CvHistogram** _dst )
{
    if( !_dst )
        CV_Error( CV_StsNullPtr, "Destination double pointer is NULL" );

    CvHistogram* dst = *_dst;

    if( !CV_IS_HIST(src) || (dst && !CV_IS_HIST(dst)) )
        CV_Error( CV_StsBadArg, "Invalid histogram header[s]" );

    bool eq = false;
    int size1[CV_MAX_DIM];
    bool is_sparse = CV_IS_SPARSE_MAT(src->bins);
    int dims1 = cvGetDims( src->bins, size1 );

    if( dst && (is_sparse == CV_IS_SPARSE_MAT(dst->bins)))
    {
        int size2[CV_MAX_DIM];
        int dims2 = cvGetDims( dst->bins, size2 );

        if( dims1 == dims2 )
        {
            int i;

            for( i = 0; i < dims1; i++ )
            {
                if( size1[i] != size2[i] )
                    break;
            }

            eq = (i == dims1);
        }
    }

    if( !eq )
    {
        cvReleaseHist( _dst );
        dst = cvCreateHist( dims1, size1, !is_sparse ? CV_HIST_ARRAY : CV_HIST_SPARSE, 0, 0 );
        *_dst = dst;
    }

    if( CV_HIST_HAS_RANGES( src ))
    {
        float* ranges[CV_MAX_DIM];
        float** thresh = 0;

        if( CV_IS_UNIFORM_HIST( src ))
        {
            for( int i = 0; i < dims1; i++ )
                ranges[i] = (float*)src->thresh[i];

            thresh = ranges;
        }
        else
        {
            thresh = src->thresh2;
        }

        cvSetHistBinRanges( dst, thresh, CV_IS_UNIFORM_HIST(src));
    }

    cvCopy( src->bins, dst->bins );
}


// Sets a value range for every histogram bin
CV_IMPL void
cvSetHistBinRanges( CvHistogram* hist, float** ranges, int uniform )
{
    int dims, size[CV_MAX_DIM], total = 0;
    int i, j;

    if( !ranges )
        CV_Error( CV_StsNullPtr, "NULL ranges pointer" );

    if( !CV_IS_HIST(hist) )
        CV_Error( CV_StsBadArg, "Invalid histogram header" );

    dims = cvGetDims( hist->bins, size );
    for( i = 0; i < dims; i++ )
        total += size[i]+1;

    if( uniform )
    {
        for( i = 0; i < dims; i++ )
        {
            if( !ranges[i] )
                CV_Error( CV_StsNullPtr, "One of <ranges> elements is NULL" );
            hist->thresh[i][0] = ranges[i][0];
            hist->thresh[i][1] = ranges[i][1];
        }

        hist->type |= CV_HIST_UNIFORM_FLAG + CV_HIST_RANGES_FLAG;
    }
    else
    {
        float* dim_ranges;

        if( !hist->thresh2 )
        {
            hist->thresh2 = (float**)cvAlloc(
                        dims*sizeof(hist->thresh2[0])+
                        total*sizeof(hist->thresh2[0][0]));
        }
        dim_ranges = (float*)(hist->thresh2 + dims);

        for( i = 0; i < dims; i++ )
        {
            float val0 = -FLT_MAX;

            if( !ranges[i] )
                CV_Error( CV_StsNullPtr, "One of <ranges> elements is NULL" );

            for( j = 0; j <= size[i]; j++ )
            {
                float val = ranges[i][j];
                if( val <= val0 )
                    CV_Error(CV_StsOutOfRange, "Bin ranges should go in ascenting order");
                val0 = dim_ranges[j] = val;
            }

            hist->thresh2[i] = dim_ranges;
            dim_ranges += size[i] + 1;
        }

        hist->type |= CV_HIST_RANGES_FLAG;
        hist->type &= ~CV_HIST_UNIFORM_FLAG;
    }
}


CV_IMPL void
cvCalcArrHist( CvArr** img, CvHistogram* hist, int accumulate, const CvArr* mask )
{
    if( !CV_IS_HIST(hist))
        CV_Error( CV_StsBadArg, "Bad histogram pointer" );

    if( !img )
        CV_Error( CV_StsNullPtr, "Null double array pointer" );

    int size[CV_MAX_DIM];
    int i, dims = cvGetDims( hist->bins, size);
    bool uniform = CV_IS_UNIFORM_HIST(hist);

    std::vector<cv::Mat> images(dims);
    for( i = 0; i < dims; i++ )
        images[i] = cv::cvarrToMat(img[i]);

    cv::Mat _mask;
    if( mask )
        _mask = cv::cvarrToMat(mask);

    const float* uranges[CV_MAX_DIM] = {0};
    const float** ranges = 0;

    if( hist->type & CV_HIST_RANGES_FLAG )
    {
        ranges = (const float**)hist->thresh2;
        if( uniform )
        {
            for( i = 0; i < dims; i++ )
                uranges[i] = &hist->thresh[i][0];
            ranges = uranges;
        }
    }

    if( !CV_IS_SPARSE_HIST(hist) )
    {
        cv::Mat H = cv::cvarrToMat(hist->bins);
        cv::calcHist( &images[0], (int)images.size(), 0, _mask,
                      H, cvGetDims(hist->bins), H.size, ranges, uniform, accumulate != 0 );
    }
    else
    {
        CvSparseMat* sparsemat = (CvSparseMat*)hist->bins;

        if( !accumulate )
            cvZero( hist->bins );
        cv::SparseMat sH;
        sparsemat->copyToSparseMat(sH);
        cv::calcHist( &images[0], (int)images.size(), 0, _mask, sH, sH.dims(),
                      sH.dims() > 0 ? sH.hdr->size : 0, ranges, uniform, accumulate != 0, true );

        if( accumulate )
            cvZero( sparsemat );

        cv::SparseMatConstIterator it = sH.begin();
        int nz = (int)sH.nzcount();
        for( i = 0; i < nz; i++, ++it )
            *(float*)cvPtrND(sparsemat, it.node()->idx, 0, -2) = (float)*(const int*)it.ptr;
    }
}


CV_IMPL void
cvCalcArrBackProject( CvArr** img, CvArr* dst, const CvHistogram* hist )
{
    if( !CV_IS_HIST(hist))
        CV_Error( CV_StsBadArg, "Bad histogram pointer" );

    if( !img )
        CV_Error( CV_StsNullPtr, "Null double array pointer" );

    int size[CV_MAX_DIM];
    int i, dims = cvGetDims( hist->bins, size );

    bool uniform = CV_IS_UNIFORM_HIST(hist);
    const float* uranges[CV_MAX_DIM] = {0};
    const float** ranges = 0;

    if( hist->type & CV_HIST_RANGES_FLAG )
    {
        ranges = (const float**)hist->thresh2;
        if( uniform )
        {
            for( i = 0; i < dims; i++ )
                uranges[i] = &hist->thresh[i][0];
            ranges = uranges;
        }
    }

    std::vector<cv::Mat> images(dims);
    for( i = 0; i < dims; i++ )
        images[i] = cv::cvarrToMat(img[i]);

    cv::Mat _dst = cv::cvarrToMat(dst);

    CV_Assert( _dst.size() == images[0].size() && _dst.depth() == images[0].depth() );

    if( !CV_IS_SPARSE_HIST(hist) )
    {
        cv::Mat H = cv::cvarrToMat(hist->bins);
        cv::calcBackProject( &images[0], (int)images.size(),
                            0, H, _dst, ranges, 1, uniform );
    }
    else
    {
        cv::SparseMat sH;
        ((const CvSparseMat*)hist->bins)->copyToSparseMat(sH);
        cv::calcBackProject( &images[0], (int)images.size(),
                             0, sH, _dst, ranges, 1, uniform );
    }
}


////////////////////// B A C K   P R O J E C T   P A T C H /////////////////////////

CV_IMPL void
cvCalcArrBackProjectPatch( CvArr** arr, CvArr* dst, CvSize patch_size, CvHistogram* hist,
                           int method, double norm_factor )
{
    CvHistogram* model = 0;

    IplImage imgstub[CV_MAX_DIM], *img[CV_MAX_DIM];
    IplROI roi;
    CvMat dststub, *dstmat;
    int i, dims;
    int x, y;
    CvSize size;

    if( !CV_IS_HIST(hist))
        CV_Error( CV_StsBadArg, "Bad histogram pointer" );

    if( !arr )
        CV_Error( CV_StsNullPtr, "Null double array pointer" );

    if( norm_factor <= 0 )
        CV_Error( CV_StsOutOfRange,
                  "Bad normalization factor (set it to 1.0 if unsure)" );

    if( patch_size.width <= 0 || patch_size.height <= 0 )
        CV_Error( CV_StsBadSize, "The patch width and height must be positive" );

    dims = cvGetDims( hist->bins );
    cvNormalizeHist( hist, norm_factor );

    for( i = 0; i < dims; i++ )
    {
        CvMat stub, *mat;
        mat = cvGetMat( arr[i], &stub, 0, 0 );
        img[i] = cvGetImage( mat, &imgstub[i] );
        img[i]->roi = &roi;
    }

    dstmat = cvGetMat( dst, &dststub, 0, 0 );
    if( CV_MAT_TYPE( dstmat->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat, "Resultant image must have 32fC1 type" );

    if( dstmat->cols != img[0]->width - patch_size.width + 1 ||
        dstmat->rows != img[0]->height - patch_size.height + 1 )
        CV_Error( CV_StsUnmatchedSizes,
            "The output map must be (W-w+1 x H-h+1), "
            "where the input images are (W x H) each and the patch is (w x h)" );

    cvCopyHist( hist, &model );

    size = cvGetMatSize(dstmat);
    roi.coi = 0;
    roi.width = patch_size.width;
    roi.height = patch_size.height;

    for( y = 0; y < size.height; y++ )
    {
        for( x = 0; x < size.width; x++ )
        {
            double result;
            roi.xOffset = x;
            roi.yOffset = y;

            cvCalcHist( img, model );
            cvNormalizeHist( model, norm_factor );
            result = cvCompareHist( model, hist, method );
            CV_MAT_ELEM( *dstmat, float, y, x ) = (float)result;
        }
    }

    cvReleaseHist( &model );
}


// Calculates Bayes probabilistic histograms
CV_IMPL void
cvCalcBayesianProb( CvHistogram** src, int count, CvHistogram** dst )
{
    int i;

    if( !src || !dst )
        CV_Error( CV_StsNullPtr, "NULL histogram array pointer" );

    if( count < 2 )
        CV_Error( CV_StsOutOfRange, "Too small number of histograms" );

    for( i = 0; i < count; i++ )
    {
        if( !CV_IS_HIST(src[i]) || !CV_IS_HIST(dst[i]) )
            CV_Error( CV_StsBadArg, "Invalid histogram header" );

        if( !CV_IS_MATND(src[i]->bins) || !CV_IS_MATND(dst[i]->bins) )
            CV_Error( CV_StsBadArg, "The function supports dense histograms only" );
    }

    cvZero( dst[0]->bins );
    // dst[0] = src[0] + ... + src[count-1]
    for( i = 0; i < count; i++ )
        cvAdd( src[i]->bins, dst[0]->bins, dst[0]->bins );

    cvDiv( 0, dst[0]->bins, dst[0]->bins );

    // dst[i] = src[i]*(1/dst[0])
    for( i = count - 1; i >= 0; i-- )
        cvMul( src[i]->bins, dst[0]->bins, dst[i]->bins );
}


CV_IMPL void
cvCalcProbDensity( const CvHistogram* hist, const CvHistogram* hist_mask,
                   CvHistogram* hist_dens, double scale )
{
    if( scale <= 0 )
        CV_Error( CV_StsOutOfRange, "scale must be positive" );

    if( !CV_IS_HIST(hist) || !CV_IS_HIST(hist_mask) || !CV_IS_HIST(hist_dens) )
        CV_Error( CV_StsBadArg, "Invalid histogram pointer[s]" );

    {
        CvArr* arrs[] = { hist->bins, hist_mask->bins, hist_dens->bins };
        CvMatND stubs[3];
        CvNArrayIterator iterator;

        cvInitNArrayIterator( 3, arrs, 0, stubs, &iterator );

        if( CV_MAT_TYPE(iterator.hdr[0]->type) != CV_32FC1 )
            CV_Error( CV_StsUnsupportedFormat, "All histograms must have 32fC1 type" );

        do
        {
            const float* srcdata = (const float*)(iterator.ptr[0]);
            const float* maskdata = (const float*)(iterator.ptr[1]);
            float* dstdata = (float*)(iterator.ptr[2]);
            int i;

            for( i = 0; i < iterator.size.width; i++ )
            {
                float s = srcdata[i];
                float m = maskdata[i];
                if( s > FLT_EPSILON )
                    if( m <= s )
                        dstdata[i] = (float)(m*scale/s);
                    else
                        dstdata[i] = (float)scale;
                else
                    dstdata[i] = (float)0;
            }
        }
        while( cvNextNArraySlice( &iterator ));
    }
}

class EqualizeHistCalcHist_Invoker : public cv::ParallelLoopBody
{
public:
    enum {HIST_SZ = 256};

    EqualizeHistCalcHist_Invoker(cv::Mat& src, int* histogram, cv::Mutex* histogramLock)
        : src_(src), globalHistogram_(histogram), histogramLock_(histogramLock)
    { }

    void operator()( const cv::Range& rowRange ) const
    {
        int localHistogram[HIST_SZ] = {0, };

        const size_t sstep = src_.step;

        int width = src_.cols;
        int height = rowRange.end - rowRange.start;

        if (src_.isContinuous())
        {
            width *= height;
            height = 1;
        }

        for (const uchar* ptr = src_.ptr<uchar>(rowRange.start); height--; ptr += sstep)
        {
            int x = 0;
            for (; x <= width - 4; x += 4)
            {
                int t0 = ptr[x], t1 = ptr[x+1];
                localHistogram[t0]++; localHistogram[t1]++;
                t0 = ptr[x+2]; t1 = ptr[x+3];
                localHistogram[t0]++; localHistogram[t1]++;
            }

            for (; x < width; ++x)
                localHistogram[ptr[x]]++;
        }

        cv::AutoLock lock(*histogramLock_);

        for( int i = 0; i < HIST_SZ; i++ )
            globalHistogram_[i] += localHistogram[i];
    }

    static bool isWorthParallel( const cv::Mat& src )
    {
        return ( src.total() >= 640*480 );
    }

private:
    EqualizeHistCalcHist_Invoker& operator=(const EqualizeHistCalcHist_Invoker&);

    cv::Mat& src_;
    int* globalHistogram_;
    cv::Mutex* histogramLock_;
};

class EqualizeHistLut_Invoker : public cv::ParallelLoopBody
{
public:
    EqualizeHistLut_Invoker( cv::Mat& src, cv::Mat& dst, int* lut )
        : src_(src),
          dst_(dst),
          lut_(lut)
    { }

    void operator()( const cv::Range& rowRange ) const
    {
        const size_t sstep = src_.step;
        const size_t dstep = dst_.step;

        int width = src_.cols;
        int height = rowRange.end - rowRange.start;
        int* lut = lut_;

        if (src_.isContinuous() && dst_.isContinuous())
        {
            width *= height;
            height = 1;
        }

        const uchar* sptr = src_.ptr<uchar>(rowRange.start);
        uchar* dptr = dst_.ptr<uchar>(rowRange.start);

        for (; height--; sptr += sstep, dptr += dstep)
        {
            int x = 0;
            for (; x <= width - 4; x += 4)
            {
                int v0 = sptr[x];
                int v1 = sptr[x+1];
                int x0 = lut[v0];
                int x1 = lut[v1];
                dptr[x] = (uchar)x0;
                dptr[x+1] = (uchar)x1;

                v0 = sptr[x+2];
                v1 = sptr[x+3];
                x0 = lut[v0];
                x1 = lut[v1];
                dptr[x+2] = (uchar)x0;
                dptr[x+3] = (uchar)x1;
            }

            for (; x < width; ++x)
                dptr[x] = (uchar)lut[sptr[x]];
        }
    }

    static bool isWorthParallel( const cv::Mat& src )
    {
        return ( src.total() >= 640*480 );
    }

private:
    EqualizeHistLut_Invoker& operator=(const EqualizeHistLut_Invoker&);

    cv::Mat& src_;
    cv::Mat& dst_;
    int* lut_;
};

CV_IMPL void cvEqualizeHist( const CvArr* srcarr, CvArr* dstarr )
{
    cv::equalizeHist(cv::cvarrToMat(srcarr), cv::cvarrToMat(dstarr));
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_equalizeHist(InputArray _src, OutputArray _dst)
{
    size_t wgs = std::min<size_t>(ocl::Device::getDefault().maxWorkGroupSize(), BINS);

    // calculation of histogram
    UMat hist;
    if (!ocl_calcHist1(_src, hist))
        return false;

    UMat lut(1, 256, CV_8UC1);
    ocl::Kernel k("calcLUT", ocl::imgproc::histogram_oclsrc, format("-D BINS=%d -D HISTS_COUNT=1 -D WGS=%d", BINS, (int)wgs));
    k.args(ocl::KernelArg::PtrWriteOnly(lut), ocl::KernelArg::PtrReadOnly(hist), (int)_src.total());

    // calculation of LUT
    if (!k.run(1, &wgs, &wgs, false))
        return false;

    // execute LUT transparently
    LUT(_src, lut, _dst);
    return true;
}

}

#endif

void cv::equalizeHist( InputArray _src, OutputArray _dst )
{
    CV_Assert( _src.type() == CV_8UC1 );

    if (_src.empty())
        return;

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_equalizeHist(_src, _dst))

    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    Mutex histogramLockInstance;

    const int hist_sz = EqualizeHistCalcHist_Invoker::HIST_SZ;
    int hist[hist_sz] = {0,};
    int lut[hist_sz];

    EqualizeHistCalcHist_Invoker calcBody(src, hist, &histogramLockInstance);
    EqualizeHistLut_Invoker      lutBody(src, dst, lut);
    cv::Range heightRange(0, src.rows);

    if(EqualizeHistCalcHist_Invoker::isWorthParallel(src))
        parallel_for_(heightRange, calcBody);
    else
        calcBody(heightRange);

    int i = 0;
    while (!hist[i]) ++i;

    int total = (int)src.total();
    if (hist[i] == total)
    {
        dst.setTo(i);
        return;
    }

    float scale = (hist_sz - 1.f)/(total - hist[i]);
    int sum = 0;

    for (lut[i++] = 0; i < hist_sz; ++i)
    {
        sum += hist[i];
        lut[i] = saturate_cast<uchar>(sum * scale);
    }

    if(EqualizeHistLut_Invoker::isWorthParallel(src))
        parallel_for_(heightRange, lutBody);
    else
        lutBody(heightRange);
}

// ----------------------------------------------------------------------

/* Implementation of RTTI and Generic Functions for CvHistogram */
#define CV_TYPE_NAME_HIST "opencv-hist"

static int icvIsHist( const void * ptr )
{
    return CV_IS_HIST( ((CvHistogram*)ptr) );
}

static CvHistogram * icvCloneHist( const CvHistogram * src )
{
    CvHistogram * dst=NULL;
    cvCopyHist(src, &dst);
    return dst;
}

static void *icvReadHist( CvFileStorage * fs, CvFileNode * node )
{
    CvHistogram * h = 0;
    int type = 0;
    int is_uniform = 0;
    int have_ranges = 0;

    h = (CvHistogram *)cvAlloc( sizeof(CvHistogram) );

    type = cvReadIntByName( fs, node, "type", 0 );
    is_uniform = cvReadIntByName( fs, node, "is_uniform", 0 );
    have_ranges = cvReadIntByName( fs, node, "have_ranges", 0 );
    h->type = CV_HIST_MAGIC_VAL | type |
        (is_uniform ? CV_HIST_UNIFORM_FLAG : 0) |
        (have_ranges ? CV_HIST_RANGES_FLAG : 0);

    if(type == CV_HIST_ARRAY)
    {
        // read histogram bins
        CvMatND* mat = (CvMatND*)cvReadByName( fs, node, "mat" );
        int i, sizes[CV_MAX_DIM];

        if(!CV_IS_MATND(mat))
            CV_Error( CV_StsError, "Expected CvMatND");

        for(i=0; i<mat->dims; i++)
            sizes[i] = mat->dim[i].size;

        cvInitMatNDHeader( &(h->mat), mat->dims, sizes, mat->type, mat->data.ptr );
        h->bins = &(h->mat);

        // take ownership of refcount pointer as well
        h->mat.refcount = mat->refcount;

        // increase refcount so freeing temp header doesn't free data
        cvIncRefData( mat );

        // free temporary header
        cvReleaseMatND( &mat );
    }
    else
    {
        h->bins = cvReadByName( fs, node, "bins" );
        if(!CV_IS_SPARSE_MAT(h->bins)){
            CV_Error( CV_StsError, "Unknown Histogram type");
        }
    }

    // read thresholds
    if(have_ranges)
    {
        int i, dims, size[CV_MAX_DIM], total = 0;
        CvSeqReader reader;
        CvFileNode * thresh_node;

        dims = cvGetDims( h->bins, size );
        for( i = 0; i < dims; i++ )
            total += size[i]+1;

        thresh_node = cvGetFileNodeByName( fs, node, "thresh" );
        if(!thresh_node)
            CV_Error( CV_StsError, "'thresh' node is missing");
        cvStartReadRawData( fs, thresh_node, &reader );

        if(is_uniform)
        {
            for(i=0; i<dims; i++)
                cvReadRawDataSlice( fs, &reader, 2, h->thresh[i], "f" );
            h->thresh2 = NULL;
        }
        else
        {
            float* dim_ranges;
            h->thresh2 = (float**)cvAlloc(
                dims*sizeof(h->thresh2[0])+
                total*sizeof(h->thresh2[0][0]));
            dim_ranges = (float*)(h->thresh2 + dims);
            for(i=0; i < dims; i++)
            {
                h->thresh2[i] = dim_ranges;
                cvReadRawDataSlice( fs, &reader, size[i]+1, dim_ranges, "f" );
                dim_ranges += size[i] + 1;
            }
        }
    }

    return h;
}

static void icvWriteHist( CvFileStorage* fs, const char* name,
                          const void* struct_ptr, CvAttrList /*attributes*/ )
{
    const CvHistogram * hist = (const CvHistogram *) struct_ptr;
    int sizes[CV_MAX_DIM];
    int dims;
    int i;
    int is_uniform, have_ranges;

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_HIST );

    is_uniform = (CV_IS_UNIFORM_HIST(hist) ? 1 : 0);
    have_ranges = (hist->type & CV_HIST_RANGES_FLAG ? 1 : 0);

    cvWriteInt( fs, "type", (hist->type & 1) );
    cvWriteInt( fs, "is_uniform", is_uniform );
    cvWriteInt( fs, "have_ranges", have_ranges );
    if(!CV_IS_SPARSE_HIST(hist))
        cvWrite( fs, "mat", &(hist->mat) );
    else
        cvWrite( fs, "bins", hist->bins );

    // write thresholds
    if(have_ranges){
        dims = cvGetDims( hist->bins, sizes );
        cvStartWriteStruct( fs, "thresh", CV_NODE_SEQ + CV_NODE_FLOW );
        if(is_uniform){
            for(i=0; i<dims; i++){
                cvWriteRawData( fs, hist->thresh[i], 2, "f" );
            }
        }
        else{
            for(i=0; i<dims; i++){
                cvWriteRawData( fs, hist->thresh2[i], sizes[i]+1, "f" );
            }
        }
        cvEndWriteStruct( fs );
    }

    cvEndWriteStruct( fs );
}


CvType hist_type( CV_TYPE_NAME_HIST, icvIsHist, (CvReleaseFunc)cvReleaseHist,
                  icvReadHist, icvWriteHist, (CvCloneFunc)icvCloneHist );

/* End of file. */
