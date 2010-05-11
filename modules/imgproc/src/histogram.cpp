/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//			  Intel License Agreement
//		  For Open Source Computer Vision Library
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

namespace cv
{

////////////////// Helper functions //////////////////////

static const size_t OUT_OF_RANGE = (size_t)1 << (sizeof(size_t)*8 - 2);

static void
calcHistLookupTables_8u( const MatND& hist, const SparseMat& shist,
                        const float** ranges,
                        const double* uniranges,
                        bool uniform, bool issparse, vector<size_t>& _tab )
{
    const int low = 0, high = 256;
    int i, j, dims = !issparse ? hist.dims : shist.dims();
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
                              vector<uchar*>& ptrs, vector<int>& deltas,
                              Size& imsize, vector<double>& uniranges )
{
    int i, j, c;
    if(!channels)
        CV_Assert( nimages == dims );
    
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
    
    if( isContinuous )
    {
        imsize.width *= imsize.height;
        imsize.height = 1;
    }
    
    if( !ranges )
    {
        CV_Assert( depth == CV_8U );
        
        uniranges.resize( dims*2 );
        for( i = 0; i < dims; i++ )
        {
            uniranges[i*2] = 1;
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
            size_t j, n = histSize[i];
            for( j = 0; j < n; j++ )
                CV_Assert( ranges[i][j] < ranges[i][j+1] );
        }
    }
}
    
    
////////////////////////////////// C A L C U L A T E    H I S T O G R A M ////////////////////////////////////        
    
template<typename T> static void
calcHist_( vector<uchar*>& _ptrs, const vector<int>& _deltas,
           Size imsize, MatND& hist, const float** _ranges,
           const double* _uniranges, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int i, x, dims = hist.dims;
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
        }
        else if( dims == 2 )
        {
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
calcHist_8u( vector<uchar*>& _ptrs, const vector<int>& _deltas,
             Size imsize, MatND& hist, const float** _ranges,
             const double* _uniranges, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int i, x, dims = hist.dims;
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    vector<size_t> _tab;
    
    calcHistLookupTables_8u( hist, SparseMat(), _ranges, _uniranges, uniform, false, _tab );
    const size_t* tab = &_tab[0];
    
    if( dims == 1 )
    {
        int d0 = deltas[0], step0 = deltas[1];
        int matH[256] = {0};
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
        
        for( i = 0; i < 256; i++ )
        {
            size_t hidx = tab[i];
            if( hidx < OUT_OF_RANGE )
                *(int*)(H + hidx) += matH[i];
        }
    }
    else if( dims == 2 )
    {
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
                    for( i = 0; i < dims; i++ )
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
            for( i = 0; i < dims; i++ )
                ptrs[i] += deltas[i*2 + 1];
        }
    }
}


void calcHist( const Mat* images, int nimages, const int* channels,
               const Mat& mask, MatND& hist, int dims, const int* histSize,
               const float** ranges, bool uniform, bool accumulate )
{
    CV_Assert(dims > 0 && histSize);
    hist.create(dims, histSize, CV_32F);    
        
    MatND ihist = hist;
    ihist.flags = (ihist.flags & ~CV_MAT_TYPE_MASK)|CV_32S;
    
    if( !accumulate )
        hist = Scalar(0.);
    else
        hist.convertTo(ihist, CV_32S);
    
    vector<uchar*> ptrs;
    vector<int> deltas;
    vector<double> uniranges;
    Size imsize;
    
    CV_Assert( !mask.data || mask.type() == CV_8UC1 );
    histPrepareImages( images, nimages, channels, mask, hist.dims, hist.size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;
    
    int depth = images[0].depth();
    if( depth == CV_8U )
        calcHist_8u(ptrs, deltas, imsize, ihist, ranges, _uniranges, uniform );
    else if( depth == CV_32F )
        calcHist_<float>(ptrs, deltas, imsize, ihist, ranges, _uniranges, uniform );
    
    ihist.convertTo(hist, CV_32F);
}

    
template<typename T> static void
calcSparseHist_( vector<uchar*>& _ptrs, const vector<int>& _deltas,
                 Size imsize, SparseMat& hist, const float** _ranges,
                 const double* _uniranges, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x, dims = hist.dims();
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
calcSparseHist_8u( vector<uchar*>& _ptrs, const vector<int>& _deltas,
                   Size imsize, SparseMat& hist, const float** _ranges,
                   const double* _uniranges, bool uniform )
{
    uchar** ptrs = (uchar**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x, dims = hist.dims();
    const uchar* mask = _ptrs[dims];
    int mstep = _deltas[dims*2 + 1];
    int idx[CV_MAX_DIM];
    vector<size_t> _tab;
    
    calcHistLookupTables_8u( MatND(), hist, _ranges, _uniranges, uniform, true, _tab );
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
        for( i = 0; i < dims; i++ )
            ptrs[i] += deltas[i*2 + 1];
    }
}   
    

static void calcHist( const Mat* images, int nimages, const int* channels,
                      const Mat& mask, SparseMat& hist, int dims, const int* histSize,
                      const float** ranges, bool uniform, bool accumulate, bool keepInt )
{
    SparseMatIterator it;
    size_t i, N;
    
    if( !accumulate )
        hist.create(dims, histSize, CV_32F);
    else
        for( i = 0, N = hist.nzcount(); i < N; i++, ++it )
        {
            int* value = (int*)it.ptr;
            *value = cvRound(*(const float*)value);
        }
    
    vector<uchar*> ptrs;
    vector<int> deltas;
    vector<double> uniranges;
    Size imsize;
    
    CV_Assert( !mask.data || mask.type() == CV_8UC1 );
    histPrepareImages( images, nimages, channels, mask, hist.dims(), hist.hdr->size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;
    
    int depth = images[0].depth();
    if( depth == CV_8U )
        calcSparseHist_8u(ptrs, deltas, imsize, hist, ranges, _uniranges, uniform );
    else if( depth == CV_32F )
        calcSparseHist_<float>(ptrs, deltas, imsize, hist, ranges, _uniranges, uniform );
    
    if( !keepInt )
        for( i = 0, N = hist.nzcount(); i < N; i++, ++it )
        {
            int* value = (int*)it.ptr;
            *(float*)value = (float)*value;
        }
}
    
    
void calcHist( const Mat* images, int nimages, const int* channels,
               const Mat& mask, SparseMat& hist, int dims, const int* histSize,
               const float** ranges, bool uniform, bool accumulate )
{
    calcHist( images, nimages, channels, mask, hist, dims, histSize,
              ranges, uniform, accumulate, false );
}

    
/////////////////////////////////////// B A C K   P R O J E C T ////////////////////////////////////    
    

template<typename T, typename BT> static void
calcBackProj_( vector<uchar*>& _ptrs, const vector<int>& _deltas,
               Size imsize, const MatND& hist, const float** _ranges,
               const double* _uniranges, float scale, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int i, x, dims = hist.dims;
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
                        if( (unsigned)idx >= (unsigned)size[i] )
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
calcBackProj_8u( vector<uchar*>& _ptrs, const vector<int>& _deltas,
                 Size imsize, const MatND& hist, const float** _ranges,
                 const double* _uniranges, float scale, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    uchar* H = hist.data;
    int i, x, dims = hist.dims;
    uchar* bproj = _ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    vector<size_t> _tab;
    
    calcHistLookupTables_8u( hist, SparseMat(), _ranges, _uniranges, uniform, false, _tab );
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
    
    
void calcBackProject( const Mat* images, int nimages, const int* channels,
                      const MatND& hist, Mat& backProject,
                      const float** ranges, double scale, bool uniform )
{
    vector<uchar*> ptrs;
    vector<int> deltas;
    vector<double> uniranges;
    Size imsize;
    
    CV_Assert( hist.dims > 0 && hist.data );
    backProject.create( images[0].size(), images[0].depth() );
    histPrepareImages( images, nimages, channels, backProject, hist.dims, hist.size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;
    
    int depth = images[0].depth();
    if( depth == CV_8U )
        calcBackProj_8u(ptrs, deltas, imsize, hist, ranges, _uniranges, (float)scale, uniform);
    else if( depth == CV_32F )
        calcBackProj_<float, float>(ptrs, deltas, imsize, hist, ranges, _uniranges, (float)scale, uniform );
}

    
template<typename T, typename BT> static void
calcSparseBackProj_( vector<uchar*>& _ptrs, const vector<int>& _deltas,
                     Size imsize, const SparseMat& hist, const float** _ranges,
                     const double* _uniranges, float scale, bool uniform )
{
    T** ptrs = (T**)&_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x, dims = hist.dims();
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
calcSparseBackProj_8u( vector<uchar*>& _ptrs, const vector<int>& _deltas,
                       Size imsize, const SparseMat& hist, const float** _ranges,
                       const double* _uniranges, float scale, bool uniform )
{
    uchar** ptrs = &_ptrs[0];
    const int* deltas = &_deltas[0];
    int i, x, dims = hist.dims();
    uchar* bproj = _ptrs[dims];
    int bpstep = _deltas[dims*2 + 1];
    vector<size_t> _tab;
    int idx[CV_MAX_DIM];
    
    calcHistLookupTables_8u( MatND(), hist, _ranges, _uniranges, uniform, true, _tab );
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
    

void calcBackProject( const Mat* images, int nimages, const int* channels,
                      const SparseMat& hist, Mat& backProject,
                      const float** ranges, double scale, bool uniform )
{
    vector<uchar*> ptrs;
    vector<int> deltas;
    vector<double> uniranges;
    Size imsize;
    
    CV_Assert( hist.dims() > 0 );
    backProject.create( images[0].size(), images[0].depth() );
    histPrepareImages( images, nimages, channels, backProject,
                       hist.dims(), hist.hdr->size, ranges,
                       uniform, ptrs, deltas, imsize, uniranges );
    const double* _uniranges = uniform ? &uniranges[0] : 0;
    
    int depth = images[0].depth();
    if( depth == CV_8U )
        calcSparseBackProj_8u(ptrs, deltas, imsize, hist, ranges, _uniranges, (float)scale, uniform);
    else if( depth == CV_32F )
        calcSparseBackProj_<float, float>(ptrs, deltas, imsize, hist, ranges, _uniranges, (float)scale, uniform );
}

    
////////////////// C O M P A R E   H I S T O G R A M S ////////////////////////    

double compareHist( const MatND& H1, const MatND& H2, int method )
{
    NAryMatNDIterator it(H1, H2);
    double result = 0;
    int i, len;
    
    CV_Assert( H1.type() == H2.type() && H1.type() == CV_32F );
    
    double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;
    
    CV_Assert( it.planes[0].isContinuous() && it.planes[1].isContinuous() );
    
    for( i = 0; i < it.nplanes; i++, ++it )
    {
        const float* h1 = (const float*)it.planes[0].data;
        const float* h2 = (const float*)it.planes[1].data;
        len = it.planes[0].rows*it.planes[0].cols;
        
        if( method == CV_COMP_CHISQR )
        {
            for( i = 0; i < len; i++ )
            {
                double a = h1[i] - h2[i];
                double b = h1[i] + h2[i];
                if( fabs(b) > FLT_EPSILON )
                    result += a*a/b;
            }
        }
        else if( method == CV_COMP_CORREL )
        {
            for( i = 0; i < len; i++ )
            {
                double a = h1[i];
                double b = h2[i];
                
                s12 += a*b;
                s1 += a;
                s11 += a*a;
                s2 += b;
                s22 += b*b;
            }
        }
        else if( method == CV_COMP_INTERSECT )
        {
            for( i = 0; i < len; i++ )
                result += std::min(h1[i], h2[i]);
        }
        else if( method == CV_COMP_BHATTACHARYYA )
        {
            for( i = 0; i < len; i++ )
            {
                double a = h1[i];
                double b = h2[i];
                result += std::sqrt(a*b);
                s1 += a;
                s2 += b;
            }
        }
        else
            CV_Error( CV_StsBadArg, "Unknown comparison method" );
    }
    
    if( method == CV_COMP_CORREL )
    {
        size_t total = 1;
        for( i = 0; i < H1.dims; i++ )
            total *= H1.size[i];
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

    
double compareHist( const SparseMat& H1, const SparseMat& H2, int method )
{
    double result = 0;
    int i, dims = H1.dims();
    
    CV_Assert( dims > 0 && dims == H2.dims() && H1.type() == H2.type() && H1.type() == CV_32F );
    for( i = 0; i < dims; i++ )
        CV_Assert( H1.size(i) == H2.size(i) );
    
    const SparseMat *PH1 = &H1, *PH2 = &H2;
    if( PH1->nzcount() > PH2->nzcount() )
        std::swap(PH1, PH2);
    
    SparseMatConstIterator it = PH1->begin();
    int N1 = (int)PH1->nzcount(), N2 = (int)PH2->nzcount();
    
    if( method == CV_COMP_CHISQR )
    {
        for( i = 0; i < N1; i++, ++it )
        {
            float v1 = it.value<float>();
            const SparseMat::Node* node = it.node();
            float v2 = PH2->value<float>(node->idx, (size_t*)&node->hashval);
            if( !v2 )
                result += v1;
            else
            {
                double a = v1 - v2;
                double b = v1 + v2;
                if( b > FLT_EPSILON )
                    result += a*a/b;
            }
        }
        
        it = PH2->begin();
        for( i = 0; i < N2; i++, ++it )
        {
            float v2 = it.value<float>();
            const SparseMat::Node* node = it.node();
            if( !PH1->find<float>(node->idx, (size_t*)&node->hashval) )
                result += v2;
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
    
    return result;
}

    
template<> void Ptr<CvHistogram>::delete_obj()
{ cvReleaseHist(&obj); }
    
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
    int i, dims, size[CV_MAX_DIM];

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
            int i;
           
            for( i = dims - 1; i >= 0; i-- )
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

        for( i = 0; i < dims; i++ )
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
        cv::MatND H1((const CvMatND*)hist1->bins), H2((const CvMatND*)hist2->bins);
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

    if( mat1->heap->active_count > mat2->heap->active_count )
    {
        CvSparseMat* t;
        CV_SWAP( mat1, mat2, t );
    }

    if( method == CV_COMP_CHISQR )
    {
        for( node1 = cvInitSparseMatIterator( mat1, &iterator );
             node1 != 0; node1 = cvGetNextSparseNode( &iterator ))
        {
            double v1 = *(float*)CV_NODE_VAL(mat1,node1);
            uchar* node2_data = cvPtrND( mat2, CV_NODE_IDX(mat1,node1), 0, 0, &node1->hashval );
            if( !node2_data )
                result += v1;
            else
            {
                double v2 = *(float*)node2_data;
                double a = v1 - v2;
                double b = v1 + v2;
                if( fabs(b) > DBL_EPSILON )
                    result += a*a/b;
            }
        }

        for( node2 = cvInitSparseMatIterator( mat2, &iterator );
             node2 != 0; node2 = cvGetNextSparseNode( &iterator ))
        {
            double v2 = *(float*)CV_NODE_VAL(mat2,node2);
            if( !cvPtrND( mat1, CV_NODE_IDX(mat2,node2), 0, 0, &node2->hashval ))
                result += v2;
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
    
    return result;
}

// copies one histogram to another
CV_IMPL void
cvCopyHist( const CvHistogram* src, CvHistogram** _dst )
{
    int eq = 0;
    int is_sparse;
    int i, dims1, dims2;
    int size1[CV_MAX_DIM], size2[CV_MAX_DIM], total = 1;
    float* ranges[CV_MAX_DIM];
    float** thresh = 0;
    CvHistogram* dst;
    
    if( !_dst )
        CV_Error( CV_StsNullPtr, "Destination double pointer is NULL" );

    dst = *_dst;

    if( !CV_IS_HIST(src) || (dst && !CV_IS_HIST(dst)) )
        CV_Error( CV_StsBadArg, "Invalid histogram header[s]" );

    is_sparse = CV_IS_SPARSE_MAT(src->bins);
    dims1 = cvGetDims( src->bins, size1 );
    for( i = 0; i < dims1; i++ )
        total *= size1[i];

    if( dst && is_sparse == CV_IS_SPARSE_MAT(dst->bins))
    {
        dims2 = cvGetDims( dst->bins, size2 );
    
        if( dims1 == dims2 )
        {
            for( i = 0; i < dims1; i++ )
                if( size1[i] != size2[i] )
                    break;
        }

        eq = i == dims1;
    }

    if( !eq )
    {
        cvReleaseHist( _dst );
        dst = cvCreateHist( dims1, size1, !is_sparse ? CV_HIST_ARRAY : CV_HIST_SPARSE, 0, 0 );
        *_dst = dst;
    }

    if( CV_HIST_HAS_RANGES( src ))
    {
        if( CV_IS_UNIFORM_HIST( src ))
        {
            for( i = 0; i < dims1; i++ )
                ranges[i] = (float*)src->thresh[i];
            thresh = ranges;
        }
        else
            thresh = src->thresh2;
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
    
    cv::vector<cv::Mat> images(dims);
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
        cv::MatND H((const CvMatND*)hist->bins);
        cv::calcHist( &images[0], (int)images.size(), 0, _mask,
                      H, H.dims, H.size, ranges, uniform, accumulate != 0 );
    }
    else
    {
        CvSparseMat* sparsemat = (CvSparseMat*)hist->bins;
        
        if( !accumulate )
            cvZero( hist->bins );
        cv::SparseMat sH(sparsemat);
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
    
    cv::vector<cv::Mat> images(dims);
    for( i = 0; i < dims; i++ )
        images[i] = cv::cvarrToMat(img[i]);
    
    cv::Mat _dst = cv::cvarrToMat(dst);
    
    CV_Assert( _dst.size() == images[0].size() && _dst.depth() == images[0].depth() );
    
    if( !CV_IS_SPARSE_HIST(hist) )
    {
        cv::MatND H((const CvMatND*)hist->bins);
        cv::calcBackProject( &images[0], (int)images.size(),
                            0, H, _dst, ranges, 1, uniform );
    }
    else
    {
        cv::SparseMat sH((const CvSparseMat*)hist->bins);
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


CV_IMPL void cvEqualizeHist( const CvArr* srcarr, CvArr* dstarr )
{
    CvMat sstub, *src = cvGetMat(srcarr, &sstub);
    CvMat dstub, *dst = cvGetMat(dstarr, &dstub);
    
    CV_Assert( CV_ARE_SIZES_EQ(src, dst) && CV_ARE_TYPES_EQ(src, dst) &&
               CV_MAT_TYPE(src->type) == CV_8UC1 );
    CvSize size = cvGetMatSize(src);
    if( CV_IS_MAT_CONT(src->type & dst->type) )
    {
        size.width *= size.height;
        size.height = 1;
    }
    int x, y;
    const int hist_sz = 256;
    int hist[hist_sz];
    memset(hist, 0, sizeof(hist));
    
    for( y = 0; y < size.height; y++ )
    {
        const uchar* sptr = src->data.ptr + src->step*y;
        for( x = 0; x < size.width; x++ )
            hist[sptr[x]]++;
    }
    
    float scale = 255.f/(size.width*size.height);
    int sum = 0;
    uchar lut[hist_sz+1];

    for( int i = 0; i < hist_sz; i++ )
    {
        sum += hist[i];
        int val = cvRound(sum*scale);
        lut[i] = CV_CAST_8U(val);
    }

    lut[0] = 0;
    for( y = 0; y < size.height; y++ )
    {
        const uchar* sptr = src->data.ptr + src->step*y;
        uchar* dptr = dst->data.ptr + dst->step*y;
        for( x = 0; x < size.width; x++ )
            dptr[x] = lut[sptr[x]];
    }
}


void cv::equalizeHist( const Mat& src, Mat& dst )
{
    dst.create( src.size(), src.type() );
    CvMat _src = src, _dst = dst;
    cvEqualizeHist( &_src, &_dst );
}

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

