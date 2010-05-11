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
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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

static void
icvComputeIntegralImages( const CvMat* matI, CvMat* matS, CvMat* matT, CvMat* _FT )
{
    int x, y, rows = matI->rows, cols = matI->cols;
    const uchar* I = matI->data.ptr;
    int *S = matS->data.i, *T = matT->data.i, *FT = _FT->data.i;
    int istep = matI->step, step = matS->step/sizeof(S[0]);
    
    assert( CV_MAT_TYPE(matI->type) == CV_8UC1 &&
        CV_MAT_TYPE(matS->type) == CV_32SC1 &&
        CV_ARE_TYPES_EQ(matS, matT) && CV_ARE_TYPES_EQ(matS, _FT) &&
        CV_ARE_SIZES_EQ(matS, matT) && CV_ARE_SIZES_EQ(matS, _FT) &&
        matS->step == matT->step && matS->step == _FT->step &&
        matI->rows+1 == matS->rows && matI->cols+1 == matS->cols );

    for( x = 0; x <= cols; x++ )
        S[x] = T[x] = FT[x] = 0;

    S += step; T += step; FT += step;
    S[0] = T[0] = 0;
    FT[0] = I[0];
    for( x = 1; x < cols; x++ )
    {
        S[x] = S[x-1] + I[x-1];
        T[x] = I[x-1];
        FT[x] = I[x] + I[x-1];
    }
    S[cols] = S[cols-1] + I[cols-1];
    T[cols] = FT[cols] = I[cols-1];

    for( y = 2; y <= rows; y++ )
    {
        I += istep, S += step, T += step, FT += step;

        S[0] = S[-step]; S[1] = S[-step+1] + I[0];
        T[0] = T[-step + 1];
        T[1] = FT[0] = T[-step + 2] + I[-istep] + I[0];
        FT[1] = FT[-step + 2] + I[-istep] + I[1] + I[0];

        for( x = 2; x < cols; x++ )
        {
            S[x] = S[x - 1] + S[-step + x] - S[-step + x - 1] + I[x - 1];
            T[x] = T[-step + x - 1] + T[-step + x + 1] - T[-step*2 + x] + I[-istep + x - 1] + I[x - 1];
            FT[x] = FT[-step + x - 1] + FT[-step + x + 1] - FT[-step*2 + x] + I[x] + I[x-1];
        }

        S[cols] = S[cols - 1] + S[-step + cols] - S[-step + cols - 1] + I[cols - 1];
        T[cols] = FT[cols] = T[-step + cols - 1] + I[-istep + cols - 1] + I[cols - 1];
    }
}

typedef struct CvStarFeature
{
    int area;
    int* p[8];
}
CvStarFeature;

static int
icvStarDetectorComputeResponses( const CvMat* img, CvMat* responses, CvMat* sizes,
                                 const CvStarDetectorParams* params )
{
    const int MAX_PATTERN = 17;
    static const int sizes0[] = {1, 2, 3, 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128, -1};
    static const int pairs[][2] = {{1, 0}, {3, 1}, {4, 2}, {5, 3}, {7, 4}, {8, 5}, {9, 6},
                                   {11, 8}, {13, 10}, {14, 11}, {15, 12}, {16, 14}, {-1, -1}};
    float invSizes[MAX_PATTERN][2];
    int sizes1[MAX_PATTERN];

#if CV_SSE2
    __m128 invSizes4[MAX_PATTERN][2];
    __m128 sizes1_4[MAX_PATTERN];
    Cv32suf absmask;
    absmask.i = 0x7fffffff;
    volatile bool useSIMD = cv::checkHardwareSupport(CV_CPU_SSE2);
#endif
    CvStarFeature f[MAX_PATTERN];

    CvMat *sum = 0, *tilted = 0, *flatTilted = 0;
    int y, i=0, rows = img->rows, cols = img->cols, step;
    int border, npatterns=0, maxIdx=0;
#ifdef _OPENMP
    int nthreads = cvGetNumThreads();
#endif

    assert( CV_MAT_TYPE(img->type) == CV_8UC1 &&
        CV_MAT_TYPE(responses->type) == CV_32FC1 &&
        CV_MAT_TYPE(sizes->type) == CV_16SC1 &&
        CV_ARE_SIZES_EQ(responses, sizes) );

    while( pairs[i][0] >= 0 && !
          ( sizes0[pairs[i][0]] >= params->maxSize 
           || sizes0[pairs[i+1][0]] + sizes0[pairs[i+1][0]]/2 >= std::min(rows, cols) ) )
    {
        ++i;
    }
    
    npatterns = i;
    npatterns += (pairs[npatterns-1][0] >= 0);
    maxIdx = pairs[npatterns-1][0];

    sum = cvCreateMat( rows + 1, cols + 1, CV_32SC1 );
    tilted = cvCreateMat( rows + 1, cols + 1, CV_32SC1 );
    flatTilted = cvCreateMat( rows + 1, cols + 1, CV_32SC1 );
    step = sum->step/CV_ELEM_SIZE(sum->type);

    icvComputeIntegralImages( img, sum, tilted, flatTilted );

    for( i = 0; i <= maxIdx; i++ )
    {
        int ur_size = sizes0[i], t_size = sizes0[i] + sizes0[i]/2;
        int ur_area = (2*ur_size + 1)*(2*ur_size + 1);
        int t_area = t_size*t_size + (t_size + 1)*(t_size + 1);

        f[i].p[0] = sum->data.i + (ur_size + 1)*step + ur_size + 1;
        f[i].p[1] = sum->data.i - ur_size*step + ur_size + 1;
        f[i].p[2] = sum->data.i + (ur_size + 1)*step - ur_size;
        f[i].p[3] = sum->data.i - ur_size*step - ur_size;

        f[i].p[4] = tilted->data.i + (t_size + 1)*step + 1;
        f[i].p[5] = flatTilted->data.i - t_size;
        f[i].p[6] = flatTilted->data.i + t_size + 1;
        f[i].p[7] = tilted->data.i - t_size*step + 1;

        f[i].area = ur_area + t_area;
        sizes1[i] = sizes0[i];
    }
    // negate end points of the size range
    // for a faster rejection of very small or very large features in non-maxima suppression.
    sizes1[0] = -sizes1[0];
    sizes1[1] = -sizes1[1];
    sizes1[maxIdx] = -sizes1[maxIdx];
    border = sizes0[maxIdx] + sizes0[maxIdx]/2;

    for( i = 0; i < npatterns; i++ )
    {
        int innerArea = f[pairs[i][1]].area;
        int outerArea = f[pairs[i][0]].area - innerArea;
        invSizes[i][0] = 1.f/outerArea;
        invSizes[i][1] = 1.f/innerArea;
    }
    
#if CV_SSE2
    if( useSIMD )
    {
        for( i = 0; i < npatterns; i++ )
        {
            _mm_store_ps((float*)&invSizes4[i][0], _mm_set1_ps(invSizes[i][0]));
            _mm_store_ps((float*)&invSizes4[i][1], _mm_set1_ps(invSizes[i][1]));
        }

        for( i = 0; i <= maxIdx; i++ )
            _mm_store_ps((float*)&sizes1_4[i], _mm_set1_ps((float)sizes1[i]));
    }
#endif

    for( y = 0; y < border; y++ )
    {
        float* r_ptr = (float*)(responses->data.ptr + responses->step*y);
        float* r_ptr2 = (float*)(responses->data.ptr + responses->step*(rows - 1 - y));
        short* s_ptr = (short*)(sizes->data.ptr + sizes->step*y);
        short* s_ptr2 = (short*)(sizes->data.ptr + sizes->step*(rows - 1 - y));
        
        memset( r_ptr, 0, cols*sizeof(r_ptr[0]));
        memset( r_ptr2, 0, cols*sizeof(r_ptr2[0]));
        memset( s_ptr, 0, cols*sizeof(s_ptr[0]));
        memset( s_ptr2, 0, cols*sizeof(s_ptr2[0]));
    }

#ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for( y = border; y < rows - border; y++ )
    {
        int x = border, i;
        float* r_ptr = (float*)(responses->data.ptr + responses->step*y);
        short* s_ptr = (short*)(sizes->data.ptr + sizes->step*y);
        
        memset( r_ptr, 0, border*sizeof(r_ptr[0]));
        memset( s_ptr, 0, border*sizeof(s_ptr[0]));
        memset( r_ptr + cols - border, 0, border*sizeof(r_ptr[0]));
        memset( s_ptr + cols - border, 0, border*sizeof(s_ptr[0]));

#if CV_SSE2
        if( useSIMD )
        {
            __m128 absmask4 = _mm_set1_ps(absmask.f);
            for( ; x <= cols - border - 4; x += 4 )
            {
                int ofs = y*step + x;
                __m128 vals[MAX_PATTERN];
                __m128 bestResponse = _mm_setzero_ps();
                __m128 bestSize = _mm_setzero_ps();

                for( i = 0; i <= maxIdx; i++ )
                {
                    const int** p = (const int**)&f[i].p[0];
                    __m128i r0 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[0]+ofs)),
                                               _mm_loadu_si128((const __m128i*)(p[1]+ofs)));
                    __m128i r1 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[3]+ofs)),
                                               _mm_loadu_si128((const __m128i*)(p[2]+ofs)));
                    __m128i r2 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[4]+ofs)),
                                               _mm_loadu_si128((const __m128i*)(p[5]+ofs)));
                    __m128i r3 = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(p[7]+ofs)),
                                               _mm_loadu_si128((const __m128i*)(p[6]+ofs)));
                    r0 = _mm_add_epi32(_mm_add_epi32(r0,r1), _mm_add_epi32(r2,r3));
                    _mm_store_ps((float*)&vals[i], _mm_cvtepi32_ps(r0));
                }

                for( i = 0; i < npatterns; i++ )
                {
                    __m128 inner_sum = vals[pairs[i][1]];
                    __m128 outer_sum = _mm_sub_ps(vals[pairs[i][0]], inner_sum);
                    __m128 response = _mm_sub_ps(_mm_mul_ps(inner_sum, invSizes4[i][1]),
                        _mm_mul_ps(outer_sum, invSizes4[i][0]));
                    __m128 swapmask = _mm_cmpgt_ps(_mm_and_ps(response,absmask4),
                        _mm_and_ps(bestResponse,absmask4));
                    bestResponse = _mm_xor_ps(bestResponse,
                        _mm_and_ps(_mm_xor_ps(response,bestResponse), swapmask));
                    bestSize = _mm_xor_ps(bestSize,
                        _mm_and_ps(_mm_xor_ps(sizes1_4[pairs[i][0]], bestSize), swapmask));
                }

                _mm_storeu_ps(r_ptr + x, bestResponse);
                _mm_storel_epi64((__m128i*)(s_ptr + x),
                    _mm_packs_epi32(_mm_cvtps_epi32(bestSize),_mm_setzero_si128()));
            }
        }
#endif        
        for( ; x < cols - border; x++ )
        {
            int ofs = y*step + x;
            int vals[MAX_PATTERN];
            float bestResponse = 0;
            int bestSize = 0;

            for( i = 0; i <= maxIdx; i++ )
            {
                const int** p = (const int**)&f[i].p[0];
                vals[i] = p[0][ofs] - p[1][ofs] - p[2][ofs] + p[3][ofs] +
                    p[4][ofs] - p[5][ofs] - p[6][ofs] + p[7][ofs];
            }
            for( i = 0; i < npatterns; i++ )
            {
                int inner_sum = vals[pairs[i][1]];
                int outer_sum = vals[pairs[i][0]] - inner_sum;
                float response = inner_sum*invSizes[i][1] - outer_sum*invSizes[i][0];
                if( fabs(response) > fabs(bestResponse) )
                {
                    bestResponse = response;
                    bestSize = sizes1[pairs[i][0]];
                }
            }

            r_ptr[x] = bestResponse;
            s_ptr[x] = (short)bestSize;
        }
    }

    cvReleaseMat(&sum);
    cvReleaseMat(&tilted);
    cvReleaseMat(&flatTilted);

    return border;
}


static bool
icvStarDetectorSuppressLines( const CvMat* responses, const CvMat* sizes, CvPoint pt,
                              const CvStarDetectorParams* params )
{
    const float* r_ptr = responses->data.fl;
    int rstep = responses->step/sizeof(r_ptr[0]);
    const short* s_ptr = sizes->data.s;
    int sstep = sizes->step/sizeof(s_ptr[0]);
    int sz = s_ptr[pt.y*sstep + pt.x];
    int x, y, delta = sz/4, radius = delta*4;
    float Lxx = 0, Lyy = 0, Lxy = 0;
    int Lxxb = 0, Lyyb = 0, Lxyb = 0;
    
    for( y = pt.y - radius; y <= pt.y + radius; y += delta )
        for( x = pt.x - radius; x <= pt.x + radius; x += delta )
        {
            float Lx = r_ptr[y*rstep + x + 1] - r_ptr[y*rstep + x - 1];
            float Ly = r_ptr[(y+1)*rstep + x] - r_ptr[(y-1)*rstep + x];
            Lxx += Lx*Lx; Lyy += Ly*Ly; Lxy += Lx*Ly;
        }
    
    if( (Lxx + Lyy)*(Lxx + Lyy) >= params->lineThresholdProjected*(Lxx*Lyy - Lxy*Lxy) )
        return true;

    for( y = pt.y - radius; y <= pt.y + radius; y += delta )
        for( x = pt.x - radius; x <= pt.x + radius; x += delta )
        {
            int Lxb = (s_ptr[y*sstep + x + 1] == sz) - (s_ptr[y*sstep + x - 1] == sz);
            int Lyb = (s_ptr[(y+1)*sstep + x] == sz) - (s_ptr[(y-1)*sstep + x] == sz);
            Lxxb += Lxb * Lxb; Lyyb += Lyb * Lyb; Lxyb += Lxb * Lyb;
        }

    if( (Lxxb + Lyyb)*(Lxxb + Lyyb) >= params->lineThresholdBinarized*(Lxxb*Lyyb - Lxyb*Lxyb) )
        return true;

    return false;
}


static void
icvStarDetectorSuppressNonmax( const CvMat* responses, const CvMat* sizes,
                               CvSeq* keypoints, int border,
                               const CvStarDetectorParams* params )
{
    int x, y, x1, y1, delta = params->suppressNonmaxSize/2;
    int rows = responses->rows, cols = responses->cols;
    const float* r_ptr = responses->data.fl;
    int rstep = responses->step/sizeof(r_ptr[0]);
    const short* s_ptr = sizes->data.s;
    int sstep = sizes->step/sizeof(s_ptr[0]);
    short featureSize = 0;

    for( y = border; y < rows - border; y += delta+1 )
        for( x = border; x < cols - border; x += delta+1 )
        {
            float maxResponse = (float)params->responseThreshold;
            float minResponse = (float)-params->responseThreshold;
            CvPoint maxPt = {-1,-1}, minPt = {-1,-1};
            int tileEndY = MIN(y + delta, rows - border - 1);
            int tileEndX = MIN(x + delta, cols - border - 1);

            for( y1 = y; y1 <= tileEndY; y1++ )
                for( x1 = x; x1 <= tileEndX; x1++ )
                {
                    float val = r_ptr[y1*rstep + x1];
                    if( maxResponse < val )
                    {
                        maxResponse = val;
                        maxPt = cvPoint(x1, y1);
                    }
                    else if( minResponse > val )
                    {
                        minResponse = val;
                        minPt = cvPoint(x1, y1);
                    }
                }

            if( maxPt.x >= 0 )
            {
                for( y1 = maxPt.y - delta; y1 <= maxPt.y + delta; y1++ )
                    for( x1 = maxPt.x - delta; x1 <= maxPt.x + delta; x1++ )
                    {
                        float val = r_ptr[y1*rstep + x1];
                        if( val >= maxResponse && (y1 != maxPt.y || x1 != maxPt.x))
                            goto skip_max;
                    }

                if( (featureSize = s_ptr[maxPt.y*sstep + maxPt.x]) >= 4 &&
                    !icvStarDetectorSuppressLines( responses, sizes, maxPt, params ))
                {
                    CvStarKeypoint kpt = cvStarKeypoint( maxPt, featureSize, maxResponse );
                    cvSeqPush( keypoints, &kpt );
                }
            }
        skip_max:
            if( minPt.x >= 0 )
            {
                for( y1 = minPt.y - delta; y1 <= minPt.y + delta; y1++ )
                    for( x1 = minPt.x - delta; x1 <= minPt.x + delta; x1++ )
                    {
                        float val = r_ptr[y1*rstep + x1];
                        if( val <= minResponse && (y1 != minPt.y || x1 != minPt.x))
                            goto skip_min;
                    }

                if( (featureSize = s_ptr[minPt.y*sstep + minPt.x]) >= 4 &&
                    !icvStarDetectorSuppressLines( responses, sizes, minPt, params ))
                {
                    CvStarKeypoint kpt = cvStarKeypoint( minPt, featureSize, minResponse );
                    cvSeqPush( keypoints, &kpt );
                }
            }
        skip_min:
            ;
        }
}

CV_IMPL CvSeq*
cvGetStarKeypoints( const CvArr* _img, CvMemStorage* storage,
                    CvStarDetectorParams params )
{
    CvMat stub, *img = cvGetMat(_img, &stub);
    CvSeq* keypoints = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvStarKeypoint), storage );
    CvMat* responses = cvCreateMat( img->rows, img->cols, CV_32FC1 );
    CvMat* sizes = cvCreateMat( img->rows, img->cols, CV_16SC1 );

    int border = icvStarDetectorComputeResponses( img, responses, sizes, &params );
    if( border >= 0 )
        icvStarDetectorSuppressNonmax( responses, sizes, keypoints, border, &params );

    cvReleaseMat( &responses );
    cvReleaseMat( &sizes );

    return border >= 0 ? keypoints : 0;
}

namespace cv
{

StarDetector::StarDetector()
{
    *(CvStarDetectorParams*)this = cvStarDetectorParams();
}

StarDetector::StarDetector(int _maxSize, int _responseThreshold,
                           int _lineThresholdProjected,
                           int _lineThresholdBinarized,
                           int _suppressNonmaxSize)
{
    *(CvStarDetectorParams*)this = cvStarDetectorParams(_maxSize, _responseThreshold,
            _lineThresholdProjected, _lineThresholdBinarized, _suppressNonmaxSize);
}

void StarDetector::operator()(const Mat& image, vector<KeyPoint>& keypoints) const
{
    CvMat _image = image;
    MemStorage storage(cvCreateMemStorage(0));
    Seq<CvStarKeypoint> kp = cvGetStarKeypoints( &_image, storage, *(const CvStarDetectorParams*)this);
    Seq<CvStarKeypoint>::iterator it = kp.begin();
    keypoints.resize(kp.size());
    size_t i, n = kp.size();
    for( i = 0; i < n; i++, ++it )
    {
        const CvStarKeypoint& kpt = *it;
        keypoints[i] = KeyPoint(kpt.pt, (float)kpt.size, -1.f, kpt.response, 0);
    }
}

}
