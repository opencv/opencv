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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

template <typename inMatType, typename outMatType> static void
computeIntegralImages( const Mat& matI, Mat& matS, Mat& matT, Mat& _FT,
                       int iiType )
{
    int x, y, rows = matI.rows, cols = matI.cols;

    matS.create(rows + 1, cols + 1, iiType );
    matT.create(rows + 1, cols + 1, iiType );
    _FT.create(rows + 1, cols + 1, iiType );

    const inMatType* I = matI.ptr<inMatType>();

    outMatType *S = matS.ptr<outMatType>();
    outMatType *T = matT.ptr<outMatType>();
    outMatType *FT = _FT.ptr<outMatType>();

    int istep = (int)(matI.step/matI.elemSize());
    int step = (int)(matS.step/matS.elemSize());

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

template <typename iiMatType> static int
StarDetectorComputeResponses( const Mat& img, Mat& responses, Mat& sizes,
                              int maxSize, int iiType )
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
    union { int i; float f; } absmask;
    absmask.i = 0x7fffffff;
    volatile bool useSIMD = cv::checkHardwareSupport(CV_CPU_SSE2) && iiType == CV_32S;
#endif

    struct StarFeature
    {
        int area;
        iiMatType* p[8];
    };

    StarFeature f[MAX_PATTERN];

    Mat sum, tilted, flatTilted;
    int y, rows = img.rows, cols = img.cols;
    int border, npatterns=0, maxIdx=0;

    responses.create( img.size(), CV_32F );
    sizes.create( img.size(), CV_16S );

    while( pairs[npatterns][0] >= 0 && !
          ( sizes0[pairs[npatterns][0]] >= maxSize
           || sizes0[pairs[npatterns+1][0]] + sizes0[pairs[npatterns+1][0]]/2 >= std::min(rows, cols) ) )
    {
        ++npatterns;
    }

    npatterns += (pairs[npatterns-1][0] >= 0);
    maxIdx = pairs[npatterns-1][0];

    // Create the integral image appropriate for our type & usage
    if ( img.type() == CV_8U )
        computeIntegralImages<uchar, iiMatType>( img, sum, tilted, flatTilted, iiType );
    else if ( img.type() == CV_8S )
        computeIntegralImages<char, iiMatType>( img, sum, tilted, flatTilted, iiType );
    else if ( img.type() == CV_16U )
        computeIntegralImages<ushort, iiMatType>( img, sum, tilted, flatTilted, iiType );
    else if ( img.type() == CV_16S )
        computeIntegralImages<short, iiMatType>( img, sum, tilted, flatTilted, iiType );
    else
        CV_Error( Error::StsUnsupportedFormat, "" );

    int step = (int)(sum.step/sum.elemSize());

    for(int i = 0; i <= maxIdx; i++ )
    {
        int ur_size = sizes0[i], t_size = sizes0[i] + sizes0[i]/2;
        int ur_area = (2*ur_size + 1)*(2*ur_size + 1);
        int t_area = t_size*t_size + (t_size + 1)*(t_size + 1);

        f[i].p[0] = sum.ptr<iiMatType>() + (ur_size + 1)*step + ur_size + 1;
        f[i].p[1] = sum.ptr<iiMatType>() - ur_size*step + ur_size + 1;
        f[i].p[2] = sum.ptr<iiMatType>() + (ur_size + 1)*step - ur_size;
        f[i].p[3] = sum.ptr<iiMatType>() - ur_size*step - ur_size;

        f[i].p[4] = tilted.ptr<iiMatType>() + (t_size + 1)*step + 1;
        f[i].p[5] = flatTilted.ptr<iiMatType>() - t_size;
        f[i].p[6] = flatTilted.ptr<iiMatType>() + t_size + 1;
        f[i].p[7] = tilted.ptr<iiMatType>() - t_size*step + 1;

        f[i].area = ur_area + t_area;
        sizes1[i] = sizes0[i];
    }
    // negate end points of the size range
    // for a faster rejection of very small or very large features in non-maxima suppression.
    sizes1[0] = -sizes1[0];
    sizes1[1] = -sizes1[1];
    sizes1[maxIdx] = -sizes1[maxIdx];
    border = sizes0[maxIdx] + sizes0[maxIdx]/2;

    for(int i = 0; i < npatterns; i++ )
    {
        int innerArea = f[pairs[i][1]].area;
        int outerArea = f[pairs[i][0]].area - innerArea;
        invSizes[i][0] = 1.f/outerArea;
        invSizes[i][1] = 1.f/innerArea;
    }

#if CV_SSE2
    if( useSIMD )
    {
        for(int i = 0; i < npatterns; i++ )
        {
            _mm_store_ps((float*)&invSizes4[i][0], _mm_set1_ps(invSizes[i][0]));
            _mm_store_ps((float*)&invSizes4[i][1], _mm_set1_ps(invSizes[i][1]));
        }

        for(int i = 0; i <= maxIdx; i++ )
            _mm_store_ps((float*)&sizes1_4[i], _mm_set1_ps((float)sizes1[i]));
    }
#endif

    for( y = 0; y < border; y++ )
    {
        float* r_ptr = responses.ptr<float>(y);
        float* r_ptr2 = responses.ptr<float>(rows - 1 - y);
        short* s_ptr = sizes.ptr<short>(y);
        short* s_ptr2 = sizes.ptr<short>(rows - 1 - y);

        memset( r_ptr, 0, cols*sizeof(r_ptr[0]));
        memset( r_ptr2, 0, cols*sizeof(r_ptr2[0]));
        memset( s_ptr, 0, cols*sizeof(s_ptr[0]));
        memset( s_ptr2, 0, cols*sizeof(s_ptr2[0]));
    }

    for( y = border; y < rows - border; y++ )
    {
        int x = border;
        float* r_ptr = responses.ptr<float>(y);
        short* s_ptr = sizes.ptr<short>(y);

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

                for(int i = 0; i <= maxIdx; i++ )
                {
                    const iiMatType** p = (const iiMatType**)&f[i].p[0];
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

                for(int i = 0; i < npatterns; i++ )
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

            for(int i = 0; i <= maxIdx; i++ )
            {
                const iiMatType** p = (const iiMatType**)&f[i].p[0];
                vals[i] = (int)(p[0][ofs] - p[1][ofs] - p[2][ofs] + p[3][ofs] +
                    p[4][ofs] - p[5][ofs] - p[6][ofs] + p[7][ofs]);
            }
            for(int i = 0; i < npatterns; i++ )
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

    return border;
}


static bool StarDetectorSuppressLines( const Mat& responses, const Mat& sizes, Point pt,
                                       int lineThresholdProjected, int lineThresholdBinarized )
{
    const float* r_ptr = responses.ptr<float>();
    int rstep = (int)(responses.step/sizeof(r_ptr[0]));
    const short* s_ptr = sizes.ptr<short>();
    int sstep = (int)(sizes.step/sizeof(s_ptr[0]));
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

    if( (Lxx + Lyy)*(Lxx + Lyy) >= lineThresholdProjected*(Lxx*Lyy - Lxy*Lxy) )
        return true;

    for( y = pt.y - radius; y <= pt.y + radius; y += delta )
        for( x = pt.x - radius; x <= pt.x + radius; x += delta )
        {
            int Lxb = (s_ptr[y*sstep + x + 1] == sz) - (s_ptr[y*sstep + x - 1] == sz);
            int Lyb = (s_ptr[(y+1)*sstep + x] == sz) - (s_ptr[(y-1)*sstep + x] == sz);
            Lxxb += Lxb * Lxb; Lyyb += Lyb * Lyb; Lxyb += Lxb * Lyb;
        }

    if( (Lxxb + Lyyb)*(Lxxb + Lyyb) >= lineThresholdBinarized*(Lxxb*Lyyb - Lxyb*Lxyb) )
        return true;

    return false;
}


static void
StarDetectorSuppressNonmax( const Mat& responses, const Mat& sizes,
                            std::vector<KeyPoint>& keypoints, int border,
                            int responseThreshold,
                            int lineThresholdProjected,
                            int lineThresholdBinarized,
                            int suppressNonmaxSize )
{
    int x, y, x1, y1, delta = suppressNonmaxSize/2;
    int rows = responses.rows, cols = responses.cols;
    const float* r_ptr = responses.ptr<float>();
    int rstep = (int)(responses.step/sizeof(r_ptr[0]));
    const short* s_ptr = sizes.ptr<short>();
    int sstep = (int)(sizes.step/sizeof(s_ptr[0]));
    short featureSize = 0;

    for( y = border; y < rows - border; y += delta+1 )
        for( x = border; x < cols - border; x += delta+1 )
        {
            float maxResponse = (float)responseThreshold;
            float minResponse = (float)-responseThreshold;
            Point maxPt(-1, -1), minPt(-1, -1);
            int tileEndY = MIN(y + delta, rows - border - 1);
            int tileEndX = MIN(x + delta, cols - border - 1);

            for( y1 = y; y1 <= tileEndY; y1++ )
                for( x1 = x; x1 <= tileEndX; x1++ )
                {
                    float val = r_ptr[y1*rstep + x1];
                    if( maxResponse < val )
                    {
                        maxResponse = val;
                        maxPt = Point(x1, y1);
                    }
                    else if( minResponse > val )
                    {
                        minResponse = val;
                        minPt = Point(x1, y1);
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
                    !StarDetectorSuppressLines( responses, sizes, maxPt, lineThresholdProjected,
                                                lineThresholdBinarized ))
                {
                    KeyPoint kpt((float)maxPt.x, (float)maxPt.y, featureSize, -1, maxResponse);
                    keypoints.push_back(kpt);
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
                    !StarDetectorSuppressLines( responses, sizes, minPt,
                                               lineThresholdProjected, lineThresholdBinarized))
                {
                    KeyPoint kpt((float)minPt.x, (float)minPt.y, featureSize, -1, maxResponse);
                    keypoints.push_back(kpt);
                }
            }
        skip_min:
            ;
        }
}

StarDetector::StarDetector(int _maxSize, int _responseThreshold,
                           int _lineThresholdProjected,
                           int _lineThresholdBinarized,
                           int _suppressNonmaxSize)
: maxSize(_maxSize), responseThreshold(_responseThreshold),
    lineThresholdProjected(_lineThresholdProjected),
    lineThresholdBinarized(_lineThresholdBinarized),
    suppressNonmaxSize(_suppressNonmaxSize)
{}


void StarDetector::detectImpl( InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask ) const
{
    Mat image = _image.getMat(), mask = _mask.getMat(), grayImage = image;
    if( image.channels() > 1 ) cvtColor( image, grayImage, COLOR_BGR2GRAY );

    (*this)(grayImage, keypoints);
    KeyPointsFilter::runByPixelsMask( keypoints, mask );
}

void StarDetector::operator()(const Mat& img, std::vector<KeyPoint>& keypoints) const
{
    Mat responses, sizes;
    int border;

    // Use 32-bit integers if we won't overflow in the integral image
    if ((img.depth() == CV_8U || img.depth() == CV_8S) &&
        (img.rows * img.cols) < 8388608 ) // 8388608 = 2 ^ (32 - 8(bit depth) - 1(sign bit))
        border = StarDetectorComputeResponses<int>( img, responses, sizes, maxSize, CV_32S );
    else
        border = StarDetectorComputeResponses<double>( img, responses, sizes, maxSize, CV_64F );

    keypoints.clear();
    if( border >= 0 )
        StarDetectorSuppressNonmax( responses, sizes, keypoints, border,
                                    responseThreshold, lineThresholdProjected,
                                    lineThresholdBinarized, suppressNonmaxSize );
}

}
