/* This is FAST corner detector, contributed to OpenCV by the author, Edward Rosten.
   Below is the original copyright and the references */

/*
Copyright (c) 2006, 2008 Edward Rosten
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

    *Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

    *Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

    *Neither the name of the University of Cambridge nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
The references are:
 * Machine learning for high-speed corner detection,
   E. Rosten and T. Drummond, ECCV 2006
 * Faster and better: A machine learning approach to corner detection
   E. Rosten, R. Porter and T. Drummond, PAMI, 2009
*/

#include "precomp.hpp"
#include "fast_score.hpp"
#include "opencl_kernels_features2d.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"
#if defined _MSC_VER
# pragma warning( disable : 4127)
#endif

namespace cv
{

template<int patternSize>
void FAST_t(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    Mat img = _img.getMat();
    const int K = patternSize/2, N = patternSize + K + 1;
#if CV_SSE2
    const int quarterPatternSize = patternSize/4;
    (void)quarterPatternSize;
#endif
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

#if CV_SSE2
    __m128i delta = _mm_set1_epi8(-128), t = _mm_set1_epi8((char)threshold), K16 = _mm_set1_epi8((char)K);
    (void)K16;
    (void)delta;
    (void)t;
#endif
    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128);
    uchar* buf[3];
    buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);

    for(i = 3; i < img.rows-2; i++)
    {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
        {
            j = 3;
    #if CV_SSE2
            if( patternSize == 16 )
            {
                for(; j < img.cols - 16 - 3; j += 16, ptr += 16)
                {
                    __m128i m0, m1;
                    __m128i v0 = _mm_loadu_si128((const __m128i*)ptr);
                    __m128i v1 = _mm_xor_si128(_mm_subs_epu8(v0, t), delta);
                    v0 = _mm_xor_si128(_mm_adds_epu8(v0, t), delta);

                    __m128i x0 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[0])), delta);
                    __m128i x1 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[quarterPatternSize])), delta);
                    __m128i x2 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[2*quarterPatternSize])), delta);
                    __m128i x3 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[3*quarterPatternSize])), delta);
                    m0 = _mm_and_si128(_mm_cmpgt_epi8(x0, v0), _mm_cmpgt_epi8(x1, v0));
                    m1 = _mm_and_si128(_mm_cmpgt_epi8(v1, x0), _mm_cmpgt_epi8(v1, x1));
                    m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x1, v0), _mm_cmpgt_epi8(x2, v0)));
                    m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x1), _mm_cmpgt_epi8(v1, x2)));
                    m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x2, v0), _mm_cmpgt_epi8(x3, v0)));
                    m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x2), _mm_cmpgt_epi8(v1, x3)));
                    m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x3, v0), _mm_cmpgt_epi8(x0, v0)));
                    m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x3), _mm_cmpgt_epi8(v1, x0)));
                    m0 = _mm_or_si128(m0, m1);
                    int mask = _mm_movemask_epi8(m0);
                    if( mask == 0 )
                        continue;
                    if( (mask & 255) == 0 )
                    {
                        j -= 8;
                        ptr -= 8;
                        continue;
                    }

                    __m128i c0 = _mm_setzero_si128(), c1 = c0, max0 = c0, max1 = c0;
                    for( k = 0; k < N; k++ )
                    {
                        __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(ptr + pixel[k])), delta);
                        m0 = _mm_cmpgt_epi8(x, v0);
                        m1 = _mm_cmpgt_epi8(v1, x);

                        c0 = _mm_and_si128(_mm_sub_epi8(c0, m0), m0);
                        c1 = _mm_and_si128(_mm_sub_epi8(c1, m1), m1);

                        max0 = _mm_max_epu8(max0, c0);
                        max1 = _mm_max_epu8(max1, c1);
                    }

                    max0 = _mm_max_epu8(max0, max1);
                    int m = _mm_movemask_epi8(_mm_cmpgt_epi8(max0, K16));

                    for( k = 0; m > 0 && k < 16; k++, m >>= 1 )
                        if(m & 1)
                        {
                            cornerpos[ncorners++] = j+k;
                            if(nonmax_suppression)
                                curr[j+k] = (uchar)cornerScore<patternSize>(ptr+k, pixel, threshold);
                        }
                }
            }
    #endif
            for( ; j < img.cols - 3; j++, ptr++ )
            {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
               (score > prev[j+1] && score > prev[j-1] &&
                score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.push_back(KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
}

#ifdef HAVE_OPENCL
template<typename pt>
struct cmp_pt
{
    bool operator ()(const pt& a, const pt& b) const { return a.y < b.y || (a.y == b.y && a.x < b.x); }
};

static bool ocl_FAST( InputArray _img, std::vector<KeyPoint>& keypoints,
                     int threshold, bool nonmax_suppression, int maxKeypoints )
{
    UMat img = _img.getUMat();
    if( img.cols < 7 || img.rows < 7 )
        return false;
    size_t globalsize[] = { (size_t)img.cols-6, (size_t)img.rows-6 };

    ocl::Kernel fastKptKernel("FAST_findKeypoints", ocl::features2d::fast_oclsrc);
    if (fastKptKernel.empty())
        return false;

    UMat kp1(1, maxKeypoints*2+1, CV_32S);

    UMat ucounter1(kp1, Rect(0,0,1,1));
    ucounter1.setTo(Scalar::all(0));

    if( !fastKptKernel.args(ocl::KernelArg::ReadOnly(img),
                            ocl::KernelArg::PtrReadWrite(kp1),
                            maxKeypoints, threshold).run(2, globalsize, 0, true))
        return false;

    Mat mcounter;
    ucounter1.copyTo(mcounter);
    int i, counter = mcounter.at<int>(0);
    counter = std::min(counter, maxKeypoints);

    keypoints.clear();

    if( counter == 0 )
        return true;

    if( !nonmax_suppression )
    {
        Mat m;
        kp1(Rect(0, 0, counter*2+1, 1)).copyTo(m);
        const Point* pt = (const Point*)(m.ptr<int>() + 1);
        for( i = 0; i < counter; i++ )
            keypoints.push_back(KeyPoint((float)pt[i].x, (float)pt[i].y, 7.f, -1, 1.f));
    }
    else
    {
        UMat kp2(1, maxKeypoints*3+1, CV_32S);
        UMat ucounter2 = kp2(Rect(0,0,1,1));
        ucounter2.setTo(Scalar::all(0));

        ocl::Kernel fastNMSKernel("FAST_nonmaxSupression", ocl::features2d::fast_oclsrc);
        if (fastNMSKernel.empty())
            return false;

        size_t globalsize_nms[] = { (size_t)counter };
        if( !fastNMSKernel.args(ocl::KernelArg::PtrReadOnly(kp1),
                                ocl::KernelArg::PtrReadWrite(kp2),
                                ocl::KernelArg::ReadOnly(img),
                                counter, counter).run(1, globalsize_nms, 0, true))
            return false;

        Mat m2;
        kp2(Rect(0, 0, counter*3+1, 1)).copyTo(m2);
        Point3i* pt2 = (Point3i*)(m2.ptr<int>() + 1);
        int newcounter = std::min(m2.at<int>(0), counter);

        std::sort(pt2, pt2 + newcounter, cmp_pt<Point3i>());

        for( i = 0; i < newcounter; i++ )
            keypoints.push_back(KeyPoint((float)pt2[i].x, (float)pt2[i].y, 7.f, -1, (float)pt2[i].z));
    }

    return true;
}
#endif


#ifdef HAVE_OPENVX
static bool openvx_FAST(InputArray _img, std::vector<KeyPoint>& keypoints,
                        int _threshold, bool nonmaxSuppression, int type)
{
    using namespace ivx;

    // Nonmax suppression is done differently in OpenCV than in OpenVX
    // 9/16 is the only supported mode in OpenVX
    if(nonmaxSuppression || type != FastFeatureDetector::TYPE_9_16)
        return false;

    Mat imgMat = _img.getMat();
    if(imgMat.empty() || imgMat.type() != CV_8UC1)
        return false;

    try
    {
        Context context = Context::create();
        Image img = Image::createFromHandle(context, Image::matTypeToFormat(imgMat.type()),
                                            Image::createAddressing(imgMat), (void*)imgMat.data);
        ivx::Scalar threshold = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, _threshold);
        vx_size capacity = imgMat.cols * imgMat.rows;
        Array corners = Array::create(context, VX_TYPE_KEYPOINT, capacity);

        ivx::Scalar numCorners = ivx::Scalar::create<VX_TYPE_SIZE>(context, 0);

        IVX_CHECK_STATUS(vxuFastCorners(context, img, threshold, (vx_bool)nonmaxSuppression, corners, numCorners));

        size_t nPoints = numCorners.getValue<vx_size>();
        keypoints.clear(); keypoints.reserve(nPoints);
        std::vector<vx_keypoint_t> vxCorners;
        corners.copyTo(vxCorners);
        for(size_t i = 0; i < nPoints; i++)
        {
            vx_keypoint_t kp = vxCorners[i];
            //if nonmaxSuppression is false, kp.strength is undefined
            keypoints.push_back(KeyPoint((float)kp.x, (float)kp.y, 7.f, -1, kp.strength));
        }

#ifdef VX_VERSION_1_1
        //we should take user memory back before release
        //(it's not done automatically according to standard)
        img.swapHandle();
#endif
    }
    catch (RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}

#endif


void FAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, int type)
{
    CV_INSTRUMENT_REGION()

#ifdef HAVE_OPENCL
  if( ocl::useOpenCL() && _img.isUMat() && type == FastFeatureDetector::TYPE_9_16 &&
      ocl_FAST(_img, keypoints, threshold, nonmax_suppression, 10000))
  {
    CV_IMPL_ADD(CV_IMPL_OCL);
    return;
  }
#endif

    CV_OVX_RUN(true,
               openvx_FAST(_img, keypoints, threshold, nonmax_suppression, type))

  switch(type) {
    case FastFeatureDetector::TYPE_5_8:
      FAST_t<8>(_img, keypoints, threshold, nonmax_suppression);
      break;
    case FastFeatureDetector::TYPE_7_12:
      FAST_t<12>(_img, keypoints, threshold, nonmax_suppression);
      break;
    case FastFeatureDetector::TYPE_9_16:
#ifdef HAVE_TEGRA_OPTIMIZATION
      if(tegra::useTegra() && tegra::FAST(_img, keypoints, threshold, nonmax_suppression))
        break;
#endif
      FAST_t<16>(_img, keypoints, threshold, nonmax_suppression);
      break;
  }
}


void FAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    CV_INSTRUMENT_REGION()

    FAST(_img, keypoints, threshold, nonmax_suppression, FastFeatureDetector::TYPE_9_16);
}


class FastFeatureDetector_Impl : public FastFeatureDetector
{
public:
    FastFeatureDetector_Impl( int _threshold, bool _nonmaxSuppression, int _type )
    : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression), type((short)_type)
    {}

    void detect( InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask )
    {
        CV_INSTRUMENT_REGION()

        Mat mask = _mask.getMat(), grayImage;
        UMat ugrayImage;
        _InputArray gray = _image;
        if( _image.type() != CV_8U )
        {
            _OutputArray ogray = _image.isUMat() ? _OutputArray(ugrayImage) : _OutputArray(grayImage);
            cvtColor( _image, ogray, COLOR_BGR2GRAY );
            gray = ogray;
        }
        FAST( gray, keypoints, threshold, nonmaxSuppression, type );
        KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    void set(int prop, double value)
    {
        if(prop == THRESHOLD)
            threshold = cvRound(value);
        else if(prop == NONMAX_SUPPRESSION)
            nonmaxSuppression = value != 0;
        else if(prop == FAST_N)
            type = cvRound(value);
        else
            CV_Error(Error::StsBadArg, "");
    }

    double get(int prop) const
    {
        if(prop == THRESHOLD)
            return threshold;
        if(prop == NONMAX_SUPPRESSION)
            return nonmaxSuppression;
        if(prop == FAST_N)
            return type;
        CV_Error(Error::StsBadArg, "");
        return 0;
    }

    void setThreshold(int threshold_) { threshold = threshold_; }
    int getThreshold() const { return threshold; }

    void setNonmaxSuppression(bool f) { nonmaxSuppression = f; }
    bool getNonmaxSuppression() const { return nonmaxSuppression; }

    void setType(int type_) { type = type_; }
    int getType() const { return type; }

    int threshold;
    bool nonmaxSuppression;
    int type;
};

Ptr<FastFeatureDetector> FastFeatureDetector::create( int threshold, bool nonmaxSuppression, int type )
{
    return makePtr<FastFeatureDetector_Impl>(threshold, nonmaxSuppression, type);
}


}
