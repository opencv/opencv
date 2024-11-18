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
#include "fast.hpp"
#include "fast_score.hpp"
#include "opencl_kernels_features.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

namespace cv
{

template<int patternSize>
void FAST_t(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    Mat img = _img.getMat();
    const int K = patternSize/2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize);

#if CV_SIMD128
    const int quarterPatternSize = patternSize/4;
    v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char)threshold), K16 = v_setall_u8((char)K);
#if CV_TRY_AVX2
    Ptr<opt_AVX2::FAST_t_patternSize16_AVX2> fast_t_impl_avx2;
    if(CV_CPU_HAS_SUPPORT_AVX2)
        fast_t_impl_avx2 = opt_AVX2::FAST_t_patternSize16_AVX2::getImpl(img.cols, threshold, nonmax_suppression, pixel);
#endif

#endif

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    uchar* buf[3] = { 0 };
    int* cpbuf[3] = { 0 };
    utils::BufferArea area;
    for (unsigned idx = 0; idx < 3; ++idx)
    {
        area.allocate(buf[idx], img.cols);
        area.allocate(cpbuf[idx], img.cols + 1);
    }
    area.commit();

    for (unsigned idx = 0; idx < 3; ++idx)
    {
        memset(buf[idx], 0, img.cols);
    }

    for(i = 3; i < img.rows-2; i++)
    {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3] + 1; // cornerpos[-1] is used to store a value
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
        {
            j = 3;
#if CV_SIMD128
            {
                if( patternSize == 16 )
                {
#if CV_TRY_AVX2
                    if (fast_t_impl_avx2)
                        fast_t_impl_avx2->process(j, ptr, curr, cornerpos, ncorners);
#endif
                    //vz if (j <= (img.cols - 27)) //it doesn't make sense using vectors for less than 8 elements
                    {
                        for (; j < img.cols - 16 - 3; j += 16, ptr += 16)
                        {
                            v_uint8x16 v = v_load(ptr);
                            v_int8x16 v0 = v_reinterpret_as_s8(v_xor(v_add(v, t), delta));
                            v_int8x16 v1 = v_reinterpret_as_s8(v_xor(v_sub(v, t), delta));

                            v_int8x16 x0 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[0]), delta));
                            v_int8x16 x1 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[quarterPatternSize]), delta));
                            v_int8x16 x2 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[2*quarterPatternSize]), delta));
                            v_int8x16 x3 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[3*quarterPatternSize]), delta));

                            v_int8x16 m0, m1;
                            m0 = v_and(v_lt(v0, x0), v_lt(v0, x1));
                            m1 = v_and(v_lt(x0, v1), v_lt(x1, v1));
                            m0 = v_or(m0, v_and(v_lt(v0, x1), v_lt(v0, x2)));
                            m1 = v_or(m1, v_and(v_lt(x1, v1), v_lt(x2, v1)));
                            m0 = v_or(m0, v_and(v_lt(v0, x2), v_lt(v0, x3)));
                            m1 = v_or(m1, v_and(v_lt(x2, v1), v_lt(x3, v1)));
                            m0 = v_or(m0, v_and(v_lt(v0, x3), v_lt(v0, x0)));
                            m1 = v_or(m1, v_and(v_lt(x3, v1), v_lt(x0, v1)));
                            m0 = v_or(m0, m1);

                            if( !v_check_any(m0) )
                                continue;
                            if( !v_check_any(v_combine_low(m0, m0)) )
                            {
                                j -= 8;
                                ptr -= 8;
                                continue;
                            }

                            v_int8x16 c0 = v_setzero_s8();
                            v_int8x16 c1 = v_setzero_s8();
                            v_uint8x16 max0 = v_setzero_u8();
                            v_uint8x16 max1 = v_setzero_u8();
                            for( k = 0; k < N; k++ )
                            {
                                v_int8x16 x = v_reinterpret_as_s8(v_xor(v_load((ptr + pixel[k])), delta));
                                m0 = v_lt(v0, x);
                                m1 = v_lt(x, v1);

                                c0 = v_and(v_sub_wrap(c0, m0), m0);
                                c1 = v_and(v_sub_wrap(c1, m1), m1);

                                max0 = v_max(max0, v_reinterpret_as_u8(c0));
                                max1 = v_max(max1, v_reinterpret_as_u8(c1));
                            }

                            max0 = v_lt(K16, v_max(max0, max1));
                            unsigned int m = v_signmask(v_reinterpret_as_s8(max0));

                            for( k = 0; m > 0 && k < 16; k++, m >>= 1 )
                            {
                                if( m & 1 )
                                {
                                    cornerpos[ncorners++] = j+k;
                                    if(nonmax_suppression)
                                    {
                                        short d[25];
                                        for (int _k = 0; _k < 25; _k++)
                                            d[_k] = (short)(ptr[k] - ptr[k + pixel[_k]]);

                                        v_int16x8 a0, b0, a1, b1;
                                        a0 = b0 = a1 = b1 = v_load(d + 8);
                                        for(int shift = 0; shift < 8; ++shift)
                                        {
                                            v_int16x8 v_nms = v_load(d + shift);
                                            a0 = v_min(a0, v_nms);
                                            b0 = v_max(b0, v_nms);
                                            v_nms = v_load(d + 9 + shift);
                                            a1 = v_min(a1, v_nms);
                                            b1 = v_max(b1, v_nms);
                                        }
                                        curr[j + k] = (uchar)(v_reduce_max(v_max(v_max(a0, a1), v_sub(v_setzero_s16(), v_min(b0, b1)))) - 1);
                                    }
                                }
                            }
                        }
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
        cornerpos = cpbuf[(i - 4 + 3)%3] + 1; // cornerpos[-1] is used to store a value
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

    ocl::Kernel fastKptKernel("FAST_findKeypoints", ocl::features::fast_oclsrc);
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

        ocl::Kernel fastNMSKernel("FAST_nonmaxSupression", ocl::features::fast_oclsrc);
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



static inline int hal_FAST(cv::Mat& src, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, FastFeatureDetector::DetectorType type)
{
    if (threshold > 20)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    cv::Mat scores(src.size(), src.type());

    int error = cv_hal_FAST_dense(src.data, src.step, scores.data, scores.step, src.cols, src.rows, type);

    if (error != CV_HAL_ERROR_OK)
        return error;

    cv::Mat suppressedScores(src.size(), src.type());

    if (nonmax_suppression)
    {
        error = cv_hal_FAST_NMS(scores.data, scores.step, suppressedScores.data, suppressedScores.step, scores.cols, scores.rows);

        if (error != CV_HAL_ERROR_OK)
            return error;
    }
    else
    {
        suppressedScores = scores;
    }

    if (!threshold && nonmax_suppression) threshold = 1;

    cv::KeyPoint kpt(0, 0, 7.f, -1, 0);

    unsigned uthreshold = (unsigned) threshold;

    int ofs = 3;

    int stride = (int)suppressedScores.step;
    const unsigned char* pscore = suppressedScores.data;

    keypoints.clear();

    for (int y = ofs; y + ofs < suppressedScores.rows; ++y)
    {
        kpt.pt.y = (float)(y);
        for (int x = ofs; x + ofs < suppressedScores.cols; ++x)
        {
            unsigned score = pscore[y * stride + x];
            if (score > uthreshold)
            {
                kpt.pt.x = (float)(x);
                kpt.response = (nonmax_suppression != 0) ? (float)((int)score - 1) : 0.f;
                keypoints.push_back(kpt);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

void FAST(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, FastFeatureDetector::DetectorType type)
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_img.isUMat() && type == FastFeatureDetector::TYPE_9_16,
               ocl_FAST(_img, keypoints, threshold, nonmax_suppression, 10000));

    cv::Mat img = _img.getMat();
    CALL_HAL(fast_dense, hal_FAST, img, keypoints, threshold, nonmax_suppression, type);

    size_t keypoints_count;
    CALL_HAL(fast, cv_hal_FAST, img.data, img.step, img.cols, img.rows,
             (uchar*)(keypoints.data()), &keypoints_count, threshold, nonmax_suppression, type);

    switch(type) {
    case FastFeatureDetector::TYPE_5_8:
        FAST_t<8>(_img, keypoints, threshold, nonmax_suppression);
        break;
    case FastFeatureDetector::TYPE_7_12:
        FAST_t<12>(_img, keypoints, threshold, nonmax_suppression);
        break;
    case FastFeatureDetector::TYPE_9_16:
        FAST_t<16>(_img, keypoints, threshold, nonmax_suppression);
        break;
    }
}


class FastFeatureDetector_Impl CV_FINAL : public FastFeatureDetector
{
public:
    FastFeatureDetector_Impl( int _threshold, bool _nonmaxSuppression, FastFeatureDetector::DetectorType _type )
    : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression), type(_type)
    {}

    void read( const FileNode& fn) CV_OVERRIDE
    {
      // if node is empty, keep previous value
      if (!fn["threshold"].empty())
        fn["threshold"] >> threshold;
      if (!fn["nonmaxSuppression"].empty())
        fn["nonmaxSuppression"] >> nonmaxSuppression;
      if (!fn["type"].empty())
        fn["type"] >> type;
    }
    void write( FileStorage& fs) const CV_OVERRIDE
    {
      if(fs.isOpened())
      {
        fs << "name" << getDefaultName();
        fs << "threshold" << threshold;
        fs << "nonmaxSuppression" << nonmaxSuppression;
        fs << "type" << type;
      }
    }

    void detect( InputArray _image, std::vector<KeyPoint>& keypoints, InputArray _mask ) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        if(_image.empty())
        {
            keypoints.clear();
            return;
        }

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
            type = static_cast<FastFeatureDetector::DetectorType>(cvRound(value));
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
            return static_cast<int>(type);
        CV_Error(Error::StsBadArg, "");
        return 0;
    }

    void setThreshold(int threshold_) CV_OVERRIDE { threshold = threshold_; }
    int getThreshold() const CV_OVERRIDE { return threshold; }

    void setNonmaxSuppression(bool f) CV_OVERRIDE { nonmaxSuppression = f; }
    bool getNonmaxSuppression() const CV_OVERRIDE { return nonmaxSuppression; }

    void setType(FastFeatureDetector::DetectorType type_) CV_OVERRIDE{ type = type_; }
    FastFeatureDetector::DetectorType getType() const CV_OVERRIDE{ return type; }

    int threshold;
    bool nonmaxSuppression;
    FastFeatureDetector::DetectorType type;
};

Ptr<FastFeatureDetector> FastFeatureDetector::create( int threshold, bool nonmaxSuppression, FastFeatureDetector::DetectorType type )
{
    return makePtr<FastFeatureDetector_Impl>(threshold, nonmaxSuppression, type);
}

String FastFeatureDetector::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".FastFeatureDetector");
}

}
