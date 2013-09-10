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

#include "test_precomp.hpp"

using namespace cv;

namespace cvtest
{
    class CV_BilateralFilterTest :
        public cvtest::BaseTest
    {
    public:
        enum
        {
            MAX_WIDTH = 1920, MIN_WIDTH = 1,
            MAX_HEIGHT = 1080, MIN_HEIGHT = 1
        };

        CV_BilateralFilterTest();
        ~CV_BilateralFilterTest();

    protected:
        virtual void run_func();
        virtual int prepare_test_case(int test_case_index);
        virtual int validate_test_results(int test_case_index);

    private:
        void reference_bilateral_filter(const Mat& src, Mat& dst, int d, double sigma_color,
            double sigma_space, int borderType = BORDER_DEFAULT);

        int getRandInt(RNG& rng, int min_value, int max_value) const;

        double _sigma_color;
        double _sigma_space;

        Mat _src;
        Mat _parallel_dst;
        int _d;
    };

    CV_BilateralFilterTest::CV_BilateralFilterTest() :
        cvtest::BaseTest(), _src(), _parallel_dst(), _d()
    {
        test_case_count = 1000;
    }

    CV_BilateralFilterTest::~CV_BilateralFilterTest()
    {
    }

    int CV_BilateralFilterTest::getRandInt(RNG& rng, int min_value, int max_value) const
    {
        double rand_value = rng.uniform(log((double)min_value), log((double)max_value + 1));
        return cvRound(exp((double)rand_value));
    }

    void CV_BilateralFilterTest::reference_bilateral_filter(const Mat &src, Mat &dst, int d,
        double sigma_color, double sigma_space, int borderType)
    {
        int cn = src.channels();
        int i, j, k, maxk, radius;
        double minValSrc = -1, maxValSrc = 1;
        const int kExpNumBinsPerChannel = 1 << 12;
        int kExpNumBins = 0;
        float lastExpVal = 1.f;
        float len, scale_index;
        Size size = src.size();

        dst.create(size, src.type());

        CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
            src.type() == dst.type() && src.size() == dst.size() &&
            src.data != dst.data );

        if( sigma_color <= 0 )
            sigma_color = 1;
        if( sigma_space <= 0 )
            sigma_space = 1;

        double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
        double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

        if( d <= 0 )
            radius = cvRound(sigma_space*1.5);
        else
            radius = d/2;
        radius = MAX(radius, 1);
        d = radius*2 + 1;
        // compute the min/max range for the input image (even if multichannel)

        minMaxLoc( src.reshape(1), &minValSrc, &maxValSrc );
        if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
        {
            src.copyTo(dst);
            return;
        }

        // temporary copy of the image with borders for easy processing
        Mat temp;
        copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );
        patchNaNs(temp);

        // allocate lookup tables
        vector<float> _space_weight(d*d);
        vector<int> _space_ofs(d*d);
        float* space_weight = &_space_weight[0];
        int* space_ofs = &_space_ofs[0];

        // assign a length which is slightly more than needed
        len = (float)(maxValSrc - minValSrc) * cn;
        kExpNumBins = kExpNumBinsPerChannel * cn;
        vector<float> _expLUT(kExpNumBins+2);
        float* expLUT = &_expLUT[0];

        scale_index = kExpNumBins/len;

        // initialize the exp LUT
        for( i = 0; i < kExpNumBins+2; i++ )
        {
            if( lastExpVal > 0.f )
            {
                double val =  i / scale_index;
                expLUT[i] = (float)std::exp(val * val * gauss_color_coeff);
                lastExpVal = expLUT[i];
            }
            else
                expLUT[i] = 0.f;
        }

        // initialize space-related bilateral filter coefficients
        for( i = -radius, maxk = 0; i <= radius; i++ )
            for( j = -radius; j <= radius; j++ )
            {
                double r = std::sqrt((double)i*i + (double)j*j);
                if( r > radius )
                    continue;
                space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
                space_ofs[maxk++] = (int)(i*(temp.step/sizeof(float)) + j*cn);
            }

        for( i = 0; i < size.height; i++ )
        {
            const float* sptr = (const float*)(temp.data + (i+radius)*temp.step) + radius*cn;
            float* dptr = (float*)(dst.data + i*dst.step);

            if( cn == 1 )
            {
                for( j = 0; j < size.width; j++ )
                {
                    float sum = 0, wsum = 0;
                    float val0 = sptr[j];
                    for( k = 0; k < maxk; k++ )
                    {
                        float val = sptr[j + space_ofs[k]];
                        float alpha = (float)(std::abs(val - val0)*scale_index);
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        float w = space_weight[k]*(expLUT[idx] + alpha*(expLUT[idx+1] - expLUT[idx]));
                        sum += val*w;
                        wsum += w;
                    }
                    dptr[j] = (float)(sum/wsum);
                }
            }
            else
            {
                assert( cn == 3 );
                for( j = 0; j < size.width*3; j += 3 )
                {
                    float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                    float b0 = sptr[j], g0 = sptr[j+1], r0 = sptr[j+2];
                    for( k = 0; k < maxk; k++ )
                    {
                        const float* sptr_k = sptr + j + space_ofs[k];
                        float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
                        float alpha = (float)((std::abs(b - b0) +
                            std::abs(g - g0) + std::abs(r - r0))*scale_index);
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        float w = space_weight[k]*(expLUT[idx] + alpha*(expLUT[idx+1] - expLUT[idx]));
                        sum_b += b*w; sum_g += g*w; sum_r += r*w;
                        wsum += w;
                    }
                    wsum = 1.f/wsum;
                    b0 = sum_b*wsum;
                    g0 = sum_g*wsum;
                    r0 = sum_r*wsum;
                    dptr[j] = b0; dptr[j+1] = g0; dptr[j+2] = r0;
                }
            }
        }
    }

    int CV_BilateralFilterTest::prepare_test_case(int /* test_case_index */)
    {
        const static int types[] = { CV_32FC1, CV_32FC3, CV_8UC1, CV_8UC3 };
        RNG& rng = ts->get_rng();
        Size size(getRandInt(rng, MIN_WIDTH, MAX_WIDTH), getRandInt(rng, MIN_HEIGHT, MAX_HEIGHT));
        int type = types[rng(sizeof(types) / sizeof(types[0]))];

        _d = rng.uniform(0., 1.) > 0.5 ? 5 : 3;

        _src.create(size, type);

        rng.fill(_src, RNG::UNIFORM, 0, 256);

        _sigma_color = _sigma_space = 1.;

        return 1;
    }

    int CV_BilateralFilterTest::validate_test_results(int test_case_index)
    {
        static const double eps = 4;

        Mat reference_dst, reference_src;
        if (_src.depth() == CV_32F)
            reference_bilateral_filter(_src, reference_dst, _d, _sigma_color, _sigma_space);
        else
        {
            int type = _src.type();
            _src.convertTo(reference_src, CV_32F);
            reference_bilateral_filter(reference_src, reference_dst, _d, _sigma_color, _sigma_space);
            reference_dst.convertTo(reference_dst, type);
        }

        double e = norm(reference_dst, _parallel_dst);
        if (e > eps)
        {
            ts->printf(cvtest::TS::CONSOLE, "actual error: %g, expected: %g", e, eps);
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }
        else
            ts->set_failed_test_info(cvtest::TS::OK);

        return BaseTest::validate_test_results(test_case_index);
    }

    void CV_BilateralFilterTest::run_func()
    {
        bilateralFilter(_src, _parallel_dst, _d, _sigma_color, _sigma_space);
    }

    TEST(Imgproc_BilateralFilter, accuracy)
    {
        CV_BilateralFilterTest test;
        test.safe_run();
    }

} // end of namespace cvtest
