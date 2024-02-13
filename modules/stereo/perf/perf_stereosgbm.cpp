/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;
using namespace testing;

static void MakeArtificialExample(Mat& dst_left_view, Mat& dst_view);

CV_ENUM(SGBMModes, StereoSGBM::MODE_SGBM, StereoSGBM::MODE_SGBM_3WAY, StereoSGBM::MODE_HH4)
typedef tuple<Size, int, SGBMModes> SGBMParams;
typedef TestBaseWithParam<SGBMParams> TestStereoCorrespSGBM;

#ifndef _DEBUG
PERF_TEST_P( TestStereoCorrespSGBM, SGBM, Combine(Values(Size(1280,720),Size(640,480)), Values(256,128), SGBMModes::all()) )
#else
PERF_TEST_P( TestStereoCorrespSGBM, DISABLED_TooLongInDebug_SGBM, Combine(Values(Size(1280,720),Size(640,480)), Values(256,128), SGBMModes::all()) )
#endif
{
    SGBMParams params = GetParam();

    Size sz              = get<0>(params);
    int num_disparities  = get<1>(params);
    int mode             = get<2>(params);

    Mat src_left(sz, CV_8UC3);
    Mat src_right(sz, CV_8UC3);
    Mat dst(sz, CV_16S);

    MakeArtificialExample(src_left,src_right);

    int wsize = 3;
    int P1 = 8*src_left.channels()*wsize*wsize;
    TEST_CYCLE()
    {
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,num_disparities,wsize,P1,4*P1,1,63,25,0,0,mode);
        sgbm->compute(src_left,src_right,dst);
    }

    SANITY_CHECK(dst, .01, ERROR_RELATIVE);
}

typedef tuple<Size, int> BMParams;
typedef TestBaseWithParam<BMParams> TestStereoCorrespBM;

PERF_TEST_P(TestStereoCorrespBM, BM, Combine(Values(Size(1280, 720), Size(640, 480)), Values(256, 128)))
{
    BMParams params = GetParam();
    Size sz = get<0>(params);
    int num_disparities = get<1>(params);

    Mat src_left(sz, CV_8UC1);
    Mat src_right(sz, CV_8UC1);
    Mat dst(sz, CV_16S);

    MakeArtificialExample(src_left, src_right);

    int wsize = 21;
    TEST_CYCLE()
    {
        Ptr<StereoBM> bm = StereoBM::create(num_disparities, wsize);
        bm->compute(src_left, src_right, dst);
    }

    SANITY_CHECK(dst, .01, ERROR_RELATIVE);
}

void MakeArtificialExample(Mat& dst_left_view, Mat& dst_right_view)
{
    RNG rng(0);
    int w = dst_left_view.cols;
    int h = dst_left_view.rows;

    //params:
    unsigned char bg_level = (unsigned char)rng.uniform(0.0,255.0);
    unsigned char fg_level = (unsigned char)rng.uniform(0.0,255.0);
    int rect_width  = (int)rng.uniform(w/16,w/2);
    int rect_height = (int)rng.uniform(h/16,h/2);
    int rect_disparity = (int)(0.15*w);
    double sigma = 3.0;

    int rect_x_offset = (w-rect_width) /2;
    int rect_y_offset = (h-rect_height)/2;

    if(dst_left_view.channels()==3)
    {
        dst_left_view  = Scalar(Vec3b(bg_level,bg_level,bg_level));
        dst_right_view = Scalar(Vec3b(bg_level,bg_level,bg_level));
    }
    else
    {
        dst_left_view  = Scalar(bg_level);
        dst_right_view = Scalar(bg_level);
    }

    Mat dst_left_view_rect = Mat(dst_left_view, Rect(rect_x_offset,rect_y_offset,rect_width,rect_height));
    if(dst_left_view.channels()==3)
        dst_left_view_rect = Scalar(Vec3b(fg_level,fg_level,fg_level));
    else
        dst_left_view_rect = Scalar(fg_level);

    rect_x_offset-=rect_disparity;

    Mat dst_right_view_rect = Mat(dst_right_view, Rect(rect_x_offset,rect_y_offset,rect_width,rect_height));
    if(dst_right_view.channels()==3)
        dst_right_view_rect = Scalar(Vec3b(fg_level,fg_level,fg_level));
    else
        dst_right_view_rect = Scalar(fg_level);

    //add some gaussian noise:
    unsigned char *l, *r;
    for(int i=0;i<h;i++)
    {
        l = dst_left_view.ptr(i);
        r = dst_right_view.ptr(i);

        if(dst_left_view.channels()==3)
        {
            for(int j=0;j<w;j++)
            {
                l[0] = saturate_cast<unsigned char>(l[0] + rng.gaussian(sigma));
                l[1] = saturate_cast<unsigned char>(l[1] + rng.gaussian(sigma));
                l[2] = saturate_cast<unsigned char>(l[2] + rng.gaussian(sigma));
                l+=3;

                r[0] = saturate_cast<unsigned char>(r[0] + rng.gaussian(sigma));
                r[1] = saturate_cast<unsigned char>(r[1] + rng.gaussian(sigma));
                r[2] = saturate_cast<unsigned char>(r[2] + rng.gaussian(sigma));
                r+=3;
            }
        }
        else
        {
            for(int j=0;j<w;j++)
            {
                l[0] = saturate_cast<unsigned char>(l[0] + rng.gaussian(sigma));
                l++;

                r[0] = saturate_cast<unsigned char>(r[0] + rng.gaussian(sigma));
                r++;
            }
        }
    }
}

}
