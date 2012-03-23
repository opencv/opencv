/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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
#include "opencv2/highgui/highgui.hpp"

#ifdef HAVE_FFMPEG

#include "ffmpeg_codecs.hpp"

using namespace cv;
using namespace std;

class CV_PositioningTest : public cvtest::BaseTest
{
public:
    void CreateTestVideo(const string& format, int codec);
    void run(int);
};

void CV_PositioningTest::CreateTestVideo(const string& format, int codec)
{
 const size_t frame_count = 2500;

 stringstream s; s << codec;

 cv::VideoWriter writer(ts->get_data_path()+"video/test_video_"+s.str()+format, codec, 25, cv::Size(640, 480), false);

 for (size_t i = 0; i < frame_count; ++i)
 {
   cv::Mat mat(480, 640, CV_8UC1);
   size_t n = 32, tmp = i;
   vector<char> tmp_code; tmp_code.clear();

    while ( tmp > 1 )
    {
        tmp_code.push_back(tmp%2);
        tmp /= 2;
    }
    tmp_code.push_back(1);

    vector<char> i_code(n);
    for (size_t j = 0; j < n; ++j)
    {
    char val = j < n - tmp_code.size() ? 0 : tmp_code.at(j+tmp_code.size()-n);
    i_code.push_back(val);
    }

    const size_t w = 480/n;

    for (size_t j = 0; j < n; ++j)
    {
        for (size_t k = w*j; k < w*(j+1); ++k)
        mat.row(k) = i_code[j] ? 255*cv::Mat::ones(1, 640, CV_8UC1) : cv::Mat::zeros(1, 640, CV_8UC1);
    }

    writer << mat;

    //imshow("test image", mat); waitKey();

 }

 writer.~VideoWriter();
}

void CV_PositioningTest::run(int)
{
    const size_t n_codec = sizeof(codec_bmp_tags)/sizeof(codec_bmp_tags[0]);

    const string format[] =  {"avi", "mov", "mp4", "mpg", "wmv"};
    const size_t n_format = sizeof(format)/sizeof(format[0]);

    for (size_t i = 0; i < n_format; ++i)
    for (size_t j = 0; j < n_codec; ++j)
    {
     CreateTestVideo(format[i], codec_bmp_tags[j].tag);

     stringstream s; s << codec_bmp_tags[j].tag;
	
    cv::VideoCapture cap(ts->get_data_path()+"video/test_video_"+s.str()+format[i]);
	cap.set(CV_CAP_PROP_POS_FRAMES, 0.0);

	int N = cap.get(CV_CAP_PROP_FRAME_COUNT);

	vector <int> idx;

	RNG rng(N);
	idx.clear();
	for( int i = 0; i < N-1; i++ )
	idx.push_back(rng.uniform(0, N));
    idx.push_back(N-1);
	swap(idx.at(rng.uniform(0, N-1)), idx.at(N-1));

    for (int i = 0; i < N; ++i)
	{
		cap.set(CV_CAP_PROP_POS_FRAMES, (double)idx[i]);

		cv::Mat img;  cap.retrieve(img);

		const double thresh = 128.0;

		const size_t n = 32, w = img.rows/n;

		int index = 0, deg = n-1;

		for (size_t j = 0; j < n; ++j)
		{
			cv::Mat mat = img.rowRange(w*j, w*(j+1)-1);
			Scalar mat_mean = cv::mean(mat);
			if (mat_mean[0] > thresh)
			{
				index += (2<<deg);
			}

			deg--;
		}

		if (index != idx[i])
		{
            ts->printf(ts->LOG, "Required position: %d   Returned position: %d\n   FAILED", idx[i], index);
		}
	}
    }
}

TEST(Highgui_Positioning, regression) { CV_PositioningTest test; test.safe_run(); }

#endif
