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
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

class CV_FFmpegWriteBigImageTest : public cvtest::BaseTest
{
	public:
		void run(int)
		{
			try
			{
				ts->printf(ts->LOG, "start  reading bit image\n");
				Mat img = imread(string(ts->get_data_path()) + "readwrite/read.png");
				ts->printf(ts->LOG, "finish reading bit image\n");
				if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
				ts->printf(ts->LOG, "start  writing bit image\n");
				imwrite(string(ts->get_data_path()) + "readwrite/write.png", img);
				ts->printf(ts->LOG, "finish writing bit image\n");
			}
			catch(...)
			{
				ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
			}
			ts->set_failed_test_info(cvtest::TS::OK);
		}
};

class CV_FFmpegWriteBigVideoTest : public cvtest::BaseTest
{
	public:
		void run(int)
		{
			const int img_r = 4096;
			const int img_c = 4096;
			Size frame_s = Size(img_c, img_r);
			const double fps = 30;
			const double time_sec = 1;

			Mat img(img_r, img_c, CV_8UC3, Scalar::all(0));
			try
			{
				VideoWriter writer(string(ts->get_data_path()) + "video/output.avi",
					CV_FOURCC('X', 'V', 'I', 'D'), fps, frame_s);

				if (writer.isOpened() == false) ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);

				for (int i = 0 ; i < static_cast<int>(fps * time_sec); i++ )
				{
					circle(img, Point2i(img_c / 2, img_r / 2), cv::min(img_r, img_c) / 2 * (i + 1), Scalar::all(255));
					writer << img;
				}
			}
			catch(...)
			{
				ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
			}
			ts->set_failed_test_info(cvtest::TS::OK);
		}
};

string ext_from_int(int ext)
{
	if (ext == 0) return ".png";
	if (ext == 1) return ".jpg";
	if (ext == 2) return ".bmp";
	if (ext == 3) return ".pgm";
	if (ext == 4) return ".tiff";
	return "";
}

//class CV_FFmpegWriteSequenceImageTest : public cvtest::BaseTest
//{
//	public:
//		void run(int)
//		{
//			try
//			{
//				const int img_r = 640;
//				const int img_c = 480;
//				Size frame_s = Size(img_c, img_r);
//
//				for (size_t ext = 0; ext < 5; ++ext) // 0 - png, 1 - jpg, 2 - bmp, 3 - pgm, 4 - tiff
//				for (size_t k = 1; k <= 3; ++k)
//					for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
//						for (size_t depth = CV_8U; depth <= CV_16U; ++depth)
//						{
//							ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", depth, num_channels, ext_from_int(ext).c_str());
//							ts->printf(ts->LOG, "creating image\n");
//							Mat img(img_r * k, img_c * k, CV_MAKETYPE(depth, num_channels), Scalar::all(0));
//							ts->printf(ts->LOG, "drawing circle\n");
//							circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), cv::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));
//							ts->printf(ts->LOG, "writing image : %s\n", string(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext)).c_str());
//							imwrite(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext), img);
//							ts->printf(ts->LOG, "reading test image : %s\n", string(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext)).c_str());
//							Mat img_test = imread(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext));
//							
//							CV_Assert(img.size() == img_test.size());
//							CV_Assert(img.type() == img_test.type());
//
//							ts->printf(ts->LOG, "checking test image\n");
//							if (countNonZero(img != img_test) != 0)
//								ts->set_failed_test_info(cvtest::TS::FAIL_GENERIC);
//						}
//			}
//			catch(...)
//			{
//				ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
//			}
//			ts->set_failed_test_info(cvtest::TS::OK);
//		}
//};


//TEST(Highgui_FFmpeg_WriteBigImage,         regression) { CV_FFmpegWriteBigImageTest      test; test.safe_run(); }
//TEST(Highgui_FFmpeg_WriteBigVideo,         regression) { CV_FFmpegWriteBigVideoTest      test; test.safe_run(); }
