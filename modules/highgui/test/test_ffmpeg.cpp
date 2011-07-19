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
                ts->printf(cvtest::TS::LOG, "start  reading bit image\n");
				Mat img = imread(string(ts->get_data_path()) + "readwrite/read.png");
                ts->printf(cvtest::TS::LOG, "finish reading bit image\n");
				if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
                ts->printf(cvtest::TS::LOG, "start  writing bit image\n");
				imwrite(string(ts->get_data_path()) + "readwrite/write.png", img);
                ts->printf(cvtest::TS::LOG, "finish writing bit image\n");
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
			const double time_sec = 2;
			const int coeff = static_cast<int>(static_cast<double>(cv::min(img_c, img_r)) / (fps * time_sec));

			Mat img(img_r, img_c, CV_8UC3, Scalar::all(0));
			try
			{
				VideoWriter writer(string(ts->get_data_path()) + "video/output.avi", CV_FOURCC('X', 'V', 'I', 'D'), fps, frame_s);

				if (writer.isOpened() == false) ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);

				for (int i = 0 ; i < static_cast<int>(fps * time_sec); i++ )
				{
					//circle(img, Point2i(img_c / 2, img_r / 2), cv::min(img_r, img_c) / 2 * (i + 1), Scalar(255, 0, 0, 0), 2);
					rectangle(img, Point2i(coeff * i, coeff * i), Point2i(coeff * (i + 1), coeff * (i + 1)), 
						Scalar::all(255 * (1.0 - static_cast<double>(i) / (fps * time_sec * 2) )), -1); 
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
	if (ext == 1) return ".bmp";
	if (ext == 2) return ".pgm";
	if (ext == 3) return ".tiff";
	return "";
}

class CV_FFmpegWriteSequenceImageTest : public cvtest::BaseTest
{
	public:
		void run(int)
		{
			try
			{
				const int img_r = 640;
				const int img_c = 480;
				Size frame_s = Size(img_c, img_r);

				for (int k = 1; k <= 5; ++k)
				{
					for (int ext = 0; ext < 4; ++ext) // 0 - png, 1 - bmp, 2 - pgm, 3 - tiff
					for (int num_channels = 1; num_channels <= 3; num_channels+=2)
					{
						ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_8U, num_channels, ext_from_int(ext).c_str());
						Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_8U, num_channels), Scalar::all(0));
						circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), cv::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));
						ts->printf(ts->LOG, "writing      image : %s\n", string(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext)).c_str());
						imwrite(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext), img);
						ts->printf(ts->LOG, "reading test image : %s\n", string(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext)).c_str());
						
						Mat img_test = imread(string(ts->get_data_path()) + "readwrite/test" + ext_from_int(ext), CV_LOAD_IMAGE_UNCHANGED);
						
						if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

						CV_Assert(img.size() == img_test.size());
						CV_Assert(img.type() == img_test.type());

						double n = norm(img, img_test);
						if ( n > 1.0)
						{
							ts->printf(ts->LOG, "norm = %f \n", n);
							ts->set_failed_test_info(ts->FAIL_MISMATCH);
						}
					}

					for (int num_channels = 1; num_channels <= 3; num_channels+=2)
					{
						// jpeg
						ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_8U, num_channels, ".jpg");
						Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_8U, num_channels), Scalar::all(0));
						circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), cv::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));
						string filename = string(ts->get_data_path() + "readwrite/test_" + char(k + 48) + "_c" + char(num_channels + 48) + "_.jpg");
						imwrite(filename, img);
						img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

						filename = string(ts->get_data_path() + "readwrite/test_" + char(k + 48) + "_c" + char(num_channels + 48) + ".jpg");
						ts->printf(ts->LOG, "reading test image : %s\n", filename.c_str());
						Mat img_test = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
						
						if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

						CV_Assert(img.size() == img_test.size());
						CV_Assert(img.type() == img_test.type());

						double n = norm(img, img_test);
						if ( n > 1.0)
						{
							ts->printf(ts->LOG, "norm = %f \n", n);
							ts->set_failed_test_info(ts->FAIL_MISMATCH);
						}
					}

					for (int num_channels = 1; num_channels <= 3; num_channels+=2)
					{
						// tiff
						ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_16U, num_channels, ".tiff");
						Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_16U, num_channels), Scalar::all(0));
						circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), cv::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));
						string filename = string(ts->get_data_path() + "readwrite/test.tiff");
						imwrite(filename, img);
						ts->printf(ts->LOG, "reading test image : %s\n", filename.c_str());
						Mat img_test = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
						
						if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

						CV_Assert(img.size() == img_test.size());

						ts->printf(ts->LOG, "img      : %d ; %d \n", img.channels(), img.depth());
						ts->printf(ts->LOG, "img_test : %d ; %d \n", img_test.channels(), img_test.depth());

						CV_Assert(img.type() == img_test.type());


						double n = norm(img, img_test);
						if ( n > 1.0)
						{
							ts->printf(ts->LOG, "norm = %f \n", n);
							ts->set_failed_test_info(ts->FAIL_MISMATCH);
						}
					}
				}
			}
			catch(const cv::Exception & e)
			{
				ts->printf(ts->LOG, "Exception: %s\n" , e.what());
				ts->set_failed_test_info(ts->FAIL_MISMATCH);
			}
		}
};

TEST(Highgui_FFmpeg_WriteBigImage,         regression) { CV_FFmpegWriteBigImageTest      test; test.safe_run(); }
TEST(Highgui_FFmpeg_WriteBigVideo,         regression) { CV_FFmpegWriteBigVideoTest      test; test.safe_run(); }
TEST(Highgui_FFmpeg_WriteSequenceImage,    regression) { CV_FFmpegWriteSequenceImageTest test; test.safe_run(); }
