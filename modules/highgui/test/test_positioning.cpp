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
#include <stdio.h>

using namespace cv;
using namespace std;

enum NAVIGATION_METHOD {PROGRESSIVE, RANDOM};

class CV_VideoPositioningTest: public cvtest::BaseTest
{
public:
	CV_VideoPositioningTest();
	~CV_VideoPositioningTest();
	virtual void run(int) = 0;

protected:
	vector <int> idx;
	void run_test(int method);

private:
	void generate_idx_seq(CvCapture *cap, int method);
};

class CV_VideoProgressivePositioningTest: public CV_VideoPositioningTest
{
public:
	CV_VideoProgressivePositioningTest() : CV_VideoPositioningTest() {};
	~CV_VideoProgressivePositioningTest();
	void run(int);
};

class CV_VideoRandomPositioningTest: public CV_VideoPositioningTest
{
public:
	CV_VideoRandomPositioningTest(): CV_VideoPositioningTest() {};
	~CV_VideoRandomPositioningTest();
	void run(int);
};

CV_VideoPositioningTest::CV_VideoPositioningTest() {}
CV_VideoPositioningTest::~CV_VideoPositioningTest() {}
CV_VideoProgressivePositioningTest::~CV_VideoProgressivePositioningTest() {}
CV_VideoRandomPositioningTest::~CV_VideoRandomPositioningTest() {}

void CV_VideoPositioningTest::generate_idx_seq(CvCapture* cap, int method)
{
	idx.clear();
    int N = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
	switch(method)
	{
	case PROGRESSIVE:
		{
			int pos = 1, step = 20;
			do
			{
				idx.push_back(pos);
				pos += step;
			}
			while (pos <= N);
			break;
		}
	case RANDOM:
		{
			RNG rng(N);
			idx.clear();
			for( int i = 0; i < N-1; i++ )
				idx.push_back(rng.uniform(0, N));
            idx.push_back(N-1);
			std::swap(idx.at(rng.uniform(0, N-1)), idx.at(N-1));
			break;
		}
	default:break;
	}
}

void CV_VideoPositioningTest::run_test(int method)
{
	const string& src_dir = ts->get_data_path(); 

    ts->printf(cvtest::TS::LOG, "\n\nSource files directory: %s\n", (src_dir+"video/").c_str());

    const string ext[] = {"avi", "mp4", "wmv"};

    size_t n = sizeof(ext)/sizeof(ext[0]);

    int failed_videos = 0;

	for (size_t i = 0; i < n; ++i)
	{
        string file_path = src_dir + "video/big_buck_bunny." + ext[i];

        printf("\nReading video file in %s...\n", file_path.c_str());

		CvCapture* cap = cvCreateFileCapture(file_path.c_str());

		if (!cap)
		{
			ts->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\nFAILED\n\n", i+1, ext[i].c_str());
            ts->printf(cvtest::TS::LOG, "Error: cannot read source video file.\n");
			ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            failed_videos++; continue;
		}

		cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, 0);

		generate_idx_seq(cap, method);

        int N = (int)idx.size(), failed_frames = 0, failed_positions = 0, failed_iterations = 0;

        for (int j = 0; j < N; ++j)
		{
            bool flag = false;

            cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, idx.at(j));

            /* IplImage* frame = cvRetrieveFrame(cap);

            if (!frame)
			{
                if (!failed_frames)
                {
                    ts->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\n", i+1, ext[i].c_str());
                }
                failed_frames++;
                ts->printf(cvtest::TS::LOG, "\nIteration: %d\n\nError: cannot read a frame with index %d.\n", j, idx.at(j));
                ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
                flag = !flag;
            } */

            int val = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES);

            if (idx.at(j) != val)
			{
                if (!(failed_frames||failed_positions))
                {
                    ts->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\n", i+1, ext[i].c_str());
                }
                failed_positions++;
                if (!failed_frames)
                {
                    ts->printf(cvtest::TS::LOG, "\nIteration: %d\n", j);
                }
                ts->printf(cvtest::TS::LOG, "Required pos: %d\nReturned pos: %d\n", idx.at(j), val);
                ts->printf(cvtest::TS::LOG, "Error: required and returned positions are not matched.\n");
				ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                if (!flag) flag = !flag;
			}

            if (flag) failed_iterations++;
		}

        ts->printf(cvtest::TS::LOG, "\nSuccessfull iterations: %d (%d%%)\n", idx.size()-failed_iterations, 100*(idx.size()-failed_iterations)/idx.size());
        ts->printf(cvtest::TS::LOG, "Failed iterations: %d (%d%%)\n", failed_iterations, 100*failed_iterations/idx.size());

        if (failed_frames||failed_positions)
        {
            ts->printf(cvtest::TS::LOG, "\nFAILED\n----------\n"); failed_videos++;
        }

		cvReleaseCapture(&cap);
	}

    ts->printf(cvtest::TS::LOG, "\nSuccessfull experiments: %d (%d%%)\n", n-failed_videos, 100*(n-failed_videos)/n);
    ts->printf(cvtest::TS::LOG, "Failed experiments: %d (%d%%)\n", failed_videos, 100*failed_videos/n);
}

void CV_VideoProgressivePositioningTest::run(int) 
{
	run_test(PROGRESSIVE);
}

void CV_VideoRandomPositioningTest::run(int)
{
	run_test(RANDOM);
}

#if BUILD_WITH_VIDEO_INPUT_SUPPORT
TEST (HighguiPositioning, progressive) { CV_VideoProgressivePositioningTest test; test.safe_run(); }
TEST (HighguiPositioning, random) { CV_VideoRandomPositioningTest test; test.safe_run(); }
#endif