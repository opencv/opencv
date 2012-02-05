#include "test_precomp.hpp"
#include "opencv2/highgui/highgui.hpp"

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

	ts->printf(cvtest::TS::LOG, "\n\nSource files directory: %s\n", (src_dir+"../perf/video/").c_str());

	const string ext[] = {"mov", "avi", "mp4", "mpg", "wmv"};

	const int time_sec = 5, fps = 25;

	size_t n = sizeof(ext)/sizeof(ext[0]);

	int failed = 0;

	for (size_t i = 0; i < n; ++i)
	{
		string file_path = src_dir + "../perf/video/big_buck_bunny." + ext[i];

		CvCapture* cap = cvCreateFileCapture(file_path.c_str());

		if (!cap)
		{
			ts->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\nFAILED\n\n", i+1, ext[i].c_str());
			ts->printf(cvtest::TS::LOG, "Error: cannot read source video file.\n");
			ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
			failed++; continue;
		}

		cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, 0);

		generate_idx_seq(cap, method);

		int N = idx.size(), failed_frames = 0;

		bool flag = false;

		for (int j = 0; j < N; ++j)
		{
			bool res = cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, (double)idx.at(j));
			
			IplImage* frame = cvRetrieveFrame(cap); 

			if (!frame)
			{
				if (!flag) failed++; flag = true;
				if (!failed_frames) ts->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\nFAILED\n\n", i+1, ext[i].c_str());
				ts->printf(cvtest::TS::LOG, "Error: cannot read a frame with index %d.\n", idx.at(j));
				ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION); failed_frames++;
			} 

			if (idx.at(j) != cvGetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES))
			{
				ts->printf(cvtest::TS::LOG, "Iteration: %d\n"\
					"Actual pos: %d Returned pos: %d", idx.at(j), cvGetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES));
				ts->printf(cvtest::TS::LOG, "Error: required and returned position are not matched.\n");
				ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
			}

			cvReleaseImage(&frame);
		}

		cvReleaseCapture(&cap);
	}

	ts->printf(cvtest::TS::LOG, "\nSuccessfull experiments: %d (%d%%)\n", n-failed, 100*(n-failed)/n);
	ts->printf(cvtest::TS::LOG, "Failed experiments: %d (%d%%)\n", failed, 100*failed/n);
}

void CV_VideoProgressivePositioningTest::run(int) 
{
	run_test(PROGRESSIVE);
}

void CV_VideoRandomPositioningTest::run(int)
{
	run_test(RANDOM);
}

TEST (HighguiPositioning, progressive) { CV_VideoProgressivePositioningTest test; test.safe_run(); }
TEST (HighguiPositioning, random) { CV_VideoRandomPositioningTest test; test.safe_run(); }
