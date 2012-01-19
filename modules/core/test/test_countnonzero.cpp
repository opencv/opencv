#include "test_precomp.hpp"
#include <time.h>

using namespace cv;
using namespace std;

#define sign(a) a > 0 ? 1 : a == 0 ? 0 : -1

const int FLOAT_TYPE [2] = {CV_32F, CV_64F};
const int INT_TYPE [5] = {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S};

#define MAX_CHANNELS 4
#define MAX_WIDTH 1e+2
#define MAX_HEIGHT 1e+2

class CV_CountNonZeroTest: public cvtest::BaseTest
{
 public:
	CV_CountNonZeroTest();
	~CV_CountNonZeroTest();

 protected:
	void run (int);

 private:
	 float eps_32; double eps_64; Mat src;
	
	 void generate_src_data(cv::Size size, int type, int channels);
	 void generate_src_data(cv::Size size, int type, int channels, int count_non_zero);
	 void generate_src_float_data(cv::Size size, int type, int channels, int distribution);

	 void checking_function_work();  
	 void checking_function_work(int count_non_zero); 
};

CV_CountNonZeroTest::CV_CountNonZeroTest(): eps_32(1e-2), eps_64(1e-4), src(Mat()) {}
CV_CountNonZeroTest::~CV_CountNonZeroTest() {}

void CV_CountNonZeroTest::generate_src_data(cv::Size size, int type, int channels)
{
 src.create(size, CV_MAKETYPE(type, channels));

 for (size_t i = 0; i < size.width; ++i)
	for (size_t j = 0; j < size.height; ++j)
		{
		 if (type == CV_8U) switch (channels)
		 {
		  case 1: {src.at<uchar>(j, i) = cv::randu<uchar>(); break;}
		  case 2: {src.at<Vec2b>(j, i) = Vec2b(cv::randu<uchar>(), cv::randu<uchar>()); break;}
		  case 3: {src.at<Vec3b>(j, i) = Vec3b(cv::randu<uchar>(), cv::randu<uchar>(), cv::randu<uchar>()); break;}
		  case 4: {src.at<Vec4b>(j, i) = Vec4b(cv::randu<uchar>(), cv::randu<uchar>(), cv::randu<uchar>(), cv::randu<uchar>()); break;}

		  default: break;
		 }

		 else if (type == CV_8S) switch (channels)
		 {
		  case 1: {src.at<char>(j,i) = cv::randu<uchar>()-128; break; }
		  case 2: {src.at<Vec<char, 2>>(j, i) = Vec<char, 2>(cv::randu<uchar>()-128, cv::randu<uchar>()-128); break;}
		  case 3: {src.at<Vec<char, 3>>(j, i) = Vec<char, 3>(cv::randu<uchar>()-128, cv::randu<uchar>()-128, cv::randu<uchar>()-128); break;}
		  case 4: {src.at<Vec<char, 4>>(j, i) = Vec<char, 4>(cv::randu<uchar>()-128, cv::randu<uchar>()-128, cv::randu<uchar>()-128, cv::randu<uchar>()-128); break;}
		  default:break;
		 }
			
		 else if (type == CV_16U) switch (channels)
		 {
		  case 1: {src.at<ushort>(j, i) = cv::randu<ushort>(); break;}
		  case 2: {src.at<Vec<ushort, 2>>(j, i) = Vec<ushort, 2>(cv::randu<ushort>(), cv::randu<ushort>()); break;}
		  case 3: {src.at<Vec<ushort, 3>>(j, i) = Vec<ushort, 3>(cv::randu<ushort>(), cv::randu<ushort>(), cv::randu<ushort>()); break;}
		  case 4: {src.at<Vec<ushort, 4>>(j, i) = Vec<ushort, 4>(cv::randu<ushort>(), cv::randu<ushort>(), cv::randu<ushort>(), cv::randu<ushort>()); break;}
		  default: break;
		 }

		 else if (type == CV_16S) switch (channels)
		 {
		  case 1: {src.at<short>(j, i) = cv::randu<short>(); break;}
		  case 2: {src.at<Vec2s>(j, i) = Vec2s(cv::randu<short>(), cv::randu<short>()); break;}
		  case 3: {src.at<Vec3s>(j, i) = Vec3s(cv::randu<short>(), cv::randu<short>(), cv::randu<short>()); break;}
		  case 4: {src.at<Vec4s>(j, i) = Vec4s(cv::randu<short>(), cv::randu<short>(), cv::randu<short>(), cv::randu<short>()); break;}
		  default: break;
		 }

		 else if (type == CV_32S) switch (channels)
		 {
		  case 1: {src.at<int>(j, i) = cv::randu<int>(); break;}
		  case 2: {src.at<Vec2i>(j, i) = Vec2i(cv::randu<int>(), cv::randu<int>()); break;}
		  case 3: {src.at<Vec3i>(j, i) = Vec3i(cv::randu<int>(), cv::randu<int>(), cv::randu<int>()); break;}
		  case 4: {src.at<Vec4i>(j, i) = Vec4i(cv::randu<int>(), cv::randu<int>(), cv::randu<int>(), cv::randu<int>()); break;}
		  default: break;
		 }

		 else if (type == CV_32F) switch (channels)
		 {
		  case 1: {src.at<float>(j, i) = cv::randu<float>(); break;}
		  case 2: {src.at<Vec2f>(j, i) = Vec2i(cv::randu<float>(), cv::randu<float>()); break;}
		  case 3: {src.at<Vec3f>(j, i) = Vec3i(cv::randu<float>(), cv::randu<float>(), cv::randu<float>()); break;}
		  case 4: {src.at<Vec4f>(j, i) = Vec4i(cv::randu<float>(), cv::randu<float>(), cv::randu<float>(), cv::randu<float>()); break;}
		  default: break;
		 }

		 else if (type == CV_64F) switch (channels)
		 {
		  case 1: {src.at<double>(j, i) = cv::randu<double>(); break;}
		  case 2: {src.at<Vec2d>(j, i) = Vec2d(cv::randu<double>(), cv::randu<double>()); break;}
		  case 3: {src.at<Vec3d>(j, i) = Vec3d(cv::randu<double>(), cv::randu<double>(), cv::randu<double>()); break;}
		  case 4: {src.at<Vec4d>(j, i) = Vec4d(cv::randu<double>(), cv::randu<double>(), cv::randu<double>(), cv::randu<double>()); break;}
		  default: break;
		 }
		}
}

void CV_CountNonZeroTest::generate_src_data(cv::Size size, int type, int channels, int count_non_zero)
{
 src = Mat::zeros(size, CV_MAKETYPE(type, channels));
 
 int n = -1;

 while (n < count_non_zero)
 {
	 RNG& rng = ts->get_rng();

	 size_t i = rng.next()%size.height, j = rng.next()%size.width;
	 
	 switch (type)
	 {
	  case CV_8U: 
		 {
		   if (channels == 1) 
		   {
			uchar value = cv::randu<uchar>();
			if (value != 0) {src.at<uchar>(i, j) = value; n++;}
		   }
			 
		   else if (channels == 2)
		   {
			Vec2b value(cv::randu<uchar>(), cv::randu<uchar>());
			if (value != Vec2b(0, 0)) {src.at<Vec2b>(i, j) = value; n++;}
		   }

		   else if (channels == 3)
		   {
			Vec3b value(cv::randu<uchar>(), cv::randu<uchar>(), cv::randu<uchar>());
			if (value != Vec3b(0, 0, 0)) {src.at<Vec3b>(i, j) = value; n++;}
		   }

		   else

		   {
		    Vec4b value(cv::randu<uchar>(), cv::randu<uchar>(), cv::randu<uchar>(), cv::randu<uchar>());
			if (value != Vec4b(0, 0, 0, 0)) {src.at<Vec4b>(i, j) = value; n++;}
		   }

		   break;
		 }

	  case CV_8S:
		  {
		   if (channels == 1) 
		   {
			char value = cv::randu<uchar>()-128;
			if (value != 0) {src.at<char>(i, j) = value; n++;}
		   }
			 
		   else if (channels == 2)
		   {
			Vec<char, 2> value(cv::randu<uchar>()-128, cv::randu<uchar>()-128);
			if (value != Vec<char, 2>(0, 0)) {src.at<Vec<char, 2>>(i, j) = value; n++;}
		   }

		   else if (channels == 3)
		   {
			Vec<char, 3> value(cv::randu<uchar>()-128, cv::randu<uchar>()-128, cv::randu<uchar>()-128);
			if (value != Vec<char, 3>(0, 0, 0)) {src.at<Vec<char, 3>>(i, j) = value; n++;}
		   }

		   else

		   {
		    Vec<char, 4> value(cv::randu<uchar>()-128, cv::randu<uchar>()-128, cv::randu<uchar>()-128, cv::randu<uchar>()-128);
			if (value != Vec<char, 4>(0, 0, 0, 0)) {src.at<Vec<char, 4>>(i, j) = value; n++;}
		   }

		   break;
		  }

	  case CV_16U:
		  {
		   if (channels == 1) 
		   {
			ushort value = cv::randu<ushort>();
			n += abs(sign(value));
			src.at<ushort>(i, j) = value;
		   }
			 
		   else if (channels == 2)
		   {
			Vec<ushort, 2> value(cv::randu<ushort>(), cv::randu<ushort>());
			if (value != Vec<ushort, 2>(0, 0)) {src.at<Vec<ushort, 2>>(i, j) = value; n++;}
		   }

		   else if (channels == 3)
		   {
		    Vec<ushort, 3> value(cv::randu<ushort>(), cv::randu<ushort>(), cv::randu<ushort>());
			if (value != Vec<ushort, 3>(0, 0, 0)) {src.at<Vec<ushort, 3>>(i, j) = value; n++;}
		   }

		   else

		   {
		    Vec<ushort, 4> value(cv::randu<ushort>(), cv::randu<ushort>(), cv::randu<ushort>(), cv::randu<ushort>());
			if (value != Vec<ushort, 4>(0, 0, 0, 0)) {src.at<Vec<ushort, 4>>(i, j) = value; n++;}
		   }

		   break;
		  }

	  case CV_16S:
		  {
		   if (channels == 1) 
		   {
			short value = cv::randu<short>();
			n += abs(sign(value));
			src.at<short>(i, j) = value;
		   }
			 
		   else if (channels == 2)
		   {
			Vec2s value(cv::randu<short>(), cv::randu<short>());
			if (value != Vec2s(0, 0)) {src.at<Vec2s>(i, j) = value; n++;}
		   }

		   else if (channels == 3)
		   {
			Vec3s value(cv::randu<short>(), cv::randu<short>(), cv::randu<short>());
			if (value != Vec3s(0, 0, 0)) {src.at<Vec3s>(i, j) = value; n++;}
		   }

		   else

		   {
		    Vec4s value(cv::randu<short>(), cv::randu<short>(), cv::randu<short>(), cv::randu<short>());
			if (value != Vec4s(0, 0, 0, 0)) {src.at<Vec4s>(i, j) = value; n++;}
		   }

		   break;
		  }

	  case CV_32S:
		  {
		   if (channels == 1) 
		   {
			int value = cv::randu<int>();
			n += abs(sign(value));
			src.at<int>(i, j) = value;
		   }
			 
		   else if (channels == 2)
		   {
			Vec2i value(cv::randu<int>(), cv::randu<int>());
			if (value != Vec2i(0, 0)) {src.at<Vec2i>(i, j) = value; n++;}
		   }

		   else if (channels == 3)
		   {
			Vec3i value(cv::randu<int>(), cv::randu<int>(), cv::randu<int>());
			if (value != Vec3i(0, 0, 0)) {src.at<Vec3i>(i, j) = value; n++;}
		   }

		   else

		   {
		    Vec4i value(cv::randu<int>(), cv::randu<int>(), cv::randu<int>(), cv::randu<int>());
			if (value != Vec4i(0, 0, 0, 0)) {src.at<Vec4i>(i, j) = value; n++;}
		   }

		   break;
		  }


	  case CV_32F:
		  {
		    if (channels == 1)
			{
			 float value = cv::randu<float>();
			 n += sign(fabs(value) > eps_32);
			 src.at<float>(i, j) = value;  
			}
			
			else 

			if (channels == 2)
			{
			 Vec2f value(cv::randu<float>(), cv::randu<float>());
			 n += sign(cv::norm(value) > eps_32); 
			 src.at<Vec2f>(i, j) = value; 
			}

			else 

			if (channels == 3)
			{
			 Vec3f value(cv::randu<float>(), cv::randu<float>(), cv::randu<float>());
			 n += sign(cv::norm(value) > eps_32); 
			 src.at<Vec3f>(i, j) = value;
			}
				
			else
				
			{
			 Vec4f value(cv::randu<float>(), cv::randu<float>(), cv::randu<float>(), cv::randu<float>());
			 n += sign(cv::norm(value) > eps_32);
			 src.at<Vec4f>(i, j) = value;
			}

			break;
		  }

	  case CV_64F:
		  {
		   if (channels == 1)
			{
			 double value = cv::randu<double>();
			 n += sign(fabs(value) > eps_64);
			 src.at<double>(i, j) = value;  
			}
			
			else 

			if (channels == 2)
			{
			 Vec2d value(cv::randu<double>(), cv::randu<double>());
			 n += sign(cv::norm(value) > eps_64); 
			 src.at<Vec2d>(i, j) = value; 
			}

			else 

			if (channels == 3)
			{
			 Vec3d value(cv::randu<double>(), cv::randu<double>(), cv::randu<double>());
			 n += sign(cv::norm(value) > eps_64); 
			 src.at<Vec3d>(i, j) = value;
			}
				
			else
				
			{
			 Vec4d value(cv::randu<double>(), cv::randu<double>(), cv::randu<double>(), cv::randu<double>());
			 n += sign(cv::norm(value) > eps_64);
			 src.at<Vec4d>(i, j) = value;
			}

		   break;
		  }

	  default: break;
	 }
 }
 
}

void CV_CountNonZeroTest::generate_src_float_data(cv::Size size, int type, int channels, int distribution)
{
 src.create(size, CV_MAKETYPE(type, channels));

 double mean = 0.0, sigma = 1.0;
 double left = -1.0, right = 1.0;

 RNG& rng = ts->get_rng();

 if (distribution == RNG::NORMAL) 
	 rng.fill(src, RNG::NORMAL, Scalar::all(mean), Scalar::all(sigma));
 else if (distribution == RNG::UNIFORM)
	 rng.fill(src, RNG::UNIFORM, Scalar::all(left), Scalar::all(right));
}

void CV_CountNonZeroTest::run(int)
{
 
}

TEST (Core_CountNonZero, accuracy) { CV_CountNonZeroTest test; test.safe_run(); }