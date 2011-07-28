#include "converters.h"

#ifdef DEBUG
#include <android/log.h>
#define MODULE_LOG_TAG "OpenCV.converters"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, MODULE_LOG_TAG, __VA_ARGS__))
#else //DEBUG
#define LOGD(...)
#endif //DEBUG

using namespace cv;

#define CHECK_MAT(cond) if(cond){ LOGD(#cond); return; }


// vector_int

void Mat_to_vector_int(Mat& mat, vector<int>& v_int)
{
	v_int.clear();
	CHECK_MAT(mat.type()!= CV_32SC1 || mat.cols!=1);
	v_int = (vector<int>) mat;
}

void vector_int_to_Mat(vector<int>& v_int, Mat& mat)
{
	mat = Mat(v_int);
}


//vector_double

void Mat_to_vector_double(Mat& mat, vector<double>& v_double)
{
	v_double.clear();
	CHECK_MAT(mat.type()!= CV_64FC1 || mat.cols!=1);
	v_double = (vector<double>) mat;
}

void vector_double_to_Mat(vector<double>& v_double, Mat& mat)
{
	mat = Mat(v_double);
}


// vector_float

void Mat_to_vector_float(Mat& mat, vector<float>& v_float)
{
	v_float.clear();
	CHECK_MAT(mat.type()!= CV_32FC1 || mat.cols!=1);
	v_float = (vector<float>) mat;
}

void vector_float_to_Mat(vector<float>& v_float, Mat& mat)
{
	mat = Mat(v_float);
}


//vector_uchar

void Mat_to_vector_uchar(Mat& mat, vector<uchar>& v_uchar)
{
	v_uchar.clear();
	CHECK_MAT(mat.type()!= CV_8UC1 || mat.cols!=1);
	v_uchar = (vector<uchar>) mat;
}


//vector_Rect

void Mat_to_vector_Rect(Mat& mat, vector<Rect>& v_rect)
{
	v_rect.clear();
	CHECK_MAT(mat.type()!= CV_32SC4 || mat.cols!=1);
	v_rect = (vector<Rect>) mat;
}

void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
	mat = Mat(v_rect);
}


//vector_Point
void Mat_to_vector_Point(Mat& mat, vector<Point>& v_point)
{
	v_point.clear();
	CHECK_MAT(mat.type()!= CV_32SC2 || mat.cols!=1);
	v_point = (vector<Point>) mat;
}


void vector_Point_to_Mat(vector<Point>& v_point, Mat& mat)
{
	mat = Mat(v_point);
}


//vector_KeyPoint
void Mat_to_vector_KeyPoint(Mat& mat, vector<KeyPoint>& v_kp)
{
    v_kp.clear();
    CHECK_MAT(mat.type()!= CV_64FC(7) || mat.cols!=1);
	for(int i=0; i<mat.rows; i++)
	{
		Vec<double, 7> v = mat.at< Vec<double, 7> >(i, 0);
		KeyPoint kp((float)v[0], (float)v[1], (float)v[2], (float)v[3], (float)v[4], (int)v[5], (int)v[6]);
		v_kp.push_back(kp);
	}
    return;
}


void vector_KeyPoint_to_Mat(vector<KeyPoint>& v_kp, Mat& mat)
{
	int count = v_kp.size();
	mat.create(count, 1, CV_64FC(7));
	for(int i=0; i<count; i++)
	{
		KeyPoint kp = v_kp[i];
		mat.at< Vec<double, 7> >(i, 0) = Vec<double, 7>(kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id);
	}
}


//vector_Mat
void Mat_to_vector_Mat(cv::Mat& mat, std::vector<cv::Mat>& v_mat)
{
	v_mat.clear();
	if(mat.type() == CV_32SC2 && mat.cols == 1)
	{
		for(int i=0; i<mat.rows; i++)
		{
			Vec<int, 2> a = mat.at< Vec<int, 2> >(i, 0);
			long long addr = (((long long)a[0])<<32) | a[1];
			Mat& m = *( (Mat*) addr );
			v_mat.push_back(m);
		}
	}
}


void vector_Mat_to_Mat(std::vector<cv::Mat>& v_mat, cv::Mat& mat)
{
	int count = v_mat.size();
	mat.create(count, 1, CV_32SC2);
	for(int i=0; i<count; i++)
	{
		long long addr = (long long) new Mat(v_mat[i]);
		mat.at< Vec<int, 2> >(i, 0) = Vec<int, 2>(addr>>32, addr&0xffffffff);
	}
}
