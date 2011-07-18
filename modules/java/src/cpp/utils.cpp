#include "utils.h"
#include <android/log.h>
#define MODULE_LOG_TAG "OpenCV.utils.cpp"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, MODULE_LOG_TAG, __VA_ARGS__))


using namespace cv;

// vector_int

void Mat_to_vector_int(Mat& mat, vector<int>& v_int)
{
	v_int.clear();

	if(mat.type()!= CV_32SC1 || mat.rows!=1)
		return;

	for(int i=0; i<mat.cols; i++)
		v_int.push_back( mat.at< int >(0, i) );

}

void vector_int_to_Mat(vector<int>& v_int, Mat& mat)
{
	mat.create(1, v_int.size(), CV_32SC1);
	for(size_t i=0; i<v_int.size(); i++)
		mat.at< int >(0, i) = v_int[i];
}


//vector_double

void Mat_to_vector_double(Mat& mat, vector<double>& v_double)
{
	v_double.clear();

	if(mat.type()!= CV_64FC1 || mat.rows!=1)
		return;

	for(int i=0; i<mat.cols; i++)
		v_double.push_back( mat.at< double >(0, i) );

}

void vector_double_to_Mat(vector<double>& v_double, Mat& mat)
{
	mat.create(1, v_double.size(), CV_64FC1);
	for(size_t i=0; i<v_double.size(); i++)
		mat.at< double >(0, i) = v_double[i];
}


// vector_float

void Mat_to_vector_float(Mat& mat, vector<float>& v_float)
{
	v_float.clear();

	if(mat.type()!= CV_32FC1 || mat.rows!=1)
		return;

	for(int i=0; i<mat.cols; i++)
		v_float.push_back( mat.at< float >(0, i) );

}

void vector_float_to_Mat(vector<float>& v_float, Mat& mat)
{
	mat.create(1, v_float.size(), CV_32FC1);
	for(size_t i=0; i<v_float.size(); i++)
		mat.at< float >(0, i) = v_float[i];
}


//vector_uchar

void Mat_to_vector_uchar(cv::Mat& mat, std::vector<uchar>& v_uchar)
{
	v_uchar.clear();

	if(mat.type()!= CV_8UC1 || mat.rows!=1)
		return;

	for(int i=0; i<mat.cols; i++)
		v_uchar.push_back( mat.at< uchar >(0, i) );

}


//vector_Rect

void Mat_to_vector_Rect(Mat& mat, vector<Rect>& v_rect)
{
	v_rect.clear();

	if(mat.type()!= CV_32SC4 || mat.rows!=1) {
		LOGD("ERROR mat.type()!= CV_32SC4 || mat.rows!=1");
		return;
	}

	for(int i=0; i<mat.cols; i++) {
		Vec<int, 4> v=mat.at< Vec<int, 4> >(0, i);
		v_rect.push_back( Rect(v[0], v[1], v[2], v[3]) );
	}
}

void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
	mat.create(1, v_rect.size(), CV_32SC4);
	for(size_t i=0; i<v_rect.size(); i++) {
		mat.at< Vec<int, 4> >(0, i) = Vec<int, 4>(v_rect[i].x, v_rect[i].y, v_rect[i].width, v_rect[i].height);
	}
}


//vector_Point
void Mat_to_vector_Point(Mat& mat, vector<Point>& v_point)
{
	v_point.clear();

	if(mat.type()!= CV_32SC2 || mat.rows!=1)
		return;

	for(int i=0; i<mat.cols; i++)
		v_point.push_back( Point( mat.at< Vec<int, 2> >(0, i) ) );
}


void vector_Point_to_Mat(vector<Point>& v_point, Mat& mat)
{
	mat.create(1, v_point.size(), CV_32SC2);
	for(size_t i=0; i<v_point.size(); i++)
		mat.at< Vec<int, 2> >(0, i) = Vec<int, 2>(v_point[i].x, v_point[i].y);
}


//vector_KeyPoint
void Mat_to_vector_KeyPoint(Mat& mat, vector<KeyPoint>& v_kp)
{
    return;
}


void vector_KeyPoint_to_Mat(vector<KeyPoint>& v_kp, Mat& mat)
{
    return;
}


//vector_Mat
void Mat_to_vector_Mat(cv::Mat& mat, std::vector<cv::Mat>& v_mat)
{
	v_mat.clear();
	if(mat.type() == CV_32SC2 && mat.rows == 1)
	{
		for(int i=0; i<mat.cols; i++)
		{
			Vec<int, 2> a = mat.at< Vec<int, 2> >(0, i);
			long long addr = (((long long)a[0])<<32) | a[1];
			Mat& m = *( (Mat*) addr );
			v_mat.push_back(m);
		}
	}
}


void vector_Mat_to_Mat(std::vector<cv::Mat>& v_mat, cv::Mat& mat)
{
	int count = v_mat.size();
	mat.create(1, count, CV_32SC2);
	for(int i=0; i<count; i++)
	{
		long long addr = (long long) &v_mat[i];
		mat.at< Vec<int, 2> >(0, i) = Vec<int, 2>(addr>>32, addr&0xffffffff);
	}
    return;
}
