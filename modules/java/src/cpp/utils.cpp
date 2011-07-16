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
	LOGD("Mat_to_vector_Rect start, mat.cols=%d", mat.cols);
	v_rect.clear();

	if(mat.type()!= CV_32SC4 || mat.rows!=1) {
		LOGD("ERROR mat.type()!= CV_32SC4 || mat.rows!=1");
		return;
	}

	for(int i=0; i<mat.cols; i++) {
		Vec<int, 4> v=mat.at< Vec<int, 4> >(0, i);
		v_rect.push_back( Rect(v[0], v[1], v[2], v[3]) );
	}
	LOGD("Mat_to_vector_Rect end, vec.size=%d", (int)v_rect.size());
}

void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
	LOGD("vector_Rect_to_Mat start, vec.size=%d", (int)v_rect.size());
	mat.create(1, v_rect.size(), CV_32SC4);
	for(size_t i=0; i<v_rect.size(); i++) {
		mat.at< Vec<int, 4> >(0, i) = Vec<int, 4>(v_rect[i].x, v_rect[i].y, v_rect[i].width, v_rect[i].height);
	}
	LOGD("vector_Rect_to_Mat end, mat.cols=%d", mat.cols);
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
    return;
}


void vector_Mat_to_Mat(std::vector<cv::Mat>& v_mat, cv::Mat& mat)
{
    return;
}

