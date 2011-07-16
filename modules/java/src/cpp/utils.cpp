#include "utils.h"

using namespace cv;

// vector_int

void Mat_to_vector_int(Mat& mat, vector<int>& v_int)
{
    return;
}

void vector_int_to_Mat(vector<int>& v_int, Mat& mat)
{
    return;
}


//vector_double

void Mat_to_vector_double(Mat& mat, vector<double>& v_double)
{
    return;
}

void vector_double_to_Mat(vector<double>& v_double, Mat& mat)
{
    return;
}


// vector_float

void Mat_to_vector_float(Mat& mat, vector<float>& v_float)
{
    return;
}

void vector_float_to_Mat(vector<float>& v_float, Mat& mat)
{
    return;
}


//vector_uchar

void Mat_to_vector_uchar(cv::Mat& mat, std::vector<uchar>& v_uchar)
{
    return;
}


//vector_Rect

void Mat_to_vector_Rect(Mat& mat, vector<Rect>& v_rect)
{
    return;
}

void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    return;
}


//vector_Point
void Mat_to_vector_Point(Mat& mat, vector<Point>& v_point)
{
	v_point.clear();
	
	if(mat.type()!= CV_32SC2 || mat.rows!=1)
		return;

	for(int i=0; i<mat.cols; i++)
		v_point.push_back( Point( mat.at< Vec<int, 2> >(0, i) ) );

    return;
}


void vector_Point_to_Mat(vector<Point>& v_point, Mat& mat)
{
	mat.create(1, v_point.size(), CV_32SC2);
	for(int i=0; i<v_point.size(); i++)
		mat.at< Vec<int, 2> >(0, i) = Vec<int, 2>(v_point[i].x, v_point[i].y);
    return;
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

