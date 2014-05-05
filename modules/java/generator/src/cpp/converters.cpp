#define LOG_TAG "org.opencv.utils.Converters"
#include "common.h"

using namespace cv;

#define CHECK_MAT(cond) if(!(cond)){ LOGD("FAILED: " #cond); return; }


// vector_int

void Mat_to_vector_int(Mat& mat, std::vector<int>& v_int)
{
    v_int.clear();
    CHECK_MAT(mat.type()==CV_32SC1 && mat.cols==1);
    v_int = (std::vector<int>) mat;
}

void vector_int_to_Mat(std::vector<int>& v_int, Mat& mat)
{
    mat = Mat(v_int, true);
}


//vector_double

void Mat_to_vector_double(Mat& mat, std::vector<double>& v_double)
{
    v_double.clear();
    CHECK_MAT(mat.type()==CV_64FC1 && mat.cols==1);
    v_double = (std::vector<double>) mat;
}

void vector_double_to_Mat(std::vector<double>& v_double, Mat& mat)
{
    mat = Mat(v_double, true);
}


// vector_float

void Mat_to_vector_float(Mat& mat, std::vector<float>& v_float)
{
    v_float.clear();
    CHECK_MAT(mat.type()==CV_32FC1 && mat.cols==1);
    v_float = (std::vector<float>) mat;
}

void vector_float_to_Mat(std::vector<float>& v_float, Mat& mat)
{
    mat = Mat(v_float, true);
}


//vector_uchar

void Mat_to_vector_uchar(Mat& mat, std::vector<uchar>& v_uchar)
{
    v_uchar.clear();
    CHECK_MAT(mat.type()==CV_8UC1 && mat.cols==1);
    v_uchar = (std::vector<uchar>) mat;
}

void vector_uchar_to_Mat(std::vector<uchar>& v_uchar, Mat& mat)
{
    mat = Mat(v_uchar, true);
}

void Mat_to_vector_char(Mat& mat, std::vector<char>& v_char)
{
    v_char.clear();
    CHECK_MAT(mat.type()==CV_8SC1 && mat.cols==1);
    v_char = (std::vector<char>) mat;
}

void vector_char_to_Mat(std::vector<char>& v_char, Mat& mat)
{
    mat = Mat(v_char, true);
}


//vector_Rect

void Mat_to_vector_Rect(Mat& mat, std::vector<Rect>& v_rect)
{
    v_rect.clear();
    CHECK_MAT(mat.type()==CV_32SC4 && mat.cols==1);
    v_rect = (std::vector<Rect>) mat;
}

void vector_Rect_to_Mat(std::vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}


//vector_Point
void Mat_to_vector_Point(Mat& mat, std::vector<Point>& v_point)
{
    v_point.clear();
    CHECK_MAT(mat.type()==CV_32SC2 && mat.cols==1);
    v_point = (std::vector<Point>) mat;
}

//vector_Point2f
void Mat_to_vector_Point2f(Mat& mat, std::vector<Point2f>& v_point)
{
    v_point.clear();
    CHECK_MAT(mat.type()==CV_32FC2 && mat.cols==1);
    v_point = (std::vector<Point2f>) mat;
}

//vector_Point2d
void Mat_to_vector_Point2d(Mat& mat, std::vector<Point2d>& v_point)
{
    v_point.clear();
    CHECK_MAT(mat.type()==CV_64FC2 && mat.cols==1);
    v_point = (std::vector<Point2d>) mat;
}


//vector_Point3i
void Mat_to_vector_Point3i(Mat& mat, std::vector<Point3i>& v_point)
{
    v_point.clear();
    CHECK_MAT(mat.type()==CV_32SC3 && mat.cols==1);
    v_point = (std::vector<Point3i>) mat;
}

//vector_Point3f
void Mat_to_vector_Point3f(Mat& mat, std::vector<Point3f>& v_point)
{
    v_point.clear();
    CHECK_MAT(mat.type()==CV_32FC3 && mat.cols==1);
    v_point = (std::vector<Point3f>) mat;
}

//vector_Point3d
void Mat_to_vector_Point3d(Mat& mat, std::vector<Point3d>& v_point)
{
    v_point.clear();
    CHECK_MAT(mat.type()==CV_64FC3 && mat.cols==1);
    v_point = (std::vector<Point3d>) mat;
}


void vector_Point_to_Mat(std::vector<Point>& v_point, Mat& mat)
{
    mat = Mat(v_point, true);
}

void vector_Point2f_to_Mat(std::vector<Point2f>& v_point, Mat& mat)
{
    mat = Mat(v_point, true);
}

void vector_Point2d_to_Mat(std::vector<Point2d>& v_point, Mat& mat)
{
    mat = Mat(v_point, true);
}

void vector_Point3i_to_Mat(std::vector<Point3i>& v_point, Mat& mat)
{
    mat = Mat(v_point, true);
}

void vector_Point3f_to_Mat(std::vector<Point3f>& v_point, Mat& mat)
{
    mat = Mat(v_point, true);
}

void vector_Point3d_to_Mat(std::vector<Point3d>& v_point, Mat& mat)
{
    mat = Mat(v_point, true);
}

#ifdef HAVE_OPENCV_FEATURES2D
//vector_KeyPoint
void Mat_to_vector_KeyPoint(Mat& mat, std::vector<KeyPoint>& v_kp)
{
    v_kp.clear();
    CHECK_MAT(mat.type()==CV_32FC(7) && mat.cols==1);
    for(int i=0; i<mat.rows; i++)
    {
        Vec<float, 7> v = mat.at< Vec<float, 7> >(i, 0);
        KeyPoint kp(v[0], v[1], v[2], v[3], v[4], (int)v[5], (int)v[6]);
        v_kp.push_back(kp);
    }
    return;
}


void vector_KeyPoint_to_Mat(std::vector<KeyPoint>& v_kp, Mat& mat)
{
    int count = (int)v_kp.size();
    mat.create(count, 1, CV_32FC(7));
    for(int i=0; i<count; i++)
    {
        KeyPoint kp = v_kp[i];
        mat.at< Vec<float, 7> >(i, 0) = Vec<float, 7>(kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, (float)kp.octave, (float)kp.class_id);
    }
}
#endif


//vector_Mat
void Mat_to_vector_Mat(cv::Mat& mat, std::vector<cv::Mat>& v_mat)
{
    v_mat.clear();
    if(mat.type() == CV_32SC2 && mat.cols == 1)
    {
        v_mat.reserve(mat.rows);
        for(int i=0; i<mat.rows; i++)
        {
            Vec<int, 2> a = mat.at< Vec<int, 2> >(i, 0);
            long long addr = (((long long)a[0])<<32) | (a[1]&0xffffffff);
            Mat& m = *( (Mat*) addr );
            v_mat.push_back(m);
        }
    } else {
        LOGD("Mat_to_vector_Mat() FAILED: mat.type() == CV_32SC2 && mat.cols == 1");
    }
}


void vector_Mat_to_Mat(std::vector<cv::Mat>& v_mat, cv::Mat& mat)
{
    int count = (int)v_mat.size();
    mat.create(count, 1, CV_32SC2);
    for(int i=0; i<count; i++)
    {
        long long addr = (long long) new Mat(v_mat[i]);
        mat.at< Vec<int, 2> >(i, 0) = Vec<int, 2>(addr>>32, addr&0xffffffff);
    }
}

#ifdef HAVE_OPENCV_FEATURES2D
//vector_DMatch
void Mat_to_vector_DMatch(Mat& mat, std::vector<DMatch>& v_dm)
{
    v_dm.clear();
    CHECK_MAT(mat.type()==CV_32FC4 && mat.cols==1);
    for(int i=0; i<mat.rows; i++)
    {
        Vec<float, 4> v = mat.at< Vec<float, 4> >(i, 0);
        DMatch dm((int)v[0], (int)v[1], (int)v[2], v[3]);
        v_dm.push_back(dm);
    }
    return;
}


void vector_DMatch_to_Mat(std::vector<DMatch>& v_dm, Mat& mat)
{
    int count = (int)v_dm.size();
    mat.create(count, 1, CV_32FC4);
    for(int i=0; i<count; i++)
    {
        DMatch dm = v_dm[i];
        mat.at< Vec<float, 4> >(i, 0) = Vec<float, 4>((float)dm.queryIdx, (float)dm.trainIdx, (float)dm.imgIdx, dm.distance);
    }
}
#endif

void Mat_to_vector_vector_Point(Mat& mat, std::vector< std::vector< Point > >& vv_pt)
{
    std::vector<Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++)
    {
        std::vector<Point> vpt;
        Mat_to_vector_Point(vm[i], vpt);
        vv_pt.push_back(vpt);
    }
}

void Mat_to_vector_vector_Point2f(Mat& mat, std::vector< std::vector< Point2f > >& vv_pt)
{
    std::vector<Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++)
    {
        std::vector<Point2f> vpt;
        Mat_to_vector_Point2f(vm[i], vpt);
        vv_pt.push_back(vpt);
    }
}

void Mat_to_vector_vector_Point3f(Mat& mat, std::vector< std::vector< Point3f > >& vv_pt)
{
    std::vector<Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++)
    {
        std::vector<Point3f> vpt;
        Mat_to_vector_Point3f(vm[i], vpt);
        vv_pt.push_back(vpt);
    }
}

#ifdef HAVE_OPENCV_FEATURES2D
void Mat_to_vector_vector_KeyPoint(Mat& mat, std::vector< std::vector< KeyPoint > >& vv_kp)
{
    std::vector<Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++)
    {
        std::vector<KeyPoint> vkp;
        Mat_to_vector_KeyPoint(vm[i], vkp);
        vv_kp.push_back(vkp);
    }
}

void vector_vector_KeyPoint_to_Mat(std::vector< std::vector< KeyPoint > >& vv_kp, Mat& mat)
{
    std::vector<Mat> vm;
    vm.reserve( vv_kp.size() );
    for(size_t i=0; i<vv_kp.size(); i++)
    {
        Mat m;
        vector_KeyPoint_to_Mat(vv_kp[i], m);
        vm.push_back(m);
    }
    vector_Mat_to_Mat(vm, mat);
}

void Mat_to_vector_vector_DMatch(Mat& mat, std::vector< std::vector< DMatch > >& vv_dm)
{
    std::vector<Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++)
    {
        std::vector<DMatch> vdm;
        Mat_to_vector_DMatch(vm[i], vdm);
        vv_dm.push_back(vdm);
    }
}

void vector_vector_DMatch_to_Mat(std::vector< std::vector< DMatch > >& vv_dm, Mat& mat)
{
    std::vector<Mat> vm;
    vm.reserve( vv_dm.size() );
    for(size_t i=0; i<vv_dm.size(); i++)
    {
        Mat m;
        vector_DMatch_to_Mat(vv_dm[i], m);
        vm.push_back(m);
    }
    vector_Mat_to_Mat(vm, mat);
}
#endif

void Mat_to_vector_vector_char(Mat& mat, std::vector< std::vector< char > >& vv_ch)
{
    std::vector<Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++)
    {
        std::vector<char> vch;
        Mat_to_vector_char(vm[i], vch);
        vv_ch.push_back(vch);
    }
}

void vector_vector_char_to_Mat(std::vector< std::vector< char > >& vv_ch, Mat& mat)
{
    std::vector<Mat> vm;
    vm.reserve( vv_ch.size() );
    for(size_t i=0; i<vv_ch.size(); i++)
    {
        Mat m;
        vector_char_to_Mat(vv_ch[i], m);
        vm.push_back(m);
    }
    vector_Mat_to_Mat(vm, mat);
}

void vector_vector_Point_to_Mat(std::vector< std::vector< Point > >& vv_pt, Mat& mat)
{
    std::vector<Mat> vm;
    vm.reserve( vv_pt.size() );
    for(size_t i=0; i<vv_pt.size(); i++)
    {
        Mat m;
        vector_Point_to_Mat(vv_pt[i], m);
        vm.push_back(m);
    }
    vector_Mat_to_Mat(vm, mat);
}

void vector_vector_Point2f_to_Mat(std::vector< std::vector< Point2f > >& vv_pt, Mat& mat)
{
    std::vector<Mat> vm;
    vm.reserve( vv_pt.size() );
    for(size_t i=0; i<vv_pt.size(); i++)
    {
        Mat m;
        vector_Point2f_to_Mat(vv_pt[i], m);
        vm.push_back(m);
    }
    vector_Mat_to_Mat(vm, mat);
}

void vector_vector_Point3f_to_Mat(std::vector< std::vector< Point3f > >& vv_pt, Mat& mat)
{
    std::vector<Mat> vm;
    vm.reserve( vv_pt.size() );
    for(size_t i=0; i<vv_pt.size(); i++)
    {
        Mat m;
        vector_Point3f_to_Mat(vv_pt[i], m);
        vm.push_back(m);
    }
    vector_Mat_to_Mat(vm, mat);
}

void vector_Vec4i_to_Mat(std::vector<Vec4i>& v_vec, Mat& mat)
{
    mat = Mat(v_vec, true);
}

void vector_Vec4f_to_Mat(std::vector<Vec4f>& v_vec, Mat& mat)
{
    mat = Mat(v_vec, true);
}

void vector_Vec6f_to_Mat(std::vector<Vec6f>& v_vec, Mat& mat)
{
    mat = Mat(v_vec, true);
}
