#define LOG_TAG "org.opencv.utils.Converters"
#include "common.h"
#include "features_converters.hpp"

using namespace cv;

#define CHECK_MAT(cond) if(!(cond)){ LOGD("FAILED: " #cond); return; }


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
