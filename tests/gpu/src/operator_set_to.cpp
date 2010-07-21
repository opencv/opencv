#include "gputest.hpp"
#include "highgui.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuMatOpSetTo : public CvTest
{
    public:
        CV_GpuMatOpSetTo();
        ~CV_GpuMatOpSetTo();
    protected:
        void print_mat(cv::Mat & mat, std::string name = "cpu mat");
        void print_mat(gpu::GpuMat & mat, std::string name = "gpu mat");
        void run(int);

        bool test_cv_8u_c1();
        bool test_cv_8u_c2();
        bool test_cv_8u_c3();
        bool test_cv_8u_c4();

        bool test_cv_16u_c4();

        bool test_cv_32f_c1();
        bool test_cv_32f_c2();
        bool test_cv_32f_c3();
        bool test_cv_32f_c4();

    private:
        int w;
        int h;
        Scalar s;
};

CV_GpuMatOpSetTo::CV_GpuMatOpSetTo(): CvTest( "GpuMatOperatorSetTo", "setTo" )
{
    w = 100;
    h = 100;

    s.val[0] = 128.0;
    s.val[1] = 128.0;
    s.val[2] = 128.0;
    s.val[3] = 128.0;

    //#define PRINT_MATRIX
}

CV_GpuMatOpSetTo::~CV_GpuMatOpSetTo() {}

void CV_GpuMatOpSetTo::print_mat(cv::Mat & mat, std::string name )
{
    cv::imshow(name, mat);
}

void CV_GpuMatOpSetTo::print_mat(gpu::GpuMat & mat, std::string name)
{
    cv::Mat newmat;
    mat.download(newmat);
    print_mat(newmat, name);
}

bool CV_GpuMatOpSetTo::test_cv_8u_c1()
{
    Mat cpumat(w, h, CV_8U, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}

bool CV_GpuMatOpSetTo::test_cv_8u_c2()
{
    Mat cpumat(w, h, CV_8UC2, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}

bool CV_GpuMatOpSetTo::test_cv_8u_c3()
{
    Mat cpumat(w, h, CV_8UC3, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}

bool CV_GpuMatOpSetTo::test_cv_8u_c4()
{
    Mat cpumat(w, h, CV_8UC4, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}

bool CV_GpuMatOpSetTo::test_cv_16u_c4()
{
    Mat cpumat(w, h, CV_16UC4, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}


bool CV_GpuMatOpSetTo::test_cv_32f_c1()
{
    Mat cpumat(w, h, CV_32F, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}

bool CV_GpuMatOpSetTo::test_cv_32f_c2()
{
    Mat cpumat(w, h, CV_32FC2, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret;
        return false;
    }
}

bool CV_GpuMatOpSetTo::test_cv_32f_c3()
{
    Mat cpumat(w, h, CV_32FC3, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret;
        return false;
    }
}

bool CV_GpuMatOpSetTo::test_cv_32f_c4()
{
    Mat cpumat(w, h, CV_32FC4, Scalar::all(0));
    GpuMat gpumat(cpumat);

    cpumat.setTo(s);
    gpumat.setTo(s);

#ifdef PRINT_MATRIX
    print_mat(cpumat);
    print_mat(gpumat);
    cv::waitKey(0);
#endif

    double ret = norm(cpumat, gpumat);

    if (ret < 0.1)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}

void CV_GpuMatOpSetTo::run( int /* start_from */)
{
    bool is_test_good = true;

    is_test_good &= test_cv_8u_c1();
    is_test_good &= test_cv_8u_c2();
    is_test_good &= test_cv_8u_c3();
    is_test_good &= test_cv_8u_c4();

    is_test_good &= test_cv_16u_c4();

    is_test_good &= test_cv_32f_c1();
    is_test_good &= test_cv_32f_c2();
    is_test_good &= test_cv_32f_c3();
    is_test_good &= test_cv_32f_c4();

    if (is_test_good == true)
        ts->set_failed_test_info(CvTS::OK);
    else
        ts->set_failed_test_info(CvTS::FAIL_GENERIC);
}

CV_GpuMatOpSetTo CV_GpuMatOpSetTo_test;
