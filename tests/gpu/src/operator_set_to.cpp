#include "gputest.hpp"
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
        void print_mat(cv::Mat & mat);
        void run(int);
};

CV_GpuMatOpSetTo::CV_GpuMatOpSetTo(): CvTest( "GpuMatOperatorSetTo", "setTo" ) {}
CV_GpuMatOpSetTo::~CV_GpuMatOpSetTo() {}

void CV_GpuMatOpSetTo::print_mat(cv::Mat & mat)
{
    for (size_t j = 0; j < mat.rows; j++)
    {
        for (size_t i = 0; i < mat.cols; i++)
        {
            std::cout << " " << int(mat.ptr<unsigned char>(j)[i]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void CV_GpuMatOpSetTo::run( int /* start_from */)
{
    Mat cpumat(1024, 1024, CV_8U, Scalar::all(0));
    GpuMat gpumat(cpumat);

    Scalar s(3);

    cpumat.setTo(s);
    gpumat.setTo(s);

    double ret = norm(cpumat, gpumat);

    /*
    std::cout << "norm() = " << ret << "\n";

    std::cout << "cpumat: \n";
    print_mat(cpumat);

    Mat newmat;
    gpumat.download(newmat);

    std::cout << "gpumat: \n";
    print_mat(newmat);
    */

    if (ret < 1.0)
        ts->set_failed_test_info(CvTS::OK);
    else
        ts->set_failed_test_info(CvTS::FAIL_GENERIC);
}

CV_GpuMatOpSetTo CV_GpuMatOpSetTo_test;
