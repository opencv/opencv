#include "gputest.hpp"
#include "highgui.h"
#include "cv.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <iomanip> // for  cout << setw()

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuMatOpCopyTo : public CvTest
{
    public:
        CV_GpuMatOpCopyTo();
        ~CV_GpuMatOpCopyTo();
    protected:

        template <typename T>
        void print_mat(const T & mat, const std::string & name) const;

        void run(int);

        bool compare_matrix(cv::Mat & cpumat, gpu::GpuMat & gpumat);

    private:
        int rows;
        int cols;
};

CV_GpuMatOpCopyTo::CV_GpuMatOpCopyTo(): CvTest( "GpuMatOperatorCopyTo", "copyTo" )
{
    rows = 234;
    cols = 123;

    //#define PRINT_MATRIX
}

CV_GpuMatOpCopyTo::~CV_GpuMatOpCopyTo() {}

template<typename T>
void CV_GpuMatOpCopyTo::print_mat(const T & mat, const std::string & name) const
{
    cv::imshow(name, mat);
}

bool CV_GpuMatOpCopyTo::compare_matrix(cv::Mat & cpumat, gpu::GpuMat & gpumat)
{
    Mat cmat(cpumat.size(), cpumat.type(), Scalar::all(0));
    GpuMat gmat(cmat);

    Mat cpumask(cpumat.size(), CV_8U);
    randu(cpumask, Scalar::all(0), Scalar::all(127));
    threshold(cpumask, cpumask, 0, 127, THRESH_BINARY);
    GpuMat gpumask(cpumask);

    //int64 time = getTickCount();
    cpumat.copyTo(cmat, cpumask);
    //int64 time1 = getTickCount();
    gpumat.copyTo(gmat, gpumask);
    //int64 time2 = getTickCount();

    //std::cout << "\ntime cpu: " << std::fixed << std::setprecision(12) << 1.0 / double((time1 - time)  / (double)getTickFrequency());
    //std::cout << "\ntime gpu: " << std::fixed << std::setprecision(12) << 1.0 / double((time2 - time1) / (double)getTickFrequency());
    //std::cout << "\n";

#ifdef PRINT_MATRIX
    print_mat(cmat, "cpu mat");
    print_mat(gmat, "gpu mat");
    print_mat(cpumask, "cpu mask");
    print_mat(gpumask, "gpu mask");
    cv::waitKey(0);
#endif

    double ret = norm(cmat, gmat);

    if (ret < 1.0)
        return true;
    else
    {
        std::cout << "return : " << ret << "\n";
        return false;
    }
}

void CV_GpuMatOpCopyTo::run( int /* start_from */)
{
    bool is_test_good = true;

    for (int i = 0 ; i < 7; i++)
    {
        Mat cpumat(rows, cols, i);
        cpumat.setTo(Scalar::all(127));

        GpuMat gpumat(cpumat);

        is_test_good &= compare_matrix(cpumat, gpumat);
    }

    if (is_test_good == true)
        ts->set_failed_test_info(CvTS::OK);
    else
        ts->set_failed_test_info(CvTS::FAIL_GENERIC);
}

CV_GpuMatOpCopyTo CV_GpuMatOpCopyTo_test;
