#include "gputest.hpp"
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

class CV_GpuMeanShift : public CvTest
{
    public:
        CV_GpuMeanShift();
    protected:
        void run(int);
};

CV_GpuMeanShift::CV_GpuMeanShift(): CvTest( "GPU-MeanShift", "meanshift" ){}

void CV_GpuMeanShift::run(int )
{
        int spatialRad = 30;
        int colorRad = 30;

        cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "meanshift/con.png");
        cv::Mat img_template = cv::imread(std::string(ts->get_data_path()) + "meanshift/con_result.png");

        cv::Mat rgba;
        cvtColor(img, rgba, CV_BGR2BGRA);

        cv::gpu::GpuMat res;

        cv::gpu::meanShiftFiltering_GPU( cv::gpu::GpuMat(rgba), res, spatialRad, colorRad );

        double norm = cv::norm(res, img_template, cv::NORM_INF);

        ts->set_failed_test_info((norm < 0.5) ? CvTS::OK : CvTS::FAIL_GENERIC);
}


CV_GpuMeanShift CV_GpuMeanShift_test;
