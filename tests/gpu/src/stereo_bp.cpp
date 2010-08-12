#include "gputest.hpp"
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

class CV_GpuStereoBP : public CvTest
{
    public:
        CV_GpuStereoBP();
    protected:
        void run(int);
};

CV_GpuStereoBP::CV_GpuStereoBP(): CvTest( "GPU-StereoBP", "StereoBP" ){}

void CV_GpuStereoBP::run(int )
{
        cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");
        cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-R.png");
        cv::Mat img_template = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-disp.png", 0);

        cv::gpu::GpuMat disp;
        cv::gpu::StereoBeliefPropagation bpm(128, 8, 4, 25, 0.1f, 15, 1);

        bpm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), disp);

        //cv::imwrite(std::string(ts->get_data_path()) + "stereobp/aloe-disp.png", disp);

        disp.convertTo(disp, img_template.type());

        double norm = cv::norm(disp, img_template, cv::NORM_INF);
        ts->set_failed_test_info((norm < 0.5) ? CvTS::OK : CvTS::FAIL_GENERIC);
}


CV_GpuStereoBP CV_GpuStereoBP_test;
