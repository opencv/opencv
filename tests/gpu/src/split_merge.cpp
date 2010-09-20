#include "gputest.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct CV_MergeTest : public CvTest
{
    CV_MergeTest() : CvTest("GPU-Merge", "merge") {}
    void can_merge(size_t rows, size_t cols);
    void can_merge_submatrixes(size_t rows, size_t cols);
    void run(int);
} merge_test;


void CV_MergeTest::can_merge(size_t rows, size_t cols)
{
    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= CV_64F; ++depth)
        {
            vector<Mat> src;
            for (size_t i = 0; i < num_channels; ++i)
                src.push_back(Mat(rows, cols, depth, Scalar::all(static_cast<double>(i))));
            
            Mat dst(rows, cols, CV_MAKETYPE(depth, num_channels));   

            cv::merge(src, dst);   

            vector<gpu::GpuMat> dev_src;
            for (size_t i = 0; i < num_channels; ++i)
                dev_src.push_back(gpu::GpuMat(src[i]));

            gpu::GpuMat dev_dst(rows, cols, CV_MAKETYPE(depth, num_channels));
            cv::gpu::merge(dev_src, dev_dst); 

            Mat host_dst = dev_dst;

            double err = norm(dst, host_dst, NORM_INF);

            if (err > 1e-3)
            {
                //ts->printf(CvTS::CONSOLE, "\nNorm: %f\n", err);
                //ts->printf(CvTS::CONSOLE, "Depth: %d\n", depth);
                //ts->printf(CvTS::CONSOLE, "Rows: %d\n", rows);
                //ts->printf(CvTS::CONSOLE, "Cols: %d\n", cols);
                //ts->printf(CvTS::CONSOLE, "NumChannels: %d\n", num_channels);
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
}


void CV_MergeTest::can_merge_submatrixes(size_t rows, size_t cols)
{
    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= CV_64F; ++depth)
        {
            vector<Mat> src;
            for (size_t i = 0; i < num_channels; ++i) 
            {
                Mat m(rows * 2, cols * 2, depth, Scalar::all(static_cast<double>(i)));
                src.push_back(m(Range(rows / 2, rows / 2 + rows), Range(cols / 2, cols / 2 + cols)));
            }

            Mat dst(rows, cols, CV_MAKETYPE(depth, num_channels));   

            cv::merge(src, dst);   

            vector<gpu::GpuMat> dev_src;
            for (size_t i = 0; i < num_channels; ++i)
                dev_src.push_back(gpu::GpuMat(src[i]));

            gpu::GpuMat dev_dst(rows, cols, CV_MAKETYPE(depth, num_channels));
            cv::gpu::merge(dev_src, dev_dst);

            Mat host_dst = dev_dst;

            double err = norm(dst, host_dst, NORM_INF);

            if (err > 1e-3)
            {
                //ts->printf(CvTS::CONSOLE, "\nNorm: %f\n", err);
                //ts->printf(CvTS::CONSOLE, "Depth: %d\n", depth);
                //ts->printf(CvTS::CONSOLE, "Rows: %d\n", rows);
                //ts->printf(CvTS::CONSOLE, "Cols: %d\n", cols);
                //ts->printf(CvTS::CONSOLE, "NumChannels: %d\n", num_channels);
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
}


void CV_MergeTest::run(int) 
{
    try
    {
        can_merge(1, 1);
        can_merge(1, 7);
        can_merge(53, 7);
        can_merge_submatrixes(1, 1);
        can_merge_submatrixes(1, 7);
        can_merge_submatrixes(53, 7);
    }
    catch(const cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw;        
    }    
}


struct CV_SplitTest : public CvTest
{
    CV_SplitTest() : CvTest("GPU-Split", "split") {}
    void can_split(size_t rows, size_t cols);    
    void can_split_submatrix(size_t rows, size_t cols);
    void run(int);
} split_test;


void CV_SplitTest::can_split(size_t rows, size_t cols)
{
    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= CV_64F; ++depth)
        {
            Mat src(rows, cols, CV_MAKETYPE(depth, num_channels), Scalar(1.0, 2.0, 3.0, 4.0));   
            vector<Mat> dst;
            cv::split(src, dst);   

            gpu::GpuMat dev_src(src);
            vector<gpu::GpuMat> dev_dst;
            cv::gpu::split(dev_src, dev_dst);

            if (dev_dst.size() != dst.size())
            {
                ts->printf(CvTS::CONSOLE, "Bad output sizes");
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            }

            for (size_t i = 0; i < num_channels; ++i)
            {
                Mat host_dst = dev_dst[i];
                double err = norm(dst[i], host_dst, NORM_INF);

                if (err > 1e-3)
                {
                    //ts->printf(CvTS::CONSOLE, "\nNorm: %f\n", err);
                    //ts->printf(CvTS::CONSOLE, "Depth: %d\n", depth);
                    //ts->printf(CvTS::CONSOLE, "Rows: %d\n", rows);
                    //ts->printf(CvTS::CONSOLE, "Cols: %d\n", cols);
                    //ts->printf(CvTS::CONSOLE, "NumChannels: %d\n", num_channels);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
        }
}



void CV_SplitTest::can_split_submatrix(size_t rows, size_t cols)
{
    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= CV_64F; ++depth)
        {
            Mat src_data(rows * 2, cols * 2, CV_MAKETYPE(depth, num_channels), Scalar(1.0, 2.0, 3.0, 4.0));   
            Mat src(src_data(Range(rows / 2, rows / 2 + rows), Range(cols / 2, cols / 2 + cols)));
            vector<Mat> dst;
            cv::split(src, dst);   

            gpu::GpuMat dev_src(src);
            vector<gpu::GpuMat> dev_dst;
            cv::gpu::split(dev_src, dev_dst);

            if (dev_dst.size() != dst.size())
            {
                ts->printf(CvTS::CONSOLE, "Bad output sizes");
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            }

            for (size_t i = 0; i < num_channels; ++i)
            {
                Mat host_dst = dev_dst[i];
                double err = norm(dst[i], host_dst, NORM_INF);

                if (err > 1e-3)
                {
                    //ts->printf(CvTS::CONSOLE, "\nNorm: %f\n", err);
                    //ts->printf(CvTS::CONSOLE, "Depth: %d\n", depth);
                    //ts->printf(CvTS::CONSOLE, "Rows: %d\n", rows);
                    //ts->printf(CvTS::CONSOLE, "Cols: %d\n", cols);
                    //ts->printf(CvTS::CONSOLE, "NumChannels: %d\n", num_channels);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
        }
}


void CV_SplitTest::run(int)
{
    try 
    {
        can_split(1, 1);
        can_split(1, 7);
        can_split(7, 53);
        can_split_submatrix(1, 1);
        can_split_submatrix(1, 7);
        can_split_submatrix(7, 53);
    }
    catch(const cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw;        
    }    
}


struct CV_SplitMergeTest : public CvTest
{
    CV_SplitMergeTest() : CvTest("GPU-SplitMerge", "split merge") {}
    void can_split_merge(size_t rows, size_t cols);    
    void run(int);
} split_merge_test;


void CV_SplitMergeTest::can_split_merge(size_t rows, size_t cols) {
    for (size_t num_channels = 1; num_channels <= 4; ++num_channels)
        for (size_t depth = CV_8U; depth <= CV_64F; ++depth)
        {
            Mat orig(rows, cols, CV_MAKETYPE(depth, num_channels), Scalar(1.0, 2.0, 3.0, 4.0));   
            gpu::GpuMat dev_orig(orig);
            vector<gpu::GpuMat> dev_vec;
            cv::gpu::split(dev_orig, dev_vec);

            gpu::GpuMat dev_final(rows, cols, CV_MAKETYPE(depth, num_channels));
            cv::gpu::merge(dev_vec, dev_final);

            double err = cv::norm((Mat)dev_orig, (Mat)dev_final, NORM_INF);
            if (err > 1e-3)
            {
                //ts->printf(CvTS::CONSOLE, "\nNorm: %f\n", err);
                //ts->printf(CvTS::CONSOLE, "Depth: %d\n", depth);
                //ts->printf(CvTS::CONSOLE, "Rows: %d\n", rows);
                //ts->printf(CvTS::CONSOLE, "Cols: %d\n", cols);
                //ts->printf(CvTS::CONSOLE, "NumChannels: %d\n", num_channels);
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
}


void CV_SplitMergeTest::run(int) 
{
    try 
    {
        can_split_merge(1, 1);
        can_split_merge(1, 7);
        can_split_merge(7, 53);
    }
    catch(const cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw;        
    }    
}
