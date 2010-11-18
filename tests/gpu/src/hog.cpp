/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "gputest.hpp"
#include <fstream>

using namespace std;

//#define DUMP

#define CHECK(pred, err) if (!(pred)) { \
    ts->printf(CvTS::LOG, "Fail: \"%s\" at line: %d\n", #pred, __LINE__); \
    ts->set_failed_test_info(err); \
    return; }

struct CV_GpuHogDetectionTest: public CvTest 
{
    CV_GpuHogDetectionTest(): CvTest( "GPU-HOG-detect", "HOGDescriptorDetection" ) {}

    void run(int) 
    {       
        try 
        {
            cv::Mat img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/road.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);

#ifdef DUMP
            f.open((std::string(ts->get_data_path()) + "hog/expected_output.bin").c_str(), std::ios_base::binary);
            CHECK(f.is_open(), CvTS::FAIL_GENERIC);          
#else
            f.open((std::string(ts->get_data_path()) + "hog/expected_output.bin").c_str(), std::ios_base::binary);
            CHECK(f.is_open(), CvTS::FAIL_MISSING_TEST_DATA);          
#endif
            
            // Test on color image
            cv::Mat img;
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            test(img);

            // Test on gray image
            cv::cvtColor(img_rgb, img, CV_BGR2GRAY);
            test(img);

            f.close();
        }
        catch (const cv::Exception& e)
        {
            f.close();
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

#ifdef DUMP
    void dump(const cv::Mat& block_hists, const std::vector<cv::Point>& locations) 
    {
        f.write((char*)&block_hists.rows, sizeof(block_hists.rows));
        f.write((char*)&block_hists.cols, sizeof(block_hists.cols));
        for (int i = 0; i < block_hists.rows; ++i)
        {
            for (int j = 0; j < block_hists.cols; ++j)
            {
                float val = block_hists.at<float>(i, j);
                f.write((char*)&val, sizeof(val));
            }
        }
        size_t nlocations = locations.size();
        f.write((char*)&nlocations, sizeof(nlocations));
        for (size_t i = 0; i < locations.size(); ++i)
            f.write((char*)&locations[i], sizeof(locations[i]));
    }
#else
    void compare(const cv::Mat& block_hists, const std::vector<cv::Point>& locations) 
    {
        int rows, cols;
        size_t nlocations;

        f.read((char*)&rows, sizeof(rows));
        f.read((char*)&cols, sizeof(cols));
        CHECK(rows == block_hists.rows, CvTS::FAIL_INVALID_OUTPUT);
        CHECK(cols == block_hists.cols, CvTS::FAIL_INVALID_OUTPUT);
        for (int i = 0; i < block_hists.rows; ++i)
        {
            for (int j = 0; j < block_hists.cols; ++j)
            {
                float val;
                f.read((char*)&val, sizeof(val));
                CHECK(fabs(val - block_hists.at<float>(i, j)) < 1e-3f, CvTS::FAIL_INVALID_OUTPUT);
            }
        }
        f.read((char*)&nlocations, sizeof(nlocations));
        CHECK(nlocations == locations.size(), CvTS::FAIL_INVALID_OUTPUT);
        for (size_t i = 0; i < nlocations; ++i)
        {
            cv::Point location;
            f.read((char*)&location, sizeof(location));
            CHECK(location == locations[i], CvTS::FAIL_INVALID_OUTPUT);
        }
    }
#endif

    void test(const cv::Mat& img) 
    {
        cv::gpu::GpuMat d_img(img);

        cv::gpu::HOGDescriptor hog;
        hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());
        //cpu detector may be updated soon
        //hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

        std::vector<cv::Point> locations;

        // Test detect
        hog.detect(d_img, locations, 0);

#ifdef DUMP
        dump(hog.block_hists, locations);
#else
        compare(hog.block_hists, locations);
#endif

        // Test detect on smaller image
        cv::gpu::GpuMat d_img2;
        cv::gpu::resize(d_img, d_img2, cv::Size(d_img.cols / 2, d_img.rows / 2)); 
        hog.detect(d_img2, locations, 0);

#ifdef DUMP
        dump(hog.block_hists, locations);
#else
        compare(hog.block_hists, locations);
#endif

        // Test detect on greater image
        cv::gpu::resize(d_img, d_img2, cv::Size(d_img.cols * 2, d_img.rows * 2)); 
        hog.detect(d_img2, locations, 0);
        
#ifdef DUMP
        dump(hog.block_hists, locations);
#else
        compare(hog.block_hists, locations);
#endif

        // Test detectMultiScale
        std::vector<cv::Rect> rects;
        size_t nrects;
        hog.detectMultiScale(d_img, rects, 0, cv::Size(8, 8), cv::Size(), 1.05, 2);

#ifdef DUMP
        nrects = rects.size();
        f.write((char*)&nrects, sizeof(nrects));
        for (size_t i = 0; i < rects.size(); ++i)
            f.write((char*)&rects[i], sizeof(rects[i]));
        dump(hog.block_hists, std::vector<cv::Point>());
#else
        f.read((char*)&nrects, sizeof(nrects));
        CHECK(nrects == rects.size(), CvTS::FAIL_INVALID_OUTPUT)
        for (size_t i = 0; i < rects.size(); ++i) 
        {
            cv::Rect rect;
            f.read((char*)&rect, sizeof(rect));
            CHECK(rect == rects[i], CvTS::FAIL_INVALID_OUTPUT);
        }
        compare(hog.block_hists, std::vector<cv::Point>());
#endif
    }

#ifdef DUMP
    std::ofstream f;
#else
    std::ifstream f;
#endif

} gpu_hog_detection_test;


struct CV_GpuHogGetDescriptorsTest: public CvTest 
{
    CV_GpuHogGetDescriptorsTest(): CvTest("GPU-HOG-getDescriptors", "HOGDescriptorGetDescriptors") {}

    void run(int)
    {
        try 
        {
            // Load image (e.g. train data, composed from windows)
            cv::Mat img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/train_data.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);

            // Convert to C4
            cv::Mat img;
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            cv::gpu::GpuMat d_img(img);

            cv::Size win_size(64, 128);
            cv::gpu::HOGDescriptor hog(win_size);

            // Convert train images into feature vectors (train table)
            cv::gpu::GpuMat descriptors;
            hog.getDescriptors(d_img, win_size, descriptors);

            // Check size of the result train table
            wins_per_img_x = 3;
            wins_per_img_y = 2;
            blocks_per_win_x = 7;
            blocks_per_win_y = 15;
            block_hist_size = 36;
            cv::Size descr_size_expected = cv::Size(blocks_per_win_x * blocks_per_win_y * block_hist_size, 
                                                    wins_per_img_x * wins_per_img_y);                                                
            CHECK(descriptors.size() == descr_size_expected, CvTS::FAIL_INVALID_OUTPUT);

            /* Now we want to extract the same feature vectors, but from single images. NOTE: results will 
            be defferent, due to border values interpolation. Using of many small images is slower, however we 
            wont't call getDescriptors and will use computeBlockHistograms instead of. computeBlockHistograms 
            works good, it can be checked in the gpu_hog sample */

            img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/positive1.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            hog.computeBlockHistograms(cv::gpu::GpuMat(img));
            // Everything is fine with interpolation for left top subimage
            CHECK(cv::norm(hog.block_hists, descriptors.rowRange(0, 1)) == 0.f, CvTS::FAIL_INVALID_OUTPUT);

            img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/positive2.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            hog.computeBlockHistograms(cv::gpu::GpuMat(img));
            compare_inner_parts(hog.block_hists, descriptors.rowRange(1, 2));

            img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/negative1.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            hog.computeBlockHistograms(cv::gpu::GpuMat(img));
            compare_inner_parts(hog.block_hists, descriptors.rowRange(2, 3));

            img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/negative2.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            hog.computeBlockHistograms(cv::gpu::GpuMat(img));
            compare_inner_parts(hog.block_hists, descriptors.rowRange(3, 4));

            img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/positive3.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            hog.computeBlockHistograms(cv::gpu::GpuMat(img));
            compare_inner_parts(hog.block_hists, descriptors.rowRange(4, 5));

            img_rgb = cv::imread(std::string(ts->get_data_path()) + "hog/negative3.png");
            CHECK(!img_rgb.empty(), CvTS::FAIL_MISSING_TEST_DATA);
            cv::cvtColor(img_rgb, img, CV_BGR2BGRA);
            hog.computeBlockHistograms(cv::gpu::GpuMat(img));
            compare_inner_parts(hog.block_hists, descriptors.rowRange(5, 6));
        }
        catch (const cv::Exception& e)
        {
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    // Does not compare border value, as interpolation leads to delta
    void compare_inner_parts(cv::Mat d1, cv::Mat d2)
    {
        for (int i = 1; i < blocks_per_win_y - 1; ++i)
            for (int j = 1; j < blocks_per_win_x - 1; ++j)
                for (int k = 0; k < block_hist_size; ++k)
                {
                    float a = d1.at<float>(0, (i * blocks_per_win_x + j) * block_hist_size);
                    float b = d2.at<float>(0, (i * blocks_per_win_x + j) * block_hist_size);
                    CHECK(a == b, CvTS::FAIL_INVALID_OUTPUT)
                }
    }

    int wins_per_img_x;
    int wins_per_img_y;
    int blocks_per_win_x;
    int blocks_per_win_y;
    int block_hist_size;
} gpu_hog_get_descriptors_test;

