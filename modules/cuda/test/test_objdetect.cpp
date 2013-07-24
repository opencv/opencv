/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

using namespace cvtest;

//#define DUMP

struct HOG : testing::TestWithParam<cv::cuda::DeviceInfo>, cv::cuda::HOGDescriptor
{
    cv::cuda::DeviceInfo devInfo;

#ifdef DUMP
    std::ofstream f;
#else
    std::ifstream f;
#endif

    int wins_per_img_x;
    int wins_per_img_y;
    int blocks_per_win_x;
    int blocks_per_win_y;
    int block_hist_size;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }

#ifdef DUMP
    void dump(const cv::Mat& blockHists, const std::vector<cv::Point>& locations)
    {
        f.write((char*)&blockHists.rows, sizeof(blockHists.rows));
        f.write((char*)&blockHists.cols, sizeof(blockHists.cols));

        for (int i = 0; i < blockHists.rows; ++i)
        {
            for (int j = 0; j < blockHists.cols; ++j)
            {
                float val = blockHists.at<float>(i, j);
                f.write((char*)&val, sizeof(val));
            }
        }

        int nlocations = locations.size();
        f.write((char*)&nlocations, sizeof(nlocations));

        for (int i = 0; i < locations.size(); ++i)
            f.write((char*)&locations[i], sizeof(locations[i]));
    }
#else
    void compare(const cv::Mat& blockHists, const std::vector<cv::Point>& locations)
    {
        int rows, cols;
        f.read((char*)&rows, sizeof(rows));
        f.read((char*)&cols, sizeof(cols));
        ASSERT_EQ(rows, blockHists.rows);
        ASSERT_EQ(cols, blockHists.cols);

        for (int i = 0; i < blockHists.rows; ++i)
        {
            for (int j = 0; j < blockHists.cols; ++j)
            {
                float val;
                f.read((char*)&val, sizeof(val));
                ASSERT_NEAR(val, blockHists.at<float>(i, j), 1e-3);
            }
        }

        int nlocations;
        f.read((char*)&nlocations, sizeof(nlocations));
        ASSERT_EQ(nlocations, static_cast<int>(locations.size()));

        for (int i = 0; i < nlocations; ++i)
        {
            cv::Point location;
            f.read((char*)&location, sizeof(location));
            ASSERT_EQ(location, locations[i]);
        }
    }
#endif

    void testDetect(const cv::Mat& img)
    {
        gamma_correction = false;
        setSVMDetector(cv::cuda::HOGDescriptor::getDefaultPeopleDetector());

        std::vector<cv::Point> locations;

        // Test detect
        detect(loadMat(img), locations, 0);

#ifdef DUMP
        dump(cv::Mat(block_hists), locations);
#else
        compare(cv::Mat(block_hists), locations);
#endif

        // Test detect on smaller image
        cv::Mat img2;
        cv::resize(img, img2, cv::Size(img.cols / 2, img.rows / 2));
        detect(loadMat(img2), locations, 0);

#ifdef DUMP
        dump(cv::Mat(block_hists), locations);
#else
        compare(cv::Mat(block_hists), locations);
#endif

        // Test detect on greater image
        cv::resize(img, img2, cv::Size(img.cols * 2, img.rows * 2));
        detect(loadMat(img2), locations, 0);

#ifdef DUMP
        dump(cv::Mat(block_hists), locations);
#else
        compare(cv::Mat(block_hists), locations);
#endif
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
                    ASSERT_FLOAT_EQ(a, b);
                }
    }
};

// desabled while resize does not fixed
CUDA_TEST_P(HOG, Detect)
{
    cv::Mat img_rgb = readImage("hog/road.png");
    ASSERT_FALSE(img_rgb.empty());

#ifdef DUMP
    f.open((std::string(cvtest::TS::ptr()->get_data_path()) + "hog/expected_output.bin").c_str(), std::ios_base::binary);
    ASSERT_TRUE(f.is_open());
#else
    f.open((std::string(cvtest::TS::ptr()->get_data_path()) + "hog/expected_output.bin").c_str(), std::ios_base::binary);
    ASSERT_TRUE(f.is_open());
#endif

    // Test on color image
    cv::Mat img;
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    testDetect(img);

    // Test on gray image
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2GRAY);
    testDetect(img);

    f.close();
}

CUDA_TEST_P(HOG, GetDescriptors)
{
    // Load image (e.g. train data, composed from windows)
    cv::Mat img_rgb = readImage("hog/train_data.png");
    ASSERT_FALSE(img_rgb.empty());

    // Convert to C4
    cv::Mat img;
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);

    cv::cuda::GpuMat d_img(img);

    // Convert train images into feature vectors (train table)
    cv::cuda::GpuMat descriptors, descriptors_by_cols;
    getDescriptors(d_img, win_size, descriptors, DESCR_FORMAT_ROW_BY_ROW);
    getDescriptors(d_img, win_size, descriptors_by_cols, DESCR_FORMAT_COL_BY_COL);

    // Check size of the result train table
    wins_per_img_x = 3;
    wins_per_img_y = 2;
    blocks_per_win_x = 7;
    blocks_per_win_y = 15;
    block_hist_size = 36;
    cv::Size descr_size_expected = cv::Size(blocks_per_win_x * blocks_per_win_y * block_hist_size,
                                            wins_per_img_x * wins_per_img_y);
    ASSERT_EQ(descr_size_expected, descriptors.size());

    // Check both formats of output descriptors are handled correctly
    cv::Mat dr(descriptors);
    cv::Mat dc(descriptors_by_cols);
    for (int i = 0; i < wins_per_img_x * wins_per_img_y; ++i)
    {
        const float* l = dr.rowRange(i, i + 1).ptr<float>();
        const float* r = dc.rowRange(i, i + 1).ptr<float>();
        for (int y = 0; y < blocks_per_win_y; ++y)
            for (int x = 0; x < blocks_per_win_x; ++x)
                for (int k = 0; k < block_hist_size; ++k)
                    ASSERT_EQ(l[(y * blocks_per_win_x + x) * block_hist_size + k],
                              r[(x * blocks_per_win_y + y) * block_hist_size + k]);
    }

    /* Now we want to extract the same feature vectors, but from single images. NOTE: results will
    be defferent, due to border values interpolation. Using of many small images is slower, however we
    wont't call getDescriptors and will use computeBlockHistograms instead of. computeBlockHistograms
    works good, it can be checked in the gpu_hog sample */

    img_rgb = readImage("hog/positive1.png");
    ASSERT_TRUE(!img_rgb.empty());
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    computeBlockHistograms(cv::cuda::GpuMat(img));
    // Everything is fine with interpolation for left top subimage
    ASSERT_EQ(0.0, cv::norm((cv::Mat)block_hists, (cv::Mat)descriptors.rowRange(0, 1)));

    img_rgb = readImage("hog/positive2.png");
    ASSERT_TRUE(!img_rgb.empty());
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    computeBlockHistograms(cv::cuda::GpuMat(img));
    compare_inner_parts(cv::Mat(block_hists), cv::Mat(descriptors.rowRange(1, 2)));

    img_rgb = readImage("hog/negative1.png");
    ASSERT_TRUE(!img_rgb.empty());
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    computeBlockHistograms(cv::cuda::GpuMat(img));
    compare_inner_parts(cv::Mat(block_hists), cv::Mat(descriptors.rowRange(2, 3)));

    img_rgb = readImage("hog/negative2.png");
    ASSERT_TRUE(!img_rgb.empty());
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    computeBlockHistograms(cv::cuda::GpuMat(img));
    compare_inner_parts(cv::Mat(block_hists), cv::Mat(descriptors.rowRange(3, 4)));

    img_rgb = readImage("hog/positive3.png");
    ASSERT_TRUE(!img_rgb.empty());
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    computeBlockHistograms(cv::cuda::GpuMat(img));
    compare_inner_parts(cv::Mat(block_hists), cv::Mat(descriptors.rowRange(4, 5)));

    img_rgb = readImage("hog/negative3.png");
    ASSERT_TRUE(!img_rgb.empty());
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    computeBlockHistograms(cv::cuda::GpuMat(img));
    compare_inner_parts(cv::Mat(block_hists), cv::Mat(descriptors.rowRange(5, 6)));
}

INSTANTIATE_TEST_CASE_P(CUDA_ObjDetect, HOG, ALL_DEVICES);

//============== caltech hog tests =====================//

struct CalTech : public ::testing::TestWithParam<std::tr1::tuple<cv::cuda::DeviceInfo, std::string> >
{
    cv::cuda::DeviceInfo devInfo;
    cv::Mat img;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());

        img = readImage(GET_PARAM(1), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty());
    }
};

CUDA_TEST_P(CalTech, HOG)
{
    cv::cuda::GpuMat d_img(img);
    cv::Mat markedImage(img.clone());

    cv::cuda::HOGDescriptor d_hog;
    d_hog.setSVMDetector(cv::cuda::HOGDescriptor::getDefaultPeopleDetector());
    d_hog.nlevels = d_hog.nlevels + 32;

    std::vector<cv::Rect> found_locations;
    d_hog.detectMultiScale(d_img, found_locations);

#if defined (LOG_CASCADE_STATISTIC)
    for (int i = 0; i < (int)found_locations.size(); i++)
    {
        cv::Rect r = found_locations[i];

        std::cout << r.x << " " << r.y  << " " << r.width << " " << r.height << std::endl;
        cv::rectangle(markedImage, r , CV_RGB(255, 0, 0));
    }

    cv::imshow("Res", markedImage); cv::waitKey();
#endif
}

INSTANTIATE_TEST_CASE_P(detect, CalTech, testing::Combine(ALL_DEVICES,
    ::testing::Values<std::string>("caltech/image_00000009_0.png", "caltech/image_00000032_0.png",
        "caltech/image_00000165_0.png", "caltech/image_00000261_0.png", "caltech/image_00000469_0.png",
        "caltech/image_00000527_0.png", "caltech/image_00000574_0.png")));




//////////////////////////////////////////////////////////////////////////////////////////
/// LBP classifier

PARAM_TEST_CASE(LBP_Read_classifier, cv::cuda::DeviceInfo, int)
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(LBP_Read_classifier, Accuracy)
{
    cv::cuda::CascadeClassifier_CUDA classifier;
    std::string classifierXmlPath = std::string(cvtest::TS::ptr()->get_data_path()) + "lbpcascade/lbpcascade_frontalface.xml";
    ASSERT_TRUE(classifier.load(classifierXmlPath));
}

INSTANTIATE_TEST_CASE_P(CUDA_ObjDetect, LBP_Read_classifier,
                        testing::Combine(ALL_DEVICES, testing::Values<int>(0)));


PARAM_TEST_CASE(LBP_classify, cv::cuda::DeviceInfo, int)
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(LBP_classify, Accuracy)
{
    std::string classifierXmlPath = std::string(cvtest::TS::ptr()->get_data_path()) + "lbpcascade/lbpcascade_frontalface.xml";
    std::string imagePath = std::string(cvtest::TS::ptr()->get_data_path()) + "lbpcascade/er.png";

    cv::CascadeClassifier cpuClassifier(classifierXmlPath);
    ASSERT_FALSE(cpuClassifier.empty());

    cv::Mat image = cv::imread(imagePath);
    image = image.colRange(0, image.cols/2);
    cv::Mat grey;
    cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    ASSERT_FALSE(image.empty());

    std::vector<cv::Rect> rects;
    cpuClassifier.detectMultiScale(grey, rects);
    cv::Mat markedImage = image.clone();

    std::vector<cv::Rect>::iterator it = rects.begin();
    for (; it != rects.end(); ++it)
        cv::rectangle(markedImage, *it, cv::Scalar(255, 0, 0));

    cv::cuda::CascadeClassifier_CUDA gpuClassifier;
    ASSERT_TRUE(gpuClassifier.load(classifierXmlPath));

    cv::cuda::GpuMat gpu_rects;
    cv::cuda::GpuMat tested(grey);
    int count = gpuClassifier.detectMultiScale(tested, gpu_rects);

#if defined (LOG_CASCADE_STATISTIC)
    cv::Mat downloaded(gpu_rects);
    const cv::Rect* faces = downloaded.ptr<cv::Rect>();
    for (int i = 0; i < count; i++)
    {
        cv::Rect r = faces[i];

        std::cout << r.x << " " << r.y  << " " << r.width << " " << r.height << std::endl;
        cv::rectangle(markedImage, r , CV_RGB(255, 0, 0));
    }
#endif

#if defined (LOG_CASCADE_STATISTIC)
    cv::imshow("Res", markedImage); cv::waitKey();
#endif
    (void)count;
}

INSTANTIATE_TEST_CASE_P(CUDA_ObjDetect, LBP_classify,
                        testing::Combine(ALL_DEVICES, testing::Values<int>(0)));

#endif // HAVE_CUDA
