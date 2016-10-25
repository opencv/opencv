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

struct HOG : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;
    cv::Ptr<cv::cuda::HOG> hog;

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

        hog = cv::cuda::HOG::create();
    }

#ifdef DUMP
    void dump(const std::vector<cv::Point>& locations)
    {
        int nlocations = locations.size();
        f.write((char*)&nlocations, sizeof(nlocations));

        for (int i = 0; i < locations.size(); ++i)
            f.write((char*)&locations[i], sizeof(locations[i]));
    }
#else
    void compare(const std::vector<cv::Point>& locations)
    {
        // skip block_hists check
        int rows, cols;
        f.read((char*)&rows, sizeof(rows));
        f.read((char*)&cols, sizeof(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                float val;
                f.read((char*)&val, sizeof(val));
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
        hog->setGammaCorrection(false);
        hog->setSVMDetector(hog->getDefaultPeopleDetector());

        std::vector<cv::Point> locations;

        // Test detect
        hog->detect(loadMat(img), locations);

#ifdef DUMP
        dump(locations);
#else
        compare(locations);
#endif

        // Test detect on smaller image
        cv::Mat img2;
        cv::resize(img, img2, cv::Size(img.cols / 2, img.rows / 2));
        hog->detect(loadMat(img2), locations);

#ifdef DUMP
        dump(locations);
#else
        compare(locations);
#endif

        // Test detect on greater image
        cv::resize(img, img2, cv::Size(img.cols * 2, img.rows * 2));
        hog->detect(loadMat(img2), locations);

#ifdef DUMP
        dump(locations);
#else
        compare(locations);
#endif
    }
};

// desabled while resize does not fixed
CUDA_TEST_P(HOG, DISABLED_Detect)
{
    cv::Mat img_rgb = readImage("hog/road.png");
    ASSERT_FALSE(img_rgb.empty());

    f.open((std::string(cvtest::TS::ptr()->get_data_path()) + "hog/expected_output.bin").c_str(), std::ios_base::binary);
    ASSERT_TRUE(f.is_open());

    // Test on color image
    cv::Mat img;
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);
    testDetect(img);

    // Test on gray image
    cv::cvtColor(img_rgb, img, cv::COLOR_BGR2GRAY);
    testDetect(img);
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

    hog->setWinStride(Size(64, 128));

    hog->setDescriptorFormat(cv::cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
    hog->compute(d_img, descriptors);

    hog->setDescriptorFormat(cv::cuda::HOG::DESCR_FORMAT_COL_BY_COL);
    hog->compute(d_img, descriptors_by_cols);

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
}
/*
INSTANTIATE_TEST_CASE_P(CUDA_ObjDetect, HOG, ALL_DEVICES);
*/
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

    cv::Ptr<cv::cuda::HOG> d_hog = cv::cuda::HOG::create();
    d_hog->setSVMDetector(d_hog->getDefaultPeopleDetector());
    d_hog->setNumLevels(d_hog->getNumLevels() + 32);

    std::vector<cv::Rect> found_locations;
    d_hog->detectMultiScale(d_img, found_locations);

#if defined (LOG_CASCADE_STATISTIC)
    for (int i = 0; i < (int)found_locations.size(); i++)
    {
        cv::Rect r = found_locations[i];

        std::cout << r.x << " " << r.y  << " " << r.width << " " << r.height << std::endl;
        cv::rectangle(markedImage, r , CV_RGB(255, 0, 0));
    }

    cv::imshow("Res", markedImage);
    cv::waitKey();
#endif
}

INSTANTIATE_TEST_CASE_P(detect, CalTech, testing::Combine(ALL_DEVICES,
    ::testing::Values<std::string>("caltech/image_00000009_0.png", "caltech/image_00000032_0.png",
        "caltech/image_00000165_0.png", "caltech/image_00000261_0.png", "caltech/image_00000469_0.png",
        "caltech/image_00000527_0.png", "caltech/image_00000574_0.png")));


//------------------------variable GPU HOG Tests------------------------//
struct Hog_var : public ::testing::TestWithParam<std::tr1::tuple<cv::cuda::DeviceInfo, std::string> >
{
    cv::cuda::DeviceInfo devInfo;
    cv::Mat img, c_img;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());

        cv::Rect roi(0, 0, 16, 32);
        img = readImage(GET_PARAM(1), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty());
        c_img = img(roi);
    }
};

CUDA_TEST_P(Hog_var, HOG)
{
    cv::cuda::GpuMat _img(c_img);
    cv::cuda::GpuMat d_img;

    int win_stride_width = 8;int win_stride_height = 8;
    int win_width = 16;
    int block_width = 8;
    int block_stride_width = 4;int block_stride_height = 4;
    int cell_width = 4;
    int nbins = 9;

    Size win_stride(win_stride_width, win_stride_height);
    Size win_size(win_width, win_width * 2);
    Size block_size(block_width, block_width);
    Size block_stride(block_stride_width, block_stride_height);
    Size cell_size(cell_width, cell_width);

    cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, nbins);

    gpu_hog->setNumLevels(13);
    gpu_hog->setHitThreshold(0);
    gpu_hog->setWinStride(win_stride);
    gpu_hog->setScaleFactor(1.05);
    gpu_hog->setGroupThreshold(8);
    gpu_hog->compute(_img, d_img);

    vector<float> gpu_desc_vec;
    ASSERT_TRUE(gpu_desc_vec.empty());
    cv::Mat R(d_img);

    cv::HOGDescriptor cpu_hog(win_size, block_size, block_stride, cell_size, nbins);
    cpu_hog.nlevels = 13;
    vector<float> cpu_desc_vec;
    ASSERT_TRUE(cpu_desc_vec.empty());
    cpu_hog.compute(c_img, cpu_desc_vec, win_stride, Size(0,0));
}

INSTANTIATE_TEST_CASE_P(detect, Hog_var, testing::Combine(ALL_DEVICES,
    ::testing::Values<std::string>("/hog/road.png")));

struct Hog_var_cell : public ::testing::TestWithParam<std::tr1::tuple<cv::cuda::DeviceInfo, std::string> >
{
    cv::cuda::DeviceInfo devInfo;
    cv::Mat img, c_img, c_img2, c_img3, c_img4;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());

        cv::Rect roi(0, 0, 48, 96);
        img = readImage(GET_PARAM(1), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty());
        c_img = img(roi);

        cv::Rect roi2(0, 0, 54, 108);
        c_img2 = img(roi2);

        cv::Rect roi3(0, 0, 64, 128);
        c_img3 = img(roi3);

        cv::Rect roi4(0, 0, 32, 64);
        c_img4 = img(roi4);
    }
};

CUDA_TEST_P(Hog_var_cell, HOG)
{
    cv::cuda::GpuMat _img(c_img);
    cv::cuda::GpuMat _img2(c_img2);
    cv::cuda::GpuMat _img3(c_img3);
    cv::cuda::GpuMat _img4(c_img4);
    cv::cuda::GpuMat d_img;

    ASSERT_FALSE(_img.empty());
    ASSERT_TRUE(d_img.empty());

    int win_stride_width = 8;int win_stride_height = 8;
    int win_width = 48;
    int block_width = 16;
    int block_stride_width = 8;int block_stride_height = 8;
    int cell_width = 8;
    int nbins = 9;

    Size win_stride(win_stride_width, win_stride_height);
    Size win_size(win_width, win_width * 2);
    Size block_size(block_width, block_width);
    Size block_stride(block_stride_width, block_stride_height);
    Size cell_size(cell_width, cell_width);

    cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, nbins);

    gpu_hog->setNumLevels(13);
    gpu_hog->setHitThreshold(0);
    gpu_hog->setWinStride(win_stride);
    gpu_hog->setScaleFactor(1.05);
    gpu_hog->setGroupThreshold(8);
    gpu_hog->compute(_img, d_img);
//------------------------------------------------------------------------------
    cv::cuda::GpuMat d_img2;
    ASSERT_TRUE(d_img2.empty());

    int win_stride_width2 = 8;int win_stride_height2 = 8;
    int win_width2 = 48;
    int block_width2 = 16;
    int block_stride_width2 = 8;int block_stride_height2 = 8;
    int cell_width2 = 4;

    Size win_stride2(win_stride_width2, win_stride_height2);
    Size win_size2(win_width2, win_width2 * 2);
    Size block_size2(block_width2, block_width2);
    Size block_stride2(block_stride_width2, block_stride_height2);
    Size cell_size2(cell_width2, cell_width2);

    cv::Ptr<cv::cuda::HOG> gpu_hog2 = cv::cuda::HOG::create(win_size2, block_size2, block_stride2, cell_size2, nbins);
    gpu_hog2->setWinStride(win_stride2);
    gpu_hog2->compute(_img, d_img2);
//------------------------------------------------------------------------------
    cv::cuda::GpuMat d_img3;
    ASSERT_TRUE(d_img3.empty());

    int win_stride_width3 = 9;int win_stride_height3 = 9;
    int win_width3 = 54;
    int block_width3 = 18;
    int block_stride_width3 = 9;int block_stride_height3 = 9;
    int cell_width3 = 6;

    Size win_stride3(win_stride_width3, win_stride_height3);
    Size win_size3(win_width3, win_width3 * 2);
    Size block_size3(block_width3, block_width3);
    Size block_stride3(block_stride_width3, block_stride_height3);
    Size cell_size3(cell_width3, cell_width3);

    cv::Ptr<cv::cuda::HOG> gpu_hog3 = cv::cuda::HOG::create(win_size3, block_size3, block_stride3, cell_size3, nbins);
    gpu_hog3->setWinStride(win_stride3);
    gpu_hog3->compute(_img2, d_img3);
//------------------------------------------------------------------------------
    cv::cuda::GpuMat d_img4;
    ASSERT_TRUE(d_img4.empty());

    int win_stride_width4 = 16;int win_stride_height4 = 16;
    int win_width4 = 64;
    int block_width4 = 32;
    int block_stride_width4 = 16;int block_stride_height4 = 16;
    int cell_width4 = 8;

    Size win_stride4(win_stride_width4, win_stride_height4);
    Size win_size4(win_width4, win_width4 * 2);
    Size block_size4(block_width4, block_width4);
    Size block_stride4(block_stride_width4, block_stride_height4);
    Size cell_size4(cell_width4, cell_width4);

    cv::Ptr<cv::cuda::HOG> gpu_hog4 = cv::cuda::HOG::create(win_size4, block_size4, block_stride4, cell_size4, nbins);
    gpu_hog4->setWinStride(win_stride4);
    gpu_hog4->compute(_img3, d_img4);
//------------------------------------------------------------------------------
    cv::cuda::GpuMat d_img5;
    ASSERT_TRUE(d_img5.empty());

    int win_stride_width5 = 16;int win_stride_height5 = 16;
    int win_width5 = 64;
    int block_width5 = 32;
    int block_stride_width5 = 16;int block_stride_height5 = 16;
    int cell_width5 = 16;

    Size win_stride5(win_stride_width5, win_stride_height5);
    Size win_size5(win_width5, win_width5 * 2);
    Size block_size5(block_width5, block_width5);
    Size block_stride5(block_stride_width5, block_stride_height5);
    Size cell_size5(cell_width5, cell_width5);

    cv::Ptr<cv::cuda::HOG> gpu_hog5 = cv::cuda::HOG::create(win_size5, block_size5, block_stride5, cell_size5, nbins);
    gpu_hog5->setWinStride(win_stride5);
    gpu_hog5->compute(_img3, d_img5);
//------------------------------------------------------------------------------
}

INSTANTIATE_TEST_CASE_P(detect, Hog_var_cell, testing::Combine(ALL_DEVICES,
    ::testing::Values<std::string>("/hog/road.png")));
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
    std::string classifierXmlPath = std::string(cvtest::TS::ptr()->get_data_path()) + "lbpcascade/lbpcascade_frontalface.xml";

    cv::Ptr<cv::cuda::CascadeClassifier> d_cascade;

    ASSERT_NO_THROW(
        d_cascade = cv::cuda::CascadeClassifier::create(classifierXmlPath);
    );

    ASSERT_FALSE(d_cascade.empty());
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

    cv::Ptr<cv::cuda::CascadeClassifier> gpuClassifier =
            cv::cuda::CascadeClassifier::create(classifierXmlPath);

    cv::cuda::GpuMat tested(grey);
    cv::cuda::GpuMat gpu_rects_buf;
    gpuClassifier->detectMultiScale(tested, gpu_rects_buf);

    std::vector<cv::Rect> gpu_rects;
    gpuClassifier->convert(gpu_rects_buf, gpu_rects);

#if defined (LOG_CASCADE_STATISTIC)
    for (size_t i = 0; i < gpu_rects.size(); i++)
    {
        cv::Rect r = gpu_rects[i];

        std::cout << r.x << " " << r.y  << " " << r.width << " " << r.height << std::endl;
        cv::rectangle(markedImage, r , CV_RGB(255, 0, 0));
    }

    cv::imshow("Res", markedImage);
    cv::waitKey();
#endif
}

INSTANTIATE_TEST_CASE_P(CUDA_ObjDetect, LBP_classify,
                        testing::Combine(ALL_DEVICES, testing::Values<int>(0)));

#endif // HAVE_CUDA
