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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#include <test_precomp.hpp>
#include <time.h>

#ifdef HAVE_CUDA
using cv::gpu::GpuMat;

// show detection results on input image with cv::imshow
#define SHOW_DETECTIONS

#if defined SHOW_DETECTIONS
# define SHOW(res)           \
    cv::imshow(#res, result);\
    cv::waitKey(0);
#else
# define SHOW(res)
#endif

#define GPU_TEST_P(fixture, name, params)                         \
    class fixture##_##name : public fixture {                     \
     public:                                                      \
      fixture##_##name() {}                                       \
     protected:                                                   \
      virtual void body();                                        \
    };                                                            \
    TEST_P(fixture##_##name, name /*none*/){ body();}             \
    INSTANTIATE_TEST_CASE_P(/*none*/, fixture##_##name, params);  \
    void fixture##_##name::body()

namespace {

    typedef cv::gpu::SCascade::Detection Detection;

    static cv::Rect getFromTable(int idx)
    {
        static const cv::Rect rois[] =
        {
            cv::Rect( 65 * 4,  20 * 4,  35 * 4, 80 * 4),
            cv::Rect( 95 * 4,  35 * 4,  45 * 4, 40 * 4),
            cv::Rect( 45 * 4,  35 * 4,  45 * 4, 40 * 4),
            cv::Rect( 25 * 4,  27 * 4,  50 * 4, 45 * 4),
            cv::Rect(100 * 4,  50 * 4,  45 * 4, 40 * 4),

            cv::Rect( 60 * 4,  30 * 4,  45 * 4, 40 * 4),
            cv::Rect( 40 * 4,  55 * 4,  50 * 4, 40 * 4),
            cv::Rect( 48 * 4,  37 * 4,  72 * 4, 80 * 4),
            cv::Rect( 48 * 4,  32 * 4,  85 * 4, 58 * 4),
            cv::Rect( 48 * 4,   0 * 4,  32 * 4, 27 * 4)
        };

        return rois[idx];
    }

    static std::string itoa(long i)
    {
        static char s[65];
        sprintf(s, "%ld", i);
        return std::string(s);
    }

    static std::string getImageName(int level)
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer [80];

        time ( &rawtime );
        timeinfo = localtime ( &rawtime );

        strftime (buffer,80,"%Y-%m-%d--%H-%M-%S",timeinfo);
        return "gpu_rec_level_" + itoa(level)+ "_" + std::string(buffer) + ".png";
    }

    static void print(std::ostream &out, const Detection& d)
    {
        out << "\x1b[32m[ detection]\x1b[0m ("
            << std::setw(4)  << d.x
            << " "
            << std::setw(4)  << d.y
            << ") ("
            << std::setw(4)  << d.w
            << " "
            << std::setw(4)  << d.h
            << ") "
            << std::setw(12) << d.confidence
            <<  std::endl;
    }

    static void printTotal(std::ostream &out, int detbytes)
    {
        out << "\x1b[32m[          ]\x1b[0m Total detections " << (detbytes / sizeof(Detection)) << std::endl;
    }

    static void writeResult(const cv::Mat& result, const int level)
    {
        std::string path = cv::tempfile(getImageName(level).c_str());
        cv::imwrite(path, result);
        std::cout << "\x1b[32m" << "[          ]" << std::endl << "[ stored in]"<< "\x1b[0m" << path << std::endl;
    }
}

typedef ::testing::TestWithParam<std::tr1::tuple<cv::gpu::DeviceInfo, std::string, std::string, int> > SCascadeTestRoi;
GPU_TEST_P(SCascadeTestRoi, detect,
    testing::Combine(
        ALL_DEVICES,
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("../cv/cascadeandhog/bahnhof/image_00000000_0.png")),
        testing::Range(0, 5)))
{
    cv::gpu::setDevice(GET_PARAM(0).deviceID());
    cv::Mat coloredCpu = cv::imread(cvtest::TS::ptr()->get_data_path() + GET_PARAM(2));
    ASSERT_FALSE(coloredCpu.empty());

    cv::gpu::SCascade cascade;

    cv::FileStorage fs(perf::TestBase::getDataPath(GET_PARAM(1)), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    GpuMat colored(coloredCpu), objectBoxes(1, 16384, CV_8UC1), rois(colored.size(), CV_8UC1), trois;
    rois.setTo(0);

    int nroi = GET_PARAM(3);
    cv::Mat result(coloredCpu);
    cv::RNG rng;
    for (int i = 0; i < nroi; ++i)
    {
        cv::Rect r = getFromTable(rng(10));
        GpuMat sub(rois, r);
        sub.setTo(1);
        cv::rectangle(result, r, cv::Scalar(0, 0, 255, 255), 1);
    }
    objectBoxes.setTo(0);
    cascade.genRoi(rois, trois);
    cascade.detect(colored, trois, objectBoxes);

    cv::Mat dt(objectBoxes);
    typedef cv::gpu::SCascade::Detection Detection;

    Detection* dts = ((Detection*)dt.data) + 1;
    int* count = dt.ptr<int>(0);

    printTotal(std::cout, *count);

    for (int i = 0; i  < *count; ++i)
    {
        Detection d = dts[i];
        print(std::cout, d);
        cv::rectangle(result, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(255, 0, 0, 255), 1);
    }

    SHOW(result);

}

typedef ::testing::TestWithParam<std::tr1::tuple<cv::gpu::DeviceInfo, std::string, std::string, int> > SCascadeTestLevel;
GPU_TEST_P(SCascadeTestLevel, detect,
        testing::Combine(
        ALL_DEVICES,
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("../cv/cascadeandhog/bahnhof/image_00000000_0.png")),
        testing::Range(0, 47)
        ))
{
    cv::gpu::setDevice(GET_PARAM(0).deviceID());

    cv::gpu::SCascade cascade;

    cv::FileStorage fs(perf::TestBase::getDataPath(GET_PARAM(1)), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat coloredCpu = cv::imread(cvtest::TS::ptr()->get_data_path() + GET_PARAM(2));
    ASSERT_FALSE(coloredCpu.empty());

    typedef cv::gpu::SCascade::Detection Detection;
    GpuMat colored(coloredCpu), objectBoxes(1, 100 * sizeof(Detection), CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(1);

    cv::gpu::GpuMat trois;
    cascade.genRoi(rois, trois);
    objectBoxes.setTo(0);
    int level = GET_PARAM(3);
    cascade.detect(colored, trois, objectBoxes, level);

    cv::Mat dt(objectBoxes);

    Detection* dts = ((Detection*)dt.data) + 1;
    int* count = dt.ptr<int>(0);

    cv::Mat result(coloredCpu);

    printTotal(std::cout, *count);
    for (int i = 0; i  < *count; ++i)
    {
        Detection d = dts[i];
        print(std::cout, d);
        cv::rectangle(result, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(255, 0, 0, 255), 1);
    }

    writeResult(result, level);
    SHOW(result);
}

TEST(SCascadeTest, readCascade)
{
    std::string xml = cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/icf-template.xml";
    cv::gpu::SCascade cascade;

    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));
}

typedef ::testing::TestWithParam<cv::gpu::DeviceInfo > SCascadeTestAll;
GPU_TEST_P(SCascadeTestAll, detect,
        ALL_DEVICES
        )
{
    cv::gpu::setDevice(GetParam().deviceID());
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml";
    cv::gpu::SCascade cascade;

    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat coloredCpu = cv::imread(cvtest::TS::ptr()->get_data_path()
        + "../cv/cascadeandhog/bahnhof/image_00000000_0.png");
    ASSERT_FALSE(coloredCpu.empty());

    GpuMat colored(coloredCpu), objectBoxes(1, 100000, CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(0);
    GpuMat sub(rois, cv::Rect(rois.cols / 4, rois.rows / 4,rois.cols / 2, rois.rows / 2));
    sub.setTo(cv::Scalar::all(1));

    cv::gpu::GpuMat trois;
    cascade.genRoi(rois, trois);
    objectBoxes.setTo(0);
    cascade.detect(colored, trois, objectBoxes);

    typedef cv::gpu::SCascade::Detection Detection;
    cv::Mat detections(objectBoxes);
    int a = *(detections.ptr<int>(0));
    ASSERT_EQ(a ,2460);
}

GPU_TEST_P(SCascadeTestAll, detectOnIntegral,
        ALL_DEVICES
        )
{
    cv::gpu::setDevice(GetParam().deviceID());
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml";
    cv::gpu::SCascade cascade;

    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    std::string intPath = cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/integrals.xml";
    cv::FileStorage fsi(intPath, cv::FileStorage::READ);
    ASSERT_TRUE(fsi.isOpened());

    GpuMat hogluv(121 * 10, 161, CV_32SC1);
    for (int i = 0; i < 10; ++i)
    {
        cv::Mat channel;
        fsi[std::string("channel") + itoa(i)] >> channel;
        GpuMat gchannel(hogluv, cv::Rect(0, 121 * i, 161, 121));
        gchannel.upload(channel);
    }

    GpuMat objectBoxes(1, 100000, CV_8UC1), rois(cv::Size(640, 480), CV_8UC1);
    rois.setTo(1);

    cv::gpu::GpuMat trois;
    cascade.genRoi(rois, trois);
    objectBoxes.setTo(0);
    cascade.detect(hogluv, trois, objectBoxes);

    typedef cv::gpu::SCascade::Detection Detection;
    cv::Mat detections(objectBoxes);
    int a = *(detections.ptr<int>(0));

    ASSERT_EQ( a ,1024);
}

GPU_TEST_P(SCascadeTestAll, detectStream,
        ALL_DEVICES
        )
{
    cv::gpu::setDevice(GetParam().deviceID());
    std::string xml =  cvtest::TS::ptr()->get_data_path() + "../cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml";
    cv::gpu::SCascade cascade;

    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat coloredCpu = cv::imread(cvtest::TS::ptr()->get_data_path()
        + "../cv/cascadeandhog/bahnhof/image_00000000_0.png");
    ASSERT_FALSE(coloredCpu.empty());

    GpuMat colored(coloredCpu), objectBoxes(1, 100000, CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(0);
    GpuMat sub(rois, cv::Rect(rois.cols / 4, rois.rows / 4,rois.cols / 2, rois.rows / 2));
    sub.setTo(cv::Scalar::all(1));

    cv::gpu::Stream s;

    cv::gpu::GpuMat trois;
    cascade.genRoi(rois, trois, s);
    objectBoxes.setTo(0);
    cascade.detect(colored, trois, objectBoxes, s);

    cudaDeviceSynchronize();

    typedef cv::gpu::SCascade::Detection Detection;
    cv::Mat detections(objectBoxes);
    int a = *(detections.ptr<int>(0));
    ASSERT_EQ(a ,2460);
}


#endif