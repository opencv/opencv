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

#include "test_precomp.hpp"
#include "opencv2/core/cuda.hpp"


#ifdef HAVE_CUDA
using std::tr1::get;

// show detection results on input image with cv::imshow
//#define SHOW_DETECTIONS

#if defined SHOW_DETECTIONS
# define SHOW(res)           \
    cv::imshow(#res, res);   \
    cv::waitKey(0);
#else
# define SHOW(res)
#endif

static std::string path(std::string relative)
{
    return cvtest::TS::ptr()->get_data_path() + "cascadeandhog/" + relative;
}

TEST(SCascadeTest, readCascade)
{
    std::string xml = path("cascades/inria_caltech-17.01.2013.xml");
    cv::FileStorage fs(xml, cv::FileStorage::READ);

    cv::softcascade::SCascade cascade;

    ASSERT_TRUE(fs.isOpened());
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));
}

namespace
{
    typedef cv::softcascade::Detection Detection;

    cv::Rect getFromTable(int idx)
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

    void print(std::ostream &out, const Detection& d)
    {
    #if defined SHOW_DETECTIONS
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
    #else
        (void)out; (void)d;
    #endif
    }

    void printTotal(std::ostream &out, int detbytes)
    {
    #if defined SHOW_DETECTIONS
        out << "\x1b[32m[          ]\x1b[0m Total detections " << (detbytes / sizeof(Detection)) << std::endl;
    #else
        (void)out; (void)detbytes;
    #endif
    }

    std::string itoa(long i)
    {
        static char s[65];
        sprintf(s, "%ld", i);
        return std::string(s);
    }

#if defined SHOW_DETECTIONS
    std::string getImageName(int level)
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer [80];

        time ( &rawtime );
        timeinfo = localtime ( &rawtime );

        strftime (buffer,80,"%Y-%m-%d--%H-%M-%S",timeinfo);
        return "gpu_rec_level_" + itoa(level)+ "_" + std::string(buffer) + ".png";
    }

    void writeResult(const cv::Mat& result, const int level)
    {
        std::string path = cv::tempfile(getImageName(level).c_str());
        cv::imwrite(path, result);
        std::cout << "\x1b[32m" << "[          ]" << std::endl << "[ stored in]"<< "\x1b[0m" << path << std::endl;
    }
#endif
}

class SCascadeTestRoi : public ::testing::TestWithParam<std::tr1::tuple<cv::cuda::DeviceInfo, std::string, std::string, int> >
{
    virtual void SetUp()
    {
        cv::cuda::setDevice(get<0>(GetParam()).deviceID());
    }
};

TEST_P(SCascadeTestRoi, Detect)
{
    cv::Mat coloredCpu = cv::imread(path(get<2>(GetParam())));
    ASSERT_FALSE(coloredCpu.empty());

    cv::softcascade::SCascade cascade;

    cv::FileStorage fs(path(get<1>(GetParam())), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::cuda::GpuMat colored(coloredCpu), objectBoxes(1, 16384, CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(0);

    int nroi = get<3>(GetParam());
    cv::Mat result(coloredCpu);
    cv::RNG rng;
    for (int i = 0; i < nroi; ++i)
    {
        cv::Rect r = getFromTable(rng(10));
        cv::cuda::GpuMat sub(rois, r);
        sub.setTo(1);
        cv::rectangle(result, r, cv::Scalar(0, 0, 255, 255), 1);
    }
    objectBoxes.setTo(0);

    cascade.detect(colored, rois, objectBoxes);

    cv::Mat dt(objectBoxes);
    typedef cv::softcascade::Detection Detection;

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

INSTANTIATE_TEST_CASE_P(cuda_accelerated, SCascadeTestRoi, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("cascades/inria_caltech-17.01.2013.xml"),
                    std::string("cascades/sc_cvpr_2012_to_opencv_new_format.xml")),
    testing::Values(std::string("images/image_00000000_0.png")),
    testing::Range(0, 5)));

namespace {

struct Fixture
{
    std::string path;
    int expected;

    Fixture(){}
    Fixture(std::string p, int e): path(p), expected(e) {}
};
}

typedef std::tr1::tuple<cv::cuda::DeviceInfo, Fixture> SCascadeTestAllFixture;
class SCascadeTestAll : public ::testing::TestWithParam<SCascadeTestAllFixture>
{
protected:
    std::string xml;
    int expected;

    virtual void SetUp()
    {
        cv::cuda::setDevice(get<0>(GetParam()).deviceID());
        xml = path(get<1>(GetParam()).path);
        expected = get<1>(GetParam()).expected;
    }
};

TEST_P(SCascadeTestAll, detect)
{
    cv::softcascade::SCascade cascade;

    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat coloredCpu = cv::imread(path("images/image_00000000_0.png"));
    ASSERT_FALSE(coloredCpu.empty());

    cv::cuda::GpuMat colored(coloredCpu), objectBoxes, rois(colored.size(), CV_8UC1);
    rois.setTo(1);

    cascade.detect(colored, rois, objectBoxes);

    typedef cv::softcascade::Detection Detection;
    cv::Mat dt(objectBoxes);


    Detection* dts = ((Detection*)dt.data) + 1;
    int* count = dt.ptr<int>(0);

    printTotal(std::cout, *count);

    for (int i = 0; i  < *count; ++i)
    {
        Detection d = dts[i];
        print(std::cout, d);
        cv::rectangle(coloredCpu, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(255, 0, 0, 255), 1);
    }

    SHOW(coloredCpu);
    ASSERT_EQ(*count, expected);
}

TEST_P(SCascadeTestAll, detectStream)
{
    cv::softcascade::SCascade cascade;

    cv::FileStorage fs(xml, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::Mat coloredCpu = cv::imread(path("images/image_00000000_0.png"));
    ASSERT_FALSE(coloredCpu.empty());

    cv::cuda::GpuMat colored(coloredCpu), objectBoxes(1, 100000, CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(cv::Scalar::all(1));

    cv::cuda::Stream s;

    objectBoxes.setTo(0);
    cascade.detect(colored, rois, objectBoxes, s);
    s.waitForCompletion();

    typedef cv::softcascade::Detection Detection;
    cv::Mat detections(objectBoxes);
    int a = *(detections.ptr<int>(0));
    ASSERT_EQ(a, expected);
}

INSTANTIATE_TEST_CASE_P(cuda_accelerated, SCascadeTestAll, testing::Combine( ALL_DEVICES,
                    testing::Values(Fixture("cascades/inria_caltech-17.01.2013.xml", 7),
                                    Fixture("cascades/sc_cvpr_2012_to_opencv_new_format.xml", 1291))));

#endif
