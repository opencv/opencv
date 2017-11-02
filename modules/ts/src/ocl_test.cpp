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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/ts/ocl_test.hpp"

namespace cvtest {
namespace ocl {

using namespace cv;

int test_loop_times = 1; // TODO Read from command line / environment

#ifdef HAVE_OPENCL

#define DUMP_PROPERTY_XML(propertyName, propertyValue) \
    do { \
        std::stringstream ssName, ssValue;\
        ssName << propertyName;\
        ssValue << (propertyValue); \
        ::testing::Test::RecordProperty(ssName.str(), ssValue.str()); \
    } while (false)

#define DUMP_MESSAGE_STDOUT(msg) \
    do { \
        std::cout << msg << std::endl; \
    } while (false)

static std::string bytesToStringRepr(size_t value)
{
    size_t b = value % 1024;
    value /= 1024;

    size_t kb = value % 1024;
    value /= 1024;

    size_t mb = value % 1024;
    value /= 1024;

    size_t gb = value;

    std::ostringstream stream;

    if (gb > 0)
        stream << gb << " GB ";
    if (mb > 0)
        stream << mb << " MB ";
    if (kb > 0)
        stream << kb << " kB ";
    if (b > 0)
        stream << b << " B";

    return stream.str();
}

void dumpOpenCLDevice()
{
    using namespace cv::ocl;

    try
    {
        if (!useOpenCL())
        {
            DUMP_MESSAGE_STDOUT("OpenCL is disabled");
            DUMP_PROPERTY_XML("cv_ocl", "disabled");
            return;
        }

        std::vector<PlatformInfo> platforms;
        cv::ocl::getPlatfomsInfo(platforms);
        if (platforms.size() > 0)
        {
            DUMP_MESSAGE_STDOUT("OpenCL Platforms: ");
            for (size_t i = 0; i < platforms.size(); i++)
            {
                const PlatformInfo* platform = &platforms[i];
                DUMP_MESSAGE_STDOUT("    " << platform->name().c_str());
                Device current_device;
                for (int j = 0; j < platform->deviceNumber(); j++)
                {
                    platform->getDevice(current_device, j);
                    const char* deviceTypeStr = current_device.type() == Device::TYPE_CPU
                        ? ("CPU") : (current_device.type() == Device::TYPE_GPU ? current_device.hostUnifiedMemory() ? "iGPU" : "dGPU" : "unknown");
                    DUMP_MESSAGE_STDOUT( "        " << deviceTypeStr << ": " << current_device.name().c_str() << " (" << current_device.version().c_str() << ")");
                    DUMP_PROPERTY_XML( cv::format("cv_ocl_platform_%d_device_%d", (int)i, (int)j ),
                        cv::format("(Platform=%s)(Type=%s)(Name=%s)(Version=%s)",
                        platform->name().c_str(), deviceTypeStr, current_device.name().c_str(), current_device.version().c_str()) );
                }
            }
        }
        else
        {
            DUMP_MESSAGE_STDOUT("OpenCL is not available");
            DUMP_PROPERTY_XML("cv_ocl", "not available");
            return;
        }

        const Device& device = Device::getDefault();
        if (!device.available())
            CV_ErrorNoReturn(CV_OpenCLInitError, "OpenCL device is not available");

        DUMP_MESSAGE_STDOUT("Current OpenCL device: ");

#if 0
        DUMP_MESSAGE_STDOUT("    Platform = "<< device.getPlatform().name());
        DUMP_PROPERTY_XML("cv_ocl_current_platformName", device.getPlatform().name());
#endif

        const char* deviceTypeStr = device.type() == Device::TYPE_CPU
            ? ("CPU") : (device.type() == Device::TYPE_GPU ? device.hostUnifiedMemory() ? "iGPU" : "dGPU" : "unknown");
        DUMP_MESSAGE_STDOUT("    Type = "<< deviceTypeStr);
        DUMP_PROPERTY_XML("cv_ocl_current_deviceType", deviceTypeStr);

        DUMP_MESSAGE_STDOUT("    Name = "<< device.name());
        DUMP_PROPERTY_XML("cv_ocl_current_deviceName", device.name());

        DUMP_MESSAGE_STDOUT("    Version = " << device.version());
        DUMP_PROPERTY_XML("cv_ocl_current_deviceVersion", device.version());

        DUMP_MESSAGE_STDOUT("    Driver version = " << device.driverVersion());
        DUMP_PROPERTY_XML("cv_ocl_current_driverVersion", device.driverVersion());

        DUMP_MESSAGE_STDOUT("    Compute units = "<< device.maxComputeUnits());
        DUMP_PROPERTY_XML("cv_ocl_current_maxComputeUnits", device.maxComputeUnits());

        DUMP_MESSAGE_STDOUT("    Max work group size = "<< device.maxWorkGroupSize());
        DUMP_PROPERTY_XML("cv_ocl_current_maxWorkGroupSize", device.maxWorkGroupSize());

        std::string localMemorySizeStr = bytesToStringRepr(device.localMemSize());
        DUMP_MESSAGE_STDOUT("    Local memory size = " << localMemorySizeStr);
        DUMP_PROPERTY_XML("cv_ocl_current_localMemSize", device.localMemSize());

        std::string maxMemAllocSizeStr = bytesToStringRepr(device.maxMemAllocSize());
        DUMP_MESSAGE_STDOUT("    Max memory allocation size = "<< maxMemAllocSizeStr);
        DUMP_PROPERTY_XML("cv_ocl_current_maxMemAllocSize", device.maxMemAllocSize());

        const char* doubleSupportStr = device.doubleFPConfig() > 0 ? "Yes" : "No";
        DUMP_MESSAGE_STDOUT("    Double support = "<< doubleSupportStr);
        DUMP_PROPERTY_XML("cv_ocl_current_haveDoubleSupport", device.doubleFPConfig() > 0);

        const char* isUnifiedMemoryStr = device.hostUnifiedMemory() ? "Yes" : "No";
        DUMP_MESSAGE_STDOUT("    Host unified memory = "<< isUnifiedMemoryStr);
        DUMP_PROPERTY_XML("cv_ocl_current_hostUnifiedMemory", device.hostUnifiedMemory());

        DUMP_MESSAGE_STDOUT("    Device extensions:");
        String extensionsStr = device.extensions();
        size_t pos = 0;
        while (pos < extensionsStr.size())
        {
            size_t pos2 = extensionsStr.find(' ', pos);
            if (pos2 == String::npos)
                pos2 = extensionsStr.size();
            if (pos2 > pos)
            {
                String extensionName = extensionsStr.substr(pos, pos2 - pos);
                DUMP_MESSAGE_STDOUT("        " << extensionName);
            }
            pos = pos2 + 1;
        }
        DUMP_PROPERTY_XML("cv_ocl_current_extensions", extensionsStr.c_str());

        const char* haveAmdBlasStr = haveAmdBlas() ? "Yes" : "No";
        DUMP_MESSAGE_STDOUT("    Has AMD Blas = "<< haveAmdBlasStr);
        DUMP_PROPERTY_XML("cv_ocl_current_AmdBlas", haveAmdBlas());

        const char* haveAmdFftStr = haveAmdFft() ? "Yes" : "No";
        DUMP_MESSAGE_STDOUT("    Has AMD Fft = "<< haveAmdFftStr);
        DUMP_PROPERTY_XML("cv_ocl_current_AmdFft", haveAmdFft());


        DUMP_MESSAGE_STDOUT("    Preferred vector width char = "<< device.preferredVectorWidthChar());
        DUMP_PROPERTY_XML("cv_ocl_current_preferredVectorWidthChar", device.preferredVectorWidthChar());

        DUMP_MESSAGE_STDOUT("    Preferred vector width short = "<< device.preferredVectorWidthShort());
        DUMP_PROPERTY_XML("cv_ocl_current_preferredVectorWidthShort", device.preferredVectorWidthShort());

        DUMP_MESSAGE_STDOUT("    Preferred vector width int = "<< device.preferredVectorWidthInt());
        DUMP_PROPERTY_XML("cv_ocl_current_preferredVectorWidthInt", device.preferredVectorWidthInt());

        DUMP_MESSAGE_STDOUT("    Preferred vector width long = "<< device.preferredVectorWidthLong());
        DUMP_PROPERTY_XML("cv_ocl_current_preferredVectorWidthLong", device.preferredVectorWidthLong());

        DUMP_MESSAGE_STDOUT("    Preferred vector width float = "<< device.preferredVectorWidthFloat());
        DUMP_PROPERTY_XML("cv_ocl_current_preferredVectorWidthFloat", device.preferredVectorWidthFloat());

        DUMP_MESSAGE_STDOUT("    Preferred vector width double = "<< device.preferredVectorWidthDouble());
        DUMP_PROPERTY_XML("cv_ocl_current_preferredVectorWidthDouble", device.preferredVectorWidthDouble());
    }
    catch (...)
    {
        DUMP_MESSAGE_STDOUT("Exception. Can't dump OpenCL info");
        DUMP_MESSAGE_STDOUT("OpenCL device not available");
        DUMP_PROPERTY_XML("cv_ocl", "not available");
    }
}
#undef DUMP_MESSAGE_STDOUT
#undef DUMP_PROPERTY_XML

#endif

Mat TestUtils::readImage(const String &fileName, int flags)
{
    return cv::imread(cvtest::TS::ptr()->get_data_path() + fileName, flags);
}

Mat TestUtils::readImageType(const String &fname, int type)
{
    Mat src = readImage(fname, CV_MAT_CN(type) == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (CV_MAT_CN(type) == 4)
    {
        Mat temp;
        cv::cvtColor(src, temp, cv::COLOR_BGR2BGRA);
        swap(src, temp);
    }
    src.convertTo(src, CV_MAT_DEPTH(type));
    return src;
}

double TestUtils::checkNorm1(InputArray m, InputArray mask)
{
    return cvtest::norm(m.getMat(), NORM_INF, mask.getMat());
}

double TestUtils::checkNorm2(InputArray m1, InputArray m2, InputArray mask)
{
    return cvtest::norm(m1.getMat(), m2.getMat(), NORM_INF, mask.getMat());
}

double TestUtils::checkSimilarity(InputArray m1, InputArray m2)
{
    Mat diff;
    matchTemplate(m1.getMat(), m2.getMat(), diff, CV_TM_CCORR_NORMED);
    return std::abs(diff.at<float>(0, 0) - 1.f);
}

double TestUtils::checkRectSimilarity(const Size & sz, std::vector<Rect>& ob1, std::vector<Rect>& ob2)
{
    double final_test_result = 0.0;
    size_t sz1 = ob1.size();
    size_t sz2 = ob2.size();

    if (sz1 != sz2)
        return sz1 > sz2 ? (double)(sz1 - sz2) : (double)(sz2 - sz1);
    else
    {
        if (sz1 == 0 && sz2 == 0)
            return 0;
        cv::Mat cpu_result(sz, CV_8UC1);
        cpu_result.setTo(0);

        for (vector<Rect>::const_iterator r = ob1.begin(); r != ob1.end(); ++r)
        {
            cv::Mat cpu_result_roi(cpu_result, *r);
            cpu_result_roi.setTo(1);
            cpu_result.copyTo(cpu_result);
        }
        int cpu_area = cv::countNonZero(cpu_result > 0);

        cv::Mat gpu_result(sz, CV_8UC1);
        gpu_result.setTo(0);
        for(vector<Rect>::const_iterator r2 = ob2.begin(); r2 != ob2.end(); ++r2)
        {
            cv::Mat gpu_result_roi(gpu_result, *r2);
            gpu_result_roi.setTo(1);
            gpu_result.copyTo(gpu_result);
        }

        cv::Mat result_;
        multiply(cpu_result, gpu_result, result_);
        int result = cv::countNonZero(result_ > 0);
        if (cpu_area!=0 && result!=0)
            final_test_result = 1.0 - (double)result/(double)cpu_area;
        else if(cpu_area==0 && result!=0)
            final_test_result = -1;
    }
    return final_test_result;
}

void TestUtils::showDiff(InputArray _src, InputArray _gold, InputArray _actual, double eps, bool alwaysShow)
{
    Mat src = _src.getMat(), actual = _actual.getMat(), gold = _gold.getMat();

    Mat diff, diff_thresh;
    absdiff(gold, actual, diff);
    diff.convertTo(diff, CV_32F);
    threshold(diff, diff_thresh, eps, 255.0, cv::THRESH_BINARY);

    if (alwaysShow || cv::countNonZero(diff_thresh.reshape(1)) > 0)
    {
#if 0
        std::cout << "Source: " << std::endl << src << std::endl;
        std::cout << "Expected: " << std::endl << gold << std::endl;
        std::cout << "Actual: " << std::endl << actual << std::endl;
#endif

        namedWindow("src", WINDOW_NORMAL);
        namedWindow("gold", WINDOW_NORMAL);
        namedWindow("actual", WINDOW_NORMAL);
        namedWindow("diff", WINDOW_NORMAL);

        imshow("src", src);
        imshow("gold", gold);
        imshow("actual", actual);
        imshow("diff", diff);

        cv::waitKey();
    }
}

} } // namespace cvtest::ocl
