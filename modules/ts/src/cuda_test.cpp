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

#include "opencv2/ts/cuda_test.hpp"
#include <stdexcept>

using namespace cv;
using namespace cv::cuda;
using namespace cvtest;
using namespace testing;
using namespace testing::internal;

namespace perf
{
    void printCudaInfo();
}

namespace cvtest
{
    //////////////////////////////////////////////////////////////////////
    // random generators

    int randomInt(int minVal, int maxVal)
    {
        RNG& rng = TS::ptr()->get_rng();
        return rng.uniform(minVal, maxVal);
    }

    double randomDouble(double minVal, double maxVal)
    {
        RNG& rng = TS::ptr()->get_rng();
        return rng.uniform(minVal, maxVal);
    }

    Size randomSize(int minVal, int maxVal)
    {
        return Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
    }

    Scalar randomScalar(double minVal, double maxVal)
    {
        return Scalar(randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal));
    }

    Mat randomMat(Size size, int type, double minVal, double maxVal)
    {
        return randomMat(TS::ptr()->get_rng(), size, type, minVal, maxVal, false);
    }

    //////////////////////////////////////////////////////////////////////
    // GpuMat create

    GpuMat createMat(Size size, int type, bool useRoi)
    {
        Size size0; Point ofs;
        return createMat(size, type, size0, ofs, useRoi);
    }

    GpuMat createMat(Size size, int type, Size& size0, Point& ofs, bool useRoi)
    {
        size0 = size;

        if (useRoi)
        {
            size0.width += randomInt(5, 15);
            size0.height += randomInt(5, 15);
        }

        GpuMat d_m(size0, type);
        if (size0 != size) {
            ofs = Point((size0.width - size.width) / 2, (size0.height - size.height) / 2);
            d_m = d_m(Rect(ofs, size));
        }

        return d_m;
    }

    GpuMat loadMat(const Mat& m, bool useRoi)
    {
        GpuMat d_m = createMat(m.size(), m.type(), useRoi);
        d_m.upload(m);
        return d_m;
    }

    //////////////////////////////////////////////////////////////////////
    // Image load

    Mat readImage(const std::string& fileName, int flags)
    {
        return imread(TS::ptr()->get_data_path() + fileName, flags);
    }

    Mat readImageType(const std::string& fname, int type)
    {
        Mat src = readImage(fname, CV_MAT_CN(type) == 1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
        if (CV_MAT_CN(type) == 4)
        {
            Mat temp;
            cvtColor(src, temp, COLOR_BGR2BGRA);
            swap(src, temp);
        }
        src.convertTo(src, CV_MAT_DEPTH(type), CV_MAT_DEPTH(type) == CV_32F ? 1.0 / 255.0 : 1.0);
        return src;
    }

    //////////////////////////////////////////////////////////////////////
    // Gpu devices

    bool supportFeature(const DeviceInfo& info, FeatureSet feature)
    {
        return TargetArchs::builtWith(feature) && info.supports(feature);
    }

    DeviceManager& DeviceManager::instance()
    {
        static DeviceManager obj;
        return obj;
    }

    void DeviceManager::load(int i)
    {
        devices_.clear();
        devices_.reserve(1);

        std::ostringstream msg;

        if (i < 0 || i >= getCudaEnabledDeviceCount())
        {
            msg << "Incorrect device number - " << i;
            throw std::runtime_error(msg.str());
        }

        DeviceInfo info(i);

        if (!info.isCompatible())
        {
            msg << "Device " << i << " [" << info.name() << "] is NOT compatible with current CUDA module build";
            throw std::runtime_error(msg.str());
        }

        devices_.push_back(info);
    }

    void DeviceManager::loadAll()
    {
        int deviceCount = getCudaEnabledDeviceCount();

        devices_.clear();
        devices_.reserve(deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            DeviceInfo info(i);
            if (info.isCompatible())
            {
                devices_.push_back(info);
            }
        }
    }

    void parseCudaDeviceOptions(int argc, char **argv)
    {
        cv::CommandLineParser cmd(argc, argv,
            "{ cuda_device | -1    | CUDA device on which tests will be executed (-1 means all devices) }"
            "{ h help      | false | Print help info                                                    }"
        );

        if (cmd.has("help"))
        {
            std::cout << "\nAvailable options besides google test option: \n";
            cmd.printMessage();
        }

        int device = cmd.get<int>("cuda_device");
        if (device < 0)
        {
            cvtest::DeviceManager::instance().loadAll();
            std::cout << "Run tests on all supported CUDA devices \n" << std::endl;
        }
        else
        {
            cvtest::DeviceManager::instance().load(device);
            cv::cuda::DeviceInfo info(device);
            std::cout << "Run tests on CUDA device " << device << " [" << info.name() << "] \n" << std::endl;
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Additional assertion

    namespace
    {
        template <typename T, typename OutT> std::string printMatValImpl(const Mat& m, Point p)
        {
            const int cn = m.channels();

            std::ostringstream ostr;
            ostr << "(";

            p.x /= cn;

            ostr << static_cast<OutT>(m.at<T>(p.y, p.x * cn));
            for (int c = 1; c < m.channels(); ++c)
            {
                ostr << ", " << static_cast<OutT>(m.at<T>(p.y, p.x * cn + c));
            }
            ostr << ")";

            return ostr.str();
        }

        std::string printMatVal(const Mat& m, Point p)
        {
            typedef std::string (*func_t)(const Mat& m, Point p);

            static const func_t funcs[] =
            {
                printMatValImpl<uchar, int>, printMatValImpl<schar, int>, printMatValImpl<ushort, int>, printMatValImpl<short, int>,
                printMatValImpl<int, int>, printMatValImpl<float, float>, printMatValImpl<double, double>
            };

            return funcs[m.depth()](m, p);
        }
    }

    void minMaxLocGold(const Mat& src, double* minVal_, double* maxVal_, Point* minLoc_, Point* maxLoc_, const Mat& mask)
    {
        if (src.depth() != CV_8S)
        {
            minMaxLoc(src, minVal_, maxVal_, minLoc_, maxLoc_, mask);
            return;
        }

        // OpenCV's minMaxLoc doesn't support CV_8S type
        double minVal = std::numeric_limits<double>::max();
        Point minLoc(-1, -1);

        double maxVal = -std::numeric_limits<double>::max();
        Point maxLoc(-1, -1);

        for (int y = 0; y < src.rows; ++y)
        {
            const schar* src_row = src.ptr<schar>(y);
            const uchar* mask_row = mask.empty() ? 0 : mask.ptr<uchar>(y);

            for (int x = 0; x < src.cols; ++x)
            {
                if (!mask_row || mask_row[x])
                {
                    schar val = src_row[x];

                    if (val < minVal)
                    {
                        minVal = val;
                        minLoc = cv::Point(x, y);
                    }

                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxLoc = cv::Point(x, y);
                    }
                }
            }
        }

        if (minVal_) *minVal_ = minVal;
        if (maxVal_) *maxVal_ = maxVal;

        if (minLoc_) *minLoc_ = minLoc;
        if (maxLoc_) *maxLoc_ = maxLoc;
    }

    Mat getMat(InputArray arr)
    {
        if (arr.kind() == _InputArray::CUDA_GPU_MAT)
        {
            Mat m;
            arr.getGpuMat().download(m);
            return m;
        }

        return arr.getMat();
    }

    AssertionResult assertMatNear(const char* expr1, const char* expr2, const char* eps_expr, InputArray m1_, InputArray m2_, double eps)
    {
        Mat m1 = getMat(m1_);
        Mat m2 = getMat(m2_);

        if (m1.size() != m2.size())
        {
            std::stringstream msg;
            msg << "Matrices \"" << expr1 << "\" and \"" << expr2 << "\" have different sizes : \""
                << expr1 << "\" [" << PrintToString(m1.size()) << "] vs \""
                << expr2 << "\" [" << PrintToString(m2.size()) << "]";
            return AssertionFailure() << msg.str();
        }

        if (m1.type() != m2.type())
        {
            std::stringstream msg;
            msg << "Matrices \"" << expr1 << "\" and \"" << expr2 << "\" have different types : \""
                << expr1 << "\" [" << PrintToString(MatType(m1.type())) << "] vs \""
                << expr2 << "\" [" << PrintToString(MatType(m2.type())) << "]";
             return AssertionFailure() << msg.str();
        }

        Mat diff;
        absdiff(m1.reshape(1), m2.reshape(1), diff);

        double maxVal = 0.0;
        Point maxLoc;
        minMaxLocGold(diff, 0, &maxVal, 0, &maxLoc);

        if (maxVal > eps)
        {
            std::stringstream msg;
            msg << "The max difference between matrices \"" << expr1 << "\" and \"" << expr2
                << "\" is " << maxVal << " at (" << maxLoc.y << ", " << maxLoc.x / m1.channels() << ")"
                << ", which exceeds \"" << eps_expr << "\", where \""
                << expr1 << "\" at (" << maxLoc.y << ", " << maxLoc.x / m1.channels() << ") evaluates to " << printMatVal(m1, maxLoc) << ", \""
                << expr2 << "\" at (" << maxLoc.y << ", " << maxLoc.x / m1.channels() << ") evaluates to " << printMatVal(m2, maxLoc) << ", \""
                << eps_expr << "\" evaluates to " << eps;
            return AssertionFailure() << msg.str();
        }

        return AssertionSuccess();
    }

    double checkSimilarity(InputArray m1, InputArray m2)
    {
        Mat diff;
        matchTemplate(getMat(m1), getMat(m2), diff, TM_CCORR_NORMED);
        return std::abs(diff.at<float>(0, 0) - 1.f);
    }

    //////////////////////////////////////////////////////////////////////
    // Helper structs for value-parameterized tests

    vector<MatType> types(int depth_start, int depth_end, int cn_start, int cn_end)
    {
        vector<MatType> v;

        v.reserve((depth_end - depth_start + 1) * (cn_end - cn_start + 1));

        for (int depth = depth_start; depth <= depth_end; ++depth)
        {
            for (int cn = cn_start; cn <= cn_end; ++cn)
            {
                v.push_back(MatType(CV_MAKE_TYPE(depth, cn)));
            }
        }

        return v;
    }

    const vector<MatType>& all_types()
    {
        static vector<MatType> v = types(CV_8U, CV_64F, 1, 4);

        return v;
    }

    void PrintTo(const UseRoi& useRoi, std::ostream* os)
    {
        if (useRoi)
            (*os) << "sub matrix";
        else
            (*os) << "whole matrix";
    }

    void PrintTo(const Inverse& inverse, std::ostream* os)
    {
        if (inverse)
            (*os) << "inverse";
        else
            (*os) << "direct";
    }

    //////////////////////////////////////////////////////////////////////
    // Other

    void dumpImage(const std::string& fileName, const Mat& image)
    {
        imwrite(TS::ptr()->get_data_path() + fileName, image);
    }

    void showDiff(InputArray gold_, InputArray actual_, double eps)
    {
        Mat gold = getMat(gold_);
        Mat actual = getMat(actual_);

        Mat diff;
        absdiff(gold, actual, diff);
        threshold(diff, diff, eps, 255.0, cv::THRESH_BINARY);

        namedWindow("gold", WINDOW_NORMAL);
        namedWindow("actual", WINDOW_NORMAL);
        namedWindow("diff", WINDOW_NORMAL);

        imshow("gold", gold);
        imshow("actual", actual);
        imshow("diff", diff);

        waitKey();
    }

    namespace
    {
        bool keyPointsEquals(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
        {
            const double maxPtDif = 1.0;
            const double maxSizeDif = 1.0;
            const double maxAngleDif = 2.0;
            const double maxResponseDif = 0.1;

            double dist = cv::norm(p1.pt - p2.pt);

            if (dist < maxPtDif &&
                fabs(p1.size - p2.size) < maxSizeDif &&
                abs(p1.angle - p2.angle) < maxAngleDif &&
                abs(p1.response - p2.response) < maxResponseDif &&
                p1.octave == p2.octave &&
                p1.class_id == p2.class_id)
            {
                return true;
            }

            return false;
        }

        struct KeyPointLess
        {
            bool operator()(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) const
            {
                return kp1.pt.y < kp2.pt.y || (kp1.pt.y == kp2.pt.y && kp1.pt.x < kp2.pt.x);
            }
        };
    }

    testing::AssertionResult assertKeyPointsEquals(const char* gold_expr, const char* actual_expr, std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
    {
        if (gold.size() != actual.size())
        {
            std::stringstream msg;
            msg << "KeyPoints size mistmach\n"
                << "\"" << gold_expr << "\" : " << gold.size() << "\n"
                << "\"" << actual_expr << "\" : " << actual.size();
            return AssertionFailure() << msg.str();
        }

        std::sort(actual.begin(), actual.end(), KeyPointLess());
        std::sort(gold.begin(), gold.end(), KeyPointLess());

        for (size_t i = 0; i < gold.size(); ++i)
        {
            const cv::KeyPoint& p1 = gold[i];
            const cv::KeyPoint& p2 = actual[i];

            if (!keyPointsEquals(p1, p2))
            {
                std::stringstream msg;
                msg << "KeyPoints differ at " << i << "\n"
                    << "\"" << gold_expr << "\" vs \"" << actual_expr << "\" : \n"
                    << "pt : " << testing::PrintToString(p1.pt) << " vs " << testing::PrintToString(p2.pt) << "\n"
                    << "size : " << p1.size << " vs " << p2.size << "\n"
                    << "angle : " << p1.angle << " vs " << p2.angle << "\n"
                    << "response : " << p1.response << " vs " << p2.response << "\n"
                    << "octave : " << p1.octave << " vs " << p2.octave << "\n"
                    << "class_id : " << p1.class_id << " vs " << p2.class_id;
                return AssertionFailure() << msg.str();
            }
        }

        return ::testing::AssertionSuccess();
    }

    int getMatchedPointsCount(std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
    {
        std::sort(actual.begin(), actual.end(), KeyPointLess());
        std::sort(gold.begin(), gold.end(), KeyPointLess());

        int validCount = 0;

        if (actual.size() == gold.size())
        {
            for (size_t i = 0; i < gold.size(); ++i)
            {
                const cv::KeyPoint& p1 = gold[i];
                const cv::KeyPoint& p2 = actual[i];

                if (keyPointsEquals(p1, p2))
                    ++validCount;
            }
        }
        else
        {
            std::vector<cv::KeyPoint>& shorter = gold;
            std::vector<cv::KeyPoint>& longer = actual;
            if (actual.size() < gold.size())
            {
                shorter = actual;
                longer = gold;
            }
            for (size_t i = 0; i < shorter.size(); ++i)
            {
                const cv::KeyPoint& p1 = shorter[i];
                const cv::KeyPoint& p2 = longer[i];
                const cv::KeyPoint& p3 = longer[i+1];

                if (keyPointsEquals(p1, p2) || keyPointsEquals(p1, p3))
                    ++validCount;
            }
        }

        return validCount;
    }

    int getMatchedPointsCount(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches)
    {
        int validCount = 0;

        for (size_t i = 0; i < matches.size(); ++i)
        {
            const cv::DMatch& m = matches[i];

            const cv::KeyPoint& p1 = keypoints1[m.queryIdx];
            const cv::KeyPoint& p2 = keypoints2[m.trainIdx];

            if (keyPointsEquals(p1, p2))
                ++validCount;
        }

        return validCount;
    }

    void printCudaInfo()
    {
        perf::printCudaInfo();
    }
}


void cv::cuda::PrintTo(const DeviceInfo& info, std::ostream* os)
{
    (*os) << info.name();
    if (info.deviceID())
        (*os) << " [ID: " << info.deviceID() << "]";
}
