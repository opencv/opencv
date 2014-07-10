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

#ifndef __OPENCV_CUDA_TEST_UTILITY_HPP__
#define __OPENCV_CUDA_TEST_UTILITY_HPP__

#include <stdexcept>
#include "cvconfig.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ts.hpp"

namespace cvtest
{
    //////////////////////////////////////////////////////////////////////
    // random generators

    CV_EXPORTS int randomInt(int minVal, int maxVal);
    CV_EXPORTS double randomDouble(double minVal, double maxVal);
    CV_EXPORTS cv::Size randomSize(int minVal, int maxVal);
    CV_EXPORTS cv::Scalar randomScalar(double minVal, double maxVal);
    CV_EXPORTS cv::Mat randomMat(cv::Size size, int type, double minVal = 0.0, double maxVal = 255.0);

    //////////////////////////////////////////////////////////////////////
    // GpuMat create

    CV_EXPORTS cv::cuda::GpuMat createMat(cv::Size size, int type, bool useRoi = false);
    CV_EXPORTS cv::cuda::GpuMat loadMat(const cv::Mat& m, bool useRoi = false);

    //////////////////////////////////////////////////////////////////////
    // Image load

    //! read image from testdata folder
    CV_EXPORTS cv::Mat readImage(const std::string& fileName, int flags = cv::IMREAD_COLOR);

    //! read image from testdata folder and convert it to specified type
    CV_EXPORTS cv::Mat readImageType(const std::string& fname, int type);

    //////////////////////////////////////////////////////////////////////
    // Gpu devices

    //! return true if device supports specified feature and gpu module was built with support the feature.
    CV_EXPORTS bool supportFeature(const cv::cuda::DeviceInfo& info, cv::cuda::FeatureSet feature);

    class CV_EXPORTS DeviceManager
    {
    public:
        static DeviceManager& instance();

        void load(int i);
        void loadAll();

        const std::vector<cv::cuda::DeviceInfo>& values() const { return devices_; }

    private:
        std::vector<cv::cuda::DeviceInfo> devices_;
    };

    #define ALL_DEVICES testing::ValuesIn(cvtest::DeviceManager::instance().values())

    //////////////////////////////////////////////////////////////////////
    // Additional assertion

    CV_EXPORTS void minMaxLocGold(const cv::Mat& src, double* minVal_, double* maxVal_ = 0, cv::Point* minLoc_ = 0, cv::Point* maxLoc_ = 0, const cv::Mat& mask = cv::Mat());

    CV_EXPORTS cv::Mat getMat(cv::InputArray arr);

    CV_EXPORTS testing::AssertionResult assertMatNear(const char* expr1, const char* expr2, const char* eps_expr, cv::InputArray m1, cv::InputArray m2, double eps);

    #define EXPECT_MAT_NEAR(m1, m2, eps) EXPECT_PRED_FORMAT3(cvtest::assertMatNear, m1, m2, eps)
    #define ASSERT_MAT_NEAR(m1, m2, eps) ASSERT_PRED_FORMAT3(cvtest::assertMatNear, m1, m2, eps)

    #define EXPECT_SCALAR_NEAR(s1, s2, eps) \
        { \
            EXPECT_NEAR(s1[0], s2[0], eps); \
            EXPECT_NEAR(s1[1], s2[1], eps); \
            EXPECT_NEAR(s1[2], s2[2], eps); \
            EXPECT_NEAR(s1[3], s2[3], eps); \
        }
    #define ASSERT_SCALAR_NEAR(s1, s2, eps) \
        { \
            ASSERT_NEAR(s1[0], s2[0], eps); \
            ASSERT_NEAR(s1[1], s2[1], eps); \
            ASSERT_NEAR(s1[2], s2[2], eps); \
            ASSERT_NEAR(s1[3], s2[3], eps); \
        }

    #define EXPECT_POINT2_NEAR(p1, p2, eps) \
        { \
            EXPECT_NEAR(p1.x, p2.x, eps); \
            EXPECT_NEAR(p1.y, p2.y, eps); \
        }
    #define ASSERT_POINT2_NEAR(p1, p2, eps) \
        { \
            ASSERT_NEAR(p1.x, p2.x, eps); \
            ASSERT_NEAR(p1.y, p2.y, eps); \
        }

    #define EXPECT_POINT3_NEAR(p1, p2, eps) \
        { \
            EXPECT_NEAR(p1.x, p2.x, eps); \
            EXPECT_NEAR(p1.y, p2.y, eps); \
            EXPECT_NEAR(p1.z, p2.z, eps); \
        }
    #define ASSERT_POINT3_NEAR(p1, p2, eps) \
        { \
            ASSERT_NEAR(p1.x, p2.x, eps); \
            ASSERT_NEAR(p1.y, p2.y, eps); \
            ASSERT_NEAR(p1.z, p2.z, eps); \
        }

    CV_EXPORTS double checkSimilarity(cv::InputArray m1, cv::InputArray m2);

    #define EXPECT_MAT_SIMILAR(mat1, mat2, eps) \
        { \
            ASSERT_EQ(mat1.type(), mat2.type()); \
            ASSERT_EQ(mat1.size(), mat2.size()); \
            EXPECT_LE(checkSimilarity(mat1, mat2), eps); \
        }
    #define ASSERT_MAT_SIMILAR(mat1, mat2, eps) \
        { \
            ASSERT_EQ(mat1.type(), mat2.type()); \
            ASSERT_EQ(mat1.size(), mat2.size()); \
            ASSERT_LE(checkSimilarity(mat1, mat2), eps); \
        }

    //////////////////////////////////////////////////////////////////////
    // Helper structs for value-parameterized tests

    #define CUDA_TEST_P(test_case_name, test_name) \
      class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
          : public test_case_name { \
       public: \
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {} \
        virtual void TestBody(); \
       private: \
        void UnsafeTestBody(); \
        static int AddToRegistry() { \
          ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
              GetTestCasePatternHolder<test_case_name>(\
                  #test_case_name, __FILE__, __LINE__)->AddTestPattern(\
                      #test_case_name, \
                      #test_name, \
                      new ::testing::internal::TestMetaFactory< \
                          GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>()); \
          return 0; \
        } \
        static int gtest_registering_dummy_; \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
      }; \
      int GTEST_TEST_CLASS_NAME_(test_case_name, \
                                 test_name)::gtest_registering_dummy_ = \
          GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
      void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() \
      { \
        try \
        { \
          UnsafeTestBody(); \
        } \
        catch (...) \
        { \
          cv::cuda::resetDevice(); \
          throw; \
        } \
      } \
      void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::UnsafeTestBody()

    #define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >
    #define GET_PARAM(k) std::tr1::get< k >(GetParam())

    #define DIFFERENT_SIZES testing::Values(cv::Size(128, 128), cv::Size(113, 113))

    // Depth

    using perf::MatDepth;

    #define ALL_DEPTH testing::Values(MatDepth(CV_8U), MatDepth(CV_8S), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32S), MatDepth(CV_32F), MatDepth(CV_64F))

    #define DEPTH_PAIRS testing::Values(std::make_pair(MatDepth(CV_8U), MatDepth(CV_8U)),   \
                                        std::make_pair(MatDepth(CV_8U), MatDepth(CV_16U)),  \
                                        std::make_pair(MatDepth(CV_8U), MatDepth(CV_16S)),  \
                                        std::make_pair(MatDepth(CV_8U), MatDepth(CV_32S)),  \
                                        std::make_pair(MatDepth(CV_8U), MatDepth(CV_32F)),  \
                                        std::make_pair(MatDepth(CV_8U), MatDepth(CV_64F)),  \
                                                                                            \
                                        std::make_pair(MatDepth(CV_16U), MatDepth(CV_16U)), \
                                        std::make_pair(MatDepth(CV_16U), MatDepth(CV_32S)), \
                                        std::make_pair(MatDepth(CV_16U), MatDepth(CV_32F)), \
                                        std::make_pair(MatDepth(CV_16U), MatDepth(CV_64F)), \
                                                                                            \
                                        std::make_pair(MatDepth(CV_16S), MatDepth(CV_16S)), \
                                        std::make_pair(MatDepth(CV_16S), MatDepth(CV_32S)), \
                                        std::make_pair(MatDepth(CV_16S), MatDepth(CV_32F)), \
                                        std::make_pair(MatDepth(CV_16S), MatDepth(CV_64F)), \
                                                                                            \
                                        std::make_pair(MatDepth(CV_32S), MatDepth(CV_32S)), \
                                        std::make_pair(MatDepth(CV_32S), MatDepth(CV_32F)), \
                                        std::make_pair(MatDepth(CV_32S), MatDepth(CV_64F)), \
                                                                                            \
                                        std::make_pair(MatDepth(CV_32F), MatDepth(CV_32F)), \
                                        std::make_pair(MatDepth(CV_32F), MatDepth(CV_64F)), \
                                                                                            \
                                        std::make_pair(MatDepth(CV_64F), MatDepth(CV_64F)))

    // Type

    using perf::MatType;

    //! return vector with types from specified range.
    CV_EXPORTS std::vector<MatType> types(int depth_start, int depth_end, int cn_start, int cn_end);

    //! return vector with all types (depth: CV_8U-CV_64F, channels: 1-4).
    CV_EXPORTS const std::vector<MatType>& all_types();

    #define ALL_TYPES testing::ValuesIn(all_types())
    #define TYPES(depth_start, depth_end, cn_start, cn_end) testing::ValuesIn(types(depth_start, depth_end, cn_start, cn_end))

    // ROI

    class UseRoi
    {
    public:
        inline UseRoi(bool val = false) : val_(val) {}

        inline operator bool() const { return val_; }

    private:
        bool val_;
    };

    CV_EXPORTS void PrintTo(const UseRoi& useRoi, std::ostream* os);

    #define WHOLE_SUBMAT testing::Values(UseRoi(false), UseRoi(true))

    // Direct/Inverse

    class Inverse
    {
    public:
        inline Inverse(bool val = false) : val_(val) {}

        inline operator bool() const { return val_; }

    private:
        bool val_;
    };

    CV_EXPORTS void PrintTo(const Inverse& useRoi, std::ostream* os);

    #define DIRECT_INVERSE testing::Values(Inverse(false), Inverse(true))

    // Param class

    #define IMPLEMENT_PARAM_CLASS(name, type) \
        class name \
        { \
        public: \
            name ( type arg = type ()) : val_(arg) {} \
            operator type () const {return val_;} \
        private: \
            type val_; \
        }; \
        inline void PrintTo( name param, std::ostream* os) \
        { \
            *os << #name <<  "(" << testing::PrintToString(static_cast< type >(param)) << ")"; \
        }

    IMPLEMENT_PARAM_CLASS(Channels, int)

    #define ALL_CHANNELS testing::Values(Channels(1), Channels(2), Channels(3), Channels(4))
    #define IMAGE_CHANNELS testing::Values(Channels(1), Channels(3), Channels(4))

    // Flags and enums

    CV_ENUM(NormCode, NORM_INF, NORM_L1, NORM_L2, NORM_TYPE_MASK, NORM_RELATIVE, NORM_MINMAX)

    CV_ENUM(Interpolation, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA)

    CV_ENUM(BorderType, BORDER_REFLECT101, BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_WRAP)
    #define ALL_BORDER_TYPES testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT), BorderType(cv::BORDER_WRAP))

    CV_FLAGS(WarpFlags, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, WARP_INVERSE_MAP)

    //////////////////////////////////////////////////////////////////////
    // Features2D

    CV_EXPORTS testing::AssertionResult assertKeyPointsEquals(const char* gold_expr, const char* actual_expr, std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual);

    #define ASSERT_KEYPOINTS_EQ(gold, actual) EXPECT_PRED_FORMAT2(assertKeyPointsEquals, gold, actual)

    CV_EXPORTS int getMatchedPointsCount(std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual);
    CV_EXPORTS int getMatchedPointsCount(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches);

    //////////////////////////////////////////////////////////////////////
    // Other

    CV_EXPORTS void dumpImage(const std::string& fileName, const cv::Mat& image);
    CV_EXPORTS void showDiff(cv::InputArray gold, cv::InputArray actual, double eps);

    CV_EXPORTS void printCudaInfo();
}

namespace cv { namespace cuda
{
    CV_EXPORTS void PrintTo(const DeviceInfo& info, std::ostream* os);
}}

#ifdef HAVE_CUDA

#define CV_CUDA_TEST_MAIN(resourcesubdir) \
    int main(int argc, char* argv[]) \
    { \
        try \
        { \
            cv::CommandLineParser cmd(argc, argv, \
                "{ h help ?            |      | Print help}" \
                "{ i info              |      | Print information about system and exit }" \
                "{ device              | -1   | Device on which tests will be executed (-1 means all devices) }" \
            ); \
            if (cmd.has("help")) \
            { \
                cmd.printMessage(); \
                return 0; \
            } \
            cvtest::printCudaInfo(); \
            if (cmd.has("info")) \
            { \
                return 0; \
            } \
            int device = cmd.get<int>("device"); \
            if (device < 0) \
            { \
                cvtest::DeviceManager::instance().loadAll(); \
                std::cout << "Run tests on all supported devices \n" << std::endl; \
            } \
            else \
            { \
                cvtest::DeviceManager::instance().load(device); \
                cv::cuda::DeviceInfo info(device); \
                std::cout << "Run tests on device " << device << " [" << info.name() << "] \n" << std::endl; \
            } \
            cvtest::TS::ptr()->init( resourcesubdir ); \
            testing::InitGoogleTest(&argc, argv); \
            return RUN_ALL_TESTS(); \
        } \
        catch (const std::exception& e) \
        { \
            std::cerr << e.what() << std::endl; \
            return -1; \
        } \
        catch (...) \
        { \
            std::cerr << "Unknown error" << std::endl; \
            return -1; \
        } \
        return 0; \
    }

#else // HAVE_CUDA

#define CV_CUDA_TEST_MAIN(resourcesubdir) \
    int main() \
    { \
        printf("OpenCV was built without CUDA support\n"); \
        return 0; \
    }

#endif // HAVE_CUDA


#endif // __OPENCV_CUDA_TEST_UTILITY_HPP__
