///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan,jlyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Zailong Wu, bullet@yeah.net
//    Yao Wang, bitwangyaoyao@gmail.com
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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////////////// HOG /////////////////////////////
PARAM_TEST_CASE(HOG, Size, MatType)
{
    Size winSize;
    int type;
    Mat img;
    UMat uimg;
    virtual void SetUp()
    {
        winSize = GET_PARAM(0);
        type = GET_PARAM(1);
        img = readImage("cascadeandhog/images/image_00000000_0.png", IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty());
        img.copyTo(uimg);
    }
};

OCL_TEST_P(HOG, GetDescriptors)
{
    HOGDescriptor hog;
    hog.gammaCorrection = true;

    hog.setSVMDetector(hog.getDefaultPeopleDetector());

    std::vector<float> cpu_descriptors;
    std::vector<float> gpu_descriptors;

    OCL_OFF(hog.compute(img, cpu_descriptors, hog.winSize));
    OCL_ON(hog.compute(uimg, gpu_descriptors, hog.winSize));

    Mat cpu_desc(cpu_descriptors), gpu_desc(gpu_descriptors);

    EXPECT_MAT_SIMILAR(cpu_desc, gpu_desc, 1e-1);
}

OCL_TEST_P(HOG, SVMDetector)
{
    HOGDescriptor hog_first, hog_second;

    // empty -> empty
    hog_first.copyTo(hog_second);

    // first -> both
    hog_first.setSVMDetector(hog_first.getDefaultPeopleDetector());
    hog_first.copyTo(hog_second);

    // both -> both
    hog_first.copyTo(hog_second);

    // second -> empty
    hog_first.setSVMDetector(cv::noArray());
    hog_first.copyTo(hog_second);
}

OCL_TEST_P(HOG, Detect)
{
    HOGDescriptor hog;
    hog.winSize = winSize;
    hog.gammaCorrection = true;

    if (winSize.width == 48 && winSize.height == 96)
        hog.setSVMDetector(hog.getDaimlerPeopleDetector());
    else
        hog.setSVMDetector(hog.getDefaultPeopleDetector());

    std::vector<Rect> cpu_found;
    std::vector<Rect> gpu_found;

    OCL_OFF(hog.detectMultiScale(img, cpu_found, 0, Size(8, 8), Size(0, 0), 1.05, 6));
    OCL_ON(hog.detectMultiScale(uimg, gpu_found, 0, Size(8, 8), Size(0, 0), 1.05, 6));

    EXPECT_LT(checkRectSimilarity(img.size(), cpu_found, gpu_found), 0.05);
}

INSTANTIATE_TEST_CASE_P(OCL_ObjDetect, HOG, testing::Combine(
                            testing::Values(Size(64, 128), Size(48, 96)),
                            testing::Values( MatType(CV_8UC1) ) ) );

namespace {

//! HOG parameters of a comparison between the OpenCL and the CPU implementation.
struct Geometry
{
    const char* name;
    Size winSize;
    Size blockSize;
    Size blockStride;
    Size cellSize;
    int nbins;
    bool signedGradient;
};

void PrintTo(const Geometry& g, std::ostream* os)
{
    *os << cv::format("%s(winSize=%dx%d blockSize=%dx%d blockStride=%dx%d cellSize=%dx%d "
                      "nbins=%d signedGradient=%d)",
                      g.name, g.winSize.width, g.winSize.height, g.blockSize.width,
                      g.blockSize.height, g.blockStride.width, g.blockStride.height,
                      g.cellSize.width, g.cellSize.height, g.nbins, int(g.signedGradient));
}

//! Window placement passed to compute() in addition to the geometry.
struct Placement
{
    const char* name;
    Size winStride;
    Size padding;
    std::vector<Point> locations;
};

void PrintTo(const Placement& p, std::ostream* os)
{
    *os << cv::format("%s(winStride=%dx%d padding=%dx%d locations=%d)", p.name,
                      p.winStride.width, p.winStride.height, p.padding.width, p.padding.height,
                      int(p.locations.size()));
}

//! Search and grouping of the detections in addition to the geometry.
struct Search
{
    const char* name;
    double groupThreshold;
    bool useMeanshiftGrouping;
    Size padding;
    double scale;
};

void PrintTo(const Search& s, std::ostream* os)
{
    *os << cv::format("%s(groupThreshold=%g useMeanshiftGrouping=%d padding=%dx%d scale=%g)",
                      s.name, s.groupThreshold, int(s.useMeanshiftGrouping), s.padding.width,
                      s.padding.height, s.scale);
}

HOGDescriptor makeDescriptor(const Geometry& g, bool gammaCorrection)
{
    HOGDescriptor hog(g.winSize, g.blockSize, g.blockStride, g.cellSize, g.nbins);
    hog.signedGradient = g.signedGradient;
    hog.gammaCorrection = gammaCorrection;
    return hog;
}

const Size defaultWinSize(64, 128);
const Size defaultBlockSize(16, 16);
const Size defaultBlockStride(8, 8);
const Size defaultCellSize(8, 8);
const int defaultNbins = 9;

const Size imageSize(100, 150);

Mat makeImage()
{
    Mat image(imageSize, CV_8UC1);
    RNG rng(0);
    rng.fill(image, RNG::UNIFORM, 0, 256);
    return image;
}

// Configurations the OpenCL kernels implement and configurations they do not,
// for which the CPU implementation has to be used instead.
const Geometry geometries[] = {
    {"Default", defaultWinSize, defaultBlockSize, defaultBlockStride, defaultCellSize, defaultNbins, false},
    {"Window48x96", Size(48, 96), defaultBlockSize, defaultBlockStride, defaultCellSize, defaultNbins, false},
    {"Window80x128", Size(80, 128), defaultBlockSize, defaultBlockStride, defaultCellSize, defaultNbins, false},
    {"BlockStride16x8", defaultWinSize, defaultBlockSize, Size(16, 8), defaultCellSize, defaultNbins, false},
    {"BlockStride8x16", defaultWinSize, defaultBlockSize, Size(8, 16), defaultCellSize, defaultNbins, false},
    {"Bins12", defaultWinSize, defaultBlockSize, defaultBlockStride, defaultCellSize, 12, false},
    {"Bins16", defaultWinSize, defaultBlockSize, defaultBlockStride, defaultCellSize, 16, false},
    {"Bins4", defaultWinSize, defaultBlockSize, defaultBlockStride, defaultCellSize, 4, false},
    {"SignedGradient", defaultWinSize, defaultBlockSize, defaultBlockStride, defaultCellSize, defaultNbins, true},
    {"Block16x24", defaultWinSize, Size(16, 24), defaultBlockStride, defaultCellSize, defaultNbins, false},
    {"Block32x32", defaultWinSize, Size(32, 32), defaultBlockStride, defaultCellSize, defaultNbins, false},
    {"Cell4x4", defaultWinSize, defaultBlockSize, defaultBlockStride, Size(4, 4), defaultNbins, false},
};

// An empty window stride selects the block stride of the geometry.
const Placement placements[] = {
    {"BlockStride", Size(), Size(), {}},
    {"WindowStride4x4", Size(4, 4), Size(), {}},
    {"WindowStride12x12", Size(12, 12), Size(), {}},
    {"WindowStride12x8", Size(12, 8), Size(), {}},
    {"WindowStride8x12", Size(8, 12), Size(), {}},
    {"Padding8x8", Size(), Size(8, 8), {}},
    {"Locations", Size(), Size(), {Point(8, 8), Point(24, 16)}},
};

// The OpenCL implementation requires a scale factor above one.
const double pyramidScale = 1.05;

const Search searches[] = {
    {"NoGrouping", 0, false, Size(), pyramidScale},
    {"RectangleGrouping", 1, false, Size(), pyramidScale},
    {"MeanShiftGrouping", 0, true, Size(), pyramidScale},
    {"Padding8x8", 0, false, Size(8, 8), pyramidScale},
    {"Scale1", 0, false, Size(), 1},
};

//! Name of a comparison made of the names of its parameters.
template <typename Variant>
std::string caseName(const testing::TestParamInfo<testing::tuple<Geometry, bool, Variant> >& info)
{
    const char* gamma = testing::get<1>(info.param) ? "Gamma" : "NoGamma";
    return std::string(testing::get<0>(info.param).name) + "_" + gamma + "_" +
           testing::get<2>(info.param).name;
}

} // namespace

PARAM_TEST_CASE(HOGCompute, Geometry, bool, Placement)
{
};

OCL_TEST_P(HOGCompute, MatchesCPU)
{
    const Geometry& g = GET_PARAM(0);
    const Placement& placement = GET_PARAM(2);
    const HOGDescriptor hog = makeDescriptor(g, GET_PARAM(1));
    const Size winStride = placement.winStride.empty() ? g.blockStride : placement.winStride;
    const double tolerance = 1e-3;
    const Mat image = makeImage();

    std::vector<float> expected;
    OCL_OFF(hog.compute(image, expected, winStride, placement.padding, placement.locations));

    // The UMat is created only once OpenCL is enabled. A UMat created while it
    // is disabled lives in host memory, which the kernels cannot bind, so the
    // OpenCL implementation would fall back to the CPU.
    std::vector<float> actual;
    OCL_ON(hog.compute(image.getUMat(ACCESS_READ), actual, winStride, placement.padding,
                       placement.locations));

    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_LE(cvtest::norm(Mat(expected), Mat(actual), NORM_INF), tolerance);
}

// The second parameter enables gamma correction, which takes the square root of
// the intensities before computing the gradients. The default constructor
// enables it, while the constructor taking the block layout disables it.
INSTANTIATE_TEST_CASE_P(OCL_ObjDetect, HOGCompute,
                        testing::Combine(testing::ValuesIn(geometries), testing::Bool(),
                                         testing::ValuesIn(placements)),
                        caseName<Placement>);

PARAM_TEST_CASE(HOGDetectMultiScale, Geometry, bool, Search)
{
    HOGDescriptor hog;
    Mat image;

    // Shifts the bias of a random detector so that the hit threshold of zero
    // lies in the widest gap between two window scores. The rounding
    // differences of OpenCL then cannot move a window across the threshold,
    // and every hit has a positive weight, which mean shift grouping requires
    // to keep it. At least half of the windows stay hits, so that rectangle
    // grouping finds overlapping hits.
    void SetUp() override
    {
        makeDescriptor(GET_PARAM(0), GET_PARAM(1)).copyTo(hog);
        // A single pyramid level keeps the detections independent of resizing.
        hog.nlevels = 1;
        image = makeImage();

        std::vector<float> svm(hog.getDescriptorSize() + 1);
        RNG svmRng(1);
        svmRng.fill(svm, RNG::UNIFORM, -1, 1);
        hog.setSVMDetector(svm);

        std::vector<Rect> all;
        std::vector<double> scores;
        OCL_OFF(hog.detectMultiScale(image, all, scores, -DBL_MAX, Size(), Size(), pyramidScale, 0));
        ASSERT_GE(scores.size(), 2u);

        std::sort(scores.begin(), scores.end());
        const size_t lastGap = scores.size() / 2;
        size_t gap = 1;
        for (size_t i = 2; i <= lastGap; ++i)
        {
            if (scores[i] - scores[i - 1] > scores[gap] - scores[gap - 1])
            {
                gap = i;
            }
        }

        svm.back() -= float((scores[gap] + scores[gap - 1]) / 2);
        hog.setSVMDetector(svm);
    }
};

OCL_TEST_P(HOGDetectMultiScale, MatchesCPU)
{
    const Search& search = GET_PARAM(2);
    const Size winStride;
    const double hitThreshold = 0;
    const double scoreTolerance = 1e-4;

    std::vector<Rect> expected;
    std::vector<double> expectedWeights;
    OCL_OFF(hog.detectMultiScale(image, expected, expectedWeights, hitThreshold, winStride,
                                 search.padding, search.scale, search.groupThreshold,
                                 search.useMeanshiftGrouping));
    ASSERT_FALSE(expected.empty());
    ASSERT_EQ(expected.size(), expectedWeights.size());

    // The UMat is created only once OpenCL is enabled. A UMat created while it
    // is disabled lives in host memory, which the kernels cannot bind, so the
    // OpenCL implementation would fall back to the CPU.
    std::vector<Rect> actual;
    OCL_ON(hog.detectMultiScale(image.getUMat(ACCESS_READ), actual, hitThreshold, winStride,
                                search.padding, search.scale, search.groupThreshold,
                                search.useMeanshiftGrouping));

    std::vector<Rect> weighted;
    std::vector<double> actualWeights;
    OCL_ON(hog.detectMultiScale(image.getUMat(ACCESS_READ), weighted, actualWeights,
                                hitThreshold, winStride,
                                search.padding, search.scale, search.groupThreshold,
                                search.useMeanshiftGrouping));
    ASSERT_EQ(weighted.size(), actualWeights.size());

    // Both implementations may report the detections in any order.
    std::vector<std::pair<Rect, double> > expectedHits;
    std::vector<std::pair<Rect, double> > weightedHits;
    auto makeHit = [](const Rect& r, double weight) { return std::make_pair(r, weight); };
    std::transform(expected.begin(), expected.end(), expectedWeights.begin(),
                   std::back_inserter(expectedHits), makeHit);
    std::transform(weighted.begin(), weighted.end(), actualWeights.begin(),
                   std::back_inserter(weightedHits), makeHit);
    auto byPosition = [](const Rect& a, const Rect& b) {
        return std::make_tuple(a.y, a.x, a.height, a.width) <
               std::make_tuple(b.y, b.x, b.height, b.width);
    };
    auto hitByPosition = [&byPosition](const std::pair<Rect, double>& a,
                                       const std::pair<Rect, double>& b) {
        return byPosition(a.first, b.first);
    };
    std::sort(expected.begin(), expected.end(), byPosition);
    std::sort(actual.begin(), actual.end(), byPosition);
    std::sort(expectedHits.begin(), expectedHits.end(), hitByPosition);
    std::sort(weightedHits.begin(), weightedHits.end(), hitByPosition);

    EXPECT_EQ(expected, actual);
    ASSERT_EQ(expectedHits.size(), weightedHits.size());
    for (size_t i = 0; i < expectedHits.size(); ++i)
    {
        EXPECT_EQ(expectedHits[i].first, weightedHits[i].first);
        EXPECT_NEAR(expectedHits[i].second, weightedHits[i].second, scoreTolerance);
    }
}

INSTANTIATE_TEST_CASE_P(OCL_ObjDetect, HOGDetectMultiScale,
                        testing::Combine(testing::ValuesIn(geometries), testing::Bool(),
                                         testing::ValuesIn(searches)),
                        caseName<Search>);

}} // namespace
#endif
