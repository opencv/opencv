// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"
#include <functional>

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

#define TEST_IMAGES testing::Values(\
    "detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "../stitching/a3.png", \
    "../stitching/s2.jpg")

PARAM_TEST_CASE(Feature2DFixture, std::function<Ptr<Feature2D>()>, std::string, double)
{
    std::string filename;
    double desc_eps;
    Mat image, descriptors;
    vector<KeyPoint> keypoints;
    UMat uimage, udescriptors;
    vector<KeyPoint> ukeypoints;
    Ptr<Feature2D> feature;

    virtual void SetUp()
    {
        feature = GET_PARAM(0)();
        filename = GET_PARAM(1);
        desc_eps = GET_PARAM(2);

        image = readImage(filename);
        ASSERT_FALSE(image.empty());
        image.copyTo(uimage);

        OCL_OFF(feature->detect(image, keypoints));
        OCL_ON(feature->detect(uimage, ukeypoints));
        OCL_OFF(feature->compute(image, keypoints, descriptors));
        OCL_ON(feature->compute(uimage, keypoints, udescriptors));
    }
};

OCL_TEST_P(Feature2DFixture, KeypointsSame)
{
    size_t count_diff = (keypoints.size() > ukeypoints.size()) ?
                        keypoints.size() - ukeypoints.size() : ukeypoints.size() - keypoints.size();
    EXPECT_LE(count_diff, (size_t)(std::min(keypoints.size(), ukeypoints.size()) * 0.20 + 1));

    std::vector<KeyPoint> cpu_sorted = keypoints, ocl_sorted = ukeypoints;
    std::sort(cpu_sorted.begin(), cpu_sorted.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.pt.x < b.pt.x || (a.pt.x == b.pt.x && a.pt.y < b.pt.y);
    });
    std::sort(ocl_sorted.begin(), ocl_sorted.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.pt.x < b.pt.x || (a.pt.x == b.pt.x && a.pt.y < b.pt.y);
    });

    int matched = 0;
    size_t j = 0;
    for (size_t i = 0; i < cpu_sorted.size(); i++)
    {
        while (j < ocl_sorted.size() && ocl_sorted[j].pt.x < cpu_sorted[i].pt.x - 2.0f)
            j++;
        for (size_t k = j; k < ocl_sorted.size() && ocl_sorted[k].pt.x <= cpu_sorted[i].pt.x + 2.0f; k++)
        {
            if (std::abs(ocl_sorted[k].pt.y - cpu_sorted[i].pt.y) < 2.0f)
            {
                matched++;
                break;
            }
        }
    }
    size_t n = std::min(cpu_sorted.size(), ocl_sorted.size());
    EXPECT_GE(matched, (int)(n * 0.70));
}

OCL_TEST_P(Feature2DFixture, DescriptorsSame)
{
    EXPECT_EQ(descriptors.size(), udescriptors.size());
    ASSERT_EQ(descriptors.type(), udescriptors.type());
    Mat udesc = udescriptors.getMat(ACCESS_READ);
    double max_diff = 0.0;
    for (int r = 0; r < descriptors.rows; r++)
    {
        for (int c = 0; c < descriptors.cols; c++)
        {
            double d;
            if (descriptors.type() == CV_8U)
                d = std::abs((double)descriptors.at<uchar>(r, c) - (double)udesc.at<uchar>(r, c));
            else
                d = std::abs((double)descriptors.at<float>(r, c) - (double)udesc.at<float>(r, c));
            max_diff = std::max(max_diff, d);
        }
    }
    EXPECT_LE(max_diff, desc_eps);
}

OCL_INSTANTIATE_TEST_CASE_P(SIFT, Feature2DFixture,
    testing::Combine(testing::Values([]() { return SIFT::create(); }), TEST_IMAGES, testing::Values(2.0)));

OCL_TEST(SIFT, NonDefaultNOctaveLayers)
{
    Mat image = TestUtils::readImage("../stitching/s2.jpg", IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());
    UMat uimage;
    image.copyTo(uimage);

    Ptr<SIFT> sift = SIFT::create(0, 4);
    vector<KeyPoint> keypoints;
    UMat descriptors;
    ASSERT_NO_THROW(sift->detectAndCompute(uimage, noArray(), keypoints, descriptors, false));
    EXPECT_GT(keypoints.size(), 20u);
    EXPECT_EQ((size_t)descriptors.rows, keypoints.size());
}

OCL_TEST(SIFT, DescriptorType)
{
    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());
    UMat uimage;
    image.copyTo(uimage);

    vector<KeyPoint> keypoints;
    UMat descriptorsFloat, descriptorsUchar;

    Ptr<SIFT> siftFloat = SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    siftFloat->detectAndCompute(uimage, noArray(), keypoints, descriptorsFloat, false);
    ASSERT_EQ(descriptorsFloat.type(), CV_32F) << "type mismatch";

    Ptr<SIFT> siftUchar = SIFT::create(0, 3, 0.04, 10, 1.6, CV_8U);
    siftUchar->detectAndCompute(uimage, noArray(), keypoints, descriptorsUchar, false);
    ASSERT_EQ(descriptorsUchar.type(), CV_8U) << "type mismatch";

    Mat df = descriptorsFloat.getMat(ACCESS_READ);
    Mat du = descriptorsUchar.getMat(ACCESS_READ);
    Mat descriptorsFloat2;
    du.assignTo(descriptorsFloat2, CV_32F);
    Mat diff = df != descriptorsFloat2;
    EXPECT_EQ(countNonZero(diff), 0) << "descriptors are not identical";
}

OCL_TEST(SIFT, Regression_26139)
{
    UMat uimage(Size(300, 300), CV_8UC1, Scalar::all(0));
    std::vector<KeyPoint> kps {
        KeyPoint(154.076813f, 136.160904f, 111.078636f, 216.195618f, 0.00000899323549f, 7)
    };
    Ptr<SIFT> extractor = SIFT::create();
    UMat descriptors;
    extractor->compute(uimage, kps, descriptors);
    ASSERT_EQ(descriptors.size(), Size(128, 1));
}

OCL_TEST(SIFT, Batch)
{
    string path = cvtest::TS::ptr()->get_data_path() + "detectors_descriptors_evaluation/images_datasets/graf";
    vector<UMat> imgs;
    vector<UMat> descriptors;
    vector<vector<KeyPoint> > keypoints;
    int n = 6;
    Ptr<SIFT> sift = SIFT::create();

    for (int i = 0; i < n; i++)
    {
        string imgname = format("%s/img%d.png", path.c_str(), i + 1);
        Mat img = imread(imgname, IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty()) << "Failed to load " << imgname;
        UMat uimg;
        img.copyTo(uimg);
        imgs.push_back(uimg);
    }

    sift->detect(imgs, keypoints);
    sift->compute(imgs, keypoints, descriptors);

    ASSERT_EQ((int)keypoints.size(), n);
    ASSERT_EQ((int)descriptors.size(), n);

    for (int i = 0; i < n; i++)
    {
        EXPECT_GT((int)keypoints[i].size(), 100);
        EXPECT_GT(descriptors[i].rows, 100);
    }
}

}//ocl
}//opencv_test

#endif //HAVE_OPENCL
