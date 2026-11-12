// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {
const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "tsukuba.png";
const string DESCRIPTOR_DIR = FEATURES2D_DIR + "/descriptor_extractors";
}} // namespace

#include "test_descriptors_regression.impl.hpp"

namespace opencv_test { namespace {

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

TEST( Features2d_DescriptorExtractor_SIFT, regression )
{
    CV_DescriptorExtractorTest<L1<float> > test( "descriptor-sift", 1.0f,
                                                SIFT::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BRISK, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-brisk",
                                             (CV_DescriptorExtractorTest<Hamming>::DistanceType)2.f,
                                            BRISK::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_ORB, regression )
{
    // TODO adjust the parameters below
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-orb",
#if CV_NEON
                                              (CV_DescriptorExtractorTest<Hamming>::DistanceType)25.f,
#else
                                              (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
#endif
                                             ORB::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_KAZE, regression )
{
    CV_DescriptorExtractorTest< L2<float> > test( "descriptor-kaze",  0.03f,
                                                 KAZE::create(),
                                                 L2<float>(), KAZE::create() );
    test.safe_run();
}

// compute() on the keypoints of detect() gives the descriptors of detectAndCompute()
TEST( Features2d_DescriptorExtractor_KAZE, compute_after_detect )
{
    Mat img(240, 320, CV_8UC1);
    theRNG().fill(img, RNG::UNIFORM, 0, 255);
    GaussianBlur(img, img, Size(9, 9), 2.0);
    rectangle(img, Point(40, 40), Point(160, 140), Scalar(240), -1);
    rectangle(img, Point(120, 90), Point(260, 200), Scalar(30), -1);
    GaussianBlur(img, img, Size(5, 5), 1.0);

    Ptr<KAZE> kaze = KAZE::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    kaze->detectAndCompute(img, noArray(), keypoints, descriptors);
    ASSERT_FALSE(keypoints.empty());

    vector<KeyPoint> keypoints2;
    Mat descriptors2;
    kaze->detect(img, keypoints2);
    kaze->compute(img, keypoints2, descriptors2);

    ASSERT_EQ(keypoints.size(), keypoints2.size());
    EXPECT_EQ(0, cvtest::norm(descriptors, descriptors2, NORM_INF));
}

TEST( Features2d_DescriptorExtractor_SIFT, compute_after_detect )
{
    Mat img(240, 320, CV_8UC1);
    theRNG().fill(img, RNG::UNIFORM, 0, 255);
    GaussianBlur(img, img, Size(9, 9), 2.0);
    rectangle(img, Point(40, 40), Point(160, 140), Scalar(240), -1);
    rectangle(img, Point(120, 90), Point(260, 200), Scalar(30), -1);
    GaussianBlur(img, img, Size(5, 5), 1.0);

    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(img, noArray(), keypoints, descriptors);
    ASSERT_FALSE(keypoints.empty());

    vector<KeyPoint> keypoints2;
    Mat descriptors2;
    sift->detect(img, keypoints2);
    sift->compute(img, keypoints2, descriptors2);

    ASSERT_EQ(keypoints.size(), keypoints2.size());
    EXPECT_EQ(0, cvtest::norm(descriptors, descriptors2, NORM_INF));
}

TEST( Features2d_DescriptorExtractor_AKAZE, compute_after_detect )
{
    Mat img(240, 320, CV_8UC3);
    theRNG().fill(img, RNG::UNIFORM, 0, 255);
    GaussianBlur(img, img, Size(9, 9), 2.0);
    rectangle(img, Point(40, 40), Point(160, 140), Scalar(240, 200, 180), -1);
    rectangle(img, Point(120, 90), Point(260, 200), Scalar(30, 60, 90), -1);
    GaussianBlur(img, img, Size(5, 5), 1.0);

    const AKAZE::DescriptorType types[] = {
        AKAZE::DESCRIPTOR_MLDB,
        AKAZE::DESCRIPTOR_MLDB_UPRIGHT,
        AKAZE::DESCRIPTOR_KAZE,
        AKAZE::DESCRIPTOR_KAZE_UPRIGHT
    };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); ++t)
    {
        Ptr<AKAZE> akaze = AKAZE::create(types[t]);
        vector<KeyPoint> keypoints, keypoints2;
        Mat descriptors, descriptors2;
        akaze->detectAndCompute(img, noArray(), keypoints, descriptors);
        ASSERT_FALSE(keypoints.empty());

        akaze->detect(img, keypoints2);
        ASSERT_EQ(keypoints.size(), keypoints2.size());
        for (size_t k = 0; k < keypoints.size(); ++k)
            EXPECT_NEAR(keypoints[k].angle, keypoints2[k].angle, 1e-5f);

        akaze->compute(img, keypoints2, descriptors2);
        EXPECT_EQ(0, cvtest::norm(descriptors, descriptors2, NORM_INF));
    }
}

TEST( Features2d_DescriptorExtractor_AKAZE, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-akaze",
                                              (CV_DescriptorExtractorTest<Hamming>::DistanceType)(486*0.05f),
                                              AKAZE::create(),
                                              Hamming(), AKAZE::create());
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_AKAZE_DESCRIPTOR_KAZE, regression )
{
    CV_DescriptorExtractorTest< L2<float> > test( "descriptor-akaze-with-kaze-desc", 0.03f,
                                              AKAZE::create(AKAZE::DESCRIPTOR_KAZE),
                                              L2<float>(), AKAZE::create(AKAZE::DESCRIPTOR_KAZE));
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor, batch_ORB )
{
    string path = string(cvtest::TS::ptr()->get_data_path() + "detectors_descriptors_evaluation/images_datasets/graf");
    vector<Mat> imgs, descriptors;
    vector<vector<KeyPoint> > keypoints;
    int i, n = 6;
    Ptr<ORB> orb = ORB::create();

    for( i = 0; i < n; i++ )
    {
        string imgname = format("%s/img%d.png", path.c_str(), i+1);
        Mat img = imread(imgname, IMREAD_GRAYSCALE);
        imgs.push_back(img);
    }

    orb->detect(imgs, keypoints);
    orb->compute(imgs, keypoints, descriptors);

    ASSERT_EQ((int)keypoints.size(), n);
    ASSERT_EQ((int)descriptors.size(), n);

    for( i = 0; i < n; i++ )
    {
        EXPECT_GT((int)keypoints[i].size(), 100);
        EXPECT_GT(descriptors[i].rows, 100);
    }
}

TEST( Features2d_DescriptorExtractor, batch_SIFT )
{
    string path = string(cvtest::TS::ptr()->get_data_path() + "detectors_descriptors_evaluation/images_datasets/graf");
    vector<Mat> imgs, descriptors;
    vector<vector<KeyPoint> > keypoints;
    int i, n = 6;
    Ptr<SIFT> sift = SIFT::create();

    for( i = 0; i < n; i++ )
    {
        string imgname = format("%s/img%d.png", path.c_str(), i+1);
        Mat img = imread(imgname, IMREAD_GRAYSCALE);
        imgs.push_back(img);
    }

    sift->detect(imgs, keypoints);
    sift->compute(imgs, keypoints, descriptors);

    ASSERT_EQ((int)keypoints.size(), n);
    ASSERT_EQ((int)descriptors.size(), n);

    for( i = 0; i < n; i++ )
    {
        EXPECT_GT((int)keypoints[i].size(), 100);
        EXPECT_GT(descriptors[i].rows, 100);
    }
}


class DescriptorImage : public TestWithParam<std::string>
{
protected:
    virtual void SetUp() {
        pattern = GetParam();
    }

    std::string pattern;
};

TEST_P(DescriptorImage, no_crash)
{
    vector<String> fnames;
    glob(cvtest::TS::ptr()->get_data_path() + pattern, fnames, false);
    std::sort(fnames.begin(), fnames.end());

    Ptr<AKAZE> akaze_mldb = AKAZE::create(AKAZE::DESCRIPTOR_MLDB);
    Ptr<AKAZE> akaze_mldb_upright = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT);
    Ptr<AKAZE> akaze_mldb_256 = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 256);
    Ptr<AKAZE> akaze_mldb_upright_256 = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 256);
    Ptr<AKAZE> akaze_kaze = AKAZE::create(AKAZE::DESCRIPTOR_KAZE);
    Ptr<AKAZE> akaze_kaze_upright = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
    Ptr<ORB> orb = ORB::create();
    Ptr<KAZE> kaze = KAZE::create();
    Ptr<BRISK> brisk = BRISK::create();
    size_t n = fnames.size();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->setMaxFeatures(5000);

    for(size_t i = 0; i < n; i++ )
    {
        printf("%d. image: %s:\n", (int)i, fnames[i].c_str());
        if( strstr(fnames[i].c_str(), "MP.png") != 0 )
        {
            printf("\tskip\n");
            continue;
        }
        bool checkCount = strstr(fnames[i].c_str(), "templ.png") == 0;

        Mat img = imread(fnames[i], -1);

        printf("\t%dx%d\n", img.cols, img.rows);

#define TEST_DETECTOR(name, descriptor) \
        keypoints.clear(); descriptors.release(); \
        printf("\t" name "\n"); fflush(stdout); \
        descriptor->detectAndCompute(img, noArray(), keypoints, descriptors); \
        printf("\t\t\t(%d keypoints, descriptor size = %d)\n", (int)keypoints.size(), descriptors.cols); fflush(stdout); \
        if (checkCount) \
        { \
            EXPECT_GT((int)keypoints.size(), 0); \
        } \
        ASSERT_EQ(descriptors.rows, (int)keypoints.size());

        TEST_DETECTOR("AKAZE:MLDB", akaze_mldb);
        TEST_DETECTOR("AKAZE:MLDB_UPRIGHT", akaze_mldb_upright);
        TEST_DETECTOR("AKAZE:MLDB_256", akaze_mldb_256);
        TEST_DETECTOR("AKAZE:MLDB_UPRIGHT_256", akaze_mldb_upright_256);
        TEST_DETECTOR("AKAZE:KAZE", akaze_kaze);
        TEST_DETECTOR("AKAZE:KAZE_UPRIGHT", akaze_kaze_upright);
        TEST_DETECTOR("KAZE", kaze);
        TEST_DETECTOR("ORB", orb);
        TEST_DETECTOR("BRISK", brisk);
    }
}

INSTANTIATE_TEST_CASE_P(Features2d, DescriptorImage,
        testing::Values(
            "shared/lena.png",
            "shared/box*.png",
            "shared/fruits*.png",
            "shared/airplane.png",
            "shared/graffiti.png",
            "shared/1_itseez-0001*.png",
            "shared/pic*.png",
            "shared/templ.png"
        )
);

}} // namespace
