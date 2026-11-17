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

TEST( Features2d_DescriptorExtractor_KAZE, compute_after_detect )
{
    RNG& rng = theRNG();
    Mat img(240, 320, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    GaussianBlur(img, img, Size(9, 9), 2.0);
    rectangle(img, Point(40, 40), Point(160, 140), Scalar(245), FILLED);
    rectangle(img, Point(120, 90), Point(260, 200), Scalar(15), FILLED);
    circle(img, Point(200, 70), 28, Scalar(235), FILLED);
    GaussianBlur(img, img, Size(5, 5), 1.0);

    for (int ext_idx = 0; ext_idx < 2; ++ext_idx)
    {
        for (int up_idx = 0; up_idx < 2; ++up_idx)
        {
            const bool extended = (ext_idx != 0);
            const bool upright = (up_idx != 0);
            Ptr<KAZE> kaze = KAZE::create(extended, upright);

            vector<KeyPoint> kp_dc, kp_det;
            Mat desc_dc, desc_comp;
            kaze->detectAndCompute(img, noArray(), kp_dc, desc_dc);
            ASSERT_FALSE(kp_dc.empty());

            kaze->detect(img, kp_det);
            ASSERT_EQ(kp_dc.size(), kp_det.size());

            kaze->compute(img, kp_det, desc_comp);
            ASSERT_EQ(desc_dc.size(), desc_comp.size());
            EXPECT_EQ(0.0, cvtest::norm(desc_dc, desc_comp, NORM_INF))
                << "extended=" << extended << " upright=" << upright;

            float max_angle_diff = 0.0f;
            for (size_t k = 0; k < kp_dc.size(); ++k)
                max_angle_diff = std::max(max_angle_diff, std::fabs(kp_dc[k].angle - kp_det[k].angle));
            EXPECT_EQ(0.0f, max_angle_diff)
                << "extended=" << extended << " upright=" << upright;

            vector<KeyPoint> kp_rev(kp_dc.rbegin(), kp_dc.rend());
            Mat desc_rev, desc_dc_flipped;
            kaze->compute(img, kp_rev, desc_rev);
            flip(desc_dc, desc_dc_flipped, 0);
            ASSERT_EQ(desc_dc_flipped.size(), desc_rev.size());
            EXPECT_EQ(0.0, cvtest::norm(desc_dc_flipped, desc_rev, NORM_INF))
                << "extended=" << extended << " upright=" << upright;

            if (upright)
            {
                UMat uimg, udesc_dc, udesc_rev, udesc_empty;
                img.copyTo(uimg);
                vector<KeyPoint> ukp_dc;
                kaze->detectAndCompute(uimg, noArray(), ukp_dc, udesc_dc);
                ASSERT_FALSE(ukp_dc.empty());

                vector<KeyPoint> ukp_rev(ukp_dc.rbegin(), ukp_dc.rend());
                kaze->compute(uimg, ukp_rev, udesc_rev);
                Mat udesc_dc_flipped;
                flip(udesc_dc.getMat(ACCESS_READ), udesc_dc_flipped, 0);
                ASSERT_EQ(udesc_dc_flipped.size(), udesc_rev.size());
                EXPECT_EQ(0.0, cvtest::norm(udesc_dc_flipped, udesc_rev.getMat(ACCESS_READ), NORM_INF))
                    << "UMat extended=" << extended;

                vector<KeyPoint> empty_kps;
                ASSERT_NO_THROW(kaze->compute(uimg, empty_kps, udesc_empty));
                EXPECT_TRUE(udesc_empty.empty());

                // a level outside the pyramid is rejected on the OpenCL path, as on the CPU path
                vector<KeyPoint> ukp_bad(1, ukp_dc[0]);
                ukp_bad[0].class_id = -1;
                UMat udesc_bad;
                EXPECT_THROW(kaze->compute(uimg, ukp_bad, udesc_bad), cv::Exception) << "UMat extended=" << extended;
            }
        }
    }

    // Level 0 (class_id == 0) is never emitted by detect(), so check caller-supplied level-0 keypoint
    Ptr<KAZE> kaze = KAZE::create();
    vector<KeyPoint> kp_lvl0(1, KeyPoint(120.0f, 90.0f, 6.4f, 0.0f, 1.0f, 0, 0));
    Mat desc_lvl0;
    kaze->compute(img, kp_lvl0, desc_lvl0);
    ASSERT_EQ(Size(64, 1), desc_lvl0.size());
    EXPECT_TRUE(checkRange(desc_lvl0));
    EXPECT_NEAR(1.0, cv::norm(desc_lvl0, NORM_L2), 1e-5);

    // BGR input with non-default diffusivity
    Mat bgr(200, 240, CV_8UC3);
    rng.fill(bgr, RNG::UNIFORM, 0, 256);
    rectangle(bgr, Point(35, 35), Point(130, 130), Scalar(250, 250, 250), FILLED);
    rectangle(bgr, Point(95, 80), Point(205, 175), Scalar(10, 10, 10), FILLED);
    Ptr<KAZE> kaze_charb = KAZE::create(false, false, 0.001f, 4, 4, KAZE::DIFF_CHARBONNIER);
    vector<KeyPoint> kp_bgr1, kp_bgr2;
    Mat desc_bgr1, desc_bgr2;
    kaze_charb->detectAndCompute(bgr, noArray(), kp_bgr1, desc_bgr1);
    ASSERT_FALSE(kp_bgr1.empty());
    kaze_charb->detect(bgr, kp_bgr2);
    kaze_charb->compute(bgr, kp_bgr2, desc_bgr2);
    EXPECT_EQ(0.0, cvtest::norm(desc_bgr1, desc_bgr2, NORM_INF));

    // Single-thread vs multi-thread parity with RAII thread-count restoration
    struct ThreadCountGuard
    {
        int prev;
        explicit ThreadCountGuard(int n) : prev(getNumThreads()) { setNumThreads(n); }
        ~ThreadCountGuard() { setNumThreads(prev); }
    };
    vector<KeyPoint> kp_ref, kp_1t, kp_mt;
    Mat desc_ref, desc_1t, desc_mt;
    kaze->detectAndCompute(img, noArray(), kp_ref, desc_ref);
    kaze->detect(img, kp_1t);
    kp_mt = kp_1t;
    const int mt_threads = std::max(2, getNumThreads());
    {
        ThreadCountGuard guard(1);
        kaze->compute(img, kp_1t, desc_1t);
    }
    {
        ThreadCountGuard guard(mt_threads);
        kaze->compute(img, kp_mt, desc_mt);
    }
    EXPECT_EQ(0.0, cvtest::norm(desc_ref, desc_1t, NORM_INF));
    EXPECT_EQ(0.0, cvtest::norm(desc_1t, desc_mt, NORM_INF));
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
