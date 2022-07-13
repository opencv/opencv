// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static
void dilateFrame(Mat& image, Mat& depth)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC1);

    CV_Assert(!depth.empty());
    CV_Assert(depth.type() == CV_32FC1);
    CV_Assert(depth.size() == image.size());

    Mat mask(image.size(), CV_8UC1, Scalar(255));
    for(int y = 0; y < depth.rows; y++)
        for(int x = 0; x < depth.cols; x++)
            if(cvIsNaN(depth.at<float>(y,x)) || depth.at<float>(y,x) > 10 || depth.at<float>(y,x) <= FLT_EPSILON)
                mask.at<uchar>(y,x) = 0;

    image.setTo(255, ~mask);
    Mat minImage;
    erode(image, minImage, Mat());

    image.setTo(0, ~mask);
    Mat maxImage;
    dilate(image, maxImage, Mat());

    depth.setTo(FLT_MAX, ~mask);
    Mat minDepth;
    erode(depth, minDepth, Mat());

    depth.setTo(0, ~mask);
    Mat maxDepth;
    dilate(depth, maxDepth, Mat());

    Mat dilatedMask;
    dilate(mask, dilatedMask, Mat(), Point(-1,-1), 1);
    for(int y = 0; y < depth.rows; y++)
        for(int x = 0; x < depth.cols; x++)
            if(!mask.at<uchar>(y,x) && dilatedMask.at<uchar>(y,x))
            {
                image.at<uchar>(y,x) = static_cast<uchar>(0.5f * (static_cast<float>(minImage.at<uchar>(y,x)) +
                                                                  static_cast<float>(maxImage.at<uchar>(y,x))));
                depth.at<float>(y,x) = 0.5f * (minDepth.at<float>(y,x) + maxDepth.at<float>(y,x));
            }
}

class OdometryTest
{
public:
    OdometryTest(OdometryType _otype,
                 OdometryAlgoType _algtype,
                 double _maxError1,
                 double _maxError5,
                 double _idError = DBL_EPSILON) :
        otype(_otype),
        algtype(_algtype),
        maxError1(_maxError1),
        maxError5(_maxError5),
        idError(_idError)
    { }

    void readData(Mat& image, Mat& depth) const;
    static Mat getCameraMatrix()
    {
        float fx = 525.0f, // default
              fy = 525.0f,
              cx = 319.5f,
              cy = 239.5f;
        Matx33f K(fx,  0, cx,
                   0, fy, cy,
                   0,  0,  1);
        return Mat(K);
    }
    static void generateRandomTransformation(Mat& R, Mat& t);

    void run();
    void checkUMats();
    void prepareFrameCheck();

    OdometryType otype;
    OdometryAlgoType algtype;
    double maxError1;
    double maxError5;
    double idError;
};


void OdometryTest::readData(Mat& image, Mat& depth) const
{
    std::string dataPath = cvtest::TS::ptr()->get_data_path();
    std::string imageFilename = dataPath + "/cv/rgbd/rgb.png";
    std::string depthFilename = dataPath + "/cv/rgbd/depth.png";

    image = imread(imageFilename,  0);
    depth = imread(depthFilename, -1);

    ASSERT_FALSE(image.empty()) << "Image " << imageFilename.c_str() << " can not be read" << std::endl;
    ASSERT_FALSE(depth.empty()) << "Depth " << depthFilename.c_str() << "can not be read" << std::endl;

    CV_DbgAssert(image.type() == CV_8UC1);
    CV_DbgAssert(depth.type() == CV_16UC1);
    {
        Mat depth_flt;
        depth.convertTo(depth_flt, CV_32FC1, 1.f/5000.f);
        depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depth_flt < FLT_EPSILON);
        depth = depth_flt;
    }
}

void OdometryTest::generateRandomTransformation(Mat& rvec, Mat& tvec)
{
    const float maxRotation = (float)(3.f / 180.f * CV_PI); //rad
    const float maxTranslation = 0.02f; //m

    RNG& rng = theRNG();
    rvec.create(3, 1, CV_64FC1);
    tvec.create(3, 1, CV_64FC1);

    randu(rvec, Scalar(-1000), Scalar(1000));
    normalize(rvec, rvec, rng.uniform(0.007f, maxRotation));

    randu(tvec, Scalar(-1000), Scalar(1000));
    normalize(tvec, tvec, rng.uniform(0.008f, maxTranslation));
}

void OdometryTest::checkUMats()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);

    OdometrySettings ods;
    ods.setCameraMatrix(K);
    Odometry odometry = Odometry(otype, ods, algtype);
    OdometryFrame odf = odometry.createOdometryFrame(OdometryFrameStoreType::UMAT);

    Mat calcRt;

    UMat uimage, udepth;
    image.copyTo(uimage);
    depth.copyTo(udepth);
    odf.setImage(uimage);
    odf.setDepth(udepth);
    uimage.release();
    udepth.release();

    odometry.prepareFrame(odf);
    bool isComputed = odometry.compute(odf, odf, calcRt);
    ASSERT_TRUE(isComputed);
    double diff = cv::norm(calcRt, Mat::eye(4, 4, CV_64FC1));
    ASSERT_FALSE(diff > idError) << "Incorrect transformation between the same frame (not the identity matrix), diff = " << diff << std::endl;
}

void OdometryTest::run()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);
    OdometrySettings ods;
    ods.setCameraMatrix(K);
    Odometry odometry = Odometry(otype, ods, algtype);
    OdometryFrame odf = odometry.createOdometryFrame();
    odf.setImage(image);
    odf.setDepth(depth);
    Mat calcRt;

    // 1. Try to find Rt between the same frame (try masks also).
    Mat mask(image.size(), CV_8UC1, Scalar(255));

    odometry.prepareFrame(odf);
    bool isComputed = odometry.compute(odf, odf, calcRt);

    ASSERT_TRUE(isComputed) << "Can not find Rt between the same frame" << std::endl;
    double ndiff = cv::norm(calcRt, Mat::eye(4,4,CV_64FC1));
    ASSERT_FALSE(ndiff > idError) << "Incorrect transformation between the same frame (not the identity matrix), diff = " << ndiff << std::endl;

    // 2. Generate random rigid body motion in some ranges several times (iterCount).
    // On each iteration an input frame is warped using generated transformation.
    // Odometry is run on the following pair: the original frame and the warped one.
    // Comparing a computed transformation with an applied one we compute 2 errors:
    // better_1time_count - count of poses which error is less than ground truth pose,
    // better_5times_count - count of poses which error is 5 times less than ground truth pose.
    int iterCount = 100;
    int better_1time_count = 0;
    int better_5times_count = 0;
    for (int iter = 0; iter < iterCount; iter++)
    {
        Mat rvec, tvec;
        generateRandomTransformation(rvec, tvec);
        Affine3d rt(rvec, tvec);

        Mat warpedImage, warpedDepth;

        warpFrame(depth, image, noArray(), rt.matrix, K, warpedDepth, warpedImage);
        dilateFrame(warpedImage, warpedDepth); // due to inaccuracy after warping

        OdometryFrame odfSrc = odometry.createOdometryFrame();
        OdometryFrame odfDst = odometry.createOdometryFrame();

        odfSrc.setImage(image);
        odfSrc.setDepth(depth);
        odfDst.setImage(warpedImage);
        odfDst.setDepth(warpedDepth);

        odometry.prepareFrames(odfSrc, odfDst);
        isComputed = odometry.compute(odfSrc, odfDst, calcRt);

        if (!isComputed)
        {
            CV_LOG_INFO(NULL, "Iter " << iter << "; Odometry compute returned false");
            continue;
        }
        Mat calcR = calcRt(Rect(0, 0, 3, 3)), calcRvec;
        cv::Rodrigues(calcR, calcRvec);
        calcRvec = calcRvec.reshape(rvec.channels(), rvec.rows);
        Mat calcTvec = calcRt(Rect(3,0,1,3));

        if (cvtest::debugLevel >= 10)
        {
            imshow("image", image);
            imshow("warpedImage", warpedImage);
            Mat resultImage, resultDepth;

            warpFrame(depth, image, noArray(), calcRt, K, resultDepth, resultImage);
            imshow("resultImage", resultImage);
            waitKey(100);
        }

        // compare rotation
        double possibleError = algtype == OdometryAlgoType::COMMON ? 0.11f : 0.015f;

        Affine3f src = Affine3f(Vec3f(rvec), Vec3f(tvec));
        Affine3f res = Affine3f(Vec3f(calcRvec), Vec3f(calcTvec));
        Affine3f src_inv = src.inv();
        Affine3f diff = res * src_inv;
        double rdiffnorm = cv::norm(diff.rvec());
        double tdiffnorm = cv::norm(diff.translation());

        if (rdiffnorm < possibleError && tdiffnorm < possibleError)
            better_1time_count++;
        if (5. * rdiffnorm < possibleError && 5 * tdiffnorm < possibleError)
            better_5times_count++;

        CV_LOG_INFO(NULL, "Iter " << iter);
        CV_LOG_INFO(NULL, "rdiff: " << Vec3f(diff.rvec()) << "; rdiffnorm: " << rdiffnorm);
        CV_LOG_INFO(NULL, "tdiff: " << Vec3f(diff.translation()) << "; tdiffnorm: " << tdiffnorm);

        CV_LOG_INFO(NULL, "better_1time_count " << better_1time_count << "; better_5time_count " << better_5times_count);
    }

    if(static_cast<double>(better_1time_count) < maxError1 * static_cast<double>(iterCount))
    {
        FAIL() << "Incorrect count of accurate poses [1st case]: "
            << static_cast<double>(better_1time_count) << " / "
            << maxError1 * static_cast<double>(iterCount) << std::endl;
    }

    if(static_cast<double>(better_5times_count) < maxError5 * static_cast<double>(iterCount))
    {
        FAIL() << "Incorrect count of accurate poses [2nd case]: "
            << static_cast<double>(better_5times_count) << " / "
            << maxError5 * static_cast<double>(iterCount) << std::endl;
    }
}

void OdometryTest::prepareFrameCheck()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);
    OdometrySettings ods;
    ods.setCameraMatrix(K);
    Odometry odometry = Odometry(otype, ods, algtype);
    OdometryFrame odf = odometry.createOdometryFrame();
    odf.setImage(image);
    odf.setDepth(depth);

    odometry.prepareFrame(odf);

    Mat points, mask;
    odf.getPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
    odf.getPyramidAt(mask, OdometryFramePyramidType::PYR_MASK, 0);

    OdometryFrame todf = odometry.createOdometryFrame();
    if (otype != OdometryType::DEPTH)
    {
        Mat img;
        odf.getPyramidAt(img, OdometryFramePyramidType::PYR_IMAGE, 0);
        todf.setPyramidLevel(1, OdometryFramePyramidType::PYR_IMAGE);
        todf.setPyramidAt(img, OdometryFramePyramidType::PYR_IMAGE, 0);
    }
    todf.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
    todf.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
    todf.setPyramidLevel(1, OdometryFramePyramidType::PYR_MASK);
    todf.setPyramidAt(mask, OdometryFramePyramidType::PYR_MASK, 0);

    odometry.prepareFrame(todf);
}

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

TEST(RGBD_Odometry_Rgbd, algorithmic)
{
    OdometryTest test(OdometryType::RGB, OdometryAlgoType::COMMON, 0.99, 0.89);
    test.run();
}

TEST(RGBD_Odometry_ICP, algorithmic)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.run();
}

TEST(RGBD_Odometry_RgbdICP, algorithmic)
{
    OdometryTest test(OdometryType::RGB_DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.run();
}

TEST(RGBD_Odometry_FastICP, algorithmic)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::FAST, 0.99, 0.89, FLT_EPSILON);
    test.run();
}


TEST(RGBD_Odometry_Rgbd, UMats)
{
    OdometryTest test(OdometryType::RGB, OdometryAlgoType::COMMON, 0.99, 0.89);
    test.checkUMats();
}

TEST(RGBD_Odometry_ICP, UMats)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.checkUMats();
}

TEST(RGBD_Odometry_RgbdICP, UMats)
{
    OdometryTest test(OdometryType::RGB_DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.checkUMats();
}

TEST(RGBD_Odometry_FastICP, UMats)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::FAST, 0.99, 0.89, FLT_EPSILON);
    test.checkUMats();
}


TEST(RGBD_Odometry_Rgbd, prepareFrame)
{
    OdometryTest test(OdometryType::RGB, OdometryAlgoType::COMMON, 0.99, 0.89);
    test.prepareFrameCheck();
}

TEST(RGBD_Odometry_ICP, prepareFrame)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.prepareFrameCheck();
}

TEST(RGBD_Odometry_RgbdICP, prepareFrame)
{
    OdometryTest test(OdometryType::RGB_DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.prepareFrameCheck();
}

TEST(RGBD_Odometry_FastICP, prepareFrame)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::FAST, 0.99, 0.89, FLT_EPSILON);
    test.prepareFrameCheck();
}


struct WarpFrameTest
{
    WarpFrameTest() :
     srcDepth(), srcRgb(), srcMask(),
     dstDepth(), dstRgb(), dstMask(),
     warpedDepth(), warpedRgb(), warpedMask()
      {}

    void run(bool needRgb, bool scaleDown, bool checkMask, bool identityTransform, int depthType, int imageType);

    Mat srcDepth, srcRgb, srcMask;
    Mat dstDepth, dstRgb, dstMask;
    Mat warpedDepth, warpedRgb, warpedMask;
};

void WarpFrameTest::run(bool needRgb, bool scaleDown, bool checkMask, bool identityTransform, int depthType, int rgbType)
{
    std::string dataPath = cvtest::TS::ptr()->get_data_path();
    std::string srcDepthFilename = dataPath + "/cv/rgbd/depth.png";
    std::string srcRgbFilename   = dataPath + "/cv/rgbd/rgb.png";
    // The depth was generated using the script at testdata/cv/rgbd/warped_depth_generator/warp_test.py
    std::string warpedDepthFilename = dataPath + "/cv/rgbd/warpedDepth.png";
    std::string warpedRgbFilename   = dataPath + "/cv/rgbd/warpedRgb.png";

    srcDepth = imread(srcDepthFilename, IMREAD_UNCHANGED);
    ASSERT_FALSE(srcDepth.empty())  << "Depth " << srcDepthFilename.c_str() << "can not be read" << std::endl;

    if (identityTransform)
    {
        warpedDepth = srcDepth;
    }
    else
    {
        warpedDepth = imread(warpedDepthFilename, IMREAD_UNCHANGED);
        ASSERT_FALSE(warpedDepth.empty()) << "Depth " << warpedDepthFilename.c_str() << "can not be read" << std::endl;
    }

    ASSERT_TRUE(srcDepth.type() == CV_16UC1);
    ASSERT_TRUE(warpedDepth.type() == CV_16UC1);

    Mat epsSrc = srcDepth > 0, epsWarped = warpedDepth > 0;

    const double depthFactor = 5000.0;
    // scale float types only
    double depthScaleCoeff = scaleDown ? ( depthType == CV_16U ? 1.          : 1./depthFactor ) :    1.;
    double transScaleCoeff = scaleDown ? ( depthType == CV_16U ? depthFactor : 1.             ) : depthFactor;

    Mat srcDepthCvt, warpedDepthCvt;
    srcDepth.convertTo(srcDepthCvt, depthType, depthScaleCoeff);
    srcDepth = srcDepthCvt;
    warpedDepth.convertTo(warpedDepthCvt, depthType, depthScaleCoeff);
    warpedDepth = warpedDepthCvt;

    Scalar badVal;
    switch (depthType)
    {
    case CV_16U:
        badVal = 0;
        break;
    case CV_32F:
        badVal = std::numeric_limits<float>::quiet_NaN();
        break;
    case CV_64F:
        badVal = std::numeric_limits<double>::quiet_NaN();
        break;
    default:
        CV_Error(Error::StsBadArg, "Unsupported depth data type");
    }

    srcDepth.setTo(badVal, ~epsSrc);
    warpedDepth.setTo(badVal, ~epsWarped);

    if (checkMask)
    {
        srcMask = epsSrc; warpedMask = epsWarped;
    }
    else
    {
        srcMask = Mat(); warpedMask = Mat();
    }

    if (needRgb)
    {
        srcRgb = imread(srcRgbFilename, rgbType == CV_8UC1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
        ASSERT_FALSE(srcRgb.empty()) << "Image " << srcRgbFilename.c_str() << "can not be read" << std::endl;

        if (identityTransform)
        {
            srcRgb.copyTo(warpedRgb, epsSrc);
        }
        else
        {
            warpedRgb = imread(warpedRgbFilename, rgbType == CV_8UC1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
            ASSERT_FALSE (warpedRgb.empty()) << "Image " << warpedRgbFilename.c_str() << "can not be read" << std::endl;
        }

        if (rgbType == CV_8UC4)
        {
            Mat newSrcRgb, newWarpedRgb;
            cvtColor(srcRgb, newSrcRgb, COLOR_RGB2RGBA);
            srcRgb = newSrcRgb;
            // let's keep alpha channel
            std::vector<Mat> warpedRgbChannels;
            split(warpedRgb, warpedRgbChannels);
            warpedRgbChannels.push_back(epsWarped);
            merge(warpedRgbChannels, newWarpedRgb);
            warpedRgb = newWarpedRgb;
        }

        ASSERT_TRUE(srcRgb.type() == rgbType);
        ASSERT_TRUE(warpedRgb.type() == rgbType);
    }
    else
    {
        srcRgb = Mat(); warpedRgb = Mat();
    }

    // test data used to generate warped depth and rgb
    // the script used to generate is in opencv_extra repo
    // at testdata/cv/rgbd/warped_depth_generator/warp_test.py
    double fx = 525.0, fy = 525.0,
           cx = 319.5, cy = 239.5;
    Matx33d K(fx,  0, cx,
               0, fy, cy,
               0,  0,  1);
    cv::Affine3d rt;
    cv::Vec3d tr(-0.04, 0.05, 0.6);
    rt = identityTransform ? cv::Affine3d() : cv::Affine3d(cv::Vec3d(0.1, 0.2, 0.3), tr * transScaleCoeff);

    warpFrame(srcDepth, srcRgb, srcMask, rt.matrix, K, dstDepth, dstRgb, dstMask);
}

typedef std::pair<int, int> WarpFrameInputTypes;
typedef testing::TestWithParam<WarpFrameInputTypes> WarpFrameInputs;

TEST_P(WarpFrameInputs, checkTypes)
{
    const double shortl2diff = 233.983;
    const double shortlidiff = 1;
    const double floatl2diff = 0.038209;
    const double floatlidiff = 0.00020004;

    int depthType = GetParam().first;
    int rgbType   = GetParam().second;

    WarpFrameTest w;
    // scale down does not happen on CV_16U
    // to avoid integer overflow
    w.run(/* needRgb */ true, /* scaleDown*/ true,
          /* checkMask */ true, /* identityTransform */ false, depthType, rgbType);

    double rgbDiff = cv::norm(w.dstRgb, w.warpedRgb, NORM_L2);
    double maskDiff = cv::norm(w.dstMask, w.warpedMask, NORM_L2);

    EXPECT_EQ(0, maskDiff);
    EXPECT_EQ(0, rgbDiff);

    double l2diff = cv::norm(w.dstDepth, w.warpedDepth, NORM_L2, w.warpedMask);
    double lidiff = cv::norm(w.dstDepth, w.warpedDepth, NORM_INF, w.warpedMask);

    double l2threshold = depthType == CV_16U ? shortl2diff : floatl2diff;
    double lithreshold = depthType == CV_16U ? shortlidiff : floatlidiff;

    EXPECT_LE(l2diff, l2threshold);
    EXPECT_LE(lidiff, lithreshold);
}

INSTANTIATE_TEST_CASE_P(RGBD_Odometry, WarpFrameInputs, ::testing::Values(
                        WarpFrameInputTypes { CV_16U, CV_8UC3 },
                        WarpFrameInputTypes { CV_32F, CV_8UC3 },
                        WarpFrameInputTypes { CV_64F, CV_8UC3 },
                        WarpFrameInputTypes { CV_32F, CV_8UC1 },
                        WarpFrameInputTypes { CV_32F, CV_8UC4 }));


TEST(RGBD_Odometry_WarpFrame, identity)
{
    WarpFrameTest w;
    w.run(/* needRgb */ true, /* scaleDown*/ true, /* checkMask */ true, /* identityTransform */ true, CV_32F, CV_8UC3);

    double rgbDiff = cv::norm(w.dstRgb, w.warpedRgb, NORM_L2);
    double maskDiff = cv::norm(w.dstMask, w.warpedMask, NORM_L2);

    ASSERT_EQ(0, rgbDiff);
    ASSERT_EQ(0, maskDiff);

    double depthDiff = cv::norm(w.dstDepth, w.warpedDepth, NORM_L2, w.dstMask);

    ASSERT_LE(depthDiff, DBL_EPSILON);
}

TEST(RGBD_Odometry_WarpFrame, noRgb)
{
    WarpFrameTest w;
    w.run(/* needRgb */ false, /* scaleDown*/ true, /* checkMask */ true, /* identityTransform */ false, CV_32F, CV_8UC3);

    double maskDiff = cv::norm(w.dstMask, w.warpedMask, NORM_L2);
    ASSERT_EQ(0, maskDiff);

    double l2diff = cv::norm(w.dstDepth, w.warpedDepth, NORM_L2, w.warpedMask);
    double lidiff = cv::norm(w.dstDepth, w.warpedDepth, NORM_INF, w.warpedMask);

    ASSERT_LE(l2diff, 0.038209);
    ASSERT_LE(lidiff, 0.00020004);
}

TEST(RGBD_Odometry_WarpFrame, nansAreMasked)
{
    WarpFrameTest w;
    w.run(/* needRgb */ true, /* scaleDown*/ true, /* checkMask */ false, /* identityTransform */ false, CV_32F, CV_8UC3);

    double rgbDiff = cv::norm(w.dstRgb, w.warpedRgb, NORM_L2);

    ASSERT_EQ(0, rgbDiff);

    Mat goodVals = (w.warpedDepth == w.warpedDepth);

    double l2diff = cv::norm(w.dstDepth, w.warpedDepth, NORM_L2, goodVals);
    double lidiff = cv::norm(w.dstDepth, w.warpedDepth, NORM_INF, goodVals);

    ASSERT_LE(l2diff, 0.038209);
    ASSERT_LE(lidiff, 0.00020004);
}

TEST(RGBD_Odometry_WarpFrame, bigScale)
{
    WarpFrameTest w;
    w.run(/* needRgb */ true, /* scaleDown*/ false, /* checkMask */ true, /* identityTransform */ false, CV_32F, CV_8UC3);

    double rgbDiff = cv::norm(w.dstRgb, w.warpedRgb, NORM_L2);
    double maskDiff = cv::norm(w.dstMask, w.warpedMask, NORM_L2);

    ASSERT_EQ(0, maskDiff);
    ASSERT_EQ(0, rgbDiff);

    double l2diff = cv::norm(w.dstDepth, w.warpedDepth, NORM_L2, w.warpedMask);
    double lidiff = cv::norm(w.dstDepth, w.warpedDepth, NORM_INF, w.warpedMask);

    ASSERT_LE(l2diff, 191.026565);
    ASSERT_LE(lidiff, 0.99951172);
}

TEST(RGBD_DepthTo3D, mask)
{
    std::string dataPath = cvtest::TS::ptr()->get_data_path();
    std::string srcDepthFilename = dataPath + "/cv/rgbd/depth.png";

    Mat srcDepth = imread(srcDepthFilename, IMREAD_UNCHANGED);
    ASSERT_FALSE(srcDepth.empty())  << "Depth " << srcDepthFilename.c_str() << "can not be read" << std::endl;
    ASSERT_TRUE(srcDepth.type() == CV_16UC1);

    Mat srcMask = srcDepth > 0;

    // test data used to generate warped depth and rgb
    // the script used to generate is in opencv_extra repo
    // at testdata/cv/rgbd/warped_depth_generator/warp_test.py
    double fx = 525.0, fy = 525.0,
           cx = 319.5, cy = 239.5;
    Matx33d intr(fx,  0, cx,
                  0, fy, cy,
                  0,  0,  1);

    Mat srcCloud;
    depthTo3d(srcDepth, intr, srcCloud, srcMask);
    size_t npts = countNonZero(srcMask);

    ASSERT_EQ(npts, srcCloud.total());
}

}} // namespace
