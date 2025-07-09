#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;

CV_ENUM(MotionType, MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY)
CV_ENUM(ReadFlag, IMREAD_GRAYSCALE, IMREAD_COLOR)

typedef std::tuple<MotionType, ReadFlag> TestParams;
typedef perf::TestBaseWithParam<TestParams> ECCPerfTest;

PERF_TEST_P(ECCPerfTest, findTransformECC,
            testing::Combine(testing::Values(MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY),
                             testing::Values(IMREAD_GRAYSCALE, IMREAD_COLOR))) {
    int transform_type = get<0>(GetParam());
    int readFlag = get<1>(GetParam());

    Mat img = imread(getDataPath("cv/shared/fruits_ecc.png"), readFlag);
    Mat templateImage;

    Mat warpMat;
    Mat warpGround;

    double angle;
    switch (transform_type) {
        case MOTION_TRANSLATION:
            warpGround = (Mat_<float>(2, 3) << 1.f, 0.f, 7.234f, 0.f, 1.f, 11.839f);

            warpAffine(img, templateImage, warpGround, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case MOTION_EUCLIDEAN:
            angle = CV_PI / 30;

            warpGround = (Mat_<float>(2, 3) << (float)cos(angle), (float)-sin(angle), 12.123f, (float)sin(angle),
                          (float)cos(angle), 14.789f);
            warpAffine(img, templateImage, warpGround, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case MOTION_AFFINE:
            warpGround = (Mat_<float>(2, 3) << 0.98f, 0.03f, 15.523f, -0.02f, 0.95f, 10.456f);
            warpAffine(img, templateImage, warpGround, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case MOTION_HOMOGRAPHY:
            warpGround = (Mat_<float>(3, 3) << 0.98f, 0.03f, 15.523f, -0.02f, 0.95f, 10.456f, 0.0002f, 0.0003f, 1.f);
            warpPerspective(img, templateImage, warpGround, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
    }

    TEST_CYCLE() {
        if (transform_type < 3)
            warpMat = Mat::eye(2, 3, CV_32F);
        else
            warpMat = Mat::eye(3, 3, CV_32F);

        findTransformECC(templateImage, img, warpMat, transform_type,
                         TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 5, -1));
    }

    // TODO: Update baseline for new test
    SANITY_CHECK_NOTHING();
}

}  // namespace opencv_test
