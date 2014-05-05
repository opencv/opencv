#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(MotionType, MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY)

typedef std::tr1::tuple<MotionType> MotionType_t;
typedef perf::TestBaseWithParam<MotionType_t> TransformationType;


PERF_TEST_P(TransformationType, findTransformECC, /*testing::ValuesIn(MotionType::all())*/
            testing::Values((int) MOTION_TRANSLATION, (int) MOTION_EUCLIDEAN,
            (int) MOTION_AFFINE, (int) MOTION_HOMOGRAPHY)
            )
{
    Mat img = imread(getDataPath("cv/shared/fruits_ecc.png"),0);
    Mat templateImage;

    int transform_type = get<0>(GetParam());

    Mat warpMat;
    Mat warpGround;

    double angle;
    switch (transform_type) {
        case MOTION_TRANSLATION:
            warpGround = (Mat_<float>(2,3) << 1.f, 0.f, 7.234f,
                0.f, 1.f, 11.839f);

            warpAffine(img, templateImage, warpGround,
                Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case MOTION_EUCLIDEAN:
            angle = CV_PI/30;

            warpGround = (Mat_<float>(2,3) << (float)cos(angle), (float)-sin(angle), 12.123f,
                (float)sin(angle), (float)cos(angle), 14.789f);
            warpAffine(img, templateImage, warpGround,
                Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case MOTION_AFFINE:
            warpGround = (Mat_<float>(2,3) << 0.98f, 0.03f, 15.523f,
                -0.02f, 0.95f, 10.456f);
            warpAffine(img, templateImage, warpGround,
                Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case MOTION_HOMOGRAPHY:
            warpGround = (Mat_<float>(3,3) << 0.98f, 0.03f, 15.523f,
                -0.02f, 0.95f, 10.456f,
                0.0002f, 0.0003f, 1.f);
            warpPerspective(img, templateImage, warpGround,
                Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
    }

    TEST_CYCLE()
    {
        if (transform_type<3)
            warpMat = Mat::eye(2,3, CV_32F);
        else
            warpMat = Mat::eye(3,3, CV_32F);

        findTransformECC(templateImage, img, warpMat, transform_type,
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 5, -1));
    }
    SANITY_CHECK(warpMat, 1e-3);
}
