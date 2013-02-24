#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(MotionType, MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY)

typedef std::tr1::tuple<MotionType> MotionType_t;
typedef perf::TestBaseWithParam<MotionType_t> TransformationType;


<<<<<<< HEAD
PERF_TEST_P(TransformationType, findTransformECC, /*testing::ValuesIn(MotionType::all())*/
            testing::Values((int) MOTION_TRANSLATION, (int) MOTION_EUCLIDEAN,
            (int) MOTION_AFFINE, (int) MOTION_HOMOGRAPHY)
            )
{

    Mat inputImage = imread(getDataPath("cv/shared/fruits.png"),0);
    Mat img;
    resize(inputImage, img, Size(216,216));
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
                Size(200,200), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
            break;
        case MOTION_EUCLIDEAN:
            angle = CV_PI/30;

            warpGround = (Mat_<float>(2,3) << (float)cos(angle), (float)-sin(angle), 12.123f,
                (float)sin(angle), (float)cos(angle), 14.789f);
            warpAffine(img, templateImage, warpGround,
                Size(200,200), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
            break;
        case MOTION_AFFINE:
            warpGround = (Mat_<float>(2,3) << 0.98f, 0.03f, 15.523f,
                -0.02f, 0.95f, 10.456f);
            warpAffine(img, templateImage, warpGround,
                Size(200,200), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
            break;
        case MOTION_HOMOGRAPHY:
            warpGround = (Mat_<float>(3,3) << 0.98f, 0.03f, 15.523f,
                -0.02f, 0.95f, 10.456f,
                0.0002f, 0.0003f, 1.f);
            warpPerspective(img, templateImage, warpGround,
                Size(200,200), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
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
=======
PERF_TEST_P(TransformationType, findTransformECC, /*testing::ValuesIn(MotionType::all())*/ 
			testing::Values((int) MOTION_TRANSLATION, (int) MOTION_EUCLIDEAN,
			(int) MOTION_AFFINE, (int) MOTION_HOMOGRAPHY)
			)
{

	declare.time(400);

	string filename1 = getDataPath("cv/shared/cameramanTemplate.png");
	string filename2 = getDataPath("cv/shared/cameramanImage.png");
    Mat templateImage = imread(filename1, IMREAD_GRAYSCALE);
    Mat inputImage = imread(filename2, IMREAD_GRAYSCALE);

	if (templateImage.empty()) FAIL() << "Unable to load template image " << filename1;
    if (inputImage.empty()) FAIL() << "Unable to load input image image " << filename2;


    int transform_type = get<0>(GetParam());

	Mat warpMat;
	if (transform_type<3)
		warpMat = (Mat_<float>(2,3,CV_32F) << 1, 0, 0, 0, 1, 0);
	else
		warpMat = Mat::eye(3,3, CV_32F);

	if (warpMat.empty()) FAIL() << "No initial warp";

	//warpMap is InputOutputArray
	declare.iterations(10).in(templateImage, inputImage);

	
	TEST_CYCLE()
	{
		//a negative epsilon means that 50 iterations will be executed
		findTransformECC(templateImage, inputImage, warpMat, transform_type,
		TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, -1));
	}

>>>>>>> 2e5332aa34a0731ea0327d094b24890e3742ba8c
    SANITY_CHECK(warpMat, 1e-3);
}
