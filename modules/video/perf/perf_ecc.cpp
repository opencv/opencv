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

    SANITY_CHECK(warpMat, 1e-3);
}
