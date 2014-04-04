#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

enum { TYPE_5_8 =FastFeatureDetector::TYPE_5_8, TYPE_7_12 = FastFeatureDetector::TYPE_7_12, TYPE_9_16 = FastFeatureDetector::TYPE_9_16 };
CV_ENUM(FastType, TYPE_5_8, TYPE_7_12)

typedef std::tr1::tuple<string, FastType> File_Type_t;
typedef TestBaseWithParam<File_Type_t> FASTFixture;

#define FAST_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

OCL_PERF_TEST_P(FASTFixture, FastDetect, testing::Combine(
                            testing::Values(FAST_IMAGES),
                            FastType::all()
                          ))
{
    string filename = getDataPath(get<0>(GetParam()));
    int type = get<1>(GetParam());
    Mat mframe = imread(filename, IMREAD_GRAYSCALE);

    if (mframe.empty())
        FAIL() << "Unable to load source image " << filename;

    UMat frame;
    mframe.copyTo(frame);
    declare.in(frame);

    Ptr<FeatureDetector> fd = Algorithm::create<FeatureDetector>("Feature2D.FAST");
    ASSERT_FALSE( fd.empty() );
    fd->set("threshold", 20);
    fd->set("nonmaxSuppression", true);
    fd->set("type", type);
    vector<KeyPoint> points;

    OCL_TEST_CYCLE() fd->detect(frame, points);

    SANITY_CHECK_KEYPOINTS(points);
}

} // ocl
} // cvtest

#endif // HAVE_OPENCL
