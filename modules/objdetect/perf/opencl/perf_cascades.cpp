#include "perf_precomp.hpp"
#include <opencv2/imgproc.hpp>

#include "opencv2/ts/ocl_perf.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<std::string, std::string, int> Cascade_Image_MinSize_t;
typedef perf::TestBaseWithParam<Cascade_Image_MinSize_t> Cascade_Image_MinSize;

#ifdef HAVE_OPENCL

OCL_PERF_TEST_P(Cascade_Image_MinSize, CascadeClassifier,
                 testing::Combine(
                    testing::Values( string("cv/cascadeandhog/cascades/haarcascade_frontalface_alt.xml"),
                                     string("cv/cascadeandhog/cascades/haarcascade_frontalface_alt_old.xml"),
                                     string("cv/cascadeandhog/cascades/lbpcascade_frontalface.xml") ),
                    testing::Values( string("cv/shared/lena.png"),
                                     string("cv/cascadeandhog/images/bttf301.png"),
                                     string("cv/cascadeandhog/images/class57.png") ),
                    testing::Values(30, 64, 90) ) )
{
    const string cascadePath = get<0>(GetParam());
    const string imagePath   = get<1>(GetParam());
    int min_size = get<2>(GetParam());
    Size minSize(min_size, min_size);

    CascadeClassifier cc( getDataPath(cascadePath) );
    if (cc.empty())
        FAIL() << "Can't load cascade file: " << getDataPath(cascadePath);

    Mat img = imread(getDataPath(imagePath), IMREAD_GRAYSCALE);
    if (img.empty())
        FAIL() << "Can't load source image: " << getDataPath(imagePath);

    vector<Rect> faces;

    equalizeHist(img, img);
    declare.in(img).time(60);

    UMat uimg = img.getUMat(ACCESS_READ);

    while(next())
    {
        faces.clear();
        cvtest::ocl::perf::safeFinish();

        startTimer();
        cc.detectMultiScale(uimg, faces, 1.1, 3, 0, minSize);
        stopTimer();
    }

    sort(faces.begin(), faces.end(), comparators::RectLess());
    SANITY_CHECK(faces, min_size/5);
}

#endif //HAVE_OPENCL
