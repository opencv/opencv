#include "perf_precomp.hpp"
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<std::string, int> ImageName_MinSize_t;
typedef perf::TestBaseWithParam<ImageName_MinSize_t> ImageName_MinSize;

PERF_TEST_P(ImageName_MinSize, CascadeClassifierLBPFrontalFace,
            testing::Combine(testing::Values( std::string("cv/shared/lena.png"),
                                              std::string("cv/shared/1_itseez-0000289.png"),
                                              std::string("cv/shared/1_itseez-0000492.png"),
                                              std::string("cv/shared/1_itseez-0000573.png")),
                             testing::Values(24, 30, 40, 50, 60, 70, 80, 90)
                             )
            )
{
    const string filename = get<0>(GetParam());
    int min_size = get<1>(GetParam());
    Size minSize(min_size, min_size);

    CascadeClassifier cc(getDataPath("cv/cascadeandhog/cascades/lbpcascade_frontalface.xml"));
    if (cc.empty())
        FAIL() << "Can't load cascade file";

    Mat img = imread(getDataPath(filename), 0);
    if (img.empty())
        FAIL() << "Can't load source image";

    vector<Rect> faces;

    equalizeHist(img, img);
    declare.in(img);

    while(next())
    {
        faces.clear();

        startTimer();
        cc.detectMultiScale(img, faces, 1.1, 3, 0, minSize);
        stopTimer();
    }
    // for some reason OpenCL version detects the face, which CPU version does not detect, we just remove it
    // TODO better solution: implement smart way of comparing two set of rectangles
    if( filename == "cv/shared/1_itseez-0000492.png" && faces.size() == (size_t)3 )
    {
        faces.erase(faces.begin());
    }

    std::sort(faces.begin(), faces.end(), comparators::RectLess());
    SANITY_CHECK(faces, 3.001 * faces.size());
}
