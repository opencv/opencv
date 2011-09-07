#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef std::tr1::tuple<const char*, int> ImageName_MinSize_t;
typedef perf::TestBaseWithParam<ImageName_MinSize_t> ImageName_MinSize;

PERF_TEST_P( ImageName_MinSize, CascadeClassifierLBPFrontalFace, testing::Values( ImageName_MinSize_t("cv/shared/lena.jpg", 10) ) )
{
    const char* filename = std::tr1::get<0>(GetParam());
    int min_size = std::tr1::get<1>(GetParam());
    Size minSize(min_size, min_size);

    CascadeClassifier cc(getDataPath("cv/cascadeandhog/cascades/lbpcascade_frontalface.xml"));

    Mat img=imread(getDataPath(filename));
    vector<Rect> res;

    declare.in(img).time(10000);
    TEST_CYCLE(100) 
    {
        res.clear();
        cc.detectMultiScale(img, res, 1.1, 3, 0, minSize);
    }
}
