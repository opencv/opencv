#include "perf_precomp.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace perf;

typedef std::tr1::tuple<std::string, int> ImageName_MinSize_t;
typedef perf::TestBaseWithParam<ImageName_MinSize_t> ImageName_MinSize;

PERF_TEST_P( ImageName_MinSize, CascadeClassifierLBPFrontalFace, 
        testing::Combine(testing::Values( std::string("cv/shared/lena.jpg"), 
                                          std::string("cv/shared/1_itseez-0000247.jpg"), 
                                          std::string("cv/shared/1_itseez-0000289.jpg"), 
                                          std::string("cv/shared/1_itseez-0000492.jpg"), 
                                          std::string("cv/shared/1_itseez-0000573.jpg"), 
                                          std::string("cv/shared/1_itseez-0000803.jpg"), 
                                          std::string("cv/shared/1_itseez-0000892.jpg"), 
                                          std::string("cv/shared/1_itseez-0000984.jpg"), 
                                          std::string("cv/shared/1_itseez-0001238.jpg"), 
                                          std::string("cv/shared/1_itseez-0001438.jpg"), 
                                          std::string("cv/shared/1_itseez-0002524.jpg")), 
            testing::Values(24, 30, 40, 50, 60, 70, 80, 90) ) )
{
    const string filename = std::tr1::get<0>(GetParam());
    int min_size = std::tr1::get<1>(GetParam());
    Size minSize(min_size, min_size);

    CascadeClassifier cc(getDataPath("cv/cascadeandhog/cascades/lbpcascade_frontalface.xml"));
    if (cc.empty())
        FAIL() << "Can't load cascade file";

    Mat img=imread(getDataPath(filename), 0);
    if (img.empty())
        FAIL() << "Can't load source image";

    vector<Rect> res;


    declare.in(img);//.out(res)

    while(next())
    {
        res.clear();

        startTimer();
        equalizeHist(img, img);
        cc.detectMultiScale(img, res, 1.1, 3, 0, minSize);
        stopTimer();
    }
}
