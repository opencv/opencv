#include "perf_precomp.hpp"
#include <opencv2/imgproc/imgproc.hpp>

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

    std::sort(faces.begin(), faces.end(), comparators::RectLess());
    SANITY_CHECK(faces, 3.001 * faces.size());
}

typedef std::tr1::tuple<std::string, std::string> fixture;
typedef perf::TestBaseWithParam<fixture> detect;


namespace {
typedef cv::SCascade::Detection detection_t;

void extractRacts(std::vector<detection_t> objectBoxes, vector<Rect>& rects)
{
    rects.clear();
    for (int i = 0; i < (int)objectBoxes.size(); ++i)
        rects.push_back(objectBoxes[i].bb);
}

}

PERF_TEST_P(detect, SCascade,
    testing::Combine(testing::Values(std::string("cv/cascadeandhog/cascades/inria_caltech-17.01.2013.xml")),
    testing::Values(std::string("cv/cascadeandhog/images/image_00000000_0.png"))))
{
    typedef cv::SCascade::Detection Detection;
    cv::Mat colored = imread(getDataPath(get<1>(GetParam())));
    ASSERT_FALSE(colored.empty());

    cv::SCascade cascade;
    cv::FileStorage fs(getDataPath(get<0>(GetParam())), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    std::vector<detection_t> objectBoxes;
    cascade.detect(colored, cv::noArray(), objectBoxes);

    TEST_CYCLE()
    {
        cascade.detect(colored, cv::noArray(), objectBoxes);
    }

    vector<Rect> rects;
    extractRacts(objectBoxes, rects);
    std::sort(rects.begin(), rects.end(), comparators::RectLess());
    SANITY_CHECK(rects);
}