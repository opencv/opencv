#include "perf_precomp.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using cv::Rect;
using std::tr1::get;

typedef std::tr1::tuple<std::string, std::string> fixture;
typedef perf::TestBaseWithParam<fixture> detect;


namespace {
typedef cv::SCascade::Detection detection_t;

void extractRacts(std::vector<detection_t> objectBoxes, std::vector<Rect>& rects)
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
    cv::Mat colored = cv::imread(getDataPath(get<1>(GetParam())));
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

    std::vector<Rect> rects;
    extractRacts(objectBoxes, rects);
    std::sort(rects.begin(), rects.end(), perf::comparators::RectLess());
    SANITY_CHECK(rects);
}