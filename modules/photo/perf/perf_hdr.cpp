#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{
static vector<float> DEFAULT_VECTOR;
void loadExposureSeq(const std::string& list_filename, vector<Mat>& images,
                     vector<float>& times = DEFAULT_VECTOR)
{
    std::ifstream list_file(list_filename);
    ASSERT_TRUE(list_file.is_open());
    string name;
    float val;
    const String path(list_filename.substr(0, list_filename.find_last_of("\\/") + 1));
    while (list_file >> name >> val)
    {
        Mat img = imread(path + name);
        ASSERT_FALSE(img.empty()) << "Could not load input image " << path + name;
        images.push_back(img);
        times.push_back(1 / val);
    }
    list_file.close();
}

PERF_TEST(HDR, Mertens)
{
    vector<Mat> images;
    loadExposureSeq(getDataPath("cv/hdr/exposures/list.txt"), images);
    Ptr<MergeMertens> merge = createMergeMertens();
    Mat result(images.front().size(), images.front().type());
    TEST_CYCLE() merge->process(images, result);
    SANITY_CHECK_NOTHING();
}

PERF_TEST(HDR, Debevec)
{
    vector<Mat> images;
    vector<float> times_seconds;
    loadExposureSeq(getDataPath("cv/hdr/exposures/list.txt"), images, times_seconds);
    Ptr<MergeDebevec> merge = createMergeDebevec();
    Mat result(images.front().size(), images.front().type());
    TEST_CYCLE() merge->process(images, result, times_seconds);
    SANITY_CHECK_NOTHING();
}

PERF_TEST(HDR, Robertson)
{
    vector<Mat> images;
    vector<float> times_seconds;
    loadExposureSeq(getDataPath("cv/hdr/exposures/list.txt"), images, times_seconds);
    Ptr<MergeRobertson> merge = createMergeRobertson();
    Mat result(images.front().size(), images.front().type());
    TEST_CYCLE() merge->process(images, result, times_seconds);
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
