// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{
struct ExposureSeq
{
    std::vector<Mat> images;
    std::vector<float> times;
};

ExposureSeq loadExposureSeq(const std::string& list_filename)
{
    std::ifstream list_file(list_filename);
    EXPECT_TRUE(list_file.is_open());
    string name;
    float val;
    const String path(list_filename.substr(0, list_filename.find_last_of("\\/") + 1));
    ExposureSeq seq;
    while (list_file >> name >> val)
    {
        Mat img = imread(path + name);
        EXPECT_FALSE(img.empty()) << "Could not load input image " << path + name;
        seq.images.push_back(img);
        seq.times.push_back(1 / val);
    }
    list_file.close();
    return seq;
}

PERF_TEST(HDR, Mertens)
{
    const ExposureSeq seq = loadExposureSeq(getDataPath("cv/hdr/exposures/list.txt"));
    Ptr<MergeMertens> merge = createMergeMertens();
    Mat result(seq.images.front().size(), seq.images.front().type());
    TEST_CYCLE() merge->process(seq.images, result);
    SANITY_CHECK_NOTHING();
}

PERF_TEST(HDR, Debevec)
{
    const ExposureSeq seq = loadExposureSeq(getDataPath("cv/hdr/exposures/list.txt"));
    Ptr<MergeDebevec> merge = createMergeDebevec();
    Mat result(seq.images.front().size(), seq.images.front().type());
    TEST_CYCLE() merge->process(seq.images, result, seq.times);
    SANITY_CHECK_NOTHING();
}

PERF_TEST(HDR, Robertson)
{
    const ExposureSeq seq = loadExposureSeq(getDataPath("cv/hdr/exposures/list.txt"));
    Ptr<MergeRobertson> merge = createMergeRobertson();
    Mat result(seq.images.front().size(), seq.images.front().type());
    TEST_CYCLE() merge->process(seq.images, result, seq.times);
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
