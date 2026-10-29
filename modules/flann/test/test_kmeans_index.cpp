// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// header, distance type, then branching_, iterations_, memoryCounter_, cb_index_, trees_
static const size_t kmeans_branching_ofs = sizeof(cvflann::IndexHeader) + sizeof(int);
static const size_t kmeans_trees_ofs = kmeans_branching_ofs + 4 * sizeof(int);

static std::vector<char> saveKMeansIndex(const Mat& data, const std::string& filename)
{
    {
        cv::flann::Index index(data, cv::flann::KMeansIndexParams(), cvflann::FLANN_DIST_L2);
        index.save(filename);
    }
    std::ifstream f(filename.c_str(), std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void writeFile(const std::string& filename, const std::vector<char>& buf)
{
    std::ofstream f(filename.c_str(), std::ios::binary);
    f.write(&buf[0], buf.size());
}

static int getInt(const std::vector<char>& buf, size_t ofs)
{
    int value = 0;
    memcpy(&value, &buf[ofs], sizeof(int));
    return value;
}

static void setInt(std::vector<char>& buf, size_t ofs, int value)
{
    memcpy(&buf[ofs], &value, sizeof(int));
}

TEST(Flann_KMeansIndex, load_more_trees_than_default)
{
    Mat data(100, 8, CV_32FC1);
    randu(data, 0.f, 1.f);
    std::string filename = cv::tempfile(".fln");
    std::vector<char> buf = saveKMeansIndex(data, filename);
    ASSERT_GT(buf.size(), kmeans_trees_ofs + sizeof(int));
    ASSERT_EQ(1, getInt(buf, kmeans_trees_ofs));

    // store the same tree twice: valid content, but more trees than the default the index was created with
    setInt(buf, kmeans_trees_ofs, 2);
    std::vector<char> tree(buf.begin() + kmeans_trees_ofs + sizeof(int), buf.end());
    buf.insert(buf.end(), tree.begin(), tree.end());
    writeFile(filename, buf);

    cv::flann::Index index;
    ASSERT_TRUE(index.load(data, filename));
    remove(filename.c_str());

    Mat indices, dists;
    index.knnSearch(data.row(7), indices, dists, 1, cv::flann::SearchParams(-1));
    EXPECT_EQ(7, indices.at<int>(0));
    // enough checks to go through every tree, not just root_[0]
    index.knnSearch(data.row(7), indices, dists, 1, cv::flann::SearchParams(2 * data.rows));
    EXPECT_EQ(7, indices.at<int>(0));
}

TEST(Flann_KMeansIndex, load_rejects_bad_branching_and_trees)
{
    Mat data(100, 8, CV_32FC1);
    randu(data, 0.f, 1.f);
    std::string filename = cv::tempfile(".fln");
    const std::vector<char> saved = saveKMeansIndex(data, filename);
    ASSERT_GT(saved.size(), kmeans_trees_ofs + sizeof(int));

    std::vector<char> buf = saved;
    setInt(buf, kmeans_trees_ofs, 0);
    writeFile(filename, buf);
    {
        cv::flann::Index index;
        EXPECT_THROW(index.load(data, filename), cv::Exception);
    }

    buf = saved;
    setInt(buf, kmeans_branching_ofs, 1);
    writeFile(filename, buf);
    {
        cv::flann::Index index;
        EXPECT_THROW(index.load(data, filename), cv::Exception);
    }
    remove(filename.c_str());
}

TEST(Flann_KMeansIndex, destroy_after_truncated_load)
{
    Mat data(100, 8, CV_32FC1);
    randu(data, 0.f, 1.f);
    cvflann::Matrix<float> dataset((float*)data.data, data.rows, data.cols);
    std::string filename = cv::tempfile(".fln");
    const std::vector<char> saved = saveKMeansIndex(data, filename);
    ASSERT_GT(saved.size(), kmeans_trees_ofs + sizeof(int));

    // every cut after trees_ stops inside the indices, a node, a pivot or a leaf offset,
    // and the destructor then frees whatever part of the tree was loaded
    for (size_t len = kmeans_trees_ofs + sizeof(int); len < saved.size(); ++len)
    {
        SCOPED_TRACE(cv::format("File length: %d", (int)len));
        writeFile(filename, std::vector<char>(saved.begin(), saved.begin() + len));
        FILE* fin = fopen(filename.c_str(), "rb");
        ASSERT_TRUE(fin != NULL);
        cvflann::load_header(fin);
        int distType = 0;
        cvflann::load_value(fin, distType);
        {
            cvflann::KMeansIndex<cvflann::L2<float> > index(dataset, cvflann::KMeansIndexParams());
            EXPECT_THROW(index.loadIndex(fin), cv::Exception);
        }
        fclose(fin);
    }
    remove(filename.c_str());
}

}} // namespace
