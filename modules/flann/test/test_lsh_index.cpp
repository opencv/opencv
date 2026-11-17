// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <fstream>

namespace opencv_test { namespace {

struct TempFile
{
    std::string path;
    explicit TempFile(const char* suffix) : path(cv::tempfile(suffix)) {}
    ~TempFile() { remove(path.c_str()); }
};

static std::vector<char> readFileBytes(const std::string& path)
{
    std::ifstream f(path.c_str(), std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void writeFileBytes(const std::string& path, const std::vector<char>& buf)
{
    std::ofstream f(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!buf.empty())
        f.write(&buf[0], (std::streamsize)buf.size());
}

static int getIntAt(const std::vector<char>& buf, size_t ofs)
{
    int v = 0;
    memcpy(&v, &buf[ofs], sizeof(int));
    return v;
}

static void setIntAt(std::vector<char>& buf, size_t ofs, int v)
{
    memcpy(&buf[ofs], &v, sizeof(int));
}

// Every dataset row sits in its own bucket, so knnSearch finds it at distance 0 regardless of the random bit masks.
static void expectRowsFound(flann::Index& index, const Mat& features)
{
    Mat indices, dists;
    index.knnSearch(features, indices, dists, 1, flann::SearchParams());
    ASSERT_EQ(features.rows, indices.rows);
    for (int i = 0; i < features.rows; i++)
    {
        ASSERT_EQ(i, indices.at<int>(i, 0));
        ASSERT_EQ(0, dists.at<int>(i, 0));
    }
}

TEST(Flann_LSH, load_with_small_key_size)
{
    Mat features(500, 32, CV_8UC1);
    theRNG().fill(features, RNG::UNIFORM, 0, 256);

    // key_size = 12 (kBitsetHash), key_size = 8 (kArray), and key_size = 1 with multi_probe_level = 2 (> key_size)
    const struct { int key_size; int multi_probe_level; } cases[] = {
        { 12, 1 },
        { 8, 1 },
        { 1, 2 },
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k)
    {
        SCOPED_TRACE(cv::format("key_size=%d multi_probe_level=%d", cases[k].key_size, cases[k].multi_probe_level));
        TempFile file(".flann");
        {
            flann::Index built(features, flann::LshIndexParams(6, cases[k].key_size, cases[k].multi_probe_level), cvflann::FLANN_DIST_HAMMING);
            expectRowsFound(built, features);
            built.save(file.path);
        }

        flann::Index loaded;
        ASSERT_TRUE(loaded.load(features, file.path));
        expectRowsFound(loaded, features);
    }

    // cvflann::LshIndexParams(6, 1) defaults multi_probe_level to 2 (> key_size)
    const cvflann::LshIndexParams generic_cases[] = {
        cvflann::LshIndexParams(6, 12, 1),
        cvflann::LshIndexParams(6, 1),
    };
    for (size_t k = 0; k < sizeof(generic_cases) / sizeof(generic_cases[0]); ++k)
    {
        TempFile genericFile(".flann");
        {
            flann::GenericIndex<cvflann::Hamming<uchar> > built(features, generic_cases[k]);
            Mat buildIndices(features.rows, 1, CV_32S), buildDists(features.rows, 1, CV_32S);
            built.knnSearch(features, buildIndices, buildDists, 1, cvflann::SearchParams());
            for (int i = 0; i < features.rows; i++)
            {
                ASSERT_EQ(i, buildIndices.at<int>(i, 0));
                ASSERT_EQ(0, buildDists.at<int>(i, 0));
            }
            built.save(genericFile.path);
        }
        flann::GenericIndex<cvflann::Hamming<uchar> > loadedGeneric(features, cvflann::SavedIndexParams(genericFile.path));
        Mat indices(features.rows, 1, CV_32S), dists(features.rows, 1, CV_32S);
        loadedGeneric.knnSearch(features, indices, dists, 1, cvflann::SearchParams());
        for (int i = 0; i < features.rows; i++)
        {
            ASSERT_EQ(i, indices.at<int>(i, 0));
            ASSERT_EQ(0, dists.at<int>(i, 0));
        }
    }
}

// The file holds the dataset, so a loaded index finds the saved rows whatever data of the same
// size load() gets (#4758 passes an uninitialized matrix); loading again replaces that copy.
TEST(Flann_LSH, load_uses_the_saved_dataset)
{
    Mat features(500, 32, CV_8UC1), placeholder(500, 32, CV_8UC1, Scalar::all(0));
    theRNG().fill(features, RNG::UNIFORM, 0, 256);
    TempFile file(".flann");
    {
        flann::Index built(features, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
        built.save(file.path);
    }
    flann::Index loaded;
    ASSERT_TRUE(loaded.load(placeholder, file.path));
    expectRowsFound(loaded, features);
    ASSERT_TRUE(loaded.load(features, file.path));
    expectRowsFound(loaded, features);
    loaded.release();
}

TEST(Flann_LSH, load_rejects_invalid_stream)
{
    Mat features(32, 8, CV_8UC1);
    theRNG().fill(features, RNG::UNIFORM, 0, 256);
    TempFile file(".flann");
    {
        flann::Index built(features, flann::LshIndexParams(4, 12, 1), cvflann::FLANN_DIST_HAMMING);
        built.save(file.path);
    }

    const std::vector<char> saved = readFileBytes(file.path);
    const size_t params_ofs = sizeof(cvflann::IndexHeader) + sizeof(int);
    ASSERT_GT(saved.size(), params_ofs + 3 * sizeof(int));
    ASSERT_EQ(4, getIntAt(saved, params_ofs));
    ASSERT_EQ(12, getIntAt(saved, params_ofs + sizeof(int)));
    ASSERT_EQ(1, getIntAt(saved, params_ofs + 2 * sizeof(int)));

    // table_number <= 0, key_size <= 0, key_size out of BucketKey range, multi_probe_level < 0
    const struct { size_t ofs; int val; } bad_params[] = {
        { params_ofs, 0 },
        { params_ofs + sizeof(int), 0 },
        { params_ofs + sizeof(int), 33 },
        { params_ofs + 2 * sizeof(int), -1 },
    };
    for (size_t i = 0; i < sizeof(bad_params) / sizeof(bad_params[0]); ++i)
    {
        SCOPED_TRACE(cv::format("case=%d", (int)i));
        std::vector<char> buf = saved;
        setIntAt(buf, bad_params[i].ofs, bad_params[i].val);
        writeFileBytes(file.path, buf);
        flann::Index loaded;
        EXPECT_THROW(loaded.load(features, file.path), cv::Exception);
    }

    // key_size larger than the saved features' bit width (2 bytes = 16 bits, with key_size = 20); the file
    // is written by hand because building an LSH index on features narrower than size_t is not supported
    Mat narrow(32, 2, CV_8UC1);
    theRNG().fill(narrow, RNG::UNIFORM, 0, 256);
    {
        const size_t matrix_ofs = params_ofs + 3 * sizeof(int);
        std::vector<char> narrowBuf(saved.begin(), saved.begin() + matrix_ofs + sizeof(cvflann::Matrix<uchar>));
        cvflann::IndexHeader hdr;
        memcpy(&hdr, &narrowBuf[0], sizeof(hdr));
        hdr.cols = (size_t)narrow.cols;
        memcpy(&narrowBuf[0], &hdr, sizeof(hdr));
        setIntAt(narrowBuf, params_ofs + sizeof(int), 20);
        cvflann::Matrix<uchar> matHdr;
        memcpy(&matHdr, &narrowBuf[matrix_ofs], sizeof(matHdr));
        matHdr.cols = matHdr.stride = (size_t)narrow.cols;
        memcpy(&narrowBuf[matrix_ofs], &matHdr, sizeof(matHdr));
        narrowBuf.insert(narrowBuf.end(), (const char*)narrow.data, (const char*)narrow.data + narrow.total());
        writeFileBytes(file.path, narrowBuf);
        flann::Index loaded;
        EXPECT_THROW(loaded.load(narrow, file.path), cv::Exception);
    }

    // saved Matrix header claiming more rows than the file holds
    {
        std::vector<char> mismatchBuf = saved;
        const size_t matrix_ofs = params_ofs + 3 * sizeof(int);
        cvflann::Matrix<uchar> matHdr;
        memcpy(&matHdr, &mismatchBuf[matrix_ofs], sizeof(matHdr));
        ASSERT_EQ((size_t)features.rows, matHdr.rows);
        matHdr.rows += 1;
        memcpy(&mismatchBuf[matrix_ofs], &matHdr, sizeof(matHdr));
        writeFileBytes(file.path, mismatchBuf);
        flann::Index loaded;
        EXPECT_THROW(loaded.load(features, file.path), cv::Exception);
    }

    // truncated serialized dataset (1 byte short) via both flann::Index and flann::GenericIndex
    writeFileBytes(file.path, std::vector<char>(saved.begin(), saved.end() - 1));
    {
        flann::Index loaded;
        EXPECT_THROW(loaded.load(features, file.path), cv::Exception);
    }

    TempFile genericFile(".flann");
    {
        flann::GenericIndex<cvflann::Hamming<uchar> > built(features, cvflann::LshIndexParams(4, 12, 1));
        built.save(genericFile.path);
    }
    std::vector<char> genericSaved = readFileBytes(genericFile.path);
    writeFileBytes(genericFile.path, std::vector<char>(genericSaved.begin(), genericSaved.end() - 1));
    EXPECT_THROW((flann::GenericIndex<cvflann::Hamming<uchar> >(features, cvflann::SavedIndexParams(genericFile.path))),
                 cv::Exception);
}

}} // namespace
