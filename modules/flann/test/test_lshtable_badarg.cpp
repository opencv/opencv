/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#define OPENCV_TEST_FLANN_GUARD_PAGE 1
#endif

namespace opencv_test { namespace {

class CV_LshTableBadArgTest : public cvtest::BadArgTest
{
protected:
    void run(int);
    void run_func(void) { }

    struct Caller
    {
        int table_number, key_size, multi_probe_level;
        Mat features;

        void operator()() const
        {
            flann::LshIndexParams indexParams(table_number, key_size, multi_probe_level);
            flann::Index lsh(features, indexParams);
        }
    };
};

void CV_LshTableBadArgTest::run( int /* start_from */ )
{
    RNG &rng = ts->get_rng();

    Caller caller;
    Size featuresSize = cvtest::randomSize(rng, 10.0);
    caller.features = cvtest::randomMat(rng, featuresSize, CV_8UC1, 0, 255, false);
    caller.table_number = 12;
    caller.multi_probe_level = 2;

    int errors = 0;
    caller.key_size = 0;
    errors += run_test_case(Error::StsBadArg, "key_size is zero", caller);

    caller.key_size = static_cast<int>(sizeof(size_t) * CHAR_BIT);
    errors += run_test_case(Error::StsBadArg, "key_size is too big", caller);

    caller.key_size += cvtest::randInt(rng) % 100;
    errors += run_test_case(Error::StsBadArg, "key_size is too big", caller);

    if (errors != 0)
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    else
        ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Flann_LshTable, badarg) { CV_LshTableBadArgTest test; test.safe_run(); }

TEST(Flann_LshTable, bad_any_cast) {
    Mat features = Mat::ones(1, 64, CV_8U);
    EXPECT_NO_THROW(flann::GenericIndex<cvflann::Hamming2<unsigned char> >(
                                            features, cvflann::LshIndexParams()));
}

#ifdef OPENCV_TEST_FLANN_GUARD_PAGE
struct GuardedTailBuffer
{
    unsigned char* mapping;
    size_t page_size;

    GuardedTailBuffer() : mapping(static_cast<unsigned char*>(MAP_FAILED)), page_size(0)
    {
        long ps = sysconf(_SC_PAGESIZE);
        if (ps > 0)
        {
            page_size = static_cast<size_t>(ps);
            void* p = mmap(NULL, 2 * page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p != MAP_FAILED)
            {
                mapping = static_cast<unsigned char*>(p);
                if (mprotect(mapping + page_size, page_size, PROT_NONE) != 0)
                {
                    munmap(mapping, 2 * page_size);
                    mapping = static_cast<unsigned char*>(MAP_FAILED);
                }
            }
        }
    }

    ~GuardedTailBuffer()
    {
        if (mapping != static_cast<unsigned char*>(MAP_FAILED))
            munmap(mapping, 2 * page_size);
    }

    unsigned char* tail(size_t n) const
    {
        if (mapping == static_cast<unsigned char*>(MAP_FAILED) || n > page_size)
            return NULL;
        return mapping + page_size - n;
    }

private:
    GuardedTailBuffer(const GuardedTailBuffer&);
    GuardedTailBuffer& operator=(const GuardedTailBuffer&);
};
#endif

TEST(Flann_LshTable, rejects_key_size_exceeding_feature_bits)
{
    const int rows = 32;
    for (int bytes = 1; bytes <= 3; ++bytes)
    {
        Mat features(rows, bytes, CV_8UC1);
        theRNG().fill(features, RNG::UNIFORM, 0, 256);

        const int badKeys[] = { bytes * CHAR_BIT + 1, 20, 32 };
        for (int k = 0; k < 3; ++k)
        {
            const int keySize = badKeys[k];
            if (keySize <= bytes * CHAR_BIT)
                continue;

            bool threwPlain = false;
            try
            {
                flann::Index idx(features, flann::LshIndexParams(4, keySize, 1), cvflann::FLANN_DIST_HAMMING);
                CV_UNUSED(idx);
            }
            catch (const cv::Exception& e)
            {
                threwPlain = true;
                EXPECT_EQ(Error::StsBadArg, e.code) << "bytes=" << bytes << " keySize=" << keySize;
            }
            EXPECT_TRUE(threwPlain) << "Index accepted keySize=" << keySize << " for bytes=" << bytes;

            bool threwGeneric = false;
            try
            {
                flann::GenericIndex<cvflann::Hamming<unsigned char> > gidx(
                    features, cvflann::LshIndexParams(4, keySize, 1));
                CV_UNUSED(gidx);
            }
            catch (const cv::Exception& e)
            {
                threwGeneric = true;
                EXPECT_EQ(Error::StsBadArg, e.code) << "bytes=" << bytes << " keySize=" << keySize;
            }
            EXPECT_TRUE(threwGeneric) << "GenericIndex accepted keySize=" << keySize << " for bytes=" << bytes;
        }
    }

    // Boundary check: key_size == 8 * bytes must remain accepted for 1..3 bytes (8, 16, 24 bits; valid on 32-bit too)
    for (int bytes = 1; bytes <= 3; ++bytes)
    {
        Mat features(rows, bytes, CV_8UC1);
        theRNG().fill(features, RNG::UNIFORM, 0, 256);
        const int maxValidKey = bytes * CHAR_BIT;
        EXPECT_NO_THROW({
            flann::Index idx(features, flann::LshIndexParams(2, maxValidKey, 1), cvflann::FLANN_DIST_HAMMING);
            CV_UNUSED(idx);
        }) << "bytes=" << bytes << " keySize=" << maxValidKey;
    }
}

TEST(Flann_LshTable, narrow_features_1_to_7_bytes)
{
    const int rows = 40;
#ifdef OPENCV_TEST_FLANN_GUARD_PAGE
    GuardedTailBuffer guarded;
#endif

    for (int bytes = 1; bytes <= 7; ++bytes)
    {
        Mat owned(rows, bytes, CV_8UC1);
        theRNG().fill(owned, RNG::UNIFORM, 0, 256);

        // Exact-size contiguous buffer (exposes both tail over-read and unaligned rows under ASan/UBSan)
        std::vector<unsigned char> exactDataset(static_cast<size_t>(rows * bytes));
        for (int r = 0; r < rows; ++r)
            memcpy(&exactDataset[static_cast<size_t>(r * bytes)], owned.ptr<unsigned char>(r), static_cast<size_t>(bytes));
        Mat features(rows, bytes, CV_8UC1, exactDataset.data());

        const int keySizes[] = { 1, std::min(12, bytes * CHAR_BIT), std::min(20, bytes * CHAR_BIT) };
        for (int k = 0; k < 3; ++k)
        {
            const int keySize = keySizes[k];
            flann::GenericIndex<cvflann::HammingLUT> gindex(
                features, cvflann::LshIndexParams(6, keySize, 1));

            const int targetRow = 7;
            std::vector<unsigned char> exactQuery(owned.ptr<unsigned char>(targetRow),
                                                  owned.ptr<unsigned char>(targetRow) + bytes);
            unsigned char* qPtr = exactQuery.data();
#ifdef OPENCV_TEST_FLANN_GUARD_PAGE
            if (unsigned char* gPtr = guarded.tail(static_cast<size_t>(bytes)))
            {
                memcpy(gPtr, exactQuery.data(), static_cast<size_t>(bytes));
                qPtr = gPtr;
            }
#endif
            Mat query(1, bytes, CV_8UC1, qPtr);
            Mat gIndices(1, 1, CV_32SC1), gDists(1, 1, CV_32SC1);
            gindex.knnSearch(query, gIndices, gDists, 1, cvflann::SearchParams());
            ASSERT_GE(gIndices.at<int>(0, 0), 0);
            ASSERT_LT(gIndices.at<int>(0, 0), rows);
            EXPECT_EQ(0, gDists.at<int>(0, 0)) << "bytes=" << bytes << " keySize=" << keySize;
            EXPECT_EQ(0, memcmp(owned.ptr<unsigned char>(gIndices.at<int>(0, 0)), exactQuery.data(), static_cast<size_t>(bytes)));

#if !CV_NEON
            // On NEON, cvflann::Hamming<uchar> (used by flann::Index) loads 16-byte vectors directly.
            flann::Index index(features, flann::LshIndexParams(6, keySize, 1), cvflann::FLANN_DIST_HAMMING);
            Mat indices, dists;
            index.knnSearch(query, indices, dists, 1, flann::SearchParams());
            ASSERT_GE(indices.at<int>(0, 0), 0);
            ASSERT_LT(indices.at<int>(0, 0), rows);
            EXPECT_EQ(0, dists.at<int>(0, 0)) << "bytes=" << bytes << " keySize=" << keySize;
            EXPECT_EQ(0, memcmp(owned.ptr<unsigned char>(indices.at<int>(0, 0)), exactQuery.data(), static_cast<size_t>(bytes)));
#endif
        }
    }

    // Verify both full-block and tail-block loads populate distinct LSH keys (1 byte and unaligned 9 bytes).
    {
        std::vector<unsigned char> narrowPair(2, 0);
        narrowPair[1] = 0xFF;
        Mat narrowMat(2, 1, CV_8UC1, narrowPair.data());
        flann::GenericIndex<cvflann::HammingLUT> narrowIdx(narrowMat, cvflann::LshIndexParams(1, 8, 0));
        Mat q0(1, 1, CV_8UC1, narrowPair.data());
        Mat ind(1, 2, CV_32SC1), dst(1, 2, CV_32SC1);
        narrowIdx.knnSearch(q0, ind, dst, 2, cvflann::SearchParams());
        EXPECT_EQ(0, ind.at<int>(0, 0));
        EXPECT_EQ(-1, ind.at<int>(0, 1));
    }
    {
        std::vector<unsigned char> multiBlock(2 * 9, 0);
        memset(&multiBlock[9], 0xFF, 8); // 64 set bits in block 0, 0 in tail byte 8 -> >=12 of 20 mask bits hit block 0
        Mat wideMat(2, 9, CV_8UC1, multiBlock.data());
        flann::GenericIndex<cvflann::HammingLUT> wideIdx(wideMat, cvflann::LshIndexParams(1, 20, 0));
        Mat q0(1, 9, CV_8UC1, multiBlock.data());
        Mat ind(1, 2, CV_32SC1), dst(1, 2, CV_32SC1);
        wideIdx.knnSearch(q0, ind, dst, 2, cvflann::SearchParams());
        EXPECT_EQ(0, ind.at<int>(0, 0));
        EXPECT_EQ(-1, ind.at<int>(0, 1));
    }
}

TEST(Flann_GenericIndex, deletes_index_when_build_throws)
{
    Mat features8u(32, 32, CV_8UC1);
    theRNG().fill(features8u, RNG::UNIFORM, 0, 256);

    const int badKeys[] = { 0, static_cast<int>(sizeof(size_t) * CHAR_BIT) };
    for (int k = 0; k < 2; ++k)
    {
        bool threw = false;
        try
        {
            flann::GenericIndex<cvflann::Hamming<unsigned char> > idx(
                features8u, cvflann::LshIndexParams(8, badKeys[k], 2));
            CV_UNUSED(idx);
        }
        catch (const cv::Exception& e)
        {
            threw = true;
            EXPECT_EQ(Error::StsBadArg, e.code);
        }
        EXPECT_TRUE(threw) << "badKey=" << badKeys[k];
    }

    Mat features32f(32, 8, CV_32FC1);
    theRNG().fill(features32f, RNG::UNIFORM, -1.f, 1.f);
    EXPECT_THROW({
        flann::GenericIndex<cvflann::L2<float> > idx(
            features32f, cvflann::KMeansIndexParams(1));
        CV_UNUSED(idx);
    }, cv::Exception);
}

}} // namespace
