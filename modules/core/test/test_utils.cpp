// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "opencv2/core/utils/logger.defines.hpp"
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

#include "opencv2/core/utils/filesystem.private.hpp"

#ifndef OPENCV_DISABLE_THREAD_SUPPORT
#include "test_utils_tls.impl.hpp"
#endif

namespace opencv_test { namespace {

static const char * const keys =
    "{ h help    |       | print help }"
    "{ i info    | false | print info }"
    "{ t true    | true  | true value }"
    "{ n unused  |       | dummy }"
;

TEST(CommandLineParser, testFailure)
{
    const char* argv[] = {"<bin>", "-q"};
    const int argc = 2;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_ANY_THROW(parser.has("q"));
    EXPECT_ANY_THROW(parser.get<bool>("q"));
    EXPECT_ANY_THROW(parser.get<bool>(0));

    parser.get<bool>("h");
    EXPECT_FALSE(parser.check());
}

TEST(CommandLineParser, testHas_noValues)
{
    const char* argv[] = {"<bin>", "-h", "--info"};
    const int argc = 3;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_TRUE(parser.has("help"));
    EXPECT_TRUE(parser.has("h"));
    EXPECT_TRUE(parser.get<bool>("help"));
    EXPECT_TRUE(parser.get<bool>("h"));
    EXPECT_TRUE(parser.has("info"));
    EXPECT_TRUE(parser.has("i"));
    EXPECT_TRUE(parser.get<bool>("info"));
    EXPECT_TRUE(parser.get<bool>("i"));
    EXPECT_TRUE(parser.get<bool>("true"));
    EXPECT_TRUE(parser.get<bool>("t"));
    EXPECT_FALSE(parser.has("n"));
    EXPECT_FALSE(parser.has("unused"));
}
TEST(CommandLineParser, testHas_TrueValues)
{
    const char* argv[] = {"<bin>", "-h=TRUE", "--info=true"};
    const int argc = 3;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_TRUE(parser.has("help"));
    EXPECT_TRUE(parser.has("h"));
    EXPECT_TRUE(parser.get<bool>("help"));
    EXPECT_TRUE(parser.get<bool>("h"));
    EXPECT_TRUE(parser.has("info"));
    EXPECT_TRUE(parser.has("i"));
    EXPECT_TRUE(parser.get<bool>("info"));
    EXPECT_TRUE(parser.get<bool>("i"));
    EXPECT_TRUE(parser.get<bool>("true"));
    EXPECT_TRUE(parser.get<bool>("t"));
    EXPECT_FALSE(parser.has("n"));
    EXPECT_FALSE(parser.has("unused"));
}
TEST(CommandLineParser, testHas_TrueValues1)
{
    const char* argv[] = {"<bin>", "-h=1", "--info=1"};
    const int argc = 3;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_TRUE(parser.has("help"));
    EXPECT_TRUE(parser.has("h"));
    EXPECT_TRUE(parser.get<bool>("help"));
    EXPECT_TRUE(parser.get<bool>("h"));
    EXPECT_TRUE(parser.has("info"));
    EXPECT_TRUE(parser.has("i"));
    EXPECT_TRUE(parser.get<bool>("info"));
    EXPECT_TRUE(parser.get<bool>("i"));
    EXPECT_TRUE(parser.get<bool>("true"));
    EXPECT_TRUE(parser.get<bool>("t"));
    EXPECT_FALSE(parser.has("n"));
    EXPECT_FALSE(parser.has("unused"));
}
TEST(CommandLineParser, testHas_FalseValues0)
{
    const char* argv[] = {"<bin>", "-h=0", "--info=0"};
    const int argc = 3;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_TRUE(parser.has("help"));
    EXPECT_TRUE(parser.has("h"));
    EXPECT_FALSE(parser.get<bool>("help"));
    EXPECT_FALSE(parser.get<bool>("h"));
    EXPECT_TRUE(parser.has("info"));
    EXPECT_TRUE(parser.has("i"));
    EXPECT_FALSE(parser.get<bool>("info"));
    EXPECT_FALSE(parser.get<bool>("i"));
    EXPECT_TRUE(parser.get<bool>("true"));
    EXPECT_TRUE(parser.get<bool>("t"));
    EXPECT_FALSE(parser.has("n"));
    EXPECT_FALSE(parser.has("unused"));
}

TEST(CommandLineParser, testBoolOption_noArgs)
{
    const char* argv[] = {"<bin>"};
    const int argc = 1;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_FALSE(parser.get<bool>("help"));
    EXPECT_FALSE(parser.get<bool>("h"));
    EXPECT_FALSE(parser.get<bool>("info"));
    EXPECT_FALSE(parser.get<bool>("i"));
    EXPECT_TRUE(parser.get<bool>("true")); // default is true
    EXPECT_TRUE(parser.get<bool>("t"));
}

TEST(CommandLineParser, testBoolOption_noValues)
{
    const char* argv[] = {"<bin>", "-h", "--info"};
    const int argc = 3;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_TRUE(parser.get<bool>("help"));
    EXPECT_TRUE(parser.get<bool>("h"));
    EXPECT_TRUE(parser.get<bool>("info"));
    EXPECT_TRUE(parser.get<bool>("i"));
    EXPECT_TRUE(parser.get<bool>("true"));
    EXPECT_TRUE(parser.get<bool>("t"));
}

TEST(CommandLineParser, testBoolOption_TrueValues)
{
    const char* argv[] = {"<bin>", "-h=TrUe", "-t=1", "--info=true", "-n=truE"};
    const int argc = 5;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_TRUE(parser.get<bool>("help"));
    EXPECT_TRUE(parser.get<bool>("h"));
    EXPECT_TRUE(parser.get<bool>("info"));
    EXPECT_TRUE(parser.get<bool>("i"));
    EXPECT_TRUE(parser.get<bool>("true"));
    EXPECT_TRUE(parser.get<bool>("t"));
    EXPECT_TRUE(parser.get<bool>("unused"));
    EXPECT_TRUE(parser.get<bool>("n"));
}

TEST(CommandLineParser, testBoolOption_FalseValues)
{
    const char* argv[] = {"<bin>", "--help=FALSE", "-t=FaLsE", "-i=false", "-n=0"};
    const int argc = 5;
    cv::CommandLineParser parser(argc, argv, keys);
    EXPECT_FALSE(parser.get<bool>("help"));
    EXPECT_FALSE(parser.get<bool>("h"));
    EXPECT_FALSE(parser.get<bool>("info"));
    EXPECT_FALSE(parser.get<bool>("i"));
    EXPECT_FALSE(parser.get<bool>("true"));
    EXPECT_FALSE(parser.get<bool>("t"));
    EXPECT_FALSE(parser.get<bool>("unused"));
    EXPECT_FALSE(parser.get<bool>("n"));
}


static const char * const keys2 =
    "{ h help    |          | print help }"
    "{ @arg1     | default1 | param1 }"
    "{ @arg2     |          | param2 }"
    "{ n unused  |          | dummy }"
;

TEST(CommandLineParser, testPositional_noArgs)
{
    const char* argv[] = {"<bin>"};
    const int argc = 1;
    cv::CommandLineParser parser(argc, argv, keys2);
    EXPECT_TRUE(parser.has("@arg1"));
    EXPECT_FALSE(parser.has("@arg2"));
    EXPECT_EQ("default1", parser.get<String>("@arg1"));
    EXPECT_EQ("default1", parser.get<String>(0));

    EXPECT_EQ("", parser.get<String>("@arg2"));
    EXPECT_EQ("", parser.get<String>(1));
}

TEST(CommandLineParser, testPositional_default)
{
    const char* argv[] = {"<bin>", "test1", "test2"};
    const int argc = 3;
    cv::CommandLineParser parser(argc, argv, keys2);
    EXPECT_TRUE(parser.has("@arg1"));
    EXPECT_TRUE(parser.has("@arg2"));
    EXPECT_EQ("test1", parser.get<String>("@arg1"));
    EXPECT_EQ("test2", parser.get<String>("@arg2"));
    EXPECT_EQ("test1", parser.get<String>(0));
    EXPECT_EQ("test2", parser.get<String>(1));
}

TEST(CommandLineParser, testPositional_withFlagsBefore)
{
    const char* argv[] = {"<bin>", "-h", "test1", "test2"};
    const int argc = 4;
    cv::CommandLineParser parser(argc, argv, keys2);
    EXPECT_TRUE(parser.has("@arg1"));
    EXPECT_TRUE(parser.has("@arg2"));
    EXPECT_EQ("test1", parser.get<String>("@arg1"));
    EXPECT_EQ("test2", parser.get<String>("@arg2"));
    EXPECT_EQ("test1", parser.get<String>(0));
    EXPECT_EQ("test2", parser.get<String>(1));
}

TEST(CommandLineParser, testPositional_withFlagsAfter)
{
    const char* argv[] = {"<bin>", "test1", "test2", "-h"};
    const int argc = 4;
    cv::CommandLineParser parser(argc, argv, keys2);
    EXPECT_TRUE(parser.has("@arg1"));
    EXPECT_TRUE(parser.has("@arg2"));
    EXPECT_EQ("test1", parser.get<String>("@arg1"));
    EXPECT_EQ("test2", parser.get<String>("@arg2"));
    EXPECT_EQ("test1", parser.get<String>(0));
    EXPECT_EQ("test2", parser.get<String>(1));
}

TEST(CommandLineParser, testEmptyStringValue)
{
    static const char * const keys3 =
            "{ @pos0 |        | empty default value }"
            "{ @pos1 | <none> | forbid empty default value }";

    const char* argv[] = {"<bin>"};
    const int argc = 1;
    cv::CommandLineParser parser(argc, argv, keys3);
    // EXPECT_TRUE(parser.has("@pos0"));
    EXPECT_EQ("", parser.get<String>("@pos0"));
    EXPECT_TRUE(parser.check());

    EXPECT_FALSE(parser.has("@pos1"));
    parser.get<String>(1);
    EXPECT_FALSE(parser.check());
}

TEST(CommandLineParser, positional_regression_5074_equal_sign)
{
    static const char * const keys3 =
            "{ @eq0 |  | }"
            "{ eq1  |  | }";

    const char* argv[] = {"<bin>", "1=0", "--eq1=1=0"};
    const int argc = 3;
    cv::CommandLineParser parser(argc, argv, keys3);
    EXPECT_EQ("1=0", parser.get<String>("@eq0"));
    EXPECT_EQ("1=0", parser.get<String>(0));
    EXPECT_EQ("1=0", parser.get<String>("eq1"));
    EXPECT_TRUE(parser.check());
}


TEST(AutoBuffer, allocate_test)
{
    AutoBuffer<int, 5> abuf(2);
    EXPECT_EQ(2u, abuf.size());

    abuf.allocate(4);
    EXPECT_EQ(4u, abuf.size());

    abuf.allocate(6);
    EXPECT_EQ(6u, abuf.size());
}

TEST(CommandLineParser, testScalar)
{
    static const char * const keys3 =
            "{ s0 | 3 4 5 | default scalar }"
            "{ s1 |       | single value scalar }"
            "{ s2 |       | two values scalar (default with zeros) }"
            "{ s3 |       | three values scalar }"
            "{ s4 |       | four values scalar }"
            "{ s5 |       | five values scalar }";

    const char* argv[] = {"<bin>", "--s1=1.1", "--s3=1.1 2.2 3",
                          "--s4=-4.2 1 0 3", "--s5=5 -4 3 2 1"};
    const int argc = 5;
    CommandLineParser parser(argc, argv, keys3);
    EXPECT_EQ(parser.get<Scalar>("s0"), Scalar(3, 4, 5));
    EXPECT_EQ(parser.get<Scalar>("s1"), Scalar(1.1));
    EXPECT_EQ(parser.get<Scalar>("s2"), Scalar(0));
    EXPECT_EQ(parser.get<Scalar>("s3"), Scalar(1.1, 2.2, 3));
    EXPECT_EQ(parser.get<Scalar>("s4"), Scalar(-4.2, 1, 0, 3));
    EXPECT_EQ(parser.get<Scalar>("s5"), Scalar(5, -4, 3, 2));
}


TEST(Logger, DISABLED_message)
{
    int id = 42;
    CV_LOG_VERBOSE(NULL, 0, "Verbose message: " << id);
    CV_LOG_VERBOSE(NULL, 1, "Verbose message: " << id);
    CV_LOG_DEBUG(NULL, "Debug message: " << id);
    CV_LOG_INFO(NULL, "Info message: " << id);
    CV_LOG_WARNING(NULL, "Warning message: " << id);
    CV_LOG_ERROR(NULL, "Error message: " << id);
    CV_LOG_FATAL(NULL, "Fatal message: " << id);
}

static int testLoggerMessageOnce(int id)
{
    CV_LOG_ONCE_VERBOSE(NULL, 0, "Verbose message: " << id++);
    CV_LOG_ONCE_VERBOSE(NULL, 1, "Verbose message: " << id++);
    CV_LOG_ONCE_DEBUG(NULL, "Debug message: " << id++);
    CV_LOG_ONCE_INFO(NULL, "Info message: " << id++);
    CV_LOG_ONCE_WARNING(NULL, "Warning message: " << id++);
    CV_LOG_ONCE_ERROR(NULL, "Error message: " << id++);
    // doesn't make sense: CV_LOG_ONCE_FATAL
    return id;
}
TEST(Logger, DISABLED_message_once)
{
    int check_id_first = testLoggerMessageOnce(42);
    EXPECT_GT(check_id_first, 42);
    int check_id_second = testLoggerMessageOnce(0);
    EXPECT_EQ(0, check_id_second);
}

TEST(Logger, DISABLED_message_if)
{
    for (int i = 0; i < 100; i++)
    {
        CV_LOG_IF_VERBOSE(NULL, 0, i == 0 || i == 42, "Verbose message: " << i);
        CV_LOG_IF_VERBOSE(NULL, 1, i == 0 || i == 42, "Verbose message: " << i);
        CV_LOG_IF_DEBUG(NULL, i == 0 || i == 42, "Debug message: " << i);
        CV_LOG_IF_INFO(NULL, i == 0 || i == 42, "Info message: " << i);
        CV_LOG_IF_WARNING(NULL, i == 0 || i == 42, "Warning message: " << i);
        CV_LOG_IF_ERROR(NULL, i == 0 || i == 42, "Error message: " << i);
        CV_LOG_IF_FATAL(NULL, i == 0 || i == 42, "Fatal message: " << i);
    }
}

#if OPENCV_HAVE_FILESYSTEM_SUPPORT
TEST(Samples, findFile)
{
    cv::utils::logging::LogLevel prev = cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
    cv::String path;
    ASSERT_NO_THROW(path = samples::findFile("lena.jpg", false));
    EXPECT_NE(std::string(), path.c_str());
    cv::utils::logging::setLogLevel(prev);
}

TEST(Samples, findFile_missing)
{
    cv::utils::logging::LogLevel prev = cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
    cv::String path;
    ASSERT_ANY_THROW(path = samples::findFile("non-existed.file", true));
    cv::utils::logging::setLogLevel(prev);
}
#endif // OPENCV_HAVE_FILESYSTEM_SUPPORT

template <typename T>
inline bool buffers_overlap(T * first, size_t first_num, T * second, size_t second_num)
{
    // cerr << "[" << (void*)first << " : " << (void*)(first + first_num) << ")";
    // cerr << " X ";
    // cerr << "[" << (void*)second << " : " << (void*)(second + second_num) << ")";
    // cerr << endl;
    bool res = false;
    res |= (second <= first) && (first < second + second_num);
    res |= (second < first + first_num) && (first + first_num < second + second_num);
    return res;
}

typedef testing::TestWithParam<bool> BufferArea;

TEST_P(BufferArea, basic)
{
    const bool safe = GetParam();
    const size_t SZ = 3;
    int * int_ptr = NULL;
    uchar * uchar_ptr = NULL;
    double * dbl_ptr = NULL;
    {
        cv::utils::BufferArea area(safe);
        area.allocate(int_ptr, SZ);
        area.allocate(uchar_ptr, SZ);
        area.allocate(dbl_ptr, SZ);
        area.commit();
        ASSERT_TRUE(int_ptr != NULL);
        ASSERT_TRUE(uchar_ptr != NULL);
        ASSERT_TRUE(dbl_ptr != NULL);
        EXPECT_EQ((size_t)0, (size_t)int_ptr % sizeof(int));
        EXPECT_EQ((size_t)0, (size_t)dbl_ptr % sizeof(double));
        for (size_t i = 0; i < SZ; ++i)
        {
            int_ptr[i] = (int)i + 1;
            uchar_ptr[i] = (uchar)i + 1;
            dbl_ptr[i] = (double)i + 1;
        }
        area.zeroFill(int_ptr);
        area.zeroFill(uchar_ptr);
        area.zeroFill(dbl_ptr);
        for (size_t i = 0; i < SZ; ++i)
        {
            EXPECT_EQ((int)0, int_ptr[i]);
            EXPECT_EQ((uchar)0, uchar_ptr[i]);
            EXPECT_EQ((double)0, dbl_ptr[i]);
        }
    }
    EXPECT_TRUE(int_ptr == NULL);
    EXPECT_TRUE(uchar_ptr == NULL);
    EXPECT_TRUE(dbl_ptr == NULL);
}

TEST_P(BufferArea, align)
{
    const bool safe = GetParam();
    const size_t SZ = 3;
    const size_t CNT = 5;
    typedef int T;
    T * buffers[CNT] = {0};
    {
        cv::utils::BufferArea area(safe);
        // allocate buffers with 3 elements with growing alignment (power of two)
        for (size_t i = 0; i < CNT; ++i)
        {
            const ushort ALIGN = static_cast<ushort>(sizeof(T) << i);
            EXPECT_TRUE(buffers[i] == NULL);
            area.allocate(buffers[i], SZ, ALIGN);
        }
        area.commit();
        for (size_t i = 0; i < CNT; ++i)
        {
            const ushort ALIGN = static_cast<ushort>(sizeof(T) << i);
            EXPECT_TRUE(buffers[i] != NULL);
            EXPECT_EQ((size_t)0, reinterpret_cast<size_t>(buffers[i]) % ALIGN);
            if (i < CNT - 1)
            {
                SCOPED_TRACE(i);
                EXPECT_FALSE(buffers_overlap(buffers[i], SZ, buffers[i + 1], SZ))
                    << "Buffers overlap: "
                    << buffers[i] << " (" << SZ << " elems)"
                    << " and "
                    << buffers[i + 1] << " (" << SZ << " elems)"
                    << " (element size: " << sizeof(T) << ")";
            }
        }
    }
    for (size_t i = 0; i < CNT; ++i)
    {
        EXPECT_TRUE(buffers[i] == NULL);
    }
}

TEST_P(BufferArea, default_align)
{
    const bool safe = GetParam();
    const size_t CNT = 100;
    const ushort ALIGN = 64;
    typedef int T;
    T * buffers[CNT] = {0};
    {
        cv::utils::BufferArea area(safe);
        // allocate buffers with 1-99 elements with default alignment
        for (size_t i = 0; i < CNT; ++ i)
        {
            EXPECT_TRUE(buffers[i] == NULL);
            area.allocate(buffers[i], i + 1, ALIGN);
        }
        area.commit();
        for (size_t i = 0; i < CNT; ++i)
        {
            EXPECT_TRUE(buffers[i] != NULL);
            EXPECT_EQ((size_t)0, reinterpret_cast<size_t>(buffers[i]) % ALIGN);
            if (i < CNT - 1)
            {
                SCOPED_TRACE(i);
                EXPECT_FALSE(buffers_overlap(buffers[i], i + 1, buffers[i + 1], i + 2))
                    << "Buffers overlap: "
                    << buffers[i] << " (" << i + 1 << " elems)"
                    << " and "
                    << buffers[i + 1] << " (" << i + 2 << " elems)"
                    << " (element size: " << sizeof(T) << ")";
            }
        }
    }
}

TEST_P(BufferArea, bad)
{
    const bool safe = GetParam();
    int * ptr = 0;
    cv::utils::BufferArea area(safe);
    EXPECT_ANY_THROW(area.allocate(ptr, 0)); // bad size
    EXPECT_ANY_THROW(area.allocate(ptr, 1, 0)); // bad alignment
    EXPECT_ANY_THROW(area.allocate(ptr, 1, 3)); // bad alignment
    ptr = (int*)1;
    EXPECT_ANY_THROW(area.allocate(ptr, 1)); // non-zero pointer
}

INSTANTIATE_TEST_CASE_P(/**/, BufferArea, testing::Values(true, false));


}} // namespace
