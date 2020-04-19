// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/core/utils/logger.hpp>
#include "../src/utils/logtagmanager.hpp"
#include "../src/utils/logtagconfigparser.hpp"

// Because "LogTagConfigParser" isn't exported from "opencv_core", the only way
// to perform test is to compile the source code into "opencv_test_core".
// This workaround may cause step debugger breakpoints to work unreliably.
#if 1
#include "../src/utils/logtagconfigparser.cpp"
#endif

using cv::utils::logging::LogTagConfigParser;

namespace opencv_test {
namespace {

typedef testing::TestWithParam<tuple<std::string, cv::utils::logging::LogLevel>> GlobalShouldSucceedTests;

TEST_P(GlobalShouldSucceedTests, globalCases)
{
    const std::string input = get<0>(GetParam());
    const cv::utils::logging::LogLevel expectedLevel = get<1>(GetParam());
    LogTagConfigParser parser;
    parser.parse(input);
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u) << "Malformed list should be empty";
    EXPECT_TRUE(parser.getGlobalConfig().isGlobal);
    EXPECT_EQ(parser.getGlobalConfig().level, expectedLevel);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u) << "Specifying global log level should not emit full names";
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u) << "Specifying global log level should not emit first name part result";
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u) << "Specifying global log level should not emit any name part result";
}

INSTANTIATE_TEST_CASE_P(Core_LogTagConfigParser, GlobalShouldSucceedTests,
    testing::Values(
        // Following test cases omit the name part
        std::make_tuple("S", cv::utils::logging::LOG_LEVEL_SILENT),
        std::make_tuple("SILENT", cv::utils::logging::LOG_LEVEL_SILENT),
        std::make_tuple("F", cv::utils::logging::LOG_LEVEL_FATAL),
        std::make_tuple("FATAL", cv::utils::logging::LOG_LEVEL_FATAL),
        std::make_tuple("E", cv::utils::logging::LOG_LEVEL_ERROR),
        std::make_tuple("ERROR", cv::utils::logging::LOG_LEVEL_ERROR),
        std::make_tuple("W", cv::utils::logging::LOG_LEVEL_WARNING),
        std::make_tuple("WARN", cv::utils::logging::LOG_LEVEL_WARNING),
        std::make_tuple("WARNING", cv::utils::logging::LOG_LEVEL_WARNING),
        std::make_tuple("I", cv::utils::logging::LOG_LEVEL_INFO),
        std::make_tuple("INFO", cv::utils::logging::LOG_LEVEL_INFO),
        std::make_tuple("D", cv::utils::logging::LOG_LEVEL_DEBUG),
        std::make_tuple("DEBUG", cv::utils::logging::LOG_LEVEL_DEBUG),
        std::make_tuple("V", cv::utils::logging::LOG_LEVEL_VERBOSE),
        std::make_tuple("VERBOSE", cv::utils::logging::LOG_LEVEL_VERBOSE),
        // Following test cases uses a single asterisk as name
        std::make_tuple("*:S", cv::utils::logging::LOG_LEVEL_SILENT),
        std::make_tuple("*:SILENT", cv::utils::logging::LOG_LEVEL_SILENT),
        std::make_tuple("*:V", cv::utils::logging::LOG_LEVEL_VERBOSE),
        std::make_tuple("*:VERBOSE", cv::utils::logging::LOG_LEVEL_VERBOSE)
    )
);

// GlobalShouldSucceedPairedTests, globalNameHandling
//
// The following tests use a strategy of performing two tests as a pair, and require the pair
// to succeed, in order to avoid false negatives due to default settings.
// The first input string is supposed to set global to SILENT, the second input string VERBOSE.

typedef testing::TestWithParam<tuple<std::string, std::string>> GlobalShouldSucceedPairedTests;

TEST_P(GlobalShouldSucceedPairedTests, globalNameHandling)
{
    const auto firstExpected = cv::utils::logging::LOG_LEVEL_SILENT;
    const auto secondExpected = cv::utils::logging::LOG_LEVEL_VERBOSE;
    //
    const std::string firstInput = get<0>(GetParam());
    LogTagConfigParser firstParser;
    firstParser.parse(firstInput);
    ASSERT_FALSE(firstParser.hasMalformed());
    ASSERT_EQ(firstParser.getMalformed().size(), 0u) << "Malformed list should be empty";
    ASSERT_TRUE(firstParser.getGlobalConfig().isGlobal);
    ASSERT_EQ(firstParser.getFullNameConfigs().size(), 0u) << "Specifying global log level should not emit full names";
    ASSERT_EQ(firstParser.getFirstPartConfigs().size(), 0u) << "Specifying global log level should not emit first name part result";
    ASSERT_EQ(firstParser.getAnyPartConfigs().size(), 0u) << "Specifying global log level should not emit any name part result";
    const cv::utils::logging::LogLevel firstActual = firstParser.getGlobalConfig().level;
    //
    const std::string secondInput = get<1>(GetParam());
    LogTagConfigParser secondParser;
    secondParser.parse(secondInput);
    ASSERT_FALSE(secondParser.hasMalformed());
    ASSERT_EQ(secondParser.getMalformed().size(), 0u) << "Malformed list should be empty";
    ASSERT_TRUE(secondParser.getGlobalConfig().isGlobal);
    ASSERT_EQ(secondParser.getFullNameConfigs().size(), 0u) << "Specifying global log level should not emit full names";
    ASSERT_EQ(secondParser.getFirstPartConfigs().size(), 0u) << "Specifying global log level should not emit first name part result";
    ASSERT_EQ(secondParser.getAnyPartConfigs().size(), 0u) << "Specifying global log level should not emit any name part result";
    const cv::utils::logging::LogLevel secondActual = secondParser.getGlobalConfig().level;
    //
    EXPECT_EQ(firstActual, firstExpected);
    EXPECT_EQ(secondActual, secondExpected);
}

// Following test cases uses lowercase "global" as name
INSTANTIATE_TEST_CASE_P(Core_LogTagConfigParser, GlobalShouldSucceedPairedTests,
    testing::Values(
        std::make_tuple("global:S", "global:V"),
        std::make_tuple("global:SILENT", "global:VERBOSE")
    )
);

// In the next few smoke tests, the use of EXPECT versus ASSERT is as follows.
// Each test will try to read the first element from one of the vector results.
// Prior to that, the vector need to be ASSERT'ed to have at least one element.
// All remaining assertions in the test body would use EXPECT instead.

TEST(Core_LogTagConfigParser, FullNameSmokeTest)
{
    LogTagConfigParser parser;
    parser.parse("something:S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    ASSERT_EQ(parser.getFullNameConfigs().size(), 1u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
    EXPECT_STREQ(parser.getFullNameConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, FirstPartSmokeTest_NoPeriod)
{
    LogTagConfigParser parser;
    parser.parse("something*:S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    ASSERT_EQ(parser.getFirstPartConfigs().size(), 1u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
    EXPECT_STREQ(parser.getFirstPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, FirstPartSmokeTest_WithPeriod)
{
    LogTagConfigParser parser;
    parser.parse("something.*:S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    ASSERT_EQ(parser.getFirstPartConfigs().size(), 1u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
    EXPECT_STREQ(parser.getFirstPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, AnyPartSmokeTest_PrecedeAsterisk_NoPeriod)
{
    LogTagConfigParser parser;
    parser.parse("*something:S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    ASSERT_EQ(parser.getAnyPartConfigs().size(), 1u);
    EXPECT_STREQ(parser.getAnyPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, AnyPartSmokeTest_PrecedeAsterisk_WithPeriod)
{
    LogTagConfigParser parser;
    parser.parse("*.something:S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    ASSERT_EQ(parser.getAnyPartConfigs().size(), 1u);
    EXPECT_STREQ(parser.getAnyPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, AnyPartSmokeTest_PrecedeFollowAsterisks_NoPeriod)
{
    LogTagConfigParser parser;
    parser.parse("*something*:S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    ASSERT_EQ(parser.getAnyPartConfigs().size(), 1u);
    EXPECT_STREQ(parser.getAnyPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, AnyPartSmokeTest_PrecedeFollowAsterisks_WithPeriod)
{
    LogTagConfigParser parser;
    parser.parse("*.something.*:S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    ASSERT_EQ(parser.getAnyPartConfigs().size(), 1u);
    EXPECT_STREQ(parser.getAnyPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, FullName_EqualSign_ShouldSucceed)
{
    LogTagConfigParser parser;
    parser.parse("something=S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    ASSERT_EQ(parser.getFullNameConfigs().size(), 1u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
    EXPECT_STREQ(parser.getFullNameConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, FirstPart_EqualSign_ShouldSucceed)
{
    LogTagConfigParser parser;
    parser.parse("something*=S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    ASSERT_EQ(parser.getFirstPartConfigs().size(), 1u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
    EXPECT_STREQ(parser.getFirstPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, AnyPart_EqualSign_ShouldSucceed)
{
    LogTagConfigParser parser;
    parser.parse("*something*=S");
    EXPECT_FALSE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 0u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    ASSERT_EQ(parser.getAnyPartConfigs().size(), 1u);
    EXPECT_STREQ(parser.getAnyPartConfigs().at(0u).namePart.c_str(), "something");
}

TEST(Core_LogTagConfigParser, DuplicateColon_ShouldFail)
{
    LogTagConfigParser parser;
    parser.parse("something::S");
    EXPECT_TRUE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 1u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
}

TEST(Core_LogTagConfigParser, DuplicateEqual_ShouldFail)
{
    LogTagConfigParser parser;
    parser.parse("something==S");
    EXPECT_TRUE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 1u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
}

TEST(Core_LogTagConfigParser, DuplicateColonAndEqual_ShouldFail)
{
    LogTagConfigParser parser;
    parser.parse("something:=S");
    EXPECT_TRUE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 1u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
}

TEST(Core_LogTagConfigParser, DuplicateEqualAndColon_ShouldFail)
{
    LogTagConfigParser parser;
    parser.parse("something=:S");
    EXPECT_TRUE(parser.hasMalformed());
    EXPECT_EQ(parser.getMalformed().size(), 1u);
    EXPECT_EQ(parser.getFullNameConfigs().size(), 0u);
    EXPECT_EQ(parser.getFirstPartConfigs().size(), 0u);
    EXPECT_EQ(parser.getAnyPartConfigs().size(), 0u);
}

}} // namespace
