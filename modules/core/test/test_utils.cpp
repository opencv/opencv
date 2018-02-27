// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

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

}} // namespace
