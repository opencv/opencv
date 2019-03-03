// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/core/utils/logger.hpp>

#include "../src/utils/logtagmanager.hpp"
#include "../src/utils/logtagconfigparser.hpp"

// Because "LogTagManager" isn't exported from "opencv_core", the only way
// to perform test is to compile the source code into "opencv_test_core".
// This workaround may cause step debugger breakpoints to work unreliably.
#if 1
#include "../src/utils/logtagmanager.cpp"
#endif

namespace opencv_test {
namespace {

using LogLevel = cv::utils::logging::LogLevel;
using LogTag = cv::utils::logging::LogTag;
using LogTagManager = cv::utils::logging::LogTagManager;

// Value to initialize log tag constructors
static constexpr const LogLevel constTestLevelBegin = cv::utils::logging::LOG_LEVEL_SILENT;

// Value to be set as part of test (to simulate runtime changes)
static constexpr const LogLevel constTestLevelChanged = cv::utils::logging::LOG_LEVEL_VERBOSE;

// An alternate value to initialize log tag constructors,
// for test cases where two distinct initialization values are needed.
static constexpr const LogLevel constTestLevelAltBegin = cv::utils::logging::LOG_LEVEL_FATAL;

// An alternate value to be set as part of test (to simulate runtime changes),
// for test cases where two distinct runtime set values are needed.
static constexpr const LogLevel constTestLevelAltChanged = cv::utils::logging::LOG_LEVEL_DEBUG;

// Enums for specifying which LogTagManager method to call.
// Used in parameterized tests.
enum class ByWhat
{
    ByFullName = 0,
    ByFirstPart = 1,
    ByAnyPart = 2
};

std::ostream& operator<< (std::ostream& strm, ByWhat byWhat)
{
    switch (byWhat)
    {
    case ByWhat::ByFullName:
        strm << "ByFullName";
        break;
    case ByWhat::ByFirstPart:
        strm << "ByFirstPart";
        break;
    case ByWhat::ByAnyPart:
        strm << "ByAnyPart";
        break;
    default:
        strm << "(invalid ByWhat enum" << (int)byWhat << ")";
        break;
    }
    return strm;
}

// Enums for describing relative timing.
// Used in parameterized tests.
enum class Timing
{
    Never = 0,
    Before = 1,
    After = 2
};

std::ostream& operator<< (std::ostream& strm, Timing timing)
{
    switch (timing)
    {
    case Timing::Never:
        strm << "Never";
        break;
    case Timing::Before:
        strm << "Before";
        break;
    case Timing::After:
        strm << "After";
        break;
    default:
        strm << "(invalid Timing enum" << (int)timing << ")";
        break;
    }
    return strm;
}

// Enums for selecting the substrings used in substring confusion tests.
enum class SubstringType
{
    Prefix = 0,
    Midstring = 1,
    Suffix = 2,
    Straddle = 3
};

std::ostream& operator<< (std::ostream& strm, SubstringType substringType)
{
    switch (substringType)
    {
    case SubstringType::Prefix:
        strm << "Prefix";
        break;
    case SubstringType::Midstring:
        strm << "Midstring";
        break;
    case SubstringType::Suffix:
        strm << "Suffix";
        break;
    case SubstringType::Straddle:
        strm << "Straddle";
        break;
    default:
        strm << "(invalid SubstringType enum: " << (int)substringType << ")";
        break;
    }
    return strm;
}

// A base fixture consisting of the LogTagManager.
// Note that an instance of LogTagManager contains its own instance of "global" log tag.
class LogTagManagerTestFixture
    : public ::testing::Test
{
protected:
    LogTagManager m_logTagManager;

public:
    LogTagManagerTestFixture(LogLevel initGlobalLogLevel)
        : m_logTagManager(initGlobalLogLevel)
    {
    }

    ~LogTagManagerTestFixture()
    {
    }
};

// LogTagManagerGlobalSmokeTest verifies that the "global" log tag works as intended.
class LogTagManagerGlobalSmokeTest
    : public LogTagManagerTestFixture
{
protected:
    LogTag* m_actualGlobalLogTag;

protected:
    static constexpr const char* m_globalTagName = "global";

public:
    LogTagManagerGlobalSmokeTest()
        : LogTagManagerTestFixture(constTestLevelBegin)
        , m_actualGlobalLogTag(nullptr)
    {
    }
};

TEST_F(LogTagManagerGlobalSmokeTest, AfterCtorCanGetGlobalWithoutAssign)
{
    EXPECT_NE(m_logTagManager.get(m_globalTagName), nullptr);
}

TEST_F(LogTagManagerGlobalSmokeTest, AfterCtorGetGlobalHasDefaultLevel)
{
    auto globalLogTag = m_logTagManager.get(m_globalTagName);
    ASSERT_NE(globalLogTag, nullptr);
    EXPECT_EQ(globalLogTag->level, constTestLevelBegin);
}

TEST_F(LogTagManagerGlobalSmokeTest, AfterCtorCanSetGlobalByFullName)
{
    m_logTagManager.setLevelByFullName(m_globalTagName, constTestLevelChanged);
    auto globalLogTag = m_logTagManager.get(m_globalTagName);
    ASSERT_NE(globalLogTag, nullptr);
    EXPECT_EQ(globalLogTag->level, constTestLevelChanged);
}

#if 0
// "global" level is not supposed to be settable by name-parts.
// Therefore this test code is supposed to fail.
TEST_F(LogTagManagerGlobalSmokeTest, DISABLED_AfterCtorCanSetGlobalByFirstPart)
{
    m_logTagManager.setLevelByFirstPart(m_globalTagName, constTestLevelChanged);
    auto globalLogTag = m_logTagManager.get(m_globalTagName);
    ASSERT_NE(globalLogTag, nullptr);
    EXPECT_EQ(globalLogTag->level, constTestLevelChanged);
}
#endif

#if 0
// "global" level is not supposed to be settable by name-parts.
// Therefore this test code is supposed to fail.
TEST_F(LogTagManagerGlobalSmokeTest, DISABLED_AfterCtorCanSetGlobalByAnyPart)
{
    m_logTagManager.setLevelByAnyPart(m_globalTagName, constTestLevelChanged);
    auto globalLogTag = m_logTagManager.get(m_globalTagName);
    ASSERT_NE(globalLogTag, nullptr);
    EXPECT_EQ(globalLogTag->level, constTestLevelChanged);
}
#endif

// LogTagManagerNonGlobalSmokeTest performs basic smoke tests to verify that
// a log tag (that is not the "global" log tag) can be assigned, and its
// log level can be configured.
class LogTagManagerNonGlobalSmokeTest
    : public LogTagManagerTestFixture
{
protected:
    LogTag m_something;

protected:
    static constexpr const char* m_somethingTagName = "something";

public:
    LogTagManagerNonGlobalSmokeTest()
        : LogTagManagerTestFixture(constTestLevelAltBegin)
        , m_something(m_somethingTagName, constTestLevelBegin)
    {
    }
};

TEST_F(LogTagManagerNonGlobalSmokeTest, CanAssignTagAndThenGet)
{
    m_logTagManager.assign(m_something.name, &m_something);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
}

TEST_F(LogTagManagerNonGlobalSmokeTest, CanAssignTagAndThenGetAndShouldHaveDefaultLevel)
{
    m_logTagManager.assign(m_something.name, &m_something);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
    EXPECT_EQ(m_logTagManager.get(m_somethingTagName)->level, constTestLevelBegin);
}

TEST_F(LogTagManagerNonGlobalSmokeTest, CanSetLevelByFullNameBeforeAssignTag)
{
    m_logTagManager.setLevelByFullName(m_somethingTagName, constTestLevelChanged);
    m_logTagManager.assign(m_something.name, &m_something);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
    EXPECT_EQ(m_logTagManager.get(m_somethingTagName)->level, constTestLevelChanged);
    EXPECT_NE(m_logTagManager.get(m_somethingTagName)->level, constTestLevelBegin) << "Should not be left unchanged (default value)";
}

TEST_F(LogTagManagerNonGlobalSmokeTest, CanSetLevelByFirstPartBeforeAssignTag)
{
    m_logTagManager.setLevelByFirstPart(m_somethingTagName, constTestLevelChanged);
    m_logTagManager.assign(m_something.name, &m_something);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
    EXPECT_EQ(m_logTagManager.get(m_somethingTagName)->level, constTestLevelChanged);
    EXPECT_NE(m_logTagManager.get(m_somethingTagName)->level, constTestLevelBegin) << "Should not be left unchanged (default value)";
}

TEST_F(LogTagManagerNonGlobalSmokeTest, CanSetLevelByAnyPartBeforeAssignTag)
{
    m_logTagManager.setLevelByAnyPart(m_somethingTagName, constTestLevelChanged);
    m_logTagManager.assign(m_something.name, &m_something);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
    EXPECT_EQ(m_logTagManager.get(m_somethingTagName)->level, constTestLevelChanged);
    EXPECT_NE(m_logTagManager.get(m_somethingTagName)->level, constTestLevelBegin) << "Should not be left unchanged (default value)";
}

TEST_F(LogTagManagerNonGlobalSmokeTest, CanSetLevelByFullNameAfterAssignTag)
{
    m_logTagManager.assign(m_something.name, &m_something);
    m_logTagManager.setLevelByFullName(m_somethingTagName, constTestLevelChanged);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
    EXPECT_EQ(m_logTagManager.get(m_somethingTagName)->level, constTestLevelChanged);
    EXPECT_NE(m_logTagManager.get(m_somethingTagName)->level, constTestLevelBegin) << "Should not be left unchanged (default value)";
}

TEST_F(LogTagManagerNonGlobalSmokeTest, CanSetLevelByFirstPartAfterAssignTag)
{
    m_logTagManager.assign(m_something.name, &m_something);
    m_logTagManager.setLevelByFirstPart(m_somethingTagName, constTestLevelChanged);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
    EXPECT_EQ(m_logTagManager.get(m_somethingTagName)->level, constTestLevelChanged);
    EXPECT_NE(m_logTagManager.get(m_somethingTagName)->level, constTestLevelBegin) << "Should not be left unchanged (default value)";
}

TEST_F(LogTagManagerNonGlobalSmokeTest, CanSetLevelByAnyPartAfterAssignTag)
{
    m_logTagManager.assign(m_something.name, &m_something);
    m_logTagManager.setLevelByAnyPart(m_somethingTagName, constTestLevelChanged);
    ASSERT_EQ(m_logTagManager.get(m_somethingTagName), &m_something);
    EXPECT_EQ(m_logTagManager.get(m_somethingTagName)->level, constTestLevelChanged);
    EXPECT_NE(m_logTagManager.get(m_somethingTagName)->level, constTestLevelBegin) << "Should not be left unchanged (default value)";
}

// Non-confusion tests use two or more (non-global) log tags with chosen names to verify
// that LogTagManager does not accidentally apply log levels to log tags other than the
// ones intended.


// LogTagManagerSubstrNonConfusionSmokeTest are non-confusion tests that focus on
// substrings. Keep in mind string matching in LogTagManager must be aligned to
// the period delimiter; substrings are not considered for matching.
class LogTagManagerSubstrNonConfusionFixture
    : public LogTagManagerTestFixture
    , public ::testing::WithParamInterface<std::tuple<SubstringType, bool, Timing, ByWhat>>
{
public:
    struct MyTestParam
    {
        const SubstringType detractorNameSelector;
        const bool initTargetTagWithCall;
        const Timing whenToAssignDetractorTag;
        const ByWhat howToSetDetractorLevel;
        MyTestParam(const std::tuple<SubstringType, bool, Timing, ByWhat>& args)
            : detractorNameSelector(std::get<0u>(args))
            , initTargetTagWithCall(std::get<1u>(args))
            , whenToAssignDetractorTag(std::get<2u>(args))
            , howToSetDetractorLevel(std::get<3u>(args))
        {
        }
    };

    // The following substrings are all derived from "soursop".
    // In particular, "ours" is a substring of "soursop".
protected:
    LogTag tagSour;
    LogTag tagSop;
    LogTag tagSoursop;
    LogTag tagSourDotSop;
    LogTag tagOurs;

protected:
    static constexpr const char* strSour = "sour";
    static constexpr const char* strSop = "sop";
    static constexpr const char* strSoursop = "soursop";
    static constexpr const char* strSourDotSop = "sour.sop";
    static constexpr const char* strOurs = "ours";

public:
    LogTagManagerSubstrNonConfusionFixture()
        : LogTagManagerTestFixture(constTestLevelAltBegin)
        , tagSour(strSour, constTestLevelBegin)
        , tagSop(strSop, constTestLevelBegin)
        , tagSoursop(strSoursop, constTestLevelBegin)
        , tagSourDotSop(strSourDotSop, constTestLevelBegin)
        , tagOurs(strOurs, constTestLevelBegin)
    {
    }

    const char* selectDetractorName(const MyTestParam& myTestParam)
    {
        switch (myTestParam.detractorNameSelector)
        {
        case SubstringType::Prefix:
            return strSour;
        case SubstringType::Midstring:
            return strOurs;
        case SubstringType::Suffix:
            return strSop;
        case SubstringType::Straddle:
            return strSourDotSop;
        default:
            throw std::logic_error("LogTagManagerSubstrNonConfusionTest::selectDetractorName");
        }
    }

    LogTag& selectDetractorTag(const MyTestParam& myTestParam)
    {
        switch (myTestParam.detractorNameSelector)
        {
        case SubstringType::Prefix:
            return tagSour;
        case SubstringType::Midstring:
            return tagOurs;
        case SubstringType::Suffix:
            return tagSop;
        case SubstringType::Straddle:
            return tagSourDotSop;
        default:
            throw std::logic_error("LogTagManagerSubstrNonConfusionTest::selectDetractorName");
        }
    }
};

INSTANTIATE_TEST_CASE_P(
    LogTagManagerSubstrNonConfusionTest,
    LogTagManagerSubstrNonConfusionFixture,
    ::testing::Combine(
        ::testing::Values(SubstringType::Prefix, SubstringType::Midstring, SubstringType::Suffix, SubstringType::Straddle),
        ::testing::Values(false, true),
        ::testing::Values(Timing::Never, Timing::Before, Timing::After),
        ::testing::Values(ByWhat::ByFullName, ByWhat::ByFirstPart, ByWhat::ByAnyPart)
    )
);

TEST_P(LogTagManagerSubstrNonConfusionFixture, ParameterizedTestFunc)
{
    const auto myTestParam = MyTestParam(GetParam());
    const char* detractorName = selectDetractorName(myTestParam);
    LogTag& detractorTag = selectDetractorTag(myTestParam);
    // Target tag is assigned
    m_logTagManager.assign(tagSoursop.name, &tagSoursop);
    // If detractor tag is to be assigned "before"
    if (myTestParam.whenToAssignDetractorTag == Timing::Before)
    {
        m_logTagManager.assign(detractorName, &detractorTag);
    }
    // Initialize target tag to constTestLevelChanged
    if (myTestParam.initTargetTagWithCall)
    {
        m_logTagManager.setLevelByFullName(strSoursop, constTestLevelChanged);
    }
    else
    {
        tagSoursop.level = constTestLevelChanged;
    }
    // If detractor tag is to be assigned "after"
    if (myTestParam.whenToAssignDetractorTag == Timing::After)
    {
        m_logTagManager.assign(detractorName, &detractorTag);
    }
    // Set the log level using the detractor name
    switch (myTestParam.howToSetDetractorLevel)
    {
    case ByWhat::ByFullName:
        m_logTagManager.setLevelByFullName(detractorName, constTestLevelAltChanged);
        break;
    case ByWhat::ByFirstPart:
        m_logTagManager.setLevelByFirstPart(detractorName, constTestLevelAltChanged);
        break;
    case ByWhat::ByAnyPart:
        m_logTagManager.setLevelByAnyPart(detractorName, constTestLevelAltChanged);
        break;
    default:
        FAIL() << "Invalid parameterized test value, check test case.";
    }
    // Verifies that the target tag is not disturbed by changes made using detractor name
    ASSERT_EQ(m_logTagManager.get(strSoursop), &tagSoursop);
    EXPECT_EQ(tagSoursop.level, constTestLevelChanged);
    EXPECT_NE(tagSoursop.level, constTestLevelAltChanged) << "Should not be changed unless confusion bug exists";
}

}} // namespace
