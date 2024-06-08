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
static const LogLevel constTestLevelBegin = cv::utils::logging::LOG_LEVEL_SILENT;

// Value to be set as part of test (to simulate runtime changes)
static const LogLevel constTestLevelChanged = cv::utils::logging::LOG_LEVEL_VERBOSE;

// An alternate value to initialize log tag constructors,
// for test cases where two distinct initialization values are needed.
static const LogLevel constTestLevelAltBegin = cv::utils::logging::LOG_LEVEL_FATAL;

// An alternate value to be set as part of test (to simulate runtime changes),
// for test cases where two distinct runtime set values are needed.
static const LogLevel constTestLevelAltChanged = cv::utils::logging::LOG_LEVEL_DEBUG;

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
    static const char* const m_globalTagName;

public:
    LogTagManagerGlobalSmokeTest()
        : LogTagManagerTestFixture(constTestLevelBegin)
        , m_actualGlobalLogTag(nullptr)
    {
    }
};

const char* const LogTagManagerGlobalSmokeTest::m_globalTagName = "global";


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
    static const char* const m_somethingTagName;

public:
    LogTagManagerNonGlobalSmokeTest()
        : LogTagManagerTestFixture(constTestLevelAltBegin)
        , m_something(m_somethingTagName, constTestLevelBegin)
    {
    }
};

const char* const LogTagManagerNonGlobalSmokeTest::m_somethingTagName = "something";


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
    static const char* const strSour; // = "sour";
    static const char* const strSop; // = "sop";
    static const char* const strSoursop; // = "soursop";
    static const char* const strSourDotSop; // = "sour.sop";
    static const char* const strOurs; // = "ours";

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

const char* const LogTagManagerSubstrNonConfusionFixture::strSour = "sour";
const char* const LogTagManagerSubstrNonConfusionFixture::strSop = "sop";
const char* const LogTagManagerSubstrNonConfusionFixture::strSoursop = "soursop";
const char* const LogTagManagerSubstrNonConfusionFixture::strSourDotSop = "sour.sop";
const char* const LogTagManagerSubstrNonConfusionFixture::strOurs = "ours";


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

// LogTagManagerNamePartNonConfusionFixture are non-confusion tests that assumes
// no substring confusions are happening, and proceed to test matching by name parts.
// In particular, setLevelByFirstPart() and setLevelByAnyPart() are the focus of these tests.
class LogTagManagerNamePartNonConfusionFixture
    : public LogTagManagerTestFixture
    , public ::testing::WithParamInterface<std::tuple<int, ByWhat, int, Timing, Timing>>
{
public:
    struct MyTestParam
    {
        // Config tag name can only specify either full name or exactly one name part.
        // When specifying exactly one name part, there is a choice of matching the
        // first name part of a tag, or matching any name part that appears in a tag
        // name regardless of its position.
        const int configTagIndex;
        const ByWhat configMatchBy;
        const int targetTagIndex;
        const Timing whenToAssignTargetTag;
        const Timing whenToAssignConfigTag;
        MyTestParam(const std::tuple<int, ByWhat, int, Timing, Timing>& args)
            : configTagIndex(std::get<0u>(args))
            , configMatchBy(std::get<1u>(args))
            , targetTagIndex(std::get<2u>(args))
            , whenToAssignTargetTag(std::get<3u>(args))
            , whenToAssignConfigTag(std::get<4u>(args))
        {
        }
    };

protected:
    LogTag m_apple;
    LogTag m_banana;
    LogTag m_coconut;
    LogTag m_orange;
    LogTag m_pineapple;
    LogTag m_bananaDotOrange;
    LogTag m_bananaDotPineapple;
    LogTag m_coconutDotPineapple;

protected:
    static const char* const strApple; // = "apple";
    static const char* const strBanana; // = "banana";
    static const char* const strCoconut; // = "coconut";
    static const char* const strOrange; // = "orange";
    static const char* const strPineapple; // = "pineapple";
    static const char* const strBananaDotOrange; // = "banana.orange";
    static const char* const strBananaDotPineapple; // = "banana.pineapple";
    static const char* const strCoconutDotPineapple; // = "coconut.pineapple";

public:
    LogTagManagerNamePartNonConfusionFixture()
        : LogTagManagerTestFixture(constTestLevelAltBegin)
        , m_apple(strApple, constTestLevelBegin)
        , m_banana(strBanana, constTestLevelBegin)
        , m_coconut(strCoconut, constTestLevelBegin)
        , m_orange(strOrange, constTestLevelBegin)
        , m_pineapple(strPineapple, constTestLevelBegin)
        , m_bananaDotOrange(strBananaDotOrange, constTestLevelBegin)
        , m_bananaDotPineapple(strBananaDotPineapple, constTestLevelBegin)
        , m_coconutDotPineapple(strCoconutDotPineapple, constTestLevelBegin)
    {
    }

protected:
    LogTag* getLogTagByIndex(int index)
    {
        switch (index)
        {
        case 0:
            return &m_apple;
        case 1:
            return &m_banana;
        case 2:
            return &m_coconut;
        case 3:
            return &m_orange;
        case 4:
            return &m_pineapple;
        case 5:
            return &m_bananaDotOrange;
        case 6:
            return &m_bananaDotPineapple;
        case 7:
            return &m_coconutDotPineapple;
        default:
            ADD_FAILURE() << "Invalid parameterized test value, check test case. "
                << "Function LogTagManagerNamePartNonConfusionFixture::getLogTagByIndex.";
            return nullptr;
        }
    }

    // findTabulatedExpectedResult returns the hard-coded expected results for parameterized
    // test cases. The tables need updated if the index, name, or ordering of test tags are
    // changed.
    bool findTabulatedExpectedResult(const MyTestParam& myTestParam) const
    {
        // expectedResultUsingFirstPart:
        // Each row ("config") specifies the tag name specifier used to call setLevelByFirstPart().
        // Each column ("target") specifies whether an actual tag with the "target"
        // name would have its log level changed because of the call to setLevelByFirstPart().
        static const bool expectedResultUsingFirstPart[5][8] =
        {
            /*byFirstPart(apple)*/ { true, false, false, false, false, false, false, false },
            /*byFirstPart(banana)*/ { false, true, false, false, false, true, true, false },
            /*byFirstPart(coconut)*/ { false, false, true, false, false, false, false, true },
            /*byFirstPart(orange)*/ { false, false, false, true, false, false, false, false },
            /*byFirstPart(pineapple)*/ { false, false, false, false, true, false, false, false },
        };

        // expectedResultUsingAnyPart:
        // Each row ("config") specifies the tag name specifier used to call setLevelByAnyPart().
        // Each column ("target") specifies whether an actual tag with the "target"
        // name would have its log level changed because of the call to setLevelByAnyPart().
        static const bool expectedResultUsingAnyPart[5][8] =
        {
            /*byAnyPart(apple)*/ { true, false, false, false, false, false, false, false },
            /*byAnyPart(banana)*/ { false, true, false, false, false, true, true, false },
            /*byAnyPart(coconut)*/ { false, false, true, false, false, false, false, true },
            /*byAnyPart(orange)*/ { false, false, false, true, false, true, false, false },
            /*byAnyPart(pineapple)*/ { false, false, false, false, true, false, true, true },
        };

        switch (myTestParam.configMatchBy)
        {
        case ByWhat::ByFirstPart:
            return expectedResultUsingFirstPart[myTestParam.configTagIndex][myTestParam.targetTagIndex];
        case ByWhat::ByAnyPart:
            return expectedResultUsingAnyPart[myTestParam.configTagIndex][myTestParam.targetTagIndex];
        default:
            ADD_FAILURE() << "Invalid parameterized test value, check test case. "
                << "Function LogTagManagerNamePartNonConfusionFixture::getLogTagByIndex.";
            return false;
        }
    }
};

const char* const LogTagManagerNamePartNonConfusionFixture::strApple = "apple";
const char* const LogTagManagerNamePartNonConfusionFixture::strBanana = "banana";
const char* const LogTagManagerNamePartNonConfusionFixture::strCoconut = "coconut";
const char* const LogTagManagerNamePartNonConfusionFixture::strOrange = "orange";
const char* const LogTagManagerNamePartNonConfusionFixture::strPineapple = "pineapple";
const char* const LogTagManagerNamePartNonConfusionFixture::strBananaDotOrange = "banana.orange";
const char* const LogTagManagerNamePartNonConfusionFixture::strBananaDotPineapple = "banana.pineapple";
const char* const LogTagManagerNamePartNonConfusionFixture::strCoconutDotPineapple = "coconut.pineapple";


INSTANTIATE_TEST_CASE_P(
    LogTagManagerNamePartNonConfusionTest,
    LogTagManagerNamePartNonConfusionFixture,
    ::testing::Combine(
        ::testing::Values(0, 1, 2, 3, 4),
        ::testing::Values(ByWhat::ByFirstPart, ByWhat::ByAnyPart),
        ::testing::Values(0, 1, 2, 3, 4, 5, 6, 7),
        ::testing::Values(Timing::Before, Timing::After),
        ::testing::Values(Timing::Before, Timing::After, Timing::Never)
    )
);

TEST_P(LogTagManagerNamePartNonConfusionFixture, NamePartTestFunc)
{
    const auto myTestParam = MyTestParam(GetParam());
    LogTag* configTag = getLogTagByIndex(myTestParam.configTagIndex);
    LogTag* targetTag = getLogTagByIndex(myTestParam.targetTagIndex);
    ASSERT_NE(configTag, nullptr) << "Invalid parameterized test value, check value of myTestParam.configTagIndex.";
    ASSERT_NE(targetTag, nullptr) << "Invalid parameterized test value, check value of myTestParam.targetTagIndex.";
    if (myTestParam.whenToAssignConfigTag == Timing::Before)
    {
        m_logTagManager.assign(configTag->name, configTag);
    }
    if (myTestParam.whenToAssignTargetTag == Timing::Before)
    {
        m_logTagManager.assign(targetTag->name, targetTag);
    }
    switch (myTestParam.configMatchBy)
    {
    case ByWhat::ByFirstPart:
        m_logTagManager.setLevelByFirstPart(configTag->name, constTestLevelChanged);
        break;
    case ByWhat::ByAnyPart:
        m_logTagManager.setLevelByAnyPart(configTag->name, constTestLevelChanged);
        break;
    default:
        FAIL() << "Invalid parameterized test value, check test case. "
            << "Fixture LogTagManagerNamePartNonConfusionFixture, Case NamePartTestFunc.";
    }
    if (myTestParam.whenToAssignConfigTag == Timing::After)
    {
        m_logTagManager.assign(configTag->name, configTag);
    }
    if (myTestParam.whenToAssignTargetTag == Timing::After)
    {
        m_logTagManager.assign(targetTag->name, targetTag);
    }
    // Verifies the registration of the log tag pointer. If fail, cannot proceed
    // because it is not certain whether the returned pointer is valid to dereference
    ASSERT_EQ(m_logTagManager.get(targetTag->name), targetTag);
    // Verifies the log level of target tag
    const bool isChangeExpected = findTabulatedExpectedResult(myTestParam);
    if (isChangeExpected)
    {
        EXPECT_EQ(targetTag->level, constTestLevelChanged);
        EXPECT_NE(targetTag->level, constTestLevelBegin);
    }
    else
    {
        EXPECT_EQ(targetTag->level, constTestLevelBegin);
        EXPECT_NE(targetTag->level, constTestLevelChanged);
    }
}

}} // namespace
