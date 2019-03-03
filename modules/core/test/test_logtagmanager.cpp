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

}} // namespace
