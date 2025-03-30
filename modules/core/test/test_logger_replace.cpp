// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <atomic>
#include <opencv2/core/utils/logger.hpp>

namespace opencv_test {
namespace {

using namespace cv::utils::logging;
using namespace cv::utils::logging::internal;

TEST(Core_Logger_Replace, WriteLogMessageRestoreCallWithNullOk)
{
    replaceWriteLogMessage(nullptr);
    writeLogMessage(cv::utils::logging::LOG_LEVEL_DEBUG, "msg");
    SUCCEED();
}

TEST(Core_Logger_Replace, WriteLogMessageExRestoreCallWithNullOk)
{
    replaceWriteLogMessageEx(nullptr);
    writeLogMessageEx(cv::utils::logging::LOG_LEVEL_DEBUG, "tag", "file", 1000, "func", "msg");
    SUCCEED();
}

std::atomic<uint32_t>& getCallFlagger()
{
    static std::atomic<uint32_t> callFlagger(0);
    return callFlagger;
}

std::atomic<uint32_t>& getCallCounter()
{
    static std::atomic<uint32_t> callCounter(0);
    return callCounter;
}

void myWriteLogMessage(LogLevel, const char*)
{
    getCallFlagger().fetch_or(1024u);
    getCallCounter().fetch_add(1u);
}

void myWriteLogMessageEx(LogLevel, const char*, const char*, int, const char*, const char*)
{
    getCallFlagger().fetch_or(2048u);
    getCallCounter().fetch_add(1u);
}

TEST(Core_Logger_Replace, WriteLogMessageReplaceRestore)
{
    uint32_t step_0 = getCallCounter().load();
    writeLogMessage(cv::utils::logging::LOG_LEVEL_DEBUG, "msg");
    uint32_t step_1 = getCallCounter().load();
    EXPECT_EQ(step_0, step_1);
    replaceWriteLogMessage(nullptr);
    uint32_t step_2 = getCallCounter().load();
    EXPECT_EQ(step_1, step_2);
    writeLogMessage(cv::utils::logging::LOG_LEVEL_DEBUG, "msg");
    uint32_t step_3 = getCallCounter().load();
    EXPECT_EQ(step_2, step_3);
    replaceWriteLogMessage(myWriteLogMessage);
    uint32_t step_4 = getCallCounter().load();
    EXPECT_EQ(step_3, step_4);
    writeLogMessage(cv::utils::logging::LOG_LEVEL_DEBUG, "msg");
    uint32_t step_5 = getCallCounter().load();
    EXPECT_EQ(step_4 + 1, step_5);
    writeLogMessage(cv::utils::logging::LOG_LEVEL_DEBUG, "msg");
    uint32_t step_6 = getCallCounter().load();
    EXPECT_EQ(step_5 + 1, step_6);
    replaceWriteLogMessage(nullptr);
    uint32_t step_7 = getCallCounter().load();
    EXPECT_EQ(step_6, step_7);
    writeLogMessage(cv::utils::logging::LOG_LEVEL_DEBUG, "msg");
    uint32_t step_8 = getCallCounter().load();
    EXPECT_EQ(step_7, step_8);
    uint32_t flags = getCallFlagger().load();
    EXPECT_NE(flags & 1024u, 0u);
}

TEST(Core_Logger_Replace, WriteLogMessageExReplaceRestore)
{
    uint32_t step_0 = getCallCounter().load();
    writeLogMessageEx(cv::utils::logging::LOG_LEVEL_DEBUG, "tag", "file", 0, "func", "msg");
    uint32_t step_1 = getCallCounter().load();
    EXPECT_EQ(step_0, step_1);
    replaceWriteLogMessageEx(nullptr);
    uint32_t step_2 = getCallCounter().load();
    EXPECT_EQ(step_1, step_2);
    writeLogMessageEx(cv::utils::logging::LOG_LEVEL_DEBUG, "tag", "file", 0, "func", "msg");
    uint32_t step_3 = getCallCounter().load();
    EXPECT_EQ(step_2, step_3);
    replaceWriteLogMessageEx(myWriteLogMessageEx);
    uint32_t step_4 = getCallCounter().load();
    EXPECT_EQ(step_3, step_4);
    writeLogMessageEx(cv::utils::logging::LOG_LEVEL_DEBUG, "tag", "file", 0, "func", "msg");
    uint32_t step_5 = getCallCounter().load();
    EXPECT_EQ(step_4 + 1, step_5);
    writeLogMessageEx(cv::utils::logging::LOG_LEVEL_DEBUG, "tag", "file", 0, "func", "msg");
    uint32_t step_6 = getCallCounter().load();
    EXPECT_EQ(step_5 + 1, step_6);
    replaceWriteLogMessageEx(nullptr);
    uint32_t step_7 = getCallCounter().load();
    EXPECT_EQ(step_6, step_7);
    writeLogMessageEx(cv::utils::logging::LOG_LEVEL_DEBUG, "tag", "file", 0, "func", "msg");
    uint32_t step_8 = getCallCounter().load();
    EXPECT_EQ(step_7, step_8);
    uint32_t flags = getCallFlagger().load();
    EXPECT_NE(flags & 2048u, 0u);
}

}} // namespace