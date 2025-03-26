// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/core/utils/logger.hpp>

#include "../src/utils/logcallbackmanager.hpp"

// Because "LogCallbackManager" isn't exported from "opencv_core", the only way
// to perform test is to compile the source code into "opencv_test_core".
// This workaround may cause step debugger breakpoints to work unreliably.
#if 1
#include "../src/utils/logcallbackmanager.cpp"
#endif

#include <atomic>

namespace opencv_test {
namespace {

using LogLevel = cv::utils::logging::LogLevel;
using LogCallbackManager = cv::utils::logging::LogCallbackManager;
using LoggingCallbackHandler = cv::utils::logging::LoggingCallbackHandler;

// A base fixture consisting of the LogCallbackManager.
class LogCallbackManagerTestFixture
    : public ::testing::Test
{
public:
    LogCallbackManagerTestFixture()
        : m_manager(cv::makePtr<LogCallbackManager>())
    {
    }

    ~LogCallbackManagerTestFixture()
    {
    }

protected:
    cv::Ptr<LogCallbackManager> m_manager;
};

class LoggingCallbackHandlerTestSpy
    : public LoggingCallbackHandler
{
public:
    struct Args
    {
        size_t spy_captured_idx;
        LogLevel logLevel;
        const char* tag;
        const char* file;
        int line;
        const char* func;
        const char* message;
    };

    void operator()(LogLevel logLevel, const char* tag,
        const char* file, int line, const char* func,
        const char* message) final
    {
        auto& this_captures = this->m_captured_args;
        size_t spy_captured_idx = this_captures.size();
        this_captures.emplace_back(Args{
            spy_captured_idx,
            logLevel,
            tag,
            file,
            line,
            func,
            message
        });
    }

    size_t capturedCount() const
    {
        return m_captured_args.size();
    }

    const Args& getCaptured(size_t idx) const
    {
        return m_captured_args.at(idx);
    }

private:
    std::vector<Args> m_captured_args;
};

class LogCallbackManagerSmokeTest
    : public LogCallbackManagerTestFixture
{
protected:
    LogCallbackManagerSmokeTest()
        : LogCallbackManagerTestFixture()
        , m_spy(cv::makePtr<LoggingCallbackHandlerTestSpy>())
        , m_spyTwo(cv::makePtr<LoggingCallbackHandlerTestSpy>())
    {
    }

    cv::Ptr<LoggingCallbackHandlerTestSpy> m_spy;
    cv::Ptr<LoggingCallbackHandlerTestSpy> m_spyTwo;
};

TEST_F(LogCallbackManagerSmokeTest, AtStartCountIsZero)
{
    EXPECT_EQ(m_manager->count(), 0u);
}

TEST_F(LogCallbackManagerSmokeTest, AtStartContainsIsFalse)
{
    EXPECT_FALSE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, AtStartRemoveSpyIsOkay)
{
    m_manager->remove(m_spy);
    EXPECT_FALSE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, AtStartRemoveSpyCountIsZero)
{
    m_manager->remove(m_spy);
    EXPECT_EQ(m_manager->count(), 0u);
}

TEST_F(LogCallbackManagerSmokeTest, AtStartRemoveAllIsOkay)
{
    m_manager->removeAll();
    EXPECT_FALSE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, AtStartRemoveAllCountIsZero)
{
    m_manager->removeAll();
    EXPECT_EQ(m_manager->count(), 0u);
}

TEST_F(LogCallbackManagerSmokeTest, AfterAddSpyCountIsOne)
{
    m_manager->add(m_spy);
    EXPECT_EQ(m_manager->count(), 1u);
}

TEST_F(LogCallbackManagerSmokeTest, AfterAddSpyContainsIsTrue)
{
    m_manager->add(m_spy);
    EXPECT_TRUE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, AfterAddSpyTwiceCountIsOne)
{
    m_manager->add(m_spy);
    m_manager->add(m_spy);
    EXPECT_EQ(m_manager->count(), 1u);
}

TEST_F(LogCallbackManagerSmokeTest, AfterAddSpyTwiceContainsIsTrue)
{
    m_manager->add(m_spy);
    m_manager->add(m_spy);
    EXPECT_TRUE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, RemoveAnyOneRemovesSame)
{
    m_manager->add(m_spy);
    m_manager->add(m_spy);
    m_manager->remove(m_spy);
    EXPECT_FALSE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, RepeatRemoveIsOkay)
{
    m_manager->add(m_spy);
    EXPECT_EQ(m_manager->count(), 1u);
    EXPECT_TRUE(m_manager->contains(m_spy));
    m_manager->add(m_spy);
    EXPECT_EQ(m_manager->count(), 1u);
    EXPECT_TRUE(m_manager->contains(m_spy));
    m_manager->remove(m_spy);
    EXPECT_EQ(m_manager->count(), 0u);
    EXPECT_FALSE(m_manager->contains(m_spy));
    m_manager->remove(m_spy);
    EXPECT_EQ(m_manager->count(), 0u);
    EXPECT_FALSE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, RemoveThenAddAgainIsOkay)
{
    m_manager->add(m_spy);
    EXPECT_EQ(m_manager->count(), 1u);
    EXPECT_TRUE(m_manager->contains(m_spy));
    m_manager->remove(m_spy);
    EXPECT_EQ(m_manager->count(), 0u);
    EXPECT_FALSE(m_manager->contains(m_spy));
    m_manager->add(m_spy);
    EXPECT_EQ(m_manager->count(), 1u);
    EXPECT_TRUE(m_manager->contains(m_spy));
    m_manager->remove(m_spy);
    EXPECT_EQ(m_manager->count(), 0u);
    EXPECT_FALSE(m_manager->contains(m_spy));
}

TEST_F(LogCallbackManagerSmokeTest, OnlyContainsSpecificOne)
{
    m_manager->add(m_spy);
    EXPECT_FALSE(m_manager->contains(m_spyTwo));
}

TEST_F(LogCallbackManagerSmokeTest, OnlySpecificOneRemoved)
{
    m_manager->add(m_spy);
    m_manager->add(m_spyTwo);
    EXPECT_EQ(m_manager->count(), 2u);
    EXPECT_TRUE(m_manager->contains(m_spy));
    EXPECT_TRUE(m_manager->contains(m_spyTwo));
    m_manager->remove(m_spy);
    EXPECT_EQ(m_manager->count(), 1u);
    EXPECT_FALSE(m_manager->contains(m_spy));
    EXPECT_TRUE(m_manager->contains(m_spyTwo));
}

TEST_F(LogCallbackManagerSmokeTest, RemoveAllRemovesAll)
{
    m_manager->add(m_spy);
    m_manager->add(m_spyTwo);
    m_manager->removeAll();
    EXPECT_EQ(m_manager->count(), 0u);
    EXPECT_FALSE(m_manager->contains(m_spy));
    EXPECT_FALSE(m_manager->contains(m_spyTwo));
}

TEST_F(LogCallbackManagerSmokeTest, ReadInto)
{
    m_manager->add(m_spy);
    m_manager->add(m_spyTwo);
    std::vector<cv::Ptr<LoggingCallbackHandler>> handlers;
    m_manager->readInto(handlers);
    EXPECT_EQ(handlers.size(), 2u);
}

TEST_F(LogCallbackManagerSmokeTest, RemoveOneThenReadInto)
{
    m_manager->add(m_spy);
    m_manager->add(m_spyTwo);
    m_manager->remove(m_spy);
    std::vector<cv::Ptr<LoggingCallbackHandler>> handlers;
    m_manager->readInto(handlers);
    EXPECT_EQ(handlers.size(), 1u);
    EXPECT_EQ(handlers.at(0), m_spyTwo);
}

TEST_F(LogCallbackManagerSmokeTest, RemoveAllThenReadInto)
{
    m_manager->add(m_spy);
    m_manager->add(m_spyTwo);
    m_manager->removeAll();
    std::vector<cv::Ptr<LoggingCallbackHandler>> handlers;
    m_manager->readInto(handlers);
    EXPECT_EQ(handlers.size(), 0u);
}

namespace log_callback_test_statics
{
    static std::atomic<int>& getAtomicIntRef()
    {
        static std::atomic<int> stc_atomic{};
        return stc_atomic;
    }

    static int getAtomicInt()
    {
        return getAtomicIntRef().load();
    }

    static void incrementAtomicInt()
    {
        getAtomicIntRef().fetch_add(1);
    }

    static void log_callback_test_statics(LogLevel, const char*, const char*, int, const char*, const char*)
    {
        incrementAtomicInt();
    }

}//log_callback_test_statics

TEST_F(LogCallbackManagerSmokeTest, StaticFuncWrapperTest)
{
    auto handler = LoggingCallbackHandler::createFromStaticFuncPtr(log_callback_test_statics::log_callback_test_statics);
    m_manager->add(handler);
    EXPECT_EQ(m_manager->count(), 1u);
    EXPECT_TRUE(m_manager->contains(handler));
    m_manager->remove(handler);
    EXPECT_EQ(m_manager->count(), 0u);
    EXPECT_FALSE(m_manager->contains(handler));
}

TEST_F(LogCallbackManagerSmokeTest, StdFunctionWrapperTest)
{
    std::function<void(LogLevel, const char*, const char*, int,
        const char*, const char*)> func = [](LogLevel, const char*,
            const char*, int, const char*, const char*) {};
    auto handler = LoggingCallbackHandler::createFromFunction(func);
    m_manager->add(handler);
    EXPECT_EQ(m_manager->count(), 1u);
    EXPECT_TRUE(m_manager->contains(handler));
    m_manager->remove(handler);
    EXPECT_EQ(m_manager->count(), 0u);
    EXPECT_FALSE(m_manager->contains(handler));
}

TEST_F(LogCallbackManagerSmokeTest, StaticFuncWrapperCallThru)
{
    int pre_count = log_callback_test_statics::getAtomicInt();
    auto handler = LoggingCallbackHandler::createFromStaticFuncPtr(log_callback_test_statics::log_callback_test_statics);
    handler->operator()(LogLevel::LOG_LEVEL_VERBOSE, "tag", "file", 42, "func", "message");
    int post_count = log_callback_test_statics::getAtomicInt();
    EXPECT_TRUE(post_count >= pre_count + 1)
        << "Expects logging callback wrapper call to increment count. "
        << "post_count=" << post_count << " pre_count=" << pre_count;
}

TEST_F(LogCallbackManagerSmokeTest, StdFunctionWrapperCallThru)
{
    bool called = false;
    std::function<void(LogLevel, const char*, const char*, int, const char*,
        const char*)> func = [&called](LogLevel, const char*, const char*, int, const char*, const char*) {
        called = true;
    };
    auto handler = LoggingCallbackHandler::createFromFunction(func);
    handler->operator()(LogLevel::LOG_LEVEL_VERBOSE, "tag", "file", 42, "func", "message");
    EXPECT_TRUE(called);
}

}} // namespace
