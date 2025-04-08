// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Intel, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_TS_EXT_HPP
#define OPENCV_TS_EXT_HPP

namespace cvtest {
void checkIppStatus();
extern bool skipUnstableTests;
extern bool runBigDataTests;
extern int testThreads;
extern int debugLevel;  ///< 0 - no debug, 1 - basic test debug information, >1 - extra debug information

void testSetUp();
void testTearDown();

bool checkBigDataTests();

}

// check for required "opencv_test" namespace
#if !defined(CV_TEST_SKIP_NAMESPACE_CHECK) && defined(__OPENCV_BUILD)
#define CV__TEST_NAMESPACE_CHECK required_opencv_test_namespace = true;
#else
#define CV__TEST_NAMESPACE_CHECK  // nothing
#endif

#define CV__TEST_INIT \
    CV__TEST_NAMESPACE_CHECK \
    if (setUpSkipped) \
        return; \
    ::cvtest::testSetUp();
#define CV__TEST_CLEANUP ::cvtest::testTearDown();
#define CV__TEST_BODY_IMPL(name) \
    { \
       CV__TRACE_APP_FUNCTION_NAME(name); \
       try { \
          CV__TEST_INIT \
          Body(); \
          CV__TEST_CLEANUP \
       } \
       catch (const cvtest::details::SkipTestExceptionBase& e) \
       { \
          printf("[     SKIP ] %s\n", e.what()); \
       } \
    } \

#define CV__TEST_SETUP_IMPL(parent_class) { \
  setUpSkipped = false; \
  try { \
    parent_class::SetUp(); \
  } catch (const cvtest::details::SkipTestExceptionBase& e) { \
    setUpSkipped = true; \
    printf("[     SKIP ] %s\n", e.what()); \
  } \
} \

struct SkipThisTest : public ::testing::Test {
  SkipThisTest(const std::string& msg_) : msg(msg_) {}

  virtual void TestBody() CV_OVERRIDE {
      printf("[     SKIP ] %s\n", msg.c_str());
  }

  std::string msg;
};

#undef TEST
#define TEST_(test_case_name, test_name, parent_class, bodyMethodName, BODY_ATTR, BODY_IMPL) \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
     public:\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
     private:\
      bool setUpSkipped = false; \
      virtual void TestBody() CV_OVERRIDE;\
      virtual void bodyMethodName() BODY_ATTR;\
      virtual void SetUp() CV_OVERRIDE; \
      static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
      GTEST_DISALLOW_COPY_AND_ASSIGN_(\
          GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
    };\
    class test_case_name##test_name##_factory : public ::testing::internal::TestFactoryBase { \
     public:\
      virtual ::testing::Test* CreateTest() CV_OVERRIDE { \
        try { \
          return new GTEST_TEST_CLASS_NAME_(test_case_name, test_name); \
        } catch (const cvtest::details::SkipTestExceptionBase& e) { \
          return new SkipThisTest(e.what()); \
        } \
      } \
    };\
    \
    ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
      ::test_info_ =\
        ::testing::internal::MakeAndRegisterTestInfo(\
            #test_case_name, #test_name, NULL, NULL, \
            ::testing::internal::CodeLocation(__FILE__, __LINE__), \
            (::testing::internal::GetTestTypeId()), \
            parent_class::SetUpTestCase, \
            parent_class::TearDownTestCase, \
            new test_case_name##test_name##_factory);\
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() BODY_IMPL( #test_case_name "_" #test_name ) \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::SetUp() CV__TEST_SETUP_IMPL(parent_class) \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::bodyMethodName()

#define TEST(test_case_name, test_name) TEST_(test_case_name, test_name, ::testing::Test, Body,, CV__TEST_BODY_IMPL)

#define CV__TEST_BIGDATA_BODY_IMPL(name) \
    { \
       if (!cvtest::checkBigDataTests()) \
       { \
           return; \
       } \
       CV__TRACE_APP_FUNCTION_NAME(name); \
       try { \
          CV__TEST_INIT \
          Body(); \
          CV__TEST_CLEANUP \
       } \
       catch (const cvtest::details::SkipTestExceptionBase& e) \
       { \
          printf("[     SKIP ] %s\n", e.what()); \
       } \
    } \

// Special type of tests which require / use or validate processing of huge amount of data (>= 2Gb)
#if defined(_M_X64) || defined(_M_ARM64) || defined(__x86_64__) || defined(__aarch64__)
#define BIGDATA_TEST(test_case_name, test_name) TEST_(BigData_ ## test_case_name, test_name, ::testing::Test, Body,, CV__TEST_BIGDATA_BODY_IMPL)
#else
#define BIGDATA_TEST(test_case_name, test_name) TEST_(BigData_ ## test_case_name, DISABLED_ ## test_name, ::testing::Test, Body,, CV__TEST_BIGDATA_BODY_IMPL)
#endif

#undef TEST_F
#define TEST_F(test_fixture, test_name)\
    class GTEST_TEST_CLASS_NAME_(test_fixture, test_name) : public test_fixture {\
     public:\
      GTEST_TEST_CLASS_NAME_(test_fixture, test_name)() {}\
     private:\
      bool setUpSkipped = false; \
      virtual void TestBody() CV_OVERRIDE;\
      virtual void Body(); \
      virtual void SetUp() CV_OVERRIDE; \
      static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
      GTEST_DISALLOW_COPY_AND_ASSIGN_(\
          GTEST_TEST_CLASS_NAME_(test_fixture, test_name));\
    };\
    class test_fixture##test_name##_factory : public ::testing::internal::TestFactoryBase { \
     public:\
      virtual ::testing::Test* CreateTest() CV_OVERRIDE { \
        try { \
          return new GTEST_TEST_CLASS_NAME_(test_fixture, test_name); \
        } catch (const cvtest::details::SkipTestExceptionBase& e) { \
          return new SkipThisTest(e.what()); \
        } \
      } \
    };\
    \
    ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_fixture, test_name)\
      ::test_info_ =\
        ::testing::internal::MakeAndRegisterTestInfo(\
            #test_fixture, #test_name, NULL, NULL, \
            ::testing::internal::CodeLocation(__FILE__, __LINE__), \
            (::testing::internal::GetTypeId<test_fixture>()), \
            test_fixture::SetUpTestCase, \
            test_fixture::TearDownTestCase, \
            new test_fixture##test_name##_factory);\
    void GTEST_TEST_CLASS_NAME_(test_fixture, test_name)::TestBody() CV__TEST_BODY_IMPL( #test_fixture "_" #test_name ) \
    void GTEST_TEST_CLASS_NAME_(test_fixture, test_name)::SetUp() CV__TEST_SETUP_IMPL(test_fixture) \
    void GTEST_TEST_CLASS_NAME_(test_fixture, test_name)::Body()

// Don't use directly
#define CV__TEST_P(test_case_name, test_name, bodyMethodName, BODY_ATTR, BODY_IMPL/*(name_str)*/) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {} \
   private: \
    bool setUpSkipped = false; \
    virtual void bodyMethodName() BODY_ATTR; \
    virtual void TestBody() CV_OVERRIDE; \
    virtual void SetUp() CV_OVERRIDE; \
    static int AddToRegistry() { \
      ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
          GetTestCasePatternHolder<test_case_name>(\
              #test_case_name, \
              ::testing::internal::CodeLocation(\
                  __FILE__, __LINE__))->AddTestPattern(\
                      #test_case_name, \
                      #test_name, \
                      new ::testing::internal::TestMetaFactory< \
                          GTEST_TEST_CLASS_NAME_(\
                              test_case_name, test_name)>()); \
      return 0; \
    } \
    static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_; \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(\
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
  }; \
  int GTEST_TEST_CLASS_NAME_(test_case_name, \
                             test_name)::gtest_registering_dummy_ = \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() BODY_IMPL( #test_case_name "_" #test_name ) \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::SetUp() CV__TEST_SETUP_IMPL(test_case_name) \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::bodyMethodName()

#undef TEST_P
#define TEST_P(test_case_name, test_name) CV__TEST_P(test_case_name, test_name, Body,, CV__TEST_BODY_IMPL)


#define CV_TEST_EXPECT_EXCEPTION_MESSAGE(statement, msg) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (::testing::internal::AlwaysTrue()) { \
    const char* msg_ = msg; \
    bool hasException = false; \
    try { \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
    } \
    catch (const cv::Exception& e) { \
      if (NULL == strstr(e.what(), msg_)) \
        ADD_FAILURE() << "Unexpected cv::Exception is raised: " << #statement << "\n  Expected message substring: '" << msg_ << "'. Actual message:\n" << e.what(); \
      hasException = true; \
    } \
    catch (const std::exception& e) { \
      ADD_FAILURE() << "Unexpected std::exception is raised: " << #statement << "\n" << e.what(); \
      hasException = true; \
    } \
    catch (...) { \
      ADD_FAILURE() << "Unexpected C++ exception is raised: " << #statement; \
      hasException = true; \
    } \
    if (!hasException) { \
      goto GTEST_CONCAT_TOKEN_(gtest_label_test_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_test_, __LINE__): \
      ADD_FAILURE() << "Failed: Expected: " #statement " throws an '" << msg << "' exception.\n" \
           "  Actual: it doesn't."


#endif  // OPENCV_TS_EXT_HPP
