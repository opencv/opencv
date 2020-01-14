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


#undef TEST
#define TEST_(test_case_name, test_name, parent_class, bodyMethodName, BODY_IMPL) \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
     public:\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
     private:\
      virtual void TestBody() CV_OVERRIDE;\
      virtual void bodyMethodName();\
      static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
      GTEST_DISALLOW_COPY_AND_ASSIGN_(\
          GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
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
            new ::testing::internal::TestFactoryImpl<\
                GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() BODY_IMPL( #test_case_name "_" #test_name ) \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::bodyMethodName()

#define TEST(test_case_name, test_name) TEST_(test_case_name, test_name, ::testing::Test, Body, CV__TEST_BODY_IMPL)

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
#define BIGDATA_TEST(test_case_name, test_name) TEST_(BigData_ ## test_case_name, test_name, ::testing::Test, Body, CV__TEST_BIGDATA_BODY_IMPL)
#else
#define BIGDATA_TEST(test_case_name, test_name) TEST_(BigData_ ## test_case_name, DISABLED_ ## test_name, ::testing::Test, Body, CV__TEST_BIGDATA_BODY_IMPL)
#endif

#undef TEST_F
#define TEST_F(test_fixture, test_name)\
    class GTEST_TEST_CLASS_NAME_(test_fixture, test_name) : public test_fixture {\
     public:\
      GTEST_TEST_CLASS_NAME_(test_fixture, test_name)() {}\
     private:\
      virtual void TestBody() CV_OVERRIDE;\
      virtual void Body(); \
      static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
      GTEST_DISALLOW_COPY_AND_ASSIGN_(\
          GTEST_TEST_CLASS_NAME_(test_fixture, test_name));\
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
            new ::testing::internal::TestFactoryImpl<\
                GTEST_TEST_CLASS_NAME_(test_fixture, test_name)>);\
    void GTEST_TEST_CLASS_NAME_(test_fixture, test_name)::TestBody() CV__TEST_BODY_IMPL( #test_fixture "_" #test_name ) \
    void GTEST_TEST_CLASS_NAME_(test_fixture, test_name)::Body()

// Don't use directly
#define CV__TEST_P(test_case_name, test_name, bodyMethodName, BODY_IMPL/*(name_str)*/) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {} \
   private: \
    virtual void bodyMethodName(); \
    virtual void TestBody() CV_OVERRIDE; \
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
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::bodyMethodName()

#undef TEST_P
#define TEST_P(test_case_name, test_name) CV__TEST_P(test_case_name, test_name, Body, CV__TEST_BODY_IMPL)

#endif  // OPENCV_TS_EXT_HPP
