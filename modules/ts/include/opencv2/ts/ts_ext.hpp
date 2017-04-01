// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Intel, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_TS_EXT_HPP
#define OPENCV_TS_EXT_HPP

namespace cvtest {
void checkIppStatus();
}

#define CV_TEST_INIT \
    cv::ipp::setIppStatus(0); \
    cv::theRNG().state = cvtest::param_seed;
#define CV_TEST_CLEANUP ::cvtest::checkIppStatus();
#define CV_TEST_BODY_IMPL \
    { \
       try { \
          CV_TEST_INIT \
          Body(); \
          CV_TEST_CLEANUP \
       } \
       catch (cvtest::SkipTestException& e) \
       { \
          printf("[     SKIP ] %s\n", e.what()); \
       } \
    } \


#undef TEST
#define TEST(test_case_name, test_name) \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public ::testing::Test {\
     public:\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
     private:\
      virtual void TestBody();\
      virtual void Body();\
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
            ::testing::Test::SetUpTestCase, \
            ::testing::Test::TearDownTestCase, \
            new ::testing::internal::TestFactoryImpl<\
                GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() CV_TEST_BODY_IMPL \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::Body()

#undef TEST_F
#define TEST_F(test_fixture, test_name)\
    class GTEST_TEST_CLASS_NAME_(test_fixture, test_name) : public test_fixture {\
     public:\
      GTEST_TEST_CLASS_NAME_(test_fixture, test_name)() {}\
     private:\
      virtual void TestBody();\
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
    void GTEST_TEST_CLASS_NAME_(test_fixture, test_name)::TestBody() CV_TEST_BODY_IMPL \
    void GTEST_TEST_CLASS_NAME_(test_fixture, test_name)::Body()

#undef TEST_P
#define TEST_P(test_case_name, test_name) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {} \
   private: \
    virtual void Body(); \
    virtual void TestBody(); \
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
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() CV_TEST_BODY_IMPL \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::Body()

#endif  // OPENCV_TS_EXT_HPP
