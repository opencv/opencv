#include "test_precomp.hpp"

// Run test with --gtest_catch_exceptions flag to avoid runtime errors in 
// the case when there is no GPU
CV_TEST_MAIN("gpu")

// TODO Add other tests from tests/gpu folder
// TODO When there is no GPU test system doesn't print error message: it fails or keeps 
//      quiet when --gtest_catch_exceptions is enabled
