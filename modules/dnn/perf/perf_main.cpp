#include "perf_precomp.hpp"

static const std::string extraTestDataPath = cvtest::safe_getenv("OPENCV_DNN_TEST_DATA_PATH");
static const std::string modelsPath = cvtest::safe_getenv("OPENCV_DNN_MODELS_PATH");

CV_PERF_TEST_MAIN(dnn,
    (void)cvtest::addDataSearchPath(extraTestDataPath),
    (void)cvtest::addDataSearchPath(modelsPath))
