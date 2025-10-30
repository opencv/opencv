#include "opencv2/ts.hpp"
#include <opencv2/core/utils/logger.hpp>
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utility.hpp"
#if !defined(__EMSCRIPTEN__)
#include "opencv2/core/private.hpp"
#endif

#ifdef GTEST_LINKED_AS_SHARED_LIBRARY
#error ts module should not have GTEST_LINKED_AS_SHARED_LIBRARY defined
#endif
