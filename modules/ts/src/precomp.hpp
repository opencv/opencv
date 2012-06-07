#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4127 4251)
#endif

#include "opencv2/core/core_c.h"
#include "opencv2/ts/ts.hpp"

#ifdef GTEST_LINKED_AS_SHARED_LIBRARY
#error ts module should not have GTEST_LINKED_AS_SHARED_LIBRARY defined
#endif
