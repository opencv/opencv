#define GTEST_CREATE_AS_SHARED_LIBRARY 1

#if _MSC_VER >= 1200
#pragma warning( disable: 4127 4251)
#endif

#include "opencv2/ts/ts.hpp"
#include "opencv2/core/core_c.h"

#if ANDROID
int wcscasecmp(const wchar_t* lhs, const wchar_t* rhs);
#endif