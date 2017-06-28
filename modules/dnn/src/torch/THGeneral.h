#ifndef TH_GENERAL_INC
#define TH_GENERAL_INC

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define TH_API

#define THError(...) CV_Error(cv::Error::StsError, cv::format(__VA_ARGS__))
#define THArgCheck(cond, ...) CV_Assert(cond)

#define THAlloc malloc
#define THRealloc realloc
#define THFree free

#endif
