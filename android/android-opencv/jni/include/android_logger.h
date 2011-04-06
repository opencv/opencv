#pragma once
#include <iostream>
#include <sstream>

#define  LOG_TAG    "libopencv"
#if ANDROID
#include <android/log.h>
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#else
#include <cstdio>
#define  LOGI(...)  printf("info:");printf("%s:",LOG_TAG); fprintf(stdout,__VA_ARGS__);printf("\n");
#define  LOGE(...)  printf("error:");printf("%s:",LOG_TAG); fprintf(stderr,__VA_ARGS__);printf("\n");
#endif

#ifndef LOGI_STREAM
#define  LOGI_STREAM(x)  {std::stringstream ss; ss << x; LOGI("%s",ss.str().c_str());}
#endif
#define  LOGE_STREAM(x)  {std::stringstream ss; ss << x; LOGE("%s",ss.str().c_str());}
