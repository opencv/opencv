/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __HIGHGUI_H_
#define __HIGHGUI_H_

#include "opencv2/highgui.hpp"

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgcodecs/imgcodecs_c.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <assert.h>

#if defined WIN32 || defined WINCE
    #include <windows.h>
    #undef small
    #undef min
    #undef max
    #undef abs
#endif

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/highgui/highgui_tegra.hpp"
#endif

/* Errors */
#define HG_OK          0 /* Don't bet on it! */
#define HG_BADNAME    -1 /* Bad window or file name */
#define HG_INITFAILED -2 /* Can't initialize HigHGUI */
#define HG_WCFAILED   -3 /* Can't create a window */
#define HG_NULLPTR    -4 /* The null pointer where it should not appear */
#define HG_BADPARAM   -5

#define __BEGIN__ __CV_BEGIN__
#define __END__  __CV_END__
#define EXIT __CV_EXIT__

#define CV_WINDOW_MAGIC_VAL     0x00420042
#define CV_TRACKBAR_MAGIC_VAL   0x00420043

//Yannick Verdie 2010, Max Kostin 2015
void cvSetModeWindow_W32(const char* name, double prop_value);
void cvSetModeWindow_GTK(const char* name, double prop_value);
void cvSetModeWindow_CARBON(const char* name, double prop_value);
void cvSetModeWindow_COCOA(const char* name, double prop_value);
void cvSetModeWindow_WinRT(const char* name, double prop_value);

double cvGetModeWindow_W32(const char* name);
double cvGetModeWindow_GTK(const char* name);
double cvGetModeWindow_CARBON(const char* name);
double cvGetModeWindow_COCOA(const char* name);
double cvGetModeWindow_WinRT(const char* name);

double cvGetPropWindowAutoSize_W32(const char* name);
double cvGetPropWindowAutoSize_GTK(const char* name);

double cvGetRatioWindow_W32(const char* name);
double cvGetRatioWindow_GTK(const char* name);

double cvGetOpenGlProp_W32(const char* name);
double cvGetOpenGlProp_GTK(const char* name);

//for QT
#if defined (HAVE_QT)
double cvGetModeWindow_QT(const char* name);
void cvSetModeWindow_QT(const char* name, double prop_value);

double cvGetPropWindow_QT(const char* name);
void cvSetPropWindow_QT(const char* name,double prop_value);

double cvGetRatioWindow_QT(const char* name);
void cvSetRatioWindow_QT(const char* name,double prop_value);

double cvGetOpenGlProp_QT(const char* name);
double cvGetPropVisible_QT(const char* name);
#endif

#endif /* __HIGHGUI_H_ */
