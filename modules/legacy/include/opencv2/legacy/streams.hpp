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

#ifndef __OPENCV_CVSTREAMS_H__
#define __OPENCV_CVSTREAMS_H__

#ifdef WIN32
#include <streams.h>  /* !!! IF YOU'VE GOT AN ERROR HERE, PLEASE READ BELOW !!! */
/***************** How to get Visual Studio understand streams.h ****************\

You need DirectShow SDK that is now a part of Platform SDK
(Windows Server 2003 SP1 SDK or later),
and DirectX SDK (2006 April or later).

1. Download the Platform SDK from
   http://www.microsoft.com/msdownload/platformsdk/sdkupdate/
   and DirectX SDK from msdn.microsoft.com/directx/
   (They are huge, but you can download it by parts).
   If it doesn't work for you, consider HighGUI that can capture video via VFW or MIL

2. Install Platform SDK together with DirectShow SDK.
   Install DirectX (with or without sample code).

3. Build baseclasses.
   See <PlatformSDKInstallFolder>\samples\multimedia\directshow\readme.txt.

4. Copy the built libraries (called strmbase.lib and strmbasd.lib
   in Release and Debug versions, respectively) to
   <PlatformSDKInstallFolder>\lib.

5. In Developer Studio add the following paths:
      <DirectXSDKInstallFolder>\include
      <PlatformSDKInstallFolder>\include
      <PlatformSDKInstallFolder>\samples\multimedia\directshow\baseclasses
    to the includes' search path
    (at Tools->Options->Directories->Include files in case of Visual Studio 6.0,
     at Tools->Options->Projects and Solutions->VC++ Directories->Include files in case
     of Visual Studio 2005)
   Add
      <DirectXSDKInstallFolder>\lib
      <PlatformSDKInstallFolder>\lib
   to the libraries' search path (in the same dialog, ...->"Library files" page)

   NOTE: PUT THE ADDED LINES ON THE VERY TOP OF THE LISTS, OTHERWISE YOU MAY STILL GET
   COMPILER OR LINKER ERRORS. This is necessary, because Visual Studio
   may include older versions of the same headers and libraries.

6. Now you can build OpenCV DirectShow filters.

\***********************************************************************************/

#endif

#endif

