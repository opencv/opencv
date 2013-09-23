/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifndef __OPENCV_CORE_INIT_HPP__
#define __OPENCV_CORE_INIT_HPP__


namespace cv {
namespace core {

CV_EXPORTS void OpenCV_Init();

CV_EXPORTS void OpenCV_Shutdown();

CV_EXPORTS void* OpenCV_RegisterShutdownCallback(void (*cleanupFn)(void* ctx), void* ctx);
CV_EXPORTS bool OpenCV_UnRegisterShutdownCallback(void* callbackHandle);

#if !defined(OPENCV_MODULE) && !defined(OPENCV_BYPASS_INIT)
namespace automatic_initialization {

inline void __cleanup()
{
    OpenCV_Shutdown();
}

inline int __init()
{
    OpenCV_Init();
    atexit(__cleanup);
    return 1;
}

template<int N>
class __Init
{
    static int result;
};

template<> int __Init<0>::result = __init();

}
#endif

}
}


#endif // __OPENCV_CORE_INIT_HPP__
