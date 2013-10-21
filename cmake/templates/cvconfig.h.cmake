/* OpenCV compiled as static or dynamic libs */
#cmakedefine BUILD_SHARED_LIBS

/* Compile for 'real' NVIDIA GPU architectures */
#define CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN}"

/* Create PTX or BIN for 1.0 compute capability */
#cmakedefine CUDA_ARCH_BIN_OR_PTX_10

/* NVIDIA GPU features are used */
#define CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES}"

/* Compile for 'virtual' NVIDIA PTX architectures */
#define CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX}"

/* AVFoundation video libraries */
#cmakedefine HAVE_AVFOUNDATION

/* V4L capturing support */
#cmakedefine HAVE_CAMV4L

/* V4L2 capturing support */
#cmakedefine HAVE_CAMV4L2

/* Carbon windowing environment */
#cmakedefine HAVE_CARBON

/* AMD's Basic Linear Algebra Subprograms Library*/
#cmakedefine HAVE_CLAMDBLAS

/* AMD's OpenCL Fast Fourier Transform Library*/
#cmakedefine HAVE_CLAMDFFT

/* Clp support */
#cmakedefine HAVE_CLP

/* Cocoa API */
#cmakedefine HAVE_COCOA

/* C= */
#cmakedefine HAVE_CSTRIPES

/* NVidia Cuda Basic Linear Algebra Subprograms (BLAS) API*/
#cmakedefine HAVE_CUBLAS

/* NVidia Cuda Runtime API*/
#cmakedefine HAVE_CUDA

/* NVidia Cuda Fast Fourier Transform (FFT) API*/
#cmakedefine HAVE_CUFFT

/* IEEE1394 capturing support */
#cmakedefine HAVE_DC1394

/* IEEE1394 capturing support - libdc1394 v2.x */
#cmakedefine HAVE_DC1394_2

/* DirectShow Video Capture library */
#cmakedefine HAVE_DSHOW

/* Eigen Matrix & Linear Algebra Library */
#cmakedefine HAVE_EIGEN

/* FFMpeg video library */
#cmakedefine HAVE_FFMPEG

/* ffmpeg's libswscale */
#cmakedefine HAVE_FFMPEG_SWSCALE

/* ffmpeg in Gentoo */
#cmakedefine HAVE_GENTOO_FFMPEG

/* GStreamer multimedia framework */
#cmakedefine HAVE_GSTREAMER

/* GTK+ 2.0 Thread support */
#cmakedefine HAVE_GTHREAD

/* GTK+ 2.x toolkit */
#cmakedefine HAVE_GTK

/* Define to 1 if you have the <inttypes.h> header file. */
#cmakedefine HAVE_INTTYPES_H 1

/* Intel Integrated Performance Primitives */
#cmakedefine HAVE_IPP

/* JPEG-2000 codec */
#cmakedefine HAVE_JASPER

/* IJG JPEG codec */
#cmakedefine HAVE_JPEG

/* libpng/png.h needs to be included */
#cmakedefine HAVE_LIBPNG_PNG_H

/* V4L/V4L2 capturing support via libv4l */
#cmakedefine HAVE_LIBV4L

/* Microsoft Media Foundation Capture library */
#cmakedefine HAVE_MSMF

/* NVidia Video Decoding API*/
#cmakedefine HAVE_NVCUVID

/* OpenCL Support */
#cmakedefine HAVE_OPENCL
#cmakedefine HAVE_OPENCL11
#cmakedefine HAVE_OPENCL12

/* OpenEXR codec */
#cmakedefine HAVE_OPENEXR

/* OpenGL support*/
#cmakedefine HAVE_OPENGL

/* OpenNI library */
#cmakedefine HAVE_OPENNI

/* PNG codec */
#cmakedefine HAVE_PNG

/* Qt support */
#cmakedefine HAVE_QT

/* Qt OpenGL support */
#cmakedefine HAVE_QT_OPENGL

/* QuickTime video libraries */
#cmakedefine HAVE_QUICKTIME

/* QTKit video libraries */
#cmakedefine HAVE_QTKIT

/* Intel Threading Building Blocks */
#cmakedefine HAVE_TBB

/* TIFF codec */
#cmakedefine HAVE_TIFF

/* Unicap video capture library */
#cmakedefine HAVE_UNICAP

/* Video for Windows support */
#cmakedefine HAVE_VFW

/* V4L2 capturing support in videoio.h */
#cmakedefine HAVE_VIDEOIO

/* Win32 UI */
#cmakedefine HAVE_WIN32UI

/* Windows Runtime support */
#cmakedefine HAVE_WINRT

/* XIMEA camera support */
#cmakedefine HAVE_XIMEA

/* Xine video library */
#cmakedefine HAVE_XINE

/* Define to 1 if your processor stores words with the most significant byte
   first (like Motorola and SPARC, unlike Intel and VAX). */
#cmakedefine WORDS_BIGENDIAN
