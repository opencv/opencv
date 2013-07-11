/* Video for Windows support */
#cmakedefine HAVE_VFW

/* V4L capturing support */
#cmakedefine HAVE_CAMV4L

/* V4L2 capturing support */
#cmakedefine HAVE_CAMV4L2

/* V4L2 capturing support in videoio.h */
#cmakedefine HAVE_VIDEOIO

/* V4L/V4L2 capturing support via libv4l */
#cmakedefine HAVE_LIBV4L

/* Carbon windowing environment */
#cmakedefine HAVE_CARBON

/* Cocoa API */
#cmakedefine HAVE_COCOA

/* IEEE1394 capturing support */
#cmakedefine HAVE_DC1394

/* libdc1394 0.9.4 or 0.9.5 */
#cmakedefine HAVE_DC1394_095

/* IEEE1394 capturing support - libdc1394 v2.x */
#cmakedefine HAVE_DC1394_2

/* ffmpeg in Gentoo */
#cmakedefine HAVE_GENTOO_FFMPEG

/* FFMpeg video library */
#cmakedefine  HAVE_FFMPEG

/* ffmpeg's libswscale */
#cmakedefine  HAVE_FFMPEG_SWSCALE

/* GStreamer multimedia framework */
#cmakedefine  HAVE_GSTREAMER

/* GTK+ 2.0 Thread support */
#cmakedefine  HAVE_GTHREAD

/* Win32 UI */
#cmakedefine HAVE_WIN32UI

/* GTK+ 2.x toolkit */
#cmakedefine  HAVE_GTK

/* OpenEXR codec */
#cmakedefine  HAVE_OPENEXR

/* Apple ImageIO Framework */
#cmakedefine  HAVE_IMAGEIO

/* JPEG-2000 codec */
#cmakedefine  HAVE_JASPER

/* IJG JPEG codec */
#cmakedefine  HAVE_JPEG

/* libpng/png.h needs to be included */
#cmakedefine  HAVE_LIBPNG_PNG_H

/* PNG codec */
#cmakedefine  HAVE_PNG

/* QuickTime video libraries */
#cmakedefine  HAVE_QUICKTIME

/* AVFoundation video libraries */
#cmakedefine  HAVE_AVFOUNDATION

/* TIFF codec */
#cmakedefine  HAVE_TIFF

/* Unicap video capture library */
#cmakedefine  HAVE_UNICAP

/* Xine video library */
#cmakedefine  HAVE_XINE

/* OpenNI library */
#cmakedefine  HAVE_OPENNI

/* Intel Integrated Performance Primitives */
#cmakedefine  HAVE_IPP

/* OpenCV compiled as static or dynamic libs */
#cmakedefine  BUILD_SHARED_LIBS

/* Define to 1 if your processor stores words with the most significant byte
   first (like Motorola and SPARC, unlike Intel and VAX). */
#cmakedefine  WORDS_BIGENDIAN

/* Intel Threading Building Blocks */
#cmakedefine  HAVE_TBB

/* C= */
#cmakedefine  HAVE_CSTRIPES

/* Eigen Matrix & Linear Algebra Library */
#cmakedefine  HAVE_EIGEN

/* NVidia Cuda Runtime API*/
#cmakedefine HAVE_CUDA

/* NVidia Cuda Fast Fourier Transform (FFT) API*/
#cmakedefine HAVE_CUFFT

/* NVidia Cuda Basic Linear Algebra Subprograms (BLAS) API*/
#cmakedefine HAVE_CUBLAS

/* NVidia Video Decoding API*/
#cmakedefine HAVE_NVCUVID

/* Compile for 'real' NVIDIA GPU architectures */
#define CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN}"

/* Compile for 'virtual' NVIDIA PTX architectures */
#define CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX}"

/* NVIDIA GPU features are used */
#define CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES}"

/* Create PTX or BIN for 1.0 compute capability */
#cmakedefine CUDA_ARCH_BIN_OR_PTX_10

/* OpenCL Support */
#cmakedefine HAVE_OPENCL

/* AMD's OpenCL Fast Fourier Transform Library*/
#cmakedefine HAVE_CLAMDFFT

/* AMD's Basic Linear Algebra Subprograms Library*/
#cmakedefine HAVE_CLAMDBLAS

/* DirectShow Video Capture library */
#cmakedefine HAVE_DSHOW

/* Microsoft Media Foundation Capture library */
#cmakedefine HAVE_MSMF

/* XIMEA camera support */
#cmakedefine HAVE_XIMEA

/* OpenGL support*/
#cmakedefine HAVE_OPENGL

/* Qt support */
#cmakedefine HAVE_QT

/* Qt OpenGL support */
#cmakedefine HAVE_QT_OPENGL
