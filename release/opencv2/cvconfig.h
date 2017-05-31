/* OpenCV compiled as static or dynamic libs */
#define BUILD_SHARED_LIBS

/* Compile for 'real' NVIDIA GPU architectures */
#define CUDA_ARCH_BIN ""

/* Create PTX or BIN for 1.0 compute capability */
/* #undef CUDA_ARCH_BIN_OR_PTX_10 */

/* NVIDIA GPU features are used */
#define CUDA_ARCH_FEATURES ""

/* Compile for 'virtual' NVIDIA PTX architectures */
#define CUDA_ARCH_PTX ""

/* AVFoundation video libraries */
#define HAVE_AVFOUNDATION

/* V4L capturing support */
/* #undef HAVE_CAMV4L */

/* V4L2 capturing support */
/* #undef HAVE_CAMV4L2 */

/* Carbon windowing environment */
/* #undef HAVE_CARBON */

/* AMD's Basic Linear Algebra Subprograms Library*/
/* #undef HAVE_CLAMDBLAS */

/* AMD's OpenCL Fast Fourier Transform Library*/
/* #undef HAVE_CLAMDFFT */

/* Clp support */
/* #undef HAVE_CLP */

/* Cocoa API */
#define HAVE_COCOA

/* C= */
/* #undef HAVE_CSTRIPES */

/* NVidia Cuda Basic Linear Algebra Subprograms (BLAS) API*/
/* #undef HAVE_CUBLAS */

/* NVidia Cuda Runtime API*/
/* #undef HAVE_CUDA */

/* NVidia Cuda Fast Fourier Transform (FFT) API*/
/* #undef HAVE_CUFFT */

/* IEEE1394 capturing support */
/* #undef HAVE_DC1394 */

/* IEEE1394 capturing support - libdc1394 v2.x */
/* #undef HAVE_DC1394_2 */

/* DirectX */
/* #undef HAVE_DIRECTX */
/* #undef HAVE_DIRECTX_NV12 */
/* #undef HAVE_D3D11 */
/* #undef HAVE_D3D10 */
/* #undef HAVE_D3D9 */

/* DirectShow Video Capture library */
/* #undef HAVE_DSHOW */

/* Eigen Matrix & Linear Algebra Library */
#define HAVE_EIGEN

/* FFMpeg video library */
/* #undef HAVE_FFMPEG */

/* Geospatial Data Abstraction Library */
/* #undef HAVE_GDAL */

/* GStreamer multimedia framework */
/* #undef HAVE_GSTREAMER */

/* GTK+ 2.0 Thread support */
/* #undef HAVE_GTHREAD */

/* GTK+ 2.x toolkit */
/* #undef HAVE_GTK */

/* Define to 1 if you have the <inttypes.h> header file. */
/* #undef HAVE_INTTYPES_H */

/* Intel Perceptual Computing SDK library */
/* #undef HAVE_INTELPERC */

/* Intel Integrated Performance Primitives */
#define HAVE_IPP
#define HAVE_IPP_ICV_ONLY

/* Intel IPP Async */
/* #undef HAVE_IPP_A */

/* JPEG-2000 codec */
#define HAVE_JASPER

/* IJG JPEG codec */
#define HAVE_JPEG

/* libpng/png.h needs to be included */
/* #undef HAVE_LIBPNG_PNG_H */

/* GDCM DICOM codec */
/* #undef HAVE_GDCM */

/* V4L/V4L2 capturing support via libv4l */
/* #undef HAVE_LIBV4L */

/* Microsoft Media Foundation Capture library */
/* #undef HAVE_MSMF */

/* NVidia Video Decoding API*/
/* #undef HAVE_NVCUVID */

/* NVidia Video Encoding API*/
/* #undef HAVE_NVCUVENC */

/* OpenCL Support */
#define HAVE_OPENCL
#define HAVE_OPENCL_STATIC
/* #undef HAVE_OPENCL_SVM */

/* OpenEXR codec */
#define HAVE_OPENEXR

/* OpenGL support*/
/* #undef HAVE_OPENGL */

/* OpenNI library */
/* #undef HAVE_OPENNI */

/* OpenNI library */
/* #undef HAVE_OPENNI2 */

/* PNG codec */
#define HAVE_PNG

/* Posix threads (pthreads) */
#define HAVE_PTHREADS

/* parallel_for with pthreads */
#define HAVE_PTHREADS_PF

/* Qt support */
/* #undef HAVE_QT */

/* Qt OpenGL support */
/* #undef HAVE_QT_OPENGL */

/* QuickTime video libraries */
/* #undef HAVE_QUICKTIME */

/* QTKit video libraries */
/* #undef HAVE_QTKIT */

/* Intel Threading Building Blocks */
/* #undef HAVE_TBB */

/* TIFF codec */
#define HAVE_TIFF

/* Unicap video capture library */
/* #undef HAVE_UNICAP */

/* Video for Windows support */
/* #undef HAVE_VFW */

/* V4L2 capturing support in videoio.h */
/* #undef HAVE_VIDEOIO */

/* Win32 UI */
/* #undef HAVE_WIN32UI */

/* XIMEA camera support */
/* #undef HAVE_XIMEA */

/* Xine video library */
/* #undef HAVE_XINE */

/* Define if your processor stores words with the most significant byte
   first (like Motorola and SPARC, unlike Intel and VAX). */
/* #undef WORDS_BIGENDIAN */

/* gPhoto2 library */
/* #undef HAVE_GPHOTO2 */

/* VA library (libva) */
/* #undef HAVE_VA */

/* Intel VA-API/OpenCL */
/* #undef HAVE_VA_INTEL */

/* Lapack */
#define HAVE_LAPACK

/* FP16 */
/* #undef HAVE_FP16 */

/* Library was compiled with functions instrumentation */
/* #undef ENABLE_INSTRUMENTATION */

/* OpenVX */
/* #undef HAVE_OPENVX */
