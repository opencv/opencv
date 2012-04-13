/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those systems.
   */
#cmakedefine  CRAY_STACKSEG_END

/* Define to 1 if using `alloca.c'. */
#cmakedefine C_ALLOCA

/* Define to 1 if you have `alloca', as a function or macro. */
#cmakedefine HAVE_ALLOCA 1

/* Define to 1 if you have <alloca.h> and it should be used (not on Ultrix).
   */
#cmakedefine HAVE_ALLOCA_H 1

/* V4L capturing support */
#cmakedefine HAVE_CAMV4L

/* V4L2 capturing support */
#cmakedefine HAVE_CAMV4L2

/* V4L/V4L2 capturing support via libv4l */
#cmakedefine HAVE_LIBV4L

/* Carbon windowing environment */
#cmakedefine HAVE_CARBON

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

/* FFMpeg version flag */
#cmakedefine  NEW_FFMPEG

/* ffmpeg's libswscale */
#cmakedefine  HAVE_FFMPEG_SWSCALE

/* GStreamer multimedia framework */
#cmakedefine  HAVE_GSTREAMER

/* GTK+ 2.0 Thread support */
#cmakedefine  HAVE_GTHREAD

/* GTK+ 2.x toolkit */
#cmakedefine  HAVE_GTK

/* OpenEXR codec */
#cmakedefine  HAVE_ILMIMF

/* Apple ImageIO Framework */
#cmakedefine  HAVE_IMAGEIO

/* Define to 1 if you have the <inttypes.h> header file. */
#cmakedefine  HAVE_INTTYPES_H 1

/* JPEG-2000 codec */
#cmakedefine  HAVE_JASPER

/* IJG JPEG codec */
#cmakedefine  HAVE_JPEG

/* Define to 1 if you have the `dl' library (-ldl). */
#cmakedefine  HAVE_LIBDL 1

/* Define to 1 if you have the `gomp' library (-lgomp). */
#cmakedefine  HAVE_LIBGOMP 1

/* Define to 1 if you have the `m' library (-lm). */
#cmakedefine  HAVE_LIBM 1

/* libpng/png.h needs to be included */
#cmakedefine  HAVE_LIBPNG_PNG_H

/* Define to 1 if you have the `pthread' library (-lpthread). */
#cmakedefine  HAVE_LIBPTHREAD 1

/* Define to 1 if you have the `lrint' function. */
#cmakedefine  HAVE_LRINT 1

/* PNG codec */
#cmakedefine  HAVE_PNG

/* Define to 1 if you have the `png_get_valid' function. */
#cmakedefine  HAVE_PNG_GET_VALID 1

/* png.h needs to be included */
#cmakedefine  HAVE_PNG_H

/* Define to 1 if you have the `png_set_tRNS_to_alpha' function. */
#cmakedefine  HAVE_PNG_SET_TRNS_TO_ALPHA 1

/* QuickTime video libraries */
#cmakedefine  HAVE_QUICKTIME

/* AVFoundation video libraries */
#cmakedefine  HAVE_AVFOUNDATION

/* TIFF codec */
#cmakedefine  HAVE_TIFF

/* Unicap video capture library */
#cmakedefine  HAVE_UNICAP

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine  HAVE_UNISTD_H 1

/* Xine video library */
#cmakedefine  HAVE_XINE

/* OpenNI library */
#cmakedefine  HAVE_OPENNI

/* LZ77 compression/decompression library (used for PNG) */
#cmakedefine  HAVE_ZLIB

/* Intel Integrated Performance Primitives */
#cmakedefine  HAVE_IPP

/* OpenCV compiled as static or dynamic libs */
#cmakedefine  BUILD_SHARED_LIBS

/* Name of package */
#define  PACKAGE "${PACKAGE}"

/* Define to the address where bug reports for this package should be sent. */
#define  PACKAGE_BUGREPORT "${PACKAGE_BUGREPORT}"

/* Define to the full name of this package. */
#define  PACKAGE_NAME "${PACKAGE_NAME}"

/* Define to the full name and version of this package. */
#define  PACKAGE_STRING "${PACKAGE_STRING}"

/* Define to the one symbol short name of this package. */
#define  PACKAGE_TARNAME "${PACKAGE_TARNAME}"

/* Define to the version of this package. */
#define  PACKAGE_VERSION "${PACKAGE_VERSION}"

/* If using the C implementation of alloca, define if you know the
   direction of stack growth for your system; otherwise it will be
   automatically deduced at runtime.
	STACK_DIRECTION > 0 => grows toward higher addresses
	STACK_DIRECTION < 0 => grows toward lower addresses
	STACK_DIRECTION = 0 => direction of growth unknown */
#cmakedefine  STACK_DIRECTION

/* Version number of package */
#define  VERSION "${PACKAGE_VERSION}"

/* Define to 1 if your processor stores words with the most significant byte
   first (like Motorola and SPARC, unlike Intel and VAX). */
#cmakedefine  WORDS_BIGENDIAN

/* Intel Threading Building Blocks */
#cmakedefine  HAVE_TBB

/* Eigen Matrix & Linear Algebra Library */
#cmakedefine  HAVE_EIGEN

/* NVidia Cuda Runtime API*/
#cmakedefine HAVE_CUDA

/* NVidia Cuda Fast Fourier Transform (FFT) API*/
#cmakedefine HAVE_CUFFT

/* NVidia Cuda Basic Linear Algebra Subprograms (BLAS) API*/
#cmakedefine HAVE_CUBLAS

/* Compile for 'real' NVIDIA GPU architectures */
#define CUDA_ARCH_BIN "${OPENCV_CUDA_ARCH_BIN}"

/* Compile for 'virtual' NVIDIA PTX architectures */
#define CUDA_ARCH_PTX "${OPENCV_CUDA_ARCH_PTX}"

/* NVIDIA GPU features are used */
#define CUDA_ARCH_FEATURES "${OPENCV_CUDA_ARCH_FEATURES}"

/* Create PTX or BIN for 1.0 compute capability */
#cmakedefine CUDA_ARCH_BIN_OR_PTX_10

/* VideoInput library */
#cmakedefine HAVE_VIDEOINPUT

/* XIMEA camera support */
#cmakedefine HAVE_XIMEA

/* OpenGL support*/
#cmakedefine HAVE_OPENGL

/* Clp support */
#cmakedefine HAVE_CLP
