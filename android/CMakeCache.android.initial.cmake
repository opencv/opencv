########################
# Initial cache settings for opencv on android
# run cmake with:
# cmake -C 
########################
#Build all examples
set(BUILD_EXAMPLES OFF CACHE BOOL "" )

#Build Reference Manual
set(BUILD_REFMAN OFF CACHE BOOL "" )

#Build LaTeX OpenCV Documentation
#set(BUILD_LATEX_DOCS OFF CACHE BOOL "" )

#Build with Python support
set(BUILD_NEW_PYTHON_SUPPORT OFF CACHE BOOL "" )

#Build a installer with the SDK
set(BUILD_PACKAGE OFF CACHE BOOL "" )

#Build shared libraries (.dll/.so CACHE BOOL "" ) instead of static ones (.lib/.a CACHE BOOL "" )
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" )

#Build 3rd party libraries
set(OPENCV_BUILD_3RDPARTY_LIBS ON CACHE BOOL "" )

#Choose the type of build, options are: None Debug Release RelWithDebInfo
# MinSizeRel.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" )

#Include IEEE1394 support
set(WITH_1394 OFF CACHE BOOL "" )

#Include NVidia Cuda Runtime support
set(WITH_CUDA OFF CACHE BOOL "" )

#Include Eigen2/Eigen3 support
set(WITH_EIGEN2 OFF CACHE BOOL "" )

#Include FFMPEG support
set(WITH_FFMPEG OFF CACHE BOOL "" )

#Include Gstreamer support
set(WITH_GSTREAMER OFF CACHE BOOL "" )

#Include GTK support
set(WITH_GTK OFF CACHE BOOL "" )

#Include Intel IPP support
set(WITH_IPP OFF CACHE BOOL "" )

#Include JPEG2K support
set(WITH_JASPER ON CACHE BOOL "" )

#Include JPEG support
set(WITH_JPEG ON CACHE BOOL "" )

#Include ILM support via OpenEXR
set(WITH_OPENEXR OFF CACHE BOOL "" )

#Include OpenNI support
set(WITH_OPENNI OFF CACHE BOOL "" )

#Include PNG support
set(WITH_PNG ON CACHE BOOL "" )

#Include Prosilica GigE support
set(WITH_PVAPI OFF CACHE BOOL "" )

#Build with Qt Backend support
set(WITH_QT OFF CACHE BOOL "" )

#Add OpenGL extension to Qt
set(WITH_QT_OPENGL OFF CACHE BOOL "" )

#Include Intel TBB support
set(WITH_TBB OFF CACHE BOOL "" )

#Include TIFF support
set(WITH_TIFF ON CACHE BOOL "" )

#Include Unicap support (GPL CACHE BOOL "" )
set(WITH_UNICAP OFF CACHE BOOL "" )

#Include Video 4 Linux support
set(WITH_V4L OFF CACHE BOOL "" )

#Include Xine support (GPL CACHE BOOL "" )
set(WITH_XINE OFF CACHE BOOL "" )

#Enable SSE instructions
SET( ENABLE_SSE OFF CACHE INTERNAL "" FORCE )

#Enable SSE2 instructions
SET( ENABLE_SSE2 OFF CACHE INTERNAL "" FORCE )

#Enable SSE3 instructions
SET( ENABLE_SSE3 OFF CACHE INTERNAL "" FORCE )

#Enable SSE4.1 instructions
SET( ENABLE_SSE41 OFF CACHE INTERNAL "" FORCE )

#Enable SSE4.2 instructions
SET( ENABLE_SSE42 OFF CACHE INTERNAL "" FORCE )

#Enable SSSE3 instructions
SET( ENABLE_SSSE3 OFF CACHE INTERNAL "" FORCE )
