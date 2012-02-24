set(the_description "High-level GUI and Media I/O")
ocv_add_module(highgui opencv_imgproc OPTIONAL opencv_androidcamera)

# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

set(GRFMT_LIBS "")

if(WITH_PNG OR WITH_TIFF OR WITH_OPENEXR)
  include_directories(${ZLIB_INCLUDE_DIR})
  set(GRFMT_LIBS ${GRFMT_LIBS} ${ZLIB_LIBRARY})
endif()

if(WITH_JPEG)
    add_definitions(-DHAVE_JPEG)
    if(JPEG_FOUND)
        include_directories(${JPEG_INCLUDE_DIR})
        set(GRFMT_LIBS ${GRFMT_LIBS} ${JPEG_LIBRARIES})
    else()
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/libjpeg")
        set(GRFMT_LIBS ${GRFMT_LIBS} libjpeg)
    endif()
endif()

if(WITH_PNG)
    add_definitions(-DHAVE_PNG)
    if(PNG_FOUND)
        add_definitions(${PNG_DEFINITIONS})
        include_directories(${PNG_INCLUDE_DIR})
        set(GRFMT_LIBS ${GRFMT_LIBS} ${PNG_LIBRARIES})
    else()
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/libpng")
        set(GRFMT_LIBS ${GRFMT_LIBS} libpng)
    endif()
endif()

if(WITH_TIFF)
    add_definitions(-DHAVE_TIFF)
    if(TIFF_FOUND)
        include_directories(${TIFF_INCLUDE_DIR})
        set(GRFMT_LIBS ${GRFMT_LIBS} ${TIFF_LIBRARIES})
    else()
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/libtiff")
        set(GRFMT_LIBS ${GRFMT_LIBS} libtiff)
    endif()
endif()

if(WITH_JASPER)
    add_definitions(-DHAVE_JASPER)
    if(JASPER_FOUND)
        include_directories(${JASPER_INCLUDE_DIR})
        set(GRFMT_LIBS ${GRFMT_LIBS} ${JASPER_LIBRARIES})
    else()
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/libjasper")
        set(GRFMT_LIBS ${GRFMT_LIBS} libjasper)
    endif()
endif()

if(WITH_OPENEXR AND OPENEXR_FOUND)
    add_definitions(-DHAVE_OPENEXR)
    include_directories(${OPENEXR_INCLUDE_PATHS})
    set(GRFMT_LIBS ${GRFMT_LIBS} ${OPENEXR_LIBRARIES})
endif()

file(GLOB grfmt_hdrs src/grfmt*.hpp)
file(GLOB grfmt_srcs src/grfmt*.cpp)
set(grfmt_hdrs src/bitstrm.hpp ${grfmt_hdrs})
set(grfmt_srcs src/bitstrm.cpp ${grfmt_srcs})

source_group("Src\\grfmts" FILES ${grfmt_hdrs} ${grfmt_srcs})

set(highgui_hdrs src/precomp.hpp src/utils.hpp)
set(highgui_srcs
    src/cap.cpp
    src/cap_images.cpp
    src/cap_ffmpeg.cpp
    src/loadsave.cpp
    src/precomp.cpp
    src/utils.cpp
    src/window.cpp
    )
file(GLOB highgui_ext_hdrs "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")

#YV
if (HAVE_QT)
    if (HAVE_QT_OPENGL)
        set(QT_USE_QTOPENGL TRUE)
    endif()
    INCLUDE(${QT_USE_FILE})
    
    SET(_RCCS_FILES src/window_QT.qrc)
    QT4_ADD_RESOURCES(_RCC_OUTFILES ${_RCCS_FILES})

    SET(_MOC_HEADERS src/window_QT.h )
    QT4_WRAP_CPP(_MOC_OUTFILES ${_MOC_HEADERS})
     
    set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${QT_LIBRARIES} ${QT_QTTEST_LIBRARY})
    set(highgui_srcs ${highgui_srcs} src/window_QT.cpp ${_MOC_OUTFILES} ${_RCC_OUTFILES} )
endif()
    
if(WIN32)
    if(NOT HAVE_QT)
        set(highgui_srcs ${highgui_srcs} src/window_w32.cpp)
    endif()
    set(highgui_srcs ${highgui_srcs} src/cap_vfw.cpp src/cap_cmu.cpp src/cap_dshow.cpp)
    if(HAVE_MIL)
        set(highgui_srcs ${highgui_srcs} src/cap_mil.cpp)
    endif()
endif()

if(UNIX)
    if(NOT HAVE_QT)        
        if(HAVE_GTK)
            set(highgui_srcs ${highgui_srcs} src/window_gtk.cpp)
        endif()
    endif()

    if(HAVE_XINE)
        set(highgui_srcs ${highgui_srcs} src/cap_xine.cpp)
    endif()

    if(HAVE_DC1394_2)
        set(highgui_srcs ${highgui_srcs} src/cap_dc1394_v2.cpp)
    endif()

    if(HAVE_DC1394)
        set(highgui_srcs ${highgui_srcs} src/cap_dc1394.cpp)
    endif()

    if(HAVE_FFMPEG)
        if(BZIP2_LIBRARIES)
            set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${BZIP2_LIBRARIES})
        endif()
    endif()
    
    if(HAVE_PVAPI)
        add_definitions(-DHAVE_PVAPI)
        set(highgui_srcs src/cap_pvapi.cpp ${highgui_srcs})
        set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} PvAPI)
    endif()

    if(HAVE_GSTREAMER)
        set(highgui_srcs ${highgui_srcs} src/cap_gstreamer.cpp)
    endif()

    if(HAVE_UNICAP)
        set(highgui_srcs ${highgui_srcs} src/cap_unicap.cpp)
    endif()

    if(HAVE_LIBV4L)
        set(highgui_srcs ${highgui_srcs} src/cap_libv4l.cpp)
    else()
        if(HAVE_CAMV4L OR HAVE_CAMV4L2)
            set(highgui_srcs ${highgui_srcs} src/cap_v4l.cpp)
        endif()
    endif()

    foreach(P ${HIGHGUI_INCLUDE_DIRS})
        include_directories(${P})
    endforeach()

    foreach(P ${HIGHGUI_LIBRARY_DIRS})
        link_directories(${P})
    endforeach()
endif()

#OpenNI
if(WITH_OPENNI AND HAVE_OPENNI)
    set(highgui_srcs ${highgui_srcs} src/cap_openni.cpp)
    include_directories(${OPENNI_INCLUDE_DIR})
endif()

#YV
if(APPLE)
	if (NOT IOS)
        add_definitions(-DHAVE_QUICKTIME=1)
	endif()

    if(NOT OPENCV_BUILD_3RDPARTY_LIBS)
        add_definitions(-DHAVE_IMAGEIO=1)
    endif()
    
    if (NOT HAVE_QT)
        if(WITH_CARBON)
            add_definitions(-DHAVE_CARBON=1)
            set(highgui_srcs ${highgui_srcs} src/window_carbon.cpp)
        else()
            add_definitions(-DHAVE_COCOA=1)
            set(highgui_srcs ${highgui_srcs} src/window_cocoa.mm)
        endif()
    endif()
    
    if(WITH_QUICKTIME)
        set(highgui_srcs ${highgui_srcs} src/cap_qt.cpp)
    else()
		if (WITH_AVFOUNDATION)
			add_definitions(-DHAVE_AVFOUNDATION=1)
		    set(highgui_srcs ${highgui_srcs} src/cap_avfoundation.mm)
		else()
            set(highgui_srcs ${highgui_srcs} src/cap_qtkit.mm)
        endif()
    endif()
endif(APPLE)

if(HAVE_opencv_androidcamera)
  set(highgui_srcs ${highgui_srcs} src/cap_android.cpp)
  add_definitions(-DHAVE_ANDROID_NATIVE_CAMERA)#TODO: remove this line
endif()

if(HAVE_XIMEA AND XIMEA_FOUND)
  set(highgui_srcs ${highgui_srcs} src/cap_ximea.cpp)
  include_directories(${XIMEA_PATH})
  link_directories(${XIMEA_LIBRARY_DIR})
  set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} m3api)
  set(highgui_srcs ${highgui_srcs} src/cap_ximea.cpp)
endif()

if(OPENNI_LIBRARY)
  set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${OPENNI_LIBRARY})
endif()

if(APPLE AND NOT IOS)
  set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} bz2 "-framework Cocoa" "-framework QuartzCore")
  if(WITH_CARBON)
    set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} "-framework Carbon")
  endif()
  if(NOT WITH_QUICKTIME)
    set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} "-framework QTKit")
  endif()
  if(WITH_CARBON OR WITH_QUICKTIME)
    set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} "-framework QuickTime" "-framework CoreFoundation")
  endif()
endif()

if(IOS)
  set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} bz2 "-framework QuartzCore" "-framework CoreFoundation" "-framework ImageIO" "-framework CoreGraphics" "-framework AVFoundation")
endif()

if(WIN32)
  link_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/lib")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/include")#for directshow
endif()

source_group("Src" FILES ${highgui_srcs} ${highgui_hdrs})
source_group("Include" FILES ${highgui_ext_hdrs})
ocv_set_module_sources(HEADERS ${highgui_ext_hdrs} SOURCES ${highgui_srcs} ${highgui_hdrs} ${grfmt_srcs} ${grfmt_hdrs})

ocv_module_include_directories()

ocv_create_module(${GRFMT_LIBS} ${HIGHGUI_LIBRARIES})

if(BUILD_SHARED_LIBS)
  add_definitions(-DHIGHGUI_EXPORTS)
endif()

if(MSVC)
  set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

#stop automatic dependencies propagation for this module
set_target_properties(${the_module} PROPERTIES LINK_INTERFACE_LIBRARIES "")

ocv_add_precompiled_headers(${the_module})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
