# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------

option(OCVMODULE_HIGHGUI "Include highgui module into the OpenCV build" ON)
if(NOT OCVMODULE_HIGHGUI)
    return()
endif()

set(the_target "opencv_highgui")
project(${the_target})

#so that the headers get installed.
file(GLOB lib_hdrs "include/opencv2/highgui/*.h*")

set(GRFMT_LIBS)

if(WITH_PNG OR WITH_TIFF OR WITH_OPENEXR)
    if(ZLIB_FOUND)
        include_directories(${ZLIB_INCLUDE_DIR})
        set(GRFMT_LIBS ${GRFMT_LIBS} ${ZLIB_LIBRARIES})
    else()
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/zlib")
        set(GRFMT_LIBS ${GRFMT_LIBS} zlib)
    endif()
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

set(highgui_srcs
    src/cap.cpp
    src/cap_images.cpp
    src/cap_ffmpeg.cpp
    src/loadsave.cpp
    src/precomp.cpp
    src/utils.cpp
    src/window.cpp
    )

set(highgui_hdrs src/precomp.hpp src/utils.hpp)

#YV
set(lib_hdr_names highgui_c.h highgui.hpp)
set(highgui_ext_hdrs)
foreach(h ${lib_hdr_names})
    list(APPEND highgui_ext_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/highgui/${h}")
endforeach()

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

if(WITH_ANDROID_CAMERA)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../androidcamera/include")
  set(highgui_srcs ${highgui_srcs} src/cap_android.cpp)
  add_definitions(-DHAVE_ANDROID_NATIVE_CAMERA)
  set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} opencv_androidcamera)
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

if(OPENCV_BUILD_3RDPARTY_LIBS AND WIN32)
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/lib")
endif()

if(COMMAND get_module_external_sources)
    set( lib_srcs "${highgui_srcs}" )
    set( lib_int_hdrs "${highgui_hdrs}" )
    get_module_external_sources(highgui)
    set( highgui_srcs "${lib_srcs}" )
    set( highgui_hdrs "${lib_int_hdrs}" )
endif()

source_group("Src" FILES ${highgui_srcs} ${highgui_hdrs})
source_group("Include" FILES ${highgui_ext_hdrs})

#message(STATUS "GRFMT: ${GRFMT_LIBS}")
#message(STATUS "OPENCV_LIBS: ${OPENCV_LINKER_LIBS}")
#message(STATUS "HIGHGUI_LIBS: ${HIGHGUI_LIBRARIES}")

# ----------------------------------------------------------------------------------
#               Define the library target:
# ----------------------------------------------------------------------------------
opencv_module_includes(opencv_core opencv_imgproc)
if(WIN32)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/include")
endif()

add_library(${the_target} ${highgui_srcs} ${grfmt_srcs} ${highgui_hdrs} ${grfmt_hdrs} ${highgui_ext_hdrs})
target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} opencv_core opencv_imgproc ${GRFMT_LIBS} ${HIGHGUI_LIBRARIES})
opencv_module_register(${the_target})

if(BUILD_SHARED_LIBS)
    add_definitions(-DHIGHGUI_EXPORTS)
endif()

if(MSVC)
    set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /NODEFAULTLIB:libcmt.lib /DEBUG")
endif()

opencv_module_setup(highgui)
define_opencv_test(highgui)
define_opencv_perf_test(highgui)
