# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------
project(opencv_highgui)

if(WITH_JPEG)
	add_definitions(-DHAVE_JPEG)
	if(NOT JPEG_FOUND)
		set(use_3rdparty TRUE)
	endif()
endif()

if(WITH_PNG)
	add_definitions(-DHAVE_PNG)
	if(NOT PNG_FOUND)
		set(use_3rdparty TRUE)
	endif()
endif()

if(WITH_TIFF)
	add_definitions(-DHAVE_TIFF)
	if(NOT TIFF_FOUND)
		set(use_3rdparty TRUE)
	endif()
endif()

if(WITH_JASPER)
	add_definitions(-DHAVE_JASPER)
	if(NOT JASPER_FOUND)
		set(use_3rdparty TRUE)
	endif()
endif()

if(WITH_OPENEXR AND OPENEXR_FOUND)
	add_definitions(-DHAVE_OPENEXR)
	include_directories(${OPENEXR_INCLUDE_PATHS})
endif()

if(use_3rdparty)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/include")
endif()

if(MSVC)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
endif()

file(GLOB grfmt_hdrs src/grfmt*.hpp)
file(GLOB grfmt_srcs src/grfmt*.cpp)
set(grfmt_hdrs src/bitstrm.hpp ${grfmt_hdrs})
set(grfmt_srcs src/bitstrm.cpp ${grfmt_srcs})

source_group("Src\\grfmts" FILES ${grfmt_hdrs} ${grfmt_srcs})

set(highgui_srcs
    src/cap.cpp
    src/cap_images.cpp
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
if(WIN32)

    if (HAVE_QT)
         if (HAVE_QT_OPENGL)
			set( QT_USE_QTOPENGL TRUE )
         endif()
		 INCLUDE(${QT_USE_FILE})
		 
		 SET(_MOC_HEADERS src/window_QT.h )
		 QT4_WRAP_CPP(_MOC_OUTFILES ${_MOC_HEADERS})
		 
		 set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${QT_LIBRARIES})
		 set(highgui_srcs ${highgui_srcs} src/window_QT.cpp ${_MOC_OUTFILES})
		 
    else()
        set(highgui_srcs ${highgui_srcs} src/window_w32.cpp)
    endif()

    set(highgui_srcs ${highgui_srcs} src/cap_vfw.cpp src/cap_cmu.cpp src/cap_w32.cpp src/cap_dshow.cpp)
    if(HAVE_MIL)
        set(highgui_srcs ${highgui_srcs} src/cap_mil.cpp)
    endif()
endif()

if(UNIX)

    if (HAVE_QT)        
         if (HAVE_QT_OPENGL)
			set( QT_USE_QTOPENGL TRUE )
			#set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${OPENGL_LIBRARIES})
         endif()
		 INCLUDE(${QT_USE_FILE})
		 
		 SET(_MOC_HEADERS src/window_QT.h)
		 QT4_WRAP_CPP(_MOC_OUTFILES ${_MOC_HEADERS})
		 
		 set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${QT_LIBRARIES})
		 set(highgui_srcs ${highgui_srcs} src/window_QT.cpp ${_MOC_OUTFILES})
		 
    else()
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
        set(highgui_srcs ${highgui_srcs} src/cap_ffmpeg.cpp)
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
        if(NOT HAVE_GSTREAMER_APP)
            set(highgui_srcs ${highgui_srcs} src/gstappsink.cpp)
            set(highgui_hdrs ${highgui_hdrs} src/gstappsink.h)
        endif()
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

	if(JPEG_FOUND)
		include_directories(${JPEG_INCLUDE_DIR})
	endif()
    if(PNG_FOUND)
		add_definitions(${PNG_DEFINITIONS})
		include_directories(${PNG_INCLUDE_DIR})
	endif()
	if(TIFF_FOUND)
		include_directories(${TIFF_INCLUDE_DIR})
	endif()
	if(JASPER_FOUND)
		include_directories(${JASPER_INCLUDE_DIR})
	endif()

    foreach(P ${HIGHGUI_INCLUDE_DIRS})
        include_directories(${P})
    endforeach()

    foreach(P ${HIGHGUI_LIBRARY_DIRS})
        link_directories(${P})
    endforeach()
endif()

#YV
if(APPLE)
    add_definitions(-DHAVE_QUICKTIME=1)
    if(NOT OPENCV_BUILD_3RDPARTY_LIBS)
        add_definitions(-DHAVE_IMAGEIO=1)
    endif()
    
    if (HAVE_QT)
         if (HAVE_QT_OPENGL)
			set( QT_USE_QTOPENGL TRUE )
         endif()
		 INCLUDE(${QT_USE_FILE})
		 
		 SET(_MOC_HEADERS src/window_QT.h)
		 QT4_WRAP_CPP(_MOC_OUTFILES ${_MOC_HEADERS})
		 
		 set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${QT_LIBRARIES})
		 set(highgui_srcs ${highgui_srcs} src/window_QT.cpp ${_MOC_OUTFILES})
		         
    else()
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
		set(highgui_srcs ${highgui_srcs} src/cap_qtkit.mm)
	endif()
endif(APPLE)

source_group("Src" FILES ${highgui_srcs} ${highgui_hdrs})
source_group("Include" FILES ${highgui_ext_hdrs})

if(OPENCV_BUILD_3RDPARTY_LIBS)
    if(WIN32)
        link_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/lib")
    endif()
    link_directories(
        "${CMAKE_BINARY_DIR}/3rdparty/lib"
        "${CMAKE_BINARY_DIR}/3rdparty/lib/${ConfigurationName}"
        )
endif()

set(lib_srcs ${highgui_srcs} ${grfmt_srcs})

# ----------------------------------------------------------------------------------
# 				Define the library target:
# ----------------------------------------------------------------------------------
set(the_target "opencv_highgui")

add_definitions(-DHIGHGUI_EXPORTS -DCVAPI_EXPORTS)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
					"${CMAKE_CURRENT_SOURCE_DIR}/../core/include"
					"${CMAKE_CURRENT_SOURCE_DIR}/../imgproc/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${CMAKE_CURRENT_BINARY_DIR}")

add_library(${the_target} ${lib_srcs} ${highgui_hdrs} ${grfmt_hdrs} ${highgui_ext_hdrs})

if(PCHSupport_FOUND)
    set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp)
    if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
        if(${CMAKE_GENERATOR} MATCHES "Visual*")
            set(${the_target}_pch "src/precomp.cpp")
        endif()            
        add_native_precompiled_header(${the_target} ${pch_header})
    elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
        add_precompiled_header(${the_target} ${pch_header})
    endif()
endif()

# For dynamic link numbering convenions
set_target_properties(${the_target} PROPERTIES
    VERSION ${OPENCV_VERSION}
    SOVERSION ${OPENCV_SOVERSION}
    OUTPUT_NAME "${the_target}${OPENCV_DLLVERSION}"
    )

# Additional target properties
set_target_properties(${the_target} PROPERTIES
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
	LINK_INTERFACE_LIBRARIES ""
	)

if(MSVC)
	set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib")
endif(MSVC)

# Dependencies of this target:
add_dependencies(${the_target} opencv_core opencv_imgproc)

# Add the required libraries for linking:
target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} opencv_core opencv_imgproc ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES} ${JASPER_LIBRARIES} ${HIGHGUI_LIBRARIES} ${OPENEXR_LIBRARIES})

if(WITH_JASPER AND NOT JASPER_FOUND)
	add_dependencies(${the_target} libjasper)
	target_link_libraries(${the_target} libjasper)
endif()

if(WITH_JPEG AND NOT JPEG_FOUND)
	add_dependencies(${the_target} libjpeg)
	target_link_libraries(${the_target} libjpeg)
endif()

if(WITH_PNG AND NOT PNG_FOUND)
	add_dependencies(${the_target} libpng zlib)
	target_link_libraries(${the_target} libpng zlib)
endif()

if(WITH_TIFF AND NOT TIFF_FOUND)
	add_dependencies(${the_target} libtiff)
	target_link_libraries(${the_target} libtiff)
endif()

#if(OPENCV_BUILD_3RDPARTY_LIBS)
#	add_dependencies(${the_target} libjasper libjpeg libpng libtiff zlib)
#	target_link_libraries(${the_target} libjasper libjpeg libpng libtiff zlib)
#endif()
if(WIN32)
	target_link_libraries(${the_target} comctl32 gdi32 ole32)
	
	if (MSVC)
		target_link_libraries(${the_target} vfw32)
	endif()

	if(MINGW)
		if(MINGW64)
			target_link_libraries(${the_target} msvfw32 avifil32 avicap32 winmm videoInput64 strmiids)
		else()
			target_link_libraries(${the_target} vfw32 winmm videoInput strmiids)
		endif()  
    endif()
endif()

if(APPLE)
	target_link_libraries(${the_target} "-lbz2 -framework Cocoa -framework QuartzCore")
	if(WITH_CARBON)
		target_link_libraries(${the_target} "-framework Carbon")
	endif()
	if(NOT WITH_QUICKTIME)
		target_link_libraries(${the_target} "-framework QTKit")
    endif()
	if(WITH_CARBON OR WITH_QUICKTIME)
        target_link_libraries(${the_target} "-framework QuickTime -framework CoreFoundation")
	endif()
endif()

install(TARGETS ${the_target}
	RUNTIME DESTINATION bin COMPONENT main
	LIBRARY DESTINATION lib COMPONENT main
	ARCHIVE DESTINATION lib COMPONENT main)

install(FILES ${highgui_ext_hdrs}
        DESTINATION include/opencv2/highgui
        COMPONENT main)
