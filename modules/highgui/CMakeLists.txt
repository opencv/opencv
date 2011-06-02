# ----------------------------------------------------------------------------
#  CMake file for highgui. See root CMakeLists.txt
#   Some parts taken from version of Hartmut Seichter, HIT Lab NZ.
#   Jose Luis Blanco, 2008
# ----------------------------------------------------------------------------
project(opencv_highgui)

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
if (HAVE_QT)
    if (HAVE_QT_OPENGL)
        set(QT_USE_QTOPENGL TRUE)
    endif()
    INCLUDE(${QT_USE_FILE})
    
    SET(_RCCS_FILES src/window_QT.qrc)
    QT4_ADD_RESOURCES(_RCC_OUTFILES ${_RCCS_FILES})

    SET(_MOC_HEADERS src/window_QT.h )
    QT4_WRAP_CPP(_MOC_OUTFILES ${_MOC_HEADERS})
     
    set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} ${QT_LIBRARIES})
    set(highgui_srcs ${highgui_srcs} src/window_QT.cpp ${_MOC_OUTFILES} ${_RCC_OUTFILES} )
endif()
    
if(WIN32)
    if(NOT HAVE_QT)
        set(highgui_srcs ${highgui_srcs} src/window_w32.cpp)
    endif()
    set(highgui_srcs ${highgui_srcs} src/cap_vfw.cpp src/cap_cmu.cpp src/cap_w32.cpp src/cap_dshow.cpp)
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
    add_definitions(-DHAVE_QUICKTIME=1)
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
        set(highgui_srcs ${highgui_srcs} src/cap_qtkit.mm)
    endif()
endif(APPLE)

if(HAVE_ANDROID_NATIVE_CAMERA)
  set(highgui_srcs ${highgui_srcs} src/cap_android.cpp)
  add_definitions(-DHAVE_ANDROID_NATIVE_CAMERA)
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
#               Define the library target:
# ----------------------------------------------------------------------------------
set(the_target "opencv_highgui")

if (BUILD_SHARED_LIBS)
    add_definitions(-DHIGHGUI_EXPORTS -DCVAPI_EXPORTS)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../core/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../imgproc/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${CMAKE_CURRENT_BINARY_DIR}")

if(WIN32)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/include")
endif()

add_library(${the_target} ${lib_srcs} ${highgui_hdrs} ${grfmt_hdrs} ${highgui_ext_hdrs})

if(PCHSupport_FOUND AND USE_PRECOMPILED_HEADERS)
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
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR lib
    LINK_INTERFACE_LIBRARIES ""
    )

if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_target} PROPERTIES FOLDER "modules")
endif() 


if(MSVC)
    set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif(MSVC)

# Dependencies of this target:
add_dependencies(${the_target} opencv_core opencv_imgproc)

# Add the required libraries for linking:

if(WITH_JASPER AND NOT JASPER_FOUND)
    add_dependencies(${the_target} libjasper)
endif()

if(WITH_JPEG AND NOT JPEG_FOUND)
    add_dependencies(${the_target} libjpeg)
endif()

if(WITH_PNG AND NOT PNG_FOUND)
    add_dependencies(${the_target} libpng)
endif()

if(WITH_TIFF AND NOT TIFF_FOUND)
    add_dependencies(${the_target} libtiff)
endif()

if(NOT ZLIB_FOUND)
    add_dependencies(${the_target} zlib)
endif()

#message(STATUS "GRFMT: ${GRFMT_LIBS}")
#message(STATUS "HIGHGUI_LIBS: ${HIGHGUI_LIBRARIES}")
#message(STATUS "OPENCV_LIBS: ${OPENCV_LINKER_LIBS}")

target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} opencv_core
                      opencv_imgproc ${GRFMT_LIBS} ${HIGHGUI_LIBRARIES})
if( OPENNI_LIBRARY )
    target_link_libraries(${the_target} ${OPENNI_LIBRARY})
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


############################# highgui tests ################################
if(BUILD_TESTS)

  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/test"
                      "${CMAKE_CURRENT_BINARY_DIR}")

  set(test_deps opencv_ts opencv_highgui opencv_imgproc)

  foreach(d ${test_deps})
      if(${d} MATCHES "opencv_")
          string(REPLACE "opencv_" "${CMAKE_CURRENT_SOURCE_DIR}/../" d_dir ${d})
          include_directories("${d_dir}/include")
      endif()
  endforeach()

  file(GLOB test_srcs "test/*.cpp")
  file(GLOB test_hdrs "test/*.h*")

  source_group("Src" FILES ${test_srcs})
  source_group("Include" FILES ${test_hdrs})

  set(the_target "opencv_test_highgui")

  add_executable(${the_target} ${test_srcs} ${test_hdrs})

  if(PCHSupport_FOUND AND USE_PRECOMPILED_HEADERS)
      set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/test/test_precomp.hpp)
      if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
          if(${CMAKE_GENERATOR} MATCHES "Visual*")
              set(${the_target}_pch "test/test_precomp.cpp")
          endif()            
          add_native_precompiled_header(${the_target} ${pch_header})
      elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
          add_precompiled_header(${the_target} ${pch_header})
      endif()
  endif()

  # Additional target properties
  set_target_properties(${the_target} PROPERTIES
      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
      RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      )
	
  if(ENABLE_SOLUTION_FOLDERS)
	  set_target_properties(${the_target} PROPERTIES FOLDER "tests")
  endif()	

  add_dependencies(${the_target} ${test_deps})

  # Add the required libraries for linking:
  target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${test_deps})

  enable_testing()
  get_target_property(LOC ${the_target} LOCATION)
  add_test(${the_target} "${LOC}")

  if(WIN32)
      if (MSVC AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
      endif()
      install(TARGETS ${the_target} RUNTIME DESTINATION bin COMPONENT main)
  endif()
endif(BUILD_TESTS)
