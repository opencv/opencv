# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------
project(python_support)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../filtered)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
endif()

# ----------------------------------- cv module ------------------------------

set_source_files_properties(cv.i PROPERTIES
    CPLUSPLUS ON
    SWIG_FLAGS -includeall
    SWIG_FLAGS -DSKIP_INCLUDES
    )

set(opencv_headers
    ${CMAKE_SOURCE_DIR}/include/opencv/cxtypes.h
    ${CMAKE_SOURCE_DIR}/include/opencv/cxcore.h
    ${CMAKE_SOURCE_DIR}/include/opencv/cvtypes.h
    ${CMAKE_SOURCE_DIR}/include/opencv/cv.h
    )

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cvaliases_autogen.i
   COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utils/extract_aliases.py
    ${opencv_headers} > ${CMAKE_CURRENT_BINARY_DIR}/cvaliases_autogen.i
   DEPENDS ${opencv_headers}
   )

set(SWIG_MODULE_cv_EXTRA_DEPS
    imagedata.i cvarr.i ${CMAKE_CURRENT_BINARY_DIR}/cvaliases_autogen.i cvaliases.i pyhelpers.i
    nointpb.i pytypemaps.i cvshadow.i cvseq.i
    ../general/cv.i ../general/memory.i ../general/typemaps.i
    ../general/extensions.i ../general/doublepointers.i
    ../general/sizeof.i ../general/cvmacros.i
    )

SWIG_ADD_MODULE(cv python cv.i cvshadow.cpp error.cpp error.h pyhelpers.cpp pyhelpers.h cvshadow.h pycvseq.hpp)
SWIG_LINK_LIBRARIES(cv ${PYTHON_LIBRARIES} cxcore cv)

# ----------------------------------- ml module ------------------------------

set_source_files_properties(ml.i PROPERTIES
    CPLUSPLUS ON
    SWIG_FLAGS -includeall
    SWIG_FLAGS -DSKIP_INCLUDES
    )

set(SWIG_MODULE_ml_EXTRA_DEPS
    nointpb.i pytypemaps.i
    ../general/memory.i ../general/typemaps.i
    ${CMAKE_SOURCE_DIR}/include/opencv/ml.h
    )

SWIG_ADD_MODULE(ml python ml.i pyhelpers.cpp pyhelpers.h)
SWIG_LINK_LIBRARIES(ml ${PYTHON_LIBRARIES} cxcore ml)

# --------------------------------highgui module ------------------------------

set_source_files_properties(highgui.i PROPERTIES
    CPLUSPLUS ON
    SWIG_FLAGS -includeall
    SWIG_FLAGS -DSKIP_INCLUDES
    )

set(SWIG_MODULE_highgui_EXTRA_DEPS
    nointpb.i pytypemaps.i
    ../general/highgui.i
    ../general/memory.i ../general/typemaps.i
    ${CMAKE_SOURCE_DIR}/include/opencv/highgui.h
    )

SWIG_ADD_MODULE(highgui python highgui.i pyhelpers.cpp pyhelpers.h)
SWIG_LINK_LIBRARIES(highgui ${PYTHON_LIBRARIES} cxcore cv highgui)

# ------------------------------ installation ----------------------------------

if(WIN32)
    set_target_properties(
        ${SWIG_MODULE_cv_REAL_NAME}
        ${SWIG_MODULE_ml_REAL_NAME}
        ${SWIG_MODULE_highgui_REAL_NAME}
        PROPERTIES SUFFIX ".pyd")
endif()

get_target_property(LOC_CV ${SWIG_MODULE_cv_REAL_NAME} LOCATION)
get_target_property(LOC_ML ${SWIG_MODULE_ml_REAL_NAME} LOCATION)
get_target_property(LOC_HIGHGUI ${SWIG_MODULE_highgui_REAL_NAME} LOCATION)

set(pyopencv_files __init__.py adaptors.py matlab_syntax.py
    ${CMAKE_CURRENT_BINARY_DIR}/cv.py
    ${CMAKE_CURRENT_BINARY_DIR}/ml.py
    ${CMAKE_CURRENT_BINARY_DIR}/highgui.py)

foreach(m ${LOC_CV} ${LOC_ML} ${LOC_HIGHGUI})
    string(REPLACE "\$(OutDir)" "\${CMAKE_INSTALL_CONFIG_NAME}" m1 ${m})
    set(pyopencv_files ${pyopencv_files} ${m1})
endforeach()


# TODO: need to compile *.py files
if(WIN32)
    install(FILES ${pyopencv_files} DESTINATION
        "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages/opencv"
        COMPONENT main)
else()
    install(FILES ${pyopencv_files} DESTINATION ${PYTHON_PLUGIN_INSTALL_PATH} COMPONENT main)
endif()

