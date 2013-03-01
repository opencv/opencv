list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(TBB)

if(NOT TBB_FOUND)
    ocv_disable_gpu_demo(stitching)
endif()

ocv_define_gpu_demo(stitching DESCRIPTION "Stitching" INCLUDE_DIRS ${TBB_INCLUDE_DIRS} LINK_LIBRARIES ${TBB_LIBRARIES})

if(BUILD_demo_stitching AND WIN32)
    string(REPLACE "lib" "bin" TBB_BINARY_DIR ${TBB_LIBRARY_DIR})
    set(ALL_TBB_DLLS "${TBB_BINARY_DIR}/tbb.dll" "${TBB_BINARY_DIR}/tbbmalloc.dll")
    install(FILES ${ALL_TBB_DLLS} DESTINATION ".")

    find_package(Qt4 REQUIRED)
    set(ALL_Qt_DLLS ${QT_BINARY_DIR}/QtCore4.dll ${QT_BINARY_DIR}/QtGui4.dll ${QT_BINARY_DIR}/QtOpenGL4.dll ${QT_BINARY_DIR}/QtTest4.dll)
    install(FILES ${ALL_Qt_DLLS} DESTINATION ".")
endif()
