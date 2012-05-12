project(opencv_world)

string(REPLACE "/INCREMENTAL:NO" "/INCREMENTAL:YES" CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
string(REPLACE "/INCREMENTAL:NO" "/INCREMENTAL:YES" CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE}")
string(REPLACE "/INCREMENTAL:NO" "/INCREMENTAL:YES" CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

set(the_description "All the selected OpenCV modules in a single binary")

set(module_list )

ocv_add_module(world opencv_core OPTIONAL opencv_flann opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_photo opencv_objdetect opencv_legacy opencv_nonfree)

set(the_module opencv_world)

ocv_glob_module_sources()
ocv_module_include_directories()

add_library(${the_module} ${OPENCV_MODULE_TYPE} ${OPENCV_MODULE_${the_module}_HEADERS} ${OPENCV_MODULE_${the_module}_SOURCES})

if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_module} PROPERTIES FOLDER "modules")
endif()

set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${OPENCV_DLLVERSION}"
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR lib
)

# For dynamic link numbering convenions
if(NOT ANDROID)
    # Android SDK build scripts can include only .so files into final .apk
    # As result we should not set version properties for Android
    set_target_properties(${the_module} PROPERTIES
        VERSION ${OPENCV_VERSION}
        SOVERSION ${OPENCV_SOVERSION}
    )
endif()

if(BUILD_SHARED_LIBS)
    if(MSVC)
        set_target_properties(${the_module} PROPERTIES DEFINE_SYMBOL CVAPI_EXPORTS)
    else()
        add_definitions(-DCVAPI_EXPORTS)
    endif()
endif()

if(MSVC)
    if(CMAKE_CROSSCOMPILING)
        set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:secchk")
    endif()
    set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libc /DEBUG")
endif()

set(objpath0 "${OpenCV_BINARY_DIR}/modules/<MODULE_NAME>")

if(CMAKE_GENERATOR MATCHES "^Visual.*$")
    set(have_cfg 1)
    set(objpath0 "${objpath0}/opencv_<MODULE_NAME>.dir/<CONFIGURATION>/<SRC_NAME_WE>.obj")
elseif (CMAKE_GENERATOR MATCHES Xcode)
    set(have_cfg 1)
    set(objpath0 "${objpath0}/OpenCV.build/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)/opencv_<MODULE_NAME>.build/Objects-normal/$(CURRENT_ARCH)/<SRC_NAME_WE>.o")
else()
    set(have_cfg 0)
    set(objpath0 "${objpath0}/CMakeFiles/opencv_<MODULE_NAME>.dir/src/<SRC_NAME>.o")
endif()

set(objlist)

foreach(m ${OPENCV_MODULES_BUILD})
    if(m STREQUAL ${the_module})
    elseif(m STREQUAL "opencv_python")
    elseif(m STREQUAL "opencv_ts")
    else()
        add_dependencies(${the_module} ${m})
        string(REGEX REPLACE "^opencv_" "" bare_m "${m}")
        string(REGEX REPLACE "<MODULE_NAME>" "${bare_m}" objpath1 "${objpath0}")
        if(m STREQUAL "opencv_highgui")
            target_link_libraries(${the_module} ${HIGHGUI_LIBRARIES} ${GRFMT_LIBS})
        endif()
        foreach(d ${OPENCV_MODULE_${m}_DEPS_EXT})
            target_link_libraries(${the_module} ${d})
        endforeach()
        set(modobjlist)
        foreach(src ${OPENCV_MODULE_${m}_SOURCES})
            get_filename_component(srcname ${src} NAME)
            if(srcname MATCHES ".*\\.[cm].*")
                get_filename_component(srcname_we ${srcname} NAME_WE)
                string(REGEX REPLACE <SRC_NAME_WE> "${srcname_we}" objpath2 "${objpath1}")
                string(REGEX REPLACE <SRC_NAME> "${srcname}" objpath3 "${objpath2}")
                set(modobjlist "${modobjlist} \"${objpath3}\"")
            endif()
        endforeach()
        set(objlist "${objlist} ${modobjlist}")
    endif()
endforeach()

if(have_cfg)
    string(REGEX REPLACE "<CONFIGURATION>" "Debug" objlist_dbg "${objlist}")
    string(REGEX REPLACE "<CONFIGURATION>" "Release" objlist_rls "${objlist}")
    set_target_properties(${the_module} PROPERTIES
        LINK_FLAGS_DEBUG ${objlist_dbg}
        LINK_FLAGS_RELEASE ${objlist_rls})
else()
    set_target_properties(${the_module} PROPERTIES
        LINK_FLAGS ${objlist})
endif()        

install(TARGETS ${the_module}
    RUNTIME DESTINATION bin COMPONENT main
    LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
    ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
)

# only "public" headers need to be installed
if(OPENCV_MODULE_${the_module}_HEADERS AND OPENCV_MODULES_PUBLIC MATCHES "(^|;)${the_module}(;|$)")
    foreach(hdr ${OPENCV_MODULE_${the_module}_HEADERS})
        string(REGEX REPLACE "^.*opencv2/" "opencv2/" hdr2 "${hdr}")
        if(hdr2 MATCHES "^(opencv2/.*)/[^/]+.h(..)?$")
            install(FILES ${hdr} DESTINATION "${OPENCV_INCLUDE_PREFIX}/${CMAKE_MATCH_1}" COMPONENT main)
        endif()
    endforeach()
endif()

ocv_add_precompiled_headers(${the_module})
