if(WIN32)
    find_path( CSTRIPES_LIB_DIR
               NAMES "С=.lib"
               DOC "The path to C= lib and dll")
    if(CSTRIPES_LIB_DIR)
        ocv_include_directories("${CSTRIPES_LIB_DIR}/..")
        link_directories("${CSTRIPES_LIB_DIR}")
        set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} "C=")
        set(HAVE_CSTRIPES 1)
    endif()
endif()
