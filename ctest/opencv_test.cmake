#
# Include CTest Ext module
#

if(NOT CTEST_EXT_INCLUDED)
    function(download_ctest_ext)
        message("Update CTest Extension module")

        find_package(Git QUIET)

        set(repo_url "https://github.com/jet47/ctest-ext.git")
        set(repo_dir "${CMAKE_CURRENT_LIST_DIR}/ctest-ext")
        set(tmp_dir "${CMAKE_CURRENT_LIST_DIR}/ctest-ext-tmp")

        if(NOT EXISTS "${repo_dir}")
            set(CTEST_CHECKOUT_COMMAND "${GIT_EXECUTABLE} clone ${repo_url} ${repo_dir}")
        endif()
        set(CTEST_UPDATE_COMMAND "${GIT_EXECUTABLE}")

        ctest_start("CTestExt" "${repo_dir}" "${tmp_dir}")
        ctest_update(SOURCE "${repo_dir}")

        file(REMOVE_RECURSE "${tmp_dir}")

        set(CTEST_EXT_MODULE_PATH "${repo_dir}/ctest_ext.cmake" PARENT_SCOPE)
    endfunction()

    if(NOT DEFINED CTEST_EXT_MODULE_PATH)
        if(DEFINED ENV{CTEST_EXT_MODULE_PATH} AND EXISTS "$ENV{CTEST_EXT_MODULE_PATH}")
            set(CTEST_EXT_MODULE_PATH "$ENV{CTEST_EXT_MODULE_PATH}")
        elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/ctest_ext.cmake")
            set(CTEST_EXT_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/ctest_ext.cmake")
        else()
            download_ctest_ext()
        endif()
    endif()

    include("${CTEST_EXT_MODULE_PATH}")
endif()

#
# Repository settings
#

set_ifndef(CTEST_PROJECT_GIT_URL    "https://github.com/jet47/opencv.git")
set_ifndef(CTEST_PROJECT_GIT_BRANCH "ctest-dashboard-testing")

set_ifndef(CTEST_WITH_UPDATE        TRUE)

#
# Initialize testing
#

ctest_ext_init()

#
# Include main testing script from project source tree
#

include("${CTEST_SOURCE_DIRECTORY}/ctest/opencv_test_internal.cmake")
