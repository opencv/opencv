# -------------------------------------------------------------------------
#  CMake file for Python samples. See root CMakeLists.txt
# -------------------------------------------------------------------------

file(GLOB PYTHON_SAMPLES *.py)
if(NOT WIN32)
install(FILES ${PYTHON_SAMPLES}
        DESTINATION share/opencv/samples/python
        PERMISSIONS OWNER_READ OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
endif()

