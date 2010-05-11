# -------------------------------------------------------------------------
#  CMake file for Octave samples. See root CMakeLists.txt
# -------------------------------------------------------------------------

file(GLOB OCTAVE_SAMPLES *.m)

if(NOT WIN32)
install(FILES ${OCTAVE_SAMPLES}
        DESTINATION share/opencv/samples/octave
        PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif()

