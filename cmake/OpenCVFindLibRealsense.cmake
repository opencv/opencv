# Main variables:
# LIBREALSENSE_LIBRARIES and LIBREALSENSE_INCLUDE to link Intel librealsense modules
# HAVE_LIBREALSENSE for conditional compilation OpenCV with/without librealsense

find_path(LIBREALSENSE_INCLUDE_DIR "librealsense2/rs.hpp" PATHS "$ENV{LIBREALSENSE_INCLUDE}" DOC "Path to librealsense interface headers")
find_library(LIBREALSENSE_LIBRARIES "realsense2" PATHS "$ENV{LIBREALSENSE_LIB}" DOC "Path to librealsense interface libraries")

if(LIBREALSENSE_INCLUDE_DIR AND LIBREALSENSE_LIBRARIES)
    set(HAVE_LIBREALSENSE TRUE)
else()
    set(HAVE_LIBREALSENSE FALSE)
    message( WARNING, " librealsense include directory (set by LIBREALSENSE_INCLUDE_DIR variable) is not found or does not have librealsense include files." )
endif() #if(LIBREALSENSE_INCLUDE_DIR AND LIBREALSENSE_LIBRARIES)

mark_as_advanced(FORCE LIBREALSENSE_LIBRARIES LIBREALSENSE_INCLUDE_DIR)