# Main variables:
# OPENNI_LIBRARY and OPENNI_INCLUDES to link OpenCV modules with OpenNI
# HAVE_OPENNI for conditional compilation OpenCV with/without OpenNI

if(NOT "${OPENNI_LIB_DIR}" STREQUAL "${OPENNI_LIB_DIR_INTERNAL}")
    unset(OPENNI_LIBRARY CACHE)
    unset(OPENNI_LIB_DIR CACHE)
endif()

if(NOT "${OPENNI_INCLUDE_DIR}" STREQUAL "${OPENNI_INCLUDE_DIR_INTERNAL}")
    unset(OPENNI_INCLUDES CACHE)
    unset(OPENNI_INCLUDE_DIR CACHE)
endif()

if(NOT "${OPENNI_PRIME_SENSOR_MODULE_BIN_DIR}" STREQUAL "${OPENNI_PRIME_SENSOR_MODULE_BIN_DIR_INTERNAL}")
    unset(OPENNI_PRIME_SENSOR_MODULE CACHE)
    unset(OPENNI_PRIME_SENSOR_MODULE_BIN_DIR CACHE)
endif()

find_file(OPENNI_INCLUDES "XnCppWrapper.h" PATHS $ENV{OPEN_NI_INCLUDE} "/usr/include/ni" "/usr/include/openni" "c:/Program Files/OpenNI/Include" DOC "OpenNI c++ interface header")
find_library(OPENNI_LIBRARY "OpenNI" PATHS $ENV{OPEN_NI_LIB} "/usr/lib" "c:/Program Files/OpenNI/Lib" DOC "OpenNI library")

if(OPENNI_LIBRARY AND OPENNI_INCLUDES)
    set(HAVE_OPENNI TRUE)
    # the check: are PrimeSensor Modules for OpenNI installed
    if(WIN32)
        find_file(OPENNI_PRIME_SENSOR_MODULE "XnCore.dll" 
				PATHS 
					"c:/Program Files/Prime Sense/Sensor/Bin" 
					"c:/Program Files (x86)/Prime Sense/Sensor/Bin"
					"c:/Program Files/PrimeSense/SensorKinect/Bin" 
					"c:/Program Files (x86)/PrimeSense/SensorKinect/Bin"
				DOC "Core library of PrimeSensor Modules for OpenNI")		
    elseif(UNIX OR APPLE)
        find_library(OPENNI_PRIME_SENSOR_MODULE "XnCore" PATHS "/usr/lib" DOC "Core library of PrimeSensor Modules for OpenNI")
    endif()

    if(OPENNI_PRIME_SENSOR_MODULE)
        set(HAVE_OPENNI_PRIME_SENSOR_MODULE TRUE)
    endif()
endif() #if(OPENNI_LIBRARY AND OPENNI_INCLUDES)

get_filename_component(OPENNI_LIB_DIR "${OPENNI_LIBRARY}" PATH CACHE)
get_filename_component(OPENNI_INCLUDE_DIR ${OPENNI_INCLUDES} PATH CACHE)
get_filename_component(OPENNI_PRIME_SENSOR_MODULE_BIN_DIR "${OPENNI_PRIME_SENSOR_MODULE}" PATH CACHE)

if(OPENNI_LIBRARY)
    set(OPENNI_LIB_DIR_INTERNAL "${OPENNI_LIB_DIR}" CACHE INTERNAL "This is the value of the last time OPENNI_LIB_DIR was set successfully." FORCE)
else()
    message( WARNING, " OpenNI library directory (set by OPENNI_LIB_DIR variable) is not found or does not have OpenNI libraries." )
endif()

if(OPENNI_INCLUDES)
    set(OPENNI_INCLUDE_DIR_INTERNAL "${OPENNI_INCLUDE_DIR}" CACHE INTERNAL "This is the value of the last time OPENNI_INCLUDE_DIR was set successfully." FORCE)
else()
    message( WARNING, " OpenNI include directory (set by OPENNI_INCLUDE_DIR variable) is not found or does not have OpenNI include files." )
endif()

if(OPENNI_PRIME_SENSOR_MODULE)
    set(OPENNI_PRIME_SENSOR_MODULE_BIN_DIR_INTERNAL "${OPENNI_PRIME_SENSOR_MODULE_BIN_DIR}" CACHE INTERNAL "This is the value of the last time OPENNI_PRIME_SENSOR_MODULE_BIN_DIR was set successfully." FORCE)
else()
    message( WARNING, " PrimeSensor Module binaries directory (set by OPENNI_PRIME_SENSOR_MODULE_BIN_DIR variable) is not found or does not have PrimeSensor Module binaries." )
endif()

mark_as_advanced(FORCE OPENNI_PRIME_SENSOR_MODULE)
mark_as_advanced(FORCE OPENNI_LIBRARY)
mark_as_advanced(FORCE OPENNI_INCLUDES)
