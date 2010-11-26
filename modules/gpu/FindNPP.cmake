###############################################################################
#
# FindNPP.cmake
#
# CUDA_NPP_LIBRARY_ROOT_DIR   -- Path to the NPP dorectory.
# CUDA_NPP_INCLUDES           -- NPP Include directories.
# CUDA_NPP_LIBRARIES          -- NPP libraries.
# NPP_VERSION                 -- NPP version in format "major.minor.build".
#
# If not found automatically, please set CUDA_NPP_LIBRARY_ROOT_DIR or
# set enviroment varivabe $CUDA_NPP_ROOT
#
# Author: Anatoly Baksheev, Itseez Ltd.
# 
# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

# We need to have at least this version to support the VERSION_LESS argument to 'if' (2.6.2) and unset (2.6.3)
cmake_policy(PUSH)
cmake_minimum_required(VERSION 2.6.3)
cmake_policy(POP)

if(NOT "${CUDA_NPP_LIBRARY_ROOT_DIR}" STREQUAL "${CUDA_NPP_LIBRARY_ROOT_DIR_INTERNAL}")
	unset(CUDA_NPP_INCLUDES CACHE)
	unset(CUDA_NPP_LIBRARIES CACHE)  
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)			
	if (UNIX OR APPLE)
		set(NPP_SUFFIX "32")				
	else()
		set(NPP_SUFFIX "-mt")
	endif()
else(CMAKE_SIZEOF_VOID_P EQUAL 4)
	if (UNIX OR APPLE)
		set(NPP_SUFFIX "64")				
	else()
		set(NPP_SUFFIX "-mt-x64")			
	endif()
endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

if(NOT CUDA_NPP_LIBRARY_ROOT_DIR OR CUDA_NPP_LIBRARY_ROOT_DIR STREQUAL "")
	unset(CUDA_NPP_LIBRARY_ROOT_DIR CACHE)	
	find_path(CUDA_NPP_LIBRARY_ROOT_DIR "common/npp/include/npp.h" PATHS ENV CUDA_NPP_ROOT DOC "NPP root directory.")	
	MESSAGE(STATUS "NPP root directory: " ${CUDA_NPP_LIBRARY_ROOT_DIR})
endif()

# Search includes in our own paths.
find_path(CUDA_NPP_INCLUDES npp.h PATHS "${CUDA_NPP_LIBRARY_ROOT_DIR}/common/npp/include")
# Search default search paths, after we search our own set of paths.
find_path(CUDA_NPP_INCLUDES device_functions.h)
mark_as_advanced(CUDA_NPP_INCLUDES)

# Find NPP library
find_library(CUDA_NPP_LIBRARIES
	NAMES "npp" "npp${NPP_SUFFIX}" "libnpp${NPP_SUFFIX}"
	PATHS "${CUDA_NPP_LIBRARY_ROOT_DIR}"    
	PATH_SUFFIXES "common/lib" "common/npp/lib"
	DOC "NPP library"	
	)	

# Search default search paths, after we search our own set of paths.
find_library(CUDA_NPP_LIBRARIES NAMES npp${NPP_SUFFIX} libnpp${NPP_SUFFIX} DOC "NPP library")
mark_as_advanced(CUDA_NPP_LIBRARIES)

if(EXISTS ${CUDA_NPP_INCLUDES}/nppversion.h)
	file( STRINGS ${CUDA_NPP_INCLUDES}/nppversion.h npp_major REGEX "#define NPP_VERSION_MAJOR.*")
	file( STRINGS ${CUDA_NPP_INCLUDES}/nppversion.h npp_minor REGEX "#define NPP_VERSION_MINOR.*")
	file( STRINGS ${CUDA_NPP_INCLUDES}/nppversion.h npp_build REGEX "#define NPP_VERSION_BUILD.*")

	string( REGEX REPLACE "#define NPP_VERSION_MAJOR[ \t]+|//.*" "" npp_major ${npp_major})
	string( REGEX REPLACE "#define NPP_VERSION_MINOR[ \t]+|//.*" "" npp_minor ${npp_minor})
	string( REGEX REPLACE "#define NPP_VERSION_BUILD[ \t]+|//.*" "" npp_build ${npp_build})

	string( REGEX MATCH "[0-9]+" npp_major ${npp_major} ) 
	string( REGEX MATCH "[0-9]+" npp_minor ${npp_minor} ) 
	string( REGEX MATCH "[0-9]+" npp_build ${npp_build} ) 	
	set( NPP_VERSION "${npp_major}.${npp_minor}.${npp_build}")	
endif()

if(NOT EXISTS ${CUDA_NPP_LIBRARIES} OR NOT EXISTS ${CUDA_NPP_INCLUDES}/npp.h)
	set(CUDA_NPP_FOUND FALSE)	
	message(FATAL_ERROR "NPP headers/libraries are not found. Please specify CUDA_NPP_LIBRARY_ROOT_DIR in CMake or set $NPP_ROOT_DIR.")	
endif()

include( FindPackageHandleStandardArgs ) 
find_package_handle_standard_args( NPP 
	REQUIRED_VARS 
		CUDA_NPP_INCLUDES 
		CUDA_NPP_LIBRARIES 
	# Don't remove!!! Please update your CMake.
	VERSION_VAR			
		NPP_VERSION)

if(APPLE)
	# We need to add the path to cudart to the linker using rpath, since the library name for the cuda libraries is prepended with @rpath.
	get_filename_component(_cuda_path_to_npp "${CUDA_NPP_LIBRARIES}" PATH)
	if(_cuda_path_to_npp)
		list(APPEND CUDA_NPP_LIBRARIES -Wl,-rpath "-Wl,${_cuda_path_to_npp}")
	endif()
endif()

set(CUDA_NPP_FOUND TRUE)
set(CUDA_NPP_LIBRARY_ROOT_DIR_INTERNAL "${CUDA_NPP_LIBRARY_ROOT_DIR}" CACHE INTERNAL "This is the value of the last time CUDA_NPP_LIBRARY_ROOT_DIR was set successfully." FORCE)


