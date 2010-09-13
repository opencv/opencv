###############################################################################
#
# FindNPP.cmake
#
# CUDA_NPP_LIBRARY_ROOT_DIR   -- Path to the NPP dorectory.
# CUDA_NPP_INCLUDES           -- NPP Include directories.
# CUDA_NPP_LIBRARIES          -- NPP libraries.
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

if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)			
	if (UNIX OR APPLE)
		set(NPP_SUFFIX "32")				
	else()
		set(NPP_SUFFIX "-mt")
	endif()
else(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
	if (UNIX OR APPLE)
		set(NPP_SUFFIX "64")				
	else()
		set(NPP_SUFFIX "-mt-x64")			
	endif()
endif(${CMAKE_SIZEOF_VOID_P} EQUAL 4)

if(NOT CUDA_NPP_LIBRARY_ROOT_DIR)
	find_path(CUDA_NPP_LIBRARY_ROOT_DIR common/npp/include/npp.h DOC "NPP root directory." NO_DEFAULT_PATH)		  
endif (NOT CUDA_NPP_LIBRARY_ROOT_DIR)

# Search includes in our own paths.
find_path(CUDA_NPP_INCLUDES npp.h PATHS "${CUDA_NPP_LIBRARY_ROOT_DIR}/common/npp/include" NO_DEFAULT_PATH)
# Search default search paths, after we search our own set of paths.
find_path(CUDA_NPP_INCLUDES device_functions.h)
mark_as_advanced(CUDA_NPP_INCLUDES)

# Find NPP library
find_library(CUDA_NPP_LIBRARIES
	NAMES npp${NPP_SUFFIX} libnpp${NPP_SUFFIX}
	PATHS "${CUDA_NPP_LIBRARY_ROOT_DIR}/common/lib"    
	DOC "NPP library"
	NO_DEFAULT_PATH
	)

# Search default search paths, after we search our own set of paths.
find_library(CUDA_NPP_LIBRARIES NAMES npp${NPP_SUFFIX} libnpp${NPP_SUFFIX} DOC "NPP library")
mark_as_advanced(CUDA_NPP_LIBRARIES)

if(NOT EXISTS ${CUDA_NPP_LIBRARIES} OR NOT EXISTS ${CUDA_NPP_INCLUDES}/npp.h)
	if(NPP_FIND_REQUIRED)
		message(FATAL_ERROR "NPP headers/libraries are not found. Specify CUDA_NPP_LIBRARY_ROOT_DIR.")
	elseif(NOT CUDA_FIND_QUIETLY)
		message("NPP headers/libraries are not found or CUDA_NPP_LIBRARY_ROOT_DIR not specified.")
	endif()	
	
	set(CUDA_FOUND FALSE)
	unset(CUDA_NPP_INCLUDES CACHE)
	unset(CUDA_NPP_LIBRARIES CACHE)
else()
	
	if(APPLE)
		# We need to add the path to cudart to the linker using rpath, since the
		# library name for the cuda libraries is prepended with @rpath.
		get_filename_component(_cuda_path_to_npp "${CUDA_NPP_LIBRARIES}" PATH)
		if(_cuda_path_to_npp)
			list(APPEND CUDA_NPP_LIBRARIES -Wl,-rpath "-Wl,${_cuda_path_to_npp}")
		endif()
	endif()	
	
	set(CUDA_NPP_FOUND TRUE)
	set(CUDA_NPP_LIBRARY_ROOT_DIR_INTERNAL "${CUDA_NPP_LIBRARY_ROOT_DIR}" CACHE INTERNAL "This is the value of the last time CUDA_NPP_LIBRARY_ROOT_DIR was set successfully." FORCE)
endif()

