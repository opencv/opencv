# COPYRIGHT
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2020, OPEN AI LAB
# Author: qtang@openailab.com or https://github.com/BUG1989
#

# Default tengine source store directory .
SET(DEFAULT_OPENCV_TENGINE_SOURCE_PATH ${OpenCV_BINARY_DIR}/3rdparty/libtengine/Tengine-1.12.0)

IF( WITH_CUDA OR WITH_OPENCL OR WITH_CUDNN OR WITH_VULKAN OR X86 OR X86_64)
	MESSAGE(STATUS "TENGINE:--  Not support . Turning Tengine_FOUND off.")

	SET(Tengine_FOUND OFF)
ELSE()
	IF(OPENCV_TENGINE_INSTALL_PATH)

		MESSAGE(STATUS "TENGINE:--  Set tengine lib dir by user ")

		SET(Tengine_FOUND ON)
		set(BUILD_TENGINE OFF)

		SET(Tengine_INCLUDE_DIR   	${OPENCV_TENGINE_INSTALL_PATH}/include)
		SET(Tengine_LIB 	${OPENCV_TENGINE_INSTALL_PATH}/lib/libtengine.a)

	ELSEIF(EXISTS ${DEFAULT_OPENCV_TENGINE_SOURCE_PATH})

		MESSAGE(STATUS "TENGINE:--  Alread exist Tengine Source Code , Only need compile .")

		SET(Tengine_FOUND ON)
		set(BUILD_TENGINE ON)

	ELSE()

		MESSAGE(STATUS "TENGINE:--  Auto download Tengine source code. ")

		SET(OCV_TENGINE_DSTDIRECTORY ${OpenCV_BINARY_DIR}/3rdparty/libtengine)

		SET(OCV_TENGINE_FILENAME "v1.12.0.zip")#name2
		SET(OCV_TENGINE_URL "https://github.com/OAID/Tengine/archive/") #url2
		SET(tengine_md5sum d97e5c379281c5aa06e28daf868166a7) #md5sum2

		MESSAGE(STATUS "**** TENGINE DOWNLOAD BEGIN ****")
		ocv_download(FILENAME ${OCV_TENGINE_FILENAME}
					HASH ${tengine_md5sum}
					URL
					"${OPENCV_TENGINE_URL}"
					"$ENV{OPENCV_TENGINE_URL}"
					"${OCV_TENGINE_URL}"
					DESTINATION_DIR ${OCV_TENGINE_DSTDIRECTORY}
					ID TENGINE
					STATUS res
					UNPACK RELATIVE_URL)

		if (NOT res)
			MESSAGE(STATUS "TENGINE DOWNLOAD FAILED .Turning Tengine_FOUND off.")
			SET(Tengine_FOUND OFF)
		else ()
			MESSAGE(STATUS "TENGINE DOWNLOAD success . ")

			SET(Tengine_FOUND ON)
			set(BUILD_TENGINE ON)
		endif()

	ENDIF()

	if( BUILD_TENGINE )
		MESSAGE(STATUS "TENGINE:--  BUILD_TENGINE is ON. ")

		set(HAVE_TENGINE 1)

		# android system 
		if(ANDROID)
			if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
				set(CONFIG_ARCH_ARM32 ON)
			elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
				set(CONFIG_ARCH_ARM64 ON)
			endif()
		endif()

		# linux system
		if(CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
			set(CONFIG_ARCH_ARM32 ON)
		elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64) ## AARCH64
			set(CONFIG_ARCH_ARM64 ON)
		endif()	

		set(BUILT_IN_OPENCV ON) ## set for tengine compile discern .
		set(Tengine_INCLUDE_DIR  ${DEFAULT_OPENCV_TENGINE_SOURCE_PATH}/core/include)
		set(Tengine_LIB   ${CMAKE_BINARY_DIR}/lib/${ANDROID_ABI}/libtengine.a)

		add_subdirectory("${DEFAULT_OPENCV_TENGINE_SOURCE_PATH}")

	else()
	# Not need build tengine . 
		# Tengine_LIB 
		#    Check include files
	#	MESSAGE(STATUS "Tengine_INCLUDE_DIR = ${Tengine_INCLUDE_DIR}")
	#	FIND_PATH(Tengine_INCLUDE_DIR NAMES cpu_device.h tengine_c_api.h tengine_c_compat.h tengine_operations.h PATHS ${Tengine_INCLUDE_DIR} NO_DEFAULT_PATH)
	#	MESSAGE(STATUS "Tengine_INCLUDE_DIR = ${Tengine_INCLUDE_DIR}")

		#    Check libraries
		IF(NOT Tengine_LIB)
			SET(Tengine_FOUND OFF)
			MESSAGE(STATUS "#### Could not find Tengine lib. Turning Tengine_FOUND off")
		ENDIF()
	endif()

	IF (Tengine_FOUND)
		IF (NOT Tengine_FIND_QUIETLY)
			MESSAGE(STATUS "Found Tengine include: ${Tengine_INCLUDE_DIR}")
			MESSAGE(STATUS "Found Tengine libraries: ${Tengine_LIB}")
			set(HAVE_TENGINE 1)
			set(TENGINE_LIBRARIES    ${Tengine_LIB})
			set(TENGINE_INCLUDE_DIRS    ${Tengine_INCLUDE_DIR})
		ENDIF (NOT Tengine_FIND_QUIETLY)
	ENDIF (Tengine_FOUND)

	MESSAGE(STATUS "Tengine include is:" ${Tengine_INCLUDE_DIR})
	MESSAGE(STATUS "Tengine library is:" ${Tengine_LIB})

	MARK_AS_ADVANCED(
	    Tengine_INCLUDE_DIR
		Tengine_LIB
		Tengine
	)
ENDIF()
