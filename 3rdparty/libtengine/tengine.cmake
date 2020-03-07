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
#         qli@openailab.com
#         sqfu@openailab.com
#

SET(TENGINE_VERSION "tengine-opencv")
SET(OCV_TENGINE_DSTDIRECTORY ${OpenCV_BINARY_DIR}/3rdparty/libtengine)
SET(DEFAULT_OPENCV_TENGINE_SOURCE_PATH ${OCV_TENGINE_DSTDIRECTORY}/Tengine-${TENGINE_VERSION})

IF(EXISTS ${DEFAULT_OPENCV_TENGINE_SOURCE_PATH})
	MESSAGE(STATUS "Tengine is exist already  .")

	SET(Tengine_FOUND ON)
	set(BUILD_TENGINE ON)
ELSE()
	SET(OCV_TENGINE_FILENAME "${TENGINE_VERSION}.zip")#name2
	SET(OCV_TENGINE_URL "https://github.com/OAID/Tengine/archive/") #url2
	SET(tengine_md5sum 9c80d91dc8413911522ec80cde013ae2) #md5sum2

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

if (BUILD_TENGINE)
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

	SET(DEFAULT_OPENCV_TENGINE_SOURCE_PATH ${OCV_TENGINE_DSTDIRECTORY}/Tengine-${TENGINE_VERSION})
	set(BUILT_IN_OPENCV ON) ## set for tengine compile discern .
	set(Tengine_INCLUDE_DIR  ${DEFAULT_OPENCV_TENGINE_SOURCE_PATH}/core/include)
	set(Tengine_LIB   ${CMAKE_BINARY_DIR}/lib/${ANDROID_ABI}/libtengine.a)
	if ( IS_DIRECTORY ${DEFAULT_OPENCV_TENGINE_SOURCE_PATH})
		add_subdirectory("${DEFAULT_OPENCV_TENGINE_SOURCE_PATH}" ${OCV_TENGINE_DSTDIRECTORY}/build)
	endif()
endif()


