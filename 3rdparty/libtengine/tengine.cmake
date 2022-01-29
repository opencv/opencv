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

SET(TENGINE_COMMIT_VERSION "e89cf8870de2ff0a80cfe626c0b52b2a16fb302e")
SET(OCV_TENGINE_DIR "${OpenCV_BINARY_DIR}/3rdparty/libtengine")
SET(OCV_TENGINE_SOURCE_PATH "${OCV_TENGINE_DIR}/Tengine-${TENGINE_COMMIT_VERSION}")

IF(EXISTS "${OCV_TENGINE_SOURCE_PATH}")
	MESSAGE(STATUS "Tengine is exist already at: ${OCV_TENGINE_SOURCE_PATH}")

	SET(Tengine_FOUND ON)
	SET(BUILD_TENGINE ON)
ELSE()
	SET(OCV_TENGINE_FILENAME "${TENGINE_COMMIT_VERSION}.zip")#name
	# Github
	SET(OCV_TENGINE_GITHUB_URL "https://github.com/OAID/Tengine/archive/") #url
	SET(tengine_GITHUB_md5sum 23f61ebb1dd419f1207d8876496289c5) #md5sum
	# Gitcode
	SET(OCV_TENGINE_GITCODE_URL "https://gitcode.net/OAID/Tengine/-/archive/")
	SET(tengine_GITCODE_md5sum 1b5908632b557275cd6e85b0c03f9690)
	# Custom
	SET(OCV_TENGINE_CUSTOM_URL "https://${OPENCV_MIRROR_CUSTOM}/OAID/Tengine/-/archive/")
	SET(tengine_CUSTOM_md5sum "")
	# if md5sum of custom source is empty, switching back to github by default
	if(NOT tengine_CUSTOM_md5sum)
		message(STATUS "Tengine: Need to assign tengine_CUSTOM_md5sum before downloading from custom source. Switching back to Github.")
		SET(OCV_TENGINE_CUSTOM_URL "${OCV_TENGINE_GITHUB_URL}")
		SET(tengine_CUSTOM_md5sum "${tengine_GITHUB_md5sum}")
	endif()

	ocv_download(FILENAME ${OCV_TENGINE_FILENAME}
						HASH ${tengine_${OPENCV_DOWNLOAD_HOST}_md5sum}
						URL
						"${OPENCV_TENGINE_URL}"
						"$ENV{OPENCV_TENGINE_URL}"
						"${OCV_TENGINE_${OPENCV_DOWNLOAD_HOST}_URL}"
						DESTINATION_DIR "${OCV_TENGINE_DIR}"
						ID TENGINE
						STATUS res
						UNPACK RELATIVE_URL)

	if (NOT res)
		MESSAGE(STATUS "TENGINE DOWNLOAD FAILED. Turning Tengine_FOUND off.")
		SET(Tengine_FOUND OFF)
	else ()
		MESSAGE(STATUS "TENGINE DOWNLOAD success . ")

		SET(Tengine_FOUND ON)
		SET(BUILD_TENGINE ON)
	endif()
ENDIF()

if(BUILD_TENGINE)
	SET(HAVE_TENGINE 1)

	if(NOT ANDROID)
		# linux system
		if(CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
			   SET(TENGINE_TOOLCHAIN_FLAG "-march=armv7-a")
		elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64) ## AARCH64
			   SET(TENGINE_TOOLCHAIN_FLAG "-march=armv8-a")
		endif()
	endif()

	SET(BUILT_IN_OPENCV ON) ## set for tengine compile discern .
	SET(Tengine_INCLUDE_DIR  "${OCV_TENGINE_SOURCE_PATH}/include" CACHE INTERNAL "")
	if(EXISTS "${OCV_TENGINE_SOURCE_PATH}/CMakeLists.txt")
		add_subdirectory("${OCV_TENGINE_SOURCE_PATH}" "${OCV_TENGINE_DIR}/build")
	else()
		message(WARNING "TENGINE: Missing 'CMakeLists.txt' in source code package: ${OCV_TENGINE_SOURCE_PATH}")
	endif()
	SET(Tengine_LIB "tengine" CACHE INTERNAL "")
endif()
