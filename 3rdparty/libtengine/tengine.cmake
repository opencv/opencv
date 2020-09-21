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
SET(TENGINE_COMMIT_VERSION "7cdf7ee0c2910ecbef3e6f4d2816c0e60723ab02")
SET(OCV_TENGINE_DIR "${OpenCV_BINARY_DIR}/3rdparty/libtengine")
SET(OCV_TENGINE_SOURCE_PATH "${OCV_TENGINE_DIR}/Tengine-${TENGINE_COMMIT_VERSION}")

IF(EXISTS "${OCV_TENGINE_SOURCE_PATH}")
	MESSAGE(STATUS "Tengine is exist already at: ${OCV_TENGINE_SOURCE_PATH}")

	SET(Tengine_FOUND ON)
	SET(BUILD_TENGINE ON)
ELSE()
	SET(OCV_TENGINE_FILENAME "${TENGINE_COMMIT_VERSION}.zip")#name2
	#SET(OCV_TENGINE_URL "https://github.com/OAID/Tengine/archive/") #url2
	SET(OCV_TENGINE_URL "https://github.com/liqi-c/Tengine/archive/") #url2.  Tmp url
	SET(tengine_md5sum 92c63441d043bc6c873a9d4c6d7cf094) #md5sum2

	#MESSAGE(STATUS "**** TENGINE DOWNLOAD BEGIN ****")
	ocv_download(FILENAME ${OCV_TENGINE_FILENAME}
						HASH ${tengine_md5sum}
						URL
						"${OPENCV_TENGINE_URL}"
						"$ENV{OPENCV_TENGINE_URL}"
						"${OCV_TENGINE_URL}"
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
