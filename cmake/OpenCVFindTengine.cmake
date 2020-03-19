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

# ----------------------------------------------------------------------------
#  Path for Tengine binaries
# ----------------------------------------------------------------------------
set(OPENCV_LIBTENGINE_ROOT_DIR "" CACHE PATH "Path to TENGINE binaries installation")

IF(OPENCV_LIBTENGINE_ROOT_DIR AND NOT BUILD_TENGINE)

	MESSAGE(STATUS "TENGINE:--  Use binaries at ${OPENCV_LIBTENGINE_ROOT_DIR}")

	SET(Tengine_FOUND ON)
	set(BUILD_TENGINE OFF)

	SET(Tengine_INCLUDE_DIR "${OPENCV_LIBTENGINE_ROOT_DIR}/include" CACHE PATH "TENGINE include dir")
	SET(Tengine_LIB "${OPENCV_LIBTENGINE_ROOT_DIR}/lib/libtengine.a" CACHE PATH "TENGINE library dir")

ELSE()
	IF(ANDROID)
		IF(OPENCV_TENGINE_FORCE_ANDROID)
			# nothing, use Android
		ELSEIF(OPENCV_TENGINE_SKIP_ANDROID)
			set(Tengine_FOUND OFF)
			set(HAVE_TENGINE FALSE)
			return()
		ELSEIF(NOT DEFINED ANDROID_NDK_REVISION)
			MESSAGE(STATUS "Android NDK version Tengine not support: ANDROID_NDK_REVISION is not defined")
			set(Tengine_FOUND OFF)
			set(HAVE_TENGINE FALSE)
			return()
		ELSEIF(ANDROID_NDK_REVISION VERSION_LESS 14)
			MESSAGE(STATUS "Android NDK version Tengine not support: ANDROID_NDK_REVISION=${ANDROID_NDK_REVISION}")
			set(Tengine_FOUND OFF)
			set(HAVE_TENGINE FALSE)
			return()
		ENDIF()
	ENDIF()
	MESSAGE(STATUS "TENGINE:--  Build Tengine from source code. ")
	include("${OpenCV_SOURCE_DIR}/3rdparty/libtengine/tengine.cmake")
ENDIF()

IF(NOT Tengine_LIB)
	SET(Tengine_FOUND OFF)
	MESSAGE(STATUS "#### Could not find Tengine lib. Turning Tengine_FOUND off")
ENDIF()

IF (Tengine_FOUND)
	MESSAGE(STATUS "Found Tengine include: ${Tengine_INCLUDE_DIR}")
	MESSAGE(STATUS "Found Tengine libraries: ${Tengine_LIB}")
	set(HAVE_TENGINE 1)
	set(TENGINE_LIBRARIES    ${Tengine_LIB})
	set(TENGINE_INCLUDE_DIRS    ${Tengine_INCLUDE_DIR})
ENDIF (Tengine_FOUND)

MARK_AS_ADVANCED(
	Tengine_INCLUDE_DIR
	Tengine_LIB
)
