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
#  Path for Tengine modules
# ----------------------------------------------------------------------------
set(OPENCV_LIBTENGINE_ROOT_DIR "" CACHE PATH "Where to look for additional OpenCV modules (can be ;-separated list of paths)")

IF(OPENCV_LIBTENGINE_ROOT_DIR)

	MESSAGE(STATUS "TENGINE:--  Set tengine lib dir by user ")

	SET(Tengine_FOUND ON)
	set(BUILD_TENGINE OFF)

	SET(Tengine_INCLUDE_DIR   	${OPENCV_LIBTENGINE_ROOT_DIR}/include)
	SET(Tengine_LIB 	${OPENCV_LIBTENGINE_ROOT_DIR}/lib/libtengine.a)

ELSE()

	MESSAGE(STATUS "TENGINE:--  Auto download Tengine source code. ")
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

MESSAGE(STATUS "Tengine include is:" ${Tengine_INCLUDE_DIR})
MESSAGE(STATUS "Tengine library is:" ${Tengine_LIB})

MARK_AS_ADVANCED(
	Tengine_INCLUDE_DIR
	Tengine_LIB
	Tengine
)
