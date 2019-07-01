// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: janstarzy / Planet artificial intelligence GmbH

#ifndef OPENCV_MAPCONVERTER_HPP
#define OPENCV_MAPCONVERTER_HPP

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"

#include <map>

std::map<int, int> Map_to_map_int_and_int(JNIEnv* env, jobject map);
std::map<int, cv::String> Map_to_map_int_and_String(JNIEnv* env, jobject map);

void Copy_map_String_and_String_to_Map(JNIEnv *env, const std::map<std::string, std::string> &src, jobject dst);

#endif //OPENCV_MAPCONVERTER_HPP
