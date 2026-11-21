// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_CONFIG_JSON_HPP
#define OPENCV_VLM_CONFIG_JSON_HPP

#include "opencv2/core.hpp"

namespace cv { namespace vlm {

FileStorage openJsonConfigOrThrow(const String& path);

int getInt(const FileNode& node, const String& name, int fallback);
float getFloat(const FileNode& node, const String& name, float fallback);
void getVec3f(const FileNode& node, const String& name, Vec3f& value, const Vec3f& fallback);

int getIntWithTextConfigFallback(const FileStorage& config, const String& name, int fallback);

}} // namespace cv::vlm

#endif // OPENCV_VLM_CONFIG_JSON_HPP
