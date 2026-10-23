// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "config_json.hpp"

namespace cv { namespace vlm {

FileStorage openJsonConfigOrThrow(const String& path)
{
    FileStorage fs(path, FileStorage::READ | FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "vlm: could not open config file: " + path);
    return fs;
}

int getInt(const FileNode& node, const String& name, int fallback)
{
    FileNode child = node[name];
    return child.empty() ? fallback : (int)child;
}

float getFloat(const FileNode& node, const String& name, float fallback)
{
    FileNode child = node[name];
    return child.empty() ? fallback : (float)child;
}

void getVec3f(const FileNode& node, const String& name, Vec3f& value, const Vec3f& fallback)
{
    FileNode child = node[name];
    if (child.empty())
    {
        value = fallback;
        return;
    }
    std::vector<float> v;
    child >> v;
    CV_CheckEQ((int)v.size(), 3, "vlm: expected a 3-element array");
    value = Vec3f(v[0], v[1], v[2]);
}

int getIntWithTextConfigFallback(const FileStorage& config, const String& name, int fallback)
{
    FileNode node = config[name];
    if (!node.empty())
        return (int)node;
    FileNode textConfig = config["text_config"];
    if (!textConfig.empty() && !textConfig[name].empty())
        return (int)textConfig[name];
    return fallback;
}

}} // namespace cv::vlm
