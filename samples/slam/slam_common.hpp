// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

// Helpers shared by the slam.cpp and visual_odometry.cpp samples.

#pragma once

#include <opencv2/ptcloud.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

inline vector<String> collectImages(const String& imagesDir)
{
    vector<String> allFiles;
    try
    {
        glob(imagesDir, allFiles, false);
    }
    catch (const Exception& e)
    {
        cerr << "glob failed on " << imagesDir << ": " << e.what() << endl;
        return {};
    }

    vector<String> imageFiles;
    imageFiles.reserve(allFiles.size());
    for (const String& file : allFiles)
        if (haveImageReader(file))
            imageFiles.push_back(file);

    sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

inline Mat parseDistCoeffs(const String& text)
{
    stringstream stream(text);
    vector<double> coeffs;
    String token;
    while (getline(stream, token, ','))
    {
        const size_t begin = token.find_first_not_of(" \t");
        const size_t end   = token.find_last_not_of(" \t");
        if (begin == String::npos) continue;
        coeffs.push_back(stod(token.substr(begin, end - begin + 1)));
    }
    return coeffs.empty() ? Mat() : Mat(coeffs, true).reshape(1, 1);
}

// `trajectory` is passed in rather than read from `vo` because callers disagree on which
// trajectory is "final": samples that run bundle adjustment/loop closure want
// getCorrectedTrajectory(), samples that only track want the raw getTrajectory().
inline bool writeColmapFiles(const Ptr<slam::VisualOdometry>& vo,
                             const vector<Matx44d>& trajectory,
                             const Matx33d& K, const Mat& distCoeffs, Size imageSize,
                             const vector<String>& poseImageNames,
                             const String& outputFolder)
{
    if (!utils::fs::createDirectories(outputFolder))
    {
        cerr << "cannot create output directory " << outputFolder << endl;
        return false;
    }

    // cameras.txt
    {
        ofstream file(utils::fs::join(outputFolder, "cameras.txt").c_str());
        if (!file.is_open()) { cerr << "cannot write cameras.txt" << endl; return false; }
        file << "# Camera list with one line of data per camera:\n"
             << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
             << "# Number of cameras: 1\n";
        file << setprecision(9);
        const bool hasDist = !distCoeffs.empty();
        file << "1 " << (hasDist ? "FULL_OPENCV" : "PINHOLE") << " "
             << imageSize.width << " " << imageSize.height << " "
             << K(0, 0) << " " << K(1, 1) << " " << K(0, 2) << " " << K(1, 2);
        if (hasDist)
            // FULL_OPENCV needs exactly 8 params (k1,k2,p1,p2,k3,k4,k5,k6); pad any shorter --dist with zeros.
            for (int i = 0; i < 8; ++i)
                file << " " << (i < (int)distCoeffs.total() ? distCoeffs.at<double>(i) : 0.0);
        file << "\n";
    }

    // images.txt; IDs are 1-based because COLMAP reserves id 0 as "invalid".
    {
        ofstream file(utils::fs::join(outputFolder, "images.txt").c_str());
        if (!file.is_open()) { cerr << "cannot write images.txt" << endl; return false; }
        file << "# Image list with two lines of data per image:\n"
             << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
             << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
             << "# Number of images: " << trajectory.size() << ", mean observations per image: 0\n";
        file << setprecision(9);
        for (size_t i = 0; i < trajectory.size(); ++i)
        {
            const Matx44d& poseCw = trajectory[i];
            const Quatd q = Quatd::createFromRotMat(poseCw.get_minor<3, 3>(0, 0));
            // COLMAP expects a name relative to the image directory, not a full path.
            const String name = i < poseImageNames.size()
                              ? poseImageNames[i].substr(poseImageNames[i].find_last_of("/\\") + 1)
                              : format("pose_%zu", i);
            file << (i + 1) << " " << q.w << " " << q.x << " " << q.y << " " << q.z << " "
                 << poseCw(0, 3) << " " << poseCw(1, 3) << " " << poseCw(2, 3)
                 << " 1 " << name << "\n"
                 // Second (POINTS2D) line left blank: 2D-3D correspondences are only known for
                 // keyframes, but this file enumerates every tracked frame, so there's no
                 // reliable per-frame keypoint list to emit here.
                 << "\n";
        }
    }

    // points3D.txt; IDs are 1-based for the same reason as images.txt.
    {
        ofstream file(utils::fs::join(outputFolder, "points3D.txt").c_str());
        if (!file.is_open()) { cerr << "cannot write points3D.txt" << endl; return false; }
        file << "# 3D point list with one line of data per point:\n"
             << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
        file << setprecision(9);
        for (const slam::MapPoint* point : vo->getMap().mapPoints())
        {
            if (!point || point->bad) continue;
            // Color is unavailable (no per-point sampling) and TRACK is left empty: observations
            // are keyed by KeyFrame*, which doesn't map to the per-frame IMAGE_IDs above.
            file << (point->id + 1) << " "
                 << point->pos.x << " " << point->pos.y << " " << point->pos.z
                 << " 128 128 128 -1\n";
        }
    }

    return true;
}
