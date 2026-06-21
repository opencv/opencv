// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_SLAM_FRAME_HPP
#define OPENCV_SLAM_FRAME_HPP

#include "../precomp.hpp"

namespace cv {
namespace slam {

/** @brief Per-frame scratch pad. Created per incoming image, never stored in the Map. */
struct Frame
{
    Mat image;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    std::vector<Point2f> undistKpts; //!< undistorted, parallel to keypoints
    Size imageSize;

    Matx44d poseCw = Matx44d::eye();
    std::vector<MapPoint*> mapPoints; //!< parallel to keypoints; nullptr = unmatched
    std::vector<bool> outliers; //!< parallel to keypoints; true = inlier check failed

    // Grid cell size in pixels — dimensions adapt to each image resolution.
    static constexpr int CELL_SIZE_PX = 20;

    int gridRows = 0;
    int gridCols = 0;
    std::unordered_map<int, std::vector<size_t>> grid; //!< cell key -> keypoint indices

    int cellKey(int row, int col) const { return row * gridCols + col; }

    void buildGrid()
    {
        grid.clear();
        if (imageSize.width <= 0 || imageSize.height <= 0) return;
        gridCols = std::max(1, (imageSize.width  + CELL_SIZE_PX - 1) / CELL_SIZE_PX);
        gridRows = std::max(1, (imageSize.height + CELL_SIZE_PX - 1) / CELL_SIZE_PX);
        for (size_t i = 0; i < undistKpts.size(); ++i)
        {
            int col = std::min(gridCols - 1, std::max(0, (int)(undistKpts[i].x / CELL_SIZE_PX)));
            int row = std::min(gridRows - 1, std::max(0, (int)(undistKpts[i].y / CELL_SIZE_PX)));
            grid[cellKey(row, col)].push_back(i);
        }
    }

    std::vector<size_t> getKeypointsInRadius(float x, float y, float r) const
    {
        std::vector<size_t> result;
        if (gridRows <= 0 || gridCols <= 0 || grid.empty()) return result;
        const int colMin = std::max(0, (int)((x - r) / CELL_SIZE_PX));
        const int colMax = std::min(gridCols-1, (int)((x + r) / CELL_SIZE_PX));
        const int rowMin = std::max(0, (int)((y - r) / CELL_SIZE_PX));
        const int rowMax = std::min(gridRows-1, (int)((y + r) / CELL_SIZE_PX));
        const float r2 = r * r;
        for (int row = rowMin; row <= rowMax; ++row)
            for (int col = colMin; col <= colMax; ++col)
            {
                auto it = grid.find(cellKey(row, col));
                if (it == grid.end()) continue;
                for (size_t idx : it->second)
                {
                    float dx = undistKpts[idx].x - x;
                    float dy = undistKpts[idx].y - y;
                    if (dx * dx + dy * dy <= r2)
                        result.push_back(idx);
                }
            }
        return result;
    }
};

}} // namespace cv::slam

#endif // OPENCV_SLAM_FRAME_HPP
