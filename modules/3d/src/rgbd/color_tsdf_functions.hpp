// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#ifndef OPENCV_3D_COLORED_TSDF_FUNCTIONS_HPP
#define OPENCV_3D_COLORED_TSDF_FUNCTIONS_HPP

#include <unordered_set>

#include "utils.hpp"
#include "tsdf_functions.hpp"

#define USE_INTERPOLATION_IN_GETNORMAL 1

namespace cv
{
void integrateColorTsdfVolumeUnit(const VolumeSettings &settings, const Matx44f &cameraPose,
                                  InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _volume);
void integrateColorTsdfVolumeUnit(const VolumeSettings &settings, const Matx44f &volumePose, const Matx44f &cameraPose,
                                  InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _volume);
void raycastColorTsdfVolumeUnit(const VolumeSettings &settings, const Matx44f &cameraPose,
                                int height, int width, InputArray intr,
                                InputArray _volume, OutputArray _points, OutputArray _normals, OutputArray _colors);
void fetchNormalsFromColorTsdfVolumeUnit(const VolumeSettings &settings, InputArray _volume,
                                         InputArray _points, OutputArray _normals);
void fetchPointsNormalsFromColorTsdfVolumeUnit(const VolumeSettings &settings, InputArray _volume,
                                               OutputArray _points, OutputArray _normals);
void fetchPointsNormalsColorsFromColorTsdfVolumeUnit(const VolumeSettings &settings, InputArray _volume,
                                                     OutputArray _points, OutputArray _normals, OutputArray _colors);

} // namespace cv

#endif
