// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_COLOR_HASH_TSDF_FUNCTIONS_HPP
#define OPENCV_3D_COLOR_HASH_TSDF_FUNCTIONS_HPP

#include "hash_tsdf_functions.hpp"
#include "color_tsdf_functions.hpp"

namespace cv
{

// ColorHashTSDF uses the same voxel layout as ColorTSDF (TSDF + running-average RGB),
// but stores voxels in spatially hashed volume units like HashTSDF.
// It is a CPU-only implementation; the Volume::Impl interface falls back to CPU
// regardless of the OpenCL availability (same as ColorTSDF).

void integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose,
    int& lastVolIndex, const int frameId, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms,
    InputOutputArray _volUnitsData, VolumeUnitIndexes& volumeUnits);

void raycastColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose,
    int height, int width, InputArray intr, const int volumeUnitDegree,
    InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    OutputArray _points, OutputArray _normals, OutputArray _colors);

void fetchNormalsFromColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const int volumeUnitDegree, InputArray _points, OutputArray _normals);

void fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const int volumeUnitDegree, OutputArray _points, OutputArray _normals, OutputArray _colors);

} // namespace cv

#endif // OPENCV_3D_COLOR_HASH_TSDF_FUNCTIONS_HPP
