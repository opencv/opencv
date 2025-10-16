// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_COLOR_HASH_TSDF_FUNCTIONS_HPP
#define OPENCV_3D_COLOR_HASH_TSDF_FUNCTIONS_HPP

#include "hash_tsdf_functions.hpp"
#include "color_tsdf_functions.hpp"

namespace cv
{

// A colored voxel for the Hashing TSDF implementation.
// It's functionally identical to RGBTsdfVoxel but is given a distinct name for clarity.
typedef RGBTsdfVoxel ColorHashTsdfVoxel;


void integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputOutputArray _volUnitsData, VolumeUnitIndexes& volumeUnits);

void raycastColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width, InputArray intr, const int volumeUnitDegree,
    InputArray _volUnitsData, const CustomHashSet& hashTable, OutputArray _points, OutputArray _normals, OutputArray _colors);

void fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const CustomHashSet& hashTable,
    const int volumeUnitDegree, OutputArray _points, OutputArray _normals, OutputArray _colors);


#ifdef HAVE_OPENCL

void ocl_integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId,
    const int volumeUnitDegree, bool enableGrowth, InputArray _depth, InputArray _rgb,
    InputArray _pixNorms, InputOutputArray _volUnitsData, VolumeUnitIndexes& volumeUnits);

void ocl_raycastColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width,
    InputArray intr, const int volumeUnitDegree, InputArray _volUnitsData,
    const CustomHashSet& hashTable, OutputArray _points, OutputArray _normals,
    OutputArray _colors);

void ocl_fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(const VolumeSettings& settings,
    InputArray _volUnits, const CustomHashSet& volumeUnits,
    int volumeUnitDegree, OutputArray points, OutputArray normals, OutputArray colors);

#endif

// Forward declarations
Point3f transformDirection(const Point3f& dir, const Matx44f& mat);
Point3f transformPoint(const Point3f& pt, const Matx44f& mat);

bool integrateColorVolumeUnit(VolumeUnit& unit, const VolumeSettings& settings, const Matx44f& cameraPose,
                            InputArray depth, InputArray rgb, InputArray pixNorms,
                            const Vec3i& unitIdx, const Point3i& volUnitDims,
                            float voxelSize, float trancDist, int maxWeight, const Intr& intrinsics);

std::vector<Vec3i> findNewVisibleVolumeUnits(const VolumeSettings& settings, const Matx44f& cameraPose,
                                           InputArray depth, const Intr& intrinsics,
                                           const VolumeUnitIndexes& existingUnits,
                                           const Point3i& volUnitDims, float voxelSize);

void raycastColorHashTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
                                 int height, int width, InputArray intr, int volumeUnitDegree,
                                 InputArray volUnitsData, const CustomHashSet& hashTable,
                                 OutputArray points, OutputArray normals, OutputArray colors);

void ocl_fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(const VolumeSettings& settings,
                                                          InputArray volUnitsData,
                                                          const CustomHashSet& hashTable,
                                                          int volumeUnitDegree,
                                                          OutputArray points,
                                                          OutputArray normals,
                                                          OutputArray colors);

bool raycastColorHashTsdf(const VolumeSettings& settings, const Point3f& rayOrigin,
                       const Point3f& rayDir, float tmin, float tmax,
                       const VolumeUnitIndexes& volumeUnits, InputArray volUnitsData,
                       const Point3i& volUnitDims, int unitResolution,
                       float voxelSize, float trancDist,
                       Point3f& hitPoint, Point3f& hitNormal, Vec3b& hitColor);

Point3f computeColorVoxelNormal(InputArray _unitData, int x, int y, int z, int volUnitSize, float voxelSize, float deltaFactor);

} // namespace cv

#endif
