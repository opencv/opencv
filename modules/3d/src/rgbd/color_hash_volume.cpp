// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "hash_tsdf_functions.hpp"
#include "color_hash_volume.hpp"

namespace cv {

void integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex,
    const int frameId, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms,
    InputOutputArray _volUnitsData, VolumeUnitIndexes& volumeUnits)
{
    CV_TRACE_FUNCTION();

    // Reuse HashTSDF allocation and volume unit management code
    // but integrate colors during voxel updates
    
    // Start with regular HashTSDF integration
    integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, 
                               frameId, volumeUnitDegree, enableGrowth,
                               _depth, _pixNorms, _volUnitsData, volumeUnits);

    // Then add color integration
    Mat depth = _depth.getMat();
    Mat color = _rgb.getMat();
    Mat& volUnitsData = _volUnitsData.getMatRef();
    
    // Iterate through visible volume units and update colors
    for (auto& vu : volumeUnits)
    {
        if (vu.second.isActive)
        {
            // Get color values at corresponding depth pixels
            ColorHashTsdfVoxel* voxelPtr = volUnitsData.ptr<ColorHashTsdfVoxel>(vu.second.index);
            
            // Update RGB values using weighted average
            // Similar to ColorTSDF integration but with hash table lookup
            // ... (color integration code) ...
        }
    }
}

// Implement remaining functions reusing HashTSDF code where possible
// and adding color support similar to ColorTSDF implementation

} // namespace cv