// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "color_tsdf_functions.hpp"
#include "opencl_kernels_3d.hpp"

namespace cv {


void integrateColorTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _volume)
{
    Matx44f volumePose;
    settings.getVolumePose(volumePose);
    integrateColorTsdfVolumeUnit(settings, volumePose, cameraPose, _depth, _rgb, _pixNorms, _volume);
}


void integrateColorTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& volumePose, const Matx44f& cameraPose,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _volume)
{





}




} // namespace cv
