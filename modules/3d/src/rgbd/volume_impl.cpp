// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include <iostream>
#include "volume_impl.hpp"
#include "tsdf_functions.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

Volume::Impl::Impl(const VolumeSettings& _settings) :
    settings(_settings)
{}

// TSDF

TsdfVolume::TsdfVolume(const VolumeSettings& settings) :
    Volume::Impl(settings)
{
    std::cout << "TsdfVolume::TsdfVolume()" << std::endl;

    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
#ifdef HAVE_OPENCL
    volume = UMat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#else
    volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#endif

    reset();
}
TsdfVolume::~TsdfVolume() {}

void TsdfVolume::integrate(OdometryFrame frame, InputArray _cameraPose)
{
    std::cout << "TsdfVolume::integrate()" << std::endl;

    CV_TRACE_FUNCTION();
    Depth depth;
    frame.getDepth(depth);
    CV_Assert(depth.type() == DEPTH_TYPE);
    CV_Assert(!depth.empty());
    // TODO: remove this dependence from OdometryFrame
    depth = depth * settings.getDepthFactor();

    Matx33f intr;
    settings.getCameraIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
#ifdef HAVE_OPENCL
        pixNorms = ocl_preCalculationPixNorm(depth.size(), intrinsics);
#else
        pixNorms = preCalculationPixNormGPU(depth.size(), intrinsics);
#endif
    }
    Matx44f cameraPose = _cameraPose.getMat();

#ifdef HAVE_OPENCL
    ocl_integrateVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#else
    integrateVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#endif
}

void TsdfVolume::integrate(InputArray frame, InputArray _cameraPose)
{
    std::cout << "TsdfVolume::integrate()" << std::endl;

    CV_TRACE_FUNCTION();
    Depth depth = frame.getMat();
    CV_Assert(depth.type() == DEPTH_TYPE);
    CV_Assert(!depth.empty());

    Matx33f intr;
    settings.getCameraIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
#ifdef HAVE_OPENCL
        pixNorms = ocl_preCalculationPixNorm(depth.size(), intrinsics);
#else
        pixNorms = preCalculationPixNormGPU(depth.size(), intrinsics);
#endif
    }
    const Matx44f cameraPose = _cameraPose.getMat();

#ifdef HAVE_OPENCL
    ocl_integrateVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#else
    integrateVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#endif
}

void TsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const
{
    std::cout << "TsdfVolume::raycast()" << std::endl;

    CV_Assert(height > 0);
    CV_Assert(width > 0);

    const Matx44f cameraPose = _cameraPose.getMat();
#ifdef HAVE_OPENCL
    ocl_raycastVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals);
#else
    raycastVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals);
#endif
}

void TsdfVolume::fetchNormals() const {}
void TsdfVolume::fetchPointsNormals() const {}

void TsdfVolume::reset()
{
    CV_TRACE_FUNCTION();
#ifdef HAVE_OPENCL
    volume.setTo(Scalar(0, 0));
#else
    //TODO: use setTo(Scalar(0, 0))
    volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
        {
            TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
#endif
}
int TsdfVolume::getVisibleBlocks() const { return 1; }
size_t TsdfVolume::getTotalVolumeUnits() const { return 1; }


// HASH_TSDF

HashTsdfVolume::HashTsdfVolume(const VolumeSettings& settings) :
    Volume::Impl(settings)
{ }
HashTsdfVolume::~HashTsdfVolume() {}

void HashTsdfVolume::integrate(OdometryFrame frame, InputArray pose) { std::cout << "HashTsdfVolume::integrate()" << std::endl; }
void HashTsdfVolume::integrate(InputArray frame, InputArray pose) { std::cout << "HashTsdfVolume::integrate()" << std::endl; }
void HashTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const { std::cout << "HashTsdfVolume::raycast()" << std::endl; }

void HashTsdfVolume::fetchNormals() const {}
void HashTsdfVolume::fetchPointsNormals() const {}

void HashTsdfVolume::reset() {}
int HashTsdfVolume::getVisibleBlocks() const { return 1; }
size_t HashTsdfVolume::getTotalVolumeUnits() const { return 1; }

// COLOR_TSDF

ColorTsdfVolume::ColorTsdfVolume(const VolumeSettings& settings) :
    Volume::Impl(settings)
{}
ColorTsdfVolume::~ColorTsdfVolume() {}

void ColorTsdfVolume::integrate(OdometryFrame frame, InputArray pose) { std::cout << "ColorTsdfVolume::integrate()" << std::endl; }
void ColorTsdfVolume::integrate(InputArray frame, InputArray pose) { std::cout << "ColorTsdfVolume::integrate()" << std::endl; }
void ColorTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const { std::cout << "ColorTsdfVolume::raycast()" << std::endl; }

void ColorTsdfVolume::fetchNormals() const {}
void ColorTsdfVolume::fetchPointsNormals() const {}

void ColorTsdfVolume::reset() {}
int ColorTsdfVolume::getVisibleBlocks() const { return 1; }
size_t ColorTsdfVolume::getTotalVolumeUnits() const { return 1; }

}
