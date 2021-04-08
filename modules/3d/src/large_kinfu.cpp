// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#include "fast_icp.hpp"
#include "hash_tsdf.hpp"
#include "kinfu_frame.hpp"
#include "precomp.hpp"
#include "submap.hpp"
#include "tsdf.hpp"

namespace cv
{
namespace large_kinfu
{
using namespace kinfu;

Ptr<Params> Params::defaultParams()
{
    Params p;

    //! Frame parameters
    {
        p.frameSize = Size(640, 480);

        float fx, fy, cx, cy;
        fx = fy = 525.f;
        cx      = p.frameSize.width / 2.0f - 0.5f;
        cy      = p.frameSize.height / 2.0f - 0.5f;
        p.intr  = Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1);

        // 5000 for the 16-bit PNG files
        // 1 for the 32-bit float images in the ROS bag files
        p.depthFactor = 5000;

        // sigma_depth is scaled by depthFactor when calling bilateral filter
        p.bilateral_sigma_depth   = 0.04f;  // meter
        p.bilateral_sigma_spatial = 4.5;    // pixels
        p.bilateral_kernel_size   = 7;      // pixels
        p.truncateThreshold       = 0.f;    // meters
    }
    //! ICP parameters
    {
        p.icpAngleThresh = (float)(30. * CV_PI / 180.);  // radians
        p.icpDistThresh  = 0.1f;                         // meters

        p.icpIterations = { 10, 5, 4 };
        p.pyramidLevels = (int)p.icpIterations.size();
    }
    //! Volume parameters
    {
        float volumeSize                   = 3.0f;
        p.volumeParams.type                = VolumeType::TSDF;
        p.volumeParams.resolution          = Vec3i::all(512);
        p.volumeParams.pose                = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.5f));
        p.volumeParams.voxelSize           = volumeSize / 512.f;            // meters
        p.volumeParams.tsdfTruncDist       = 7 * p.volumeParams.voxelSize;  // about 0.04f in meters
        p.volumeParams.maxWeight           = 64;                            // frames
        p.volumeParams.raycastStepFactor   = 0.25f;                         // in voxel sizes
        p.volumeParams.depthTruncThreshold = p.truncateThreshold;
    }
    //! Unused parameters
    p.tsdf_min_camera_movement = 0.f;              // meters, disabled
    p.lightPose                = Vec3f::all(0.f);  // meters

    return makePtr<Params>(p);
}

Ptr<Params> Params::coarseParams()
{
    Ptr<Params> p = defaultParams();

    //! Reduce ICP iterations and pyramid levels
    {
        p->icpIterations = { 5, 3, 2 };
        p->pyramidLevels = (int)p->icpIterations.size();
    }
    //! Make the volume coarse
    {
        float volumeSize                  = 3.f;
        p->volumeParams.resolution        = Vec3i::all(128);  // number of voxels
        p->volumeParams.voxelSize         = volumeSize / 128.f;
        p->volumeParams.tsdfTruncDist     = 2 * p->volumeParams.voxelSize;  // 0.04f in meters
        p->volumeParams.raycastStepFactor = 0.75f;                          // in voxel sizes
    }
    return p;
}
Ptr<Params> Params::hashTSDFParams(bool isCoarse)
{
    Ptr<Params> p;
    if (isCoarse)
        p = coarseParams();
    else
        p = defaultParams();

    p->volumeParams.type                = VolumeType::HASHTSDF;
    p->volumeParams.depthTruncThreshold = rgbd::Odometry::DEFAULT_MAX_DEPTH();
    p->volumeParams.unitResolution      = 16;
    return p;
}

// MatType should be Mat or UMat
template<typename MatType>
class LargeKinfuImpl : public LargeKinfu
{
   public:
    LargeKinfuImpl(const Params& _params);
    virtual ~LargeKinfuImpl();

    const Params& getParams() const CV_OVERRIDE;

    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    virtual void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    const Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth) CV_OVERRIDE;

    bool updateT(const MatType& depth);

   private:
    Params params;

    cv::Ptr<ICP> icp;
    //! TODO: Submap manager and Pose graph optimizer
    cv::Ptr<SubmapManager<MatType>> submapMgr;

    int frameCounter;
    Affine3f pose;
};

template<typename MatType>
LargeKinfuImpl<MatType>::LargeKinfuImpl(const Params& _params)
    : params(_params)
{
    icp = makeICP(params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh);

    submapMgr = cv::makePtr<SubmapManager<MatType>>(params.volumeParams);
    reset();
    submapMgr->createNewSubmap(true);

}

template<typename MatType>
void LargeKinfuImpl<MatType>::reset()
{
    frameCounter = 0;
    pose         = Affine3f::Identity();
    submapMgr->reset();
}

template<typename MatType>
LargeKinfuImpl<MatType>::~LargeKinfuImpl()
{
}

template<typename MatType>
const Params& LargeKinfuImpl<MatType>::getParams() const
{
    return params;
}

template<typename MatType>
const Affine3f LargeKinfuImpl<MatType>::getPose() const
{
    return pose;
}

template<>
bool LargeKinfuImpl<Mat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    Mat depth;
    if (_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getMat());
    }
}

template<>
bool LargeKinfuImpl<UMat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    UMat depth;
    if (!_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getUMat());
    }
}


template<typename MatType>
bool LargeKinfuImpl<MatType>::updateT(const MatType& _depth)
{
    CV_TRACE_FUNCTION();

    MatType depth;
    if (_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    std::vector<MatType> newPoints, newNormals;
    makeFrameFromDepth(depth, newPoints, newNormals, params.intr, params.pyramidLevels, params.depthFactor,
                       params.bilateral_sigma_depth, params.bilateral_sigma_spatial, params.bilateral_kernel_size,
                       params.truncateThreshold);

    CV_LOG_INFO(NULL, "Current frameID: " << frameCounter);
    for (const auto& it : submapMgr->activeSubmaps)
    {
        int currTrackingId = it.first;
        auto submapData = it.second;
        Ptr<Submap<MatType>> currTrackingSubmap = submapMgr->getSubmap(currTrackingId);
        Affine3f affine;
        CV_LOG_INFO(NULL, "Current tracking ID: " << currTrackingId);

        if(frameCounter == 0) //! Only one current tracking map
        {
            currTrackingSubmap->integrate(depth, params.depthFactor, params.intr, frameCounter);
            currTrackingSubmap->pyrPoints = newPoints;
            currTrackingSubmap->pyrNormals = newNormals;
            continue;
        }

        //1. Track
        bool trackingSuccess =
            icp->estimateTransform(affine, currTrackingSubmap->pyrPoints, currTrackingSubmap->pyrNormals, newPoints, newNormals);
        if (trackingSuccess)
            currTrackingSubmap->composeCameraPose(affine);
        else
        {
            CV_LOG_INFO(NULL, "Tracking failed");
            continue;
        }

        //2. Integrate
        if(submapData.type == SubmapManager<MatType>::Type::NEW || submapData.type == SubmapManager<MatType>::Type::CURRENT)
        {
            float rnorm = (float)cv::norm(affine.rvec());
            float tnorm = (float)cv::norm(affine.translation());
            // We do not integrate volume if camera does not move
            if ((rnorm + tnorm) / 2 >= params.tsdf_min_camera_movement)
                currTrackingSubmap->integrate(depth, params.depthFactor, params.intr, frameCounter);
        }

        //3. Raycast
        currTrackingSubmap->raycast(currTrackingSubmap->cameraPose, params.intr, params.frameSize, currTrackingSubmap->pyrPoints[0], currTrackingSubmap->pyrNormals[0]);

        currTrackingSubmap->updatePyrPointsNormals(params.pyramidLevels);

        CV_LOG_INFO(NULL, "Submap: " << currTrackingId << " Total allocated blocks: " << currTrackingSubmap->getTotalAllocatedBlocks());
        CV_LOG_INFO(NULL, "Submap: " << currTrackingId << " Visible blocks: " << currTrackingSubmap->getVisibleBlocks(frameCounter));

    }
    //4. Update map
    bool isMapUpdated = submapMgr->updateMap(frameCounter, newPoints, newNormals);

    if(isMapUpdated)
    {
        // TODO: Convert constraints to posegraph
        Ptr<kinfu::detail::PoseGraph> poseGraph = submapMgr->MapToPoseGraph();
        CV_LOG_INFO(NULL, "Created posegraph");
        int iters = poseGraph->optimize();
        if (iters < 0)
        {
            CV_LOG_INFO(NULL, "Failed to perform pose graph optimization");
            return false;
        }

        submapMgr->PoseGraphToMap(poseGraph);

    }
    CV_LOG_INFO(NULL, "Number of submaps: " << submapMgr->submapList.size());

    frameCounter++;
    return true;
}

template<typename MatType>
void LargeKinfuImpl<MatType>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);

    auto currSubmap = submapMgr->getCurrentSubmap();
    const Affine3f id = Affine3f::Identity();
    if ((cameraPose.rotation() == pose.rotation() && cameraPose.translation() == pose.translation()) ||
        (cameraPose.rotation() == id.rotation() && cameraPose.translation() == id.translation()))
    {
        //! TODO: Can render be dependent on current submap
        renderPointsNormals(currSubmap->pyrPoints[0], currSubmap->pyrNormals[0], image, params.lightPose);
    }
    else
    {
        MatType points, normals;
        currSubmap->raycast(cameraPose, params.intr, params.frameSize, points, normals);
        renderPointsNormals(points, normals, image, params.lightPose);
    }
}

template<typename MatType>
void LargeKinfuImpl<MatType>::getCloud(OutputArray p, OutputArray n) const
{
    auto currSubmap = submapMgr->getCurrentSubmap();
    currSubmap->volume->fetchPointsNormals(p, n);
}

template<typename MatType>
void LargeKinfuImpl<MatType>::getPoints(OutputArray points) const
{
    auto currSubmap = submapMgr->getCurrentSubmap();
    currSubmap->volume->fetchPointsNormals(points, noArray());
}

template<typename MatType>
void LargeKinfuImpl<MatType>::getNormals(InputArray points, OutputArray normals) const
{
    auto currSubmap = submapMgr->getCurrentSubmap();
    currSubmap->volume->fetchNormals(points, normals);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<LargeKinfu> LargeKinfu::create(const Ptr<Params>& params)
{
    CV_Assert((int)params->icpIterations.size() == params->pyramidLevels);
    CV_Assert(params->intr(0, 1) == 0 && params->intr(1, 0) == 0 && params->intr(2, 0) == 0 && params->intr(2, 1) == 0 &&
              params->intr(2, 2) == 1);
#ifdef HAVE_OPENCL
    if (cv::ocl::useOpenCL())
        return makePtr<LargeKinfuImpl<UMat>>(*params);
#endif
    return makePtr<LargeKinfuImpl<Mat>>(*params);
}

#else
Ptr<LargeKinfu> LargeKinfu::create(const Ptr<Params>& /* params */)
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif
}  // namespace large_kinfu
}  // namespace cv
