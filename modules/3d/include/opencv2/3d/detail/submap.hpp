// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_DETAIL_SUBMAP_HPP
#define OPENCV_3D_DETAIL_SUBMAP_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>
#include "opencv2/3d/detail/optimizer.hpp"

//TODO: remove it when it is rewritten to robust pose graph
#include "opencv2/core/dualquaternion.hpp"

#include <type_traits>
#include <vector>
#include <map>
#include <unordered_map>


namespace cv
{
namespace detail
{
template<typename MatType>
class Submap
{
public:
    struct PoseConstraint
    {
        Affine3f estimatedPose;
        int weight;

        PoseConstraint() : weight(0){};

        void accumulatePose(const Affine3f& _pose, int _weight = 1)
        {
            DualQuatf accPose = DualQuatf::createFromAffine3(estimatedPose) * float(weight) + DualQuatf::createFromAffine3(_pose) * float(_weight);
            weight += _weight;
            accPose = accPose / float(weight);
            estimatedPose = accPose.toAffine3();
        }
    };
    typedef std::map<int, PoseConstraint> Constraints;

    Submap(int _id, const VolumeSettings& settings, const cv::Affine3f& _pose = cv::Affine3f::Identity(),
           int _startFrameId = 0)
        : id(_id), pose(_pose), cameraPose(Affine3f::Identity()), startFrameId(_startFrameId),
          volume(VolumeType::HashTSDF, settings)
    { }
    virtual ~Submap() = default;

    virtual void integrate(InputArray _depth, const int currframeId);
    virtual void raycast(const cv::Affine3f& cameraPose, cv::Size frameSize, cv::Matx33f K,
                         OutputArray points = noArray(), OutputArray normals = noArray());

    virtual int getTotalAllocatedBlocks() const { return int(volume.getTotalVolumeUnits()); };
    virtual int getVisibleBlocks(int currFrameId) const
    {
        CV_Assert(currFrameId >= startFrameId);
        //return volume.getVisibleBlocks(currFrameId, FRAME_VISIBILITY_THRESHOLD);
        return volume.getVisibleBlocks();

    }

    float calcVisibilityRatio(int currFrameId) const
    {
        int allocate_blocks = getTotalAllocatedBlocks();
        int visible_blocks  = getVisibleBlocks(currFrameId);
        return float(visible_blocks) / float(allocate_blocks);
    }

    // Adding new Edge for Loop Closure Detection. Return true or false to indicate whether adding success.
    bool addEdgeToSubmap(const int tarSubmapID, const Affine3f& tarPose);

    //! TODO: Possibly useless
    virtual void setStartFrameId(int _startFrameId) { startFrameId = _startFrameId; };
    virtual void setStopFrameId(int _stopFrameId) { stopFrameId = _stopFrameId; };

    void composeCameraPose(const cv::Affine3f& _relativePose) { cameraPose = cameraPose * _relativePose; }
    PoseConstraint& getConstraint(const int _id)
    {
        //! Creates constraints if doesn't exist yet
        return constraints[_id];
    }

public:
    const int id;
    cv::Affine3f pose;
    cv::Affine3f cameraPose;
    Constraints constraints;

    int startFrameId;
    int stopFrameId;
    //! TODO: Should we support submaps for regular volumes?
    static constexpr int FRAME_VISIBILITY_THRESHOLD = 5;

    //! TODO: Add support for GPU arrays (UMat)
    OdometryFrame frame;
    OdometryFrame renderFrame;

    Volume volume;
};

template<typename MatType>

void Submap<MatType>::integrate(InputArray _depth, const int currFrameId)
{
    CV_Assert(currFrameId >= startFrameId);
    volume.integrate(_depth, cameraPose.matrix);
}

template<typename MatType>
void Submap<MatType>::raycast(const cv::Affine3f& _cameraPose, cv::Size frameSize, cv::Matx33f K,
                              OutputArray points, OutputArray normals)
{
    if (!points.needed() && !normals.needed())
    {
        MatType pts, nrm;
        //TODO: get depth instead of pts from raycast
        volume.raycast(_cameraPose.matrix, frameSize.height, frameSize.width, K, pts, nrm);

        std::vector<MatType> pch(3);
        split(pts, pch);

        renderFrame = frame;

        frame = OdometryFrame(pch[2]);
    }
    else
    {
        volume.raycast(_cameraPose.matrix, frameSize.height, frameSize.width, K, points, normals);
    }
}

template<typename MatType>
bool Submap<MatType>::addEdgeToSubmap(const int tarSubmapID, const Affine3f& tarPose)
{
    auto iter = constraints.find(tarSubmapID);

    // if there is NO edge of currSubmap to tarSubmap.
    if(iter == constraints.end())
    {
        // Frome pose to tarPose transformation
        Affine3f estimatePose = tarPose * pose.inv();

        // Create new Edge.
        PoseConstraint& preConstrain = getConstraint(tarSubmapID);
        preConstrain.accumulatePose(estimatePose, 1);

        return true;
    } else
    {
        return false;
    }
}

/**
 * @brief: Manages all the created submaps for a particular scene
 */
template<typename MatType>
class SubmapManager
{
public:
    enum class Type
    {
        NEW            = 0,
        CURRENT        = 1,
        RELOCALISATION = 2,
        LOOP_CLOSURE   = 3,
        LOST           = 4
    };

    struct ActiveSubmapData
    {
        Type type;
        std::vector<Affine3f> constraints;
        int trackingAttempts;
    };

    typedef Submap<MatType> SubmapT;
    typedef std::map<int, Ptr<SubmapT>> IdToSubmapPtr;
    typedef std::unordered_map<int, ActiveSubmapData> IdToActiveSubmaps;

    explicit SubmapManager(const VolumeSettings& _volumeSettings) : volumeSettings(_volumeSettings) {}
    virtual ~SubmapManager() = default;

    void reset() { submapList.clear(); };

    bool shouldCreateSubmap(int frameId);
    bool shouldChangeCurrSubmap(int _frameId, int toSubmapId);

    //! Adds a new submap/volume into the current list of managed/Active submaps
    int createNewSubmap(bool isCurrentActiveMap, const int currFrameId = 0, const Affine3f& pose = cv::Affine3f::Identity());

    void removeSubmap(int _id);
    size_t numOfSubmaps(void) const { return submapList.size(); };
    size_t numOfActiveSubmaps(void) const { return activeSubmaps.size(); };

    Ptr<SubmapT> getSubmap(int _id) const;
    Ptr<SubmapT> getCurrentSubmap(void) const;

    int estimateConstraint(int fromSubmapId, int toSubmapId, int& inliers, Affine3f& inlierPose);
    bool updateMap(int frameId, const OdometryFrame& frame);

    bool addEdgeToCurrentSubmap(const int currentSubmapID, const int tarSubmapID);

    Ptr<detail::PoseGraph> MapToPoseGraph();
    void PoseGraphToMap(const Ptr<detail::PoseGraph>& updatedPoseGraph);

    VolumeSettings volumeSettings;

    std::vector<Ptr<SubmapT>> submapList;
    IdToActiveSubmaps activeSubmaps;

    Ptr<detail::PoseGraph> poseGraph;
};

template<typename MatType>
int SubmapManager<MatType>::createNewSubmap(bool isCurrentMap, int currFrameId, const Affine3f& pose)
{
    int newId = int(submapList.size());

    Ptr<SubmapT> newSubmap = cv::makePtr<SubmapT>(newId, volumeSettings, pose, currFrameId);
    submapList.push_back(newSubmap);

    ActiveSubmapData newSubmapData;
    newSubmapData.trackingAttempts = 0;
    newSubmapData.type             = isCurrentMap ? Type::CURRENT : Type::NEW;
    activeSubmaps[newId]           = newSubmapData;

    return newId;
}

template<typename MatType>
Ptr<Submap<MatType>> SubmapManager<MatType>::getSubmap(int _id) const
{
    CV_Assert(submapList.size() > 0);
    CV_Assert(_id >= 0 && _id < int(submapList.size()));
    return submapList.at(_id);
}

template<typename MatType>
Ptr<Submap<MatType>> SubmapManager<MatType>::getCurrentSubmap(void) const
{
    for (const auto& it : activeSubmaps)
    {
        if (it.second.type == Type::CURRENT)
            return getSubmap(it.first);
    }
    return nullptr;
}

template<typename MatType>
bool SubmapManager<MatType>::shouldCreateSubmap(int currFrameId)
{
    int currSubmapId = -1;
    for (const auto& it : activeSubmaps)
    {
        auto submapData = it.second;
        // No more than 1 new submap at a time!
        if (submapData.type == Type::NEW)
        {
            return false;
        }
        if (submapData.type == Type::CURRENT)
        {
            currSubmapId = it.first;
        }
    }
    //! TODO: This shouldn't be happening? since there should always be one active current submap
    if (currSubmapId < 0)
    {
        return false;
    }

    Ptr<SubmapT> currSubmap = getSubmap(currSubmapId);
    float ratio             = currSubmap->calcVisibilityRatio(currFrameId);

    //TODO: fix this when a new pose graph is ready
    // if (ratio < 0.2f)
    if (ratio < 0.5f)
        return true;
    return false;
}

template<typename MatType>
int SubmapManager<MatType>::estimateConstraint(int fromSubmapId, int toSubmapId, int& inliers, Affine3f& inlierPose)
{
    static constexpr int MAX_ITER                    = 10;
    static constexpr float CONVERGE_WEIGHT_THRESHOLD = 0.01f;
    static constexpr float INLIER_WEIGHT_THRESH      = 0.8f;
    static constexpr int MIN_INLIERS                 = 10;
    static constexpr int MAX_TRACKING_ATTEMPTS = 25;

    //! thresh = HUBER_THRESH
    auto huberWeight = [](float residual, float thresh = 0.1f) -> float {
        float rAbs = abs(residual);
        if (rAbs < thresh)
            return 1.0;
        float numerator = sqrt(2 * thresh * rAbs - thresh * thresh);
        return numerator / rAbs;
    };

    Ptr<SubmapT> fromSubmap          = getSubmap(fromSubmapId);
    Ptr<SubmapT> toSubmap            = getSubmap(toSubmapId);
    ActiveSubmapData& fromSubmapData = activeSubmaps.at(fromSubmapId);

    Affine3f TcameraToFromSubmap = fromSubmap->cameraPose;
    Affine3f TcameraToToSubmap   = toSubmap->cameraPose;

    // FromSubmap -> ToSubmap transform
    Affine3f candidateConstraint = TcameraToToSubmap * TcameraToFromSubmap.inv();
    fromSubmapData.trackingAttempts++;
    fromSubmapData.constraints.push_back(candidateConstraint);

    std::vector<float> weights(fromSubmapData.constraints.size() + 1, 1.0f);

    Affine3f prevConstraint = fromSubmap->getConstraint(toSubmap->id).estimatedPose;
    int prevWeight          = fromSubmap->getConstraint(toSubmap->id).weight;

    // Iterative reweighted least squares with huber threshold to find the inliers in the past observations
    Vec6f meanConstraint;
    float sumWeight = 0.0f;
    for (int i = 0; i < MAX_ITER; i++)
    {
        Vec6f constraintVec;
        for (int j = 0; j < int(weights.size() - 1); j++)
        {
            Affine3f currObservation = fromSubmapData.constraints[j];
            cv::vconcat(currObservation.rvec(), currObservation.translation(), constraintVec);
            meanConstraint += weights[j] * constraintVec;
            sumWeight += weights[j];
        }
        // Heavier weight given to the estimatedPose
        cv::vconcat(prevConstraint.rvec(), prevConstraint.translation(), constraintVec);
        meanConstraint += weights.back() * prevWeight * constraintVec;
        sumWeight += prevWeight;
        meanConstraint /= float(sumWeight);

        float residual = 0.0f;
        float diff     = 0.0f;
        for (int j = 0; j < int(weights.size()); j++)
        {
            int w;
            if (j == int(weights.size() - 1))
            {
                cv::vconcat(prevConstraint.rvec(), prevConstraint.translation(), constraintVec);
                w = prevWeight;
            }
            else
            {
                Affine3f currObservation = fromSubmapData.constraints[j];
                cv::vconcat(currObservation.rvec(), currObservation.translation(), constraintVec);
                w = 1;
            }

            cv::Vec6f residualVec = (constraintVec - meanConstraint);
            residual         = float(norm(residualVec));
            float newWeight = huberWeight(residual);
            diff += w * abs(newWeight - weights[j]);
            weights[j] = newWeight;
        }

        if (diff / (prevWeight + weights.size() - 1) < CONVERGE_WEIGHT_THRESHOLD)
            break;
    }

    int localInliers = 0;
    DualQuatf inlierConstraint;
    for (int i = 0; i < int(weights.size()); i++)
    {
        if (weights[i] > INLIER_WEIGHT_THRESH)
        {
            localInliers++;
            if (i == int(weights.size() - 1))
                inlierConstraint += DualQuatf::createFromMat(prevConstraint.matrix);
            else
                inlierConstraint += DualQuatf::createFromMat(fromSubmapData.constraints[i].matrix);
        }
    }
    inlierConstraint = inlierConstraint * 1.0f/float(max(localInliers, 1));
    inlierPose = inlierConstraint.toAffine3();
    inliers    = localInliers;

    if (inliers >= MIN_INLIERS)
    {
        return 1;
    }
    if(fromSubmapData.trackingAttempts - inliers > (MAX_TRACKING_ATTEMPTS - MIN_INLIERS))
    {
        return -1;
    }

    return 0;
}

template<typename MatType>
bool SubmapManager<MatType>::shouldChangeCurrSubmap(int _frameId, int toSubmapId)
{
    auto toSubmap         = getSubmap(toSubmapId);
    auto toSubmapData     = activeSubmaps.at(toSubmapId);
    auto currActiveSubmap = getCurrentSubmap();

    int blocksInNewMap = toSubmap->getTotalAllocatedBlocks();
    float newRatio     = toSubmap->calcVisibilityRatio(_frameId);

    float currRatio = currActiveSubmap->calcVisibilityRatio(_frameId);

    //! TODO: Check for a specific threshold?
    if (blocksInNewMap <= 0)
        return false;
    if ((newRatio > currRatio) && (toSubmapData.type == Type::NEW))
        return true;

    return false;
}

template<typename MatType>
bool SubmapManager<MatType>::addEdgeToCurrentSubmap(const int currentSubmapID, const int tarSubmapID)
{
    Ptr<SubmapT> currentSubmap = getSubmap(currentSubmapID);
    Ptr<SubmapT> tarSubmap = getSubmap(tarSubmapID);

    return currentSubmap->addEdgeToSubmap(tarSubmapID, tarSubmap->pose);
}

template<typename MatType>
bool SubmapManager<MatType>::updateMap(int _frameId, const OdometryFrame& _frame)
{
    bool mapUpdated = false;
    int changedCurrentMapId = -1;

    const int currSubmapId  = getCurrentSubmap()->id;

    for (auto& it : activeSubmaps)
    {
        int submapId     = it.first;
        auto& submapData = it.second;
        if (submapData.type == Type::NEW || submapData.type == Type::LOOP_CLOSURE)
        {
            // Check with previous estimate
            int inliers;
            Affine3f inlierPose;
            int constraintUpdate = estimateConstraint(submapId, currSubmapId, inliers, inlierPose);
            if (constraintUpdate == 1)
            {
                typename SubmapT::PoseConstraint& submapConstraint = getSubmap(submapId)->getConstraint(currSubmapId);
                submapConstraint.accumulatePose(inlierPose, inliers);
                submapData.constraints.clear();
                submapData.trackingAttempts = 0;

                if (shouldChangeCurrSubmap(_frameId, submapId))
                {
                    changedCurrentMapId = submapId;
                }
                mapUpdated = true;
            }
            else if(constraintUpdate == -1)
            {
                submapData.type = Type::LOST;
            }
        }
    }

    std::vector<int> createNewConstraintsList;
    for (auto& it : activeSubmaps)
    {
        int submapId     = it.first;
        auto& submapData = it.second;

        if (submapId == changedCurrentMapId)
        {
            submapData.type = Type::CURRENT;
        }
        if ((submapData.type == Type::CURRENT) && (changedCurrentMapId >= 0) && (submapId != changedCurrentMapId))
        {
            submapData.type = Type::LOST;
            createNewConstraintsList.push_back(submapId);
        }
        if ((submapData.type == Type::NEW || submapData.type == Type::LOOP_CLOSURE) && (changedCurrentMapId >= 0))
        {
            //! TODO: Add a new type called NEW_LOST?
            submapData.type = Type::LOST;
            createNewConstraintsList.push_back(submapId);
        }
    }

    for (typename IdToActiveSubmaps::iterator it = activeSubmaps.begin(); it != activeSubmaps.end();)
    {
        auto& submapData = it->second;
        if (submapData.type == Type::LOST)
            it = activeSubmaps.erase(it);
        else
            it++;
    }

    for (std::vector<int>::const_iterator it = createNewConstraintsList.begin(); it != createNewConstraintsList.end(); ++it)
    {
        int dataId = *it;
        ActiveSubmapData newSubmapData;
        newSubmapData.trackingAttempts = 0;
        newSubmapData.type             = Type::LOOP_CLOSURE;
        activeSubmaps[dataId]          = newSubmapData;
    }

    if (shouldCreateSubmap(_frameId))
    {
        Ptr<SubmapT> currActiveSubmap = getCurrentSubmap();
        Affine3f newSubmapPose        = currActiveSubmap->pose * currActiveSubmap->cameraPose;
        int submapId                  = createNewSubmap(false, _frameId, newSubmapPose);
        auto newSubmap                = getSubmap(submapId);
        newSubmap->frame              = _frame;
    }

    return mapUpdated;
}

template<typename MatType>
Ptr<detail::PoseGraph> SubmapManager<MatType>::MapToPoseGraph()
{
    Ptr<detail::PoseGraph> localPoseGraph = detail::PoseGraph::create();

    for(const auto& currSubmap : submapList)
    {
        const typename SubmapT::Constraints& constraintList = currSubmap->constraints;
        for(const auto& currConstraintPair : constraintList)
        {
            // TODO: Handle case with duplicate constraints A -> B and B -> A
            /* Matx66f informationMatrix = Matx66f::eye() * (currConstraintPair.second.weight/10); */
            Matx66f informationMatrix = Matx66f::eye();
            localPoseGraph->addEdge(currSubmap->id, currConstraintPair.first, currConstraintPair.second.estimatedPose, informationMatrix);
        }
    }

    for(const auto& currSubmap : submapList)
    {
        localPoseGraph->addNode(currSubmap->id, currSubmap->pose, (currSubmap->id == 0));
    }

    return localPoseGraph;
}

template <typename MatType>
void SubmapManager<MatType>::PoseGraphToMap(const Ptr<detail::PoseGraph>& updatedPoseGraph)
{
    for(const auto& currSubmap : submapList)
    {
        Affine3d pose = updatedPoseGraph->getNodePose(currSubmap->id);
        if(!updatedPoseGraph->isNodeFixed(currSubmap->id))
            currSubmap->pose = pose;
    }
}

}  // namespace detail
}  // namespace cv

#endif // include guard
