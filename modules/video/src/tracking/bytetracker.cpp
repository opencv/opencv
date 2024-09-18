// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "detail/bytetracker_strack.hpp"
#include <map>
#include <vector>
#include <unordered_map>
#include <iostream>
#include "lapjv.h"

namespace cv {

using namespace cv::detail::tracking;

std::map<int, int> lapjv(const Mat &cost, float matchThreshold)
{
    std::map<int, int> ret;
    if (cost.rows == 0 || cost.cols == 0)
        return ret;
    int maxI = cost.rows;
    int maxJ = cost.cols;
    int n = max(maxJ, maxI);

    std::vector<std::vector<double>> cost_ptr(n, std::vector<double>(n));
    std::vector<int> x_c(n);
    std::vector<int> y_c(n);

    std::vector<double*> cost_ptr_ptr(n);
    for (int i = 0; i < n; i++)
    {
        cost_ptr_ptr[i] = cost_ptr[i].data();
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < maxI && j < maxJ && cost.at<float>(i, j) < matchThreshold) // verify
            {
                cost_ptr[i][j] = static_cast<double>(cost.at<float>(i, j));
            }
            else
            {
                cost_ptr[i][j] = LARGE;
            }
        }
        x_c[i] = -1;
        y_c[i] = -1;
    }
    lapjv_internal(n, cost_ptr_ptr.data(), x_c.data(), y_c.data());
    for (int i = 0; i < n; i++)
    {
        if (i < maxI && x_c[i] < maxJ) // verify
        {
            ret[i] = x_c[i];
        }
    }

    return ret;
}

void lapjv(InputArray costMatrix, OutputArray assignedPairs, float matchThreshold)
{
    auto ret = lapjv(costMatrix.getMat(), matchThreshold);
    auto numPairs = ret.size();
    assignedPairs.create(static_cast<int>(numPairs), 2, CV_32S);

    auto c_assigned_pairs = assignedPairs.getMat();
    for (auto const& x : ret) {
        c_assigned_pairs.at<int>(x.first, 0) = x.first;
        c_assigned_pairs.at<int>(x.first, 1) = x.second;
    }
}

ByteTracker::ByteTracker()
{
    //nothing
}

ByteTracker::~ByteTracker()
{
    //nothing
}

ByteTracker::Params::Params()
{
    frameRate = 30;
    frameBuffer = 30;
}

class ByteTrackerImpl : public ByteTracker
{
public:
    ByteTrackerImpl(const ByteTracker::Params& parameters) : params(parameters)
    {
        trackThreshold = 0.5f;
        matchThreshold = 0.8f;
        lastId = 0;
        frame = 0;
        maxTimeLost = static_cast<int>(params.frameRate) / 30.0f * params.frameBuffer;
    }

    void init(InputArray image, Rect& boundingBox);
    bool update(InputArray inputDetections,CV_OUT OutputArray& outputTracks) CV_OVERRIDE;

    void update(const std::vector<Detection>& detections, CV_OUT std::vector<Track>& tracks) CV_OVERRIDE;
    int getFrame();
    void incrementFrame();
    std::map<int, int> lapjv(InputArray &cost);
    Mat getCostMatrix(const cv::Mat, const cv::Mat) CV_OVERRIDE;

protected:
    ByteTracker::Params params;
    float trackThreshold;
    float matchThreshold;
    std::unordered_map<int, Strack> trackedStracks_;
    std::unordered_map<int, Strack> lostStracks_;
    int lastId;
    int frame;
    float maxTimeLost;

    void getDetections(std::vector<Detection> inputObjects, std::vector<Strack>& detections,
        std::vector<Strack>& detectionsLow);


    void addNewDetectedTracks(std::unordered_map<int, Strack> trackedMap,
        std::vector<Strack>& inactiveStracks, std::vector<Strack>& trackedStracks);


    Mat getCostMatrix(const std::vector<Strack>& tracks, const std::vector<Strack>& btracks);
    Mat getCostMatrix(const std::unordered_map<int, Strack>& atracks,const std::vector<Strack> &btracks);


    Mat calculateIous(const std::vector<Rect2f>& atlwhs,const std::vector<Rect2f>& btlwhs);


    std::unordered_map<int, Strack> joinStracks(const std::vector<Strack>& trackA, std::vector<Strack>& trackB);
    std::unordered_map<int, Strack> joinStracks(const std::vector<Strack>& trackVector,
                                                std::unordered_map<int, Strack>& trackMap, bool inplace);

    std::unordered_map<int, Strack> vectorToMap(const std::vector<Strack>& stracks);
    static bool compareTracksByTrackId(const Track& track1, const Track& track2);
};

Ptr<ByteTracker> ByteTracker::create(const ByteTracker::Params& parameters)
{
    return makePtr<ByteTrackerImpl>(parameters);
}


bool ByteTrackerImpl::update(InputArray inputDetections,CV_OUT OutputArray& outputTracks)
{
    Mat dets = inputDetections.getMat();
    std::vector<Detection> detections;
    std::vector<Track> tracks;

    for (int i = 0; i < dets.rows; i++)
    {
        Rect2f box;
        float score;
        int classId;

        box.x = dets.at<float>(i, 0);
        box.y = dets.at<float>(i, 1);
        box.width = dets.at<float>(i, 2);
        box.height = dets.at<float>(i, 3);
        classId = static_cast<int>(dets.at<float>(i, 4));
        score = dets.at<float>(i, 5);

        Detection detection(box, classId, score);
        detections.push_back(detection);
    }
    ByteTrackerImpl::update(detections, tracks);

    cv::Mat trackData(static_cast<int>(tracks.size()), 7, CV_32F);
    int row = 0;
    for (auto &track : tracks)
    {
        float* data = trackData.ptr<float>(row);
        Rect2f tlwh = track.rect;
        data[0] = tlwh.x;
        data[1] = tlwh.y;
        data[2] = tlwh.width;
        data[3] = tlwh.height;
        data[4] = static_cast<float>(track.classLabel);
        data[5] = track.classScore;
        data[6] = static_cast<float>(track.trackingId);

        ++row;
    }

    trackData.copyTo(outputTracks);

    return true;
}

void ByteTrackerImpl::update(const std::vector<Detection>& inputDetections, CV_OUT std::vector<Track>& tracks)
{
    std::vector<Strack> detections;
    std::vector<Strack> detectionsLow;
    std::vector<Strack> remainDets;
    std::vector<Strack> activatedStracks;
    std::vector<Strack> reRemainTracks;

    getDetections(inputDetections, detections, detectionsLow); // objects -> D and Dlow
    std::vector<Strack> inactiveStracks;
    std::vector<Strack> trackedStracks;

    // trackedStracks_ -> inactive and active
    addNewDetectedTracks(trackedStracks_, inactiveStracks, trackedStracks);

    std::unordered_map<int, Strack> strackPool;
    strackPool = joinStracks(trackedStracks, lostStracks_, false);
    // remember that in the first association we consider the lost tracks too
    // we need to predict the tracks to do association
    // it updates strackPool with prediction
    for (auto& pair : strackPool)
    {
        Strack& track = pair.second;
        cv::Rect2f prediction = track.predict(); // cx cy w h
        prediction.x -= prediction.width;
        prediction.y -= prediction.height;
        track.setTlwh(prediction);
    }

    // getting map keys from the indexes
    std::unordered_map<int, int> indexToKey;
    int index = 0;
    for (const auto &pair : strackPool)
    {
        int key = pair.first;
        indexToKey[index] = key;
        ++index;
    }

    // First association with IoU
    //To do: add enum for cost type
    Mat dists; // IoU distances
    dists = getCostMatrix(strackPool, detections);

    std::vector<Strack> remainTracks;
    std::vector<int> strackIndex;
    std::vector<int> detectionsIndex;
    std::map<int, int> matches;

    matches = this->lapjv(dists); // returns a map of matched indexes (track,detection)

    // Find unmatched track indexes
    for (
        int trackIndex = 0;
        trackIndex < static_cast<int>(strackPool.size());                  ++trackIndex)
    {
        if (matches.find(trackIndex) == matches.end())
        {
            strackIndex.push_back(trackIndex);
        }
    }

    // Find unmatched detection indexes
    for (
        int detectionIndex = 0;
        detectionIndex < static_cast<int>(detections.size());
        ++detectionIndex)
    {
        bool matched = false;
        for (const auto &match : matches)
        {
            int matchedDetectionIndex = match.second;
            if (detectionIndex == matchedDetectionIndex)
            {
                matched = true;
                break;
            }
        }
        if (!matched)
        {
            detectionsIndex.push_back(detectionIndex);
        }
    }

    // remaining tracks and dets
    for (size_t i = 0; i < strackIndex.size(); i++)
    {
        int key = indexToKey[strackIndex[i]];
        Strack track = strackPool[key];
        remainTracks.push_back(track);
    }
    for (size_t j = 0; j < detectionsIndex.size(); j++)
    {
        remainDets.push_back(detections[detectionsIndex[j]]);
    }

    //Matched tracks-dets
    for (auto &pair : matches)
    {
        int key = indexToKey[pair.first];
        Strack &track = strackPool[key];
        Strack &detection = detections[pair.second];

        // if it's tracked, update it, else reactivate it
        if (track.getState() == TrackState::TRACKED)
        {
            track.update(detection);
            activatedStracks.push_back(track);
        }
        else
        {
            track.reactivate(detection, getFrame());
            activatedStracks.push_back(track);
            lostStracks_.erase(track.getId());
        }
    }

    //Second association for low score detections
    dists = getCostMatrix(remainTracks, detectionsLow);
    strackIndex.clear();
    detectionsIndex.clear();
    matches = lapjv(dists);

    for (
        int trackIndex = 0;
        trackIndex < static_cast<int>(remainTracks.size());
        ++trackIndex)
    {
        if (matches.find(trackIndex) == matches.end())
        {
            strackIndex.push_back(trackIndex);
        }
    }

    for (size_t i = 0; i < strackIndex.size(); i++)
    {
        reRemainTracks.push_back(remainTracks[strackIndex[i]]);
    }

    for (auto pair : matches)
    {
        Strack &track = remainTracks[pair.first];
        Strack &detection = detectionsLow[pair.second];

        if (track.getState() == TrackState::TRACKED)
        {
            track.update(detection);
            activatedStracks.push_back(track);
        }
        else
        {
            track.reactivate(detection, frame);
            activatedStracks.push_back(track);
            lostStracks_.erase(track.getId());
        }
    }

    //Deal with unconfirmed tracks, usually tracks with only one beginning frame
    dists = getCostMatrix(inactiveStracks, remainDets);
    strackIndex.clear();
    detectionsIndex.clear();
    matches = lapjv(dists);

    for (auto pair : matches)
    {
        Strack &track = inactiveStracks[pair.first];
        Strack &detection = remainDets[pair.second];
        track.reactivate(detection,getFrame());
        activatedStracks.push_back(track);
    }

    // initialize new tracks
    for (size_t i = 0; i < remainDets.size(); i++)
    {
        Strack newTrack = remainDets[i];

        if (newTrack.getScore() < trackThreshold + 0.1)
        {
            continue;
        }

        newTrack.activate(getFrame(), lastId++);
        activatedStracks.push_back(newTrack);
    }

    trackedStracks_ = vectorToMap(activatedStracks);
    lostStracks_ = vectorToMap(reRemainTracks);

    // deal with lost tracks and save them in an attribute
    std::vector<int> keysToRemove;
    for (auto& pair : lostStracks_)
    {
        Strack& track = pair.second;
        if (track.getState() != TrackState::LOST) //fist time that it enters the map
        {
            track.setTrackletLen(1);
            track.setState(TrackState::LOST);
        }
        else
            track.incrementTrackletLen();

        if ((track.getTrackletLen()) > maxTimeLost)
            keysToRemove.push_back(pair.first);
    }

    for (int key : keysToRemove)
    {
        lostStracks_.erase(key);
    }
    if (!tracks.empty())
    {
        tracks.clear();
    }

    for (auto& strack : activatedStracks)
    {
        Track track(strack.getTlwh(), strack.getId(), strack.getClass(), strack.getScore());
        tracks.push_back(track);
    }

    for (auto& pair: lostStracks_)
    {
        Strack& strack = pair.second;
        Track track(strack.getTlwh(), strack.getId(), strack.getClass(), strack.getScore());
        tracks.push_back(track);
    }

    std::sort(tracks.begin(), tracks.end(), compareTracksByTrackId);
}

void ByteTrackerImpl::getDetections(std::vector<Detection> inputObjects, std::vector<Strack>& detections,
    std::vector<Strack>& detectionsLow)
{
    for (const Detection& detection : inputObjects)
    {
        Rect2f box = detection.rect;
        int classId = detection.classLabel;
        float score = detection.classScore;

        Strack strack(box, classId, score);
        if (score >= trackThreshold) // + 0.05 Dhigh or Dlow
        {
            detections.push_back(strack);
        }
        else if (score > 0.1)
        {
            detectionsLow.push_back(strack);
        }
    }
}

void ByteTrackerImpl::addNewDetectedTracks(std::unordered_map<int, Strack> trackedMap,
    std::vector<Strack> &inactiveStracks, std::vector<Strack> &trackedStracks)
{
    // checks if the trackedStracks are activated to keep them in the std::vector(same name)
    for (auto pair : trackedMap)
    {
        Strack track = pair.second;
        if (track.getState() == TrackState::TRACKED)
            trackedStracks.push_back(track);
        else
            inactiveStracks.push_back(track);
    }
}

Mat ByteTrackerImpl::getCostMatrix(const std::vector<Strack> &atracks, const std::vector<Strack> &btracks)
{
    Mat costMatrix;
    if (atracks.size() == 0 || btracks.size() == 0)
    {
        return costMatrix; // returns empty matrix
    }
    std::vector<Rect2f> atlwhs,btlwhs;
    for (auto& track : atracks)
    {
        atlwhs.push_back(track.getTlwh());
    }
    for (auto& track : btracks)
    {
        btlwhs.push_back(track.getTlwh());
    }

    costMatrix = calculateIous(atlwhs, btlwhs);
    subtract(1, costMatrix, costMatrix); //costMatrix = 1 - costMatrix

    return costMatrix;
}

//overload
Mat ByteTrackerImpl::getCostMatrix(const std::unordered_map<int, Strack> &atracks, const std::vector<Strack> &btracks)
{
    Mat costMatrix;
    if (atracks.size() == 0 && btracks.size() == 0)
    {
        return costMatrix;
    }

    std::vector<Rect2f> atlwhs, btlwhs;
    for (auto &pair : atracks)
    {
        Rect2f tlwh = pair.second.getTlwh();
        atlwhs.push_back(tlwh);
    }

    for (auto &track : btracks)
    {
        Rect2f tlwh = track.getTlwh();
        btlwhs.push_back(tlwh);
    }

    costMatrix = calculateIous(atlwhs, btlwhs);
    std::cout << "cost matrix: " << std::endl << costMatrix << std::endl;
    subtract(1, costMatrix, costMatrix); //costMatrix = 1 - costMatrix

    return costMatrix;
}

Mat ByteTrackerImpl::getCostMatrix(const Mat amat, const Mat bmat)
{
    Mat costMatrix;
    if (amat.rows == 0 && bmat.rows == 0)
    {
        return costMatrix;
    }

    std::vector<Rect2f> atlwhs, btlwhs;
    for (int i = 0; i < amat.rows; i++)
    {
        cv::Rect2f rect(
            amat.at<float>(0, 0), // x
            amat.at<float>(0, 1), // y
            amat.at<float>(0, 2), // width
            amat.at<float>(0, 3)  // height
        );
        Rect2f tlwh = rect;
        atlwhs.push_back(tlwh);
    }

    for (int i = 0; i < bmat.rows; i++)
    {
        cv::Rect2f rect(
            bmat.at<float>(0, 0), // x
            bmat.at<float>(0, 1), // y
            bmat.at<float>(0, 2), // width
            bmat.at<float>(0, 3)  // height
        );
        Rect2f tlwh = rect;
        btlwhs.push_back(tlwh);
    }

    costMatrix = calculateIous(atlwhs, btlwhs);
    subtract(1, costMatrix, costMatrix); //costMatrix = 1 - costMatrix

    return costMatrix;
}


Mat ByteTrackerImpl::calculateIous(const std::vector<Rect2f> &atlwhs, const std::vector<Rect2f> &btlwhs)
{
    Mat iousMatrix;
    if (atlwhs.empty() || btlwhs.empty())
    {
        printf("one of them is empty\n");
        return iousMatrix;
    }
    int m = static_cast<int>(atlwhs.size());
    int n = static_cast<int>(btlwhs.size());

    iousMatrix.create(m, n, CV_32F);

    // bbox_ious
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cv::Rect2f intersection = atlwhs[i] & btlwhs[j];
            cv::Rect2f unionRect = atlwhs[i] | btlwhs[j];
            float intersectionArea = intersection.area();
            float unionArea = unionRect.area();
            iousMatrix.at<float>(i, j) = intersectionArea / unionArea;
        }
    }

    return iousMatrix;
}


std::unordered_map<int, Strack> ByteTrackerImpl::joinStracks(
    const std::vector<Strack>& trackA, std::vector<Strack>& trackB)
{
    std::unordered_map<int, Strack> joinedTracks;

    for (const auto &track : trackA)
    {
        joinedTracks.emplace(track.getId(), track);
    }

    for (const auto &track : trackB)
    {
        joinedTracks.emplace(track.getId(), track);
    }

    return joinedTracks;
}

// overload to receive a hashmap
std::unordered_map<int, Strack> ByteTrackerImpl::joinStracks(const std::vector<Strack>& trackVector,
    std::unordered_map<int, Strack>& trackMap, bool inplace)
{
    if (inplace)
    {
        for (const auto& track : trackVector)
        {
            trackMap[track.getId()] = track;
        }
        return trackMap;
    }

    std::unordered_map<int, Strack> joinedTracks = trackMap;
    for (const auto &track : trackVector)
    {
        joinedTracks.emplace(track.getId(), track);
    }

    return joinedTracks;

}

std::map<int, int> ByteTrackerImpl::lapjv(InputArray &cost)
{
    Mat _cost = cost.getMat();
    std::map<int, int> ret;
    if (_cost.rows == 0 || _cost.cols == 0)
        return ret;
    int maxI = _cost.rows;
    int maxJ = _cost.cols;
    int n = max(maxJ, maxI);

    std::vector<std::vector<double>> cost_ptr(n, std::vector<double>(n));
    std::vector<int> x_c(n);
    std::vector<int> y_c(n);

    std::vector<double*> cost_ptr_ptr(n);
    for (int i=0; i < n; i++)
    {
        cost_ptr_ptr[i] = cost_ptr[i].data();
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < maxI && j < maxJ && _cost.at<float>(i, j) < matchThreshold)
            {
                cost_ptr[i][j] = static_cast<double>(_cost.at<float>(i, j));
            }
            else
            {
                cost_ptr[i][j] = LARGE;
            }
        }
        x_c[i] = -1;
        y_c[i] = -1;
    }
    lapjv_internal(n, cost_ptr_ptr.data(), x_c.data(), y_c.data());
    for (int i = 0; i < n; i++)
    {
        if (i < maxI && x_c[i] < maxJ)
        {
            ret[i] = x_c[i];
        }
    }

    return ret;
}

int ByteTrackerImpl::getFrame()
{
    return frame;
}

void ByteTrackerImpl::incrementFrame()
{
    frame++;
}

std::unordered_map<int, Strack> ByteTrackerImpl::vectorToMap(const std::vector<Strack>& stracks)
{
    std::unordered_map<int, Strack> strackMap;
    for (const Strack& strack : stracks)
    {
        int id = strack.getId();
        strackMap.emplace(id, strack);
    }
    return strackMap;
}

bool ByteTrackerImpl::compareTracksByTrackId(const Track& track1, const Track& track2) {
    return track1.trackingId< track2.trackingId;
}

}// namespace cv
