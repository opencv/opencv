// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.



#include "../precomp.hpp"

//#include "opencv2/video/detail/bytetracker.hpp"
// /#include "opencv2/video/detail/bytetracker_strack.hpp"
#include "../lapjv/lapjv.hpp"
//#include "opencv2/video/detail/tracking.detail.hpp"
#include <map>
#include <unordered_map>

// #include "detail/bytetracker.hpp"
// #include "detail/bytetracker_strack.hpp"
// #include "detail/lapjv.hpp"

using namespace std;
using namespace cv;

namespace cv {

//using cv::detail::tracking::Strack;
//using cv::detail::tracking::Detection;
//using cv::detail::tracking::TrackState;

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
    //ByteTracker(int, int);
    ByteTrackerImpl(const ByteTracker::Params& parameters) : params_(parameters)
    {
        trackThreshold_ = 0.5f;
        matchThreshold_ = 0.7f;
        lastId_ = 0;
        frame_ = 0;
        maxTimeLost_ = static_cast<int>(params_.frameRate / 30.0f * params_.frameBuffer);
    }

    void init(InputArray image, Rect& boundingBox);
    //std::vector<std::vector<float>> update(std::vector<std::vector<float>>)
    bool update(InputArray inputDetections,CV_OUT OutputArray& outputTracks) CV_OVERRIDE;
    //Scalar get_color(int idx);
    int getFrame();
    void incrementFrame();
    map<int, int> lapjv(InputArray &cost);

protected:
    ByteTracker::Params params_;
    float trackThreshold_;
    float matchThreshold_;
    unordered_map<int, Strack> trackedStracks_;
    unordered_map<int, Strack> lostStracks_;
    int lastId_;
    int frame_;
    int maxTimeLost_;

    void getDetections(InputArray& inputObjects, vector<Strack>& detections, 
        vector<Strack>& detectionsLow);


    void addNewDetectedTracks(unordered_map<int, Strack> trackedMap,
        vector<Strack>& inactiveStracks, vector<Strack>& trackedStracks);


    Mat getCostMatrix(vector<Strack>& tracks, vector<Strack>& btracks);
    Mat getCostMatrix(unordered_map<int, Strack>& atracks, vector<Strack> &btracks);

        
    Mat calculateIous(vector<Rect>& atlwhs, vector<Rect> &btlwhs);


    unordered_map<int, Strack> joinStracks(const vector<Strack>& trackA, vector<Strack>& trackB);
    unordered_map<int, Strack> joinStracks(const vector<Strack>& trackVector,
        unordered_map<int, Strack>& trackMap, bool inplace);

};

Ptr<ByteTracker> ByteTracker::create(const ByteTracker::Params& parameters)
{
    return makePtr<ByteTrackerImpl>(parameters);
}


bool ByteTrackerImpl::update(InputArray inputDetections,CV_OUT OutputArray& outputTracks)
{
    cout<<"This is update loop";
    // Detetions, Dk = Detections(fk)
    vector<Strack> detections; // consider changing to cv::Mat_<Strack>
    vector<Strack> detectionsLow;
    vector<Strack> remainDets;
    vector<Strack> activatedStracks;
    vector<Strack> reRemainTracks;

    getDetections(inputDetections, detections, detectionsLow); // objects -> D and Dlow
    vector<Strack> inactiveStracks;
    vector<Strack> trackedStracks;

    addNewDetectedTracks(trackedStracks_, inactiveStracks, trackedStracks); // trackedStracks_ -> inactive and active
    //trackedStracks_.copyTo(outputTracks);

    unordered_map<int, Strack> strackPool;
    strackPool = joinStracks(trackedStracks, lostStracks_, false);
    // remember that in the first association we consider the lost tracks too
    // we need to predict the tracks to do association
    // it updates strackPool with prediction, maybe pass strackPool by reference
    for (auto& track : strackPool)
    {
        cv::Rect prediction = track.second.predict(); // cx cy w h
        prediction.x -= prediction.width;
        prediction.y -= prediction.height;
        track.second.setTlwh(prediction);
    }
    strackPool.copyTo(outputTracks)

    // getting map keys from the indexes
    unordered_map<int, int> indexToKey;
    int index = 0;
    for (const auto &pair : strackPool)
    {
        int key = pair.first;
        indexToKey[index] = key;
        ++index;
    }
    
    // First association with IoU
    Mat dists; // IoU distances, maybe change it to mat type?
    dists = getCostMatrix(strackPool, detections);

    vector<Strack> remainTracks;
    vector<int> strackIndex;
    vector<int> detectionsIndex;
    map<int, int> matches;

    matches = lapjv(dists); // returns a map (track_i,matched_det_index)
    // cout << "\n Num of matches: " << matches.size();

    // Find unmatched track indexes
    for (int trackIndex = 0; trackIndex < static_cast<int>(strackPool.size()); ++trackIndex)
    {
        if (matches.find(trackIndex) == matches.end())
        {
            strackIndex.push_back(trackIndex);
        }
    }

    // Find unmatched detection indexes
    for (int detectionIndex = 0; detectionIndex < static_cast<int>(detections.size());
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

    // remain tracks and dets
    for (int i = 0; i < static_cast<int>(strackIndex.size()); i++)
    {
        int key = indexToKey[strackIndex[i]];
        Strack track = strackPool[key];
        remainTracks.push_back(track);
    }
    for (int j = 0; j < static_cast<int>(detectionsIndex.size()); j++)
    {
        remainDets.push_back(detections[detectionsIndex[j]]);
    }

    for (auto &pair : matches) // row
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

    dists = getCostMatrix(remainTracks, detectionsLow);
    strackIndex.clear();
    detectionsIndex.clear();
    matches = lapjv(dists);

    // cout << "\n Num of low matches: " << matches.size();

    // Find unmatched track indexes
    for (int trackIndex = 0; trackIndex < static_cast<int>(remainTracks.size()); ++trackIndex)
    {
        if (matches.find(trackIndex) == matches.end())
        {
            strackIndex.push_back(trackIndex);
        }
    }


    for (int i = 0; i < static_cast<int>(strackIndex.size()); i++)
    {
        reRemainTracks.push_back(remainTracks[strackIndex[i]]);
    }

    for (auto pair : matches) // row
    {
        Strack &track = remainTracks[pair.first];
        Strack &detection = detectionsLow[pair.second];

        // if it's tracked, update it, else re_activate it
        if (track.getState() == TrackState::TRACKED)
        {
            track.update(detection);
            activatedStracks.push_back(track);
        }
        else
        {
            track.reactivate(detection, frame_);
            activatedStracks.push_back(track);
            lostStracks_.erase(track.getId());
        }
    }

    // initialize new tracks
    for (int i = 0; i < static_cast<int>(remainDets.size()); i++)
    {
        Strack newTrack = remainDets[i];
        newTrack.activate(getFrame(), lastId_++);
        activatedStracks.push_back(newTrack);
    }
    joinStracks(activatedStracks, trackedStracks_, true); //"true" means replacing in place
    joinStracks(reRemainTracks, lostStracks_, true);

    // deal with lost tracks and save them in an attribute
    vector<int> keysToRemove;
    for (auto& track : lostStracks_)
    {
        track.second.incrementTrackletLen();
        if ((track.second.getTrackletLen()) >maxTimeLost_)
            keysToRemove.push_back(track.first);
        else
            track.second.setState(TrackState::LOST);
    }

    for (int key : keysToRemove)
    {
        lostStracks_.erase(key);
    }


    cv::Mat trackData(trackedStracks_.size(), 7, CV_32F);
    int row = 0;
    for (auto &track : trackedStracks_)
    {

        float* data = trackData.ptr<float>(row);
        Rect tlwh = track.second.getTlwh();
        data[0] = tlwh.x;
        data[1] = tlwh.y;
        data[2] = tlwh.width;
        data[3] = tlwh.height;
        data[4] = static_cast<float>(track.second.getClass());
        data[5] = track.second.getScore();
        data[6] = static_cast<float>(track.second.getId());

        ++row;
    }

    //trackData.copyTo(outputTracks);

    //cout<<outputTracks;

    return true;
}

void ByteTrackerImpl::getDetections(InputArray& inputObjects, vector<Strack>& detections, 
    vector<Strack>& detectionsLow)
{
    Mat objects = inputObjects.getMat();

    incrementFrame(); // update frame
    for (int i = 0; i < objects.rows; i++)
    {
        Rect box;
        float score;
        int classId;

        box.x = objects.at<float>(i, 0);
        box.y = objects.at<float>(i, 1);
        box.width = objects.at<float>(i, 2);
        box.height = objects.at<float>(i, 3);
        classId = static_cast<int>(objects.at<float>(i, 4));
        score = objects.at<float>(i, 5);
    
        Strack strack(box, classId, score);
        if (score >= trackThreshold_) // Dhigh or Dlow
        {
            detections.push_back(strack);
        }
        else
        {
            detectionsLow.push_back(strack);
        }
    }
}

void ByteTrackerImpl::addNewDetectedTracks(unordered_map<int, Strack> trackedMap, 
    vector<Strack> &inactiveStracks, vector<Strack> &trackedStracks)
{
    // checks if the trackedStracks are activated to keep them in the vector(same name)
    for (auto pair : trackedMap)
    {
        Strack track = pair.second;
        if (track.getState() == TrackState::TRACKED)
            trackedStracks.push_back(track);
        else
            inactiveStracks.push_back(track);
    }
}

Mat ByteTrackerImpl::getCostMatrix(vector<Strack> &atracks, vector<Strack> &btracks)
{
    Mat costMatrix;
    if (atracks.size() == 0 || btracks.size() == 0)
    {
        return costMatrix; // returns empty matrix
    }

    vector<Rect> atlwhs, btlwhs;
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

Mat ByteTrackerImpl::getCostMatrix(unordered_map<int, Strack> &atracks, vector<Strack> &btracks)
{
    Mat costMatrix;
    if (atracks.size() == 0 && btracks.size() == 0)
    {
        return costMatrix; // returns empty matrix
    }

    vector<Rect> atlwhs, btlwhs;
    for (auto &pair : atracks)
    {
        Rect tlwh = pair.second.getTlwh();
        atlwhs.push_back(tlwh);
    }

    for (auto &track : btracks)
    {
        Rect tlwh = track.getTlwh();
        btlwhs.push_back(tlwh);
    }

    costMatrix = calculateIous(atlwhs, btlwhs);
    subtract(1, costMatrix, costMatrix); //costMatrix = 1 - costMatrix

    return costMatrix;
}


Mat ByteTrackerImpl::calculateIous(vector<Rect> &atlwhs, vector<Rect> &btlwhs)
{
    Mat iousMatrix;
    if (atlwhs.empty() || btlwhs.empty())
    {
        return iousMatrix;
    }

    iousMatrix.create(static_cast<int>(atlwhs.size()), static_cast<int>(btlwhs.size()), CV_32F);
    
    // bbox_ious
    for (int i = 0; i < static_cast<int>(atlwhs.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(btlwhs.size()); ++j)
        {
            cv::Rect intersection = atlwhs[i] & btlwhs[j];
            cv::Rect unionRect = atlwhs[i] | btlwhs[j];
            float intersectionArea = intersection.area();
            float unionArea = unionRect.area();
            iousMatrix.at<float>(i, j) = intersectionArea / unionArea;
        }
    }

    return iousMatrix;
}


unordered_map<int, Strack> ByteTrackerImpl::joinStracks(
    const vector<Strack>& trackA, vector<Strack>& trackB)
{
    unordered_map<int, Strack> joinedTracks;

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
unordered_map<int, Strack> ByteTrackerImpl::joinStracks(const vector<Strack>& trackVector,
    unordered_map<int, Strack>& trackMap, bool inplace)
{
    if (inplace)
    {
        for (const auto& track : trackVector)
        {
            trackMap.emplace(track.getId(), track);
        }
        return trackMap;
    }

    unordered_map<int, Strack> joinedTracks = trackMap;
    for (const auto &track : trackVector)
    {
        joinedTracks.emplace(track.getId(), track);
    }

    return joinedTracks;

}

map<int, int> ByteTrackerImpl::lapjv(InputArray &cost)
{
    Mat _cost = cost.getMat();
    map<int, int> ret;
    if (_cost.rows == 0 || _cost.cols == 0)
        return ret;
    int maxI = _cost.rows;
    int maxJ = _cost.cols;
    int n = max(maxJ, maxI);
    double **cost_ptr;
    double *u = new double[sizeof(double) * n];
    double *v = new double[sizeof(double) * n];
    int *x_c = new int[n];
    int *y_c = new int[n];
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
    {
        cost_ptr[i] = new double[sizeof(double) * n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < maxI && j < maxJ && _cost.at<double>(i, j) < matchThreshold_) // verify
            {
                cost_ptr[i][j] = (double)_cost.at<double>(i, j);
            }
            else
            {
                cost_ptr[i][j] = LARGE;
            }
        }
        x_c[i] = -1.0;
        y_c[i] = -1.0;
    }
    lapjv_internal(n, cost_ptr, x_c, y_c);
    for (int i = 0; i < n; i++)
    {
        if (i < maxI && x_c[i] < maxJ) // verify
        {
            ret[i] = x_c[i];
        }
    }

    for (int i = 0; i < n; i++)
    {
        delete[] cost_ptr[i];
    }
    delete[] cost_ptr;
    delete[] x_c;
    delete[] y_c;
    delete[] u;
    delete[] v;

    return ret;
}



/*
map<int, int> ByteTrackerImpl::lapjv(vector<vector<float>> &cost)
{
    map<int, int> ret;
    if (cost.size() == 0 || cost[0].size() == 0)
        return ret;
    int maxI = cost.size();
    int maxJ = cost[0].size();
    int n = max(maxJ, maxI);
    double **cost_ptr;
    double *u = new double[sizeof(double) * n];
    double *v = new double[sizeof(double) * n];
    int *x_c = new int[n];
    int *y_c = new int[n];
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
    {
        cost_ptr[i] = new double[sizeof(double) * n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < maxI && j < maxJ && cost[i][j] < matchThreshold_) // verify
            {
                cost_ptr[i][j] = (double)cost[i][j];
            }
            else
            {
                cost_ptr[i][j] = LARGE;
            }
        }
        x_c[i] = -1.0;
        y_c[i] = -1.0;
    }
    lapjv_internal(n, cost_ptr, x_c, y_c);
    for (int i = 0; i < n; i++)
    {
        if (i < maxI && x_c[i] < maxJ) // verify
        {
            ret[i] = x_c[i];
        }
    }

    for (int i = 0; i < n; i++)
    {
        delete[] cost_ptr[i];
    }
    delete[] cost_ptr;
    delete[] x_c;
    delete[] y_c;
    delete[] u;
    delete[] v;

    return ret;
}
*/

int ByteTrackerImpl::getFrame()
{
    return frame_;
}

void ByteTrackerImpl::incrementFrame()
{
    frame_++;
}

}
