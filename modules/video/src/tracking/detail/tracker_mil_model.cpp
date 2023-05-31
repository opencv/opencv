// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "tracker_mil_model.hpp"

/**
 * TrackerMILModel
 */

namespace cv {
inline namespace tracking {
namespace impl {

TrackerMILModel::TrackerMILModel(const Rect& boundingBox)
{
    currentSample.clear();
    mode = MODE_POSITIVE;
    width = boundingBox.width;
    height = boundingBox.height;

    Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> initState = Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState>(
            new TrackerStateEstimatorMILBoosting::TrackerMILTargetState(Point2f((float)boundingBox.x, (float)boundingBox.y), boundingBox.width, boundingBox.height,
                    true, Mat()));
    trajectory.push_back(initState);
}

void TrackerMILModel::responseToConfidenceMap(const std::vector<Mat>& responses, ConfidenceMap& confidenceMap)
{
    if (currentSample.empty())
    {
        CV_Error(-1, "The samples in Model estimation are empty");
    }

    for (size_t i = 0; i < responses.size(); i++)
    {
        //for each column (one sample) there are #num_feature
        //get informations from currentSample
        for (int j = 0; j < responses.at(i).cols; j++)
        {

            Size currentSize;
            Point currentOfs;
            currentSample.at(j).locateROI(currentSize, currentOfs);
            bool foreground = false;
            if (mode == MODE_POSITIVE || mode == MODE_ESTIMATON)
            {
                foreground = true;
            }
            else if (mode == MODE_NEGATIVE)
            {
                foreground = false;
            }

            //get the column of the HAAR responses
            Mat singleResponse = responses.at(i).col(j);

            //create the state
            Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> currentState = Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState>(
                    new TrackerStateEstimatorMILBoosting::TrackerMILTargetState(currentOfs, width, height, foreground, singleResponse));

            confidenceMap.push_back(std::make_pair(currentState, 0.0f));
        }
    }
}

void TrackerMILModel::modelEstimationImpl(const std::vector<Mat>& responses)
{
    responseToConfidenceMap(responses, currentConfidenceMap);
}

void TrackerMILModel::modelUpdateImpl()
{
}

void TrackerMILModel::setMode(int trainingMode, const std::vector<Mat>& samples)
{
    currentSample.clear();
    currentSample = samples;

    mode = trainingMode;
}

}}}  // namespace cv::tracking::impl
