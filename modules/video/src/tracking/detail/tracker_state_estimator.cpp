// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.private.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerStateEstimator::~TrackerStateEstimator()
{
}

Ptr<TrackerTargetState> TrackerStateEstimator::estimate(const std::vector<ConfidenceMap>& confidenceMaps)
{
    if (confidenceMaps.empty())
        return Ptr<TrackerTargetState>();

    return estimateImpl(confidenceMaps);
}

void TrackerStateEstimator::update(std::vector<ConfidenceMap>& confidenceMaps)
{
    if (confidenceMaps.empty())
        return;

    return updateImpl(confidenceMaps);
}

Ptr<TrackerStateEstimator> TrackerStateEstimator::create(const String& trackeStateEstimatorType)
{

    if (trackeStateEstimatorType.find("SVM") == 0)
    {
        return Ptr<TrackerStateEstimatorSVM>(new TrackerStateEstimatorSVM());
    }

    if (trackeStateEstimatorType.find("BOOSTING") == 0)
    {
        return Ptr<TrackerStateEstimatorMILBoosting>(new TrackerStateEstimatorMILBoosting());
    }

    CV_Error(-1, "Tracker state estimator type not supported");
}

String TrackerStateEstimator::getClassName() const
{
    return className;
}

/**
 * TrackerStateEstimatorMILBoosting::TrackerMILTargetState
 */
TrackerStateEstimatorMILBoosting::TrackerMILTargetState::TrackerMILTargetState(const Point2f& position, int width, int height, bool foreground,
        const Mat& features)
{
    setTargetPosition(position);
    setTargetWidth(width);
    setTargetHeight(height);
    setTargetFg(foreground);
    setFeatures(features);
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setTargetFg(bool foreground)
{
    isTarget = foreground;
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setFeatures(const Mat& features)
{
    targetFeatures = features;
}

bool TrackerStateEstimatorMILBoosting::TrackerMILTargetState::isTargetFg() const
{
    return isTarget;
}

Mat TrackerStateEstimatorMILBoosting::TrackerMILTargetState::getFeatures() const
{
    return targetFeatures;
}

TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting(int nFeatures)
{
    className = "BOOSTING";
    trained = false;
    numFeatures = nFeatures;
}

TrackerStateEstimatorMILBoosting::~TrackerStateEstimatorMILBoosting()
{
}

void TrackerStateEstimatorMILBoosting::setCurrentConfidenceMap(ConfidenceMap& confidenceMap)
{
    currentConfidenceMap.clear();
    currentConfidenceMap = confidenceMap;
}

uint TrackerStateEstimatorMILBoosting::max_idx(const std::vector<float>& v)
{
    const float* findPtr = &(*std::max_element(v.begin(), v.end()));
    const float* beginPtr = &(*v.begin());
    return (uint)(findPtr - beginPtr);
}

Ptr<TrackerTargetState> TrackerStateEstimatorMILBoosting::estimateImpl(const std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
    //run ClfMilBoost classify in order to compute next location
    if (currentConfidenceMap.empty())
        return Ptr<TrackerTargetState>();

    Mat positiveStates;
    Mat negativeStates;

    prepareData(currentConfidenceMap, positiveStates, negativeStates);

    std::vector<float> prob = boostMILModel.classify(positiveStates);

    int bestind = max_idx(prob);
    //float resp = prob[bestind];

    return currentConfidenceMap.at(bestind).first;
}

void TrackerStateEstimatorMILBoosting::prepareData(const ConfidenceMap& confidenceMap, Mat& positive, Mat& negative)
{

    int posCounter = 0;
    int negCounter = 0;

    for (size_t i = 0; i < confidenceMap.size(); i++)
    {
        Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at(i).first.staticCast<TrackerMILTargetState>();
        if (currentTargetState->isTargetFg())
            posCounter++;
        else
            negCounter++;
    }

    positive.create(posCounter, numFeatures, CV_32FC1);
    negative.create(negCounter, numFeatures, CV_32FC1);

    //TODO change with mat fast access
    //initialize trainData (positive and negative)

    int pc = 0;
    int nc = 0;
    for (size_t i = 0; i < confidenceMap.size(); i++)
    {
        Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at(i).first.staticCast<TrackerMILTargetState>();
        Mat stateFeatures = currentTargetState->getFeatures();

        if (currentTargetState->isTargetFg())
        {
            for (int j = 0; j < stateFeatures.rows; j++)
            {
                //fill the positive trainData with the value of the feature j for sample i
                positive.at<float>(pc, j) = stateFeatures.at<float>(j, 0);
            }
            pc++;
        }
        else
        {
            for (int j = 0; j < stateFeatures.rows; j++)
            {
                //fill the negative trainData with the value of the feature j for sample i
                negative.at<float>(nc, j) = stateFeatures.at<float>(j, 0);
            }
            nc++;
        }
    }
}

void TrackerStateEstimatorMILBoosting::updateImpl(std::vector<ConfidenceMap>& confidenceMaps)
{

    if (!trained)
    {
        //this is the first time that the classifier is built
        //init MIL
        boostMILModel.init();
        trained = true;
    }

    ConfidenceMap lastConfidenceMap = confidenceMaps.back();
    Mat positiveStates;
    Mat negativeStates;

    prepareData(lastConfidenceMap, positiveStates, negativeStates);
    //update MIL
    boostMILModel.update(positiveStates, negativeStates);
}

/**
 * TrackerStateEstimatorAdaBoosting
 */
TrackerStateEstimatorAdaBoosting::TrackerStateEstimatorAdaBoosting(int numClassifer, int initIterations, int nFeatures, Size patchSize, const Rect& ROI)
{
    className = "ADABOOSTING";
    numBaseClassifier = numClassifer;
    numFeatures = nFeatures;
    iterationInit = initIterations;
    initPatchSize = patchSize;
    trained = false;
    sampleROI = ROI;
}

Rect TrackerStateEstimatorAdaBoosting::getSampleROI() const
{
    return sampleROI;
}

void TrackerStateEstimatorAdaBoosting::setSampleROI(const Rect& ROI)
{
    sampleROI = ROI;
}

/**
 * TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState
 */
TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState(const Point2f& position, int width, int height,
        bool foreground, const Mat& responses)
{
    setTargetPosition(position);
    setTargetWidth(width);
    setTargetHeight(height);

    setTargetFg(foreground);
    setTargetResponses(responses);
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetFg(bool foreground)
{
    isTarget = foreground;
}

bool TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::isTargetFg() const
{
    return isTarget;
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetResponses(const Mat& responses)
{
    targetResponses = responses;
}

Mat TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::getTargetResponses() const
{
    return targetResponses;
}

TrackerStateEstimatorAdaBoosting::~TrackerStateEstimatorAdaBoosting()
{
}
void TrackerStateEstimatorAdaBoosting::setCurrentConfidenceMap(ConfidenceMap& confidenceMap)
{
    currentConfidenceMap.clear();
    currentConfidenceMap = confidenceMap;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeReplacedClassifier()
{
    return replacedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSwappedClassifier()
{
    return swappedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSelectedWeakClassifier()
{
    return boostClassifier->getSelectedWeakClassifier();
}

Ptr<TrackerTargetState> TrackerStateEstimatorAdaBoosting::estimateImpl(const std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
    //run classify in order to compute next location
    if (currentConfidenceMap.empty())
        return Ptr<TrackerTargetState>();

    std::vector<Mat> images;

    for (size_t i = 0; i < currentConfidenceMap.size(); i++)
    {
        Ptr<TrackerAdaBoostingTargetState> currentTargetState = currentConfidenceMap.at(i).first.staticCast<TrackerAdaBoostingTargetState>();
        images.push_back(currentTargetState->getTargetResponses());
    }

    int bestIndex;
    boostClassifier->classifySmooth(images, sampleROI, bestIndex);

    // get bestIndex from classifySmooth
    return currentConfidenceMap.at(bestIndex).first;
}

void TrackerStateEstimatorAdaBoosting::updateImpl(std::vector<ConfidenceMap>& confidenceMaps)
{
    if (!trained)
    {
        //this is the first time that the classifier is built
        int numWeakClassifier = numBaseClassifier * 10;

        bool useFeatureExchange = true;
        boostClassifier = Ptr<StrongClassifierDirectSelection>(
                new StrongClassifierDirectSelection(numBaseClassifier, numWeakClassifier, initPatchSize, sampleROI, useFeatureExchange, iterationInit));
        //init base classifiers
        boostClassifier->initBaseClassifier();

        trained = true;
    }

    ConfidenceMap lastConfidenceMap = confidenceMaps.back();
    bool featureEx = boostClassifier->getUseFeatureExchange();

    replacedClassifier.clear();
    replacedClassifier.resize(lastConfidenceMap.size(), -1);
    swappedClassifier.clear();
    swappedClassifier.resize(lastConfidenceMap.size(), -1);

    for (size_t i = 0; i < lastConfidenceMap.size() / 2; i++)
    {
        Ptr<TrackerAdaBoostingTargetState> currentTargetState = lastConfidenceMap.at(i).first.staticCast<TrackerAdaBoostingTargetState>();

        int currentFg = 1;
        if (!currentTargetState->isTargetFg())
            currentFg = -1;
        Mat res = currentTargetState->getTargetResponses();

        boostClassifier->update(res, currentFg);
        if (featureEx)
        {
            replacedClassifier[i] = boostClassifier->getReplacedClassifier();
            swappedClassifier[i] = boostClassifier->getSwappedClassifier();
            if (replacedClassifier[i] >= 0 && swappedClassifier[i] >= 0)
                boostClassifier->replaceWeakClassifier(replacedClassifier[i]);
        }
        else
        {
            replacedClassifier[i] = -1;
            swappedClassifier[i] = -1;
        }

        int mapPosition = (int)(i + lastConfidenceMap.size() / 2);
        Ptr<TrackerAdaBoostingTargetState> currentTargetState2 = lastConfidenceMap.at(mapPosition).first.staticCast<TrackerAdaBoostingTargetState>();

        currentFg = 1;
        if (!currentTargetState2->isTargetFg())
            currentFg = -1;
        const Mat res2 = currentTargetState2->getTargetResponses();

        boostClassifier->update(res2, currentFg);
        if (featureEx)
        {
            replacedClassifier[mapPosition] = boostClassifier->getReplacedClassifier();
            swappedClassifier[mapPosition] = boostClassifier->getSwappedClassifier();
            if (replacedClassifier[mapPosition] >= 0 && swappedClassifier[mapPosition] >= 0)
                boostClassifier->replaceWeakClassifier(replacedClassifier[mapPosition]);
        }
        else
        {
            replacedClassifier[mapPosition] = -1;
            swappedClassifier[mapPosition] = -1;
        }
    }
}

/**
 * TrackerStateEstimatorSVM
 */
TrackerStateEstimatorSVM::TrackerStateEstimatorSVM()
{
    className = "SVM";
}

TrackerStateEstimatorSVM::~TrackerStateEstimatorSVM()
{
}

Ptr<TrackerTargetState> TrackerStateEstimatorSVM::estimateImpl(const std::vector<ConfidenceMap>& confidenceMaps)
{
    return confidenceMaps.back().back().first;
}

void TrackerStateEstimatorSVM::updateImpl(std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
}

}}}  // namespace cv::detail::tracking
