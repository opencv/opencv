#pragma once

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>

class DetectionBasedTracker
{
    public:
        struct Parameters
        {
            int minObjectSize;
            int maxObjectSize;
            double scaleFactor;
            int maxTrackLifetime;
            int minNeighbors;
            int minDetectionPeriod; //the minimal time between run of the big object detector (on the whole frame) in ms (1000 mean 1 sec), default=0

            Parameters();
        };

        DetectionBasedTracker(const std::string& cascadeFilename, const Parameters& params);
        virtual ~DetectionBasedTracker();

        virtual bool run();
        virtual void stop();
        virtual void resetTracking();

        virtual void process(const cv::Mat& imageGray);

        bool setParameters(const Parameters& params);
        const Parameters& getParameters();


        typedef std::pair<cv::Rect, int> Object;
        virtual void getObjects(std::vector<cv::Rect>& result) const;
        virtual void getObjects(std::vector<Object>& result) const;

    protected:
        class SeparateDetectionWork;
        cv::Ptr<SeparateDetectionWork> separateDetectionWork;
        friend void* workcycleObjectDetectorFunction(void* p);


        struct InnerParameters
        {
            int numLastPositionsToTrack;
            int numStepsToWaitBeforeFirstShow;
            int numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown;
            int numStepsToShowWithoutDetecting;

            float coeffTrackingWindowSize;
            float coeffObjectSizeToTrack;
            float coeffObjectSpeedUsingInPrediction;

            InnerParameters();
        };
        Parameters parameters;
        InnerParameters innerParameters;

        struct TrackedObject
        {
            typedef std::vector<cv::Rect> PositionsVector;

            PositionsVector lastPositions;

            int numDetectedFrames;
            int numFramesNotDetected;
            int id;

            TrackedObject(const cv::Rect& rect):numDetectedFrames(1), numFramesNotDetected(0)
            {
                lastPositions.push_back(rect);
                id=getNextId();
            };

            static int getNextId()
            {
                static int _id=0;
                return _id++;
            }
        };

        int numTrackedSteps;
        std::vector<TrackedObject> trackedObjects;

        std::vector<float> weightsPositionsSmoothing;
        std::vector<float> weightsSizesSmoothing;

        cv::CascadeClassifier cascadeForTracking;


        void updateTrackedObjects(const std::vector<cv::Rect>& detectedObjects);
        cv::Rect calcTrackedObjectPositionToShow(int i) const;
        void detectInRegion(const cv::Mat& img, const cv::Rect& r, std::vector<cv::Rect>& detectedObjectsInRegions);
};

namespace cv
{
    using ::DetectionBasedTracker;
} //end of cv namespace

#endif
