#pragma once

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(__ANDROID__)

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>

namespace cv
{
class CV_EXPORTS DetectionBasedTracker
{
    public:
        struct Parameters
        {
            int maxTrackLifetime;
            int minDetectionPeriod; //the minimal time between run of the big object detector (on the whole frame) in ms (1000 mean 1 sec), default=0

            Parameters();
        };

        class IDetector
        {
            public:
                IDetector():
                    minObjSize(96, 96),
                    maxObjSize(INT_MAX, INT_MAX),
                    minNeighbours(2),
                    scaleFactor(1.1f)
                {}

                virtual void detect(const cv::Mat& image, std::vector<cv::Rect>& objects) = 0;

                void setMinObjectSize(const cv::Size& min)
                {
                    minObjSize = min;
                }
                void setMaxObjectSize(const cv::Size& max)
                {
                    maxObjSize = max;
                }
                cv::Size getMinObjectSize() const
                {
                    return minObjSize;
                }
                cv::Size getMaxObjectSize() const
                {
                    return maxObjSize;
                }
                float getScaleFactor()
                {
                    return scaleFactor;
                }
                void setScaleFactor(float value)
                {
                    scaleFactor = value;
                }
                int getMinNeighbours()
                {
                    return minNeighbours;
                }
                void setMinNeighbours(int value)
                {
                    minNeighbours = value;
                }
                virtual ~IDetector() {}

            protected:
                cv::Size minObjSize;
                cv::Size maxObjSize;
                int minNeighbours;
                float scaleFactor;
        };

        DetectionBasedTracker(cv::Ptr<IDetector> mainDetector, cv::Ptr<IDetector> trackingDetector, const Parameters& params);
        virtual ~DetectionBasedTracker();

        virtual bool run();
        virtual void stop();
        virtual void resetTracking();

        virtual void process(const cv::Mat& imageGray);

        bool setParameters(const Parameters& params);
        const Parameters& getParameters() const;


        typedef std::pair<cv::Rect, int> Object;
        virtual void getObjects(std::vector<cv::Rect>& result) const;
        virtual void getObjects(std::vector<Object>& result) const;

        enum ObjectStatus
        {
            DETECTED_NOT_SHOWN_YET,
            DETECTED,
            DETECTED_TEMPORARY_LOST,
            WRONG_OBJECT
        };
        struct ExtObject
        {
            int id;
            cv::Rect location;
            ObjectStatus status;
            ExtObject(int _id, cv::Rect _location, ObjectStatus _status)
                :id(_id), location(_location), status(_status)
            {
            }
        };
        virtual void getObjects(std::vector<ExtObject>& result) const;


        virtual int addObject(const cv::Rect& location); //returns id of the new object

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

        cv::Ptr<IDetector> cascadeForTracking;

        void updateTrackedObjects(const std::vector<cv::Rect>& detectedObjects);
        cv::Rect calcTrackedObjectPositionToShow(int i) const;
        cv::Rect calcTrackedObjectPositionToShow(int i, ObjectStatus& status) const;
        void detectInRegion(const cv::Mat& img, const cv::Rect& r, std::vector<cv::Rect>& detectedObjectsInRegions);
};
} //end of cv namespace
#endif
