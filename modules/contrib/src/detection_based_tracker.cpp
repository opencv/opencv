#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)
#include "opencv2/contrib/detection_based_tracker.hpp"
#include "opencv2/core/utility.hpp"

#include <pthread.h>

#if defined(DEBUG) || defined(_DEBUG)
#undef DEBUGLOGS
#define DEBUGLOGS 1
#endif

#ifndef DEBUGLOGS
#define DEBUGLOGS 0
#endif

#ifdef ANDROID
#include <android/log.h>
#define LOG_TAG "OBJECT_DETECTOR"
#define LOGD0(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI0(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGW0(...) ((void)__android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__))
#define LOGE0(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#else

#include <stdio.h>

#define LOGD0(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#define LOGI0(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#define LOGW0(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#define LOGE0(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#endif

#if DEBUGLOGS
#define LOGD(_str, ...) LOGD0(_str , ## __VA_ARGS__)
#define LOGI(_str, ...) LOGI0(_str , ## __VA_ARGS__)
#define LOGW(_str, ...) LOGW0(_str , ## __VA_ARGS__)
#define LOGE(_str, ...) LOGE0(_str , ## __VA_ARGS__)
#else
#define LOGD(...) do{} while(0)
#define LOGI(...) do{} while(0)
#define LOGW(...) do{} while(0)
#define LOGE(...) do{} while(0)
#endif


using namespace cv;

static inline cv::Point2f centerRect(const cv::Rect& r)
{
    return cv::Point2f(r.x+((float)r.width)/2, r.y+((float)r.height)/2);
}

static inline cv::Rect scale_rect(const cv::Rect& r, float scale)
{
    cv::Point2f m=centerRect(r);
    float width  = r.width  * scale;
    float height = r.height * scale;
    int x=cvRound(m.x - width/2);
    int y=cvRound(m.y - height/2);

    return cv::Rect(x, y, cvRound(width), cvRound(height));
}

namespace cv
{
    void* workcycleObjectDetectorFunction(void* p);
}

class cv::DetectionBasedTracker::SeparateDetectionWork
{
    public:
        SeparateDetectionWork(cv::DetectionBasedTracker& _detectionBasedTracker, cv::Ptr<DetectionBasedTracker::IDetector> _detector);
        virtual ~SeparateDetectionWork();
        bool communicateWithDetectingThread(const Mat& imageGray, std::vector<Rect>& rectsWhereRegions);
        bool run();
        void stop();
        void resetTracking();

        inline bool isWorking()
        {
            return (stateThread==STATE_THREAD_WORKING_SLEEPING) || (stateThread==STATE_THREAD_WORKING_WITH_IMAGE);
        }
        inline void lock()
        {
            pthread_mutex_lock(&mutex);
        }
        inline void unlock()
        {
            pthread_mutex_unlock(&mutex);
        }

    protected:

        DetectionBasedTracker& detectionBasedTracker;
        cv::Ptr<DetectionBasedTracker::IDetector> cascadeInThread;

        pthread_t second_workthread;
        pthread_mutex_t mutex;
        pthread_cond_t objectDetectorRun;
        pthread_cond_t objectDetectorThreadStartStop;

        std::vector<cv::Rect> resultDetect;
        volatile bool isObjectDetectingReady;
        volatile bool shouldObjectDetectingResultsBeForgot;

        enum StateSeparatedThread {
            STATE_THREAD_STOPPED=0,
            STATE_THREAD_WORKING_SLEEPING,
            STATE_THREAD_WORKING_WITH_IMAGE,
            STATE_THREAD_WORKING,
            STATE_THREAD_STOPPING
        };
        volatile StateSeparatedThread stateThread;

        cv::Mat imageSeparateDetecting;

        void workcycleObjectDetector();
        friend void* workcycleObjectDetectorFunction(void* p);

        long long  timeWhenDetectingThreadStartedWork;
};

cv::DetectionBasedTracker::SeparateDetectionWork::SeparateDetectionWork(DetectionBasedTracker& _detectionBasedTracker, cv::Ptr<DetectionBasedTracker::IDetector> _detector)
    :detectionBasedTracker(_detectionBasedTracker),
    cascadeInThread(),
    isObjectDetectingReady(false),
    shouldObjectDetectingResultsBeForgot(false),
    stateThread(STATE_THREAD_STOPPED),
    timeWhenDetectingThreadStartedWork(-1)
{
    CV_Assert(_detector);

    cascadeInThread = _detector;

    int res=0;
    res=pthread_mutex_init(&mutex, NULL);//TODO: should be attributes?
    if (res) {
        LOGE("ERROR in DetectionBasedTracker::SeparateDetectionWork::SeparateDetectionWork in pthread_mutex_init(&mutex, NULL) is %d", res);
        throw(std::exception());
    }
    res=pthread_cond_init (&objectDetectorRun, NULL);
    if (res) {
        LOGE("ERROR in DetectionBasedTracker::SeparateDetectionWork::SeparateDetectionWork in pthread_cond_init(&objectDetectorRun,, NULL) is %d", res);
        pthread_mutex_destroy(&mutex);
        throw(std::exception());
    }
    res=pthread_cond_init (&objectDetectorThreadStartStop, NULL);
    if (res) {
        LOGE("ERROR in DetectionBasedTracker::SeparateDetectionWork::SeparateDetectionWork in pthread_cond_init(&objectDetectorThreadStartStop,, NULL) is %d", res);
        pthread_cond_destroy(&objectDetectorRun);
        pthread_mutex_destroy(&mutex);
        throw(std::exception());
    }
}

cv::DetectionBasedTracker::SeparateDetectionWork::~SeparateDetectionWork()
{
    if(stateThread!=STATE_THREAD_STOPPED) {
        LOGE("\n\n\nATTENTION!!! dangerous algorithm error: destructor DetectionBasedTracker::DetectionBasedTracker::~SeparateDetectionWork is called before stopping the workthread");
    }

    pthread_cond_destroy(&objectDetectorThreadStartStop);
    pthread_cond_destroy(&objectDetectorRun);
    pthread_mutex_destroy(&mutex);
}
bool cv::DetectionBasedTracker::SeparateDetectionWork::run()
{
    LOGD("DetectionBasedTracker::SeparateDetectionWork::run() --- start");
    pthread_mutex_lock(&mutex);
    if (stateThread != STATE_THREAD_STOPPED) {
        LOGE("DetectionBasedTracker::SeparateDetectionWork::run is called while the previous run is not stopped");
        pthread_mutex_unlock(&mutex);
        return false;
    }
    stateThread=STATE_THREAD_WORKING_SLEEPING;
    pthread_create(&second_workthread, NULL, workcycleObjectDetectorFunction, (void*)this); //TODO: add attributes?
    pthread_cond_wait(&objectDetectorThreadStartStop, &mutex);
    pthread_mutex_unlock(&mutex);
    LOGD("DetectionBasedTracker::SeparateDetectionWork::run --- end");
    return true;
}

#ifdef __GNUC__
#define CATCH_ALL_AND_LOG(_block)                                                       \
do {                                                                               \
    try {                                                                                   \
        _block;                                                                             \
        break;                                                                              \
    }                                                                                       \
    catch(cv::Exception& e) {                                                               \
        LOGE0("\n %s: ERROR: OpenCV Exception caught: \n'%s'\n\n", __func__, e.what());      \
    } catch(std::exception& e) {                                                            \
        LOGE0("\n %s: ERROR: Exception caught: \n'%s'\n\n", __func__, e.what());             \
    } catch(...) {                                                                          \
        LOGE0("\n %s: ERROR: UNKNOWN Exception caught\n\n", __func__);                       \
    }                                                                                       \
} while(0)
#else
#define CATCH_ALL_AND_LOG(_block)                                                       \
do {                                                                               \
    try {                                                                                   \
        _block;                                                                             \
        break;                                                                              \
    }                                                                                       \
    catch(cv::Exception& e) {                                                               \
        LOGE0("\n ERROR: OpenCV Exception caught: \n'%s'\n\n", e.what());                    \
    } catch(std::exception& e) {                                                            \
        LOGE0("\n ERROR: Exception caught: \n'%s'\n\n", e.what());                           \
    } catch(...) {                                                                          \
        LOGE0("\n ERROR: UNKNOWN Exception caught\n\n");                                     \
    }                                                                                       \
} while(0)
#endif

void* cv::workcycleObjectDetectorFunction(void* p)
{
    CATCH_ALL_AND_LOG({ ((cv::DetectionBasedTracker::SeparateDetectionWork*)p)->workcycleObjectDetector(); });
    try{
        ((cv::DetectionBasedTracker::SeparateDetectionWork*)p)->stateThread = cv::DetectionBasedTracker::SeparateDetectionWork::STATE_THREAD_STOPPED;
    } catch(...) {
        LOGE0("DetectionBasedTracker: workcycleObjectDetectorFunction: ERROR concerning pointer, received as the function parameter");
    }
    return NULL;
}

void cv::DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector()
{
    static double freq = getTickFrequency();
    LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- start");
    std::vector<Rect> objects;

    CV_Assert(stateThread==STATE_THREAD_WORKING_SLEEPING);
    pthread_mutex_lock(&mutex);
    {
        pthread_cond_signal(&objectDetectorThreadStartStop);

        LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- before waiting");
        CV_Assert(stateThread==STATE_THREAD_WORKING_SLEEPING);
        pthread_cond_wait(&objectDetectorRun, &mutex);
        if (isWorking()) {
            stateThread=STATE_THREAD_WORKING_WITH_IMAGE;
        }
        LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- after waiting");
    }
    pthread_mutex_unlock(&mutex);

    bool isFirstStep=true;

    isObjectDetectingReady=false;

    while(isWorking())
    {
        LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- next step");

        if (! isFirstStep) {
            LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- before waiting");
            CV_Assert(stateThread==STATE_THREAD_WORKING_SLEEPING);

            pthread_mutex_lock(&mutex);
            if (!isWorking()) {//it is a rare case, but may cause a crash
                LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- go out from the workcycle from inner part of lock just before waiting");
                pthread_mutex_unlock(&mutex);
                break;
            }
            CV_Assert(stateThread==STATE_THREAD_WORKING_SLEEPING);
            pthread_cond_wait(&objectDetectorRun, &mutex);
            if (isWorking()) {
                stateThread=STATE_THREAD_WORKING_WITH_IMAGE;
            }
            pthread_mutex_unlock(&mutex);

            LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- after waiting");
        } else {
            isFirstStep=false;
        }

        if (!isWorking()) {
            LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- go out from the workcycle just after waiting");
            break;
        }


        if (imageSeparateDetecting.empty()) {
            LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- imageSeparateDetecting is empty, continue");
            continue;
        }
        LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- start handling imageSeparateDetecting, img.size=%dx%d, img.data=0x%p",
                imageSeparateDetecting.size().width, imageSeparateDetecting.size().height, (void*)imageSeparateDetecting.data);


        int64 t1_detect=getTickCount();

        cascadeInThread->detect(imageSeparateDetecting, objects);

        /*cascadeInThread.detectMultiScale( imageSeparateDetecting, objects,
                detectionBasedTracker.parameters.scaleFactor, detectionBasedTracker.parameters.minNeighbors, 0
                |CV_HAAR_SCALE_IMAGE
                ,
                min_objectSize,
                max_objectSize
                );
        */

        LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- end handling imageSeparateDetecting");

        if (!isWorking()) {
            LOGD("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- go out from the workcycle just after detecting");
            break;
        }

        int64 t2_detect = getTickCount();
        int64 dt_detect = t2_detect-t1_detect;
        double dt_detect_ms=((double)dt_detect)/freq * 1000.0;
        (void)(dt_detect_ms);

        LOGI("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector() --- objects num==%d, t_ms=%.4f", (int)objects.size(), dt_detect_ms);

        pthread_mutex_lock(&mutex);
        if (!shouldObjectDetectingResultsBeForgot) {
            resultDetect=objects;
            isObjectDetectingReady=true;
        } else { //shouldObjectDetectingResultsBeForgot==true
            resultDetect.clear();
            isObjectDetectingReady=false;
            shouldObjectDetectingResultsBeForgot=false;
        }
        if(isWorking()) {
            stateThread=STATE_THREAD_WORKING_SLEEPING;
        }
        pthread_mutex_unlock(&mutex);

        objects.clear();
    }// while(isWorking())


    pthread_mutex_lock(&mutex);

    stateThread=STATE_THREAD_STOPPED;

    isObjectDetectingReady=false;
    shouldObjectDetectingResultsBeForgot=false;

    pthread_cond_signal(&objectDetectorThreadStartStop);

    pthread_mutex_unlock(&mutex);

    LOGI("DetectionBasedTracker::SeparateDetectionWork::workcycleObjectDetector: Returning");
}

void cv::DetectionBasedTracker::SeparateDetectionWork::stop()
{
    //FIXME: TODO: should add quickStop functionality
    pthread_mutex_lock(&mutex);
    if (!isWorking()) {
        pthread_mutex_unlock(&mutex);
        LOGE("SimpleHighguiDemoCore::stop is called but the SimpleHighguiDemoCore pthread is not active");
        return;
    }
    stateThread=STATE_THREAD_STOPPING;
    LOGD("DetectionBasedTracker::SeparateDetectionWork::stop: before going to sleep to wait for the signal from the workthread");
    pthread_cond_signal(&objectDetectorRun);
    pthread_cond_wait(&objectDetectorThreadStartStop, &mutex);
    LOGD("DetectionBasedTracker::SeparateDetectionWork::stop: after receiving the signal from the workthread, stateThread=%d", (int)stateThread);
    pthread_mutex_unlock(&mutex);
}

void cv::DetectionBasedTracker::SeparateDetectionWork::resetTracking()
{
    LOGD("DetectionBasedTracker::SeparateDetectionWork::resetTracking");
    pthread_mutex_lock(&mutex);

    if (stateThread == STATE_THREAD_WORKING_WITH_IMAGE) {
        LOGD("DetectionBasedTracker::SeparateDetectionWork::resetTracking: since workthread is detecting objects at the moment, we should make cascadeInThread stop detecting and forget the detecting results");
        shouldObjectDetectingResultsBeForgot=true;
        //cascadeInThread.setStopFlag();//FIXME: TODO: this feature also should be contributed to OpenCV
    } else {
        LOGD("DetectionBasedTracker::SeparateDetectionWork::resetTracking: since workthread is NOT detecting objects at the moment, we should NOT make any additional actions");
    }

    resultDetect.clear();
    isObjectDetectingReady=false;


    pthread_mutex_unlock(&mutex);

}

bool cv::DetectionBasedTracker::SeparateDetectionWork::communicateWithDetectingThread(const Mat& imageGray, std::vector<Rect>& rectsWhereRegions)
{
    static double freq = getTickFrequency();

    bool shouldCommunicateWithDetectingThread = (stateThread==STATE_THREAD_WORKING_SLEEPING);
    LOGD("DetectionBasedTracker::SeparateDetectionWork::communicateWithDetectingThread: shouldCommunicateWithDetectingThread=%d", (shouldCommunicateWithDetectingThread?1:0));

    if (!shouldCommunicateWithDetectingThread) {
        return false;
    }

    bool shouldHandleResult = false;
    pthread_mutex_lock(&mutex);

    if (isObjectDetectingReady) {
        shouldHandleResult=true;
        rectsWhereRegions = resultDetect;
        isObjectDetectingReady=false;

        double lastBigDetectionDuration = 1000.0 * (((double)(getTickCount()  - timeWhenDetectingThreadStartedWork )) / freq);
        (void)(lastBigDetectionDuration);
        LOGD("DetectionBasedTracker::SeparateDetectionWork::communicateWithDetectingThread: lastBigDetectionDuration=%f ms", (double)lastBigDetectionDuration);
    }

    bool shouldSendNewDataToWorkThread = true;
    if (timeWhenDetectingThreadStartedWork > 0) {
        double time_from_previous_launch_in_ms=1000.0 * (((double)(getTickCount()  - timeWhenDetectingThreadStartedWork )) / freq); //the same formula as for lastBigDetectionDuration
        shouldSendNewDataToWorkThread = (time_from_previous_launch_in_ms >= detectionBasedTracker.parameters.minDetectionPeriod);
        LOGD("DetectionBasedTracker::SeparateDetectionWork::communicateWithDetectingThread: shouldSendNewDataToWorkThread was 1, now it is %d, since time_from_previous_launch_in_ms=%.2f, minDetectionPeriod=%d",
                (shouldSendNewDataToWorkThread?1:0), time_from_previous_launch_in_ms, detectionBasedTracker.parameters.minDetectionPeriod);
    }

    if (shouldSendNewDataToWorkThread) {

        imageSeparateDetecting.create(imageGray.size(), CV_8UC1);

        imageGray.copyTo(imageSeparateDetecting);//may change imageSeparateDetecting ptr. But should not.


        timeWhenDetectingThreadStartedWork = getTickCount() ;

        pthread_cond_signal(&objectDetectorRun);
    }

    pthread_mutex_unlock(&mutex);
    LOGD("DetectionBasedTracker::SeparateDetectionWork::communicateWithDetectingThread: result: shouldHandleResult=%d", (shouldHandleResult?1:0));

    return shouldHandleResult;
}

cv::DetectionBasedTracker::Parameters::Parameters()
{
    maxTrackLifetime=5;
    minDetectionPeriod=0;
}

cv::DetectionBasedTracker::InnerParameters::InnerParameters()
{
    numLastPositionsToTrack=4;
    numStepsToWaitBeforeFirstShow=6;
    numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown=3;
    numStepsToShowWithoutDetecting=3;

    coeffTrackingWindowSize=2.0;
    coeffObjectSizeToTrack=0.85;
    coeffObjectSpeedUsingInPrediction=0.8;

}

cv::DetectionBasedTracker::DetectionBasedTracker(cv::Ptr<IDetector> mainDetector, cv::Ptr<IDetector> trackingDetector, const Parameters& params)
    :separateDetectionWork(),
    parameters(params),
    innerParameters(),
    numTrackedSteps(0),
    cascadeForTracking(trackingDetector)
{
    CV_Assert( (params.maxTrackLifetime >= 0)
//            && mainDetector
            && trackingDetector );

    if (mainDetector) {
        separateDetectionWork.reset(new SeparateDetectionWork(*this, mainDetector));
    }

    weightsPositionsSmoothing.push_back(1);
    weightsSizesSmoothing.push_back(0.5);
    weightsSizesSmoothing.push_back(0.3);
    weightsSizesSmoothing.push_back(0.2);
}

cv::DetectionBasedTracker::~DetectionBasedTracker()
{
}

void DetectionBasedTracker::process(const Mat& imageGray)
{
    CV_Assert(imageGray.type()==CV_8UC1);

    if ( separateDetectionWork && !separateDetectionWork->isWorking() ) {
        separateDetectionWork->run();
    }

    static double freq = getTickFrequency();
    static long long time_when_last_call_started=getTickCount();

    {
        double delta_time_from_prev_call=1000.0 * (((double)(getTickCount()  - time_when_last_call_started)) / freq);
        (void)(delta_time_from_prev_call);
        LOGD("DetectionBasedTracker::process: time from the previous call is %f ms", (double)delta_time_from_prev_call);
        time_when_last_call_started=getTickCount();
    }

    Mat imageDetect=imageGray;

    std::vector<Rect> rectsWhereRegions;
    bool shouldHandleResult=false;
    if (separateDetectionWork) {
        shouldHandleResult = separateDetectionWork->communicateWithDetectingThread(imageGray, rectsWhereRegions);
    }

    if (shouldHandleResult) {
        LOGD("DetectionBasedTracker::process: get _rectsWhereRegions were got from resultDetect");
    } else {
        LOGD("DetectionBasedTracker::process: get _rectsWhereRegions from previous positions");
        for(size_t i = 0; i < trackedObjects.size(); i++) {
            int n = trackedObjects[i].lastPositions.size();
            CV_Assert(n > 0);

            Rect r = trackedObjects[i].lastPositions[n-1];
            if(r.area() == 0) {
                LOGE("DetectionBasedTracker::process: ERROR: ATTENTION: strange algorithm's behavior: trackedObjects[i].rect() is empty");
                continue;
            }

            //correction by speed of rectangle
            if (n > 1) {
                Point2f center = centerRect(r);
                Point2f center_prev = centerRect(trackedObjects[i].lastPositions[n-2]);
                Point2f shift = (center - center_prev) * innerParameters.coeffObjectSpeedUsingInPrediction;

                r.x += cvRound(shift.x);
                r.y += cvRound(shift.y);
            }


            rectsWhereRegions.push_back(r);
        }
    }
    LOGI("DetectionBasedTracker::process: tracked objects num==%d", (int)trackedObjects.size());

    std::vector<Rect> detectedObjectsInRegions;

    LOGD("DetectionBasedTracker::process: rectsWhereRegions.size()=%d", (int)rectsWhereRegions.size());
    for(size_t i=0; i < rectsWhereRegions.size(); i++) {
        Rect r = rectsWhereRegions[i];

        detectInRegion(imageDetect, r, detectedObjectsInRegions);
    }
    LOGD("DetectionBasedTracker::process: detectedObjectsInRegions.size()=%d", (int)detectedObjectsInRegions.size());

    updateTrackedObjects(detectedObjectsInRegions);
}

void cv::DetectionBasedTracker::getObjects(std::vector<cv::Rect>& result) const
{
    result.clear();

    for(size_t i=0; i < trackedObjects.size(); i++) {
        Rect r=calcTrackedObjectPositionToShow(i);
        if (r.area()==0) {
            continue;
        }
        result.push_back(r);
        LOGD("DetectionBasedTracker::process: found a object with SIZE %d x %d, rect={%d, %d, %d x %d}", r.width, r.height, r.x, r.y, r.width, r.height);
    }
}

void cv::DetectionBasedTracker::getObjects(std::vector<Object>& result) const
{
    result.clear();

    for(size_t i=0; i < trackedObjects.size(); i++) {
        Rect r=calcTrackedObjectPositionToShow(i);
        if (r.area()==0) {
            continue;
        }
        result.push_back(Object(r, trackedObjects[i].id));
        LOGD("DetectionBasedTracker::process: found a object with SIZE %d x %d, rect={%d, %d, %d x %d}", r.width, r.height, r.x, r.y, r.width, r.height);
    }
}
void cv::DetectionBasedTracker::getObjects(std::vector<ExtObject>& result) const
{
    result.clear();

    for(size_t i=0; i < trackedObjects.size(); i++) {
        ObjectStatus status;
        Rect r=calcTrackedObjectPositionToShow(i, status);
        result.push_back(ExtObject(trackedObjects[i].id, r, status));
        LOGD("DetectionBasedTracker::process: found a object with SIZE %d x %d, rect={%d, %d, %d x %d}, status = %d", r.width, r.height, r.x, r.y, r.width, r.height, (int)status);
    }
}

bool cv::DetectionBasedTracker::run()
{
    if (separateDetectionWork) {
        return separateDetectionWork->run();
    }
    return false;
}

void cv::DetectionBasedTracker::stop()
{
    if (separateDetectionWork) {
        separateDetectionWork->stop();
    }
}

void cv::DetectionBasedTracker::resetTracking()
{
    if (separateDetectionWork) {
        separateDetectionWork->resetTracking();
    }
    trackedObjects.clear();
}

void cv::DetectionBasedTracker::updateTrackedObjects(const std::vector<Rect>& detectedObjects)
{
    enum {
        NEW_RECTANGLE=-1,
        INTERSECTED_RECTANGLE=-2
    };

    int N1=trackedObjects.size();
    int N2=detectedObjects.size();
    LOGD("DetectionBasedTracker::updateTrackedObjects: N1=%d, N2=%d", N1, N2);

    for(int i=0; i < N1; i++) {
        trackedObjects[i].numDetectedFrames++;
    }

    std::vector<int> correspondence(detectedObjects.size(), NEW_RECTANGLE);
    correspondence.clear();
    correspondence.resize(detectedObjects.size(), NEW_RECTANGLE);

    for(int i=0; i < N1; i++) {
        LOGD("DetectionBasedTracker::updateTrackedObjects: i=%d", i);
        TrackedObject& curObject=trackedObjects[i];

        int bestIndex=-1;
        int bestArea=-1;

        int numpositions=curObject.lastPositions.size();
        CV_Assert(numpositions > 0);
        Rect prevRect=curObject.lastPositions[numpositions-1];
        LOGD("DetectionBasedTracker::updateTrackedObjects: prevRect[%d]={%d, %d, %d x %d}", i, prevRect.x, prevRect.y, prevRect.width, prevRect.height);

        for(int j=0; j < N2; j++) {
            LOGD("DetectionBasedTracker::updateTrackedObjects: j=%d", j);
            if (correspondence[j] >= 0) {
                LOGD("DetectionBasedTracker::updateTrackedObjects: j=%d is rejected, because it has correspondence=%d", j, correspondence[j]);
                continue;
            }
            if (correspondence[j] !=NEW_RECTANGLE) {
                LOGD("DetectionBasedTracker::updateTrackedObjects: j=%d is rejected, because it is intersected with another rectangle", j);
                continue;
            }
            LOGD("DetectionBasedTracker::updateTrackedObjects: detectedObjects[%d]={%d, %d, %d x %d}",
                    j, detectedObjects[j].x, detectedObjects[j].y, detectedObjects[j].width, detectedObjects[j].height);

            Rect r=prevRect & detectedObjects[j];
            if ( (r.width > 0) && (r.height > 0) ) {
                LOGD("DetectionBasedTracker::updateTrackedObjects: There is intersection between prevRect and detectedRect, r={%d, %d, %d x %d}",
                        r.x, r.y, r.width, r.height);
                correspondence[j]=INTERSECTED_RECTANGLE;

                if ( r.area() > bestArea) {
                    LOGD("DetectionBasedTracker::updateTrackedObjects: The area of intersection is %d, it is better than bestArea=%d", r.area(), bestArea);
                    bestIndex=j;
                    bestArea=r.area();
                }
            }
        }
        if (bestIndex >= 0) {
            LOGD("DetectionBasedTracker::updateTrackedObjects: The best correspondence for i=%d is j=%d", i, bestIndex);
            correspondence[bestIndex]=i;

            for(int j=0; j < N2; j++) {
                if (correspondence[j] >= 0)
                    continue;

                Rect r=detectedObjects[j] & detectedObjects[bestIndex];
                if ( (r.width > 0) && (r.height > 0) ) {
                    LOGD("DetectionBasedTracker::updateTrackedObjects: Found intersection between "
                            "rectangles j=%d and bestIndex=%d, rectangle j=%d is marked as intersected", j, bestIndex, j);
                    correspondence[j]=INTERSECTED_RECTANGLE;
                }
            }
        } else {
            LOGD("DetectionBasedTracker::updateTrackedObjects: There is no correspondence for i=%d ", i);
            curObject.numFramesNotDetected++;
        }
    }

    LOGD("DetectionBasedTracker::updateTrackedObjects: start second cycle");
    for(int j=0; j < N2; j++) {
        LOGD("DetectionBasedTracker::updateTrackedObjects: j=%d", j);
        int i=correspondence[j];
        if (i >= 0) {//add position
            LOGD("DetectionBasedTracker::updateTrackedObjects: add position");
            trackedObjects[i].lastPositions.push_back(detectedObjects[j]);
            while ((int)trackedObjects[i].lastPositions.size() > (int) innerParameters.numLastPositionsToTrack) {
                trackedObjects[i].lastPositions.erase(trackedObjects[i].lastPositions.begin());
            }
            trackedObjects[i].numFramesNotDetected=0;
        } else if (i==NEW_RECTANGLE){ //new object
            LOGD("DetectionBasedTracker::updateTrackedObjects: new object");
            trackedObjects.push_back(detectedObjects[j]);
        } else {
            LOGD("DetectionBasedTracker::updateTrackedObjects: was auxiliary intersection");
        }
    }

    std::vector<TrackedObject>::iterator it=trackedObjects.begin();
    while( it != trackedObjects.end() ) {
        if ( (it->numFramesNotDetected > parameters.maxTrackLifetime)
                ||
                (
                 (it->numDetectedFrames <= innerParameters.numStepsToWaitBeforeFirstShow)
                 &&
                 (it->numFramesNotDetected > innerParameters.numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown)
                )
           )
        {
            int numpos=it->lastPositions.size();
            CV_Assert(numpos > 0);
            Rect r = it->lastPositions[numpos-1];
            (void)(r);
            LOGD("DetectionBasedTracker::updateTrackedObjects: deleted object {%d, %d, %d x %d}",
                    r.x, r.y, r.width, r.height);
            it=trackedObjects.erase(it);
        } else {
            it++;
        }
    }
}

int cv::DetectionBasedTracker::addObject(const Rect& location)
{
    LOGD("DetectionBasedTracker::addObject: new object {%d, %d %dx%d}",location.x, location.y, location.width, location.height);
    trackedObjects.push_back(TrackedObject(location));
    int newId = trackedObjects.back().id;
    LOGD("DetectionBasedTracker::addObject: newId = %d", newId);
    return newId;
}

Rect cv::DetectionBasedTracker::calcTrackedObjectPositionToShow(int i) const
{
    ObjectStatus status;
    return calcTrackedObjectPositionToShow(i, status);
}
Rect cv::DetectionBasedTracker::calcTrackedObjectPositionToShow(int i, ObjectStatus& status) const
{
    if ( (i < 0) || (i >= (int)trackedObjects.size()) ) {
        LOGE("DetectionBasedTracker::calcTrackedObjectPositionToShow: ERROR: wrong i=%d", i);
        status = WRONG_OBJECT;
        return Rect();
    }
    if (trackedObjects[i].numDetectedFrames <= innerParameters.numStepsToWaitBeforeFirstShow){
        LOGI("DetectionBasedTracker::calcTrackedObjectPositionToShow: trackedObjects[%d].numDetectedFrames=%d <= numStepsToWaitBeforeFirstShow=%d --- return empty Rect()",
                i, trackedObjects[i].numDetectedFrames, innerParameters.numStepsToWaitBeforeFirstShow);
        status = DETECTED_NOT_SHOWN_YET;
        return Rect();
    }
    if (trackedObjects[i].numFramesNotDetected > innerParameters.numStepsToShowWithoutDetecting) {
        status = DETECTED_TEMPORARY_LOST;
        return Rect();
    }

    const TrackedObject::PositionsVector& lastPositions=trackedObjects[i].lastPositions;

    int N=lastPositions.size();
    if (N<=0) {
        LOGE("DetectionBasedTracker::calcTrackedObjectPositionToShow: ERROR: no positions for i=%d", i);
        status = WRONG_OBJECT;
        return Rect();
    }

    int Nsize=std::min(N, (int)weightsSizesSmoothing.size());
    int Ncenter= std::min(N, (int)weightsPositionsSmoothing.size());

    Point2f center;
    double w=0, h=0;
    if (Nsize > 0) {
        double sum=0;
        for(int j=0; j < Nsize; j++) {
            int k=N-j-1;
            w += lastPositions[k].width  * weightsSizesSmoothing[j];
            h += lastPositions[k].height * weightsSizesSmoothing[j];
            sum+=weightsSizesSmoothing[j];
        }
        w /= sum;
        h /= sum;
    } else {
        w=lastPositions[N-1].width;
        h=lastPositions[N-1].height;
    }

    if (Ncenter > 0) {
        double sum=0;
        for(int j=0; j < Ncenter; j++) {
            int k=N-j-1;
            Point tl(lastPositions[k].tl());
            Point br(lastPositions[k].br());
            Point2f c1;
            c1=tl;
            c1=c1* 0.5f;
            Point2f c2;
            c2=br;
            c2=c2*0.5f;
            c1=c1+c2;

            center=center+  (c1  * weightsPositionsSmoothing[j]);
            sum+=weightsPositionsSmoothing[j];
        }
        center *= (float)(1 / sum);
    } else {
        int k=N-1;
        Point tl(lastPositions[k].tl());
        Point br(lastPositions[k].br());
        Point2f c1;
        c1=tl;
        c1=c1* 0.5f;
        Point2f c2;
        c2=br;
        c2=c2*0.5f;

        center=c1+c2;
    }
    Point2f tl=center-(Point2f(w,h)*0.5);
    Rect res(cvRound(tl.x), cvRound(tl.y), cvRound(w), cvRound(h));
    LOGD("DetectionBasedTracker::calcTrackedObjectPositionToShow: Result for i=%d: {%d, %d, %d x %d}", i, res.x, res.y, res.width, res.height);

    status = DETECTED;
    return res;
}

void cv::DetectionBasedTracker::detectInRegion(const Mat& img, const Rect& r, std::vector<Rect>& detectedObjectsInRegions)
{
    Rect r0(Point(), img.size());
    Rect r1 = scale_rect(r, innerParameters.coeffTrackingWindowSize);
    r1 = r1 & r0;

    if ( (r1.width <=0) || (r1.height <= 0) ) {
        LOGD("DetectionBasedTracker::detectInRegion: Empty intersection");
        return;
    }

    int d = cvRound(std::min(r.width, r.height) * innerParameters.coeffObjectSizeToTrack);

    std::vector<Rect> tmpobjects;

    Mat img1(img, r1);//subimage for rectangle -- without data copying
    LOGD("DetectionBasedTracker::detectInRegion: img1.size()=%d x %d, d=%d",
            img1.size().width, img1.size().height, d);

    cascadeForTracking->setMinObjectSize(Size(d, d));
    cascadeForTracking->detect(img1, tmpobjects);
            /*
            detectMultiScale( img1, tmpobjects,
            parameters.scaleFactor, parameters.minNeighbors, 0
            |CV_HAAR_FIND_BIGGEST_OBJECT
            |CV_HAAR_SCALE_IMAGE
            ,
            Size(d,d),
            max_objectSize
            );*/

    for(size_t i=0; i < tmpobjects.size(); i++) {
        Rect curres(tmpobjects[i].tl() + r1.tl(), tmpobjects[i].size());
        detectedObjectsInRegions.push_back(curres);
    }
}

bool cv::DetectionBasedTracker::setParameters(const Parameters& params)
{
    if ( params.maxTrackLifetime < 0 )
    {
        LOGE("DetectionBasedTracker::setParameters: ERROR: wrong parameters value");
        return false;
    }

    if (separateDetectionWork) {
        separateDetectionWork->lock();
    }
    parameters=params;
    if (separateDetectionWork) {
        separateDetectionWork->unlock();
    }
    return true;
}

const cv::DetectionBasedTracker::Parameters& DetectionBasedTracker::getParameters() const
{
    return parameters;
}

#endif
