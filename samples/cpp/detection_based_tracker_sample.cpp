#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)

#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/contrib/detection_based_tracker.hpp"

#include <vector>
#include <iostream>
#include <stdio.h>

#define DEBUGLOGS 1


#ifdef ANDROID
#include <android/log.h>
#define LOG_TAG "DETECTIONBASEDTRACKER__TEST_APPLICAT"
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
using namespace std;

#define ORIGINAL 0
#define SHOULD_USE_EXTERNAL_BUFFERS 1

static void usage()
{
    LOGE0("usage: filepattern outfilepattern cascadefile");
    LOGE0("\t where ");
    LOGE0("\t filepattern --- pattern for the paths to the source images");
    LOGE0("\t       (e.g.\"./Videos/FACESJPG2/Faces2_%%08d.jpg\" ");
    LOGE0("\t outfilepattern --- pattern for the paths for images which will be generated");
    LOGE0("\t       (e.g.\"./resFaces2_%%08d.jpg\" ");
    LOGE0("\t cascadefile --- path to the cascade file");
    LOGE0("\t       (e.g.\"opencv/data/lbpcascades/lbpcascade_frontalface.xml\" ");
}

static int test_FaceDetector(int argc, char *argv[])
{
    if (argc < 4) {
        usage();
        return -1;
    }

    const char* filepattern=argv[1];
    const char* outfilepattern=argv[2];
    const char* cascadefile=argv[3];
    LOGD0("filepattern='%s'", filepattern);
    LOGD0("outfilepattern='%s'", outfilepattern);
    LOGD0("cascadefile='%s'", cascadefile);

    vector<Mat> images;
    {
        char filename[256];
        for(int n=1; ; n++) {
            snprintf(filename, sizeof(filename), filepattern, n);
            LOGD("filename='%s'", filename);
            Mat m0;
            m0=imread(filename);
            if (m0.empty()) {
                LOGI0("Cannot read the file --- break");
                break;
            }
            images.push_back(m0);
        }
        LOGD("read %d images", (int)images.size());
    }

    DetectionBasedTracker::Parameters params;
    std::string cascadeFrontalfilename=cascadefile;

    DetectionBasedTracker fd(cascadeFrontalfilename, params);

    fd.run();

    Mat gray;
    Mat m;

    int64 tprev=getTickCount();
    double freq=getTickFrequency();

    int num_images=images.size();
    for(int n=1; n <= num_images; n++) {
        int64 tcur=getTickCount();
        int64 dt=tcur-tprev;
        tprev=tcur;
        double t_ms=((double)dt)/freq * 1000.0;
        LOGD("\n\nSTEP n=%d        from prev step %f ms\n\n", n, t_ms);
        m=images[n-1];
        CV_Assert(! m.empty());
        cvtColor(m, gray, CV_BGR2GRAY);

        fd.process(gray);

        vector<Rect> result;
        fd.getObjects(result);





        for(size_t i=0; i < result.size(); i++) {
            Rect r=result[i];
            CV_Assert(r.area() > 0);
            Point tl=r.tl();
            Point br=r.br();
            Scalar color=Scalar(0, 250, 0);
            rectangle(m, tl, br, color, 3);
        }
    }
    {
        char outfilename[256];
        for(int n=1; n <= num_images; n++) {
            snprintf(outfilename, sizeof(outfilename), outfilepattern, n);
            LOGD("outfilename='%s'", outfilename);
            m=images[n-1];
            imwrite(outfilename, m);
        }
    }

    fd.stop();

    return 0;
}



int main(int argc, char *argv[])
{
    return test_FaceDetector(argc, argv);
}

#else // #if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)

#include <stdio.h>
int main()
{
    printf("This sample works for UNIX or ANDROID only\n");
    return 0;
}

#endif
