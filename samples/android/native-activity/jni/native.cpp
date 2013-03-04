#include <android_native_app_glue.h>

#include <errno.h>
#include <jni.h>
#include <sys/time.h>
#include <time.h>
#include <android/log.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define  LOG_TAG    "OCV:libnative_activity"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

struct Engine
{
    android_app* app;
    cv::Ptr<cv::VideoCapture> capture;
};

cv::Size calcOptimalResolution(const char* supported, int width, int height)
{
    int frameWidth = 0;
    int frameHeight = 0;

    size_t prev_idx = 0;
    size_t idx = 0;
    float minDiff = FLT_MAX;

    do
    {
        int tmp_width;
        int tmp_height;

        prev_idx = idx;
        while ((supported[idx] != '\0') && (supported[idx] != ','))
            idx++;

        sscanf(&supported[prev_idx], "%dx%d", &tmp_width, &tmp_height);

        int w_diff = width - tmp_width;
        int h_diff = height - tmp_height;
        if ((h_diff >= 0) && (w_diff >= 0))
        {
            if ((h_diff <= minDiff) && (tmp_height <= 720))
            {
                frameWidth = tmp_width;
                frameHeight = tmp_height;
                minDiff = h_diff;
            }
        }

        idx++; // to skip coma symbol

    } while(supported[idx-1] != '\0');

    return cv::Size(frameWidth, frameHeight);
}

static void engine_draw_frame(Engine* engine, const cv::Mat& frame)
{
    if (engine->app->window == NULL)
    {
        return; // No window.
    }

    ANativeWindow_Buffer buffer;
    if (ANativeWindow_lock(engine->app->window, &buffer, NULL) < 0)
    {
        LOGW("Unable to lock window buffer");
        return;
    }

    void* pixels = buffer.bits;

    for (int yy = 0; yy < std::min(frame.rows, buffer.height); yy++)
    {
        unsigned char* line = (unsigned char*)pixels;
        memcpy(line, frame.ptr<unsigned char>(yy),
               std::min(frame.cols, buffer.width)*4*sizeof(unsigned char));
        // go to next line
        pixels = (int32_t*)pixels + buffer.stride;
    }
    ANativeWindow_unlockAndPost(engine->app->window);
}

static void engine_handle_cmd(android_app* app, int32_t cmd)
{
    Engine* engine = (Engine*)app->userData;
    switch (cmd)
    {
        case APP_CMD_INIT_WINDOW:
            if (app->window != NULL)
            {
                LOGI("APP_CMD_INIT_WINDOW");

                engine->capture = new cv::VideoCapture(0);

                union {double prop; const char* name;} u;
                u.prop = engine->capture->get(CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING);

                cv::Size resolution;
                if (u.name)
                    resolution = calcOptimalResolution(u.name,
                                                       ANativeWindow_getWidth(app->window),
                                                       ANativeWindow_getHeight(app->window));
                else
                {
                    LOGE("Cannot get supported camera resolutions");
                    resolution = cv::Size(ANativeWindow_getWidth(app->window),
                                          ANativeWindow_getHeight(app->window));
                }

                if ((resolution.width != 0) && (resolution.height != 0))
                {
                    engine->capture->set(CV_CAP_PROP_FRAME_WIDTH, resolution.width);
                    engine->capture->set(CV_CAP_PROP_FRAME_HEIGHT, resolution.height);
                }

                if (ANativeWindow_setBuffersGeometry(app->window, resolution.width,
                    resolution.height, WINDOW_FORMAT_RGBA_8888) < 0)
                {
                    LOGE("Cannot set pixel format!");
                    return;
                }

                LOGI("Camera initialized at resoution %dx%d", resolution.width, resolution.height);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            LOGI("APP_CMD_TERM_WINDOW");

            engine->capture->release();
            break;
    }
}

void android_main(android_app* app)
{
    Engine engine;

    // Make sure glue isn't stripped.
    app_dummy();

    memset(&engine, 0, sizeof(engine));
    app->userData = &engine;
    app->onAppCmd = engine_handle_cmd;
    engine.app = app;

    float fps = 0;
    cv::Mat drawingFrame;
    bool firstFrame = true;
    std::queue<int64> timeQueue;

    // loop waiting for stuff to do.
    while (1)
    {
        // Read all pending events.
        int ident;
        int events;
        android_poll_source* source;

        // Process system events
        while ((ident=ALooper_pollAll(0, NULL, &events, (void**)&source)) >= 0)
        {
            // Process this event.
            if (source != NULL)
            {
                source->process(app, source);
            }

            // Check if we are exiting.
            if (app->destroyRequested != 0)
            {
                LOGI("Engine thread destroy requested!");
                return;
            }
        }

        int64 then;
        int64 now = cv::getTickCount();
        timeQueue.push(now);

        // Capture frame from camera and draw it
        if (!engine.capture.empty())
        {
            if (engine.capture->grab())
            {
                engine.capture->retrieve(drawingFrame, CV_CAP_ANDROID_COLOR_FRAME_RGBA);
//                 if (firstFrame)
//                 {
//                     firstFrame = false;
//                     engine.capture->set(CV_CAP_PROP_AUTOGRAB, 1);
//                 }
            }
             char buffer[256];
             sprintf(buffer, "Display performance: %dx%d @ %.3f", drawingFrame.cols, drawingFrame.rows, fps);
             cv::putText(drawingFrame, std::string(buffer), cv::Point(8,64), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,255,0,255));
             engine_draw_frame(&engine, drawingFrame);
        }

        if (timeQueue.size() >= 2)
            then = timeQueue.front();
        else
            then = 0;

        if (timeQueue.size() >= 25)
            timeQueue.pop();

        fps = 1.f*timeQueue.size()*(float)cv::getTickFrequency()/(float)(now-then);
    }
}
