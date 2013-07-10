#include <android_native_app_glue.h>

#include <errno.h>
#include <jni.h>
#include <sys/time.h>
#include <time.h>
#include <android/log.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

static cv::Size calc_optimal_camera_resolution(const char* supported, int width, int height)
{
    int frame_width = 0;
    int frame_height = 0;

    size_t prev_idx = 0;
    size_t idx = 0;
    float min_diff = FLT_MAX;

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
            if ((h_diff <= min_diff) && (tmp_height <= 720))
            {
                frame_width = tmp_width;
                frame_height = tmp_height;
                min_diff = h_diff;
            }
        }

        idx++; // to skip comma symbol

    } while(supported[idx-1] != '\0');

    return cv::Size(frame_width, frame_height);
}

static void engine_draw_frame(Engine* engine, const cv::Mat& frame)
{
    if (engine->app->window == NULL)
        return; // No window.

    ANativeWindow_Buffer buffer;
    if (ANativeWindow_lock(engine->app->window, &buffer, NULL) < 0)
    {
        LOGW("Unable to lock window buffer");
        return;
    }

    int32_t* pixels = (int32_t*)buffer.bits;

    int left_indent = (buffer.width-frame.cols)/2;
    int top_indent = (buffer.height-frame.rows)/2;

    if (top_indent > 0)
    {
        memset(pixels, 0, top_indent*buffer.stride*sizeof(int32_t));
        pixels += top_indent*buffer.stride;
    }

    for (int yy = 0; yy < frame.rows; yy++)
    {
        if (left_indent > 0)
        {
            memset(pixels, 0, left_indent*sizeof(int32_t));
            memset(pixels+left_indent+frame.cols, 0, (buffer.stride-frame.cols-left_indent)*sizeof(int32_t));
        }
        int32_t* line = pixels + left_indent;
        size_t line_size = frame.cols*4*sizeof(unsigned char);
        memcpy(line, frame.ptr<unsigned char>(yy), line_size);
        // go to next line
        pixels += buffer.stride;
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

                int view_width = ANativeWindow_getWidth(app->window);
                int view_height = ANativeWindow_getHeight(app->window);

                cv::Size camera_resolution;
                if (u.name)
                    camera_resolution = calc_optimal_camera_resolution(u.name, 640, 480);
                else
                {
                    LOGE("Cannot get supported camera camera_resolutions");
                    camera_resolution = cv::Size(ANativeWindow_getWidth(app->window),
                                          ANativeWindow_getHeight(app->window));
                }

                if ((camera_resolution.width != 0) && (camera_resolution.height != 0))
                {
                    engine->capture->set(CV_CAP_PROP_FRAME_WIDTH, camera_resolution.width);
                    engine->capture->set(CV_CAP_PROP_FRAME_HEIGHT, camera_resolution.height);
                }

                float scale = std::min((float)view_width/camera_resolution.width,
                                       (float)view_height/camera_resolution.height);

                if (ANativeWindow_setBuffersGeometry(app->window, (int)(view_width/scale),
                    int(view_height/scale), WINDOW_FORMAT_RGBA_8888) < 0)
                {
                    LOGE("Cannot set pixel format!");
                    return;
                }

                LOGI("Camera initialized at resolution %dx%d", camera_resolution.width, camera_resolution.height);
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

    size_t engine_size = sizeof(engine); // for Eclipse CDT parser
    memset((void*)&engine, 0, engine_size);
    app->userData = &engine;
    app->onAppCmd = engine_handle_cmd;
    engine.app = app;

    float fps = 0;
    cv::Mat drawing_frame;
    std::queue<int64> time_queue;

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
        time_queue.push(now);

        // Capture frame from camera and draw it
        if (!engine.capture.empty())
        {
            if (engine.capture->grab())
                engine.capture->retrieve(drawing_frame, CV_CAP_ANDROID_COLOR_FRAME_RGBA);

             char buffer[256];
             sprintf(buffer, "Display performance: %dx%d @ %.3f", drawing_frame.cols, drawing_frame.rows, fps);
             cv::putText(drawing_frame, std::string(buffer), cv::Point(8,64),
                         cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,255,0,255));
             engine_draw_frame(&engine, drawing_frame);
        }

        if (time_queue.size() >= 2)
            then = time_queue.front();
        else
            then = 0;

        if (time_queue.size() >= 25)
            time_queue.pop();

        fps = time_queue.size() * (float)cv::getTickFrequency() / (now-then);
    }
}
