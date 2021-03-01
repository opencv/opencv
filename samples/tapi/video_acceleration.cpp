#include <iostream>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

const char* keys =
"{ i input    |        | input video file }"
"{ o output   |        | output video file, or specify 'null' to measure decoding without rendering to screen}"
"{ backend    | any    | VideoCapture and VideoWriter backend, valid values: 'any', 'ffmpeg', 'msmf', 'gstreamer' }"
"{ accel      | any    | GPU Video Acceleration, valid values: 'none', 'any', 'd3d11', 'vaapi', 'mfx' }"
"{ device     | -1     | Video Acceleration device (GPU) index (-1 means default device) }"
"{ out_w      |        | output width (resize by calling cv::resize) }"
"{ out_h      |        | output height (resize by calling cv::resize) }"
"{ bitwise_not| false  | apply simple image processing - bitwise_not pixels by calling cv::bitwise_not }"
"{ opencl     | true   | use OpenCL (inside VideoCapture/VideoWriter and for image processing) }"
"{ codec      | H264   | codec id (four characters string) of output file encoder }"
"{ h help     |        | print help message }";

struct {
    cv::VideoCaptureAPIs backend;
    const char* str;
} backend_strings[] = {
    { cv::CAP_ANY, "any" },
    { cv::CAP_FFMPEG, "ffmpeg" },
    { cv::CAP_MSMF, "msmf" },
    { cv::CAP_GSTREAMER, "gstreamer" },
};

struct {
    VideoAccelerationType acceleration;
    const char* str;
} acceleration_strings[] = {
    { VIDEO_ACCELERATION_NONE, "none" },
    { VIDEO_ACCELERATION_ANY, "any" },
    { VIDEO_ACCELERATION_D3D11, "d3d11" },
    { VIDEO_ACCELERATION_VAAPI, "vaapi" },
    { VIDEO_ACCELERATION_MFX, "mfx" },
};

class FPSCounter {
public:
    FPSCounter(double _interval) : interval(_interval) {
    }

    ~FPSCounter() {
        NewFrame(true);
    }

    void NewFrame(bool last_frame = false) {
        num_frames++;
        auto now = std::chrono::high_resolution_clock::now();
        if (!last_time.time_since_epoch().count()) {
            last_time = now;
        }

        double sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_time).count();
        if (sec >= interval || last_frame) {
            printf("FPS(last %.2f sec) = %.2f\n", sec, num_frames / sec);
            fflush(stdout);
            num_frames = 0;
            last_time = now;
        }
    }

private:
    double interval = 1;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time;
    int num_frames = 0;
};

int main(int argc, char** argv)
{
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cout << "Usage : video_acceleration [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    string infile = cmd.get<string>("i");
    string outfile = cmd.get<string>("o");
    string codec = cmd.get<string>("codec");
    int device = cmd.get<int>("device");
    int out_w = cmd.get<int>("out_w");
    int out_h = cmd.get<int>("out_h");
    bool use_opencl = cmd.get<bool>("opencl");
    bool bitwise_not = cmd.get<bool>("bitwise_not");

    cv::VideoCaptureAPIs backend = cv::CAP_ANY;
    string backend_str = cmd.get<string>("backend");
    for (size_t i = 0; i < sizeof(backend_strings)/sizeof(backend_strings[0]); i++) {
        if (backend_str == backend_strings[i].str) {
            backend = backend_strings[i].backend;
            break;
        }
    }

    VideoAccelerationType accel = VIDEO_ACCELERATION_ANY;
    string accel_str = cmd.get<string>("accel");
    for (size_t i = 0; i < sizeof(acceleration_strings) / sizeof(acceleration_strings[0]); i++) {
        if (accel_str == acceleration_strings[i].str) {
            accel = acceleration_strings[i].acceleration;
            break;
        }
    }

    ocl::setUseOpenCL(use_opencl);

    VideoCapture capture(infile, backend, {
            CAP_PROP_HW_ACCELERATION, (int)accel,
            CAP_PROP_HW_DEVICE, device
    });
    if (!capture.isOpened()) {
        cerr << "Failed to open VideoCapture" << endl;
        return 1;
    }
    cout << "VideoCapture backend = " << capture.getBackendName() << endl;
    VideoAccelerationType actual_accel = static_cast<VideoAccelerationType>(static_cast<int>(capture.get(CAP_PROP_HW_ACCELERATION)));
    for (size_t i = 0; i < sizeof(acceleration_strings) / sizeof(acceleration_strings[0]); i++) {
        if (actual_accel == acceleration_strings[i].acceleration) {
            cout << "VideoCapture acceleration = " << acceleration_strings[i].str << endl;
            cout << "VideoCapture acceleration device = " << (int)capture.get(CAP_PROP_HW_DEVICE) << endl;
            break;
        }
    }

    VideoWriter writer;
    if (!outfile.empty() && outfile != "null") {
        const char* codec_str = codec.c_str();
        int fourcc = VideoWriter::fourcc(codec_str[0], codec_str[1], codec_str[2], codec_str[3]);
        double fps = capture.get(CAP_PROP_FPS);
        Size frameSize = { out_w, out_h };
        if (!out_w || !out_h) {
            frameSize = { (int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT) };
        }
        writer = VideoWriter(outfile, backend, fourcc, fps, frameSize, {
                VIDEOWRITER_PROP_HW_ACCELERATION, (int)accel,
                VIDEOWRITER_PROP_HW_DEVICE, device
        });
        if (!writer.isOpened()) {
            cerr << "Failed to open VideoWriter" << endl;
            return 1;
        }
        cout << "VideoWriter backend = " << writer.getBackendName() << endl;
        actual_accel = static_cast<VideoAccelerationType>(static_cast<int>(writer.get(CAP_PROP_HW_ACCELERATION)));
        for (size_t i = 0; i < sizeof(acceleration_strings) / sizeof(acceleration_strings[0]); i++) {
            if (actual_accel == acceleration_strings[i].acceleration) {
                cout << "VideoWriter acceleration = " << acceleration_strings[i].str << endl;
                cout << "VideoWriter acceleration device = " << (int)writer.get(CAP_PROP_HW_DEVICE) << endl;
                break;
            }
        }
    }

    cout << "\nStarting frame loop. Press ESC to exit\n";

    FPSCounter fps_counter(0.5); // print FPS every 0.5 seconds

    UMat frame, frame2, frame3;

    for (;;)
    {
        capture.read(frame);
        if (frame.empty()) {
            cout << "End of stream" << endl;
            break;
        }

        if (out_w && out_h) {
            cv::resize(frame, frame2, cv::Size(out_w, out_h));
            //cv::cvtColor(frame, outframe, COLOR_BGRA2RGBA);
        }
        else {
            frame2 = frame;
        }

        if (bitwise_not) {
            cv::bitwise_not(frame2, frame3);
        }
        else {
            frame3 = frame2;
        }

        if (writer.isOpened()) {
            writer.write(frame3);
        }

        if (outfile.empty()) {
            imshow("output", frame3);
            char key = (char) waitKey(1);
            if (key == 27)
                break;
            else if (key == 'm') {
                ocl::setUseOpenCL(!cv::ocl::useOpenCL());
                cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL enabled" : "CPU") << " mode\n";
            }
        }
        fps_counter.NewFrame();
    }

    return EXIT_SUCCESS;
}
