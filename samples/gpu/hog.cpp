#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

//#define WRITE_VIDEO

class Args
{
public:
    Args();
    static Args read(int argc, char** argv);

    string src;
    bool src_is_video;
    bool src_is_camera;
    int camera_id;

    bool make_gray;

    bool resize_src;
    int resized_width, resized_height;

    double scale;
    int nlevels;
    int gr_threshold;
    double hit_threshold;

    int win_width;
    int win_stride_width, win_stride_height;

    bool gamma_corr;
};


class App
{
public:
    App(const Args& s);
    void run();

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string message() const;

private:
    App operator=(App&);

    Args args;
    bool running;

    bool use_gpu;
    bool make_gray;
    double scale;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

    int64 hog_work_begin;
    double hog_work_fps;

    int64 work_begin;
    double work_fps;
};


int main(int argc, char** argv)
{
    try
    {
        cout << "Histogram of Oriented Gradients descriptor and detector sample.\n";
        if (argc < 2)
        {
            cout << "\nUsage: hog_gpu\n"
                << "  --src <path> # it's image file by default\n"
                << "  [--src-is-video <true/false>] # says to interpretate src as video\n"
                << "  [--src-is-camera <true/false>] # says to interpretate src as camera\n"
                << "  [--make-gray <true/false>] # convert image to gray one or not\n"
                << "  [--resize-src <true/false>] # do resize of the source image or not\n"
                << "  [--src-width <int>] # resized image width\n"
                << "  [--src-height <int>] # resized image height\n"
                << "  [--hit-threshold <double>] # classifying plane distance threshold (0.0 usually)\n"
                << "  [--scale <double>] # HOG window scale factor\n"
                << "  [--nlevels <int>] # max number of HOG window scales\n"
                << "  [--win-width <int>] # width of the window (48 or 64)\n"
                << "  [--win-stride-width <int>] # distance by OX axis between neighbour wins\n"
                << "  [--win-stride-height <int>] # distance by OY axis between neighbour wins\n"
                << "  [--gr-threshold <int>] # merging similar rects constant\n"
                << "  [--gamma-correct <int>] # do gamma correction or not\n";
            return 1;
        }
        App app(Args::read(argc, argv));
        app.run();
    }
    catch (const Exception& e) { return cout << "Error: "  << e.what() << endl, 1; }
    catch (const exception& e) { return cout << "Error: "  << e.what() << endl, 1; }
    catch(...) { return cout << "Unknown exception" << endl, 1; }
    return 0;
}


Args::Args()
{
    src_is_video = false;
    src_is_camera = false;
    camera_id = 0;

    make_gray = false;

    resize_src = false;
    resized_width = 640;
    resized_height = 480;

    scale = 1.05;
    nlevels = 13;
    gr_threshold = 8;
    hit_threshold = 1.4;

    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;

    gamma_corr = true;
}


Args Args::read(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc - 1; i += 2)
    {
        string key = argv[i];
        string val = argv[i + 1];
        if (key == "--src") args.src = val;
        else if (key == "--src-is-video") args.src_is_video = (val == "true");        
        else if (key == "--src-is-camera") args.src_is_camera = (val == "true");        
        else if (key == "--camera-id") args.camera_id = atoi(val.c_str());
        else if (key == "--make-gray") args.make_gray = (val == "true");
        else if (key == "--resize-src") args.resize_src = (val == "true");
        else if (key == "--src-width") args.resized_width = atoi(val.c_str());
        else if (key == "--src-height") args.resized_height = atoi(val.c_str());
        else if (key == "--hit-threshold") args.hit_threshold = atof(val.c_str());
        else if (key == "--scale") args.scale = atof(val.c_str());
        else if (key == "--nlevels") args.nlevels = atoi(val.c_str());
        else if (key == "--win-width") args.win_width = atoi(val.c_str());
        else if (key == "--win-stride-width") args.win_stride_width = atoi(val.c_str());
        else if (key == "--win-stride-height") args.win_stride_height = atoi(val.c_str());
        else if (key == "--gr-threshold") args.gr_threshold = atoi(val.c_str());
        else if (key == "--gamma-correct") args.gamma_corr = atoi(val.c_str()) != 0;
        else throw runtime_error((string("unknown key: ") + key));
    }
    return args;
}


App::App(const Args& s)
{
    args = s;
    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tm - change mode GPU <-> CPU\n"
         << "\tg - convert image to gray or not\n"
         << "\t1/q - increase/decrease HOG scale\n"
         << "\t2/w - increase/decrease levels count\n"
         << "\t3/e - increase/decrease HOG group threshold\n"
         << "\t4/r - increase/decrease hit threshold\n"
         << endl;

    use_gpu = true;
    make_gray = args.make_gray;
    scale = args.scale;
    gr_threshold = args.gr_threshold;
    nlevels = args.nlevels;
    hit_threshold = args.hit_threshold;
    gamma_corr = args.gamma_corr;

    if (args.win_width != 64 && args.win_width != 48)
        args.win_width = 64;

    cout << "Scale: " << scale << endl;
    if (args.resize_src)
        cout << "Source size: (" << args.resized_width << ", " << args.resized_height << ")\n";
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: " << args.win_width << endl;
    cout << "Win stride: (" << args.win_stride_width << ", " << args.win_stride_height << ")\n";
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}


void App::run()
{
    running = true;

    Size win_size(args.win_width, args.win_width * 2); //(64, 128) or (48, 96)
    Size win_stride(args.win_stride_width, args.win_stride_height);

    vector<float> detector;
    if (win_size == Size(64, 128)) 
        detector = cv::gpu::HOGDescriptor::getPeopleDetector_64x128();
    else
        detector = cv::gpu::HOGDescriptor::getPeopleDetector_48x96();

    cv::gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 
                                   cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr, 
                                   cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
    cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1, 
                              HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
    gpu_hog.setSVMDetector(detector);
    cpu_hog.setSVMDetector(detector);

#ifdef WRITE_VIDEO
    cv::VideoWriter video_writer;
    video_writer.open("output.avi", CV_FOURCC('x','v','i','d'), 24., cv::Size(640, 480), true);
    if (!video_writer.isOpened())
        throw std::runtime_error("can't create video writer");
#endif

    while (running)
    {
        VideoCapture vc;
        Mat frame;

        if (args.src_is_video)
        {
            vc.open(args.src.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + args.src));
            vc >> frame;
        }
        else if (args.src_is_camera)
        {
            vc.open(args.camera_id);
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + args.src));
            vc >> frame;
        }
        else
        {
            frame = imread(args.src);
            if (frame.empty())
                throw runtime_error(string("can't open image file: " + args.src));
        }

        Mat img_aux, img, img_to_show;
        gpu::GpuMat gpu_img;

        // Iterate over all frames
        while (running && !frame.empty())
        {
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, CV_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, CV_BGR2BGRA);
            else img_aux = frame;

            // Resize image
            if (args.resize_src) resize(img_aux, img, Size(args.resized_width, args.resized_height));
            else img = img_aux;
            img_to_show = img;

            gpu_hog.nlevels = nlevels;
            cpu_hog.nlevels = nlevels;

            vector<Rect> found;

            // Perform HOG classification
            hogWorkBegin();
            if (use_gpu)
            {
                gpu_img = img;
                gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride, 
                                         Size(0, 0), scale, gr_threshold);
            }
            else cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride, 
                                          Size(0, 0), scale, gr_threshold);
            hogWorkEnd();

            // Draw positive classified windows
            for (size_t i = 0; i < found.size(); i++)
            {
                Rect r = found[i];
                rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
            }

            putText(img_to_show, "FPS (HOG only): " + hogWorkFps(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (total): " + workFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            imshow("opencv_gpu_hog", img_to_show);
            handleKey((char)waitKey(3));

            if (args.src_is_video || args.src_is_camera) vc >> frame;

            workEnd();

#ifdef WRITE_VIDEO
            cvtColor(img_to_show, img, CV_BGRA2BGR);
            video_writer << img;
#endif
        }
    }
}


void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'm':
    case 'M':
        use_gpu = !use_gpu;
        cout << "Switched to " << (use_gpu ? "CUDA" : "CPU") << " mode\n";
        break;
    case 'g':
    case 'G':
        make_gray = !make_gray;
        cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
        break;
    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case 'q':
    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;
    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;
    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case '4':
        hit_threshold+=0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;
    }
}


inline void App::hogWorkBegin() { hog_work_begin = getTickCount(); }

inline void App::hogWorkEnd()
{
    int64 delta = getTickCount() - hog_work_begin;
    double freq = getTickFrequency();
    hog_work_fps = freq / delta;
}

inline string App::hogWorkFps() const
{
    stringstream ss;
    ss << hog_work_fps;
    return ss.str();
}


inline void App::workBegin() { work_begin = getTickCount(); }

inline void App::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string App::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}
