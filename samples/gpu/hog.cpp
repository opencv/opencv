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


/** Contains all properties of application (including those which can be
changed by user in runtime) */
class Settings
{
public:
    /** Sets default values */
    Settings();

    /** Reads settings from command args */
    static Settings Read(int argc, char** argv);

    string src;
    bool src_is_video;
    bool make_gray;
    bool resize_src;
    double resize_src_scale;
    double scale;
    int nlevels;
    int gr_threshold;
    double hit_threshold;
    int win_width;
    int win_stride_width;
    int win_stride_height;
    bool gamma_corr;
};


/** Describes aplication logic */
class App
{
public:
    /** Initializes application */
    App(const Settings& s);

    /** Runs demo using OpenCV highgui module for GUI building */
    void RunOpencvGui();

    /** Processes user keybord input */
    void HandleKey(char key);

    void HogWorkBegin();
    void HogWorkEnd();
    double HogWorkFps() const;

    void WorkBegin();
    void WorkEnd();
    double WorkFps() const;

    const string GetPerformanceSummary() const;

private:
    App operator=(App&);

    Settings settings;
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
        if (argc < 2)
        {
            cout << "Usage:\nsample_hog\n"
                << "  -src <path_to_the_source>\n"
                << "  [-src_is_video <true/false>] # says to interp. src as img or as video\n"
                << "  [-make_gray <true/false>] # convert image to gray one or not\n"
                << "  [-resize_src <true/false>] # do resize of the source image or not\n"
                << "  [-resize_src_scale <double>] # preprocessing image scale factor\n"
                << "  [-hit_threshold <double>] # classifying plane dist. threshold (0.0 usually)\n"
                << "  [-scale <double>] # HOG window scale factor\n"
                << "  [-nlevels <int>] # max number of HOG window scales\n"
                << "  [-win_width <int>] # width of the window (48 or 64)\n"
                << "  [-win_stride_width <int>] # distance by OX axis between neighbour wins\n"
                << "  [-win_stride_height <int>] # distance by OY axis between neighbour wins\n"
                << "  [-gr_threshold <int>] # merging similar rects constant\n"
                << "  [-gamma_corr <int>] # do gamma correction or not\n";
            return 1;
        }
        App app(Settings::Read(argc, argv));
        app.RunOpencvGui();
    }
    catch (const Exception& e) { return cout << "Error: "  << e.what() << endl, 1; }
    catch (const exception& e) { return cout << "Error: "  << e.what() << endl, 1; }
    catch(...) { return cout << "Unknown exception" << endl, 1; }
    return 0;
}


Settings::Settings()
{
    src_is_video = false;
    make_gray = false;
    resize_src = true;
    resize_src_scale = 1.5;
    scale = 1.05;
    nlevels = 13;
    gr_threshold = 8;
    hit_threshold = 1.4;
    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;
    gamma_corr = true;
}


Settings Settings::Read(int argc, char** argv)
{
    cout << "Parsing command args" << endl;

    Settings settings;
    for (int i = 1; i < argc - 1; i += 2)
    {
        string key = argv[i];
        string val = argv[i + 1];
        if (key == "-src") settings.src = val;
        else if (key == "-src_is_video") settings.src_is_video = (val == "true");        
        else if (key == "-make_gray") settings.make_gray = (val == "true");
        else if (key == "-resize_src") settings.resize_src = (val == "true");
        else if (key == "-resize_src_scale") settings.resize_src_scale = atof(val.c_str());
        else if (key == "-hit_threshold") settings.hit_threshold = atof(val.c_str());
        else if (key == "-scale") settings.scale = atof(val.c_str());
        else if (key == "-nlevels") settings.nlevels = atoi(val.c_str());
        else if (key == "-win_width") settings.win_width = atoi(val.c_str());
        else if (key == "-win_stride_width") settings.win_stride_width = atoi(val.c_str());
        else if (key == "-win_stride_height") settings.win_stride_height = atoi(val.c_str());
        else if (key == "-gr_threshold") settings.gr_threshold = atoi(val.c_str());
        else if (key == "-gamma_corr") settings.gamma_corr = atoi(val.c_str()) != 0;
        else throw runtime_error((string("Unknown key: ") + key));
    }

    cout << "Command args are parsed\n";
    return settings;
}


App::App(const Settings &s)
{
    settings = s;
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
    make_gray = settings.make_gray;
    scale = settings.scale;
    gr_threshold = settings.gr_threshold;
    nlevels = settings.nlevels;
    hit_threshold = settings.hit_threshold;
    gamma_corr = settings.gamma_corr;

    if (settings.win_width != 64 && settings.win_width != 48)
        settings.win_width = 64;

    cout << "Scale: " << scale << endl;
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: " << settings.win_width << endl;
    cout << "Win stride: (" << settings.win_stride_width << ", " << settings.win_stride_height << ")\n";
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}


void App::RunOpencvGui()
{
    running = true;

    Size win_size(settings.win_width, settings.win_width * 2); //(64, 128) or (48, 96)
    Size win_stride(settings.win_stride_width, settings.win_stride_height);

    vector<float> detector;

    if (win_size == Size(64, 128))
        detector = cv::gpu::HOGDescriptor::getPeopleDetector_64x128();
    else
        detector = cv::gpu::HOGDescriptor::getPeopleDetector_48x96();

    // GPU's HOG classifier
    cv::gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 
                                   cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr, 
                                   cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
    gpu_hog.setSVMDetector(detector);

    // CPU's HOG classifier
    cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1, 
                              HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
    cpu_hog.setSVMDetector(detector);

#ifdef WRITE_VIDEO
    cv::VideoWriter video_writer;

    video_writer.open("output.avi", CV_FOURCC('x','v','i','d'), 24., cv::Size(640, 480), true);

    if (!video_writer.isOpened())
        throw std::runtime_error("can't create video writer");
#endif

    // Make endless cycle from video (if src is video)
    while (running)
    {
        VideoCapture vc;
        Mat frame;

        if (settings.src_is_video)
        {
            vc.open(settings.src.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("Can't open video file: " + settings.src));
            vc >> frame;
        }
        else
        {
            frame = imread(settings.src);
            if (frame.empty())
                throw runtime_error(string("Can't open image file: " + settings.src));
        }

        Mat img_aux, img, img_to_show;
        gpu::GpuMat gpu_img;

        // Iterate over all frames
        while (running && !frame.empty())
        {
            WorkBegin();

            vector<Rect> found;

            // Change format of the image (input must be 8UC3)
            if (make_gray)
                cvtColor(frame, img_aux, CV_BGR2GRAY);
            else if (use_gpu)
                cvtColor(frame, img_aux, CV_BGR2BGRA);
            else
                img_aux = frame;

            // Resize image
            if (settings.resize_src)
                resize(img_aux, img, Size(int(frame.cols * settings.resize_src_scale), int(frame.rows * settings.resize_src_scale)));
            else
                img = img_aux;
            img_to_show = img;

            gpu_hog.nlevels = nlevels;
            cpu_hog.nlevels = nlevels;

            // Perform HOG classification
            HogWorkBegin();
            if (use_gpu)
            {
                gpu_img = img;
                gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride, Size(0, 0), scale, gr_threshold);
            }
            else
                cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride, Size(0, 0), scale, gr_threshold);
            HogWorkEnd();

            // Draw positive classified windows
            for (size_t i = 0; i < found.size(); i++)
            {
                Rect r = found[i];
                rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
            }

            // Show results
            putText(img_to_show, GetPerformanceSummary(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            imshow("opencv_gpu_hog", img_to_show);
            HandleKey((char)waitKey(3));

            if (settings.src_is_video)
            {
                vc >> frame;
            }

            WorkEnd();

#ifdef WRITE_VIDEO
            cvtColor(img_to_show, img, CV_BGRA2BGR);
            video_writer << img;
#endif
        }
    }
}


void App::HandleKey(char key)
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


inline void App::HogWorkBegin() { hog_work_begin = getTickCount(); }


inline void App::HogWorkEnd()
{
    int64 delta = getTickCount() - hog_work_begin;
    double freq = getTickFrequency();
    hog_work_fps = freq / delta;
}


inline double App::HogWorkFps() const { return hog_work_fps; }


inline void App::WorkBegin() { work_begin = getTickCount(); }


inline void App::WorkEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}


inline double App::WorkFps() const { return work_fps; }


inline const string App::GetPerformanceSummary() const
{
    stringstream ss;
    ss << (use_gpu ? "GPU" : "CPU") << " HOG FPS: " << setiosflags(ios::left) << setprecision(4) <<
       setw(7) << HogWorkFps() << " Total FPS: " << setprecision(4) << setw(7) << WorkFps();
    return ss.str();
}
