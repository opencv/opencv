#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class App
{
public:
    App(CommandLineParser& cmd);
    void run();
    void handleKey(char key);
    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;
    void workBegin();
    void workEnd();
    string workFps() const;
    string message() const;


// This function test if gpu_rst matches cpu_rst.
// If the two vectors are not equal, it will return the difference in vector size
// Else if will return
// (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
    double checkRectSimilarity(Size sz,
                               std::vector<Rect>& cpu_rst,
                               std::vector<Rect>& gpu_rst);
private:
    App operator=(App&);

    //Args args;
    bool running;
    bool use_gpu;
    bool make_gray;
    double scale;
    double resize_scale;
    int win_width;
    int win_stride_width, win_stride_height;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

    int64 hog_work_begin;
    double hog_work_fps;
    int64 work_begin;
    double work_fps;

    string img_source;
    string vdo_source;
    string output;
    int camera_id;
    bool write_once;
};

int main(int argc, char** argv)
{
    const char* keys =
        "{ h |  help    | false          | print help message }"
        "{ i |  input   |                | specify input image}"
        "{ c | camera   | -1             | enable camera capturing }"
        "{ v | video    |                | use video as input }"
        "{ g |  gray    | false          | convert image to gray one or not}"
        "{ s |  scale   | 1.0            | resize the image before detect}"
        "{ l |larger_win| false          | use 64x128 window}"
        "{ o |  output  |                | specify output path when input is images}";
    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        cout << "Usage : hog [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        return EXIT_SUCCESS;
    }

    App app(cmd);
    try
    {
        app.run();
    }
    catch (const Exception& e)
    {
        return cout << "error: "  << e.what() << endl, 1;
    }
    catch (const exception& e)
    {
        return cout << "error: "  << e.what() << endl, 1;
    }
    catch(...)
    {
        return cout << "unknown exception" << endl, 1;
    }
    return EXIT_SUCCESS;
}

App::App(CommandLineParser& cmd)
{
    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tm - change mode GPU <-> CPU\n"
         << "\tg - convert image to gray or not\n"
         << "\to - save output image once, or switch on/off video save\n"
         << "\t1/q - increase/decrease HOG scale\n"
         << "\t2/w - increase/decrease levels count\n"
         << "\t3/e - increase/decrease HOG group threshold\n"
         << "\t4/r - increase/decrease hit threshold\n"
         << endl;


    use_gpu = true;
    make_gray = cmd.get<bool>("g");
    resize_scale = cmd.get<double>("s");
    win_width = cmd.get<bool>("l") == true ? 64 : 48;
    vdo_source = cmd.get<string>("v");
    img_source = cmd.get<string>("i");
    output = cmd.get<string>("o");
    camera_id = cmd.get<int>("c");

    win_stride_width = 8;
    win_stride_height = 8;
    gr_threshold = 8;
    nlevels = 13;
    hit_threshold = win_width == 48 ? 1.4 : 0.;
    scale = 1.05;
    gamma_corr = true;
    write_once = false;

    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: " << win_width << endl;
    cout << "Win stride: (" << win_stride_width << ", " << win_stride_height << ")\n";
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}

void App::run()
{
    running = true;
    VideoWriter video_writer;

    Size win_size(win_width, win_width * 2);
    Size win_stride(win_stride_width, win_stride_height);

    // Create HOG descriptors and detectors here
    vector<float> detector;
    if (win_size == Size(64, 128))
        detector = ocl::HOGDescriptor::getPeopleDetector64x128();
    else
        detector = ocl::HOGDescriptor::getPeopleDetector48x96();


    ocl::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
                               ocl::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
                               ocl::HOGDescriptor::DEFAULT_NLEVELS);
    HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
                          HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
    gpu_hog.setSVMDetector(detector);
    cpu_hog.setSVMDetector(detector);

    while (running)
    {
        VideoCapture vc;
        Mat frame;

        if (vdo_source!="")
        {
            vc.open(vdo_source.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + vdo_source));
            vc >> frame;
        }
        else if (camera_id != -1)
        {
            vc.open(camera_id);
            if (!vc.isOpened())
            {
                stringstream msg;
                msg << "can't open camera: " << camera_id;
                throw runtime_error(msg.str());
            }
            vc >> frame;
        }
        else
        {
            frame = imread(img_source);
            if (frame.empty())
                throw runtime_error(string("can't open image file: " + img_source));
        }

        Mat img_aux, img, img_to_show;
        ocl::oclMat gpu_img;

        // Iterate over all frames
        bool verify = false;
        while (running && !frame.empty())
        {
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, CV_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, CV_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (abs(scale-1.0)>0.001)
            {
                Size sz((int)((double)img_aux.cols/resize_scale), (int)((double)img_aux.rows/resize_scale));
                resize(img_aux, img, sz);
            }
            else img = img_aux;
            img_to_show = img;
            gpu_hog.nlevels = nlevels;
            cpu_hog.nlevels = nlevels;
            vector<Rect> found;

            // Perform HOG classification
            hogWorkBegin();
            if (use_gpu)
            {
                gpu_img.upload(img);
                gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride,
                                         Size(0, 0), scale, gr_threshold);
                if (!verify)
                {
                    // verify if GPU output same objects with CPU at 1st run
                    verify = true;
                    vector<Rect> ref_rst;
                    cvtColor(img, img, CV_BGRA2BGR);
                    cpu_hog.detectMultiScale(img, ref_rst, hit_threshold, win_stride,
                                             Size(0, 0), scale, gr_threshold-2);
                    double accuracy = checkRectSimilarity(img.size(), ref_rst, found);
                    cout << "\naccuracy value: " << accuracy << endl;
                }
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

            if (use_gpu)
                putText(img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            else
                putText(img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (HOG only): " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            imshow("opencv_gpu_hog", img_to_show);
            if (vdo_source!="" || camera_id!=-1) vc >> frame;

            workEnd();

            if (output!="" && write_once)
            {
                if (img_source!="")     // wirte image
                {
                    write_once = false;
                    imwrite(output, img_to_show);
                }
                else                    //write video
                {
                    if (!video_writer.isOpened())
                    {
                        video_writer.open(output, CV_FOURCC('x','v','i','d'), 24,
                                          img_to_show.size(), true);
                        if (!video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray) cvtColor(img_to_show, img, CV_GRAY2BGR);
                    else cvtColor(img_to_show, img, CV_BGRA2BGR);

                    video_writer << img;
                }
            }

            handleKey((char)waitKey(3));
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
    case 'o':
    case 'O':
        write_once = !write_once;
        break;
    }
}


inline void App::hogWorkBegin()
{
    hog_work_begin = getTickCount();
}

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

inline void App::workBegin()
{
    work_begin = getTickCount();
}

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


double App::checkRectSimilarity(Size sz,
                                std::vector<Rect>& ob1,
                                std::vector<Rect>& ob2)
{
    double final_test_result = 0.0;
    size_t sz1 = ob1.size();
    size_t sz2 = ob2.size();

    if(sz1 != sz2)
    {
        return sz1 > sz2 ? (double)(sz1 - sz2) : (double)(sz2 - sz1);
    }
    else
    {
        if(sz1==0 && sz2==0)
            return 0;
        cv::Mat cpu_result(sz, CV_8UC1);
        cpu_result.setTo(0);


        for(vector<Rect>::const_iterator r = ob1.begin(); r != ob1.end(); r++)
        {
            cv::Mat cpu_result_roi(cpu_result, *r);
            cpu_result_roi.setTo(1);
            cpu_result.copyTo(cpu_result);
        }
        int cpu_area = cv::countNonZero(cpu_result > 0);


        cv::Mat gpu_result(sz, CV_8UC1);
        gpu_result.setTo(0);
        for(vector<Rect>::const_iterator r2 = ob2.begin(); r2 != ob2.end(); r2++)
        {
            cv::Mat gpu_result_roi(gpu_result, *r2);
            gpu_result_roi.setTo(1);
            gpu_result.copyTo(gpu_result);
        }

        cv::Mat result_;
        multiply(cpu_result, gpu_result, result_);
        int result = cv::countNonZero(result_ > 0);
        if(cpu_area!=0 && result!=0)
            final_test_result = 1.0 - (double)result/(double)cpu_area;
        else if(cpu_area==0 && result!=0)
            final_test_result = -1;
    }
    return final_test_result;
}
