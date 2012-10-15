#include <iostream>
#include <math.h>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility_lib/utility_lib.h"
#include <tbb/tbb.h>

enum stitchers
{
    USE_GPU = 1,
    USE_CPU = 0
};

static char imgs_count_tpl[] = "Sources images: ";
static char imgs_resolution_tpl[] = "Images resolution: ";

static char pano_resolution_tpl[] = "Panorama resolution: ";
static char pano_time_tpl[] = "Panorama composition time: ";
static char gpu_speedup_tpl[] = "GPU optimizations speedup: ";

static char pano_2_path[] = "data/stitching/2/IMG_0%d.JPG";
static int pano_2_imgs = 9;

static char pano_3_path[] = "data/stitching/3/IMG_0%d.JPG";
static int pano_3_imgs = 6;

static char pano_4_path[] = "data/stitching/4/IMG_c500d_%02d.jpg";
static int pano_4_imgs = 10;

tbb::concurrent_queue<double> pano_queue;
tbb::atomic<int> mit;
tbb::atomic<int> stop_thread;
tbb::atomic<double> pano_time;
tbb::atomic<stitchers> stitcher_type;
std::vector<cv::Mat> imgs;
cv::Mat final_pano;

class StitchingTask: public tbb::task
{
    tbb::task* execute()
    {
        {
            while(mit) ;
            cv::Stitcher _stitcher = cv::Stitcher::createDefault(_type);
            //this one crashed in deallocation stage
            //_stitcher.setFeaturesMatcher(new cv::detail::BestOf2NearestMatcher(_type, 0.5));

            int64 proc_start = cv::getTickCount();

            _stitcher.estimateTransform(imgs);

            _stitcher.composePanorama(imgs, final_pano);

            pano_time.fetch_and_store((cv::getTickCount() - proc_start) / cv::getTickFrequency());
            if (_type)
                gpu_time = pano_time;
            else
                cpu_time = pano_time;

            stitcher_type.fetch_and_store(_type);
            final_pano = final_pano.clone();

            mit++;
        }

        _type = (stitchers)((_type + 1) % 2);
        if (!stop_thread)
            self().recycle_to_reexecute();
        return 0;
    }

public:
    StitchingTask(stitchers type): _type(type),cpu_time(1.0), gpu_time(1.0){}

    void setType(stitchers type)
    {
        _type = type;
    }

    volatile stitchers getType()
    {
        return _type;
    }

    volatile double speedUp()
    {
        return cpu_time / gpu_time;
    }

private:
    volatile stitchers _type;
    volatile double gpu_time;
    volatile double cpu_time;
};

class App : public BaseApp
{
public:
    App(){defaultPanoIdx=2;}

protected:
    void process();
    bool processKey(int key);
    void printHelp();
    bool parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[]);
    bool parseCmdArgs(int& i, int argc, const char* argv[]);

private:
    int defaultPanoIdx;
};

void App::process()
{
    StitchingTask* task = new( tbb::task::allocate_root() )StitchingTask(USE_CPU);
    stop_thread.fetch_and_store(0);

    size_t w_width = 1920, w_height = 1080, w_grid_border = 30, w_half_border = w_grid_border >> 1;

    int max_imgs= 20;
    if (!sources.size())
    {
        switch (defaultPanoIdx)
        {
        case 3:
            sources.push_back(new ImagesVideoSource(pano_3_path));
            max_imgs = pano_3_imgs;
            break;
        case 4:
            sources.push_back(new ImagesVideoSource(pano_4_path));
            max_imgs = pano_4_imgs;
            break;
        case 2:
        default:
            sources.push_back(new ImagesVideoSource(pano_2_path));
            max_imgs = pano_2_imgs;
            break;
        };
    }

    int frames_counter = 0;
    while (frames_counter++ < max_imgs)
    {
        cv::Mat frame;
        sources[0]->next(frame);
        if (frame.empty()) break;
        imgs.push_back(frame.clone());
    }

    cv::namedWindow("stitching_gemo", CV_WINDOW_NORMAL);
    cv::Mat imgGrid(w_height, w_width, CV_8UC3, cv::Scalar::all(0));

    int src_size = imgs.size();
    int h = (int)sqrt((double)src_size);
    int w = ((h+1) * h <= src_size )? h+1 : h;

    int t, t_com = 0;
    t = src_size - w * h;

    cvSetWindowProperty("stitching_gemo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    cvSetWindowProperty("stitching_gemo", CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);

    size_t w_cell = (w_width - 2 * w_grid_border) / (w + (t ? 1 : 0));
    size_t h_cell = (w_height - 2 * w_grid_border) / h;

    int k = 0;
    for(size_t i = w_grid_border; i < w_width - w_grid_border; i += w_cell)
    {
        t_com = 0;
        for(size_t j = w_grid_border; j < w_height - w_grid_border; j +=h_cell)
        {
            size_t start_sub_pos = ( t > 0 && t_com >= t) ? (w_cell >> 1): 0;
            if (k >= src_size ) break;
            const cv::Mat sub(imgGrid, cv::Rect(i + w_half_border  + start_sub_pos, j + w_half_border, w_cell - w_half_border, h_cell - w_half_border));
            cv::resize(imgs[k++], sub, sub.size());
            if (t) t_com++;
        }
    }

    tbb::task::enqueue(*task);
    int track = 10;
    cv::Mat imgGridForDraw = imgGrid.clone();
    while(!exited)
    {
        std::ostringstream msg;
        if(track <=10 )
        {
            cv::imshow("stitching_gemo", imgGridForDraw);
            cv::displayOverlay("stitching_gemo", "GTC presentation stitching OpenCV GPU vs. CPU comparisonn", 1);

            msg << imgs_count_tpl << imgs.size();

            addText(imgGridForDraw, msg.str(), cv::Point(w_width >> 2 , w_height >> 2 ),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");

            msg << imgs_resolution_tpl << cvRound(imgs[0].cols * imgs[0].rows / 100000)/10.0f << " Mp";
            addText(imgGridForDraw, msg.str(), cv::Point( w_width >> 2, (w_height >> 2) + 70),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");

            msg << "Used " << ( (task->getType() == USE_CPU)? "CPU" : "GPU" ) << " version";
            addText(imgGridForDraw, msg.str(), cv::Point(w_width >> 2, (w_height >> 2) + 140),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");
            cv::imshow("stitching_gemo", imgGridForDraw);
        }

        cv::Mat prog(imgGridForDraw, cv::Rect(w_width >> 2, (w_height >> 2) + 210, track, 40));
        prog.setTo(cv::Scalar(255,255,255,0));
        cv::imshow("stitching_gemo", imgGridForDraw);
        processKey(cv::waitKey(300));
        track +=10;
        if (track >= (w_width >> 1)) track = w_width >> 1;
        if(mit)
        {

            cv::Mat dst(w_height, w_width, CV_8UC3, cv::Scalar::all(0));
            int sub_rows = (w_height * final_pano.rows) / final_pano.cols;
            cv::Mat sub_dst(dst, cv::Rect(0, (w_height - sub_rows) >> 1, w_width, sub_rows ));
            cv::resize(final_pano, sub_dst, sub_dst.size());

            msg << pano_resolution_tpl << cvRound(final_pano.cols * final_pano.rows / 100000)/10.0f << " Mp";
            addText(dst, msg.str(), cv::Point(w_width >> 2 , w_height >> 4 ),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");
            cv::Scalar time_color;
            if (stitcher_type == USE_GPU)
                time_color = cv::Scalar(124,250,0,0);
            else
                time_color = cv::Scalar(255,19,19,0);

            msg << pano_time_tpl << pano_time  << " sec";
            addText(dst, msg.str(), cv::Point( w_width >> 2, (w_height >> 4) + 70),
                    cv::fontQt("CV_FONT_BLACK", 40, time_color));
            msg.str("");

            if (stitcher_type == USE_GPU)
            {
                msg << gpu_speedup_tpl << task->speedUp() << "x";
                addText(dst, msg.str(), cv::Point(w_width >> 2, (w_height >> 4) + 140),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 19, 19, 0)));
                msg.str("");
            }

            cv::imshow("stitching_gemo", dst);
            processKey(cv::waitKey(10000) & 0xff);
            imgGridForDraw = imgGrid.clone();
            track = 10;
            mit.fetch_and_store(0);
        }
    }
}

bool App::processKey(int key)
{
    switch (key & 0xff)
    {
    case 27:
    {
        exited = true;
        stop_thread.fetch_and_store(1);
        break;
    }

    default:
        return BaseApp::processKey(key);
    }
    return true;
}

void App::printHelp()
{
    std::cout << "Rotation model images stitcher.\n\n"
         << "Usage: -v <images template path>\n";
    BaseApp::printHelp();
}

bool App::parseCmdArgs(int& i, int argc, const char* argv[])
{
    std::string arg(argv[i]);

    if (arg == "--pano")
    {
        ++i;

        if (i >= argc)
            throw std::runtime_error("Missing pano index after --pano");

        defaultPanoIdx = atoi(argv[i]);
    }
    else
        return false;

    return true;
}

bool App::parseFrameSourcesCmdArgs(int& i, int argc, const char* argv[])
{
    std::string arg(argv[i]);

    if (arg == "-i")
    {
        ++i;

        if (i >= argc)
            throw std::runtime_error("Missing file name after -i");

        sources.push_back(new ImageSource(argv[i]));
    }
    else if (arg == "-v")
    {
        ++i;

        if (i >= argc)
            throw std::runtime_error("Missing file name after -v");

        sources.push_back(new ImagesVideoSource(argv[i]));
    }
    else if (arg == "-w")
    {
        ++i;

        if (i >= argc)
            throw std::runtime_error("Missing value after -w");

        frame_width = atoi(argv[i]);
    }
    else if (arg == "-h")
    {
        ++i;

        if (i >= argc)
            throw std::runtime_error("Missing value after -h");

        frame_height = atoi(argv[i]);
    }
    else if (arg == "-c")
    {
        ++i;

        if (i >= argc)
            throw std::runtime_error("Missing value after -c");

        sources.push_back(new CameraSource(atoi(argv[i]), frame_width, frame_height));
    }
    else
        return false;

    return true;
}

RUN_APP(App)
