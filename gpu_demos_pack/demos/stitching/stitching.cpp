#include <iostream>
#include <math.h>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <tbb/tbb.h>

#include "utility.h"

// 0 - wait
// 1 - process
// 2 - finished
tbb::atomic<int> taskState;
tbb::atomic<int> stopThread;

std::vector<cv::Mat> imgs;
cv::Mat final_pano;

struct StitchingTask : tbb::task
{
    volatile bool useGpu;
    volatile double gpu_time;
    volatile double cpu_time;

    StitchingTask() : useGpu(false), cpu_time(1.0), gpu_time(1.0)
    {
    }

    tbb::task* execute()
    {
        while (taskState != 1)
            tbb::this_tbb_thread::sleep(tbb::tick_count::interval_t(0.03));

        int64 proc_start = cv::getTickCount();
        {
            cv::Stitcher stitcher = cv::Stitcher::createDefault(useGpu);

            stitcher.estimateTransform(imgs);

            stitcher.composePanorama(imgs, final_pano);
        }

        double pano_time = (cv::getTickCount() - proc_start) / cv::getTickFrequency();
        if (useGpu)
            gpu_time = pano_time;
        else
            cpu_time = pano_time;

        final_pano = final_pano.clone();
        taskState.store(2);

        if (!stopThread)
            self().recycle_to_reexecute();

        return 0;
    }
};

class App : public BaseApp
{
public:
    App();

protected:
    void process();
    bool processKey(int key);
    void printHelp();

private:
    std::vector< std::pair<std::string, int> > panoSources;
    int panoIdx;
};

App::App()
{
    panoSources.push_back(std::make_pair("data/stitching_0%d.jpg", 9));
    panoIdx = 0;
}

void App::process()
{
    sources.clear();
    for (size_t i = 0; i < panoSources.size(); ++i)
        sources.push_back(new ImagesVideoSource(panoSources[i].first));

    const int w_width = 1920;
    const int w_height = 1080;
    const int w_grid_border = 30;
    const int w_half_border = w_grid_border >> 1;

    cv::namedWindow("Stitching Demo", CV_WINDOW_NORMAL);
    cvSetWindowProperty("Stitching Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    cvSetWindowProperty("Stitching Demo", CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);

    int track = 10;

    taskState.store(0);
    stopThread.store(0);
    StitchingTask* task = new(tbb::task::allocate_root()) StitchingTask;
    tbb::task::enqueue(*task);

    cv::Mat imgGrid(w_height, w_width, CV_8UC3, cv::Scalar::all(0));
    cv::Mat imgGridForDraw;

    while(!exited)
    {
        if (taskState == 0)
        {
            imgs.clear();
            int frames_counter = 0;
            while (frames_counter++ < panoSources[panoIdx].second)
            {
                cv::Mat frame;
                sources[panoIdx]->next(frame);
                if (frame.empty()) break;
                imgs.push_back(frame.clone());
            }

            size_t src_size = imgs.size();
            int h = (int)sqrt((double)src_size);
            int w = ((h+1) * h <= src_size )? h+1 : h;

            int t = static_cast<int>(src_size) - w * h;
            int t_com = 0;

            imgGrid.setTo(cv::Scalar::all(0));

            int w_cell = (w_width - 2 * w_grid_border) / (w + (t ? 1 : 0));
            int h_cell = (w_height - 2 * w_grid_border) / h;

            int k = 0;
            for(int i = w_grid_border; i < w_width - w_grid_border; i += w_cell)
            {
                t_com = 0;
                for(int j = w_grid_border; j < w_height - w_grid_border; j +=h_cell)
                {
                    int start_sub_pos = ( t > 0 && t_com >= t) ? (w_cell >> 1): 0;
                    if (k >= src_size ) break;
                    const cv::Mat sub(imgGrid, cv::Rect(i + w_half_border  + start_sub_pos, j + w_half_border, w_cell - w_half_border, h_cell - w_half_border));
                    cv::resize(imgs[k++], sub, sub.size());
                    if (t) t_com++;
                }
            }

            panoIdx = (panoIdx + 1) % panoSources.size();

            imgGrid.copyTo(imgGridForDraw);

            taskState.store(1);
        }

        std::ostringstream msg;
        if (track <= 10)
        {
            cv::imshow("Stitching Demo", imgGridForDraw);
            cv::displayOverlay("Stitching Demo", "GTC presentation stitching OpenCV GPU vs. CPU comparison", 1);

            msg << "Sources images: " << imgs.size();

            addText(imgGridForDraw, msg.str(), cv::Point(w_width >> 2 , w_height >> 2 ),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");

            msg << "Images resolution: " << cvRound(imgs[0].cols * imgs[0].rows / 100000)/10.0f << " Mp";
            addText(imgGridForDraw, msg.str(), cv::Point( w_width >> 2, (w_height >> 2) + 70),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");

            msg << "Used " << ( task->useGpu ? "GPU" : "CPU") << " version";
            addText(imgGridForDraw, msg.str(), cv::Point(w_width >> 2, (w_height >> 2) + 140),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");
            cv::imshow("Stitching Demo", imgGridForDraw);
        }

        cv::Mat prog(imgGridForDraw, cv::Rect(w_width >> 2, (w_height >> 2) + 210, track, 40));
        prog.setTo(cv::Scalar(255,255,255,0));
        cv::imshow("Stitching Demo", imgGridForDraw);

        processKey(cv::waitKey(300));
        track +=10;
        if (track >= (w_width >> 1))
            track = w_width >> 1;

        if (taskState == 2)
        {
            cv::Mat dst(w_height, w_width, CV_8UC3, cv::Scalar::all(0));
            int sub_rows = (w_height * final_pano.rows) / final_pano.cols;
            cv::Mat sub_dst(dst, cv::Rect(0, (w_height - sub_rows) >> 1, w_width, sub_rows ));
            cv::resize(final_pano, sub_dst, sub_dst.size());

            msg << "Panorama resolution: " << cvRound(final_pano.cols * final_pano.rows / 100000)/10.0f << " Mp";
            addText(dst, msg.str(), cv::Point(w_width >> 2 , w_height >> 4 ),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 255, 255, 0)));
            msg.str("");
            cv::Scalar time_color;
            if (task->useGpu)
                time_color = cv::Scalar(124,250,0,0);
            else
                time_color = cv::Scalar(255,19,19,0);

            msg << "Panorama composition time: " << (task->useGpu ? task->gpu_time : task->cpu_time)  << " sec";
            addText(dst, msg.str(), cv::Point( w_width >> 2, (w_height >> 4) + 70),
                    cv::fontQt("CV_FONT_BLACK", 40, time_color));
            msg.str("");

            if (task->useGpu)
            {
                msg << "GPU optimization speedup: " << task->cpu_time / task->gpu_time << "x";
                addText(dst, msg.str(), cv::Point(w_width >> 2, (w_height >> 4) + 140),
                    cv::fontQt("CV_FONT_BLACK", 40, cv::Scalar(255, 19, 19, 0)));
                msg.str("");
            }

            cv::imshow("Stitching Demo", dst);
            processKey(cv::waitKey(10000));

            track = 10;

            task->useGpu = !task->useGpu;
            if (!task->useGpu)
                taskState.store(0);
            else
            {
                imgGrid.copyTo(imgGridForDraw);
                taskState.store(1);
            }
        }
    }
}

bool App::processKey(int key)
{
    switch (key & 0xff)
    {
    case 27:
        exited = true;
        stopThread.store(1);
        break;

    default:
        return BaseApp::processKey(key);
    }

    return true;
}

void App::printHelp()
{
    std::cout << "This sample demonstrates Rotation model images stitcher" << std::endl;
    std::cout << "Usage: demo_stereo_matching [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    BaseApp::printHelp();
}

RUN_APP(App)
