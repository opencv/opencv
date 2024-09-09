#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>  // cv::Canny()
#include <iostream>
#include <cstdlib> // For system()

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
using std::string;

int main(int argc, char** argv)
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <video_file_path>" << endl;
        return -1;
    }

    string videoFilePath = argv[1];
    Mat frame;
    cout << "Opening video file: " << videoFilePath << endl;
    VideoCapture capture(videoFilePath);
    if (!capture.isOpened())
    {
        cerr << "ERROR: Can't open video file" << endl;
        return 1;
    }

    cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << endl;

    cout << endl << "Start processing..." << endl;

    size_t nFrames = 0;
    bool enableProcessing = false;
    int64 t0 = cv::getTickCount();
    int64 processingTime = 0;

    int frameWidth = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(CAP_PROP_FPS);
    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');

    system("mkdir -p videocapture_camera");
    string outputFilePath = "videocapture_camera/output.avi";
    VideoWriter writer(outputFilePath, fourcc, fps, Size(frameWidth, frameHeight));

    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cerr << "ERROR: Can't grab video frame." << endl;
            break;
        }
        nFrames++;
        if (nFrames % 10 == 0)
        {
            const int N = 10;
            int64 t1 = cv::getTickCount();
            cout << "Frames captured: " << cv::format("%5lld", (long long int)nFrames)
                 << "    Average FPS: " << cv::format("%9.1f", (double)getTickFrequency() * N / (t1 - t0))
                 << "    Average time per frame: " << cv::format("%9.2f ms", (double)(t1 - t0) * 1000.0f / (N * getTickFrequency()))
                 << "    Average processing time: " << cv::format("%9.2f ms", (double)(processingTime) * 1000.0f / (N * getTickFrequency()))
                 << std::endl;
            t0 = t1;
            processingTime = 0;
        }
        if (!enableProcessing)
        {
            // imshow("Frame", frame); // 注释掉
            writer.write(frame);
        }
        else
        {
            int64 tp0 = cv::getTickCount();
            Mat processed;
            cv::Canny(frame, processed, 400, 1000, 5);
            processingTime += cv::getTickCount() - tp0;
            // imshow("Frame", processed); // 注释掉
            writer.write(processed);
        }
        // int key = waitKey(1); // 注释掉
        // if (key == 27/*ESC*/) // 注释掉
        //     break; // 注释掉
        // if (key == 32/*SPACE*/) // 注释掉
        // {
        //     enableProcessing = !enableProcessing; // 注释掉
        //     cout << "Enable frame processing ('space' key): " << enableProcessing << endl; // 注释掉
        // }
    }

    cout << "Number of captured frames: " << nFrames << endl;
    cout << "Video saved to " << outputFilePath << endl;
    return nFrames > 0 ? 0 : 1;
}

