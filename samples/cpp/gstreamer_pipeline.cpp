#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

string getGstDemuxPlugin(string container);
string getGstAvDecodePlugin(string codec);

int main(int argc, char *argv[])
{
    const string keys =
          "{h help usage ? |           | print help messages   }"
          "{p pipeline     |gst-default| pipeline name  (supported: 'gst-default', 'gst-vaapi', 'gst-libav', 'ffmpeg') }"
          "{ct container   |mp4        | container name (supported: 'mp4', 'mov', 'avi', 'mkv') }"
          "{cd codec       |h264       | codec name     (supported: 'h264', 'h265', 'mpeg2', 'mpeg4', 'mjpeg', 'vp8') }"
          "{f file path    |           | path to file }"
          "{fm fast        |           | fast measure fps }";

    CommandLineParser parser(argc, argv, keys);

    parser.about("This program shows how to read a video file with GStreamer pipeline with OpenCV.");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    bool   arg_fast_measure = parser.has("fast");              // fast measure fps
    string arg_pipeline     = parser.get<string>("pipeline"),  // GStreamer pipeline type
           arg_container    = parser.get<string>("container"), // container type
           arg_codec        = parser.get<string>("codec"),     // codec type
           arg_file_name    = parser.get<string>("file");      // path to videofile
    VideoCapture cap;

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    // Choose the constructed GStreamer pipeline
    if (arg_pipeline.find("gst") == 0)
    {
        ostringstream pipeline;
        pipeline << "filesrc location=\"" << arg_file_name << "\"";
        pipeline << " ! " << getGstDemuxPlugin(arg_container);

        if (arg_pipeline.find("default") == 4) {
            pipeline << " ! decodebin";
        }
        else if (arg_pipeline.find("vaapi1710") == 4)
        {
            pipeline << " ! vaapidecodebin";
            if (arg_container == "mkv")
            {
                pipeline << " ! autovideoconvert";
            }
            else
            {
                pipeline << " ! video/x-raw, format=YV12";
            }
        }
        else if (arg_pipeline.find("libav") == 4)
        {
            pipeline << " ! " << getGstAvDecodePlugin(arg_codec);
        }
        else
        {
            parser.printMessage();
            cout << "Unsupported pipeline: " << arg_pipeline << endl;
            return -4;
        }

        pipeline << " ! videoconvert";
        pipeline << "     n-threads=" << getNumThreads();
        pipeline << " ! appsink sync=false";
        cap = VideoCapture(pipeline.str(), CAP_GSTREAMER);
    }
    else if (arg_pipeline == "ffmpeg")
    {
        cap = VideoCapture(arg_file_name, CAP_FFMPEG);
    }
    else
    {
        parser.printMessage();
        cout << "Unsupported pipeline: " << arg_pipeline << endl;
        return -4;
    }

    // Choose a show video or only measure fps
    cout << "_____________________________________" << '\n';
    cout << "Start measure frame per seconds (fps)" << '\n';
    cout << "Loading ..." << '\n';

    Mat frame;
    vector<double> tick_counts;

    cout << "Start decoding: " << arg_file_name;
    cout << " (" << arg_pipeline << ")" << endl;

    while(true)
    {
        int64 temp_count_tick = getTickCount();
        cap >> frame;
        temp_count_tick = getTickCount() - temp_count_tick;
        if (frame.empty()) { break; }
        tick_counts.push_back(static_cast<double>(temp_count_tick));
        if (arg_fast_measure && (tick_counts.size() > 1000)) { break; }

    }
    double time_fps = sum(tick_counts)[0] / getTickFrequency();

    if (tick_counts.size() != 0)
    {
        cout << "Finished: " << tick_counts.size() << " in " << time_fps <<" sec ~ " ;
        cout << tick_counts.size() / time_fps <<" fps " << endl;
    }
    else
    {
        cout << "Failed decoding: " << arg_file_name;
        cout << " (" << arg_pipeline << ")" << endl;
        return -5;
    }
    return 0;
}

// Choose a video container
string getGstDemuxPlugin(string container) {
    if      (container == "avi") { return "avidemux"; }
    else if (container == "mp4") { return "qtdemux"; }
    else if (container == "mov") { return "qtdemux"; }
    else if (container == "mkv") { return "matroskademux"; }
    return string();
}

// Choose a codec
string getGstAvDecodePlugin(string codec) {
    if      (codec == "h264")  { return "h264parse ! avdec_h264"; }
    else if (codec == "h265")  { return "h265parse ! avdec_h265"; }
    else if (codec == "mpeg2") { return "avdec_mpeg2video"; }
    else if (codec == "mpeg4") { return "avdec_mpeg4"; }
    else if (codec == "mjpeg") { return "avdec_mjpeg"; }
    else if (codec == "vp8")   { return "avdec_vp8"; }
    return string();
}
