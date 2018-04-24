#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

class GStreamerPipeline
{
 public:
    // Preprocessing arguments command line
    GStreamerPipeline(int argc, char *argv[])
    {
        const string keys =
              "{h help usage ? |           | print help messages   }"
              "{m mode         |           | coding mode (supported: encode, decode) }"
              "{p pipeline     |default    | pipeline name  (supported: 'default', 'gst-basic', 'gst-vaapi', 'gst-libav', 'ffmpeg') }"
              "{ct container   |mp4        | container name (supported: 'mp4', 'mov', 'avi', 'mkv') }"
              "{cd codec       |h264       | codec name     (supported: 'h264', 'h265', 'mpeg2', 'mpeg4', 'mjpeg', 'vp8') }"
              "{f file path    |           | path to file }"
              "{vr resolution  |720p       | video resolution for encoding (supported: '720p', '1080p', '4k') }"
              "{fps            |30         | fix frame per second for encoding (supported: fps > 0) }"
              "{fm fast        |           | fast measure fps }";
        cmd_parser = new CommandLineParser(argc, argv, keys);
        cmd_parser->about("This program shows how to read a video file with GStreamer pipeline with OpenCV.");

        if (cmd_parser->has("help"))
        {
            cmd_parser->printMessage();
            exit_code = -1;
        }

        fast_measure = cmd_parser->has("fast");               // fast measure fps
        fix_fps      = cmd_parser->get<int>("fps");           // fixed frame per second
        pipeline     = cmd_parser->get<string>("pipeline"),   // gstreamer pipeline type
        container    = cmd_parser->get<string>("container"),  // container type
        mode         = cmd_parser->get<string>("mode"),       // coding mode
        codec        = cmd_parser->get<string>("codec"),      // codec type
        file_name    = cmd_parser->get<string>("file"),       // path to videofile
        resolution   = cmd_parser->get<string>("resolution"); // video resolution

        if (!cmd_parser->check())
        {
            cmd_parser->printErrors();
            exit_code = -1;
        }
        exit_code = 0;
    }

    ~GStreamerPipeline() { delete cmd_parser; }

    // Start pipeline
    int run()
    {
        if (exit_code < 0) { return exit_code; }
        if      (mode == "decode") { if (createDecodePipeline() < 0) return -1; }
        else if (mode == "encode") { if (createEncodePipeline() < 0) return -1; }
        else
        {
            cout << "Unsupported mode: " << mode << endl;
            cmd_parser->printErrors();
            return -1;
        }
        cout << "_____________________________________" << endl;
        cout << "Pipeline " << mode << ":" << endl;
        cout << stream_pipeline.str() << endl;
        // Choose a show video or only measure fps
        cout << "_____________________________________" << endl;
        cout << "Start measure frame per seconds (fps)" << endl;
        cout << "Loading ..." << endl;

        vector<double> tick_counts;

        cout << "Start " << mode << ": " << file_name;
        cout << " (" << pipeline << ")" << endl;

        while(true)
        {
            int64 temp_count_tick = 0;
            if (mode == "decode")
            {
                Mat frame;
                temp_count_tick = getTickCount();
                cap >> frame;
                temp_count_tick = getTickCount() - temp_count_tick;
                if (frame.empty()) { break; }
            }
            else if (mode == "encode")
            {
                Mat element;
                while(!cap.grab());
                cap.retrieve(element);
                temp_count_tick = getTickCount();
                wrt << element;
                temp_count_tick = getTickCount() - temp_count_tick;
            }

            tick_counts.push_back(static_cast<double>(temp_count_tick));
            if (((mode == "decode") && fast_measure && (tick_counts.size() > 1e3)) ||
                ((mode == "encode") && (tick_counts.size() > 3e3)) ||
                ((mode == "encode") && fast_measure && (tick_counts.size() > 1e2)))
            { break; }

        }
        double time_fps = sum(tick_counts)[0] / getTickFrequency();

        if (tick_counts.size() != 0)
        {
            cout << "Finished: " << tick_counts.size() << " in " << time_fps <<" sec ~ " ;
            cout << tick_counts.size() / time_fps <<" fps " << endl;
        }
        else
        {
            cout << "Failed " << mode << ": " << file_name;
            cout << " (" << pipeline << ")" << endl;
            return -1;
        }
        return 0;
    }

    // Free video resource
    void close()
    {
        cap.release();
        wrt.release();
    }

 private:
    // Choose the constructed GStreamer pipeline for decode
    int createDecodePipeline()
    {
        if (pipeline == "default") {
            cap = VideoCapture(file_name, CAP_GSTREAMER);
        }
        else if (pipeline.find("gst") == 0)
        {
            stream_pipeline << "filesrc location=\"" << file_name << "\"";
            stream_pipeline << " ! " << getGstMuxPlugin();

            if (pipeline.find("basic") == 4)
            {
                stream_pipeline << getGstDefaultCodePlugin();
            }
            else if (pipeline.find("vaapi1710") == 4)
            {
                stream_pipeline << getGstVaapiCodePlugin();
            }
            else if (pipeline.find("libav") == 4)
            {
                stream_pipeline << getGstAvCodePlugin();
            }
            else
            {
                cout << "Unsupported pipeline: " << pipeline << endl;
                cmd_parser->printErrors();
                return -1;
            }

            stream_pipeline << " ! videoconvert n-threads=" << getNumThreads();
            stream_pipeline << " ! appsink sync=false";
            cap = VideoCapture(stream_pipeline.str(), CAP_GSTREAMER);
        }
        else if (pipeline == "ffmpeg")
        {
            cap = VideoCapture(file_name, CAP_FFMPEG);
            stream_pipeline << "default pipeline for ffmpeg" << endl;
        }
        else
        {
            cout << "Unsupported pipeline: " << pipeline << endl;
            cmd_parser->printErrors();
            return -1;
        }
        return 0;
    }

    // Choose the constructed GStreamer pipeline for encode
    int createEncodePipeline()
    {
        if (checkConfiguration() < 0) return -1;
        ostringstream test_pipeline;
        test_pipeline << "videotestsrc pattern=smpte";
        test_pipeline << " ! video/x-raw, " << getVideoSettings();
        test_pipeline << " ! appsink sync=false";
        cap = VideoCapture(test_pipeline.str(), CAP_GSTREAMER);

        if (pipeline == "default") {
            wrt = VideoWriter(file_name, CAP_GSTREAMER, getFourccCode(), fix_fps, fix_size, true);
        }
        else if (pipeline.find("gst") == 0)
        {
            stream_pipeline << "appsrc ! videoconvert n-threads=" << getNumThreads() << " ! ";

            if (pipeline.find("basic") == 4)
            {
                stream_pipeline << getGstDefaultCodePlugin();
            }
            else if (pipeline.find("vaapi1710") == 4)
            {
                stream_pipeline << getGstVaapiCodePlugin();
            }
            else if (pipeline.find("libav") == 4)
            {
                stream_pipeline << getGstAvCodePlugin();
            }
            else
            {
                cout << "Unsupported pipeline: " << pipeline << endl;
                cmd_parser->printErrors();
                return -1;
            }

            stream_pipeline << " ! " << getGstMuxPlugin();
            stream_pipeline << " ! filesink location=\"" << file_name << "\"";
            wrt = VideoWriter(stream_pipeline.str(), CAP_GSTREAMER, 0, fix_fps, fix_size, true);
        }
        else if (pipeline == "ffmpeg")
        {
            wrt = VideoWriter(file_name, CAP_FFMPEG, getFourccCode(), fix_fps, fix_size, true);
            stream_pipeline << "default pipeline for ffmpeg" << endl;
        }
        else
        {
            cout << "Unsupported pipeline: " << pipeline << endl;
            cmd_parser->printErrors();
            return -1;
        }
        return 0;
    }

    // Choose video resolution for encoding
    string getVideoSettings()
    {
        ostringstream video_size;
        if (fix_fps > 0) { video_size << "framerate=" << fix_fps << "/1, "; }
        else
        {
            cout << "Unsupported fps (< 0): " << fix_fps << endl;
            cmd_parser->printErrors();
            return string();
        }

        if      (resolution == "720p")  { fix_size = Size(1280, 720);  }
        else if (resolution == "1080p") { fix_size = Size(1920, 1080); }
        else if (resolution == "4k")    { fix_size = Size(3840, 2160); }
        else
        {
            cout << "Unsupported video resolution: " << resolution << endl;
            cmd_parser->printErrors();
            return string();
        }

        video_size << "width=" << fix_size.width << ", height=" << fix_size.height;
        return video_size.str();
    }

    // Choose a video container
    string getGstMuxPlugin()
    {
        ostringstream plugin;
        if      (container == "avi") { plugin << "avi"; }
        else if (container == "mp4") { plugin << "qt";  }
        else if (container == "mov") { plugin << "qt";  }
        else if (container == "mkv") { plugin << "matroska"; }
        else
        {
            cout << "Unsupported container: " << container << endl;
            cmd_parser->printErrors();
            return string();
        }

        if      (mode == "decode") { plugin << "demux"; }
        else if (mode == "encode") { plugin << "mux"; }
        else
        {
            cout << "Unsupported mode: " << mode << endl;
            cmd_parser->printErrors();
            return string();
        }

        return plugin.str();
    }

    // Choose a libav codec
    string getGstAvCodePlugin()
    {
        ostringstream plugin;
        if (mode == "decode")
        {
            if      (codec == "h264")  { plugin << "h264parse ! "; }
            else if (codec == "h265")  { plugin << "h265parse ! "; }
            plugin << "avdec_";
        }
        else if (mode == "encode") { plugin << "avenc_"; }
        else
        {
            cout << "Unsupported mode: " << mode << endl;
            cmd_parser->printErrors();
            return string();
        }

        if      (codec == "h264")  { plugin << "h264";       }
        else if (codec == "h265")  { plugin << "h265";       }
        else if (codec == "mpeg2") { plugin << "mpeg2video"; }
        else if (codec == "mpeg4") { plugin << "mpeg4";      }
        else if (codec == "mjpeg") { plugin << "mjpeg";      }
        else if (codec == "vp8")   { plugin << "vp8";        }
        else
        {
            cout << "Unsupported libav codec: " << codec << endl;
            cmd_parser->printErrors();
            return string();
        }

        return plugin.str();
    }

    // Choose a vaapi codec
    string getGstVaapiCodePlugin()
    {
        ostringstream plugin;
        if (mode == "decode")
        {
            plugin << "vaapidecodebin";
            if (container == "mkv") { plugin << " ! autovideoconvert"; }
            else { plugin << " ! video/x-raw, format=YV12"; }
        }
        else if (mode == "encode")
        {
            if      (codec == "h264")     { plugin << "vaapih264enc";  }
            else if (codec == "h265")     { plugin << "vaapih265enc";  }
            else if (codec == "mpeg2")    { plugin << "vaapimpeg2enc"; }
            else if (codec == "mjpeg")    { plugin << "vaapijpegenc";  }
            else if (codec == "vp8")      { plugin << "vaapivp8enc";   }
            else
            {
                cout << "Unsupported vaapi codec: " << codec << endl;
                cmd_parser->printErrors();
                return string();
            }
        }
        else
        {
            cout << "Unsupported mode: " << resolution << endl;
            cmd_parser->printErrors();
            return string();
        }
        return plugin.str();
    }

    // Choose a default codec
    string getGstDefaultCodePlugin()
    {
        ostringstream plugin;
        if (mode == "decode")
        {
            plugin << " ! decodebin";
        }
        else if (mode == "encode")
        {
            if      (codec == "h264")     { plugin << "x264enc";  }
            else if (codec == "h265")     { plugin << "x265enc";  }
            else if (codec == "mpeg2")    { plugin << "mpeg2enc"; }
            else if (codec == "mjpeg")    { plugin << "jpegenc";  }
            else if (codec == "vp8")      { plugin << "vp8enc";   }
            else
            {
                cout << "Unsupported default codec: " << codec << endl;
                cmd_parser->printErrors();
                return string();
            }
        }
        else
        {
            cout << "Unsupported mode: " << resolution << endl;
            cmd_parser->printErrors();
            return string();
        }
        return plugin.str();
    }
    // Get fourcc for codec
    int getFourccCode()
    {
        if      (codec == "h264")  { return VideoWriter::fourcc('H','2','6','4'); }
        else if (codec == "h265")  { return VideoWriter::fourcc('H','E','V','C'); }
        else if (codec == "mpeg2") { return VideoWriter::fourcc('M','P','E','G'); }
        else if (codec == "mpeg4") { return VideoWriter::fourcc('M','P','4','2'); }
        else if (codec == "mjpeg") { return VideoWriter::fourcc('M','J','P','G'); }
        else if (codec == "vp8")   { return VideoWriter::fourcc('V','P','8','0'); }
        else
        {
            cout << "Unsupported ffmpeg codec: " << codec << endl;
            cmd_parser->printErrors();
            return 0;
        }
    }

    // Check bad configuration
    int checkConfiguration()
    {
        if ((codec == "mpeg2" && getGstMuxPlugin() == "qtmux")  ||
            (codec == "h265"  && getGstMuxPlugin() == "avimux") ||
            (pipeline == "gst-libav" && (codec == "h264" || codec == "h265"))   ||
            (pipeline == "gst-vaapi1710" && codec=="mpeg2" && resolution=="4k") ||
            (pipeline == "gst-vaapi1710" && codec=="mpeg2" && resolution=="1080p" && fix_fps > 30))
        {
            cout << "Unsupported configuration" << endl;
            cmd_parser->printErrors();
            return -1;
        }
        return 0;
    }

    bool   fast_measure;     // fast measure fps
    string pipeline,         // gstreamer pipeline type
           container,        // container type
           mode,             // coding mode
           codec,            // codec type
           file_name,        // path to videofile
           resolution;       // video resolution
    int    fix_fps;          // fixed frame per second
    Size   fix_size;         // fixed frame size
    int    exit_code;
    VideoWriter  wrt;
    VideoCapture cap;
    ostringstream stream_pipeline;
    CommandLineParser* cmd_parser;
};

int main(int argc, char *argv[])
{
    GStreamerPipeline pipe(argc, argv);
    return pipe.run();
}
