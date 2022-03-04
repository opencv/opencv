#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>
#include <map>

using namespace std;
using namespace cv;

//================================================================================

template<typename M>
inline typename M::mapped_type getValue(const M &dict, const typename M::key_type &key, const string & errorMessage)
{
    typename M::const_iterator it = dict.find(key);
    if (it == dict.end())
    {
        CV_Error(Error::StsBadArg, errorMessage);
    }
    return it->second;
}

inline map<string, Size> sizeByResolution()
{
    map<string, Size> res;
    res["720p"] = Size(1280, 720);
    res["1080p"] = Size(1920, 1080);
    res["4k"] = Size(3840, 2160);
    return res;
}

inline map<string, int> fourccByCodec()
{
    map<string, int> res;
    res["h264"] = VideoWriter::fourcc('H','2','6','4');
    res["h265"] = VideoWriter::fourcc('H','E','V','C');
    res["mpeg2"] = VideoWriter::fourcc('M','P','E','G');
    res["mpeg4"] = VideoWriter::fourcc('M','P','4','2');
    res["mjpeg"] = VideoWriter::fourcc('M','J','P','G');
    res["vp8"] = VideoWriter::fourcc('V','P','8','0');
    return res;
}

inline map<string, string> defaultEncodeElementByCodec()
{
    map<string, string> res;
    res["h264"] = "x264enc";
    res["h265"] = "x265enc";
    res["mpeg2"] = "mpeg2enc";
    res["mjpeg"] = "jpegenc";
    res["vp8"] = "vp8enc";
    return res;
}

inline map<string, string> VAAPIEncodeElementByCodec()
{
    map<string, string> res;
    res["h264"] = "parsebin ! vaapih264enc";
    res["h265"] = "parsebin ! vaapih265enc";
    res["mpeg2"] = "parsebin ! vaapimpeg2enc";
    res["mjpeg"] = "parsebin ! vaapijpegenc";
    res["vp8"] = "parsebin ! vaapivp8enc";
    return res;
}

inline map<string, string> mfxDecodeElementByCodec()
{
    map<string, string> res;
    res["h264"] = "parsebin ! mfxh264dec";
    res["h265"] = "parsebin ! mfxhevcdec";
    res["mpeg2"] = "parsebin ! mfxmpeg2dec";
    res["mjpeg"] = "parsebin ! mfxjpegdec";
    return res;
}

inline map<string, string> mfxEncodeElementByCodec()
{
    map<string, string> res;
    res["h264"] = "mfxh264enc";
    res["h265"] = "mfxhevcenc";
    res["mpeg2"] = "mfxmpeg2enc";
    res["mjpeg"] = "mfxjpegenc";
    return res;
}

inline map<string, string> libavDecodeElementByCodec()
{
    map<string, string> res;
    res["h264"] = "parsebin ! avdec_h264";
    res["h265"] = "parsebin ! avdec_h265";
    res["mpeg2"] = "parsebin ! avdec_mpeg2video";
    res["mpeg4"] = "parsebin ! avdec_mpeg4";
    res["mjpeg"] = "parsebin ! avdec_mjpeg";
    res["vp8"] = "parsebin ! avdec_vp8";
    return res;
}

inline map<string, string> libavEncodeElementByCodec()
{
    map<string, string> res;
    res["h264"] = "avenc_h264";
    res["h265"] = "avenc_h265";
    res["mpeg2"] = "avenc_mpeg2video";
    res["mpeg4"] = "avenc_mpeg4";
    res["mjpeg"] = "avenc_mjpeg";
    res["vp8"] = "avenc_vp8";
    return res;
}

inline map<string, string> demuxPluginByContainer()
{
    map<string, string> res;
    res["avi"] = "avidemux";
    res["mp4"] = "qtdemux";
    res["mov"] = "qtdemux";
    res["mkv"] = "matroskademux";
    return res;
}

inline map<string, string> muxPluginByContainer()
{
    map<string, string> res;
    res["avi"] = "avimux";
    res["mp4"] = "qtmux";
    res["mov"] = "qtmux";
    res["mkv"] = "matroskamux";
    return res;
}

//================================================================================

inline string containerByName(const string &name)
{
    size_t found = name.rfind(".");
    if (found != string::npos)
    {
        return name.substr(found + 1);  // container type
    }
    return string();
}

//================================================================================

inline Ptr<VideoCapture> createCapture(const string &backend, const string &file_name, const string &codec)
{
    if (backend == "gst-default")
    {
        cout << "Created GStreamer capture ( " << file_name << " )" << endl;
        return makePtr<VideoCapture>(file_name, CAP_GSTREAMER);
    }
    else if (backend.find("gst") == 0)
    {
        ostringstream line;
        line << "filesrc location=\"" << file_name << "\"";
        line << " ! ";
        line << getValue(demuxPluginByContainer(), containerByName(file_name), "Invalid container");
        line << " ! ";
        if (backend.find("basic") == 4)
            line << "decodebin";
        else if (backend.find("vaapi") == 4)
            line << "vaapidecodebin";
        else if (backend.find("libav") == 4)
            line << getValue(libavDecodeElementByCodec(), codec, "Invalid codec");
        else if (backend.find("mfx") == 4)
            line << getValue(mfxDecodeElementByCodec(), codec, "Invalid or unsupported codec");
        else
            return Ptr<VideoCapture>();
        line << " ! videoconvert n-threads=" << getNumThreads();
        line << " ! appsink sync=false";
        cout << "Created GStreamer capture  ( " << line.str() << " )" << endl;
        return makePtr<VideoCapture>(line.str(), CAP_GSTREAMER);
    }
    else if (backend == "ffmpeg")
    {
        cout << "Created FFmpeg capture ( " << file_name << " )" << endl;
        return makePtr<VideoCapture>(file_name, CAP_FFMPEG);
    }
    return Ptr<VideoCapture>();
}

inline Ptr<VideoCapture> createSynthSource(Size sz, unsigned fps)
{
    ostringstream line;
    line << "videotestsrc pattern=smpte";
    line << " ! video/x-raw";
    line << ",width=" << sz.width << ",height=" << sz.height;
    if (fps > 0)
        line << ",framerate=" << fps << "/1";
    line << " ! appsink sync=false";
    cout << "Created synthetic video source ( " << line.str() << " )" << endl;
    return makePtr<VideoCapture>(line.str(), CAP_GSTREAMER);
}

inline Ptr<VideoWriter> createWriter(const string &backend, const string &file_name, const string &codec, Size sz, unsigned fps)
{
    if (backend == "gst-default")
    {
        cout << "Created GStreamer writer ( " << file_name << ", FPS=" << fps << ", Size=" << sz << ")" << endl;
        return makePtr<VideoWriter>(file_name, CAP_GSTREAMER, getValue(fourccByCodec(), codec, "Invalid codec"), fps, sz, true);
    }
    else if (backend.find("gst") == 0)
    {
        ostringstream line;
        line << "appsrc ! videoconvert n-threads=" << getNumThreads() << " ! ";
        if (backend.find("basic") == 4)
            line << getValue(defaultEncodeElementByCodec(), codec, "Invalid codec");
        else if (backend.find("vaapi") == 4)
            line << getValue(VAAPIEncodeElementByCodec(), codec, "Invalid codec");
        else if (backend.find("libav") == 4)
            line << getValue(libavEncodeElementByCodec(), codec, "Invalid codec");
        else if (backend.find("mfx") == 4)
            line << getValue(mfxEncodeElementByCodec(), codec, "Invalid codec");
        else
            return Ptr<VideoWriter>();
        line << " ! ";
        line << getValue(muxPluginByContainer(), containerByName(file_name), "Invalid container");
        line << " ! ";
        line << "filesink location=\"" << file_name << "\"";
        cout << "Created GStreamer writer ( " << line.str() << " )" << endl;
        return makePtr<VideoWriter>(line.str(), CAP_GSTREAMER, 0, fps, sz, true);
    }
    else if (backend == "ffmpeg")
    {
        cout << "Created FFmpeg writer ( " << file_name << ", FPS=" << fps << ", Size=" << sz << " )" << endl;
        return makePtr<VideoWriter>(file_name, CAP_FFMPEG, getValue(fourccByCodec(), codec, "Invalid codec"), fps, sz, true);
    }
    return Ptr<VideoWriter>();
}

//================================================================================

int main(int argc, char *argv[])
{
    const string keys =
        "{h help usage ? |           | print help messages   }"
        "{m mode         |decode     | coding mode (supported: encode, decode) }"
        "{b backend      |default    | video backend (supported: 'gst-default', 'gst-basic', 'gst-vaapi', 'gst-libav', 'gst-mfx', 'ffmpeg') }"
        "{c codec        |h264       | codec name     (supported: 'h264', 'h265', 'mpeg2', 'mpeg4', 'mjpeg', 'vp8') }"
        "{f file path    |           | path to file }"
        "{r resolution   |720p       | video resolution for encoding (supported: '720p', '1080p', '4k') }"
        "{fps            |30         | fix frame per second for encoding (supported: fps > 0) }"
        "{fast           |           | fast measure fps }";
    CommandLineParser cmd_parser(argc, argv, keys);
    cmd_parser.about("This program measures performance of video encoding and decoding using different backends OpenCV.");
    if (cmd_parser.has("help"))
    {
        cmd_parser.printMessage();
        return 0;
    }
    bool fast_measure = cmd_parser.has("fast");               // fast measure fps
    unsigned fix_fps      = cmd_parser.get<unsigned>("fps");      // fixed frame per second
    string backend     = cmd_parser.get<string>("backend");   // video backend
    string mode         = cmd_parser.get<string>("mode");       // coding mode
    string codec        = cmd_parser.get<string>("codec");      // codec type
    string file_name    = cmd_parser.get<string>("file");       // path to videofile
    string resolution   = cmd_parser.get<string>("resolution"); // video resolution
    if (!cmd_parser.check())
    {
        cmd_parser.printErrors();
        return -1;
    }
    if (mode != "encode" && mode != "decode")
    {
        cout << "Unsupported mode: " << mode << endl;
        return -1;
    }
    if (mode == "decode")
    {
        file_name = samples::findFile(file_name);
    }
    cout << "Mode: " << mode << ", Backend: " << backend << ", File: " << file_name << ", Codec: " << codec << endl;

    TickMeter total;
    Ptr<VideoCapture> cap;
    Ptr<VideoWriter> wrt;
    try
    {
        if (mode == "decode")
        {
            cap = createCapture(backend, file_name, codec);
            if (!cap)
            {
                cout << "Failed to create video capture" << endl;
                return -3;
            }
            if (!cap->isOpened())
            {
                cout << "Capture is not opened" << endl;
                return -4;
            }
        }
        else if (mode == "encode")
        {
            Size sz = getValue(sizeByResolution(), resolution, "Invalid resolution");
            cout << "FPS: " << fix_fps << ", Frame size: " << sz << endl;
            cap = createSynthSource(sz, fix_fps);
            wrt = createWriter(backend, file_name, codec, sz, fix_fps);
            if (!cap || !wrt)
            {
                cout << "Failed to create synthetic video source or video writer" << endl;
                return -3;
            }
            if (!cap->isOpened() || !wrt->isOpened())
            {
                cout << "Synthetic video source or video writer is not opened" << endl;
                return -4;
            }
        }
    }
    catch (...)
    {
        cout << "Unsupported parameters" << endl;
        return -2;
    }

    TickMeter tick;
    Mat frame;
    Mat element;
    total.start();
    while(true)
    {
        if (mode == "decode")
        {
            tick.start();
            if (!cap->grab())
            {
                cout << "No more frames - break" << endl;
                break;
            }
            if (!cap->retrieve(frame))
            {
                cout << "Failed to retrieve frame - break" << endl;
                break;
            }
            if (frame.empty())
            {
                cout << "Empty frame received - break" << endl;
                break;
            }
            tick.stop();
        }
        else if (mode == "encode")
        {
            int limit = 100;
            while (!cap->grab() && --limit != 0)
            {
                cout << "Skipping empty input frame - " << limit << endl;
            }
            cap->retrieve(element);
            tick.start();
            *wrt << element;
            tick.stop();
        }

        if (fast_measure && tick.getCounter() >= 1000)
        {
            cout << "Fast mode frame limit reached - break" << endl;
            break;
        }
        if (mode == "encode" && tick.getCounter() >= 1000)
        {
            cout << "Encode frame limit reached - break" << endl;
            break;
        }
    }
    total.stop();
    if (tick.getCounter() == 0)
    {
        cout << "No frames have been processed" << endl;
        return -10;
    }
    else
    {
        double res_fps = tick.getCounter() / tick.getTimeSec();
        cout << tick.getCounter() << " frames in " << tick.getTimeSec() << " sec ~ " << res_fps << " FPS" << " (total time: " << total.getTimeSec() << " sec)" << endl;
    }
    return 0;
}
