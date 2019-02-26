// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef BACKEND_HPP_DEFINED
#define BACKEND_HPP_DEFINED

#include "cap_interface.hpp"
#include "opencv2/videoio/registry.hpp"

namespace cv {

// TODO: move to public interface
// TODO: allow runtime backend registration
class IBackend
{
public:
    Ptr<IVideoCapture> tryOpenCapture(const std::string & backendName, const std::string & filename, int cameraNum) const;
    Ptr<IVideoWriter> tryOpenWriter(const std::string & backendName, const std::string& filename, int _fourcc, double fps, const Size &frameSize, bool isColor) const;
protected:
    virtual Ptr<IVideoCapture> createCapture(const std::string &filename, int camera) const = 0;
    virtual Ptr<IVideoWriter> createWriter(const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor) const = 0;
    virtual ~IBackend() {}
};

//==================================================================================================

class StaticBackend : public IBackend
{
    typedef Ptr<IVideoCapture> (*OpenFileFun)(const std::string &);
    typedef Ptr<IVideoCapture> (*OpenCamFun)(int);
    typedef Ptr<IVideoWriter> (*OpenWriterFun)(const std::string&, int, double, const Size&, bool);
private:
    OpenFileFun FUN_FILE;
    OpenCamFun FUN_CAM;
    OpenWriterFun FUN_WRITE;
public:
    StaticBackend(OpenFileFun f1, OpenCamFun f2, OpenWriterFun f3)
        : FUN_FILE(f1), FUN_CAM(f2), FUN_WRITE(f3)
    {
    }
protected:
    Ptr<IVideoCapture> createCapture(const std::string &filename, int camera) const CV_OVERRIDE
    {
        if (filename.empty() && FUN_CAM)
            return FUN_CAM(camera);
        if (FUN_FILE)
            return FUN_FILE(filename);
        return 0;
    }
    Ptr<IVideoWriter> createWriter(const std::string &filename, int fourcc, double fps, const Size &sz, bool isColor) const CV_OVERRIDE
    {
        if (FUN_WRITE)
            return FUN_WRITE(filename, fourcc, fps, sz, isColor);
        return 0;
    }
};

//==================================================================================================

class DynamicBackend : public IBackend
{
public:
    class CaptureTable;
    class WriterTable;
    class DynamicLib;
private:
    DynamicLib * lib;
    CaptureTable const * cap_tbl;
    WriterTable const * wri_tbl;
public:
    DynamicBackend(const std::string &filename);
    ~DynamicBackend();
    static Ptr<DynamicBackend> load(VideoCaptureAPIs api, int mode);
protected:
    bool canCreateCapture(cv::VideoCaptureAPIs api) const;
    bool canCreateWriter(VideoCaptureAPIs api) const;
    Ptr<IVideoCapture> createCapture(const std::string &filename, int camera) const CV_OVERRIDE;
    Ptr<IVideoWriter> createWriter(const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor) const CV_OVERRIDE;
};

} // cv::


#endif // BACKEND_HPP_DEFINED
