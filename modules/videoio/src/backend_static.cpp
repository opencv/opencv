// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "backend.hpp"

namespace cv {

class StaticBackend: public IBackend
{
public:
    FN_createCaptureFile fn_createCaptureFile_;
    FN_createCaptureCamera fn_createCaptureCamera_;
    FN_createWriter fn_createWriter_;
    FN_createWriterWithProperties fn_createWriterWithProperties_;

    StaticBackend(FN_createCaptureFile fn_createCaptureFile,
                  FN_createCaptureCamera fn_createCaptureCamera,
                  FN_createWriter fn_createWriter,
                  FN_createWriterWithProperties fn_createWriterWithProperties)
        : fn_createCaptureFile_(fn_createCaptureFile),
        fn_createCaptureCamera_(fn_createCaptureCamera),
        fn_createWriter_(fn_createWriter),
        fn_createWriterWithProperties_(fn_createWriterWithProperties)
    {
        // nothing
    }
    ~StaticBackend() CV_OVERRIDE {}

    Ptr<IVideoCapture> createCapture(int camera) const CV_OVERRIDE
    {
        if (fn_createCaptureCamera_)
            return fn_createCaptureCamera_(camera);
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoCapture> createCapture(const std::string &filename) const CV_OVERRIDE
    {
        if (fn_createCaptureFile_)
            return fn_createCaptureFile_(filename);
        return Ptr<IVideoCapture>();
    }
    Ptr<IVideoWriter> createWriter(const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor) const CV_OVERRIDE
    {
        VideoWriterPropertyList emptyProperties;
        return createWriter(filename, fourcc, fps, sz, isColor, emptyProperties);
    }
    Ptr<IVideoWriter> createWriter(const std::string &filename, int fourcc, double fps, const cv::Size &sz, bool isColor, const VideoWriterPropertyList& properties) const CV_OVERRIDE
    {
        if (fn_createWriterWithProperties_)
            return fn_createWriterWithProperties_(filename, fourcc, fps, sz, isColor, properties);
        if (fn_createWriter_)
            return fn_createWriter_(filename, fourcc, fps, sz, isColor);
        return Ptr<IVideoWriter>();
    }
}; // StaticBackend

class StaticBackendFactory : public IBackendFactory
{
protected:
    Ptr<StaticBackend> backend;

public:
    StaticBackendFactory(FN_createCaptureFile createCaptureFile, FN_createCaptureCamera createCaptureCamera, FN_createWriter createWriter, FN_createWriterWithProperties createWriterWithProperties)
        : backend(makePtr<StaticBackend>(createCaptureFile, createCaptureCamera, createWriter, createWriterWithProperties))
    {
        // nothing
    }

    ~StaticBackendFactory() CV_OVERRIDE {}

    Ptr<IBackend> getBackend() const CV_OVERRIDE
    {
        return backend.staticCast<IBackend>();
    }
};

Ptr<IBackendFactory> createBackendFactory(FN_createCaptureFile createCaptureFile,
                                          FN_createCaptureCamera createCaptureCamera,
                                          FN_createWriter createWriter,
                                          FN_createWriterWithProperties createWriterWithProperties)
{
    return makePtr<StaticBackendFactory>(
        createCaptureFile,
        createCaptureCamera,
        createWriter,
        createWriterWithProperties)
        .staticCast<IBackendFactory>();
}

} // namespace
