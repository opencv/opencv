// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//
// Not a standalone header.
//

namespace cv { namespace impl { namespace legacy {

//==================================================================================================

class PluginCapture : public cv::IVideoCapture
{
    const OpenCV_VideoIO_Plugin_API_preview* plugin_api_;
    CvPluginCapture capture_;

public:
    static
    Ptr<PluginCapture> create(const OpenCV_VideoIO_Plugin_API_preview* plugin_api,
            const std::string &filename, int camera)
    {
        CV_Assert(plugin_api);
        CvPluginCapture capture = NULL;
        if (plugin_api->v0.Capture_open)
        {
            CV_Assert(plugin_api->v0.Capture_release);
            if (CV_ERROR_OK == plugin_api->v0.Capture_open(filename.empty() ? 0 : filename.c_str(), camera, &capture))
            {
                CV_Assert(capture);
                return makePtr<PluginCapture>(plugin_api, capture);
            }
        }
        return Ptr<PluginCapture>();
    }

    PluginCapture(const OpenCV_VideoIO_Plugin_API_preview* plugin_api, CvPluginCapture capture)
        : plugin_api_(plugin_api), capture_(capture)
    {
        CV_Assert(plugin_api_); CV_Assert(capture_);
    }

    ~PluginCapture()
    {
        CV_DbgAssert(plugin_api_->v0.Capture_release);
        if (CV_ERROR_OK != plugin_api_->v0.Capture_release(capture_))
            CV_LOG_ERROR(NULL, "Video I/O: Can't release capture by plugin '" << plugin_api_->api_header.api_description << "'");
        capture_ = NULL;
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        double val = -1;
        if (plugin_api_->v0.Capture_getProperty)
            if (CV_ERROR_OK != plugin_api_->v0.Capture_getProperty(capture_, prop, &val))
                val = -1;
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        if (plugin_api_->v0.Capture_setProperty)
            if (CV_ERROR_OK == plugin_api_->v0.Capture_setProperty(capture_, prop, val))
                return true;
        return false;
    }
    bool grabFrame() CV_OVERRIDE
    {
        if (plugin_api_->v0.Capture_grab)
            if (CV_ERROR_OK == plugin_api_->v0.Capture_grab(capture_))
                return true;
        return false;
    }
    static CvResult CV_API_CALL retrieve_callback(int stream_idx, const unsigned char* data, int step, int width, int height, int cn, void* userdata)
    {
        CV_UNUSED(stream_idx);
        cv::_OutputArray* dst = static_cast<cv::_OutputArray*>(userdata);
        if (!dst)
            return CV_ERROR_FAIL;
        cv::Mat(cv::Size(width, height), CV_MAKETYPE(CV_8U, cn), (void*)data, step).copyTo(*dst);
        return CV_ERROR_OK;
    }
    bool retrieveFrame(int idx, cv::OutputArray img) CV_OVERRIDE
    {
        bool res = false;
        if (plugin_api_->v0.Capture_retreive)
            if (CV_ERROR_OK == plugin_api_->v0.Capture_retreive(capture_, idx, retrieve_callback, (cv::_OutputArray*)&img))
                res = true;
        return res;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return capture_ != NULL;  // TODO always true
    }
    int getCaptureDomain() CV_OVERRIDE
    {
        return plugin_api_->v0.captureAPI;
    }
};


//==================================================================================================

class PluginWriter : public cv::IVideoWriter
{
    const OpenCV_VideoIO_Plugin_API_preview* plugin_api_;
    CvPluginWriter writer_;

public:
    static
    Ptr<PluginWriter> create(const OpenCV_VideoIO_Plugin_API_preview* plugin_api,
            const std::string& filename, int fourcc, double fps, const cv::Size& sz,
            const VideoWriterParameters& params)
    {
        CV_Assert(plugin_api);
        CvPluginWriter writer = NULL;
        if (plugin_api->api_header.api_version >= 1 && plugin_api->v1.Writer_open_with_params)
        {
            CV_Assert(plugin_api->v0.Writer_release);
            CV_Assert(!filename.empty());
            std::vector<int> vint_params = params.getIntVector();
            int* c_params = &vint_params[0];
            unsigned n_params = (unsigned)(vint_params.size() / 2);

            if (CV_ERROR_OK == plugin_api->v1.Writer_open_with_params(filename.c_str(), fourcc, fps, sz.width, sz.height, c_params, n_params, &writer))
            {
                CV_Assert(writer);
                return makePtr<PluginWriter>(plugin_api, writer);
            }
        }
        else if (plugin_api->v0.Writer_open)
        {
            CV_Assert(plugin_api->v0.Writer_release);
            CV_Assert(!filename.empty());
            const bool isColor = params.get(VIDEOWRITER_PROP_IS_COLOR, true);
            const int depth = params.get(VIDEOWRITER_PROP_DEPTH, CV_8U);
            if (depth != CV_8U)
            {
                CV_LOG_WARNING(NULL, "Video I/O plugin doesn't support (due to lower API level) creation of VideoWriter with depth != CV_8U");
                return Ptr<PluginWriter>();
            }
            if (CV_ERROR_OK == plugin_api->v0.Writer_open(filename.c_str(), fourcc, fps, sz.width, sz.height, isColor, &writer))
            {
                CV_Assert(writer);
                return makePtr<PluginWriter>(plugin_api, writer);
            }
        }
        return Ptr<PluginWriter>();
    }

    PluginWriter(const OpenCV_VideoIO_Plugin_API_preview* plugin_api, CvPluginWriter writer)
        : plugin_api_(plugin_api), writer_(writer)
    {
        CV_Assert(plugin_api_); CV_Assert(writer_);
    }

    ~PluginWriter()
    {
        CV_DbgAssert(plugin_api_->v0.Writer_release);
        if (CV_ERROR_OK != plugin_api_->v0.Writer_release(writer_))
            CV_LOG_ERROR(NULL, "Video I/O: Can't release writer by plugin '" << plugin_api_->api_header.api_description << "'");
        writer_ = NULL;
    }
    double getProperty(int prop) const CV_OVERRIDE
    {
        double val = -1;
        if (plugin_api_->v0.Writer_getProperty)
            if (CV_ERROR_OK != plugin_api_->v0.Writer_getProperty(writer_, prop, &val))
                val = -1;
        return val;
    }
    bool setProperty(int prop, double val) CV_OVERRIDE
    {
        if (plugin_api_->v0.Writer_setProperty)
            if (CV_ERROR_OK == plugin_api_->v0.Writer_setProperty(writer_, prop, val))
                return true;
        return false;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return writer_ != NULL;  // TODO always true
    }
    void write(cv::InputArray arr) CV_OVERRIDE
    {
        cv::Mat img = arr.getMat();
        CV_DbgAssert(writer_);
        CV_Assert(plugin_api_->v0.Writer_write);
        if (CV_ERROR_OK != plugin_api_->v0.Writer_write(writer_, img.data, (int)img.step[0], img.cols, img.rows, img.channels()))
        {
            CV_LOG_DEBUG(NULL, "Video I/O: Can't write frame by plugin '" << plugin_api_->api_header.api_description << "'");
        }
        // TODO return bool result?
    }
    int getCaptureDomain() const CV_OVERRIDE
    {
        return plugin_api_->v0.captureAPI;
    }
};


}}}  // namespace
