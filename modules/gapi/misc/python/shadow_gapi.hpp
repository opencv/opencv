#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
struct GAPI_EXPORTS_W_SIMPLE GCompileArg
{
    GAPI_WRAP GCompileArg(GKernelPackage arg);
    GAPI_WRAP GCompileArg(gapi::GNetPackage arg);
    GAPI_WRAP GCompileArg(gapi::streaming::queue_capacity arg);
};

class GAPI_EXPORTS_W_SIMPLE GInferInputs
{
public:
    GAPI_WRAP GInferInputs();
    GAPI_WRAP GInferInputs& setInput(const std::string& name, const cv::GMat&   value);
    GAPI_WRAP GInferInputs& setInput(const std::string& name, const cv::GFrame& value);
};

class GAPI_EXPORTS_W_SIMPLE GInferListInputs
{
public:
    GAPI_WRAP GInferListInputs();
    GAPI_WRAP GInferListInputs setInput(const std::string& name, const cv::GArray<cv::GMat>& value);
    GAPI_WRAP GInferListInputs setInput(const std::string& name, const cv::GArray<cv::Rect>& value);
};

class GAPI_EXPORTS_W_SIMPLE GInferOutputs
{
public:
    GAPI_WRAP GInferOutputs();
    GAPI_WRAP cv::GMat at(const std::string& name);
};

class GAPI_EXPORTS_W_SIMPLE GInferListOutputs
{
public:
    GAPI_WRAP GInferListOutputs();
    GAPI_WRAP cv::GArray<cv::GMat> at(const std::string& name);
};

namespace gapi
{
namespace wip
{
class GAPI_EXPORTS_W IStreamSource { };
namespace draw
{
    // NB: These render primitives are partially wrapped in shadow file
    // because cv::Rect conflicts with cv::gapi::wip::draw::Rect in python generator
    // and cv::Rect2i breaks standalone mode.
    struct Rect
    {
        GAPI_WRAP Rect(const cv::Rect2i& rect_,
                       const cv::Scalar& color_,
                       int thick_ = 1,
                       int lt_ = 8,
                       int shift_ = 0);
    };

    struct Mosaic
    {
        GAPI_WRAP Mosaic(const cv::Rect2i& mos_, int cellSz_, int decim_);
    };
} // namespace draw
} // namespace wip
namespace streaming
{
    // FIXME: Extend to work with an arbitrary G-type.
    cv::GOpaque<int64_t> GAPI_EXPORTS_W timestamp(cv::GMat);
    cv::GOpaque<int64_t> GAPI_EXPORTS_W seqNo(cv::GMat);
    cv::GOpaque<int64_t> GAPI_EXPORTS_W seq_id(cv::GMat);

    GAPI_EXPORTS_W cv::GMat desync(const cv::GMat &g);
} // namespace streaming
} // namespace gapi

namespace detail
{
    gapi::GNetParam GAPI_EXPORTS_W strip(gapi::ie::PyParams params);
} // namespace detail
} // namespace cv
