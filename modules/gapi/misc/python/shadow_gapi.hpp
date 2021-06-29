#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
struct GAPI_EXPORTS_W_SIMPLE GCompileArg {
   GAPI_WRAP GCompileArg(gapi::GKernelPackage pkg);
   GAPI_WRAP GCompileArg(gapi::GNetPackage pkg);
};

// NB: This classes doesn't exist in *.so
// HACK: Mark them as a class to force python wrapper generate code for this entities
class GAPI_EXPORTS_W_SIMPLE GProtoArg { };
class GAPI_EXPORTS_W_SIMPLE GProtoInputArgs { };
class GAPI_EXPORTS_W_SIMPLE GProtoOutputArgs { };
class GAPI_EXPORTS_W_SIMPLE GRunArg { };
class GAPI_EXPORTS_W_SIMPLE GMetaArg { GAPI_WRAP GMetaArg(); };

using GProtoInputArgs  = GIOProtoArgs<In_Tag>;
using GProtoOutputArgs = GIOProtoArgs<Out_Tag>;

class GAPI_EXPORTS_W_SIMPLE GInferInputs
{
public:
   GAPI_WRAP GInferInputs();
   GAPI_WRAP void setInput(const std::string& name, const cv::GMat&   value);
   GAPI_WRAP void setInput(const std::string& name, const cv::GFrame& value);
};

class GAPI_EXPORTS_W_SIMPLE GInferListInputs
{
public:
   GAPI_WRAP GInferListInputs();
   GAPI_WRAP void setInput(const std::string& name, const cv::GArray<cv::GMat>& value);
   GAPI_WRAP void setInput(const std::string& name, const cv::GArray<cv::Rect>& value);
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

namespace detail
{
   struct GAPI_EXPORTS_W_SIMPLE ExtractArgsCallback { };
   struct GAPI_EXPORTS_W_SIMPLE ExtractMetaCallback { };
} // namespace detail

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
} // namespace gapi
} // namespace cv
