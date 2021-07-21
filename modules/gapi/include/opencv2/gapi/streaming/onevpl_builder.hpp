#ifndef OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP

#include <map>
#include <memory>
#include <string>

#include <opencv2/gapi/streaming/source.hpp>

#ifdef HAVE_ONEVPL

#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {

using CFGParamName = std::string;
using CFGParamValue = mfxVariant;
using CFGParams = std::map<CFGParamName, CFGParamValue>;

class GAPI_EXPORTS oneVPLBulder
{
public:
    template<typename... Param>
    oneVPLBulder(Param&& ...params)
    {
        set(std::forward<Param>(params)...);
    }

    template<typename... Param>
    void set(Param&& ...params)
    {
        std::array<bool, sizeof...(params)> expander {
        (set_arg(params), true)...};
        (void)expander;
    }

    std::shared_ptr<IStreamSource> build() const;

private:
    void set_arg(const std::string& file_path);
    void set_arg(const CFGParams& params);

    std::string filePath;
    CFGParams cfg_params;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP
