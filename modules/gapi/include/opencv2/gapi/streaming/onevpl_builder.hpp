#ifndef OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP

#include <map>
#include <memory>
#include <string>

#include <opencv2/gapi/streaming/source.hpp>

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif // MFX_VERSION

#include <vpl/mfx.h>
#endif // HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {

using CFGParamName = std::string;

#ifdef HAVE_ONEVPL
using CFGParamValue = mfxVariant;
#else
namespace detail {
class EmptyParamValue {
    class {} Type;
    class {} Data;
};
}
using CFGParamValue = detail::EmptyParamValue;
#endif // HAVE_ONEVPL

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
        (set_arg(std::forward<Param>(params)), true)...};
        (void)expander;
    }

    std::shared_ptr<IStreamSource> build() const;

private:
    void set_arg(const std::string& file_path);
    void set_arg(const CFGParams& params);
    /*template<typename Param>
    void set_arg(Param&& p) {
        (void)p;
        abort();
        static_assert(std::is_same<typename std::decay<Param>::type,CFGParams>::value, "Unexpected param");
    }*/
    
    std::string filePath;
    CFGParams cfg_params;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_PRIV_BUILDER_HPP
