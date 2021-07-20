#include <opencv2/gapi/util/throw.hpp>

#include "streaming/onevpl_builder.hpp"
#include "streaming/vpl/vpl_source_impl.hpp"



namespace cv {
namespace gapi {
namespace wip {
oneVPLBulder::oneVPLBulder()
{
}

std::unique_ptr<OneVPLCapture::IPriv> oneVPLBulder::build() const
{
#ifdef HAVE_ONEVPL
    if (filePath.empty())
    {
        util::throw_error(std::logic_error("Cannot create 'OneVPLCapture' on empty source file name"));
    }
    return std::unique_ptr<VPLSourceImpl>(new VPLSourceImpl(filePath, cfg_params));
#else
    abort();
#endif
}


void oneVPLBulder::set_arg(const std::string& file_path)
{
    this->filePath = file_path;
}

void oneVPLBulder::set_arg(const CFGParams& params)
{
    cfg_params = params;
}
} // namespace wip
} // namespace gapi
} // namespace cv
