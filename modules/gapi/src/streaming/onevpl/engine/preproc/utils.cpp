#include <type_traits>

#include "streaming/onevpl/engine/preproc/utils.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
namespace utils {

cv::MediaFormat fourcc_to_MediaFormat(int value) {
    switch (value)
    {
        case MFX_FOURCC_BGRP:
            return cv::MediaFormat::BGR;
        case MFX_FOURCC_NV12:
            return cv::MediaFormat::NV12;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported FourCC format requested: " << value <<
                                     ". Cannot cast to cv::MediaFrame");
            GAPI_Assert(false && "Unsupported FOURCC");

    }
}

int MediaFormat_to_fourcc(cv::MediaFormat value) {
    switch (value)
    {
        case cv::MediaFormat::BGR:
            return MFX_FOURCC_BGRP;
        case cv::MediaFormat::NV12:
            return MFX_FOURCC_NV12;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported cv::MediaFormat format requested: " <<
                                      static_cast<typename std::underlying_type<cv::MediaFormat>::type>(value) <<
                                     ". Cannot cast to FourCC");
            GAPI_Assert(false && "Unsupported cv::MediaFormat");
    }
}
int MediaFormat_to_chroma(cv::MediaFormat value) {
    switch (value)
    {
        case cv::MediaFormat::BGR:
            return MFX_CHROMAFORMAT_MONOCHROME;
        case cv::MediaFormat::NV12:
            return MFX_CHROMAFORMAT_YUV420;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported cv::MediaFormat format requested: " <<
                                      static_cast<typename std::underlying_type<cv::MediaFormat>::type>(value) <<
                                     ". Cannot cast to ChromaFormateIdc");
            GAPI_Assert(false && "Unsupported cv::MediaFormat");
    }
}
} // namespace utils
} // namespace cv
} // namespace gapi
} // namespace wip
} // namespace onevpl

#endif // HAVE_ONEVPL
