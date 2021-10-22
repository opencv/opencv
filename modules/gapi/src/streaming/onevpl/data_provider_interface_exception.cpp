#include <errno.h>
#include <string.h>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
DataProviderSystemErrorException::DataProviderSystemErrorException(int error_code,
                                                                   const std::string& description) {
    reason = description + ", error: " + std::to_string(error_code) + ", description: " + strerror(error_code);
}

DataProviderSystemErrorException::~DataProviderSystemErrorException() = default;

const char* DataProviderSystemErrorException::what() const noexcept {
    return reason.c_str();
}

DataProviderUnsupportedException::DataProviderUnsupportedException(const std::string& description) {
    reason = description;
}

DataProviderUnsupportedException::~DataProviderUnsupportedException() = default;

const char* DataProviderUnsupportedException::what() const noexcept {
    return reason.c_str();
}


const char *IDataProvider::to_cstr(IDataProvider::CodecID codec) {
    switch(codec) {
        case IDataProvider::CodecID::AVC:
            return "AVC";
        case IDataProvider::CodecID::HEVC:
            return "HEVC";
        case IDataProvider::CodecID::MPEG2:
            return "MPEG2";
        case IDataProvider::CodecID::VC1:
            return "VC1";
        case IDataProvider::CodecID::VP9:
            return "VP9";
        case IDataProvider::CodecID::AV1:
            return "AV1";
        case IDataProvider::CodecID::JPEG:
            return "JPEG";
        default:
            return "<unsupported>";
    }
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
