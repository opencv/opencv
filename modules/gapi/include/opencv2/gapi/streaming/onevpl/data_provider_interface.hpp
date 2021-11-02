// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#include <exception>
#include <limits>
#include <string>

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#endif // HAVE_ONEVPL

#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct GAPI_EXPORTS DataProviderException : public std::exception {
    virtual ~DataProviderException() {}
};

struct GAPI_EXPORTS DataProviderSystemErrorException : public DataProviderException {
    DataProviderSystemErrorException(int error_code, const std::string& desription = std::string());
    virtual ~DataProviderSystemErrorException();
    virtual const char* what() const noexcept override;

private:
    std::string reason;
};

struct GAPI_EXPORTS DataProviderUnsupportedException : public DataProviderException {
    DataProviderUnsupportedException(const std::string& desription);
    virtual ~DataProviderUnsupportedException();
    virtual const char* what() const noexcept override;

private:
    std::string reason;
};

/**
 * @brief Public interface allows to customize extraction of video stream data
 * used by onevpl::GSource instead of reading stream from file (by default).
 *
 * Interface implementation constructor MUST provide consistency and creates fully operable object.
 * If error happened implementation MUST throw `DataProviderException` kind exceptions
 *
 * @note Interface implementation MUST manage stream and other constructed resources by itself to avoid any kind of leak.
 * For simple interface implementation example please see `StreamDataProvider` in `tests/streaming/gapi_streaming_tests.cpp`
 */
struct GAPI_EXPORTS IDataProvider {
    using Ptr = std::shared_ptr<IDataProvider>;

    enum class CodecID : uint16_t {
        AVC,
        HEVC,
        MPEG2,
        VC1,
        VP9,
        AV1,
        JPEG,

        UNCOMPRESSED = std::numeric_limits<uint16_t>::max()
    };

    static const char *to_cstr(CodecID codec);
    static int codec_id_to_mfx(IDataProvider::CodecID codec);

    virtual ~IDataProvider() = default;

    /**
     * The function is used by onevpl::GSource to extract codec id from data
     *
     */
    virtual CodecID get_codec() const = 0;

    /**
     * The function is used by onevpl::GSource to extract binary data stream from @ref IDataProvider
     * implementation.
     *
     * It MUST throw `DataProviderException` kind exceptions in fail cases.
     * It MUST return 0 in EOF which considered as not-fail case.
     *
     * @param out_data_bytes_size the available capacity of out_data buffer.
     * @param out_data the output consumer buffer with capacity out_data_bytes_size.
     * @return fetched bytes count.
     */
#ifdef HAVE_ONEVPL
    virtual mfxStatus fetch_bitstream_data(std::shared_ptr<mfxBitstream> &out_bitsream) = 0;
#endif // HAVE_ONEVPL
    /**
     * The function is used by onevpl::GSource to check more binary data availability.
     *
     * It MUST return TRUE in case of EOF and NO_THROW exceptions.
     *
     * @return boolean value which detects end of stream
     */
    virtual bool empty() const = 0;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
