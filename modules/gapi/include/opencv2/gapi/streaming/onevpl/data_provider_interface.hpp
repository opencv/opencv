// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#include <exception>
#include <memory>
#include <string>

#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct GAPI_EXPORTS DataProviderException : public std::exception {
    DataProviderException(const std::string& descr);
    DataProviderException(std::string&& descr);

    virtual ~DataProviderException() = default;
    virtual const char* what() const noexcept override;
private:
    std::string reason;
};

struct GAPI_EXPORTS DataProviderSystemErrorException final : public DataProviderException {
    DataProviderSystemErrorException(int error_code, const std::string& description = std::string());
    ~DataProviderSystemErrorException() = default;
};

struct GAPI_EXPORTS DataProviderUnsupportedException final : public DataProviderException {
    DataProviderUnsupportedException(const std::string& description);
    ~DataProviderUnsupportedException() = default;
};

struct GAPI_EXPORTS DataProviderImplementationException : public DataProviderException {
    DataProviderImplementationException(const std::string& description);
    ~DataProviderImplementationException() = default;
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
    using mfx_codec_id_type = uint32_t;

    /**
     * NB: here is supposed to be forward declaration of mfxBitstream
     * But according to current oneVPL implementation it is impossible to forward
     * declare untagged struct mfxBitstream.
     *
     * IDataProvider makes sense only for HAVE_VPL is ON and to keep IDataProvider
     * interface API/ABI compliant between core library and user application layer
     * let's introduce wrapper mfx_bitstream which inherits mfxBitstream in private
     * G-API code section and declare forward for wrapper mfx_bitstream here
     */
    struct mfx_bitstream;

    virtual ~IDataProvider() = default;

    /**
     * The function is used by onevpl::GSource to extract codec id from data
     *
     */
    virtual mfx_codec_id_type get_mfx_codec_id() const = 0;

    /**
     * The function is used by onevpl::GSource to extract binary data stream from @ref IDataProvider
     * implementation.
     *
     * It MUST throw `DataProviderException` kind exceptions in fail cases.
     * It MUST return MFX_ERR_MORE_DATA in EOF which considered as not-fail case.
     *
     * @param in_out_bitsream the input-output reference on MFX bitstream buffer which MUST be empty at the first request
     * to allow implementation to allocate it by itself and to return back. Subsequent invocation of `fetch_bitstream_data`
     * MUST use the previously used in_out_bitsream to avoid skipping rest of frames which haven't been consumed
     * @return true for fetched data, false on EOF and throws exception on error
     */
    virtual bool fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &in_out_bitsream) = 0;

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
