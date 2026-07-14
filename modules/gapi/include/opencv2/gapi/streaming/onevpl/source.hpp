// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
using CfgParams = std::vector<CfgParam>;

/**
 * @brief G-API streaming source based on OneVPL implementation.
 *
 * This class implements IStreamSource interface.
 * Its constructor takes source file path (in usual way) or @ref onevpl::IDataProvider
 * interface implementation (for not file-based sources). It also allows to pass-through
 * oneVPL configuration parameters by using several @ref onevpl::CfgParam.
 *
 * @note stream sources are passed to G-API via shared pointers, so
 *  please gapi::make_onevpl_src<> to create objects and ptr() to pass a
 *  GSource to cv::gin().
 */
class GAPI_EXPORTS GSource : public IStreamSource
{
public:
    struct Priv;

    GSource(const std::string& filePath,
            const CfgParams& cfg_params = CfgParams{});

    GSource(const std::string& filePath,
            const CfgParams& cfg_params,
            const std::string& device_id,
            void* accel_device_ptr,
            void* accel_ctx_ptr);

    GSource(const std::string& filePath,
            const CfgParams& cfg_params,
            const Device &device, const Context &ctx);

    GSource(const std::string& filePath,
            const CfgParams& cfg_params,
            std::shared_ptr<IDeviceSelector> selector);


    GSource(std::shared_ptr<IDataProvider> source,
            const CfgParams& cfg_params = CfgParams{});

    GSource(std::shared_ptr<IDataProvider> source,
            const CfgParams& cfg_params,
            const std::string& device_id,
            void* accel_device_ptr,
            void* accel_ctx_ptr);

    GSource(std::shared_ptr<IDataProvider> source,
            const CfgParams& cfg_params,
            std::shared_ptr<IDeviceSelector> selector);

    ~GSource() override;

    bool pull(cv::gapi::wip::Data& data) override;
    GMetaArg descr_of() const override;

private:
    explicit GSource(std::unique_ptr<Priv>&& impl);
    std::unique_ptr<Priv> m_priv;
};
} // namespace onevpl

using GVPLSource = onevpl::GSource;

template<class... Args>
GAPI_EXPORTS_W cv::Ptr<IStreamSource> inline make_onevpl_src(Args&&... args)
{
    return make_src<onevpl::GSource>(std::forward<Args>(args)...);
}

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_SOURCE_HPP
