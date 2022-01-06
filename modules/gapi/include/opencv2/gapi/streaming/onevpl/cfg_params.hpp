// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_CFG_PARAMS_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_CFG_PARAMS_HPP

#include <map>
#include <memory>
#include <string>

#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/util/variant.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

/**
 * @brief Public class is using for creation of onevpl::GSource instances.
 *
 * Class members availaible through methods @ref CfgParam::get_name() and @ref CfgParam::get_value() are used by
 * onevpl::GSource inner logic to create or find oneVPL particular implementation
 * (software/hardware, specific API version and etc.).
 *
 * @note Because oneVPL may provide several implementations which are satisfying with multiple (or single one) @ref CfgParam
 * criteria therefore it is possible to configure `preferred` parameters. This kind of CfgParams are created
 * using `is_major = false` argument in @ref CfgParam::create method and are not used by creating oneVPL particular implementations.
 * Instead they fill out a "score table" to select preferrable implementation from available list. Implementation are satisfying
 * with most of these optional params would be chosen.
 * If no one optional CfgParam params were present then first of available oneVPL implementation would be applied.
 * Please get on https://spec.oneapi.io/versions/latest/elements/oneVPL/source/API_ref/VPL_disp_api_func.html?highlight=mfxcreateconfig#mfxsetconfigfilterproperty
 * for using OneVPL configuration. In this schema `mfxU8 *name` represents @ref CfgParam::get_name() and
 * `mfxVariant value` is @ref CfgParam::get_value()
 */
struct GAPI_EXPORTS CfgParam {
    using name_t = std::string;
    using value_t = cv::util::variant<uint8_t, int8_t,
                                      uint16_t, int16_t,
                                      uint32_t, int32_t,
                                      uint64_t, int64_t,
                                      float_t,
                                      double_t,
                                      void*,
                                      std::string>;
    /**
     * @brief frames_pool_size_name
     *
     * Special configuration parameter name for onevp::GSource:
     *
     * @note frames_pool_size_name allows to allocate surfaces pool appropriate size to keep
     * decoded frames in accelerator memory ready before
     * they would be consumed by onevp::GSource::pull operation. If you see
     * a lot of WARNING about lack of free surface then it's time to increase
     * frames_pool_size_name but be aware of accelerator free memory volume.
     * If not set then MFX implementation use
     * mfxFrameAllocRequest::NumFrameSuggested behavior
     *
     */
    static constexpr const char *frames_pool_size_name() { return "frames_pool_size"; }
    static CfgParam create_frames_pool_size(size_t value);

    /**
     * @brief acceleration_mode_name
     *
     * Special configuration parameter names for onevp::GSource:
     *
     * @note acceleration_mode_name allows to activate hardware acceleration &
     * device memory management.
     * Supported values:
     * - MFX_ACCEL_MODE_VIA_D3D11   Will activate DX11 acceleration and will produces
     * MediaFrames with data allocated in DX11 device memory
     *
     * If not set then MFX implementation will use default acceleration behavior:
     * all decoding operation uses default GPU resources but MediaFrame produces
     * data allocated by using host RAM
     *
     */
    static constexpr const char *acceleration_mode_name() { return "mfxImplDescription.AccelerationMode"; }
    static CfgParam create_acceleration_mode(uint32_t value);
    static CfgParam create_acceleration_mode(const char* value);

    /**
     * @brief decoder_id_name
     *
     * Special configuration parameter names for onevp::GSource:
     *
     * @note decoder_id_name allows to specify VPL decoder type which MUST present
     * in case of RAW video input data and MUST NOT present as CfgParam if video
     * stream incapsulated into container(*.mp4, *.mkv and so on). In latter case
     * onevp::GSource will determine it automatically
     * Supported values:
     * - MFX_CODEC_AVC
     * - MFX_CODEC_HEVC
     * - MFX_CODEC_MPEG2
     * - MFX_CODEC_VC1
     * - MFX_CODEC_CAPTURE
     * - MFX_CODEC_VP9
     * - MFX_CODEC_AV1
     *
     */
    static constexpr const char *decoder_id_name() { return "mfxImplDescription.mfxDecoderDescription.decoder.CodecID"; }
    static CfgParam create_decoder_id(uint32_t value);
    static CfgParam create_decoder_id(const char* value);

    static constexpr const char *implementation_name() { return "mfxImplDescription.Impl"; }
    static CfgParam create_implementation(uint32_t value);
    static CfgParam create_implementation(const char* value);

    /**
     * Create generic onevp::GSource configuration parameter.
     *
     *@param name           name of parameter.
     *@param value          value of parameter.
     *@param is_major       TRUE if parameter MUST be provided by OneVPL inner implementation, FALSE for optional (for resolve multiple available implementations).
     *
     */
    template<typename ValueType>
    static CfgParam create(const std::string& name, ValueType&& value, bool is_major = true) {
        CfgParam param(name, CfgParam::value_t(std::forward<ValueType>(value)), is_major);
        return param;
    }

    struct Priv;

    const name_t& get_name() const;
    const value_t& get_value() const;
    bool is_major() const;
    bool operator==(const CfgParam& rhs) const;
    bool operator< (const CfgParam& rhs) const;
    bool operator!=(const CfgParam& rhs) const;

    CfgParam& operator=(const CfgParam& src);
    CfgParam& operator=(CfgParam&& src);
    CfgParam(const CfgParam& src);
    CfgParam(CfgParam&& src);
    ~CfgParam();
private:
    CfgParam(const std::string& param_name, value_t&& param_value, bool is_major_param);
    std::shared_ptr<Priv> m_priv;
};

} //namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_CFG_PARAMS_HPP
